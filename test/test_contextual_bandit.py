import unittest
from unittest.mock import patch

import numpy as np

from aiphy.contextual_bandit import (
    AdaptiveTheoryReward,
    ExperimentDescriptor,
    LinUCBExperimentSelector,
    RewardBreakdown,
    SelectionDecision,
    TheorySnapshot,
    TheoryStateEncoder,
)


def snapshot(*, epoch=0, concepts=0, laws=0, valid_per_law=0, complexity=None):
    general_laws = {
        f"R_{i}": {
            "prop": f"law_{i}",
            "type": "IsZero",
            "indication": "ForAll",
            "valid_experiments": [f"exp_{j}" for j in range(valid_per_law)],
            "father": None,
            "complexity": 10.0,
        }
        for i in range(laws)
    }
    return TheorySnapshot(
        epoch=epoch,
        concepts={f"c_{i}": f"concept_{i}" for i in range(concepts)},
        intrinsic_concepts={},
        conclusions={},
        general_laws=general_laws,
        completion_exclusions={},
        specific_conclusions={},
        marked_general_laws=list(general_laws),
        experiment_history_sizes={},
        law_complexity=10.0 * laws if complexity is None else complexity,
    )


class AdaptiveRewardTests(unittest.TestCase):
    def test_early_novelty_is_rewarded(self):
        before = snapshot()
        after = snapshot(concepts=3, laws=1)
        after.general_laws["R_0"]["valid_experiments"] = ["exp_0"]

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertGreater(reward.total, 0.0)
        self.assertGreater(reward.novelty_term, 0.0)
        self.assertEqual(reward.added_concepts, 3)
        self.assertEqual(reward.added_laws, 1)

    def test_large_concept_growth_without_theory_is_penalized(self):
        before = snapshot(concepts=100)
        after = snapshot(concepts=130)

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertLess(reward.total, 0.0)
        self.assertGreater(reward.growth_penalty, 0.0)

    def test_derived_concept_is_not_counted_as_active_novelty(self):
        before = snapshot(concepts=5)
        after = snapshot(concepts=6)
        after.derived_concepts = ["c_5"]

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertEqual(reward.added_concepts, 0)
        self.assertEqual(reward.added_derived_concepts, 1)
        self.assertGreaterEqual(reward.growth_penalty, 0.1)
        self.assertLess(reward.total, 0.0)

    def test_validation_reward_saturates(self):
        before = snapshot(concepts=30, laws=1)
        after = snapshot(concepts=30, laws=1, valid_per_law=100)

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertGreater(reward.validation_term, 0.0)
        self.assertLess(reward.validation_term, 10.0)

    def test_cached_scope_of_new_law_is_not_rewarded_as_new_evidence(self):
        before = snapshot(concepts=10)
        after = snapshot(concepts=11, laws=1, valid_per_law=40)

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertEqual(reward.new_validations, 0)
        self.assertEqual(reward.backfilled_validations, 40)
        self.assertEqual(reward.validation_term, 0.0)
        self.assertGreater(reward.total, 0.0)

    def test_unresolved_growth_is_penalized_for_a_mature_theory(self):
        before = snapshot(epoch=500, concepts=500, laws=60, valid_per_law=8)
        after = snapshot(epoch=501, concepts=501, laws=61, valid_per_law=8)
        # The extra law has not explained or survived any new experiment.
        after.general_laws["R_60"]["valid_experiments"] = []

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertGreater(before.maturity, 0.99)
        self.assertLess(reward.novelty_term, 0.0)
        self.assertLess(reward.total, 0.0)

    def test_revision_and_new_evidence_receive_large_reward(self):
        before = snapshot(concepts=30, laws=1, complexity=20.0)
        after = snapshot(concepts=31, laws=2, complexity=18.0)
        after.general_laws["R_1"]["father"] = "R_0"
        after.completion_exclusions = {"R_0": ["critical_exp"]}

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertEqual(reward.revised_laws, 1)
        self.assertEqual(reward.new_contradictions, 1)
        self.assertGreater(reward.revision_term, 5.0)

    def test_failed_scope_of_new_hypothesis_is_not_falsification(self):
        before = snapshot(concepts=5)
        after = snapshot(concepts=6, laws=1)
        after.completion_exclusions = {"R_0": ["exp_outside_scope"]}

        reward = AdaptiveTheoryReward().evaluate(before, after)

        self.assertEqual(reward.new_contradictions, 0)


class LinUCBPersistenceTests(unittest.TestCase):
    @staticmethod
    def descriptor(name):
        return ExperimentDescriptor(
            name=name,
            spatial_dimension=1,
            objects=[],
            original_data=[],
            original_concepts=[],
            known_valid_laws=[],
            known_excluded_laws=[],
            testable_unknown_laws=[],
            non_applicable_laws=[],
            selection_count=0,
            computational_limit=5.0,
        )

    def test_periodic_coverage_selects_least_tried_arm(self):
        selector = LinUCBExperimentSelector(hash_dim=8, coverage_interval=4)
        selector.action_counts = {"overused": 10, "underused": 1}
        selector.total_updates = 4
        with patch.object(
            TheoryStateEncoder,
            "describe_experiment",
            side_effect=lambda _, name: self.descriptor(name),
        ):
            decision = selector.select(object(), ["overused", "underused"], snapshot())

        self.assertEqual(decision.experiment, "underused")
        self.assertEqual(decision.selection_reason, "coverage")

    def test_hard_cooldown_prevents_fourth_consecutive_selection(self):
        selector = LinUCBExperimentSelector(hash_dim=8, max_consecutive=3)
        selector.action_counts = {"repeated": 5, "alternative": 5}
        selector.action_reward_ema = {"repeated": 10.0, "alternative": 0.0}
        selector.total_updates = 5
        selector.last_experiment = "repeated"
        selector.consecutive_selections = 3
        with patch.object(
            TheoryStateEncoder,
            "describe_experiment",
            side_effect=lambda _, name: self.descriptor(name),
        ):
            decision = selector.select(object(), ["repeated", "alternative"], snapshot())

        self.assertEqual(decision.experiment, "alternative")
        self.assertEqual(decision.selection_reason, "cooldown")

    def test_compute_budget_only_grows_for_timed_out_arm(self):
        selector = LinUCBExperimentSelector(hash_dim=8)

        selector.record_compute_outcome("slow", "timeout", 5.0, 100.0, 3.0)

        self.assertEqual(selector.compute_limit_for("slow", 5.0, 100.0), 15.0)
        self.assertEqual(selector.compute_limit_for("other", 5.0, 100.0), 5.0)

    def test_repeated_no_progress_does_not_collapse_to_one_arm(self):
        selector = LinUCBExperimentSelector(hash_dim=8, coverage_interval=4)
        candidates = [f"exp_{i}" for i in range(5)]
        no_progress = RewardBreakdown(total=-0.35, maturity=0.0, outcome="success")
        with patch.object(
            TheoryStateEncoder,
            "describe_experiment",
            side_effect=lambda _, name: self.descriptor(name),
        ):
            for _ in range(100):
                decision = selector.select(object(), candidates, snapshot())
                selector.update(decision, no_progress)

        self.assertLessEqual(max(selector.action_counts.values()), 30)
        self.assertGreaterEqual(min(selector.action_counts.values()), 10)

    def test_exact_state_archive_roundtrip(self):
        selector = LinUCBExperimentSelector(hash_dim=8)
        state = snapshot(epoch=12, concepts=7, laws=2, valid_per_law=1)

        state_hash = selector.remember_state(state)
        restored_selector = LinUCBExperimentSelector.from_json(selector.to_json())
        restored = restored_selector.load_state(state_hash)

        self.assertEqual(restored.to_json(), state.to_json())
        self.assertEqual(restored.state_hash, state_hash)

    def test_update_and_json_roundtrip(self):
        selector = LinUCBExperimentSelector(hash_dim=8)
        feature = np.linspace(0.0, 1.0, selector.encoder.dimension)
        decision = SelectionDecision(
            experiment="exp_a",
            feature=feature.tolist(),
            state_hash="state",
            descriptor={},
            predicted_reward=0.0,
            uncertainty=1.0,
            score=1.0,
            forced_exploration=True,
        )
        reward = RewardBreakdown(total=7.5, maturity=0.2, outcome="success")

        selector.update(decision, reward)
        restored = LinUCBExperimentSelector.from_json(selector.to_json())

        self.assertEqual(restored.action_counts, {"exp_a": 1})
        self.assertEqual(restored.action_reward_ema, {"exp_a": 7.5})
        self.assertEqual(restored.last_experiment, "exp_a")
        self.assertEqual(restored.consecutive_selections, 1)
        self.assertEqual(restored.total_updates, 1)
        np.testing.assert_allclose(restored.a, selector.a)
        np.testing.assert_allclose(restored.b, selector.b)
        self.assertEqual(restored.history[-1]["reward"]["total"], 7.5)


if __name__ == "__main__":
    unittest.main()
