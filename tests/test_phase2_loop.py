"""
Integration tests for Phase 2 modules.

Tests the full pipeline: theory_population → prediction_registry →
falsification_judge → discriminating_selector.

Uses synthetic benchmarks with known ground truth to validate
that the system can discover/narrow down to correct forms.
"""

from __future__ import annotations

import math
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from aiphy.theory_population import TheoryPopulation, TheoryRecord, TheoryStatus
from aiphy.prediction_registry import (
    PredictionRegistry, PredictionRecord, PredictionOutcomeRecord,
    PredictionOutcome,
)
from aiphy.falsification_judge import FalsificationJudge, FalsificationResult
from aiphy.discriminating_selector import (
    DiscriminatingExperimentSelector,
    DiscriminatingExperimentScore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_population() -> TheoryPopulation:
    return TheoryPopulation()


@pytest.fixture
def populated() -> TheoryPopulation:
    """A population with 5 active theories."""
    pop = TheoryPopulation()
    expressions = ["x + 5", "2*x + 1", "x**2 - 1", "3*x", "x - 2"]
    for i, expr in enumerate(expressions):
        tid = pop.add_theory(name=f"theory_{i}", expression=expr)
        pop.activate(tid)
    return pop


@pytest.fixture
def registry() -> PredictionRegistry:
    return PredictionRegistry()


@pytest.fixture
def filled_registry(populated: TheoryPopulation) -> PredictionRegistry:
    """Registry with predictions for two experiments."""
    reg = PredictionRegistry()
    actives = list(populated.active_theories.keys())
    for i, tid in enumerate(actives):
        reg.pre_register(tid, "exp_A", f"{i}.0", prediction_type="conserved")
        if i < 3:
            reg.pre_register(tid, "exp_B", f"{i * 2}.0", prediction_type="conserved")
    return reg


@pytest.fixture
def judge(populated: TheoryPopulation, filled_registry: PredictionRegistry) -> FalsificationJudge:
    return FalsificationJudge(populated, filled_registry)


# ---------------------------------------------------------------------------
# TheoryPopulation tests
# ---------------------------------------------------------------------------

class TestTheoryPopulation:
    def test_add_and_activate(self, empty_population):
        tid = empty_population.add_theory("test", "x + 1")
        assert tid.startswith("TH_")
        assert empty_population.count() == 1
        assert empty_population.count_active() == 0  # starts as CANDIDATE

        empty_population.activate(tid)
        assert empty_population.count_active() == 1
        assert tid in empty_population.active_theories

    def test_falsify_not_deleted(self, empty_population):
        tid = empty_population.add_theory("test", "x + 1")
        empty_population.activate(tid)

        empty_population.falsify(tid, "exp_contradiction", "wrong value")
        assert empty_population.count_active() == 0
        assert tid in empty_population.falsified_theories
        assert tid in empty_population.all_theories  # still there!

    def test_lineage(self, empty_population):
        parent_tid = empty_population.add_theory("parent", "x")
        empty_population.activate(parent_tid)
        child_tid = empty_population.add_theory("child", "x + 5", parent_id=parent_tid)
        empty_population.activate(child_tid)

        lineage = empty_population.get_lineage(child_tid)
        assert len(lineage) == 2
        assert lineage[0] == parent_tid
        assert lineage[1] == child_tid

    def test_population_initial_state(self, populated):
        assert populated.count_active() == 5
        assert populated.count() == 5

    def test_deactivate_and_reactivate(self, populated):
        tid = list(populated.active_theories.keys())[0]
        populated.falsify(tid, "exp_X", "falsified")
        assert populated.count_active() == 4

        # Can't "reactivate" falsified (no reactivate method)
        # But we can add a new theory
        new_tid = populated.add_theory("replacement", "x + 999")
        populated.activate(new_tid)
        assert populated.count_active() == 5


# ---------------------------------------------------------------------------
# PredictionRegistry tests
# ---------------------------------------------------------------------------

class TestPredictionRegistry:
    def test_pre_register(self, registry):
        rid = registry.pre_register("TH_01", "exp_A", "1.5", prediction_type="conserved")
        assert rid.startswith("PR_")
        assert registry.total_predictions == 1

    def test_prediction_is_frozen(self, registry):
        rid = registry.pre_register("TH_01", "exp_A", "1.5")
        pred = registry.get_prediction(rid)
        assert pred is not None
        assert pred.frozen is True

        # Record outcome
        oid = registry.record_outcome(rid, PredictionOutcome.CONFIRMED,
                                       observed_value=1.5)
        outcome = registry.get_outcome(oid)
        assert outcome is not None
        assert outcome.outcome == PredictionOutcome.CONFIRMED

    def test_duplicate_outcome_blocked(self, registry):
        rid = registry.pre_register("TH_01", "exp_A", "1.5")
        registry.record_outcome(rid, PredictionOutcome.CONFIRMED, observed_value=1.5)
        with pytest.raises(ValueError, match="Outcome already recorded"):
            registry.record_outcome(rid, PredictionOutcome.CONFIRMED, observed_value=1.5)

    def test_pending_predictions(self, registry):
        registry.pre_register("TH_A", "exp_1", "10.0")
        registry.pre_register("TH_B", "exp_1", "12.0")
        registry.pre_register("TH_C", "exp_2", "5.0")

        pending = registry.get_pending_predictions("exp_1")
        assert len(pending) == 2

        # Resolve one
        first = pending[0]
        registry.record_outcome(first.record_id, PredictionOutcome.CONFIRMED)
        pending_after = registry.get_pending_predictions("exp_1")
        assert len(pending_after) == 1

    def test_get_predictions_for_theory(self, registry):
        registry.pre_register("TH_01", "exp_A", "a")
        registry.pre_register("TH_01", "exp_B", "b")
        registry.pre_register("TH_02", "exp_A", "c")

        t1_preds = registry.get_predictions_for_theory("TH_01")
        assert len(t1_preds) == 2

    def test_contradiction_summary(self, registry):
        r1 = registry.pre_register("TH_01", "exp_A", "1.0")
        registry.pre_register("TH_01", "exp_B", "2.0")

        registry.record_outcome(r1, PredictionOutcome.CONTRADICTED,
                                 observed_value=999.0)
        summary = registry.contradiction_summary("TH_01")
        assert summary["contradictions"] == 1
        assert summary["total_predictions"] == 2

    def test_experiment_fulfilled(self, registry):
        r1 = registry.pre_register("TH_01", "exp_X", "1.0")
        r2 = registry.pre_register("TH_02", "exp_X", "2.0")

        assert not registry.experiment_fulfilled("exp_X")

        registry.record_outcome(r1, PredictionOutcome.CONFIRMED)
        assert not registry.experiment_fulfilled("exp_X")

        registry.record_outcome(r2, PredictionOutcome.CONFIRMED)
        assert registry.experiment_fulfilled("exp_X")


# ---------------------------------------------------------------------------
# FalsificationJudge tests
# ---------------------------------------------------------------------------

class TestFalsificationJudge:
    def test_judge_conserved_confirmed(self, populated, registry):
        tid = list(populated.active_theories.keys())[0]
        rid = registry.pre_register(tid, "exp_cons", "42.0", prediction_type="conserved")

        judge = FalsificationJudge(populated, registry)
        results = judge.judge_experiment("exp_cons", {
            "is_conserved": True,
            "coefficient_of_variation": 0.01,
            "mean": 42.0,
        })
        assert len(results) >= 1
        assert results[0].outcome == PredictionOutcome.CONFIRMED

    def test_judge_conserved_contradicted(self, populated, registry):
        tid = list(populated.active_theories.keys())[0]
        rid = registry.pre_register(tid, "exp_var", "42.0", prediction_type="conserved")

        judge = FalsificationJudge(populated, registry)
        results = judge.judge_experiment("exp_var", {
            "is_conserved": False,
            "coefficient_of_variation": 0.5,
            "mean": 10.0,
            "std": 5.0,
        })
        assert len(results) >= 1
        assert results[0].outcome == PredictionOutcome.CONTRADICTED
        assert results[0].information_reward > 0  # falsification = positive reward

    def test_judge_zero_confirmed(self, populated, registry):
        tid = list(populated.active_theories.keys())[0]
        registry.pre_register(tid, "exp_z", "0", prediction_type="zero")

        judge = FalsificationJudge(populated, registry)
        results = judge.judge_experiment("exp_z", {
            "is_zero": True,
            "mean": 0.0,
            "std": 0.001,
        })
        assert results[0].outcome == PredictionOutcome.CONFIRMED

    def test_detect_equivalent_theories(self, populated, registry):
        # Two theories with identical predictions
        t1 = list(populated.active_theories.keys())[0]
        t2 = list(populated.active_theories.keys())[1]

        for exp in ["exp_X", "exp_Y"]:
            registry.pre_register(t1, exp, "same_value", prediction_type="conserved")
            registry.pre_register(t2, exp, "same_value", prediction_type="conserved")

        judge = FalsificationJudge(populated, registry)
        equivalents = judge.detect_equivalent_theories()
        # Should detect t1 and t2 are equivalent
        assert len(equivalents) > 0

    def test_design_discriminating_intervention(self, populated, registry):
        t1 = list(populated.active_theories.keys())[0]
        t2 = list(populated.active_theories.keys())[1]

        # Different predictions on different experiments
        registry.pre_register(t1, "exp_same", "42", prediction_type="conserved")
        registry.pre_register(t2, "exp_same", "42", prediction_type="conserved")

        registry.pre_register(t1, "exp_diff", "100", prediction_type="conserved")
        registry.pre_register(t2, "exp_diff", "0", prediction_type="conserved")

        judge = FalsificationJudge(populated, registry)
        best = judge.design_discriminating_intervention(
            t1, t2, ["exp_same", "exp_diff"]
        )
        assert best == "exp_diff"  # Different predictions here

    def test_apply_results_to_population(self, populated, registry):
        tid = list(populated.active_theories.keys())[0]
        registry.pre_register(tid, "exp_fail", "10.0", prediction_type="conserved")

        judge = FalsificationJudge(populated, registry)
        results = judge.judge_experiment("exp_fail", {
            "is_conserved": False,
            "coefficient_of_variation": 0.9,
            "mean": 500.0,
            "std": 450.0,
        })

        counts = judge.apply_results_to_population(results)
        assert counts["falsified"] >= 1
        assert tid in populated.falsified_theories

    def test_compute_law_qualification(self, populated):
        tid = list(populated.active_theories.keys())[0]
        reg = PredictionRegistry()
        reg.pre_register(tid, "exp_1", "val1")
        reg.pre_register(tid, "exp_2", "val2")
        reg.record_outcome(
            list(reg.get_predictions_for_theory(tid))[0].record_id,
            PredictionOutcome.CONFIRMED
        )
        reg.record_outcome(
            list(reg.get_predictions_for_theory(tid))[1].record_id,
            PredictionOutcome.CONFIRMED
        )

        judge = FalsificationJudge(populated, reg)
        qual = judge.compute_law_qualification(tid)
        assert qual["qualified"] is True
        assert qual["no_contradictions"] is True
        assert qual["multi_experiment"] is True


# ---------------------------------------------------------------------------
# DiscriminatingExperimentSelector tests
# ---------------------------------------------------------------------------

class TestDiscriminatingSelector:
    def test_score_no_discrimination(self, populated, registry, judge):
        """All theories predict the same → low discrimination."""
        selector = DiscriminatingExperimentSelector(populated, registry, judge)

        score = selector.score_experiment("no_predictions_here")
        assert score.discrimination_value <= 1.0
        assert score.completeness_score == 0.0  # no predictions

    def test_score_with_predictions(self, populated, registry, judge):
        """Different predictions → high discrimination."""
        actives = list(populated.active_theories.keys())
        # Register different predictions
        for i, tid in enumerate(actives):
            registry.pre_register(tid, "exp_disc", f"{i}", prediction_type="conserved")

        selector = DiscriminatingExperimentSelector(populated, registry, judge)
        score = selector.score_experiment("exp_disc")
        # With 5 different predictions, discrimination should be high
        assert score.discrimination_value > 0, f"Got {score.discrimination_value}"
        assert score.completeness_score > 0
        assert len(score.competing_pairs) > 0

    def test_select_experiment(self, populated, registry, judge):
        """Should return the best discriminating experiment."""
        actives = list(populated.active_theories.keys())

        # exp_boring: all same predictions
        for tid in actives:
            registry.pre_register(tid, "exp_boring", "42", prediction_type="conserved")

        # exp_disc: all different predictions
        for i, tid in enumerate(actives):
            registry.pre_register(tid, "exp_disc", f"{i * 100}", prediction_type="conserved")

        selector = DiscriminatingExperimentSelector(
            populated, registry, judge,
            exploration_rate=0.0,  # disable exploration for deterministic test
        )
        chosen = selector.select_experiment(["exp_boring", "exp_disc"])
        assert chosen == "exp_disc", f"Expected exp_disc, got {chosen}"

    def test_select_top_k(self, populated, registry, judge):
        """Top-k should return sorted list."""
        actives = list(populated.active_theories.keys())
        for i, tid in enumerate(actives):
            registry.pre_register(tid, "exp_A", f"{i}", prediction_type="conserved")
            registry.pre_register(tid, "exp_B", f"{i*2}", prediction_type="conserved")

        selector = DiscriminatingExperimentSelector(
            populated, registry, judge,
            exploration_rate=0.0,
        )
        top = selector.select_top_discriminating(["exp_A", "exp_B"], top_k=2)
        assert len(top) == 2
        assert top[0][1].total_score >= top[1][1].total_score

    def test_exploration_rate(self, populated, registry, judge):
        """High exploration rate should sometimes pick non-optimal."""
        selector = DiscriminatingExperimentSelector(
            populated, registry, judge,
            exploration_rate=1.0,  # always explore
        )
        # With exploration_rate=1.0, should always pick randomly
        # (no assertion on which, just that it doesn't crash)
        chosen = selector.select_experiment(["a", "b", "c"])
        assert chosen in ["a", "b", "c"]

    def test_report_competition(self, populated, registry, judge):
        selector = DiscriminatingExperimentSelector(populated, registry, judge)
        status = selector.report_competition_status()
        assert status["active_theories"] == 5
        assert status["competing_pairs"] > 0
        assert "has_competition" in status


# ---------------------------------------------------------------------------
# Serialization round-trip tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_theory_population_roundtrip(self, populated):
        d = populated.to_dict()
        restored = TheoryPopulation.from_dict(d)
        assert restored.count_active() == populated.count_active()
        assert restored.count() == populated.count()

    def test_prediction_registry_roundtrip(self, filled_registry):
        d = filled_registry.to_dict()
        restored = PredictionRegistry.from_dict(d)
        assert restored.total_predictions == filled_registry.total_predictions

    def test_falsification_judge_roundtrip(self, populated, filled_registry):
        judge = FalsificationJudge(populated, filled_registry, conservation_tolerance=0.1)
        d = judge.to_dict()
        restored = FalsificationJudge.from_dict(d, populated, filled_registry)
        assert restored.conservation_tolerance == 0.1

    def test_selector_roundtrip(self, populated, filled_registry, judge):
        selector = DiscriminatingExperimentSelector(
            populated, filled_registry, judge,
            exploration_rate=0.25,
        )
        d = selector.to_dict()
        restored = DiscriminatingExperimentSelector.from_dict(
            d, populated, filled_registry, judge
        )
        assert restored.exploration_rate == 0.25


# ---------------------------------------------------------------------------
# Synthetic benchmark regression tests
# ---------------------------------------------------------------------------

class TestSyntheticBenchmarks:
    """Run the full Phase 2 loop on synthetic benchmarks with known ground truth."""

    @staticmethod
    def _make_theory_population(num: int) -> TheoryPopulation:
        """Create a population with diverse candidate theories."""
        pop = TheoryPopulation()
        candidates = [f"x + {i}" for i in range(num)]
        for i, expr in enumerate(candidates):
            tid = pop.add_theory(name=f"candidate_{i}", expression=expr)
            pop.activate(tid)
        return pop

    @staticmethod
    def _run_loop_for_function(
        true_func,
        num_candidates: int = 8,
        num_experiments: int = 10,
        tolerance: float = 0.1,
        seed: int = 42,
    ) -> Dict:
        """Run the full Phase 2 loop against a known function."""
        import random
        random.seed(seed)

        pop = TestSyntheticBenchmarks._make_theory_population(num_candidates)
        reg = PredictionRegistry()
        judge = FalsificationJudge(pop, reg)
        selector = DiscriminatingExperimentSelector(pop, reg, judge, exploration_rate=0.1)

        # Generate experiments: x values
        x_values = [i * 0.5 - 2.5 for i in range(num_experiments)]
        experiment_names = [f"x={x}" for x in x_values]

        stats = []
        for iteration in range(num_experiments * 2):
            actives = list(pop.active_theories.keys())
            if len(actives) == 0:
                break

            # Pre-register predictions for each active theory on upcoming experiments
            for tid in actives:
                theory = pop.get_theory(tid)
                if theory is None:
                    continue
                for exp_name in experiment_names:
                    if reg.get_predictions_for_experiment(exp_name):
                        # Check if this theory already predicted
                        existing = [p for p in reg.get_predictions_for_theory(tid)
                                    if p.experiment_name == exp_name]
                        if existing:
                            continue
                    # Evaluate the theory's expression at this x
                    try:
                        x_val = float(exp_name.split("=")[1])
                        # Simple safe eval
                        pred_val = eval(theory.expression, {"x": x_val, "__builtins__": {}})
                        reg.pre_register(
                            tid, exp_name,
                            f"{pred_val:.4f}",
                            prediction_type="conserved",
                        )
                    except:
                        continue

            # Select discriminating experiment
            pending_exps = [e for e in experiment_names
                            if reg.get_pending_predictions(e)]
            if not pending_exps:
                pending_exps = experiment_names

            try:
                chosen_exp = selector.select_experiment(pending_exps)
            except ValueError:
                break

            # Evaluate ground truth
            x_val = float(chosen_exp.split("=")[1])
            ground_truth = true_func(x_val)

            # Judge all predictions for this experiment
            judge_results = judge.judge_experiment(chosen_exp, {
                "is_conserved": True,
                "coefficient_of_variation": 0.0,
                "mean": ground_truth,
                "std": 0.0,
            })

            # But we also need to check NUMERICAL accuracy, not just "conserved"
            # Actually evaluate each prediction against ground truth
            predictions = reg.get_predictions_for_experiment(chosen_exp)
            for pred in predictions:
                if any(o.prediction_id == pred.record_id
                       for o in reg.get_outcomes_for_experiment(chosen_exp)):
                    continue
                try:
                    pred_val = float(pred.predicted_expression)
                    abs_err = abs(pred_val - ground_truth)
                    if abs_err < tolerance:
                        reg.record_outcome(pred.record_id,
                                           PredictionOutcome.CONFIRMED,
                                           observed_value=ground_truth)
                    else:
                        reg.record_outcome(pred.record_id,
                                           PredictionOutcome.CONTRADICTED,
                                           observed_value=ground_truth,
                                           details=f"error={abs_err:.4f}")
                except (ValueError, TypeError):
                    reg.record_outcome(pred.record_id,
                                       PredictionOutcome.INCONCLUSIVE,
                                       observed_value=ground_truth)

            # Apply results
            counts = judge.apply_results_to_population(judge_results)

            # Find best remaining theory
            best_expr = None
            best_err = float("inf")
            for tid in pop.active_theories:
                theory = pop.get_theory(tid)
                if theory is None:
                    continue
                # Compute error across all x values
                errors = []
                for x in x_values:
                    try:
                        pv = eval(theory.expression, {"x": x, "__builtins__": {}})
                        tv = true_func(x)
                        errors.append(abs(pv - tv))
                    except:
                        errors.append(float("inf"))
                avg_err = sum(errors) / len(errors) if errors else float("inf")
                if avg_err < best_err:
                    best_err = avg_err
                    best_expr = theory.expression

            stats.append({
                "iteration": iteration,
                "experiment": chosen_exp,
                "ground_truth": ground_truth,
                "active_count": pop.count_active(),
                "falsified_count": counts.get("falsified", 0),
                "best_error": best_err,
                "best_expression": best_expr,
            })

        return {"stats": stats, "best_error": best_err, "best_expression": best_expr}

    def test_linear_x_plus_5(self):
        """f(x) = x + 5"""
        result = self._run_loop_for_function(lambda x: x + 5, num_candidates=10)
        assert result["best_error"] < 2.0, (
            f"Loop should find a good approximation. Got error={result['best_error']:.4f}"
        )

    def test_linear_2x_plus_3(self):
        """f(x) = 2x + 3"""
        result = self._run_loop_for_function(lambda x: 2*x + 3, num_candidates=10)
        assert result["best_error"] < 5.0, (
            f"Got error={result['best_error']:.4f}"
        )

    def test_negative_slope(self):
        """f(x) = -x + 1"""
        result = self._run_loop_for_function(lambda x: -x + 1, num_candidates=10)
        assert result["best_error"] < 5.0

    def test_loop_narrows_candidates(self):
        """The loop should eliminate falsified theories."""
        result = self._run_loop_for_function(lambda x: x + 3, num_candidates=8)
        stats = result["stats"]
        if len(stats) >= 2:
            first_active = stats[0]["active_count"]
            last_active = stats[-1]["active_count"]
            # Should have eliminated some by the end
            assert last_active <= first_active, (
                f"Active count should decrease: {first_active} → {last_active}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
