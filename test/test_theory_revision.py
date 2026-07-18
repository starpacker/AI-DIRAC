import os
import unittest

from aiphy.classical_revision_experiments import (
    HOOKE_LINEAR_LAW,
    PENDULUM_LINEAR_LAW,
    duffing_revision_spec,
    duffing_revision_training_data,
    duffing_revision_validation_data,
    pendulum_revision_spec,
    pendulum_revision_training_data,
    pendulum_revision_validation_data,
)
from aiphy.pysr_regression import PySRConfig, PySRDiscovery
from aiphy.theory_revision import (
    ActiveTheoryLaw,
    GeneralLawRevisionAdapter,
    RevisionControllerConfig,
    RevisionTheoryState,
    TheoryRevisionController,
)


class NoEquationRegressor:
    def __init__(self, **kwargs):
        self.equations_ = []

    def fit(self, matrix, target, variable_names):
        return self


class SineRegressor(NoEquationRegressor):
    def __init__(self, **kwargs):
        self.equations_ = [{"equation": "sin(g0)", "complexity": 2}]


def controller(regressor, *, acceptance=1e-6):
    discovery = PySRDiscovery(
        PySRConfig(
            max_features=8,
            max_train_nrmse=1e-6,
            max_holdout_nrmse=1e-6,
            max_holdout_bias=1e-6,
        ),
        regressor_class=regressor,
    )
    return TheoryRevisionController(
        discovery, RevisionControllerConfig(acceptance_nrmse=acceptance)
    )


class FixedClassicalTheoryRevisionTests(unittest.TestCase):
    def test_pendulum_internal_revision_atomically_replaces_old_law(self):
        old = ActiveTheoryLaw(
            "small_angle_pendulum",
            PENDULUM_LINEAR_LAW,
            "angularacceleration[1]",
            valid_experiments=["small_angle_reference"],
        )
        state = RevisionTheoryState.from_law(old)
        outcome = controller(SineRegressor).revise(
            state,
            old.name,
            pendulum_revision_training_data(),
            pendulum_revision_validation_data(),
            pendulum_revision_spec(),
            strategies=("internal",),
        )
        self.assertEqual(outcome.status, "revised")
        self.assertEqual(outcome.strategy, "internal")
        self.assertEqual(outcome.replaced_variable, "angle[1]")
        self.assertIn("sin(angle[1])", outcome.new_law.formula)
        self.assertNotIn(old.name, state.active_laws)
        self.assertIn(old.name, state.archived_laws)
        self.assertEqual(len(state.active_laws), 1)
        self.assertGreater(outcome.reward.total, 0.0)
        self.assertLess(
            max(item.nrmse for item in outcome.new_validation), 1e-6
        )

    def test_real_general_law_basis_commits_pendulum_revision_and_reward(self):
        from aiphy.core import Exp, Proposition
        from aiphy.experiment.revision_pendulum import (
            revision_pendulum_critical_config,
            revision_pendulum_small_config,
        )
        from aiphy.general_model import GeneralLawBase
        from aiphy.main import Theorist

        theorist = Theorist(num_threads=1)
        theorist.register_experiments([
            revision_pendulum_small_config(),
            revision_pendulum_critical_config(),
        ])
        old_residual = Exp(
            "angularacceleration[1] + "
            "gravity[1]*angle[1]/revisionlength[1]"
        )
        old_concept = theorist.knowledge.generalize_to_normal_concept(
            "revision_pendulum_small", old_residual
        )
        old_law = theorist.general.register_general_law(
            Proposition.Zero(old_concept),
            valid_experiments=["revision_pendulum_small"],
        )
        theorist.general.completion_exclude[old_law.name] = [
            "revision_pendulum_small"
        ]
        theorist.general.service.marked_general_laws.add(old_law.name)
        theorist.general.service.diffalg_for_marked_gl[old_law.name] = {}

        outcome = GeneralLawRevisionAdapter(
            theorist, controller(SineRegressor)
        ).revise_and_commit(
            old_law.name,
            "revision_pendulum_critical",
            pendulum_revision_training_data(),
            pendulum_revision_validation_data(),
            pendulum_revision_spec(),
            old_formula=PENDULUM_LINEAR_LAW,
            strategies=("internal",),
            scope_experiments=(
                "revision_pendulum_small",
                "revision_pendulum_critical",
            ),
            observation_mapping={"length[1]": "revisionlength[1]"},
        )

        self.assertEqual(outcome.status, "revised")
        self.assertEqual(outcome.commit.certificate["status"], "independent")
        self.assertNotIn(old_law.name, theorist.general.general_laws)
        self.assertEqual(len(theorist.general.general_laws), 1)
        new_law = theorist.general.general_laws[outcome.commit.new_law]
        self.assertIn("sin(angle[1])", str(new_law.prop))
        self.assertEqual(new_law.father_name, old_law.name)
        # Extended AST nodes are represented by deterministic opaque jets in
        # the legacy KeyValue index.  They therefore remain registerable,
        # while semantic dependence is decided by the numerical reducer.
        self.assertTrue(new_law.knowledge_registered)
        self.assertEqual(
            set(new_law.valid_experiments),
            {"revision_pendulum_small", "revision_pendulum_critical"},
        )
        self.assertEqual(outcome.adaptive_reward.revised_laws, 1)
        self.assertEqual(outcome.adaptive_reward.new_contradictions, 1)
        self.assertGreater(outcome.adaptive_reward.total, 10.0)
        archive = theorist.general.service.derived_general_laws[old_law.name]
        self.assertEqual(archive["replaced_by"], new_law.name)
        self.assertEqual(archive["archive_reason"], "falsified_and_revised")

        # A save/load cycle must retain an archived parent even though it no
        # longer exists in the active dictionary or the algebraic core index.
        restored = GeneralLawBase.from_json(
            theorist.general.to_json(),
            theorist.knowledge,
            theorist.specific,
            theorist.memory,
            theorist.objmodel,
        )
        restored_law = restored.general_laws[new_law.name]
        self.assertEqual(restored_law.father_name, old_law.name)
        self.assertTrue(restored_law.knowledge_registered)
        self.assertEqual(
            set(restored.service.marked_general_laws),
            set(restored.general_laws),
        )

    def test_duffing_additive_dimensional_fallback_replaces_hooke_law(self):
        old = ActiveTheoryLaw(
            "linear_hooke",
            HOOKE_LINEAR_LAW,
            "acceleration[1]",
            valid_experiments=["linear_reference"],
        )
        state = RevisionTheoryState.from_law(old)
        outcome = controller(NoEquationRegressor).revise(
            state,
            old.name,
            duffing_revision_training_data(),
            duffing_revision_validation_data(),
            duffing_revision_spec(),
            strategies=("additive",),
        )
        self.assertEqual(outcome.status, "revised")
        self.assertEqual(outcome.strategy, "additive")
        self.assertIn("position[1] ** 3", outcome.new_law.formula)
        self.assertIn("cubic[1]", outcome.new_law.formula)
        self.assertNotIn(old.name, state.active_laws)
        self.assertEqual(len(state.active_laws), 1)
        self.assertEqual(outcome.reducer_certificate["status"], "independent")
        self.assertLess(
            max(item.nrmse for item in outcome.new_validation), 1e-6
        )

    def test_real_general_law_basis_commits_duffing_revision(self):
        from aiphy.core import Exp, Proposition
        from aiphy.experiment.revision_duffing import (
            revision_duffing_critical_config,
            revision_duffing_linear_config,
        )
        from aiphy.main import Theorist

        theorist = Theorist(num_threads=1)
        theorist.register_experiments([
            revision_duffing_linear_config(),
            revision_duffing_critical_config(),
        ])
        old_concept = theorist.knowledge.generalize_to_normal_concept(
            "revision_duffing_linear",
            Exp(
                "revisionacceleration[1] + "
                "revisionstiffness[1]*revisionposition[1]/revisionmass[1]"
            ),
        )
        old = theorist.general.register_general_law(
            Proposition.Zero(old_concept),
            valid_experiments=["revision_duffing_linear"],
        )
        theorist.general.completion_exclude[old.name] = [
            "revision_duffing_linear"
        ]
        theorist.general.service.marked_general_laws.add(old.name)
        theorist.general.service.diffalg_for_marked_gl[old.name] = {}
        outcome = GeneralLawRevisionAdapter(
            theorist, controller(NoEquationRegressor)
        ).revise_and_commit(
            old.name,
            "revision_duffing_critical",
            duffing_revision_training_data(),
            duffing_revision_validation_data(),
            duffing_revision_spec(),
            old_formula=HOOKE_LINEAR_LAW,
            strategies=("additive",),
            scope_experiments=(
                "revision_duffing_linear",
                "revision_duffing_critical",
            ),
            observation_mapping={
                "acceleration[1]": "revisionacceleration[1]",
                "position[1]": "revisionposition[1]",
                "mass[1]": "revisionmass[1]",
                "stiffness[1]": "revisionstiffness[1]",
                "cubic[1]": "revisioncubic[1]",
            },
        )
        self.assertEqual(outcome.status, "revised")
        self.assertEqual(outcome.commit.certificate["status"], "independent")
        self.assertEqual(len(theorist.general.general_laws), 1)
        new_law = theorist.general.general_laws[outcome.commit.new_law]
        self.assertIn("revisionposition[1] ** 3", str(new_law.prop))
        self.assertEqual(new_law.father_name, old.name)
        self.assertEqual(outcome.adaptive_reward.revised_laws, 1)
        self.assertEqual(outcome.adaptive_reward.new_contradictions, 1)


@unittest.skipUnless(
    os.getenv("AI_NEWTON_RUN_SLOW_REVISION") == "1",
    "set AI_NEWTON_RUN_SLOW_REVISION=1 for genuine PySR revision",
)
class GenuinePySRTheoryRevisionTest(unittest.TestCase):
    def test_pysr_discovers_and_commits_pendulum_internal_substitution(self):
        from aiphy.core import Exp, Proposition
        from aiphy.experiment.revision_pendulum import (
            revision_pendulum_critical_config,
            revision_pendulum_small_config,
        )
        from aiphy.main import Theorist

        discovery = PySRDiscovery(
            PySRConfig(
                niterations=50,
                populations=6,
                max_complexity=10,
                max_features=6,
                timeout_seconds=50,
                random_state=17,
                max_train_nrmse=1e-6,
                max_holdout_nrmse=1e-6,
                max_holdout_bias=1e-6,
            )
        )
        theorist = Theorist(num_threads=1)
        theorist.register_experiments([
            revision_pendulum_small_config(),
            revision_pendulum_critical_config(),
        ])
        old_concept = theorist.knowledge.generalize_to_normal_concept(
            "revision_pendulum_small",
            Exp(
                "angularacceleration[1] + "
                "gravity[1]*angle[1]/revisionlength[1]"
            ),
        )
        old = theorist.general.register_general_law(
            Proposition.Zero(old_concept),
            valid_experiments=["revision_pendulum_small"],
        )
        theorist.general.completion_exclude[old.name] = [
            "revision_pendulum_small"
        ]
        theorist.general.service.marked_general_laws.add(old.name)
        theorist.general.service.diffalg_for_marked_gl[old.name] = {}
        outcome = GeneralLawRevisionAdapter(
            theorist,
            TheoryRevisionController(
                discovery, RevisionControllerConfig(acceptance_nrmse=1e-6)
            ),
        ).revise_and_commit(
            old.name,
            "revision_pendulum_critical",
            pendulum_revision_training_data(),
            pendulum_revision_validation_data(),
            pendulum_revision_spec(),
            old_formula=PENDULUM_LINEAR_LAW,
            strategies=("internal",),
            scope_experiments=(
                "revision_pendulum_small",
                "revision_pendulum_critical",
            ),
            observation_mapping={"length[1]": "revisionlength[1]"},
        )
        self.assertEqual(outcome.status, "revised")
        self.assertIn("sin(angle[1])", outcome.search.new_law.formula)
        self.assertEqual(outcome.commit.certificate["status"], "independent")
        self.assertEqual(outcome.adaptive_reward.revised_laws, 1)
        self.assertEqual(outcome.adaptive_reward.new_contradictions, 1)
        self.assertEqual(
            outcome.search.search_audit["structured_candidates"], 0
        )
        self.assertEqual(outcome.search.search_audit["domain_templates"], 0)


if __name__ == "__main__":
    unittest.main()
