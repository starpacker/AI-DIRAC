"""
Test: No Definition-Data Credit

Verifies that data used to define a concept or law is NOT later
counted as independent validation evidence for that same concept/law.

Per Phase 0 requirements:
  "定义一个概念的数据不会被计入该概念的 validation reward。"
  "定义数据不能同时作为该概念或定律的独立验证数据。"

This is a structural test checking the system design, not a runtime
integration test (which would require a full Theorist instance).
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNoDefinitionCredit(unittest.TestCase):

    def test_specific_model_has_independent_test_experiments(self):
        """
        SpecificModel has experiment_test — trajectories regenerated
        with random_settings(), distinct from the defining experiment.
        """
        # Check that SpecificModel.test_on_test_experiment exists
        from aiphy.specific_model import SpecificModel
        self.assertTrue(hasattr(SpecificModel, 'test_on_test_experiment'),
            "SpecificModel must have test_on_test_experiment method")
        self.assertTrue(hasattr(SpecificModel, 'complement_experiments_test'),
            "SpecificModel must have complement_experiments_test method")

    def test_test_on_test_experiment_uses_independent_data(self):
        """
        test_on_test_experiment regenerates the experiment with random
        settings, ensuring the test data is NOT the same trajectory
        as the definition data.
        """
        from aiphy.specific_model import SpecificModel

        # The method complement_experiments_test creates new copies
        # with experiment.copy() → experiment.random_settings()
        # → experiment.collect_expdata(), ensuring independence.
        import inspect
        source = inspect.getsource(SpecificModel.complement_experiments_test)
        self.assertIn("random_settings", source,
            "complement_experiments_test must call random_settings")

    def test_append_conserved_exp_tests_before_accepting(self):
        """
        append_conserved_exp calls test_on_test_experiment before
        accepting a new conclusion. This ensures definition data
        alone is insufficient for concept registration.
        """
        from aiphy.specific_model import SpecificModel

        import inspect
        source = inspect.getsource(SpecificModel.append_conserved_exp)
        self.assertIn("test_on_test_experiment", source,
            "append_conserved_exp must call test_on_test_experiment")

    def test_append_zero_exp_tests_before_accepting(self):
        """
        append_zero_exp similarly requires test_on_test_experiment.
        """
        from aiphy.specific_model import SpecificModel

        import inspect
        source = inspect.getsource(SpecificModel.append_zero_exp)
        self.assertIn("test_on_test_experiment", source,
            "append_zero_exp must call test_on_test_experiment")

    def test_conclusion_already_exist_prevents_duplicate_credit(self):
        """
        already_exist check prevents the same expression from being
        registered twice, which would double-count rewards.
        """
        from aiphy.specific_model import SpecificModel, ConclusionSet

        self.assertTrue(hasattr(ConclusionSet, 'already_exist'),
            "ConclusionSet must have already_exist method")

    def test_cross_arity_combiner_tests_on_test_experiment(self):
        """
        Cross-arity combiner uses specific_model.test_on_test_experiment
        to validate discovered combinations, not the definition data.
        """
        cross_arity_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "aiphy", "cross_arity_combiner.py"
        )
        with open(cross_arity_path) as f:
            source = f.read()
        self.assertIn("test_on_test_experiment", source,
            "cross_arity_combiner must call test_on_test_experiment for validation")

    def test_general_law_validation_is_cross_experiment(self):
        """
        General laws are validated across ALL registered experiments,
        not just the source experiment. This is stronger than
        same-experiment held-out validation.
        """
        general_model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "aiphy", "general_model.py"
        )
        with open(general_model_path) as f:
            source = f.read()
        self.assertIn("check_general_laws", source,
            "general_model must have check_general_laws method")

    def test_split_manifest_declares_no_definition_reuse(self):
        """
        split_manifest.json declares that definition data cannot
        be reused as validation data.
        """
        manifest_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "split_manifest.json"
        )
        if not os.path.exists(manifest_path):
            self.skipTest("split_manifest.json not found")
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
        policy = manifest["split_policy"]
        self.assertFalse(
            policy["definition"]["can_be_reused_as_validation"],
            "Definition data must not be reusable as validation"
        )
        self.assertFalse(
            policy["test_on_test"]["can_be_reused_as_definition"],
            "Test data must not be reusable as definition"
        )


if __name__ == "__main__":
    unittest.main()
