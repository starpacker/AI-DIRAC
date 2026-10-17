import unittest
from unittest.mock import patch

import numpy as np

from aiphy.capability_experiments import (
    boltzmann_capability_data,
    real_gas_generic_spec,
    lorentz_force_capability_data,
    nabla_capability_fields,
    real_gas_capability_data,
)
from aiphy.pysr_regression import PySRConfig, PySRDiscovery
from aiphy.regression import search_relations_pysr


def capability_engine(**overrides):
    values = dict(
        niterations=1,
        populations=1,
        max_complexity=12,
        max_targets=1,
        max_features=8,
        max_candidates=1,
        timeout_seconds=15,
        max_train_nrmse=1e-4,
        max_holdout_nrmse=1e-4,
        max_holdout_bias=1e-4,
        enable_domain_templates=True,
    )
    values.update(overrides)
    return PySRDiscovery(PySRConfig(**values))


class ExtendedSymbolicCapabilities(unittest.TestCase):
    def test_lorentz_force_cross_product_is_discovered(self):
        result = capability_engine().discover(lorentz_force_capability_data())
        self.assertEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("charge[1]", law)
        self.assertIn("velocity", law)
        self.assertIn("bfield", law)

    def test_boltzmann_exponential_is_discovered(self):
        result = capability_engine().discover(boltzmann_capability_data())
        self.assertEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("exp(", law)
        self.assertIn("energy[1]", law)
        self.assertIn("probability[1]", law)

    def test_nabla_grad_div_curl_match_analytic_fields(self):
        grid, scalar, expected_grad, vector, expected_div, expected_curl = (
            nabla_capability_fields()
        )
        actual_grad = grid.grad(scalar)
        actual_div = grid.div(vector)
        actual_curl = grid.curl(vector)
        for actual, expected in zip(actual_grad, expected_grad):
            np.testing.assert_allclose(actual, expected, atol=6e-3)
        np.testing.assert_allclose(actual_div, expected_div, atol=1e-12)
        for actual, expected in zip(actual_curl, expected_curl):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_nabla_divergence_relation_is_discovered(self):
        grid, scalar, _expected_grad, vector, expected_div, _expected_curl = (
            nabla_capability_fields()
        )
        result = capability_engine().discover_spatial_operators(
            grid,
            "source[1]",
            expected_div,
            scalar_fields={"potential[1]": scalar},
            vector_fields={"field": vector},
        )
        self.assertEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("source[1]", law)
        self.assertIn("Partial[fieldx[1]]/Partial[spacex[0]]", law)
        self.assertIn("Partial[fieldz[1]]/Partial[spacez[0]]", law)

    def test_real_gas_data_rejects_ideal_and_discovers_correction(self):
        data = real_gas_capability_data()
        engine = capability_engine(max_features=6)
        features = engine.extract_features(data)
        by_name = {feature.name: feature for feature in features}
        pressure = next(
            feature.values for feature in features if feature.expression == "pressure[1]"
        )
        ideal_residual = pressure - by_name["ideal_gas_pressure"].values
        corrected_residual = pressure - (
            by_name["real_gas_repulsion"].values
            - by_name["real_gas_attraction"].values
        )
        self.assertGreater(
            np.sqrt(np.mean(ideal_residual**2)) / np.std(pressure), 0.05
        )
        self.assertLess(
            np.sqrt(np.mean(corrected_residual**2)) / np.std(pressure), 1e-7
        )
        result = engine.discover(data)
        self.assertEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("covolume[1]", law)
        self.assertIn("attraction[1]", law)
        self.assertIn("pressure[1]", law)

    def test_real_gas_is_discovered_without_domain_templates(self):
        engine = PySRDiscovery(
            PySRConfig(
                max_train_nrmse=1e-7,
                max_holdout_nrmse=1e-7,
                max_holdout_bias=1e-7,
            ),
            use_pysr=False,
        )
        result = engine.discover(
            real_gas_capability_data(),
            specification=real_gas_generic_spec(),
        )
        self.assertEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("volume[1] - (amount[1] * covolume[1])", law)
        self.assertIn("attraction[1]", law)
        self.assertEqual(engine.last_search_audit["domain_templates"], 0)
        self.assertEqual(engine.last_search_audit["structured_candidates"], 0)
        self.assertEqual(engine.last_search_audit["entrypoint"], "discover")
        self.assertLess(engine.last_search_audit["holdout_nrmse"], 1e-7)

    @patch("aiphy.regression.search_relations_ver3_and_pca", return_value=[])
    def test_production_pysr_entrypoint_discovers_real_gas(self, _baseline):
        result = search_relations_pysr(
            real_gas_capability_data(), exp_name="capability_real_gas"
        )
        self.assertTrue(result)
        laws = "\n".join(str(expression) for expression, _ in result)
        self.assertIn("covolume[1]", laws)
        self.assertIn("attraction[1]", laws)


if __name__ == "__main__":
    unittest.main()
