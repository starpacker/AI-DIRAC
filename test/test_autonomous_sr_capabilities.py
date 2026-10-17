"""Slow, genuine-search capability tests.

Run explicitly with ``AI_NEWTON_RUN_SLOW_SR=1``.  These tests invoke Julia
PySR and intentionally prohibit all structured/domain formula seeds.
"""

import os
import unittest

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData

from aiphy.capability_experiments import (
    boltzmann_capability_data,
    boltzmann_generic_spec,
    lorentz_force_capability_data,
    lorentz_generic_spec,
    nabla_capability_fields,
)
from aiphy.pysr_regression import PySRConfig, PySRDiscovery


RUN_SLOW = os.getenv("AI_NEWTON_RUN_SLOW_SR") == "1"


@unittest.skipUnless(RUN_SLOW, "set AI_NEWTON_RUN_SLOW_SR=1 for real PySR searches")
class AutonomousSymbolicRegressionCapabilities(unittest.TestCase):
    def assert_genuine_search(self, engine):
        self.assertEqual(engine.last_search_audit["structured_candidates"], 0)
        self.assertEqual(engine.last_search_audit["domain_templates"], 0)
        self.assertGreaterEqual(engine.last_search_audit["model_fit_calls"], 1)
        self.assertGreater(engine.last_search_audit["model_equations"], 0)

    def test_lorentz_force_from_generic_typed_operators(self):
        engine = PySRDiscovery(
            PySRConfig(
                niterations=60, populations=8, max_complexity=10,
                max_features=20, timeout_seconds=60, random_state=2,
                max_train_nrmse=1e-6, max_holdout_nrmse=1e-6,
                max_holdout_bias=1e-6,
            )
        )
        result = engine.discover(
            lorentz_force_capability_data(),
            specification=lorentz_generic_spec(),
        )
        self.assertGreaterEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("charge[1]", law)
        self.assertIn("velocityy[1] * bfieldz[1]", law)
        self.assert_genuine_search(engine)

    def test_boltzmann_exponential_from_generic_operators(self):
        engine = PySRDiscovery(
            PySRConfig(
                niterations=80, populations=8, max_complexity=12,
                timeout_seconds=60, random_state=1,
                max_train_nrmse=1e-6, max_holdout_nrmse=1e-6,
                max_holdout_bias=1e-6,
            )
        )
        result = engine.discover(
            boltzmann_capability_data(),
            specification=boltzmann_generic_spec(),
        )
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("exp(", str(result[0][0]))
        self.assert_genuine_search(engine)

    def test_nabla_operator_is_selected_by_real_pysr(self):
        grid, scalar, _gradient, vector, divergence, _curl = nabla_capability_fields()
        engine = PySRDiscovery(
            PySRConfig(
                niterations=12, populations=4, max_complexity=8,
                timeout_seconds=25, random_state=3,
                max_train_nrmse=1e-8, max_holdout_nrmse=1e-8,
                max_holdout_bias=1e-8,
            )
        )
        result = engine.discover_spatial_operators(
            grid,
            "source[1]",
            divergence,
            scalar_fields={"potential[1]": scalar},
            vector_fields={"field": vector},
            generic_search=True,
        )
        self.assertGreaterEqual(len(result), 1)
        self.assertIn("Partial[fieldx[1]]", str(result[0][0]))
        self.assert_genuine_search(engine)

    def test_anonymous_noninteger_power_from_generic_operator(self):
        # Names and units are intentionally content-free.  Success therefore
        # demonstrates a genuine power-law capability, not a thermodynamics
        # template or a gamma-specific feature generator.
        coordinate = np.linspace(0.6, 3.4, 240)
        response = 3.125 * coordinate ** -1.37
        time = np.linspace(0.0, 1.0, coordinate.size)
        data = DataStruct.empty()
        for name, values in {
            "t[0]": time,
            "u[1]": coordinate,
            "w[1]": response,
        }.items():
            data.add_data(
                AtomExp(name),
                ExpData(np.repeat(values[None, :], 3, axis=0).tolist()),
            )
        engine = PySRDiscovery(
            PySRConfig(
                niterations=100, populations=10, max_complexity=9,
                max_features=4, timeout_seconds=90, random_state=7,
                max_train_nrmse=3e-4, max_holdout_nrmse=3e-4,
                max_holdout_bias=3e-4, max_rational_denominator=1000,
            )
        )
        from aiphy.pysr_regression import GenericSearchSpec
        result = engine.discover(
            data,
            specification=GenericSearchSpec(
                "w[1]", {"u[1]": "", "w[1]": ""}, {}
            ),
        )
        self.assertGreaterEqual(len(result), 1)
        law = str(result[0][0])
        self.assertIn("u[1] **", law)
        self.assert_genuine_search(engine)


if __name__ == "__main__":
    unittest.main()
