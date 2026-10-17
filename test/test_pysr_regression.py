import unittest
from unittest.mock import patch

import numpy as np

from aiphy.core import AtomExp, DataStruct, Exp, ExpData
from aiphy.pysr_regression import GenericSearchSpec, PySRConfig, PySRDiscovery
from aiphy.regression import search_relations_pysr
from aiphy.sr_metadata import infer_generic_specifications
from aiphy.capability_experiments import (
    lorentz_force_capability_data,
    lorentz_generic_spec,
    real_gas_capability_data,
)


def oscillation_data() -> DataStruct:
    time = np.linspace(0.0, 4.0 * np.pi, 240)
    offsets = np.array([-2.0, -1.0, 1.0, 2.0])[:, None] * 1e-7
    traces = np.sin(time)[None, :] + offsets
    data = DataStruct.empty()
    data.add_data(AtomExp("t[0]"), ExpData(np.repeat(time[None, :], 4, axis=0).tolist()))
    data.add_data(AtomExp("posx[1]"), ExpData(traces.tolist()))
    return data


class CorrectRegressor:
    def __init__(self, **kwargs):
        self.equations_ = [{"equation": "-v0", "complexity": 2}]

    def fit(self, matrix, target, variable_names):
        self.variable_names = variable_names
        return self


class WrongRegressor(CorrectRegressor):
    def __init__(self, **kwargs):
        self.equations_ = [{"equation": "v0", "complexity": 2}]


class RedundantRegressor(CorrectRegressor):
    def __init__(self, **kwargs):
        self.equations_ = [
            {"equation": "-v0 + v0 / 100000", "complexity": 5},
            {"equation": "-v0", "complexity": 2},
        ]


class CosineRegressor(CorrectRegressor):
    def __init__(self, **kwargs):
        self.equations_ = []

    def fit(self, matrix, target, variable_names):
        self.variable_names = variable_names
        self.equations_ = [
            {"equation": "(cos(v1) + 1) / 2", "complexity": 5}
        ]
        return self


class NoEquationRegressor(CorrectRegressor):
    def __init__(self, **kwargs):
        self.equations_ = []


class AnonymousPowerRegressor(CorrectRegressor):
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.equations_ = [
            {"equation": "2.75 * (g0 ^ -1.375)", "complexity": 5}
        ]


def equation_regressor(equation):
    class EquationRegressor:
        def __init__(self, **kwargs):
            self.equations_ = [{"equation": equation, "complexity": 5}]

        def fit(self, matrix, target, variable_names):
            return self

    return EquationRegressor


def update_data():
    x = np.linspace(0.2, 1.2, 120)
    y = x + x**2
    data = DataStruct.empty()
    time = np.linspace(0.0, 1.0, x.size)
    data.add_data(AtomExp("t[0]"), ExpData(np.repeat(time[None, :], 4, axis=0).tolist()))
    data.add_data(AtomExp("x[1]"), ExpData(np.repeat(x[None, :], 4, axis=0).tolist()))
    data.add_data(AtomExp("y[1]"), ExpData(np.repeat(y[None, :], 4, axis=0).tolist()))
    return data


def anonymous_power_data():
    x = np.linspace(0.55, 3.2, 180)
    y = 2.75 * x ** -1.375
    data = DataStruct.empty()
    time = np.linspace(0.0, 1.0, x.size)
    for name, values in {"t[0]": time, "x[1]": x, "y[1]": y}.items():
        data.add_data(
            AtomExp(name),
            ExpData(np.repeat(values[None, :], 3, axis=0).tolist()),
        )
    return data


class TestPySRRegression(unittest.TestCase):
    def setUp(self):
        self.config = PySRConfig(
            max_targets=1,
            max_features=8,
            max_train_nrmse=0.03,
            max_holdout_nrmse=0.04,
            max_holdout_bias=0.03,
        )

    def test_core_ast_and_data_support_exact_rational_powers(self):
        expression = Exp("x[1] ** (3 / 2)")
        self.assertIn("x[1] ** (3 / 2)", str(expression))

        positive = ExpData(
            [[4.0, 9.0, 16.0, 25.0], [4.0, 9.0, 16.0, 25.0]]
        ).__powq__(1, 2)
        np.testing.assert_allclose(
            np.asarray(positive.normal_data.data),
            [[2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0]],
        )
        negative = ExpData(
            [[-8.0, -27.0, -64.0, -125.0], [-8.0, -27.0, -64.0, -125.0]]
        ).__powq__(1, 3)
        np.testing.assert_allclose(
            np.asarray(negative.normal_data.data),
            [[-2.0, -3.0, -4.0, -5.0], [-2.0, -3.0, -4.0, -5.0]],
        )

    def test_feature_extraction_averages_all_repeats(self):
        discovery = PySRDiscovery(self.config, use_pysr=False)
        features = discovery.extract_features(oscillation_data())
        raw = next(feature for feature in features if feature.name == "v0")
        expected = np.sin(np.linspace(0.0, 4.0 * np.pi, 240))
        np.testing.assert_allclose(raw.values, expected, atol=1e-12)

    def test_metadata_target_scoring_handles_const_data_without_rust_panic(self):
        time = np.linspace(0.0, 1.0, 120)
        data = DataStruct.empty()
        data.add_data(
            AtomExp("t[0]"),
            ExpData(np.repeat(time[None, :], 4, axis=0).tolist()),
        )
        data.add_data(
            AtomExp("posx[1]"),
            ExpData(np.repeat(np.sin(time)[None, :], 4, axis=0).tolist()),
        )
        data.add_data(AtomExp("length[1]"), ExpData.from_const(2.0, 1e-9))

        specifications = infer_generic_specifications(data, max_targets=2)

        self.assertEqual(specifications[0].target, "posx[1]")
        self.assertEqual([item.target for item in specifications], ["posx[1]"])
        self.assertIn("length[1]", specifications[0].scalars)

    def test_dimensional_aliases_prefer_formula_without_constant_nuisance(self):
        time = np.linspace(0.0, 1.0, 96)
        amount = 1.2 + 0.1 * np.sin(3.0 * time)
        temperature = 2.0 + 0.2 * np.cos(5.0 * time)
        volume = 1.5 + 0.15 * np.sin(7.0 * time)
        gasconstant = np.full_like(time, 1.1)
        gamma = np.full_like(time, 1.4)
        pressure = amount * gasconstant * temperature / volume
        data = DataStruct.empty()
        for name, values in {
            "amount[1]": amount,
            "temperature[1]": temperature,
            "volume[1]": volume,
            "gasconstant[1]": gasconstant,
            "gamma[1]": gamma,
            "pressure[1]": pressure,
        }.items():
            data.add_data(
                AtomExp(name),
                ExpData(np.repeat(values[None, :], 3, axis=0).tolist()),
            )
        discovery = PySRDiscovery(self.config, use_pysr=False)
        specification = GenericSearchSpec(
            "pressure[1]",
            {
                "amount[1]": "mol",
                "gamma[1]": "",
                "gasconstant[1]": "kg*m^2/(s^2*mol*K)",
                "temperature[1]": "K",
                "volume[1]": "m^3",
                "pressure[1]": "kg/(m*s^2)",
            },
            {},
        )

        results = discovery.discover_dimensional_typed(data, specification)

        self.assertTrue(results)
        expression = next(
            str(result[0]) for result in results
            if "gasconstant[1]" in str(result[0])
        )
        self.assertNotIn("gamma", expression)
        self.assertIn("amount[1]", expression)
        self.assertIn("gasconstant[1]", expression)
        self.assertIn("temperature[1]", expression)
        self.assertIn("volume[1]", expression)

    def test_dimensional_search_can_discover_a_new_unitful_constant(self):
        time = np.linspace(0.0, 1.0, 96)
        mass = 1.1 + 0.2 * np.sin(2.0 * time)
        molar_mass = 0.8 + 0.1 * np.cos(3.0 * time)
        temperature = 2.0 + 0.3 * np.sin(5.0 * time)
        volume = 1.4 + 0.2 * np.cos(7.0 * time)
        pressure = 1.2 * mass * temperature / (molar_mass * volume)
        data = DataStruct.empty()
        for name, values in {
            "samplemass[1]": mass,
            "molarmass[1]": molar_mass,
            "temperature[1]": temperature,
            "volume[1]": volume,
            "pressure[1]": pressure,
        }.items():
            data.add_data(
                AtomExp(name),
                ExpData(np.repeat(values[None, :], 3, axis=0).tolist()),
            )
        discovery = PySRDiscovery(self.config, use_pysr=False)
        specification = GenericSearchSpec(
            "pressure[1]",
            {
                "samplemass[1]": "kg",
                "molarmass[1]": "kg/mol",
                "temperature[1]": "K",
                "volume[1]": "m^3",
                "pressure[1]": "kg/(m*s^2)",
            },
            {},
        )

        results = discovery.discover_dimensional_typed(data, specification)

        self.assertTrue(results)
        expression = next(
            str(result[0]) for result in results
            if "molarmass[1]" in str(result[0])
        )
        self.assertIn("samplemass[1]", expression)
        self.assertIn("molarmass[1]", expression)
        self.assertIn("temperature[1]", expression)
        self.assertIn("volume[1]", expression)

    def test_invalid_typed_target_is_skipped_without_aborting_other_targets(self):
        data = oscillation_data()
        discovery = PySRDiscovery(
            PySRConfig(
                max_targets=2,
                max_features=8,
                max_train_nrmse=0.03,
                max_holdout_nrmse=0.04,
                max_holdout_bias=0.03,
            ),
            regressor_class=NoEquationRegressor,
        )
        invalid = GenericSearchSpec(
            "missing[1]", {"missing[1]": "", "posx[1]": "m"}, {}
        )
        valid = GenericSearchSpec(
            "posx[1]", {"posx[1]": "m", "t[0]": "s"}, {}
        )

        self.assertEqual(
            discovery.discover(data, specification=[invalid, valid]), []
        )
        self.assertEqual(len(discovery.last_search_audit["targets"]), 2)

    def test_default_extractor_contains_no_domain_formula_features(self):
        discovery = PySRDiscovery(self.config, use_pysr=False)
        names = {feature.name for feature in discovery.extract_features(real_gas_capability_data())}
        self.assertFalse(any(name.startswith("real_gas_") for name in names))
        self.assertFalse(any(name.startswith("lorentz_") for name in names))

    def test_generic_vector_operators_survive_default_feature_budget(self):
        discovery = PySRDiscovery(
            PySRConfig(max_features=8), regressor_class=NoEquationRegressor
        )
        discovery.discover(
            lorentz_force_capability_data(), specification=lorentz_generic_spec()
        )
        audited = discovery.last_search_audit["features"]
        self.assertTrue(any(item["base"].startswith("Cross(") for item in audited))
        self.assertTrue(any(item["expression"] == "charge[1]" for item in audited))

    def test_holdout_validated_equation_is_admitted(self):
        discovery = PySRDiscovery(
            self.config, regressor_class=CorrectRegressor
        )
        results = discovery.discover(oscillation_data())
        self.assertTrue(results)
        expression, data = results[0]
        self.assertIn("D[D[posx[1]]/D[t[0]]]", str(expression))
        self.assertTrue(data.is_conserved)

    def test_bad_equation_is_rejected(self):
        discovery = PySRDiscovery(self.config, regressor_class=WrongRegressor)
        # Isolate the injected regressor proposal.  The production discovery
        # also has a structured affine seed which independently and correctly
        # recovers x'' = -x from this benchmark.
        with patch.object(discovery, "_structured_candidates", return_value=[]):
            self.assertEqual(discovery.discover(oscillation_data()), [])

    @patch("aiphy.pysr_regression.propose_relations_pysr")
    @patch("aiphy.regression.search_relations_ver3_and_pca")
    def test_main_pysr_runs_even_when_baseline_has_dynamics(self, baseline, propose):
        expected = [(Exp("D[posx[1]]/D[t[0]]"), ExpData.from_const(1.0, 1e-8))]
        baseline.return_value = expected
        self.assertEqual(search_relations_pysr(DataStruct.empty()), expected)
        propose.assert_called_once()

    @patch("aiphy.pysr_regression.propose_relations_pysr")
    @patch("aiphy.regression.search_relations_ver3_and_pca")
    def test_pysr_runs_when_baseline_has_no_dynamics(self, baseline, propose):
        baseline.return_value = [(Exp("posx[1]"), ExpData.from_const(1.0, 1e-8))]
        proposed = (Exp("D[D[posx[1]]/D[t[0]]]/D[t[0]] + posx[1]"),
                    ExpData.from_const(0.0, 1e-8))
        propose.return_value = [proposed]
        result = search_relations_pysr(DataStruct.empty())
        self.assertIn(proposed, result)
        propose.assert_called_once()

    def test_only_simplest_validated_equation_for_a_target_is_kept(self):
        discovery = PySRDiscovery(self.config, regressor_class=RedundantRegressor)
        results = discovery.discover(oscillation_data())
        self.assertEqual(len(results), 1)
        self.assertEqual(
            str(results[0][0]),
            "(D[D[posx[1]]/D[t[0]]]/D[t[0]] + posx[1])",
        )

    def test_missing_optional_dependency_is_a_clean_noop(self):
        discovery = PySRDiscovery(self.config, use_pysr=False)
        self.assertFalse(discovery.available)
        self.assertEqual(discovery.discover(oscillation_data()), [])

    def test_transcendental_candidate_roundtrips_through_dsl(self):
        time = np.linspace(0.0, 2.0 * np.pi, 240)
        data = DataStruct.empty()
        repeated_time = np.repeat(time[None, :], 4, axis=0)
        data.add_data(AtomExp("t[0]"), ExpData(repeated_time.tolist()))
        data.add_data(AtomExp("phase[1]"), ExpData(repeated_time.tolist()))
        intensity = 0.5 * (1.0 + np.cos(repeated_time))
        data.add_data(AtomExp("intensity[1]"), ExpData(intensity.tolist()))
        discovery = PySRDiscovery(
            PySRConfig(
                max_targets=1,
                max_features=6,
                max_train_nrmse=1e-5,
                max_holdout_nrmse=1e-5,
                max_holdout_bias=1e-5,
                enable_domain_templates=True,
            ),
            regressor_class=CosineRegressor,
        )
        results = discovery.discover(data)
        self.assertEqual(len(results), 1)
        self.assertIn("cos(phase[1])", str(results[0][0]))

    def test_anonymous_fractional_power_roundtrips_through_native_ast(self):
        discovery = PySRDiscovery(
            PySRConfig(
                max_targets=1,
                max_features=4,
                max_train_nrmse=1e-8,
                max_holdout_nrmse=1e-8,
                max_holdout_bias=1e-8,
                max_rational_denominator=64,
            ),
            regressor_class=AnonymousPowerRegressor,
        )
        specification = GenericSearchSpec(
            "y[1]", {"x[1]": "", "y[1]": ""}, {}
        )

        results = discovery.discover(
            anonymous_power_data(), specification=specification
        )

        self.assertGreaterEqual(len(results), 2)
        laws = " ".join(str(expression) for expression, _data in results)
        self.assertIn("x[1] ** (11 / 8)", laws)
        self.assertNotIn("gamma", laws.lower())
        self.assertTrue(any(data.is_conserved for _expression, data in results))
        self.assertIn("^", AnonymousPowerRegressor.last_kwargs["binary_operators"])
        self.assertEqual(
            AnonymousPowerRegressor.last_kwargs["constraints"]["^"], (-1, 1)
        )

    def test_additive_theory_update_searches_residual(self):
        discovery = PySRDiscovery(
            self.config, regressor_class=equation_regressor("g0 ** 2")
        )
        results = discovery.discover_theory_updates(
            update_data(),
            GenericSearchSpec("y[1]", {"x[1]": "", "y[1]": ""}, {}),
            "x[1]",
            strategies=("additive",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].strategy, "additive")
        self.assertIn("x[1] ** 2", str(results[0].expression))

    def test_multiplicative_theory_update_searches_ratio(self):
        discovery = PySRDiscovery(
            self.config, regressor_class=equation_regressor("g0 + 1")
        )
        results = discovery.discover_theory_updates(
            update_data(),
            GenericSearchSpec("y[1]", {"x[1]": "", "y[1]": ""}, {}),
            "x[1]",
            strategies=("multiplicative",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].strategy, "multiplicative")

    def test_internal_ast_substitution_replaces_old_variable(self):
        discovery = PySRDiscovery(
            self.config, regressor_class=equation_regressor("g0 + g0 ** 2")
        )
        results = discovery.discover_theory_updates(
            update_data(),
            GenericSearchSpec("y[1]", {"x[1]": "", "y[1]": ""}, {}),
            "x[1]",
            strategies=("internal",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].replaced_variable, "x[1]")

    def test_skeleton_update_returns_mutated_full_formula(self):
        discovery = PySRDiscovery(
            self.config, regressor_class=equation_regressor("g0 + g0 ** 2")
        )
        results = discovery.discover_theory_updates(
            update_data(),
            GenericSearchSpec("y[1]", {"x[1]": "", "y[1]": ""}, {}),
            "x[1]",
            strategies=("skeleton",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].strategy, "skeleton")


if __name__ == "__main__":
    unittest.main()
