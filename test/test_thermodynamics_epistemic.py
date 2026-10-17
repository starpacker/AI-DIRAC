import unittest

import sympy as sp

from aiphy import Exp, ExpData
from aiphy.core import Knowledge as CoreKnowledge
from aiphy.experiment.thermodynamics_epistemic import (
    epistemic_gas_state_scan_config,
)
from aiphy.interface import Knowledge
from aiphy.main import Theorist
from aiphy.operational_vocabulary import _ratio_candidates
from aiphy.experiment.challenge_thermodynamics import (
    rod_four_node_diffusion_config,
    rod_three_node_diffusion_config,
    three_body_chain_relaxation_config,
    three_body_ring_relaxation_config,
    two_body_contact_equal_config,
    two_body_contact_unequal_config,
)


class EpistemicThermodynamicsTests(unittest.TestCase):
    def test_native_ast_and_key_index_accept_exact_fractional_powers(self):
        core = CoreKnowledge.default()
        expressions = (
            "pressure[1] * volume[1] ** (7/5)",
            "pressure[1] / temperature[1] ** (7/2)",
            "volume[1] ** (1/2)",
            "volume[1] ** (-1/3)",
        )

        for source in expressions:
            expression = Exp(source)
            self.assertIn("**", str(expression))
            key = core.eval_exp_keyvaluehashed(expression, None)
            self.assertFalse(key.is_none)
            self.assertFalse(key.is_const)

    def test_fractional_power_real_domain_evaluation(self):
        sqrt_four = ExpData.from_const(4.0, 0.0).__powq__(1, 2)
        cbrt_negative = ExpData.from_const(-27.0, 0.0).__powq__(1, 3)
        two_thirds = ExpData.from_const(-8.0, 0.0).__powq__(2, 3)
        invalid_even_root = ExpData.from_const(-4.0, 0.0).__powq__(1, 2)

        self.assertAlmostEqual(sqrt_four.const_data.mean, 2.0)
        self.assertAlmostEqual(cbrt_negative.const_data.mean, -3.0)
        self.assertAlmostEqual(two_thirds.const_data.mean, 4.0)
        self.assertTrue(invalid_even_root.is_err)

    def test_fractional_power_general_laws_register_without_panicking(self):
        knowledge = Knowledge.default()
        for name in ("pressure", "volume", "temperature"):
            self.assertEqual(
                knowledge.register_basic_concept(
                    f"(1->Particle) |- {name}[1]"
                ),
                name,
            )

        first = knowledge.register_conclusion(
            "(1->Particle) |- "
            "(pressure[1] * (volume[1] ** (7 / 5))) is conserved"
        )
        second = knowledge.register_conclusion(
            "(1->Particle) |- "
            "(pressure[1] / (temperature[1] ** (7 / 2))) is conserved"
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIn("** (7 / 5)", str(knowledge.fetch_conclusions[first]))
        self.assertIn("** (7 / 2)", str(knowledge.fetch_conclusions[second]))

    def test_difficult_thermal_benchmarks_do_not_embed_answer_laws(self):
        factories = (
            two_body_contact_equal_config,
            two_body_contact_unequal_config,
            three_body_chain_relaxation_config,
            three_body_ring_relaxation_config,
            rod_three_node_diffusion_config,
            rod_four_node_diffusion_config,
        )
        self.assertTrue(all(not factory().geometry_info for factory in factories))

    def test_operational_miner_prefers_simplest_unit_ratio(self):
        t = sp.Symbol("t_0")
        pressure = sp.Function("pressure_1")(t)
        volume = sp.Function("volume_1")(t)
        temperature = sp.Function("temperature_1")(t)
        samplemass = sp.Function("samplemass_1")(t)
        molarmass = sp.Function("molarmass_1")(t)
        expression = (
            pressure
            - sp.Rational(4157, 500) * samplemass * temperature
            / (molarmass * volume)
        )

        candidates = _ratio_candidates(expression)

        self.assertTrue(candidates)
        self.assertEqual(
            candidates[0]["definition"],
            "(1->Particle) |- (samplemass[1] / molarmass[1])",
        )

    def test_hidden_simulator_quantities_are_not_observations(self):
        theorist = Theorist(num_threads=1)
        theorist.register_experiment(epistemic_gas_state_scan_config())
        structure = theorist.knowledge.fetch_expstruct("epistemic_gas_state_scan")
        observed = {
            item.atomexp_name for item in structure.original_concept
        }

        self.assertEqual(
            observed,
            {"pressure", "volume", "temperature", "samplemass", "molarmass", "t"},
        )
        self.assertNotIn("amount", observed)
        self.assertNotIn("gasconstant", observed)
        self.assertNotIn("gamma", observed)


if __name__ == "__main__":
    unittest.main()
