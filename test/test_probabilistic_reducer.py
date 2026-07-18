import unittest
from types import MethodType, SimpleNamespace

import sympy as sp

from aiphy.probabilistic_reducer import (
    compare_localized_zero_ideals,
    ProbabilisticReducerConfig,
    RankCertificate,
    RandomizedKahlerRankOracle,
    evaluate_mod_prime,
    rational_jet_signature,
    rank_mod_prime,
)
from aiphy.general_model import GeneralLawService
from aiphy.manager import Manager


class FiniteFieldTests(unittest.TestCase):
    def test_rational_evaluation_and_rank_are_exact(self):
        x = sp.Symbol("x")
        prime = 1_000_000_007
        value = evaluate_mod_prime((x + 1) / 2, {x: 3}, prime)
        self.assertEqual(value, 2)
        self.assertEqual(rank_mod_prime([[1, 2], [2, 4]], prime), 1)
        self.assertEqual(rank_mod_prime([[1, 2], [2, 5]], prime), 2)

    def test_canonical_signature_merges_algebra_but_preserves_semantics(self):
        x, t = sp.symbols("x t")
        vx = sp.Function("vx")(t)
        vy = sp.Function("vy")(t)
        self.assertEqual(
            rational_jet_signature([x * (x + 1)]),
            rational_jet_signature([x ** 2 + x]),
        )
        self.assertNotEqual(
            rational_jet_signature([sp.diff(vx, t)]),
            rational_jet_signature([sp.diff(vy, t)]),
        )


class RandomizedKahlerRankTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y = sp.symbols("x y")
        self.oracle = RandomizedKahlerRankOracle(
            ProbabilisticReducerConfig(trials_per_prime=2)
        )

    def test_power_of_existing_invariant_is_dependent(self):
        result = self.oracle.compare([self.x], self.x ** 7, key="power")
        self.assertEqual(result.status, "dependent")

    def test_fractional_power_uses_numeric_kahler_fallback(self):
        result = self.oracle.compare(
            [self.x], self.x ** sp.Rational(3, 2), key="fractional-power"
        )
        self.assertEqual(result.status, "dependent")
        self.assertEqual(result.method, "numeric_transcendental_kahler_rank")
        self.assertEqual(result.rank_gain, 0)
        self.assertGreaterEqual(result.confidence_lower_bound, 0.999)

    def test_new_coordinate_is_independent(self):
        result = self.oracle.compare([self.x], self.y, key="coordinate")
        self.assertEqual(result.status, "independent")
        self.assertEqual(result.rank_gain, 1)

    def test_polynomial_closure_of_two_invariants_is_dependent(self):
        candidate = self.x ** 4 * self.y ** 3
        result = self.oracle.compare([self.x, self.y], candidate, key="closure")
        self.assertEqual(result.status, "dependent")

    def test_two_generators_are_kept_before_later_closure(self):
        first = self.x * self.y
        second = self.x * self.y ** 2
        third = self.x ** 2 * self.y ** 3
        self.assertEqual(self.oracle.compare([first], second).status, "independent")
        self.assertEqual(self.oracle.compare([first, second], third).status, "dependent")

    def test_transcendental_closure_is_dependent(self):
        result = self.oracle.compare([self.x], sp.sin(self.x), key="sine")
        self.assertEqual(result.status, "dependent")
        self.assertEqual(result.rank_gain, 0)

    def test_transcendental_new_coordinate_is_independent(self):
        result = self.oracle.compare([self.x], sp.sin(self.y), key="new-sine")
        self.assertEqual(result.status, "independent")
        self.assertEqual(result.rank_gain, 1)


class LocalizedZeroIdealTests(unittest.TestCase):
    def test_exact_linear_consequence_is_dependent(self):
        ax, ay, x, y = sp.symbols("ax ay x y")
        result = compare_localized_zero_ideals([ax, ay], x * ay - y * ax)
        self.assertEqual(result.status, "dependent")
        self.assertEqual(result.confidence_lower_bound, 1.0)

    def test_algebraic_equality_plus_conservation_implies_conservation(self):
        t = sp.Symbol("t_0")
        state = sp.Function("state")(t)
        constant = sp.Function("constant")(t)
        equality = state - constant
        result = compare_localized_zero_ideals(
            [equality, sp.diff(equality, t), sp.diff(constant, t)],
            sp.diff(state, t),
        )
        self.assertEqual(result.status, "dependent")

    def test_two_conserved_values_do_not_imply_their_equality(self):
        t = sp.Symbol("t_0")
        left = sp.Function("left")(t)
        right = sp.Function("right")(t)
        result = compare_localized_zero_ideals(
            [sp.diff(left, t), sp.diff(right, t)],
            left - right,
        )
        self.assertEqual(result.status, "independent")

    def test_rational_conserved_consequence_uses_true_localization(self):
        t = sp.Symbol("t_0")
        amount = sp.Function("amount")(t)
        gasconstant = sp.Function("gasconstant")(t)
        temperature = sp.Function("temperature")(t)
        pressure = sp.Function("pressure")(t)
        volume = sp.Function("volume")(t)
        state_equation = (
            pressure - amount * gasconstant * temperature / volume
        )
        conserved_ratio = amount * temperature / (pressure * volume)
        result = compare_localized_zero_ideals(
            [
                state_equation,
                sp.diff(state_equation, t),
                sp.diff(gasconstant, t),
            ],
            sp.diff(conserved_ratio, t),
        )
        self.assertEqual(result.status, "dependent")

    def test_isobaric_ratio_is_a_state_equation_consequence(self):
        t = sp.Symbol("t_0")
        amount = sp.Function("amount")(t)
        gasconstant = sp.Function("gasconstant")(t)
        temperature = sp.Function("temperature")(t)
        pressure = sp.Function("pressure")(t)
        volume = sp.Function("volume")(t)
        state_equation = (
            pressure - amount * gasconstant * temperature / volume
        )
        result = compare_localized_zero_ideals(
            [
                state_equation,
                sp.diff(state_equation, t),
                sp.diff(amount, t),
                sp.diff(gasconstant, t),
                sp.diff(pressure, t),
            ],
            sp.diff(temperature / volume, t),
        )
        self.assertEqual(result.status, "dependent")

    def test_applied_function_monomial_union_keeps_its_branches(self):
        t = sp.Symbol("t_0")
        left = sp.Function("left")(t)
        right = sp.Function("right")(t)
        result = compare_localized_zero_ideals([left * right], left)
        self.assertEqual(result.status, "independent")


class MinimumBasisExchangeTests(unittest.TestCase):
    def test_new_cheaper_generator_displaces_only_dependent_heavier_law(self):
        priorities = {"cheap": (0,), "new": (1,), "heavy": (2,), "other": (3,)}

        class Reducer:
            @staticmethod
            def priority_key(name):
                return priorities[name]

            @staticmethod
            def classify(name, basis):
                dependent = name == "heavy" and "new" in basis
                return RankCertificate(
                    status="dependent" if dependent else "independent",
                    basis_laws=["new"] if dependent else [],
                )

        service = SimpleNamespace(
            general_law_base=SimpleNamespace(
                general_laws={name: object() for name in priorities}
            ),
            marked_general_laws=set(priorities),
            rank_reducer=Reducer(),
            independence_certificates={},
        )
        archived = []

        def record(this, name, certificate):
            archived.append(name)
            this.marked_general_laws.remove(name)
            this.general_law_base.general_laws.pop(name)

        service._record_derived_law = MethodType(record, service)
        removed = GeneralLawService.exchange_heavier_basis_laws(service, "new")
        self.assertEqual(removed, ["heavy"])
        self.assertEqual(archived, ["heavy"])
        self.assertIn("cheap", service.marked_general_laws)
        self.assertIn("other", service.marked_general_laws)

    def test_scope_change_revalidates_and_triggers_exchange(self):
        law = SimpleNamespace(valid_experiments=["old", "new_scope"])

        class Reducer:
            @staticmethod
            def priority_key(name):
                return (0, name)

            @staticmethod
            def classify(name, basis):
                return RankCertificate(
                    status="independent",
                    candidate_scope_size=2,
                    candidate_experiments=["old", "new_scope"],
                    independent_experiments=["old", "new_scope"],
                )

        service = SimpleNamespace(
            general_law_base=SimpleNamespace(general_laws={"law": law}),
            marked_general_laws={"law"},
            rank_reducer=Reducer(),
            independence_certificates={
                "law": {"candidate_experiments": ["old"]}
            },
        )
        exchanged = []

        def exchange(this, name):
            exchanged.append(name)
            return []

        service.exchange_heavier_basis_laws = MethodType(exchange, service)
        service._record_derived_law = MethodType(lambda *_: None, service)
        removed = GeneralLawService.refresh_scope_changed_basis_laws(service)
        self.assertEqual(removed, [])
        self.assertEqual(exchanged, ["law"])
        self.assertEqual(
            service.independence_certificates["law"]["candidate_experiments"],
            ["old", "new_scope"],
        )


class ConceptCanonicalGateTests(unittest.TestCase):
    def test_existing_algebraic_aliases_are_indexed_and_removed_from_actions(self):
        class FakeConcept:
            def __init__(self, expression, complexity=3):
                self.exp = expression
                self.complexity = complexity
                self.is_sum = False
                self.objtype_id_map = {"Particle": {1}}

            def __str__(self):
                return str(self.exp)

        concepts = {
            "C_1": FakeConcept("x*(x+1)", 3),
            "C_2": FakeConcept("x**2+x", 4),
            "C_3": FakeConcept("y", 1),
        }

        class FakeCore:
            @staticmethod
            def raw_definition_exp(expression, _experiment):
                return expression

            @staticmethod
            def parse_exp_to_sympy_str(expression, _time):
                return str(expression)

        class FakeKnowledge:
            K = FakeCore()
            fetch_concepts = concepts

            @staticmethod
            def fetch_concept_by_name(name):
                return concepts.get(name)

        harness = object.__new__(Manager)
        harness.knowledge = FakeKnowledge()
        harness.memory = SimpleNamespace(
            concept_signatures={},
            concept_signatures_indexed=False,
            redundant_concepts=set(),
            action_bandit=SimpleNamespace(actions=set(concepts)),
        )
        harness._ensure_concept_signature_index()
        self.assertTrue(harness.memory.concept_signatures_indexed)
        self.assertEqual(harness.memory.redundant_concepts, {"C_2"})
        self.assertNotIn("C_2", harness.memory.action_bandit.actions)
        canonical, _ = harness._find_canonical_concept(
            FakeConcept("(x+1)*x", 5), None
        )
        self.assertEqual(canonical, "C_1")
        concepts.pop("C_1")
        canonical, _ = harness._find_canonical_concept(
            FakeConcept("x+x**2", 5), None
        )
        self.assertEqual(canonical, "C_2")
        self.assertEqual(harness.memory.redundant_concepts, set())
        self.assertIn("C_2", harness.memory.action_bandit.actions)


class LocalizedZeroIdealMoreTests(unittest.TestCase):

    def test_generic_coordinate_factors_are_localized_explicitly(self):
        c, y, z = sp.symbols("c y z")
        result = compare_localized_zero_ideals([(c - 1) * y], (c - 1) * z)
        self.assertEqual(result.status, "dependent")
        assumptions = " ".join(result.generic_nonzero_assumptions)
        self.assertIn("y", assumptions)
        self.assertIn("z", assumptions)

    def test_union_of_coordinate_branches_is_not_arbitrarily_collapsed(self):
        x, y = sp.symbols("x y")
        result = compare_localized_zero_ideals([x * y], x)
        self.assertEqual(result.status, "independent")

    def test_repeated_factor_has_same_radical_constraint(self):
        x = sp.Symbol("x")
        result = compare_localized_zero_ideals([x ** 4], x)
        self.assertEqual(result.status, "dependent")


if __name__ == "__main__":
    unittest.main()
