import unittest

import sympy as sp

from aiphy.probabilistic_reducer import RandomizedKahlerRankOracle
from aiphy.typed_expression import (
    Cos,
    Cross,
    Curl,
    Div,
    Dot,
    ExpFn,
    Grad,
    Log,
    Sin,
    TensorExpression,
    TypedSymbolTable,
    ValueKind,
    VectorExpression,
)


class TypedExpressionTests(unittest.TestCase):
    def setUp(self):
        symbols = TypedSymbolTable()
        self.position = symbols.vector(
            "r", ["posx[1]", "posy[1]", "posz[1]"]
        )
        self.velocity = symbols.vector(
            "v",
            [
                "D[posx[1]]/D[t[0]]",
                "D[posy[1]]/D[t[0]]",
                "D[posz[1]]/D[t[0]]",
            ],
        )
        self.field = symbols.vector(
            "B", ["bfieldx[1]", "bfieldy[1]", "bfieldz[1]"]
        )

    def test_dot_and_cross_are_typed_and_expand_to_scalar_dsl(self):
        energy = Dot(self.velocity, self.velocity)
        force = Cross(self.velocity, self.field)
        self.assertEqual(energy.kind, ValueKind.SCALAR)
        self.assertEqual(force.kind, ValueKind.VECTOR)
        self.assertEqual(force.dimension, 3)
        self.assertIn("bfieldz[1]", str(force[0]))
        self.assertIn("D[posy[1]]/D[t[0]]", str(force[0]))

    def test_grad_div_curl_have_formal_component_semantics(self):
        potential = Sin("posx[1]") + ExpFn("posy[1]") + Log("posz[1]")
        gradient = Grad(potential, self.position)
        divergence = Div(self.field, self.position)
        curl = Curl(self.field, self.position)
        self.assertEqual(gradient.dimension, 3)
        self.assertIn("Partial[", str(gradient[0]))
        self.assertIn("Partial[", str(divergence))
        self.assertEqual(curl.dimension, 3)

    def test_tensor_vector_product_checks_shape(self):
        identity = TensorExpression([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        transformed = identity * self.velocity
        self.assertEqual(transformed.dimension, 3)
        with self.assertRaises(TypeError):
            TensorExpression([[1, 0], [0, 1]]) * self.velocity

    def test_invalid_vector_operations_fail_early(self):
        with self.assertRaises(TypeError):
            Dot(self.velocity, VectorExpression(["posx[1]", "posy[1]"]))
        with self.assertRaises(TypeError):
            Cross(self.velocity, VectorExpression(["posx[1]", "posy[1]"]))


class TranscendentalReducerTests(unittest.TestCase):
    def test_function_of_existing_generator_is_dependent(self):
        x = sp.Symbol("x")
        certificate = RandomizedKahlerRankOracle().compare([x], sp.sin(x))
        self.assertEqual(certificate.status, "dependent")
        self.assertEqual(certificate.method, "numeric_transcendental_kahler_rank")

    def test_new_transcendental_coordinate_is_independent(self):
        x, y = sp.symbols("x y")
        certificate = RandomizedKahlerRankOracle().compare([x], sp.sin(y))
        self.assertEqual(certificate.status, "independent")

    def test_trigonometric_identity_canonicalizes(self):
        x = sp.Symbol("x")
        certificate = RandomizedKahlerRankOracle().compare(
            [], sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1
        )
        self.assertEqual(certificate.status, "dependent")


if __name__ == "__main__":
    unittest.main()
