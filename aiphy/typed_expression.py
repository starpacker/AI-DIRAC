"""Typed physical expressions built on the scalar AI-Newton ``Exp`` DSL.

The existing knowledge, experiment and reducer layers store scalar laws.  A
vector or tensor expression is therefore represented as a checked collection
of scalar ``Exp`` components.  Vector identities expand deterministically to
those components, so no information is lost and all existing validation code
continues to apply component by component.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .core import AtomExp, Exp


class ValueKind(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    TENSOR = "tensor"


def _exp(value: Exp | str | int) -> Exp:
    if isinstance(value, Exp):
        return value
    return Exp(str(value))


@dataclass(frozen=True)
class ScalarExpression:
    expression: Exp

    def __init__(self, expression: Exp | str | int):
        object.__setattr__(self, "expression", _exp(expression))

    @property
    def kind(self) -> ValueKind:
        return ValueKind.SCALAR

    def __str__(self) -> str:
        return str(self.expression)

    def _binary(self, other, operator: str) -> "ScalarExpression":
        other = as_scalar(other)
        return ScalarExpression(f"({self}) {operator} ({other})")

    def __add__(self, other):
        return self._binary(other, "+")

    def __sub__(self, other):
        return self._binary(other, "-")

    def __mul__(self, other):
        if isinstance(other, VectorExpression):
            return other * self
        if isinstance(other, TensorExpression):
            return other * self
        return self._binary(other, "*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._binary(other, "/")

    def __neg__(self):
        return ScalarExpression(f"-({self})")

    def __pow__(self, power: int):
        if not isinstance(power, int):
            raise TypeError("AI-Newton scalar powers must be integers")
        return ScalarExpression(f"({self}) ** {power}")


@dataclass(frozen=True)
class VectorExpression:
    components: tuple[ScalarExpression, ...]

    def __init__(self, components: Sequence[ScalarExpression | Exp | str | int]):
        converted = tuple(as_scalar(component) for component in components)
        if not converted:
            raise ValueError("A vector must contain at least one component")
        object.__setattr__(self, "components", converted)

    @property
    def kind(self) -> ValueKind:
        return ValueKind.VECTOR

    @property
    def dimension(self) -> int:
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __getitem__(self, index: int) -> ScalarExpression:
        return self.components[index]

    def _same_dimension(self, other: "VectorExpression") -> None:
        if self.dimension != other.dimension:
            raise TypeError(
                f"Vector dimensions differ: {self.dimension} != {other.dimension}"
            )

    def __add__(self, other):
        other = as_vector(other)
        self._same_dimension(other)
        return VectorExpression([left + right for left, right in zip(self, other)])

    def __sub__(self, other):
        other = as_vector(other)
        self._same_dimension(other)
        return VectorExpression([left - right for left, right in zip(self, other)])

    def __neg__(self):
        return VectorExpression([-component for component in self])

    def __mul__(self, scalar):
        scalar = as_scalar(scalar)
        return VectorExpression([component * scalar for component in self])

    def __rmul__(self, scalar):
        return self * scalar

    def __truediv__(self, scalar):
        scalar = as_scalar(scalar)
        return VectorExpression([component / scalar for component in self])


@dataclass(frozen=True)
class TensorExpression:
    rows: tuple[tuple[ScalarExpression, ...], ...]

    def __init__(self, rows: Sequence[Sequence[ScalarExpression | Exp | str | int]]):
        converted = tuple(tuple(as_scalar(value) for value in row) for row in rows)
        if not converted or not converted[0]:
            raise ValueError("A tensor must have a non-empty rectangular shape")
        width = len(converted[0])
        if any(len(row) != width for row in converted):
            raise ValueError("Tensor rows must have equal length")
        object.__setattr__(self, "rows", converted)

    @property
    def kind(self) -> ValueKind:
        return ValueKind.TENSOR

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    def __mul__(self, other):
        if isinstance(other, VectorExpression):
            if self.shape[1] != other.dimension:
                raise TypeError(f"Tensor/vector shapes differ: {self.shape} and {other.dimension}")
            return VectorExpression(
                [
                    sum_scalars(value * component for value, component in zip(row, other))
                    for row in self.rows
                ]
            )
        scalar = as_scalar(other)
        return TensorExpression([[value * scalar for value in row] for row in self.rows])

    def __rmul__(self, scalar):
        return self * scalar


def as_scalar(value) -> ScalarExpression:
    if isinstance(value, ScalarExpression):
        return value
    if isinstance(value, (VectorExpression, TensorExpression)):
        raise TypeError(f"Expected scalar, received {value.kind.value}")
    return ScalarExpression(value)


def as_vector(value) -> VectorExpression:
    if not isinstance(value, VectorExpression):
        kind = getattr(value, "kind", type(value).__name__)
        raise TypeError(f"Expected vector, received {kind}")
    return value


def sum_scalars(values: Iterable[ScalarExpression]) -> ScalarExpression:
    values = iter(values)
    try:
        result = next(values)
    except StopIteration:
        return ScalarExpression(0)
    for value in values:
        result = result + value
    return result


def Sin(value) -> ScalarExpression:
    value = as_scalar(value)
    return ScalarExpression(f"sin({value})")


def Cos(value) -> ScalarExpression:
    value = as_scalar(value)
    return ScalarExpression(f"cos({value})")


def ExpFn(value) -> ScalarExpression:
    value = as_scalar(value)
    return ScalarExpression(f"exp({value})")


def Log(value) -> ScalarExpression:
    value = as_scalar(value)
    return ScalarExpression(f"log({value})")


def Dot(left, right) -> ScalarExpression:
    left, right = as_vector(left), as_vector(right)
    left._same_dimension(right)
    return sum_scalars(a * b for a, b in zip(left, right))


def Cross(left, right) -> VectorExpression:
    left, right = as_vector(left), as_vector(right)
    if left.dimension != 3 or right.dimension != 3:
        raise TypeError("Cross product is defined only for three-dimensional vectors")
    a, b, c = left
    x, y, z = right
    return VectorExpression([b * z - c * y, c * x - a * z, a * y - b * x])


def _coordinate_atoms(coordinates: VectorExpression) -> tuple[AtomExp, ...]:
    atoms: list[AtomExp] = []
    for coordinate in coordinates:
        expression_atoms = list(coordinate.expression.all_atoms)
        if len(expression_atoms) != 1 or str(expression_atoms[0]) != str(coordinate.expression):
            raise TypeError("Spatial coordinates must be atomic scalar expressions")
        atoms.append(expression_atoms[0])
    return tuple(atoms)


def _partial(value: ScalarExpression, coordinate: AtomExp) -> ScalarExpression:
    return ScalarExpression(value.expression.__partial__(coordinate))


def Grad(scalar, coordinates) -> VectorExpression:
    scalar = as_scalar(scalar)
    coordinates = as_vector(coordinates)
    return VectorExpression([_partial(scalar, atom) for atom in _coordinate_atoms(coordinates)])


def Div(vector, coordinates) -> ScalarExpression:
    vector, coordinates = as_vector(vector), as_vector(coordinates)
    vector._same_dimension(coordinates)
    return sum_scalars(
        _partial(component, atom)
        for component, atom in zip(vector, _coordinate_atoms(coordinates))
    )


def Curl(vector, coordinates) -> VectorExpression:
    vector, coordinates = as_vector(vector), as_vector(coordinates)
    if vector.dimension != 3 or coordinates.dimension != 3:
        raise TypeError("Curl is defined only for three-dimensional fields")
    fx, fy, fz = vector
    x, y, z = _coordinate_atoms(coordinates)
    return VectorExpression(
        [
            _partial(fz, y) - _partial(fy, z),
            _partial(fx, z) - _partial(fz, x),
            _partial(fy, x) - _partial(fx, y),
        ]
    )


class TypedSymbolTable:
    """Explicitly declare physical variables and prevent type collisions."""

    def __init__(self) -> None:
        self._symbols: dict[str, ScalarExpression | VectorExpression | TensorExpression] = {}

    def _register(self, name: str, value):
        if name in self._symbols:
            raise ValueError(f"Symbol {name!r} is already declared")
        self._symbols[name] = value
        return value

    def scalar(self, name: str, expression: Exp | str | None = None) -> ScalarExpression:
        return self._register(name, ScalarExpression(expression or name))

    def vector(self, name: str, components: Sequence[Exp | str]) -> VectorExpression:
        return self._register(name, VectorExpression(components))

    def tensor(self, name: str, rows: Sequence[Sequence[Exp | str]]) -> TensorExpression:
        return self._register(name, TensorExpression(rows))

    def __getitem__(self, name: str):
        return self._symbols[name]


# Lower-case aliases mirror the scalar DSL spelling.
sin = Sin
cos = Cos
exp = ExpFn
log = Log
dot = Dot
cross = Cross
grad = Grad
div = Div
curl = Curl

