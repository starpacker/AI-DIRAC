from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from aiphy.experiment import Concept


DataKey = Tuple[Concept, List[str]]


class ColumnRole(str, Enum):
    DEFAULT = "DEFAULT"
    UNLOCKED = "UNLOCKED"
    CONTROL = "CONTROL"
    HIDDEN = "HIDDEN"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    key: DataKey
    role: ColumnRole = ColumnRole.DEFAULT
    provenance: str = ""
    unlock: str | None = None

    @property
    def is_open(self) -> bool:
        return self.role is not ColumnRole.HIDDEN

    @property
    def required_unlock(self) -> str | None:
        if self.role is not ColumnRole.UNLOCKED:
            return None
        return self.unlock or self.key[0].atomexp_name


def concept(expr: str) -> Concept:
    return Concept(expr)


def particle_concept(name: str) -> Concept:
    return Concept(f"(1->Particle) |- {name}[1]")


def clock_concept(name: str) -> Concept:
    return Concept(f"(1->Clock) |- {name}[1]")


def key(concept_obj: Concept, *objects: str) -> DataKey:
    return (concept_obj, list(objects))


def open_data_info(specs: Sequence[ColumnSpec]) -> List[DataKey]:
    return [spec.key for spec in specs if spec.is_open]


def open_names(specs: Sequence[ColumnSpec]) -> set[str]:
    return {spec.name for spec in specs if spec.is_open}


def hidden_names(specs: Sequence[ColumnSpec]) -> set[str]:
    return {spec.name for spec in specs if not spec.is_open}


def apply_role_overrides(
    specs: Sequence[ColumnSpec],
    overrides: Mapping[str, ColumnRole | str] | None = None,
) -> List[ColumnSpec]:
    """Return a copy of specs with selected column roles replaced.

    This is the strict-ladder switch: the same simulator column can be hidden in
    its discovery experiment and opened later once the ladder registry certifies
    the corresponding concept.
    """
    if not overrides:
        return list(specs)
    known = {spec.name for spec in specs}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise KeyError(f"Unknown column role override(s): {unknown}")
    normalized = {
        name: role if isinstance(role, ColumnRole) else ColumnRole[str(role).upper()]
        for name, role in overrides.items()
    }
    return [
        ColumnSpec(spec.name, spec.key, normalized.get(spec.name, spec.role), spec.provenance, spec.unlock)
        for spec in specs
    ]


def resolve_column_specs(
    specs: Sequence[ColumnSpec],
    unlocked_concepts: Iterable[str] = (),
    overrides: Mapping[str, ColumnRole | str] | None = None,
) -> List[ColumnSpec]:
    """Resolve candidate UNLOCKED columns against certified framework state."""
    specs = apply_role_overrides(specs, overrides)
    unlocked = set(unlocked_concepts)
    resolved: List[ColumnSpec] = []
    for spec in specs:
        role = spec.role
        if spec.role is ColumnRole.UNLOCKED and spec.required_unlock not in unlocked:
            role = ColumnRole.HIDDEN
        resolved.append(ColumnSpec(spec.name, spec.key, role, spec.provenance, spec.unlock))
    return resolved


def insert_open_columns(data_struct, specs: Sequence[ColumnSpec], values: Mapping[str, Iterable[float]]) -> None:
    allowed = open_names(specs)
    for spec in specs:
        if spec.name not in allowed:
            continue
        if spec.name not in values:
            raise KeyError(f"Open column `{spec.name}` has no generated values.")
        data_struct.insert_data(spec.key, np.asarray(values[spec.name], dtype=float))


def audit_specs(specs: Sequence[ColumnSpec], target_names: Iterable[str] = ()) -> None:
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate column specs: {duplicates}")
    target_set = {name.lower() for name in target_names}
    leaked = [spec.name for spec in specs if spec.is_open and spec.name.lower() in target_set]
    if leaked:
        raise ValueError(f"Target-named columns are open: {leaked}")
