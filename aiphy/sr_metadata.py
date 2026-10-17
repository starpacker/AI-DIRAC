"""Type/unit metadata routing for the production symbolic-regression path.

This registry describes quantities, never equations.  It is deliberately
separate from PySR feature generation so experiment modules can add new
observables without adding a candidate physical law.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from .core import DataStruct


@dataclass(frozen=True)
class QuantityMetadata:
    unit: str = ""
    vector_family: str | None = None
    axis: str | None = None


_QUANTITIES: dict[str, QuantityMetadata] = {}
_EXPERIMENT_TARGETS: dict[str, tuple[str, ...]] = {}


def register_quantity(
    name: str,
    *,
    unit: str = "",
    vector_family: str | None = None,
    axis: str | None = None,
) -> None:
    if axis is not None and axis not in "xyz":
        raise ValueError("Vector axes must be x, y, or z")
    _QUANTITIES[name.lower()] = QuantityMetadata(unit, vector_family, axis)


def register_vector(stem: str, *, unit: str, family: str | None = None) -> None:
    family = family or stem
    for axis in "xyz":
        register_quantity(
            f"{stem}{axis}", unit=unit, vector_family=family, axis=axis
        )


def register_experiment_targets(exp_name: str, targets: Sequence[str]) -> None:
    """Optionally declare measured responses, not a formula or predictor set."""

    _EXPERIMENT_TARGETS[exp_name] = tuple(targets)


def quantity_metadata(name: str) -> QuantityMetadata:
    """Return public measurement metadata for operational-concept mining."""

    return _QUANTITIES.get(name.lower(), QuantityMetadata())


def _base(expression: str) -> str:
    return expression.split("[", 1)[0].lower()


def infer_generic_specifications(
    data: DataStruct,
    *,
    exp_name: str | None = None,
    max_targets: int = 3,
):
    """Build typed target problems from registered quantity metadata."""

    # Lazy import avoids making the registry depend on Julia/PySR at import time.
    from .pysr_regression import GenericSearchSpec, VectorVariableSpec

    expressions = sorted(
        (str(key) for key in data.data_keys if str(key) != "t[0]")
    )
    if not expressions:
        return []
    metadata = {
        expression: _QUANTITIES.get(_base(expression), QuantityMetadata())
        for expression in expressions
    }
    vector_groups: dict[tuple[str, str], dict[str, str]] = {}
    for expression, item in metadata.items():
        if item.vector_family is None or item.axis is None:
            continue
        ids_match = re.search(r"\[([^\]]+)\]$", expression)
        ids = ids_match.group(1) if ids_match else ""
        vector_groups.setdefault((item.vector_family, ids), {})[item.axis] = expression

    vectors = {}
    consumed: set[str] = set()
    for (family, ids), components in vector_groups.items():
        if set(components) != {"x", "y", "z"}:
            continue
        ordered = tuple(components[axis] for axis in "xyz")
        unit = metadata[ordered[0]].unit
        name = f"{family}[{ids}]" if ids else family
        vectors[name] = VectorVariableSpec(ordered, unit)
        consumed.update(ordered)

    scalars = {
        expression: metadata[expression].unit
        for expression in expressions
        if expression not in consumed
    }
    # Time is a universally typed predictor when it is present.
    if "t[0]" in {str(key) for key in data.data_keys}:
        scalars["t[0]"] = "s"

    declared_targets = _EXPERIMENT_TARGETS.get(exp_name or "", ())
    if declared_targets:
        target_candidates = [
            expression for expression in expressions
            if expression in declared_targets or _base(expression) in declared_targets
        ]
    else:
        # No semantic target priority: prefer statistically informative traces
        # and use a stable expression tie-break.
        scored = []
        for expression in expressions:
            item = data.fetch_data_by_str(expression)
            try:
                # Accessing ``normal_data`` on ConstData/ZeroData is not a
                # normal Python AttributeError: the pyo3 getter panics in the
                # Rust core.  Check the tagged variant before unwrapping it.
                if item is None or not item.is_normal:
                    # Constant/zero observations are useful predictors, but
                    # fitting them as supervised targets is ill-posed and is
                    # already covered by the baseline conservation search.
                    continue
                values = np.asarray(item.normal_data.data, dtype=float)
                score = float(np.nanstd(values))
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            scored.append((-score, expression))
        target_candidates = [expression for _, expression in sorted(scored)]

    specifications = []
    for target in target_candidates[:max_targets]:
        target_scalars = dict(scalars)
        # A vector component target obtains its unit from the vector declaration.
        if target not in target_scalars and not any(
            target in vector.components for vector in vectors.values()
        ):
            target_scalars[target] = metadata[target].unit
        specifications.append(GenericSearchSpec(target, target_scalars, dict(vectors)))
    return specifications


# Common physical quantity declarations.  These are type/unit facts only.
register_vector("pos", unit="m", family="position")
register_vector("velocity", unit="m/s")
register_vector("force", unit="kg*m/s^2")
register_vector("efield", unit="kg*m/(A*s^3)", family="electric_field")
register_vector("bfield", unit="kg/(A*s^2)", family="magnetic_field")
register_quantity("charge", unit="A*s")
register_quantity("energy", unit="kg*m^2/s^2")
register_quantity("probability", unit="")
register_quantity("density", unit="")
register_quantity("pressure", unit="kg/(m*s^2)")
register_quantity("volume", unit="m^3")
register_quantity("amount", unit="mol")
register_quantity("temperature", unit="K")
register_quantity("temp", unit="K")
register_quantity("attraction", unit="kg*m^5/(s^2*mol^2)")
register_quantity("covolume", unit="m^3/mol")
register_quantity("gasconstant", unit="kg*m^2/(s^2*mol*K)")
register_quantity("samplemass", unit="kg")
register_quantity("molarmass", unit="kg/mol")
