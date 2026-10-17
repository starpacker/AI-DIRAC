"""Adapter from AI-Newton conserved relations to operational trials.

Experiment modules may declare apparatus affordances (observable channel,
reference slot, and controllable nuisances).  They do not declare the hidden
quantity or its equation.  If the ordinary symbolic-regression pipeline later
finds a linear conserved relation over that observable, this router asks the
independent additive-invariant miner to recover the anonymous object weights
from measured traces and places the result on probation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from .operational_concepts import (
    AdditiveInvariantMiner,
    ChannelOrigin,
    ChannelRef,
    InvariantObservation,
    MeasurementProtocol,
)


@dataclass(frozen=True)
class AdditiveOperationalExperimentSpec:
    experiment_name: str
    observable: str
    reference_object_id: int
    entity_type: str
    nuisance_controls: tuple[str, ...] = ()
    alternate_experiment_families: tuple[str, ...] = ()
    applicable_experiment_families: tuple[str, ...] = ()


_ADDITIVE_SPECS: dict[str, AdditiveOperationalExperimentSpec] = {}


def register_additive_operational_experiment(
    spec: AdditiveOperationalExperimentSpec,
) -> None:
    """Declare apparatus capabilities, never a target coefficient or law."""

    _ADDITIVE_SPECS[spec.experiment_name] = spec


def additive_operational_spec(
    experiment_name: str,
) -> AdditiveOperationalExperimentSpec | None:
    return _ADDITIVE_SPECS.get(experiment_name)


def _function_identity(atom: AppliedUndef) -> tuple[str, int] | None:
    match = re.fullmatch(r"(.+)_([0-9]+)", atom.func.__name__)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _linear_entities(
    expression: sp.Expr, observable: str
) -> tuple[int, ...] | None:
    """Recognize a linear additive relation without using its coefficients."""

    atoms = []
    for atom in expression.atoms(AppliedUndef):
        identity = _function_identity(atom)
        if identity is None or identity[0] != observable:
            return None
        atoms.append((atom, identity[1]))
    if len(atoms) < 2 or len({object_id for _, object_id in atoms}) != len(atoms):
        return None
    reconstructed = sp.Integer(0)
    for atom, _ in atoms:
        coefficient = sp.simplify(sp.diff(expression, atom))
        if not coefficient.is_number:
            return None
        reconstructed += coefficient * atom
    remainder = sp.simplify(expression - reconstructed)
    if not remainder.is_number:
        return None
    return tuple(sorted(object_id for _, object_id in atoms))


def _mean_trace(data, expression: str) -> np.ndarray | None:
    item = data.fetch_data_by_str(expression)
    if item is None or not item.is_normal:
        return None
    values = np.asarray(item.normal_data.data, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return values.mean(axis=0)
    return None


def _same_trial_exists(theorist, spec, fit) -> bool:
    rounded = {key: round(value, 8) for key, value in fit.weights.items()}
    for record in theorist.operational_registry.records.values():
        if (
            record.protocol.experiment_family == spec.experiment_name
            and record.fit.observable == spec.observable
            and {key: round(value, 8) for key, value in record.fit.weights.items()}
            == rounded
        ):
            return True
    return False


def route_operational_relations(
    theorist,
    experiment_name: str,
    propositions: Iterable,
) -> list[str]:
    """Route eligible conserved relations into the trial registry.

    The function is deliberately a no-op unless the experiment declared an
    apparatus spec.  It examines a relation only as a trigger; numerical
    weights are re-estimated from measured traces, not copied from the equation.
    """

    spec = additive_operational_spec(experiment_name)
    if spec is None:
        return []
    specific = theorist.specific[experiment_name]
    measured = specific.experiment.data_info
    created = []
    for proposition in propositions:
        if not proposition.is_conserved:
            continue
        expanded = specific.expand_exp(proposition.unwrap_exp)
        raw = specific._sympy_of_raw_defi(expanded).doit()
        object_ids = _linear_entities(raw, spec.observable)
        if object_ids is None or spec.reference_object_id not in object_ids:
            continue
        traces = []
        missing = False
        for object_id in object_ids:
            trace = _mean_trace(measured, f"{spec.observable}[{object_id}]")
            if trace is None:
                missing = True
                break
            traces.append(trace)
        if missing or len({len(trace) for trace in traces}) != 1:
            continue

        epoch = getattr(theorist.memory, "epoch", 0)
        observation = InvariantObservation.from_state_series(
            observation_id=f"{experiment_name}:epoch:{epoch}:additive",
            experiment_id=f"{experiment_name}:epoch:{epoch}",
            experiment_family=experiment_name,
            observable=spec.observable,
            entity_ids=tuple(f"{spec.entity_type}:{object_id}" for object_id in object_ids),
            states=np.column_stack(traces),
            channels=tuple(
                ChannelRef(
                    f"{spec.observable}[{object_id}]", ChannelOrigin.RAW_SENSOR
                )
                for object_id in object_ids
            ),
            interventions={"source": "main_symbolic_regression"},
        )
        reference_entity = f"{spec.entity_type}:{spec.reference_object_id}"
        try:
            fit = AdditiveInvariantMiner().fit(
                [observation], reference_entity=reference_entity
            )
        except ValueError:
            continue
        if _same_trial_exists(theorist, spec, fit):
            continue
        protocol = MeasurementProtocol(
            experiment_family=experiment_name,
            observable=spec.observable,
            estimator=(
                "fit anonymous object weights whose measured-state changes "
                "have a one-dimensional additive null space"
            ),
            required_channels=tuple(
                [f"{spec.observable}[{object_id}]" for object_id in object_ids]
                + ["t[0]"]
            ),
            reference_entity=reference_entity,
            reference_value=1.0,
            nuisance_controls=spec.nuisance_controls,
            alternate_experiment_families=spec.alternate_experiment_families,
            notes=(
                "Proposed from a conserved relation, but coefficients were "
                "re-estimated from raw measured traces."
            ),
        )
        record = theorist.operational_registry.propose_additive_weight(
            fit,
            protocol,
            entity_type=spec.entity_type,
            created_epoch=epoch,
            applicable_experiment_families=spec.applicable_experiment_families,
        )
        created.append(record.candidate_id)
    return created
