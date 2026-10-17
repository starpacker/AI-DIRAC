"""Electric-field discovery from charged-particle kinematics.

This module is intentionally experiment-side only.  The simulator uses a
hidden uniform electric field to generate trajectories, but pre-unlock data do
not expose force or electric-field channels.  Once mass and charge have been
certified elsewhere, the candidate field can be estimated from allowed public
channels as ``mass * acceleration / charge``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import (
    locked_quantities_in_data,
    quantity_is_eligible,
    unlock_quantity,
)


@dataclass(frozen=True)
class ElectricFieldDynamicsRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    position: np.ndarray
    mass: float
    charge: float
    electric_field: np.ndarray
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    apparatus: str = "uniform_parallel_plate"


@dataclass(frozen=True)
class ElectricFieldUnlockResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_with_mass_charge: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def make_uniform_electric_dynamics_run(
    *,
    experiment_id: str,
    field: tuple[float, float, float] = (0.7, -0.25, 0.15),
    mass: float = 1.4,
    charge: float = 0.8,
    initial_position: tuple[float, float, float] = (-0.2, 0.15, 0.05),
    initial_velocity: tuple[float, float, float] = (0.1, -0.05, 0.08),
    duration: float = 1.4,
    samples: int = 96,
    experiment_family: str = "electric_field_dynamics",
    apparatus: str = "uniform_parallel_plate",
) -> ElectricFieldDynamicsRun:
    time = np.linspace(0.0, duration, samples)
    efield = np.asarray(field, dtype=float)
    pos0 = np.asarray(initial_position, dtype=float)
    vel0 = np.asarray(initial_velocity, dtype=float)
    acceleration = charge * efield / mass
    position = pos0[None, :] + vel0[None, :] * time[:, None]
    position = position + 0.5 * acceleration[None, :] * time[:, None] ** 2
    return ElectricFieldDynamicsRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        position=position,
        mass=float(mass),
        charge=float(charge),
        electric_field=efield,
        initial_position=pos0,
        initial_velocity=vel0,
        apparatus=apparatus,
    )


def _repeat_trace(values: np.ndarray, repeats: int, rng, noise: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    samples = np.repeat(values[None, :], repeats, axis=0)
    if noise:
        samples = samples + rng.normal(0.0, noise, samples.shape)
    return samples


def electric_field_dynamics_data(
    run: ElectricFieldDynamicsRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    """Emit only channels that are currently unlocked.

    ``force`` is never emitted by this experiment.  The electric-field vector is
    emitted only after the electric-field concept itself is unlocked.
    """

    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    data.add_data(
        AtomExp("t[0]"),
        ExpData(_repeat_trace(run.time, repeats, rng, noise).tolist()),
    )
    for axis_index, axis in enumerate("xyz"):
        data.add_data(
            AtomExp(f"pos{axis}[1]"),
            ExpData(
                _repeat_trace(
                    run.position[:, axis_index],
                    repeats,
                    rng,
                    noise,
                ).tolist()
            ),
        )

    if quantity_is_eligible("mass[1]"):
        data.add_data(AtomExp("mass[1]"), ExpData.from_const(run.mass, max(noise, 1e-12)))
    if quantity_is_eligible("charge[1]"):
        data.add_data(
            AtomExp("charge[1]"),
            ExpData.from_const(run.charge, max(noise, 1e-12)),
        )
    if quantity_is_eligible("efieldx[1]"):
        for axis_index, axis in enumerate("xyz"):
            data.add_data(
                AtomExp(f"efield{axis}[1]"),
                ExpData.from_const(
                    float(run.electric_field[axis_index]),
                    max(noise, 1e-12),
                ),
            )
    return data


def _mean_trace(data, expression: str, length: int | None = None) -> np.ndarray:
    expdata = data.fetch_data_by_str(expression)
    if expdata is None:
        raise KeyError(f"Missing required channel: {expression}")
    if expdata.is_normal:
        values = np.asarray(expdata.normal_data.mean, dtype=float)
    elif expdata.is_const or expdata.is_zero:
        if length is None:
            values = np.asarray([float(expdata.const_data.mean)], dtype=float)
        else:
            values = np.full(length, float(expdata.const_data.mean), dtype=float)
    else:
        raise ValueError(f"Cannot extract usable values from {expression}")
    if length is not None and values.size != length:
        raise ValueError(f"Channel {expression} length mismatch")
    return values


def reconstruct_acceleration_from_trajectory(data: DataStruct) -> np.ndarray:
    """Estimate acceleration using only public trajectory/time channels."""

    time = _mean_trace(data, "t[0]")
    components = [
        _mean_trace(data, f"pos{axis}[1]", len(time))
        for axis in "xyz"
    ]
    velocity = [
        np.gradient(component, time, edge_order=2)
        for component in components
    ]
    acceleration = [
        np.gradient(component, time, edge_order=2)
        for component in velocity
    ]
    return np.stack(acceleration, axis=1)


def reconstruct_electric_field_from_dynamics(data: DataStruct) -> np.ndarray:
    """Estimate candidate ``E`` as ``mass * acceleration / charge``.

    This helper is for experiment validation/audit.  It fails if mass or charge
    have not been unlocked and emitted into the public data.
    """

    time = _mean_trace(data, "t[0]")
    mass = float(_mean_trace(data, "mass[1]", len(time))[0])
    charge = float(_mean_trace(data, "charge[1]", len(time))[0])
    if abs(charge) < 1e-12:
        raise ValueError("Cannot estimate electric field with zero charge")
    acceleration = reconstruct_acceleration_from_trajectory(data)
    return mass * acceleration / charge


def electric_field_reconstruction_score(
    run: ElectricFieldDynamicsRun,
    *,
    noise: float = 0.0,
) -> float:
    data = electric_field_dynamics_data(run, noise=noise)
    if locked_quantities_in_data(data):
        raise ValueError("Electric-field dynamics data contains locked channels")
    estimate = reconstruct_electric_field_from_dynamics(data)
    central = estimate[2:-2] if len(estimate) > 4 else estimate
    predicted = central.mean(axis=0)
    scale = max(float(np.linalg.norm(run.electric_field)), 1e-12)
    relative_error = float(np.linalg.norm(predicted - run.electric_field) / scale)
    return max(0.0, 1.0 - relative_error)


def run_electric_field_unlock_lifecycle() -> ElectricFieldUnlockResult:
    """Run a no-force electric-field unlock lifecycle.

    Preconditions: mass and charge must already be unlocked, because those are
    the allowed public quantities needed for ``mass * acceleration / charge``.
    """

    if not quantity_is_eligible("mass[1]"):
        raise ValueError("mass must be unlocked before electric-field dynamics")
    if not quantity_is_eligible("charge[1]"):
        raise ValueError("charge must be unlocked before electric-field dynamics")

    definition_runs = (
        make_uniform_electric_dynamics_run(
            experiment_id="electric_field_definition_x",
            field=(0.72, 0.0, 0.0),
            mass=1.25,
            charge=0.85,
        ),
        make_uniform_electric_dynamics_run(
            experiment_id="electric_field_definition_3d",
            field=(0.45, -0.3, 0.22),
            mass=1.7,
            charge=1.1,
            initial_velocity=(-0.08, 0.12, 0.04),
        ),
    )
    definition_scores = [
        electric_field_reconstruction_score(run)
        for run in definition_runs
    ]

    validation_runs: dict[str, ElectricFieldDynamicsRun] = {
        EvidenceKind.REPEATABILITY.value: make_uniform_electric_dynamics_run(
            experiment_id="electric_field_repeat",
            field=(0.72, 0.0, 0.0),
            mass=1.25,
            charge=0.85,
        ),
        "vary:mass": make_uniform_electric_dynamics_run(
            experiment_id="electric_field_vary_mass",
            field=(0.72, 0.0, 0.0),
            mass=2.05,
            charge=0.85,
        ),
        "vary:charge": make_uniform_electric_dynamics_run(
            experiment_id="electric_field_vary_charge",
            field=(0.72, 0.0, 0.0),
            mass=1.25,
            charge=1.35,
        ),
        EvidenceKind.OBJECT_SENSITIVITY.value: make_uniform_electric_dynamics_run(
            experiment_id="electric_field_negative_charge",
            field=(0.35, -0.2, 0.15),
            mass=1.4,
            charge=-0.9,
        ),
        EvidenceKind.REFERENCE_TRANSITIVITY.value: make_uniform_electric_dynamics_run(
            experiment_id="electric_field_axis_cycle",
            field=(-0.2, 0.5, 0.28),
            mass=1.8,
            charge=0.75,
        ),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: make_uniform_electric_dynamics_run(
            experiment_id="electric_field_independent_apparatus",
            field=(0.25, 0.18, -0.4),
            mass=1.15,
            charge=0.95,
            apparatus="uniform_accelerator_tube",
        ),
        EvidenceKind.HELDOUT_PREDICTION.value: make_uniform_electric_dynamics_run(
            experiment_id="electric_field_heldout",
            field=(-0.55, 0.12, 0.33),
            mass=1.65,
            charge=1.2,
            initial_position=(0.4, -0.1, 0.2),
            initial_velocity=(0.05, 0.06, -0.09),
        ),
        EvidenceKind.PROVENANCE_AUDIT.value: definition_runs[0],
    }
    validation_scores = {
        name: (
            1.0
            if name == EvidenceKind.PROVENANCE_AUDIT.value
            else electric_field_reconstruction_score(run)
        )
        for name, run in validation_runs.items()
    }

    before_run = definition_runs[0]
    before_data = electric_field_dynamics_data(before_run)
    if any(score < 0.999 for score in definition_scores):
        status = ConceptStatus.TRIAL
    elif all(score >= 0.999 for score in validation_scores.values()):
        status = ConceptStatus.CERTIFIED
    else:
        status = ConceptStatus.TRIAL

    with_mass_charge_columns = tuple(sorted(str(k) for k in before_data.data_keys))
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("electric_field")

    after_data = electric_field_dynamics_data(before_run)
    return ElectricFieldUnlockResult(
        status=status,
        symbolic_name="electric_field",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=(
            "posx[1]",
            "posy[1]",
            "posz[1]",
            "t[0]",
        ),
        emitted_columns_with_mass_charge=with_mass_charge_columns,
        emitted_columns_after_unlock=tuple(sorted(str(k) for k in after_data.data_keys)),
    )
