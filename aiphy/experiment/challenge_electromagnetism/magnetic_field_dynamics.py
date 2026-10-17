"""Magnetic-field discovery experiments from charged-particle dynamics.

The experiments here are deliberately data-only.  They do not precompute a
Lorentz residual or a candidate magnetic field.  The simulator may use hidden
``B`` to generate trajectories, but public columns are gated by the concept
registry:

* trajectory/time are always emitted;
* velocity, acceleration, mass, charge, and electric field are emitted only if
  their concepts are already eligible;
* magnetic field is emitted only after ``magnetic_field`` itself is unlocked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.experiment.challenge_electromagnetism.common import lorentz_rhs
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import (
    locked_quantities_in_data,
    quantity_is_eligible,
    unlock_quantity,
)


@dataclass(frozen=True)
class MagneticFieldDynamicsRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    mass: float
    charge: float
    electric_field: np.ndarray
    magnetic_field: np.ndarray
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    apparatus: str = "helmholtz_like_uniform_field"


@dataclass(frozen=True)
class MagneticFieldUnlockResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]
    estimated_fields: dict[str, tuple[float, float, float]]


def make_uniform_magnetic_dynamics_run(
    *,
    experiment_id: str,
    magnetic_field: tuple[float, float, float] = (0.0, 0.0, 1.1),
    electric_field: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = 1.4,
    charge: float = 0.8,
    initial_position: tuple[float, float, float] = (0.25, 0.0, 0.0),
    initial_velocity: tuple[float, float, float] = (0.0, 0.35, 0.08),
    duration: float = 1.6,
    samples: int = 128,
    experiment_family: str = "magnetic_field_dynamics",
    apparatus: str = "helmholtz_like_uniform_field",
) -> MagneticFieldDynamicsRun:
    time = np.linspace(0.0, duration, samples)
    evec = np.asarray(electric_field, dtype=float)
    bvec = np.asarray(magnetic_field, dtype=float)
    state0 = np.concatenate(
        [
            np.asarray(initial_position, dtype=float),
            np.asarray(initial_velocity, dtype=float),
        ]
    )
    solution = odeint(
        lorentz_rhs(
            charge,
            mass,
            lambda _time, _pos, _vel: evec,
            lambda _time, _pos, _vel: bvec,
        ),
        state0,
        time,
    )
    position = solution[:, :3]
    velocity = solution[:, 3:]
    acceleration = np.asarray(
        [
            charge * (evec + np.cross(vel, bvec)) / mass
            for vel in velocity
        ],
        dtype=float,
    )
    return MagneticFieldDynamicsRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        mass=float(mass),
        charge=float(charge),
        electric_field=evec,
        magnetic_field=bvec,
        initial_position=state0[:3],
        initial_velocity=state0[3:],
        apparatus=apparatus,
    )


def magnetic_field_discovery_runs() -> tuple[MagneticFieldDynamicsRun, ...]:
    """Return a small held-out suite for discovering magnetic-field structure."""

    return (
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_definition_pure_bz",
            magnetic_field=(0.0, 0.0, 1.2),
            initial_velocity=(0.0, 0.42, 0.0),
        ),
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_velocity_direction_x",
            magnetic_field=(0.0, 0.0, 1.2),
            initial_velocity=(0.38, 0.0, 0.0),
        ),
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_negative_charge",
            magnetic_field=(0.0, 0.0, 1.2),
            charge=-0.7,
            initial_velocity=(0.0, 0.42, 0.0),
        ),
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_parallel_velocity_control",
            magnetic_field=(0.0, 0.0, 1.2),
            initial_velocity=(0.0, 0.0, 0.45),
        ),
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_tilted_field_heldout",
            magnetic_field=(0.35, -0.15, 0.9),
            initial_velocity=(0.18, 0.33, -0.05),
        ),
        make_uniform_magnetic_dynamics_run(
            experiment_id="magnetic_crossed_eb_transfer",
            magnetic_field=(0.0, 0.0, 1.05),
            electric_field=(0.22, -0.08, 0.0),
            initial_velocity=(0.12, 0.36, 0.0),
            apparatus="crossed_plate_and_coil",
        ),
    )


def _repeat_trace(values: np.ndarray, repeats: int, rng, noise: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    samples = np.repeat(values[None, :], repeats, axis=0)
    if noise:
        samples = samples + rng.normal(0.0, noise, samples.shape)
    return samples


def _add_vector_series(
    data: DataStruct,
    stem: str,
    values: np.ndarray,
    *,
    repeats: int,
    rng,
    noise: float,
) -> None:
    for axis_index, axis in enumerate("xyz"):
        data.add_data(
            AtomExp(f"{stem}{axis}[1]"),
            ExpData(
                _repeat_trace(
                    values[:, axis_index],
                    repeats,
                    rng,
                    noise,
                ).tolist()
            ),
        )


def _add_const_vector(
    data: DataStruct,
    stem: str,
    values: np.ndarray,
    *,
    noise: float,
) -> None:
    for axis_index, axis in enumerate("xyz"):
        data.add_data(
            AtomExp(f"{stem}{axis}[1]"),
            ExpData.from_const(float(values[axis_index]), max(noise, 1e-12)),
        )


def magnetic_field_dynamics_data(
    run: MagneticFieldDynamicsRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    """Emit a magnetic-field discovery trial with no hidden derived residuals."""

    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    data.add_data(
        AtomExp("t[0]"),
        ExpData(_repeat_trace(run.time, repeats, rng, noise).tolist()),
    )
    _add_vector_series(
        data,
        "pos",
        run.position,
        repeats=repeats,
        rng=rng,
        noise=noise,
    )
    if quantity_is_eligible("velocityx[1]"):
        _add_vector_series(
            data,
            "velocity",
            run.velocity,
            repeats=repeats,
            rng=rng,
            noise=noise,
        )
    if quantity_is_eligible("accelerationx[1]"):
        _add_vector_series(
            data,
            "acceleration",
            run.acceleration,
            repeats=repeats,
            rng=rng,
            noise=noise,
        )
    if quantity_is_eligible("mass[1]"):
        data.add_data(AtomExp("mass[1]"), ExpData.from_const(run.mass, max(noise, 1e-12)))
    if quantity_is_eligible("charge[1]"):
        data.add_data(
            AtomExp("charge[1]"),
            ExpData.from_const(run.charge, max(noise, 1e-12)),
        )
    if quantity_is_eligible("efieldx[1]"):
        _add_const_vector(data, "efield", run.electric_field, noise=noise)
    if quantity_is_eligible("bfieldx[1]"):
        _add_const_vector(data, "bfield", run.magnetic_field, noise=noise)

    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError(
            "Magnetic-field dynamics data contains locked channels: "
            + ", ".join(locked)
        )
    return data


def _mean_trace(
    data: DataStruct,
    expression: str,
    length: int | None = None,
) -> np.ndarray:
    expdata = data.fetch_data_by_str(expression)
    if expdata is None:
        raise KeyError(f"Missing required channel: {expression}")
    if expdata.is_normal:
        values = np.asarray(expdata.normal_data.mean, dtype=float)
    elif expdata.is_const:
        value = float(expdata.const_data.mean)
        values = np.asarray([value], dtype=float) if length is None else np.full(length, value)
    elif expdata.is_zero:
        values = np.asarray([0.0], dtype=float) if length is None else np.zeros(length)
    else:
        raise ValueError(f"Cannot extract usable values from {expression}")
    if length is not None and values.size != length:
        raise ValueError(f"Channel {expression} length mismatch")
    return values


def _vector_trace(data: DataStruct, stem: str, length: int) -> np.ndarray:
    return np.stack(
        [_mean_trace(data, f"{stem}{axis}[1]", length) for axis in "xyz"],
        axis=1,
    )


def estimate_magnetic_field_from_dynamics(data: DataStruct) -> np.ndarray:
    """Estimate candidate ``B`` from public dynamics channels.

    This is lifecycle audit code, not an emitted experiment channel.  It uses
    only already-unlocked quantities: velocity, acceleration, mass, charge, and
    electric field.
    """

    time = _mean_trace(data, "t[0]")
    length = len(time)
    velocity = _vector_trace(data, "velocity", length)
    acceleration = _vector_trace(data, "acceleration", length)
    electric_field = _vector_trace(data, "efield", length)
    mass = float(_mean_trace(data, "mass[1]", length)[0])
    charge = float(_mean_trace(data, "charge[1]", length)[0])
    if abs(charge) < 1e-12:
        raise ValueError("Cannot estimate magnetic field with zero charge")

    target = mass * acceleration / charge - electric_field
    rows = []
    values = []
    for vel, item in zip(velocity, target):
        vx, vy, vz = vel
        rows.extend(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ]
        )
        values.extend(item.tolist())
    matrix = np.asarray(rows, dtype=float)
    vector = np.asarray(values, dtype=float)
    estimate, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
    return estimate


def magnetic_field_reconstruction_score(run: MagneticFieldDynamicsRun) -> float:
    data = magnetic_field_dynamics_data(run)
    if locked_quantities_in_data(data):
        raise ValueError("Magnetic-field dynamics data contains locked channels")
    estimate = estimate_magnetic_field_from_dynamics(data)
    scale = max(float(np.linalg.norm(run.magnetic_field)), 1e-12)
    relative_error = float(np.linalg.norm(estimate - run.magnetic_field) / scale)
    return max(0.0, 1.0 - relative_error)


def magnetic_parallel_control_score(run: MagneticFieldDynamicsRun) -> float:
    expected = run.charge * run.electric_field / run.mass
    error = np.linalg.norm(run.acceleration - expected[None, :]) / max(
        np.sqrt(len(run.time)),
        1.0,
    )
    scale = max(float(np.linalg.norm(expected)), 1e-12)
    if scale <= 1e-12:
        scale = max(float(np.linalg.norm(run.magnetic_field)), 1.0)
    return max(0.0, 1.0 - float(error / scale))


def run_magnetic_field_unlock_lifecycle() -> MagneticFieldUnlockResult:
    """Run the no-force magnetic-field unlock lifecycle."""

    for expression in (
        "velocityx[1]",
        "accelerationx[1]",
        "mass[1]",
        "charge[1]",
        "efieldx[1]",
    ):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "magnetic-field dynamics requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )

    runs = magnetic_field_discovery_runs()
    before_data = magnetic_field_dynamics_data(runs[-1])
    definition_runs = (runs[0], runs[1], runs[4])
    validation_runs = {
        EvidenceKind.REPEATABILITY.value: runs[0],
        "vary:velocity_direction": runs[1],
        EvidenceKind.OBJECT_SENSITIVITY.value: runs[2],
        "parallel_velocity_control": runs[3],
        EvidenceKind.HELDOUT_PREDICTION.value: runs[4],
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: runs[5],
        EvidenceKind.PROVENANCE_AUDIT.value: runs[-1],
    }

    estimated_fields = {
        run.experiment_id: tuple(float(value) for value in estimate_magnetic_field_from_dynamics(
            magnetic_field_dynamics_data(run)
        ))
        for run in (*definition_runs, runs[2], runs[4], runs[5])
    }
    definition_scores = [
        magnetic_field_reconstruction_score(run)
        for run in definition_runs
    ]
    validation_scores: dict[str, float] = {}
    for name, run in validation_runs.items():
        if name == EvidenceKind.PROVENANCE_AUDIT.value:
            validation_scores[name] = 1.0
        elif name == "parallel_velocity_control":
            validation_scores[name] = magnetic_parallel_control_score(run)
        else:
            validation_scores[name] = magnetic_field_reconstruction_score(run)

    if min(definition_scores) >= 0.999 and all(
        score >= 0.999 for score in validation_scores.values()
    ):
        status = ConceptStatus.CERTIFIED
        unlock_quantity("magnetic_field")
    else:
        status = ConceptStatus.TRIAL

    after_data = magnetic_field_dynamics_data(runs[-1])
    return MagneticFieldUnlockResult(
        status=status,
        symbolic_name="magnetic_field",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(k) for k in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(k) for k in after_data.data_keys)),
        estimated_fields=estimated_fields,
    )
