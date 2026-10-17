"""Kinematics prerequisite experiments for the EM curriculum.

Named velocity and acceleration channels are locked until they are certified
from position/time trajectories.  The lifecycle here validates the derivative
construction without exposing those columns before unlock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import locked_quantities_in_data, quantity_is_eligible, unlock_quantity


@dataclass(frozen=True)
class KinematicsRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True)
class KinematicsUnlockResult:
    status: ConceptStatus
    unlocked_concepts: tuple[str, ...]
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def make_kinematics_run(
    *,
    experiment_id: str,
    initial_position: tuple[float, float, float] = (0.0, 0.1, -0.2),
    initial_velocity: tuple[float, float, float] = (0.35, -0.2, 0.12),
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
    duration: float = 1.6,
    samples: int = 96,
    experiment_family: str = "kinematics_trajectory",
) -> KinematicsRun:
    time = np.linspace(0.0, duration, samples)
    pos0 = np.asarray(initial_position, dtype=float)
    vel0 = np.asarray(initial_velocity, dtype=float)
    acc = np.asarray(acceleration, dtype=float)
    position = pos0[None, :] + vel0[None, :] * time[:, None]
    position = position + 0.5 * acc[None, :] * time[:, None] ** 2
    velocity = vel0[None, :] + acc[None, :] * time[:, None]
    acceleration_values = np.repeat(acc[None, :], samples, axis=0)
    return KinematicsRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration_values,
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


def kinematics_data(
    run: KinematicsRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
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
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError(
            "Kinematics data contains locked channels: " + ", ".join(locked)
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


def reconstruct_kinematics_from_positions(data: DataStruct) -> tuple[np.ndarray, np.ndarray]:
    time = _mean_trace(data, "t[0]")
    positions = np.stack(
        [_mean_trace(data, f"pos{axis}[1]", len(time)) for axis in "xyz"],
        axis=1,
    )
    velocity = np.stack(
        [
            np.gradient(positions[:, axis_index], time, edge_order=2)
            for axis_index in range(3)
        ],
        axis=1,
    )
    acceleration = np.stack(
        [
            np.gradient(velocity[:, axis_index], time, edge_order=2)
            for axis_index in range(3)
        ],
        axis=1,
    )
    return velocity, acceleration


def kinematics_reconstruction_score(run: KinematicsRun) -> float:
    data = kinematics_data(run)
    velocity, acceleration = reconstruct_kinematics_from_positions(data)
    central = slice(2, -2) if len(run.time) > 4 else slice(None)
    velocity_scale = max(float(np.linalg.norm(run.velocity[central])), 1e-9)
    acceleration_scale = max(float(np.linalg.norm(run.acceleration[central])), 1e-9)
    velocity_error = float(
        np.linalg.norm(velocity[central] - run.velocity[central]) / velocity_scale
    )
    acceleration_error = float(
        np.linalg.norm(acceleration[central] - run.acceleration[central])
        / acceleration_scale
    )
    return max(0.0, 1.0 - max(velocity_error, acceleration_error))


def run_kinematics_unlock_lifecycle() -> KinematicsUnlockResult:
    definition_runs = (
        make_kinematics_run(
            experiment_id="kinematics_definition_free_3d",
            acceleration=(0.0, 0.0, 0.0),
        ),
        make_kinematics_run(
            experiment_id="kinematics_definition_accelerated",
            initial_velocity=(0.1, 0.25, -0.15),
            acceleration=(0.35, -0.12, 0.08),
        ),
    )
    before_data = kinematics_data(definition_runs[0])
    definition_scores = [
        kinematics_reconstruction_score(run)
        for run in definition_runs
    ]
    validation_runs = {
        EvidenceKind.REPEATABILITY.value: make_kinematics_run(
            experiment_id="kinematics_repeat",
            acceleration=(0.35, -0.12, 0.08),
        ),
        "vary:initial_velocity": make_kinematics_run(
            experiment_id="kinematics_vary_initial_velocity",
            initial_velocity=(-0.22, 0.4, 0.16),
            acceleration=(0.35, -0.12, 0.08),
        ),
        "vary:axis": make_kinematics_run(
            experiment_id="kinematics_vary_axis",
            initial_velocity=(0.0, -0.2, 0.35),
            acceleration=(0.0, 0.24, -0.16),
        ),
        EvidenceKind.OBJECT_SENSITIVITY.value: make_kinematics_run(
            experiment_id="kinematics_replaced_particle",
            initial_position=(0.3, -0.2, 0.1),
            initial_velocity=(0.18, 0.22, -0.31),
            acceleration=(-0.12, 0.04, 0.22),
        ),
        EvidenceKind.REFERENCE_TRANSITIVITY.value: make_kinematics_run(
            experiment_id="kinematics_clock_transfer",
            initial_velocity=(0.25, -0.18, 0.05),
            acceleration=(0.08, 0.11, -0.04),
        ),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: make_kinematics_run(
            experiment_id="kinematics_independent_camera",
            experiment_family="kinematics_independent_camera",
            initial_velocity=(-0.1, 0.15, 0.2),
            acceleration=(0.18, 0.0, -0.09),
        ),
        EvidenceKind.HELDOUT_PREDICTION.value: make_kinematics_run(
            experiment_id="kinematics_heldout",
            initial_position=(-0.4, 0.25, 0.2),
            initial_velocity=(0.32, 0.08, -0.2),
            acceleration=(-0.07, 0.2, 0.13),
        ),
        EvidenceKind.PROVENANCE_AUDIT.value: definition_runs[0],
    }
    validation_scores = {
        name: (
            1.0
            if name == EvidenceKind.PROVENANCE_AUDIT.value
            else kinematics_reconstruction_score(run)
        )
        for name, run in validation_runs.items()
    }
    if min(definition_scores) >= 0.999 and all(
        score >= 0.999 for score in validation_scores.values()
    ):
        unlock_quantity("velocity")
        unlock_quantity("acceleration")
        status = ConceptStatus.CERTIFIED
    else:
        status = ConceptStatus.TRIAL
    after_data = kinematics_data(definition_runs[0])
    return KinematicsUnlockResult(
        status=status,
        unlocked_concepts=("velocity", "acceleration") if status is ConceptStatus.CERTIFIED else (),
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(k) for k in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(k) for k in after_data.data_keys)),
    )
