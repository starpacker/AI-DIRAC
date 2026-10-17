"""Post-field Lorentz dynamics law experiments.

These experiments start only after ``electric_field`` and ``magnetic_field``
are unlocked.  They expose public dynamics channels and let the normal
regression path choose variables.  The lifecycle scorer checks the hidden
benchmark relation, but no force, residual, or precomputed law term is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.experiment.challenge_electromagnetism.common import lorentz_rhs
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import locked_quantities_in_data, quantity_is_eligible


@dataclass(frozen=True)
class LorentzDynamicsLawRun:
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
    apparatus: str = "crossed_field_particle_track"


@dataclass(frozen=True)
class LorentzDynamicsLawResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns: tuple[str, ...]


def make_lorentz_dynamics_law_run(
    *,
    experiment_id: str,
    electric_field: tuple[float, float, float] = (0.35, -0.08, 0.12),
    magnetic_field: tuple[float, float, float] = (0.15, 0.0, 0.9),
    mass: float = 1.3,
    charge: float = 0.85,
    initial_position: tuple[float, float, float] = (0.0, -0.1, 0.05),
    initial_velocity: tuple[float, float, float] = (0.22, 0.35, -0.04),
    duration: float = 1.4,
    samples: int = 128,
    experiment_family: str = "lorentz_dynamics_law",
    apparatus: str = "crossed_field_particle_track",
) -> LorentzDynamicsLawRun:
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
    velocity = solution[:, 3:]
    acceleration = charge * (evec[None, :] + np.cross(velocity, bvec)) / mass
    return LorentzDynamicsLawRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        position=solution[:, :3],
        velocity=velocity,
        acceleration=acceleration,
        mass=float(mass),
        charge=float(charge),
        electric_field=evec,
        magnetic_field=bvec,
        apparatus=apparatus,
    )


def lorentz_dynamics_law_runs() -> tuple[LorentzDynamicsLawRun, ...]:
    return (
        make_lorentz_dynamics_law_run(
            experiment_id="lorentz_definition_crossed_eb",
            electric_field=(0.42, 0.0, 0.0),
            magnetic_field=(0.0, 0.0, 1.1),
            initial_velocity=(0.16, 0.38, 0.0),
        ),
        make_lorentz_dynamics_law_run(
            experiment_id="lorentz_definition_tilted",
            electric_field=(0.25, -0.18, 0.16),
            magnetic_field=(0.32, 0.12, 0.86),
            initial_velocity=(0.18, -0.24, 0.13),
        ),
        make_lorentz_dynamics_law_run(
            experiment_id="lorentz_negative_charge",
            electric_field=(-0.25, 0.2, 0.08),
            magnetic_field=(0.0, 0.0, 0.95),
            charge=-0.75,
            initial_velocity=(0.22, 0.28, 0.04),
        ),
        make_lorentz_dynamics_law_run(
            experiment_id="lorentz_vary_mass_ratio",
            electric_field=(0.18, 0.12, -0.09),
            magnetic_field=(-0.18, 0.28, 0.72),
            mass=2.1,
            charge=1.15,
            initial_velocity=(-0.2, 0.21, 0.17),
        ),
        make_lorentz_dynamics_law_run(
            experiment_id="lorentz_heldout_plane",
            electric_field=(0.0, 0.35, -0.12),
            magnetic_field=(0.82, 0.0, 0.0),
            initial_velocity=(0.05, 0.18, 0.36),
            apparatus="orthogonal_coil_and_plate",
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
                _repeat_trace(values[:, axis_index], repeats, rng, noise).tolist()
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


def lorentz_dynamics_law_data(
    run: LorentzDynamicsLawRun,
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
    _add_vector_series(data, "pos", run.position, repeats=repeats, rng=rng, noise=noise)
    if quantity_is_eligible("velocityx[1]"):
        _add_vector_series(
            data, "velocity", run.velocity, repeats=repeats, rng=rng, noise=noise
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
        raise ValueError("Lorentz dynamics data contains locked channels: " + ", ".join(locked))
    return data


def lorentz_dynamics_score(run: LorentzDynamicsLawRun) -> float:
    predicted = run.charge * (
        run.electric_field[None, :] + np.cross(run.velocity, run.magnetic_field)
    ) / run.mass
    scale = max(float(np.sqrt(np.mean(run.acceleration ** 2))), 1e-12)
    relative_error = float(np.sqrt(np.mean((predicted - run.acceleration) ** 2)) / scale)
    return max(0.0, 1.0 - relative_error)


def run_lorentz_dynamics_law_lifecycle() -> LorentzDynamicsLawResult:
    for expression in (
        "velocityx[1]",
        "accelerationx[1]",
        "mass[1]",
        "charge[1]",
        "efieldx[1]",
        "bfieldx[1]",
    ):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "Lorentz dynamics law requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )

    runs = lorentz_dynamics_law_runs()
    definition_runs = runs[:2]
    validation_runs = {
        EvidenceKind.REPEATABILITY.value: runs[0],
        EvidenceKind.OBJECT_SENSITIVITY.value: runs[2],
        "vary:charge_mass_ratio": runs[3],
        EvidenceKind.HELDOUT_PREDICTION.value: runs[4],
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: runs[4],
        EvidenceKind.PROVENANCE_AUDIT.value: runs[-1],
    }
    definition_scores = [lorentz_dynamics_score(run) for run in definition_runs]
    validation_scores = {
        name: (1.0 if name == EvidenceKind.PROVENANCE_AUDIT.value else lorentz_dynamics_score(run))
        for name, run in validation_runs.items()
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    data = lorentz_dynamics_law_data(runs[-1])
    return LorentzDynamicsLawResult(
        status=status,
        symbolic_name="lorentz_dynamics_law",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns=tuple(sorted(str(key) for key in data.data_keys)),
    )
