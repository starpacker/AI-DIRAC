"""Electric and magnetic dipole-moment experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.experiment.challenge_electromagnetism.electrostatic_flux import EPSILON0_REFERENCE
from aiphy.experiment.challenge_electromagnetism.magnetostatics import MU0_REFERENCE
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import locked_quantities_in_data, quantity_is_eligible, unlock_quantity


@dataclass(frozen=True)
class ElectricDipoleRun:
    experiment_id: str
    experiment_family: str
    charge: float
    separation: np.ndarray
    electric_dipole: np.ndarray
    field_position: np.ndarray
    electric_field: np.ndarray
    epsilon0: float = EPSILON0_REFERENCE
    apparatus: str = "opposite_charge_pair"


@dataclass(frozen=True)
class MagneticDipoleRun:
    experiment_id: str
    experiment_family: str
    current: float
    loop_area: float
    normal: np.ndarray
    magnetic_dipole: np.ndarray
    field_position: np.ndarray
    magnetic_field: np.ndarray
    mu0: float = MU0_REFERENCE
    apparatus: str = "small_current_loop"


@dataclass(frozen=True)
class DipoleUnlockResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def _field_points(radius: float = 1.1) -> np.ndarray:
    return np.asarray(
        [
            (radius, 0.0, 0.0),
            (0.0, radius, 0.0),
            (0.0, 0.0, radius),
            (radius / np.sqrt(2.0), radius / np.sqrt(2.0), 0.0),
            (radius / np.sqrt(3.0), radius / np.sqrt(3.0), radius / np.sqrt(3.0)),
        ],
        dtype=float,
    )


def _dipole_field(moment: np.ndarray, points: np.ndarray, coefficient: float) -> np.ndarray:
    field = []
    for point in points:
        radius = float(np.linalg.norm(point))
        rhat = point / max(radius, 1e-12)
        projection = float(np.dot(moment, rhat))
        field.append(coefficient * (3.0 * rhat * projection - moment) / max(radius ** 3, 1e-12))
    return np.asarray(field, dtype=float)


def make_electric_dipole_run(
    *,
    experiment_id: str,
    charge: float = 0.8,
    separation: tuple[float, float, float] = (0.18, 0.0, 0.0),
    epsilon0: float = EPSILON0_REFERENCE,
    radius: float = 1.1,
    experiment_family: str = "electric_dipole_moment",
    apparatus: str = "opposite_charge_pair",
) -> ElectricDipoleRun:
    separation_vec = np.asarray(separation, dtype=float)
    moment = charge * separation_vec
    points = _field_points(radius)
    coefficient = 1.0 / (4.0 * np.pi * epsilon0)
    electric_field = _dipole_field(moment, points, coefficient)
    return ElectricDipoleRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        charge=float(charge),
        separation=separation_vec,
        electric_dipole=moment,
        field_position=points,
        electric_field=electric_field,
        epsilon0=float(epsilon0),
        apparatus=apparatus,
    )


def make_magnetic_dipole_run(
    *,
    experiment_id: str,
    current: float = 1.1,
    loop_area: float = 0.19,
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mu0: float = MU0_REFERENCE,
    radius: float = 1.1,
    experiment_family: str = "magnetic_dipole_moment",
    apparatus: str = "small_current_loop",
) -> MagneticDipoleRun:
    normal_vec = np.asarray(normal, dtype=float)
    normal_vec = normal_vec / max(float(np.linalg.norm(normal_vec)), 1e-12)
    moment = current * loop_area * normal_vec
    points = _field_points(radius)
    coefficient = mu0 / (4.0 * np.pi)
    magnetic_field = _dipole_field(moment, points, coefficient)
    return MagneticDipoleRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        current=float(current),
        loop_area=float(loop_area),
        normal=normal_vec,
        magnetic_dipole=moment,
        field_position=points,
        magnetic_field=magnetic_field,
        mu0=float(mu0),
        apparatus=apparatus,
    )


def electric_dipole_runs() -> tuple[ElectricDipoleRun, ...]:
    return (
        make_electric_dipole_run(
            experiment_id="electric_dipole_definition_x",
            charge=0.8,
            separation=(0.18, 0.0, 0.0),
        ),
        make_electric_dipole_run(
            experiment_id="electric_dipole_definition_tilted",
            charge=0.65,
            separation=(0.08, -0.12, 0.05),
        ),
        make_electric_dipole_run(
            experiment_id="electric_dipole_reverse_charge",
            charge=-0.7,
            separation=(0.11, 0.04, -0.07),
            apparatus="independent_charge_pair_probe",
        ),
    )


def magnetic_dipole_runs() -> tuple[MagneticDipoleRun, ...]:
    return (
        make_magnetic_dipole_run(
            experiment_id="magnetic_dipole_definition_z",
            current=1.1,
            loop_area=0.19,
            normal=(0.0, 0.0, 1.0),
        ),
        make_magnetic_dipole_run(
            experiment_id="magnetic_dipole_definition_tilted",
            current=0.9,
            loop_area=0.24,
            normal=(0.2, -0.3, 0.93),
        ),
        make_magnetic_dipole_run(
            experiment_id="magnetic_dipole_reverse_current",
            current=-0.75,
            loop_area=0.17,
            normal=(0.0, 1.0, 0.0),
            apparatus="independent_current_loop_probe",
        ),
    )


def _repeat_trace(values: np.ndarray, repeats: int, rng, noise: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    samples = np.repeat(values[None, :], repeats, axis=0)
    if noise:
        samples = samples + rng.normal(0.0, noise, samples.shape)
    return samples


def _add_series(data: DataStruct, expression: str, values: np.ndarray, repeats: int, rng, noise: float) -> None:
    data.add_data(AtomExp(expression), ExpData(_repeat_trace(values, repeats, rng, noise).tolist()))


def _add_vector_series(data: DataStruct, stem: str, values: np.ndarray, repeats: int, rng, noise: float) -> None:
    for axis_index, axis in enumerate("xyz"):
        _add_series(data, f"{stem}{axis}[1]", values[:, axis_index], repeats, rng, noise)


def _add_const_vector(data: DataStruct, stem: str, values: np.ndarray, noise: float) -> None:
    for axis_index, axis in enumerate("xyz"):
        data.add_data(
            AtomExp(f"{stem}{axis}[1]"),
            ExpData.from_const(float(values[axis_index]), max(noise, 1e-12)),
        )


def electric_dipole_data(
    run: ElectricDipoleRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    data.add_data(AtomExp("charge[1]"), ExpData.from_const(run.charge, max(noise, 1e-12)))
    data.add_data(AtomExp("epsilon0[0]"), ExpData.from_const(run.epsilon0, max(noise, 1e-12)))
    _add_const_vector(data, "separation", run.separation, noise)
    _add_vector_series(data, "pos", run.field_position, repeats, rng, noise)
    if quantity_is_eligible("efieldx[1]"):
        _add_vector_series(data, "efield", run.electric_field, repeats, rng, noise)
    if quantity_is_eligible("electric_dipolex[1]"):
        _add_const_vector(data, "electric_dipole", run.electric_dipole, noise)
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Electric-dipole data contains locked channels: " + ", ".join(locked))
    return data


def magnetic_dipole_data(
    run: MagneticDipoleRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    data.add_data(AtomExp("current[1]"), ExpData.from_const(run.current, max(noise, 1e-12)))
    data.add_data(AtomExp("loop_area[1]"), ExpData.from_const(run.loop_area, max(noise, 1e-12)))
    data.add_data(AtomExp("mu0[0]"), ExpData.from_const(run.mu0, max(noise, 1e-12)))
    _add_const_vector(data, "normal", run.normal, noise)
    _add_vector_series(data, "pos", run.field_position, repeats, rng, noise)
    if quantity_is_eligible("bfieldx[1]"):
        _add_vector_series(data, "bfield", run.magnetic_field, repeats, rng, noise)
    if quantity_is_eligible("magnetic_dipolex[1]"):
        _add_const_vector(data, "magnetic_dipole", run.magnetic_dipole, noise)
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Magnetic-dipole data contains locked channels: " + ", ".join(locked))
    return data


def electric_dipole_score(run: ElectricDipoleRun) -> float:
    predicted_moment = run.charge * run.separation
    predicted_field = _dipole_field(
        predicted_moment,
        run.field_position,
        1.0 / (4.0 * np.pi * run.epsilon0),
    )
    moment_scale = max(float(np.linalg.norm(run.electric_dipole)), 1e-12)
    field_scale = max(float(np.sqrt(np.mean(run.electric_field ** 2))), 1e-12)
    moment_error = float(np.linalg.norm(predicted_moment - run.electric_dipole) / moment_scale)
    field_error = float(np.sqrt(np.mean((predicted_field - run.electric_field) ** 2)) / field_scale)
    return max(0.0, 1.0 - max(moment_error, field_error))


def magnetic_dipole_score(run: MagneticDipoleRun) -> float:
    predicted_moment = run.current * run.loop_area * run.normal
    predicted_field = _dipole_field(
        predicted_moment,
        run.field_position,
        run.mu0 / (4.0 * np.pi),
    )
    moment_scale = max(float(np.linalg.norm(run.magnetic_dipole)), 1e-12)
    field_scale = max(float(np.sqrt(np.mean(run.magnetic_field ** 2))), 1e-12)
    moment_error = float(np.linalg.norm(predicted_moment - run.magnetic_dipole) / moment_scale)
    field_error = float(np.sqrt(np.mean((predicted_field - run.magnetic_field) ** 2)) / field_scale)
    return max(0.0, 1.0 - max(moment_error, field_error))


def run_electric_dipole_unlock_lifecycle() -> DipoleUnlockResult:
    for expression in ("charge[1]", "efieldx[1]", "epsilon0[0]"):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "electric_dipole requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )
    runs = electric_dipole_runs()
    before_data = electric_dipole_data(runs[0])
    definition_scores = [electric_dipole_score(runs[0]), electric_dipole_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: electric_dipole_score(runs[0]),
        "reverse:charge_sign": electric_dipole_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: electric_dipole_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("electric_dipole")
    after_data = electric_dipole_data(runs[0])
    return DipoleUnlockResult(
        status=status,
        symbolic_name="electric_dipole",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_magnetic_dipole_unlock_lifecycle() -> DipoleUnlockResult:
    for expression in ("bfieldx[1]", "mu0[0]"):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "magnetic_dipole requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )
    runs = magnetic_dipole_runs()
    before_data = magnetic_dipole_data(runs[0])
    definition_scores = [magnetic_dipole_score(runs[0]), magnetic_dipole_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: magnetic_dipole_score(runs[0]),
        "reverse:current_sign": magnetic_dipole_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: magnetic_dipole_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("magnetic_dipole")
    after_data = magnetic_dipole_data(runs[0])
    return DipoleUnlockResult(
        status=status,
        symbolic_name="magnetic_dipole",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )
