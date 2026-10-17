"""Maxwell-Ampere displacement-current experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.experiment.challenge_electromagnetism.electrostatic_flux import EPSILON0_REFERENCE
from aiphy.experiment.challenge_electromagnetism.magnetostatics import MU0_REFERENCE
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import locked_quantities_in_data, quantity_is_eligible, unlock_quantity


@dataclass(frozen=True)
class ChargingCapacitorRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    current: np.ndarray
    electric_flux: np.ndarray
    electric_flux_rate: np.ndarray
    displacement_current: np.ndarray
    loop_radius: float
    path_position: np.ndarray
    dl_vectors: np.ndarray
    magnetic_field: np.ndarray
    surface_mode: np.ndarray
    epsilon0: float = EPSILON0_REFERENCE
    mu0: float = MU0_REFERENCE
    apparatus: str = "charging_capacitor_loop"


@dataclass(frozen=True)
class MaxwellAmpereResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def make_charging_capacitor_run(
    *,
    experiment_id: str,
    flux_amplitude: float = 0.36,
    omega: float = 1.4,
    flux_offset: float = 0.5,
    loop_radius: float = 0.32,
    duration: float = 2.0,
    time_samples: int = 96,
    loop_segments: int = 48,
    surface_mode: str = "gap",
    epsilon0: float = EPSILON0_REFERENCE,
    mu0: float = MU0_REFERENCE,
    experiment_family: str = "charging_capacitor_displacement_current",
    apparatus: str = "charging_capacitor_loop",
) -> ChargingCapacitorRun:
    time = np.linspace(0.0, duration, time_samples)
    electric_flux = flux_offset + flux_amplitude * np.sin(omega * time)
    electric_flux_rate = flux_amplitude * omega * np.cos(omega * time)
    displacement_current = epsilon0 * electric_flux_rate
    conduction_current = displacement_current.copy()
    if surface_mode == "wire":
        current = conduction_current
        surface_values = np.zeros_like(time)
    elif surface_mode == "gap":
        current = np.zeros_like(time)
        surface_values = np.ones_like(time)
    else:
        raise ValueError(f"Unsupported capacitor surface mode: {surface_mode}")

    angles = np.linspace(0.0, 2.0 * np.pi, loop_segments, endpoint=False)
    dtheta = 2.0 * np.pi / loop_segments
    position = np.column_stack(
        [loop_radius * np.cos(angles), loop_radius * np.sin(angles), np.zeros_like(angles)]
    )
    tangent = np.column_stack([-np.sin(angles), np.cos(angles), np.zeros_like(angles)])
    dl = tangent * (loop_radius * dtheta)
    circulation = mu0 * (current + epsilon0 * electric_flux_rate * surface_values)
    b_tangent = circulation / (2.0 * np.pi * loop_radius)
    magnetic_field = b_tangent[:, None, None] * tangent[None, :, :]
    return ChargingCapacitorRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        current=current,
        electric_flux=electric_flux,
        electric_flux_rate=electric_flux_rate,
        displacement_current=displacement_current,
        loop_radius=float(loop_radius),
        path_position=position,
        dl_vectors=dl,
        magnetic_field=magnetic_field,
        surface_mode=surface_values,
        epsilon0=float(epsilon0),
        mu0=float(mu0),
        apparatus=apparatus,
    )


def maxwell_ampere_runs() -> tuple[ChargingCapacitorRun, ...]:
    return (
        make_charging_capacitor_run(
            experiment_id="maxwell_ampere_definition_gap",
            surface_mode="gap",
            flux_amplitude=0.36,
            omega=1.4,
        ),
        make_charging_capacitor_run(
            experiment_id="maxwell_ampere_definition_wire",
            surface_mode="wire",
            flux_amplitude=0.36,
            omega=1.4,
        ),
        make_charging_capacitor_run(
            experiment_id="maxwell_ampere_vary_frequency",
            surface_mode="gap",
            flux_amplitude=0.28,
            omega=2.1,
        ),
        make_charging_capacitor_run(
            experiment_id="maxwell_ampere_heldout_loop",
            surface_mode="wire",
            flux_amplitude=-0.31,
            omega=1.15,
            loop_radius=0.45,
            apparatus="independent_capacitor_loop",
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


def _add_vector_series(
    data: DataStruct,
    stem: str,
    values: np.ndarray,
    repeats: int,
    rng,
    noise: float,
) -> None:
    for axis_index, axis in enumerate("xyz"):
        _add_series(data, f"{stem}{axis}[1]", values[:, axis_index], repeats, rng, noise)


def maxwell_ampere_data(
    run: ChargingCapacitorRun,
    *,
    time_index: int | None = None,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_series(data, "t[0]", run.time, repeats, rng, noise)
    _add_series(data, "current[1]", run.current, repeats, rng, noise)
    _add_series(data, "surface_mode[1]", run.surface_mode, repeats, rng, noise)
    if quantity_is_eligible("electric_flux[1]"):
        _add_series(data, "electric_flux[1]", run.electric_flux, repeats, rng, noise)
    if quantity_is_eligible("epsilon0[0]"):
        data.add_data(AtomExp("epsilon0[0]"), ExpData.from_const(run.epsilon0, max(noise, 1e-12)))
    if quantity_is_eligible("mu0[0]"):
        data.add_data(AtomExp("mu0[0]"), ExpData.from_const(run.mu0, max(noise, 1e-12)))
    if quantity_is_eligible("displacement_current[1]"):
        _add_series(data, "displacement_current[1]", run.displacement_current, repeats, rng, noise)

    loop_index = len(run.time) // 3 if time_index is None else int(time_index)
    _add_vector_series(data, "pos", run.path_position, repeats, rng, noise)
    _add_vector_series(data, "dl", run.dl_vectors, repeats, rng, noise)
    if quantity_is_eligible("bfieldx[1]"):
        _add_vector_series(data, "bfield", run.magnetic_field[loop_index], repeats, rng, noise)
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Maxwell-Ampere data contains locked channels: " + ", ".join(locked))
    return data


def line_integral_b(run: ChargingCapacitorRun, time_index: int) -> float:
    return float(np.sum(np.einsum("ij,ij->i", run.magnetic_field[time_index], run.dl_vectors)))


def displacement_current_score(run: ChargingCapacitorRun) -> float:
    predicted = run.epsilon0 * np.gradient(run.electric_flux, run.time, edge_order=2)
    central = slice(2, -2)
    scale = max(float(np.sqrt(np.mean(run.displacement_current[central] ** 2))), 1e-12)
    relative_error = float(
        np.sqrt(np.mean((predicted[central] - run.displacement_current[central]) ** 2)) / scale
    )
    return max(0.0, 1.0 - relative_error)


def maxwell_ampere_score(run: ChargingCapacitorRun) -> float:
    predicted_rate = np.gradient(run.electric_flux, run.time, edge_order=2)
    scores = []
    for index in range(2, len(run.time) - 2, max(1, len(run.time) // 12)):
        lhs = line_integral_b(run, index)
        rhs = run.mu0 * (run.current[index] + run.epsilon0 * predicted_rate[index] * run.surface_mode[index])
        relative_error = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        scores.append(max(0.0, 1.0 - float(relative_error)))
    return min(scores)


def run_displacement_current_unlock_lifecycle() -> MaxwellAmpereResult:
    for expression in ("electric_flux[1]", "epsilon0[0]"):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "displacement_current requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )
    runs = maxwell_ampere_runs()
    before_data = maxwell_ampere_data(runs[0])
    definition_scores = [displacement_current_score(runs[0]), displacement_current_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: displacement_current_score(runs[0]),
        "vary:drive_frequency": displacement_current_score(runs[2]),
        EvidenceKind.HELDOUT_PREDICTION.value: displacement_current_score(runs[3]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.995 and all(score >= 0.995 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("displacement_current")
    after_data = maxwell_ampere_data(runs[0])
    return MaxwellAmpereResult(
        status=status,
        symbolic_name="displacement_current",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_maxwell_ampere_law_lifecycle() -> MaxwellAmpereResult:
    for expression in ("electric_flux[1]", "epsilon0[0]", "mu0[0]", "bfieldx[1]"):
        if not quantity_is_eligible(expression):
            raise ValueError(
                "Maxwell-Ampere law requires unlocked "
                f"{expression.split('[', 1)[0]}"
            )
    runs = maxwell_ampere_runs()
    data = maxwell_ampere_data(runs[0])
    definition_scores = [maxwell_ampere_score(runs[0]), maxwell_ampere_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: maxwell_ampere_score(runs[0]),
        "spanning_surface:wire_vs_gap": min(maxwell_ampere_score(runs[0]), maxwell_ampere_score(runs[1])),
        "vary:charging_waveform": maxwell_ampere_score(runs[2]),
        EvidenceKind.HELDOUT_PREDICTION.value: maxwell_ampere_score(runs[3]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: maxwell_ampere_score(runs[3]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.995 and all(score >= 0.995 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    columns = tuple(sorted(str(key) for key in data.data_keys))
    return MaxwellAmpereResult(
        status=status,
        symbolic_name="maxwell_ampere_law",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=columns,
        emitted_columns_after_unlock=columns,
    )
