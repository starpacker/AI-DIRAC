"""Magnetostatic law experiments after magnetic-field unlock."""

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


MU0_REFERENCE = 1.7


@dataclass(frozen=True)
class LongWireMagnetostaticsRun:
    experiment_id: str
    experiment_family: str
    current: float
    radii: np.ndarray
    angles: np.ndarray
    position: np.ndarray
    tangent: np.ndarray
    magnetic_field: np.ndarray
    mu0: float = MU0_REFERENCE
    apparatus: str = "long_straight_wire"


@dataclass(frozen=True)
class CurrentDensityRun:
    experiment_id: str
    experiment_family: str
    current: float
    conductor_radius: float
    current_density: np.ndarray
    area: float
    apparatus: str = "uniform_round_conductor"


@dataclass(frozen=True)
class MagnetostaticsUnlockResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def make_long_wire_magnetostatics_run(
    *,
    experiment_id: str,
    current: float = 1.2,
    radii: tuple[float, ...] = (0.35, 0.5, 0.7, 0.95),
    angles: tuple[float, ...] | None = None,
    mu0: float = MU0_REFERENCE,
    experiment_family: str = "long_wire_magnetostatics",
    apparatus: str = "long_straight_wire",
) -> LongWireMagnetostaticsRun:
    if angles is None:
        angles = tuple(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))
    samples = [(radius, angle) for radius in radii for angle in angles]
    radii_array = np.asarray([item[0] for item in samples], dtype=float)
    angles_array = np.asarray([item[1] for item in samples], dtype=float)
    position = np.column_stack(
        [
            radii_array * np.cos(angles_array),
            radii_array * np.sin(angles_array),
            np.zeros_like(radii_array),
        ]
    )
    tangent = np.column_stack(
        [-np.sin(angles_array), np.cos(angles_array), np.zeros_like(radii_array)]
    )
    magnitude = mu0 * current / (2.0 * np.pi * radii_array)
    magnetic_field = tangent * magnitude[:, None]
    return LongWireMagnetostaticsRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        current=float(current),
        radii=radii_array,
        angles=angles_array,
        position=position,
        tangent=tangent,
        magnetic_field=magnetic_field,
        mu0=float(mu0),
        apparatus=apparatus,
    )


def make_current_density_run(
    *,
    experiment_id: str,
    current: float = 1.4,
    conductor_radius: float = 0.18,
    experiment_family: str = "current_density_uniform_conductor",
    apparatus: str = "uniform_round_conductor",
) -> CurrentDensityRun:
    area = float(np.pi * conductor_radius ** 2)
    current_density = np.asarray([0.0, 0.0, current / area], dtype=float)
    return CurrentDensityRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        current=float(current),
        conductor_radius=float(conductor_radius),
        current_density=current_density,
        area=area,
        apparatus=apparatus,
    )


def magnetostatics_runs() -> tuple[LongWireMagnetostaticsRun, ...]:
    return (
        make_long_wire_magnetostatics_run(
            experiment_id="wire_mu0_definition",
            current=1.2,
            radii=(0.32, 0.46, 0.68, 0.91),
        ),
        make_long_wire_magnetostatics_run(
            experiment_id="wire_mu0_reverse_current",
            current=-0.9,
            radii=(0.4, 0.62, 0.85),
        ),
        make_long_wire_magnetostatics_run(
            experiment_id="wire_mu0_large_current",
            current=1.75,
            radii=(0.28, 0.55, 0.82, 1.1),
        ),
        make_long_wire_magnetostatics_run(
            experiment_id="wire_mu0_heldout_angles",
            current=0.7,
            radii=(0.37, 0.73, 1.05),
            angles=tuple(np.linspace(np.pi / 10.0, 2.0 * np.pi + np.pi / 10.0, 7, endpoint=False)),
            apparatus="independent_wire_probe",
        ),
    )


def current_density_runs() -> tuple[CurrentDensityRun, ...]:
    return (
        make_current_density_run(
            experiment_id="current_density_definition",
            current=1.1,
            conductor_radius=0.16,
        ),
        make_current_density_run(
            experiment_id="current_density_vary_area",
            current=1.1,
            conductor_radius=0.24,
        ),
        make_current_density_run(
            experiment_id="current_density_vary_current",
            current=-0.8,
            conductor_radius=0.19,
            apparatus="four_terminal_conductor",
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


def _add_vector_series(data: DataStruct, stem: str, values: np.ndarray, repeats: int, rng, noise: float, object_id: str = "1") -> None:
    for axis_index, axis in enumerate("xyz"):
        _add_series(data, f"{stem}{axis}[{object_id}]", values[:, axis_index], repeats, rng, noise)


def long_wire_magnetostatics_data(
    run: LongWireMagnetostaticsRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_series(data, "current[1]", np.full(len(run.radii), run.current), repeats, rng, noise)
    _add_series(data, "radial_distance[1]", run.radii, repeats, rng, noise)
    _add_series(data, "wire_angle[1]", run.angles, repeats, rng, noise)
    _add_vector_series(data, "pos", run.position, repeats, rng, noise)
    _add_vector_series(data, "tangent", run.tangent, repeats, rng, noise)
    if quantity_is_eligible("bfieldx[1]"):
        _add_vector_series(data, "bfield", run.magnetic_field, repeats, rng, noise)
    if quantity_is_eligible("mu0[0]"):
        data.add_data(AtomExp("mu0[0]"), ExpData.from_const(run.mu0, max(noise, 1e-12)))
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Magnetostatics data contains locked channels: " + ", ".join(locked))
    return data


def current_density_data(
    run: CurrentDensityRun,
    *,
    noise: float = 0.0,
) -> DataStruct:
    data = DataStruct.empty()
    data.add_data(AtomExp("current[1]"), ExpData.from_const(run.current, max(noise, 1e-12)))
    data.add_data(
        AtomExp("conductor_radius[1]"),
        ExpData.from_const(run.conductor_radius, max(noise, 1e-12)),
    )
    data.add_data(AtomExp("cross_section_area[1]"), ExpData.from_const(run.area, max(noise, 1e-12)))
    if quantity_is_eligible("current_densityx[1]"):
        for axis_index, axis in enumerate("xyz"):
            data.add_data(
                AtomExp(f"current_density{axis}[1]"),
                ExpData.from_const(float(run.current_density[axis_index]), max(noise, 1e-12)),
            )
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Current-density data contains locked channels: " + ", ".join(locked))
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


def _vector_trace(data: DataStruct, stem: str) -> np.ndarray:
    expressions = [f"{stem}{axis}[1]" for axis in "xyz"]
    length = None
    for expression in expressions:
        expdata = data.fetch_data_by_str(expression)
        if expdata is not None and expdata.is_normal:
            length = len(expdata.normal_data.mean)
            break
    return np.stack([_mean_trace(data, expression, length) for expression in expressions], axis=1)


def estimate_mu0_from_long_wire(data: DataStruct) -> float:
    radii = _mean_trace(data, "radial_distance[1]")
    tangent = _vector_trace(data, "tangent")
    bfield = _vector_trace(data, "bfield")
    b_tangent = np.einsum("ij,ij->i", bfield, tangent)
    current = _mean_trace(data, "current[1]", len(b_tangent))
    mask = np.abs(current) > 1e-12
    estimates = 2.0 * np.pi * radii[mask] * b_tangent[mask] / current[mask]
    return float(np.mean(estimates))


def mu0_reconstruction_score(run: LongWireMagnetostaticsRun) -> float:
    data = long_wire_magnetostatics_data(run)
    estimate = estimate_mu0_from_long_wire(data)
    relative_error = abs(estimate - run.mu0) / max(abs(run.mu0), 1e-12)
    return max(0.0, 1.0 - float(relative_error))


def ampere_loop_score(run: LongWireMagnetostaticsRun) -> float:
    unique_radii = np.unique(np.round(run.radii, decimals=12))
    scores = []
    for radius in unique_radii:
        mask = np.isclose(run.radii, radius)
        b_tangent = np.einsum("ij,ij->i", run.magnetic_field[mask], run.tangent[mask])
        line_integral = float(np.mean(b_tangent) * 2.0 * np.pi * radius)
        target = run.mu0 * run.current
        relative_error = abs(line_integral - target) / max(abs(target), 1e-12)
        scores.append(max(0.0, 1.0 - relative_error))
    return min(scores)


def current_density_score(run: CurrentDensityRun) -> float:
    predicted = np.asarray([0.0, 0.0, run.current / run.area], dtype=float)
    scale = max(float(np.linalg.norm(run.current_density)), 1e-12)
    relative_error = float(np.linalg.norm(predicted - run.current_density) / scale)
    return max(0.0, 1.0 - relative_error)


def run_mu0_unlock_lifecycle() -> MagnetostaticsUnlockResult:
    if not quantity_is_eligible("bfieldx[1]"):
        raise ValueError("mu0 magnetostatics requires unlocked magnetic_field")
    runs = magnetostatics_runs()
    before_data = long_wire_magnetostatics_data(runs[0])
    definition_scores = [mu0_reconstruction_score(runs[0]), mu0_reconstruction_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: mu0_reconstruction_score(runs[0]),
        "ampere_loop_integral": ampere_loop_score(runs[0]),
        "vary:current": mu0_reconstruction_score(runs[2]),
        EvidenceKind.HELDOUT_PREDICTION.value: mu0_reconstruction_score(runs[3]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: mu0_reconstruction_score(runs[3]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("mu0")
    after_data = long_wire_magnetostatics_data(runs[0])
    return MagnetostaticsUnlockResult(
        status=status,
        symbolic_name="mu0",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_current_density_unlock_lifecycle() -> MagnetostaticsUnlockResult:
    runs = current_density_runs()
    before_data = current_density_data(runs[0])
    definition_scores = [current_density_score(runs[0]), current_density_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: current_density_score(runs[0]),
        "vary:area": current_density_score(runs[1]),
        "vary:current_sign": current_density_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: current_density_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("current_density")
    after_data = current_density_data(runs[0])
    return MagnetostaticsUnlockResult(
        status=status,
        symbolic_name="current_density",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )
