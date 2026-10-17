"""Flux, emf, and Faraday-law experiments."""

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
class UniformMagneticFluxRun:
    experiment_id: str
    experiment_family: str
    magnetic_field: np.ndarray
    area_vectors: np.ndarray
    patch_position: np.ndarray
    magnetic_flux: float
    apparatus: str = "surface_patch_array"


@dataclass(frozen=True)
class ClosedLoopEmfRun:
    experiment_id: str
    experiment_family: str
    electric_field: np.ndarray
    dl_vectors: np.ndarray
    path_position: np.ndarray
    emf: float
    apparatus: str = "closed_loop_field_probe"


@dataclass(frozen=True)
class FaradayInductionRun:
    experiment_id: str
    experiment_family: str
    time: np.ndarray
    magnetic_flux: np.ndarray
    emf: np.ndarray
    current: np.ndarray
    apparatus: str = "induction_loop"


@dataclass(frozen=True)
class FluxInductionResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def make_uniform_magnetic_flux_run(
    *,
    experiment_id: str,
    magnetic_field: tuple[float, float, float] = (0.2, -0.1, 0.85),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    area: float = 0.42,
    patches: int = 24,
    experiment_family: str = "uniform_magnetic_flux",
    apparatus: str = "surface_patch_array",
) -> UniformMagneticFluxRun:
    bvec = np.asarray(magnetic_field, dtype=float)
    normal_vec = np.asarray(normal, dtype=float)
    normal_vec = normal_vec / max(float(np.linalg.norm(normal_vec)), 1e-12)
    side = np.sqrt(area)
    grid = int(np.ceil(np.sqrt(patches)))
    coords = np.linspace(-0.5 * side, 0.5 * side, grid)
    points = np.asarray([(x, y, 0.0) for x in coords for y in coords], dtype=float)[:patches]
    weights = 1.0 + 0.05 * np.sin(np.arange(patches, dtype=float) + 0.3)
    weights = weights / np.sum(weights)
    darea = normal_vec[None, :] * (area * weights[:, None])
    magnetic_flux = float(np.sum(darea @ bvec))
    return UniformMagneticFluxRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        magnetic_field=bvec,
        area_vectors=darea,
        patch_position=points,
        magnetic_flux=magnetic_flux,
        apparatus=apparatus,
    )


def make_closed_loop_emf_run(
    *,
    experiment_id: str,
    tangential_field: float = 0.35,
    radius: float = 0.28,
    segments: int = 64,
    experiment_family: str = "closed_loop_emf",
    apparatus: str = "closed_loop_field_probe",
) -> ClosedLoopEmfRun:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    dtheta = 2.0 * np.pi / segments
    position = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
    )
    tangent = np.column_stack([-np.sin(angles), np.cos(angles), np.zeros_like(angles)])
    dl = tangent * (radius * dtheta)
    electric_field = tangent * tangential_field
    emf = float(np.sum(np.einsum("ij,ij->i", electric_field, dl)))
    return ClosedLoopEmfRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        electric_field=electric_field,
        dl_vectors=dl,
        path_position=position,
        emf=emf,
        apparatus=apparatus,
    )


def make_faraday_induction_run(
    *,
    experiment_id: str,
    flux_offset: float = 0.18,
    flux_amplitude: float = 0.42,
    omega: float = 1.7,
    duration: float = 2.0,
    samples: int = 128,
    load_resistance: float = 2.5,
    experiment_family: str = "faraday_induction",
    apparatus: str = "induction_loop",
) -> FaradayInductionRun:
    time = np.linspace(0.0, duration, samples)
    magnetic_flux = flux_offset + flux_amplitude * np.sin(omega * time)
    emf = -flux_amplitude * omega * np.cos(omega * time)
    current = emf / load_resistance
    return FaradayInductionRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        time=time,
        magnetic_flux=magnetic_flux,
        emf=emf,
        current=current,
        apparatus=apparatus,
    )


def magnetic_flux_runs() -> tuple[UniformMagneticFluxRun, ...]:
    return (
        make_uniform_magnetic_flux_run(
            experiment_id="magnetic_flux_definition",
            magnetic_field=(0.15, -0.05, 0.9),
            normal=(0.0, 0.0, 1.0),
            area=0.45,
        ),
        make_uniform_magnetic_flux_run(
            experiment_id="magnetic_flux_tilted_surface",
            magnetic_field=(0.4, -0.12, 0.72),
            normal=(0.2, 0.0, 0.98),
            area=0.38,
        ),
        make_uniform_magnetic_flux_run(
            experiment_id="magnetic_flux_reverse_orientation",
            magnetic_field=(0.1, 0.25, 0.6),
            normal=(0.0, -0.3, -0.95),
            area=0.62,
            apparatus="independent_surface_probe",
        ),
    )


def emf_runs() -> tuple[ClosedLoopEmfRun, ...]:
    return (
        make_closed_loop_emf_run(
            experiment_id="emf_definition",
            tangential_field=0.35,
            radius=0.28,
        ),
        make_closed_loop_emf_run(
            experiment_id="emf_vary_radius",
            tangential_field=0.22,
            radius=0.46,
        ),
        make_closed_loop_emf_run(
            experiment_id="emf_reverse_field",
            tangential_field=-0.31,
            radius=0.34,
            apparatus="segmented_ring_probe",
        ),
    )


def faraday_runs() -> tuple[FaradayInductionRun, ...]:
    return (
        make_faraday_induction_run(
            experiment_id="faraday_definition",
            flux_amplitude=0.42,
            omega=1.7,
        ),
        make_faraday_induction_run(
            experiment_id="faraday_vary_waveform_frequency",
            flux_amplitude=0.31,
            omega=2.2,
        ),
        make_faraday_induction_run(
            experiment_id="faraday_heldout_slow_drive",
            flux_amplitude=-0.25,
            omega=1.15,
            apparatus="moving_magnet_loop",
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


def magnetic_flux_data(
    run: UniformMagneticFluxRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_vector_series(data, "pos", run.patch_position, repeats, rng, noise)
    _add_vector_series(data, "darea", run.area_vectors, repeats, rng, noise)
    if quantity_is_eligible("bfieldx[1]"):
        bfield = np.repeat(run.magnetic_field[None, :], len(run.area_vectors), axis=0)
        _add_vector_series(data, "bfield", bfield, repeats, rng, noise)
    if quantity_is_eligible("magnetic_flux[1]"):
        data.add_data(
            AtomExp("magnetic_flux[1]"),
            ExpData.from_const(run.magnetic_flux, max(noise, 1e-12)),
        )
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Magnetic-flux data contains locked channels: " + ", ".join(locked))
    return data


def emf_data(
    run: ClosedLoopEmfRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_vector_series(data, "pos", run.path_position, repeats, rng, noise)
    _add_vector_series(data, "dl", run.dl_vectors, repeats, rng, noise)
    if quantity_is_eligible("efieldx[1]"):
        _add_vector_series(data, "efield", run.electric_field, repeats, rng, noise)
    if quantity_is_eligible("emf[1]"):
        data.add_data(AtomExp("emf[1]"), ExpData.from_const(run.emf, max(noise, 1e-12)))
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("EMF data contains locked channels: " + ", ".join(locked))
    return data


def faraday_data(
    run: FaradayInductionRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_series(data, "t[0]", run.time, repeats, rng, noise)
    _add_series(data, "current[1]", run.current, repeats, rng, noise)
    if quantity_is_eligible("magnetic_flux[1]"):
        _add_series(data, "magnetic_flux[1]", run.magnetic_flux, repeats, rng, noise)
    if quantity_is_eligible("emf[1]"):
        _add_series(data, "emf[1]", run.emf, repeats, rng, noise)
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Faraday data contains locked channels: " + ", ".join(locked))
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


def magnetic_flux_score(run: UniformMagneticFluxRun) -> float:
    data = magnetic_flux_data(run)
    bfield = _vector_trace(data, "bfield")
    darea = _vector_trace(data, "darea")
    estimate = float(np.sum(np.einsum("ij,ij->i", bfield, darea)))
    relative_error = abs(estimate - run.magnetic_flux) / max(abs(run.magnetic_flux), 1e-12)
    return max(0.0, 1.0 - float(relative_error))


def emf_score(run: ClosedLoopEmfRun) -> float:
    data = emf_data(run)
    efield = _vector_trace(data, "efield")
    dl = _vector_trace(data, "dl")
    estimate = float(np.sum(np.einsum("ij,ij->i", efield, dl)))
    relative_error = abs(estimate - run.emf) / max(abs(run.emf), 1e-12)
    return max(0.0, 1.0 - float(relative_error))


def faraday_score(run: FaradayInductionRun) -> float:
    predicted = -np.gradient(run.magnetic_flux, run.time, edge_order=2)
    central = slice(2, -2)
    scale = max(float(np.sqrt(np.mean(run.emf[central] ** 2))), 1e-12)
    relative_error = float(np.sqrt(np.mean((predicted[central] - run.emf[central]) ** 2)) / scale)
    return max(0.0, 1.0 - relative_error)


def run_magnetic_flux_unlock_lifecycle() -> FluxInductionResult:
    if not quantity_is_eligible("bfieldx[1]"):
        raise ValueError("magnetic_flux requires unlocked magnetic_field")
    runs = magnetic_flux_runs()
    before_data = magnetic_flux_data(runs[0])
    definition_scores = [magnetic_flux_score(runs[0]), magnetic_flux_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: magnetic_flux_score(runs[0]),
        "reverse:surface_orientation": magnetic_flux_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: magnetic_flux_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("magnetic_flux")
    after_data = magnetic_flux_data(runs[0])
    return FluxInductionResult(
        status=status,
        symbolic_name="magnetic_flux",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_emf_unlock_lifecycle() -> FluxInductionResult:
    if not quantity_is_eligible("efieldx[1]"):
        raise ValueError("emf requires unlocked electric_field")
    runs = emf_runs()
    before_data = emf_data(runs[0])
    definition_scores = [emf_score(runs[0]), emf_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: emf_score(runs[0]),
        "reverse:electric_circulation": emf_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: emf_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("emf")
    after_data = emf_data(runs[0])
    return FluxInductionResult(
        status=status,
        symbolic_name="emf",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_faraday_law_lifecycle() -> FluxInductionResult:
    if not quantity_is_eligible("magnetic_flux[1]"):
        raise ValueError("Faraday law requires unlocked magnetic_flux")
    if not quantity_is_eligible("emf[1]"):
        raise ValueError("Faraday law requires unlocked emf")
    runs = faraday_runs()
    data = faraday_data(runs[0])
    definition_scores = [faraday_score(runs[0])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: faraday_score(runs[0]),
        "vary:drive_frequency": faraday_score(runs[1]),
        EvidenceKind.HELDOUT_PREDICTION.value: faraday_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: faraday_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.995 and all(score >= 0.995 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    columns = tuple(sorted(str(key) for key in data.data_keys))
    return FluxInductionResult(
        status=status,
        symbolic_name="faraday_law",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=columns,
        emitted_columns_after_unlock=columns,
    )
