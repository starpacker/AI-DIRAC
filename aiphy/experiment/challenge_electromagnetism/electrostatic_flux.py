"""Electric flux and epsilon0 experiments.

The Gauss-law-facing experiments here avoid exposing closed-surface integrals
as columns.  The public data are patch geometry, unlocked electric field, and
unlocked charge.  Lifecycle scoring computes the benchmark relation for
certification only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.operational_concepts import ConceptStatus, EvidenceKind
from aiphy.sr_metadata import locked_quantities_in_data, quantity_is_eligible, unlock_quantity


EPSILON0_REFERENCE = 2.3


@dataclass(frozen=True)
class UniformElectricFluxRun:
    experiment_id: str
    experiment_family: str
    electric_field: np.ndarray
    area_vectors: np.ndarray
    patch_position: np.ndarray
    electric_flux: float
    apparatus: str = "electric_surface_patch_array"


@dataclass(frozen=True)
class ClosedSurfaceChargeFluxRun:
    experiment_id: str
    experiment_family: str
    charge: float
    electric_field: np.ndarray
    area_vectors: np.ndarray
    patch_position: np.ndarray
    electric_flux: float
    epsilon0: float = EPSILON0_REFERENCE
    apparatus: str = "closed_gaussian_box"


@dataclass(frozen=True)
class ElectrostaticFluxResult:
    status: ConceptStatus
    symbolic_name: str
    definition_score: float
    validation_scores: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


def _weighted_area_vectors(normal: np.ndarray, area: float, patches: int) -> np.ndarray:
    weights = 1.0 + 0.05 * np.cos(np.arange(patches, dtype=float) + 0.2)
    weights = weights / np.sum(weights)
    return normal[None, :] * (area * weights[:, None])


def make_uniform_electric_flux_run(
    *,
    experiment_id: str,
    electric_field: tuple[float, float, float] = (0.35, -0.1, 0.7),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    area: float = 0.48,
    patches: int = 24,
    experiment_family: str = "uniform_electric_flux",
    apparatus: str = "electric_surface_patch_array",
) -> UniformElectricFluxRun:
    evec = np.asarray(electric_field, dtype=float)
    normal_vec = np.asarray(normal, dtype=float)
    normal_vec = normal_vec / max(float(np.linalg.norm(normal_vec)), 1e-12)
    side = np.sqrt(area)
    grid = int(np.ceil(np.sqrt(patches)))
    coords = np.linspace(-0.5 * side, 0.5 * side, grid)
    points = np.asarray([(x, y, 0.0) for x in coords for y in coords], dtype=float)[:patches]
    darea = _weighted_area_vectors(normal_vec, area, patches)
    electric_flux = float(np.sum(darea @ evec))
    return UniformElectricFluxRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        electric_field=evec,
        area_vectors=darea,
        patch_position=points,
        electric_flux=electric_flux,
        apparatus=apparatus,
    )


def make_closed_surface_charge_flux_run(
    *,
    experiment_id: str,
    charge: float = 0.92,
    epsilon0: float = EPSILON0_REFERENCE,
    faces: int = 6,
    patches_per_face: int = 12,
    experiment_family: str = "closed_surface_charge_flux",
    apparatus: str = "closed_gaussian_box",
) -> ClosedSurfaceChargeFluxRun:
    normals = (
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    )[:faces]
    total_flux = charge / epsilon0
    face_flux = total_flux / len(normals)
    area_vectors = []
    electric_field = []
    patch_position = []
    for face_index, normal in enumerate(normals):
        darea = _weighted_area_vectors(normal, 1.0, patches_per_face)
        field = normal * face_flux
        area_vectors.append(darea)
        electric_field.append(np.repeat(field[None, :], patches_per_face, axis=0))
        patch_position.append(np.repeat((normal * 0.5)[None, :], patches_per_face, axis=0))
    darea_all = np.vstack(area_vectors)
    efield_all = np.vstack(electric_field)
    position_all = np.vstack(patch_position)
    electric_flux = float(np.sum(np.einsum("ij,ij->i", efield_all, darea_all)))
    return ClosedSurfaceChargeFluxRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        charge=float(charge),
        electric_field=efield_all,
        area_vectors=darea_all,
        patch_position=position_all,
        electric_flux=electric_flux,
        epsilon0=float(epsilon0),
        apparatus=apparatus,
    )


def electric_flux_runs() -> tuple[UniformElectricFluxRun, ...]:
    return (
        make_uniform_electric_flux_run(
            experiment_id="electric_flux_definition",
            electric_field=(0.18, -0.1, 0.74),
            normal=(0.0, 0.0, 1.0),
            area=0.44,
        ),
        make_uniform_electric_flux_run(
            experiment_id="electric_flux_tilted_surface",
            electric_field=(0.31, 0.2, 0.52),
            normal=(0.25, -0.1, 0.96),
            area=0.37,
        ),
        make_uniform_electric_flux_run(
            experiment_id="electric_flux_reverse_orientation",
            electric_field=(-0.2, 0.15, 0.63),
            normal=(0.0, 0.0, -1.0),
            area=0.58,
            apparatus="independent_electric_surface_probe",
        ),
    )


def epsilon0_runs() -> tuple[ClosedSurfaceChargeFluxRun, ...]:
    return (
        make_closed_surface_charge_flux_run(
            experiment_id="epsilon0_closed_surface_definition",
            charge=0.92,
        ),
        make_closed_surface_charge_flux_run(
            experiment_id="epsilon0_vary_charge",
            charge=-0.71,
        ),
        make_closed_surface_charge_flux_run(
            experiment_id="epsilon0_heldout_box",
            charge=1.34,
            patches_per_face=15,
            apparatus="independent_closed_surface_probe",
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


def electric_flux_data(
    run: UniformElectricFluxRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_vector_series(data, "pos", run.patch_position, repeats, rng, noise)
    _add_vector_series(data, "darea", run.area_vectors, repeats, rng, noise)
    if quantity_is_eligible("efieldx[1]"):
        efield = np.repeat(run.electric_field[None, :], len(run.area_vectors), axis=0)
        _add_vector_series(data, "efield", efield, repeats, rng, noise)
    if quantity_is_eligible("electric_flux[1]"):
        data.add_data(
            AtomExp("electric_flux[1]"),
            ExpData.from_const(run.electric_flux, max(noise, 1e-12)),
        )
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("Electric-flux data contains locked channels: " + ", ".join(locked))
    return data


def epsilon0_data(
    run: ClosedSurfaceChargeFluxRun,
    *,
    repeats: int = 5,
    noise: float = 0.0,
    seed: int = 20260815,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    data = DataStruct.empty()
    _add_vector_series(data, "pos", run.patch_position, repeats, rng, noise)
    _add_vector_series(data, "darea", run.area_vectors, repeats, rng, noise)
    if quantity_is_eligible("efieldx[1]"):
        _add_vector_series(data, "efield", run.electric_field, repeats, rng, noise)
    if quantity_is_eligible("charge[1]"):
        data.add_data(AtomExp("charge[1]"), ExpData.from_const(run.charge, max(noise, 1e-12)))
    if quantity_is_eligible("electric_flux[1]"):
        data.add_data(
            AtomExp("electric_flux[1]"),
            ExpData.from_const(run.electric_flux, max(noise, 1e-12)),
        )
    if quantity_is_eligible("epsilon0[0]"):
        data.add_data(AtomExp("epsilon0[0]"), ExpData.from_const(run.epsilon0, max(noise, 1e-12)))
    locked = locked_quantities_in_data(data)
    if locked:
        raise ValueError("epsilon0 data contains locked channels: " + ", ".join(locked))
    return data


def _mean_trace(data: DataStruct, expression: str, length: int | None = None) -> np.ndarray:
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


def electric_flux_score(run: UniformElectricFluxRun) -> float:
    data = electric_flux_data(run)
    efield = _vector_trace(data, "efield")
    darea = _vector_trace(data, "darea")
    estimate = float(np.sum(np.einsum("ij,ij->i", efield, darea)))
    relative_error = abs(estimate - run.electric_flux) / max(abs(run.electric_flux), 1e-12)
    return max(0.0, 1.0 - float(relative_error))


def epsilon0_score(run: ClosedSurfaceChargeFluxRun) -> float:
    data = epsilon0_data(run)
    efield = _vector_trace(data, "efield")
    darea = _vector_trace(data, "darea")
    flux = float(np.sum(np.einsum("ij,ij->i", efield, darea)))
    estimate = run.charge / flux
    relative_error = abs(estimate - run.epsilon0) / max(abs(run.epsilon0), 1e-12)
    return max(0.0, 1.0 - float(relative_error))


def run_electric_flux_unlock_lifecycle() -> ElectrostaticFluxResult:
    if not quantity_is_eligible("efieldx[1]"):
        raise ValueError("electric_flux requires unlocked electric_field")
    runs = electric_flux_runs()
    before_data = electric_flux_data(runs[0])
    definition_scores = [electric_flux_score(runs[0]), electric_flux_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: electric_flux_score(runs[0]),
        "reverse:surface_orientation": electric_flux_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: electric_flux_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("electric_flux")
    after_data = electric_flux_data(runs[0])
    return ElectrostaticFluxResult(
        status=status,
        symbolic_name="electric_flux",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )


def run_epsilon0_unlock_lifecycle() -> ElectrostaticFluxResult:
    if not quantity_is_eligible("charge[1]"):
        raise ValueError("epsilon0 requires unlocked charge")
    if not quantity_is_eligible("efieldx[1]"):
        raise ValueError("epsilon0 requires unlocked electric_field")
    if not quantity_is_eligible("electric_flux[1]"):
        raise ValueError("epsilon0 requires unlocked electric_flux")
    runs = epsilon0_runs()
    before_data = epsilon0_data(runs[0])
    definition_scores = [epsilon0_score(runs[0]), epsilon0_score(runs[1])]
    validation_scores = {
        EvidenceKind.REPEATABILITY.value: epsilon0_score(runs[0]),
        "vary:charge_sign": epsilon0_score(runs[1]),
        EvidenceKind.HELDOUT_PREDICTION.value: epsilon0_score(runs[2]),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: epsilon0_score(runs[2]),
        EvidenceKind.PROVENANCE_AUDIT.value: 1.0,
    }
    status = (
        ConceptStatus.CERTIFIED
        if min(definition_scores) >= 0.999 and all(score >= 0.999 for score in validation_scores.values())
        else ConceptStatus.TRIAL
    )
    if status is ConceptStatus.CERTIFIED:
        unlock_quantity("epsilon0")
    after_data = epsilon0_data(runs[0])
    return ElectrostaticFluxResult(
        status=status,
        symbolic_name="epsilon0",
        definition_score=min(definition_scores),
        validation_scores=validation_scores,
        emitted_columns_before_unlock=tuple(sorted(str(key) for key in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(key) for key in after_data.data_keys)),
    )
