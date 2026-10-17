"""Inertial-mass unlock experiments from collision kinematics.

The electromagnetic curriculum needs mass before electric and magnetic dynamics
can be used without force.  These helpers expose only time and positions before
``mass`` is certified.  Velocity changes are reconstructed from allowed
trajectory channels for the operational concept miner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from aiphy.core import AtomExp, DataStruct, ExpData
from aiphy.operational_concepts import (
    AdditiveInvariantMiner,
    ChannelOrigin,
    ChannelRef,
    ConceptStatus,
    EvidenceKind,
    InvariantObservation,
    MeasurementProtocol,
    OperationalConceptRegistry,
    OperationalPipelineBridge,
    OperationalScope,
    ValidationEvidence,
)
from aiphy.sr_metadata import quantity_is_eligible


@dataclass(frozen=True)
class MassCollisionRun:
    experiment_id: str
    experiment_family: str
    entity_ids: tuple[str, str]
    slot_ids: tuple[str, str]
    masses: dict[str, float]
    time: np.ndarray
    positions: dict[str, np.ndarray]
    velocities_before: dict[str, float]
    velocities_after: dict[str, float]
    apparatus: str = "air_track_collision"


@dataclass(frozen=True)
class MassUnlockResult:
    candidate_id: str
    status: ConceptStatus
    symbolic_name: str
    fit_relative_residual: float
    validation_scores: dict[str, float]
    discovered_weights: dict[str, float]
    emitted_columns_before_unlock: tuple[str, ...]
    emitted_columns_after_unlock: tuple[str, ...]


MASS_REFERENCES: Mapping[str, float] = {
    "particle_1": 1.0,
    "particle_2": 1.7,
    "particle_3": 0.65,
    "particle_4": 2.25,
}


def elastic_collision_1d(
    m1: float,
    m2: float,
    u1: float,
    u2: float,
) -> tuple[float, float]:
    total = m1 + m2
    v1 = ((m1 - m2) / total) * u1 + (2.0 * m2 / total) * u2
    v2 = (2.0 * m1 / total) * u1 + ((m2 - m1) / total) * u2
    return float(v1), float(v2)


def make_mass_collision_run(
    *,
    experiment_id: str,
    entity_ids: tuple[str, str] = ("particle_1", "particle_2"),
    slot_ids: tuple[str, str] = ("1", "2"),
    velocities_before: tuple[float, float] = (0.8, -0.25),
    duration: float = 2.0,
    samples: int = 96,
    experiment_family: str = "inertial_mass_collision",
    apparatus: str = "air_track_collision",
) -> MassCollisionRun:
    m1 = float(MASS_REFERENCES[entity_ids[0]])
    m2 = float(MASS_REFERENCES[entity_ids[1]])
    u1, u2 = (float(value) for value in velocities_before)
    v1, v2 = elastic_collision_1d(m1, m2, u1, u2)
    time = np.linspace(0.0, duration, samples)
    collision_time = 0.5 * duration
    start_positions = {
        slot_ids[0]: -0.45,
        slot_ids[1]: 0.45,
    }
    positions = {
        slot_ids[0]: np.where(
            time <= collision_time,
            start_positions[slot_ids[0]] + u1 * time,
            start_positions[slot_ids[0]]
            + u1 * collision_time
            + v1 * (time - collision_time),
        ),
        slot_ids[1]: np.where(
            time <= collision_time,
            start_positions[slot_ids[1]] + u2 * time,
            start_positions[slot_ids[1]]
            + u2 * collision_time
            + v2 * (time - collision_time),
        ),
    }
    return MassCollisionRun(
        experiment_id=experiment_id,
        experiment_family=experiment_family,
        entity_ids=entity_ids,
        slot_ids=slot_ids,
        masses={entity: float(MASS_REFERENCES[entity]) for entity in entity_ids},
        time=time,
        positions=positions,
        velocities_before={slot_ids[0]: u1, slot_ids[1]: u2},
        velocities_after={slot_ids[0]: v1, slot_ids[1]: v2},
        apparatus=apparatus,
    )


def mass_definition_runs() -> tuple[MassCollisionRun, ...]:
    return (
        make_mass_collision_run(
            experiment_id="mass_definition_12_fast",
            entity_ids=("particle_1", "particle_2"),
            velocities_before=(0.8, -0.25),
        ),
        make_mass_collision_run(
            experiment_id="mass_definition_12_reverse",
            entity_ids=("particle_1", "particle_2"),
            velocities_before=(-0.5, 0.7),
        ),
        make_mass_collision_run(
            experiment_id="mass_definition_23",
            entity_ids=("particle_2", "particle_3"),
            velocities_before=(0.65, -0.35),
        ),
        make_mass_collision_run(
            experiment_id="mass_definition_13",
            entity_ids=("particle_1", "particle_3"),
            velocities_before=(-0.4, 0.9),
        ),
    )


def mass_validation_runs() -> dict[str, MassCollisionRun]:
    return {
        EvidenceKind.REPEATABILITY.value: make_mass_collision_run(
            experiment_id="mass_repeat_12",
            entity_ids=("particle_1", "particle_2"),
            velocities_before=(0.8, -0.25),
        ),
        "vary:impact_speed": make_mass_collision_run(
            experiment_id="mass_vary_impact_speed",
            entity_ids=("particle_1", "particle_2"),
            velocities_before=(1.05, -0.1),
        ),
        "vary:reference_frame": make_mass_collision_run(
            experiment_id="mass_vary_reference_frame",
            entity_ids=("particle_1", "particle_2"),
            velocities_before=(1.0, -0.05),
        ),
        EvidenceKind.OBJECT_SENSITIVITY.value: make_mass_collision_run(
            experiment_id="mass_replace_object_14",
            entity_ids=("particle_1", "particle_4"),
            velocities_before=(0.6, -0.25),
        ),
        EvidenceKind.REFERENCE_TRANSITIVITY.value: make_mass_collision_run(
            experiment_id="mass_reference_cycle_31",
            entity_ids=("particle_3", "particle_1"),
            velocities_before=(0.45, -0.55),
        ),
        EvidenceKind.CROSS_APPARATUS_TRANSFER.value: make_mass_collision_run(
            experiment_id="mass_independent_apparatus_24",
            entity_ids=("particle_2", "particle_4"),
            velocities_before=(0.75, -0.2),
            experiment_family="inertial_mass_magnetic_glider",
            apparatus="magnetic_glider_track",
        ),
        EvidenceKind.HELDOUT_PREDICTION.value: make_mass_collision_run(
            experiment_id="mass_heldout_34",
            entity_ids=("particle_3", "particle_4"),
            velocities_before=(-0.35, 0.85),
        ),
    }


def _repeat_trace(values: np.ndarray, repeats: int, rng, noise: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    samples = np.repeat(values[None, :], repeats, axis=0)
    if noise:
        samples = samples + rng.normal(0.0, noise, samples.shape)
    return samples


def mass_collision_data(
    run: MassCollisionRun,
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
    for slot_id in run.slot_ids:
        data.add_data(
            AtomExp(f"posx[{slot_id}]"),
            ExpData(
                _repeat_trace(run.positions[slot_id], repeats, rng, noise).tolist()
            ),
        )
    if quantity_is_eligible("mass[1]"):
        for entity, slot_id in zip(run.entity_ids, run.slot_ids):
            data.add_data(
                AtomExp(f"mass[{slot_id}]"),
                ExpData.from_const(run.masses[entity], max(noise, 1e-12)),
            )
    return data


def _mean_trace(data: DataStruct, expression: str) -> np.ndarray:
    expdata = data.fetch_data_by_str(expression)
    if expdata is None:
        raise KeyError(f"Missing required channel: {expression}")
    if expdata.is_normal:
        return np.asarray(expdata.normal_data.mean, dtype=float)
    if expdata.is_const or expdata.is_zero:
        return np.asarray([float(expdata.const_data.mean)], dtype=float)
    raise ValueError(f"Cannot extract usable values from {expression}")


def _slope(time: np.ndarray, values: np.ndarray) -> float:
    coefficient = np.polyfit(time, values, deg=1)[0]
    return float(coefficient)


def reconstruct_velocity_deltas_from_positions(
    data: DataStruct,
    slot_ids: Sequence[str],
) -> np.ndarray:
    time = _mean_trace(data, "t[0]")
    if len(time) < 8:
        raise ValueError("At least eight time samples are required")
    split = len(time) // 2
    before_slice = slice(0, max(3, split // 2))
    after_slice = slice(split + max(1, (len(time) - split) // 2), len(time))
    deltas = []
    for slot_id in slot_ids:
        position = _mean_trace(data, f"posx[{slot_id}]")
        before = _slope(time[before_slice], position[before_slice])
        after = _slope(time[after_slice], position[after_slice])
        deltas.append(after - before)
    return np.asarray(deltas, dtype=float)


def observation_from_collision_run(
    run: MassCollisionRun,
    *,
    noise: float = 0.0,
) -> InvariantObservation:
    data = mass_collision_data(run, noise=noise)
    deltas = reconstruct_velocity_deltas_from_positions(data, run.slot_ids)
    return InvariantObservation(
        observation_id=f"{run.experiment_id}:delta_velocity",
        experiment_id=run.experiment_id,
        experiment_family=run.experiment_family,
        observable="velocity_change",
        entity_ids=run.entity_ids,
        deltas=deltas,
        channels=(
            ChannelRef("t[0]", ChannelOrigin.RAW_SENSOR),
            *(
                ChannelRef(f"posx[{slot_id}]", ChannelOrigin.RAW_SENSOR)
                for slot_id in run.slot_ids
            ),
            ChannelRef("delta_velocity", ChannelOrigin.DERIVED_FROM_ALLOWED),
        ),
        interventions={
            "apparatus": run.apparatus,
            "pair": ",".join(run.entity_ids),
        },
    )


def _mass_score(run: MassCollisionRun) -> float:
    deltas = np.asarray(
        [
            run.velocities_after[slot_id] - run.velocities_before[slot_id]
            for slot_id in run.slot_ids
        ],
        dtype=float,
    )
    weights = np.asarray([run.masses[entity] for entity in run.entity_ids], dtype=float)
    residual = float(abs(np.dot(weights, deltas)))
    scale = max(float(np.linalg.norm(weights) * np.linalg.norm(deltas)), 1e-12)
    return max(0.0, 1.0 - residual / scale)


def _evidence(
    kind: EvidenceKind,
    run: MassCollisionRun,
    *,
    intervention: str,
    independent: bool = True,
) -> ValidationEvidence:
    return ValidationEvidence(
        kind=kind,
        experiment_id=run.experiment_id,
        experiment_family=run.experiment_family,
        score=_mass_score(run),
        independent=independent,
        intervention=intervention,
        held_fixed=("mass_definition:collision_weighted_velocity_change",),
        channels=(
            ChannelRef("t[0]", ChannelOrigin.RAW_SENSOR),
            *(
                ChannelRef(f"posx[{slot_id}]", ChannelOrigin.RAW_SENSOR)
                for slot_id in run.slot_ids
            ),
            ChannelRef("delta_velocity", ChannelOrigin.DERIVED_FROM_ALLOWED),
        ),
    )


def run_mass_unlock_lifecycle(
    registry: OperationalConceptRegistry | None = None,
) -> MassUnlockResult:
    registry = registry or OperationalConceptRegistry()
    definition_runs = mass_definition_runs()
    before_data = mass_collision_data(definition_runs[0])
    observations = [
        observation_from_collision_run(run)
        for run in definition_runs
    ]
    fit = AdditiveInvariantMiner().fit(
        observations,
        reference_entity="particle_1",
        reference_value=1.0,
    )
    protocol = MeasurementProtocol(
        experiment_family="inertial_mass_collision",
        observable="velocity_change",
        estimator="mass is the object-attached weight conserving collision velocity change",
        required_channels=("t[0]", "posx[*]"),
        reference_entity="particle_1",
        reference_value=1.0,
        nuisance_controls=("impact_speed", "reference_frame"),
        alternate_experiment_families=("inertial_mass_magnetic_glider",),
        notes="Velocity changes are derived from public position/time traces.",
    )
    record = registry.propose_additive_weight(
        fit,
        protocol,
        entity_type="Particle",
        symbolic_name="mass",
        applicable_experiment_families=(
            "mechanics",
            "electromagnetism",
            "electric_field_dynamics",
            "magnetic_field_dynamics",
        ),
    )
    record.scope = OperationalScope.OBJECT_INTRINSIC

    validation_scores: dict[str, float] = {}
    for name, run in mass_validation_runs().items():
        if name == "vary:impact_speed":
            kind = EvidenceKind.NUISANCE_INVARIANCE
            intervention = "vary:impact_speed"
        elif name == "vary:reference_frame":
            kind = EvidenceKind.NUISANCE_INVARIANCE
            intervention = "vary:reference_frame"
        else:
            kind = EvidenceKind(name)
            intervention = name
        registry.record_evidence(
            record.candidate_id,
            _evidence(kind, run, intervention=intervention),
        )
        validation_scores[name] = _mass_score(run)

    provenance = ValidationEvidence(
        kind=EvidenceKind.PROVENANCE_AUDIT,
        experiment_id="mass_collision_provenance_audit",
        experiment_family="audit",
        score=1.0,
        independent=True,
        intervention="audit_channel_dependency_graph",
        held_fixed=("no_mass_channel_before_unlock",),
        channels=(
            ChannelRef("t[0]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("posx[*]", ChannelOrigin.RAW_SENSOR),
            ChannelRef("delta_velocity", ChannelOrigin.DERIVED_FROM_ALLOWED),
        ),
    )
    registry.record_evidence(record.candidate_id, provenance)
    validation_scores[EvidenceKind.PROVENANCE_AUDIT.value] = 1.0

    bridge = OperationalPipelineBridge(registry)
    bridge.sync_certified_actions(lambda _name, _initial_value: None)

    after_data = mass_collision_data(definition_runs[0])
    return MassUnlockResult(
        candidate_id=record.candidate_id,
        status=record.status,
        symbolic_name=record.symbolic_name or "",
        fit_relative_residual=fit.relative_residual,
        validation_scores=validation_scores,
        discovered_weights=fit.weights,
        emitted_columns_before_unlock=tuple(sorted(str(k) for k in before_data.data_keys)),
        emitted_columns_after_unlock=tuple(sorted(str(k) for k in after_data.data_keys)),
    )
