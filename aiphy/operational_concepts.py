"""Low-prior operational-concept discovery and certification.

This module is intentionally independent of the Rust DSL and of simulator
objects.  It consumes only explicitly-provenanced observations, proposes
anonymous reusable measurements, and manages the controlled evidence required
before a measurement is allowed into the normal law-discovery action pool.

The first miner covers object-attached scalar weights in additive invariants.
For measured state changes ``delta_y_i`` it searches for weights ``w_i`` with

    sum_i w_i * delta_y_i ~= 0.

This covers, without naming either target, collision weights over velocity
changes and thermal weights over temperature changes.  A reference object fixes
the unavoidable overall scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


class ForbiddenEvidenceError(ValueError):
    """Raised when discovery attempts to consume a forbidden data channel."""


class UnidentifiableInvariantError(ValueError):
    """Raised when the supplied interventions do not identify one weight vector."""


class ChannelOrigin(str, Enum):
    RAW_SENSOR = "raw_sensor"
    CONTROL_SETTING = "control_setting"
    DERIVED_FROM_ALLOWED = "derived_from_allowed"
    CERTIFIED_OPERATIONAL_CONCEPT = "certified_operational_concept"
    SIMULATOR_HIDDEN = "simulator_hidden"
    ANSWER_TEMPLATE = "answer_template"

    @property
    def allowed_for_discovery(self) -> bool:
        return self not in {self.SIMULATOR_HIDDEN, self.ANSWER_TEMPLATE}


class ConceptStatus(str, Enum):
    PROPOSED = "proposed"
    TRIAL = "trial"
    CERTIFIED = "certified"
    REJECTED = "rejected"


class OperationalScope(str, Enum):
    OBJECT_INTRINSIC = "object_intrinsic"
    STATE = "state"
    FIELD = "field"
    EVENT = "event"
    PATTERN = "pattern"
    FLUX = "flux"


class EvidenceKind(str, Enum):
    REPEATABILITY = "repeatability"
    NUISANCE_INVARIANCE = "nuisance_invariance"
    OBJECT_SENSITIVITY = "object_sensitivity"
    REFERENCE_TRANSITIVITY = "reference_transitivity"
    CROSS_APPARATUS_TRANSFER = "cross_apparatus_transfer"
    HELDOUT_PREDICTION = "heldout_prediction"
    PROVENANCE_AUDIT = "provenance_audit"
    FALSIFICATION = "falsification"


class ValidationTaskKind(str, Enum):
    REPEAT = "repeat"
    VARY_NUISANCE = "vary_nuisance"
    REPLACE_ARGUMENT = "replace_argument"
    REPLACE_REFERENCE = "replace_reference"
    CROSS_APPARATUS = "cross_apparatus"
    HELDOUT_USE = "heldout_use"
    AUDIT_PROVENANCE = "audit_provenance"


@dataclass(frozen=True)
class ChannelRef:
    name: str
    origin: ChannelOrigin

    def ensure_allowed(self) -> None:
        if not self.origin.allowed_for_discovery:
            raise ForbiddenEvidenceError(
                f"Channel {self.name!r} has forbidden origin {self.origin.value!r}"
            )

    def to_dict(self) -> dict:
        return {"name": self.name, "origin": self.origin.value}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ChannelRef":
        return cls(str(data["name"]), ChannelOrigin(data["origin"]))


@dataclass
class InvariantObservation:
    """Measured changes from one controlled experiment.

    ``deltas`` has shape ``(transitions, entities)``.  It should be computed
    from allowed observations before construction; for a state trajectory a
    useful choice is ``state[t] - state[t0]``.  Object identities, rather than
    apparatus slot names, should be used when the same physical object is
    transported between experiments.
    """

    observation_id: str
    experiment_id: str
    experiment_family: str
    observable: str
    entity_ids: tuple[str, ...]
    deltas: np.ndarray
    channels: tuple[ChannelRef, ...]
    interventions: dict[str, str | float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entity_ids = tuple(self.entity_ids)
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("entity_ids must be unique within an observation")
        array = np.asarray(self.deltas, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != len(self.entity_ids):
            raise ValueError(
                "deltas must have shape (transitions, len(entity_ids))"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("deltas contain non-finite values")
        self.deltas = array
        self.channels = tuple(self.channels)

    @classmethod
    def from_state_series(
        cls,
        *,
        observation_id: str,
        experiment_id: str,
        experiment_family: str,
        observable: str,
        entity_ids: Sequence[str],
        states: np.ndarray,
        channels: Sequence[ChannelRef],
        interventions: Mapping[str, str | float | int] | None = None,
        baseline_index: int = 0,
    ) -> "InvariantObservation":
        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(entity_ids):
            raise ValueError("states must have shape (samples, len(entity_ids))")
        if not 0 <= baseline_index < len(values):
            raise IndexError("baseline_index is outside the state series")
        deltas = values - values[baseline_index]
        deltas = np.delete(deltas, baseline_index, axis=0)
        return cls(
            observation_id=observation_id,
            experiment_id=experiment_id,
            experiment_family=experiment_family,
            observable=observable,
            entity_ids=tuple(entity_ids),
            deltas=deltas,
            channels=tuple(channels),
            interventions=dict(interventions or {}),
        )

    def ensure_allowed(self) -> None:
        if not self.channels:
            raise ForbiddenEvidenceError(
                f"Observation {self.observation_id!r} has no provenance"
            )
        for channel in self.channels:
            channel.ensure_allowed()


@dataclass(frozen=True)
class AdditiveInvariantFit:
    observable: str
    reference_entity: str
    reference_value: float
    weights: dict[str, float]
    standard_errors: dict[str, float]
    relative_residual: float
    spectral_gap: float
    singular_values: tuple[float, ...]
    observation_ids: tuple[str, ...]
    definition_experiments: tuple[str, ...]
    definition_families: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "observable": self.observable,
            "reference_entity": self.reference_entity,
            "reference_value": self.reference_value,
            "weights": self.weights,
            "standard_errors": self.standard_errors,
            "relative_residual": self.relative_residual,
            "spectral_gap": self.spectral_gap,
            "singular_values": list(self.singular_values),
            "observation_ids": list(self.observation_ids),
            "definition_experiments": list(self.definition_experiments),
            "definition_families": list(self.definition_families),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "AdditiveInvariantFit":
        return cls(
            observable=str(data["observable"]),
            reference_entity=str(data["reference_entity"]),
            reference_value=float(data["reference_value"]),
            weights={str(k): float(v) for k, v in data["weights"].items()},
            standard_errors={
                str(k): float(v) for k, v in data.get("standard_errors", {}).items()
            },
            relative_residual=float(data["relative_residual"]),
            spectral_gap=float(data["spectral_gap"]),
            singular_values=tuple(float(x) for x in data["singular_values"]),
            observation_ids=tuple(data["observation_ids"]),
            definition_experiments=tuple(data["definition_experiments"]),
            definition_families=tuple(data["definition_families"]),
        )


@dataclass(frozen=True)
class AdditiveInvariantMinerConfig:
    max_relative_residual: float = 0.03
    min_spectral_gap: float = 5.0
    min_nonzero_rows: int = 4
    row_zero_tolerance: float = 1e-12
    bootstrap_samples: int = 64
    bootstrap_seed: int = 20260718


class AdditiveInvariantMiner:
    """Discover one shared object-weight vector from controlled transitions."""

    def __init__(self, config: AdditiveInvariantMinerConfig | None = None):
        self.config = config or AdditiveInvariantMinerConfig()

    @staticmethod
    def _null_vector(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
        return vh[-1], singular_values

    def fit(
        self,
        observations: Sequence[InvariantObservation],
        *,
        reference_entity: str,
        reference_value: float = 1.0,
    ) -> AdditiveInvariantFit:
        observations = tuple(observations)
        if not observations:
            raise UnidentifiableInvariantError("No observations were supplied")
        for observation in observations:
            observation.ensure_allowed()
        observables = {observation.observable for observation in observations}
        if len(observables) != 1:
            raise ValueError("All observations must use the same observable")

        entity_ids = sorted(
            {entity for observation in observations for entity in observation.entity_ids}
        )
        if reference_entity not in entity_ids:
            raise UnidentifiableInvariantError(
                f"Reference entity {reference_entity!r} was never observed"
            )
        if len(entity_ids) < 2:
            raise UnidentifiableInvariantError("At least two entities are required")
        index = {entity: position for position, entity in enumerate(entity_ids)}

        rows = []
        for observation in observations:
            for local_row in observation.deltas:
                row = np.zeros(len(entity_ids), dtype=float)
                for local_index, entity in enumerate(observation.entity_ids):
                    row[index[entity]] = local_row[local_index]
                norm = float(np.linalg.norm(row))
                if norm > self.config.row_zero_tolerance:
                    rows.append(row / norm)
        if len(rows) < max(self.config.min_nonzero_rows, len(entity_ids)):
            raise UnidentifiableInvariantError(
                "Too few nonzero controlled transitions to identify the weights"
            )
        matrix = np.asarray(rows, dtype=float)
        vector, singular_values = self._null_vector(matrix)
        reference_coefficient = float(vector[index[reference_entity]])
        if abs(reference_coefficient) <= self.config.row_zero_tolerance:
            raise UnidentifiableInvariantError(
                "The discovered null vector cannot be normalized by the reference"
            )
        vector = vector * (reference_value / reference_coefficient)
        relative_residual = float(
            np.linalg.norm(matrix @ vector)
            / (np.sqrt(len(matrix)) * max(np.linalg.norm(vector), 1e-15))
        )
        if len(singular_values) < 2:
            spectral_gap = float("inf")
        else:
            spectral_gap = float(
                singular_values[-2] / max(singular_values[-1], 1e-15)
            )
        if relative_residual > self.config.max_relative_residual:
            raise UnidentifiableInvariantError(
                f"Best invariant residual {relative_residual:.4g} exceeds "
                f"{self.config.max_relative_residual:.4g}"
            )
        if spectral_gap < self.config.min_spectral_gap:
            raise UnidentifiableInvariantError(
                f"Null direction is ambiguous: spectral gap {spectral_gap:.4g} "
                f"is below {self.config.min_spectral_gap:.4g}"
            )

        standard_errors = self._bootstrap_errors(
            matrix, entity_ids, index[reference_entity], reference_value
        )
        return AdditiveInvariantFit(
            observable=next(iter(observables)),
            reference_entity=reference_entity,
            reference_value=float(reference_value),
            weights={entity: float(vector[index[entity]]) for entity in entity_ids},
            standard_errors=standard_errors,
            relative_residual=relative_residual,
            spectral_gap=spectral_gap,
            singular_values=tuple(float(value) for value in singular_values),
            observation_ids=tuple(sorted({item.observation_id for item in observations})),
            definition_experiments=tuple(
                sorted({item.experiment_id for item in observations})
            ),
            definition_families=tuple(
                sorted({item.experiment_family for item in observations})
            ),
        )

    def _bootstrap_errors(
        self,
        matrix: np.ndarray,
        entity_ids: Sequence[str],
        reference_index: int,
        reference_value: float,
    ) -> dict[str, float]:
        if self.config.bootstrap_samples <= 1:
            return {entity: 0.0 for entity in entity_ids}
        rng = np.random.default_rng(self.config.bootstrap_seed)
        estimates = []
        for _ in range(self.config.bootstrap_samples):
            sample = matrix[rng.integers(0, len(matrix), size=len(matrix))]
            vector, _ = self._null_vector(sample)
            reference = float(vector[reference_index])
            if abs(reference) <= self.config.row_zero_tolerance:
                continue
            normalized = vector * (reference_value / reference)
            if np.all(np.isfinite(normalized)) and np.max(np.abs(normalized)) < 1e8:
                estimates.append(normalized)
        if len(estimates) < 2:
            return {entity: float("inf") for entity in entity_ids}
        errors = np.std(np.asarray(estimates), axis=0, ddof=1)
        return {entity: float(errors[i]) for i, entity in enumerate(entity_ids)}


@dataclass(frozen=True)
class MeasurementProtocol:
    experiment_family: str
    observable: str
    estimator: str
    required_channels: tuple[str, ...]
    reference_entity: str | None = None
    reference_value: float | None = None
    nuisance_controls: tuple[str, ...] = ()
    alternate_experiment_families: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_family": self.experiment_family,
            "observable": self.observable,
            "estimator": self.estimator,
            "required_channels": list(self.required_channels),
            "reference_entity": self.reference_entity,
            "reference_value": self.reference_value,
            "nuisance_controls": list(self.nuisance_controls),
            "alternate_experiment_families": list(
                self.alternate_experiment_families
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MeasurementProtocol":
        return cls(
            experiment_family=str(data["experiment_family"]),
            observable=str(data["observable"]),
            estimator=str(data["estimator"]),
            required_channels=tuple(data.get("required_channels", ())),
            reference_entity=data.get("reference_entity"),
            reference_value=(
                None
                if data.get("reference_value") is None
                else float(data["reference_value"])
            ),
            nuisance_controls=tuple(data.get("nuisance_controls", ())),
            alternate_experiment_families=tuple(
                data.get("alternate_experiment_families", ())
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class ValidationEvidence:
    kind: EvidenceKind
    experiment_id: str
    experiment_family: str
    score: float
    independent: bool = True
    intervention: str = ""
    held_fixed: tuple[str, ...] = ()
    channels: tuple[ChannelRef, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Evidence score must lie in [0, 1]")
        for channel in self.channels:
            channel.ensure_allowed()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "experiment_id": self.experiment_id,
            "experiment_family": self.experiment_family,
            "score": self.score,
            "independent": self.independent,
            "intervention": self.intervention,
            "held_fixed": list(self.held_fixed),
            "channels": [channel.to_dict() for channel in self.channels],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "ValidationEvidence":
        return cls(
            kind=EvidenceKind(data["kind"]),
            experiment_id=str(data["experiment_id"]),
            experiment_family=str(data["experiment_family"]),
            score=float(data["score"]),
            independent=bool(data.get("independent", True)),
            intervention=str(data.get("intervention", "")),
            held_fixed=tuple(data.get("held_fixed", ())),
            channels=tuple(
                ChannelRef.from_dict(channel) for channel in data.get("channels", ())
            ),
            notes=str(data.get("notes", "")),
        )


@dataclass
class OperationalConceptRecord:
    candidate_id: str
    status: ConceptStatus
    scope: OperationalScope
    entity_type: str
    protocol: MeasurementProtocol
    fit: AdditiveInvariantFit
    created_epoch: int = 0
    version: int = 1
    symbolic_name: str | None = None
    applicable_experiment_families: tuple[str, ...] = ()
    evidence: list[ValidationEvidence] = field(default_factory=list)
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "scope": self.scope.value,
            "entity_type": self.entity_type,
            "protocol": self.protocol.to_dict(),
            "fit": self.fit.to_dict(),
            "created_epoch": self.created_epoch,
            "version": self.version,
            "symbolic_name": self.symbolic_name,
            "applicable_experiment_families": list(
                self.applicable_experiment_families
            ),
            "evidence": [item.to_dict() for item in self.evidence],
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "OperationalConceptRecord":
        return cls(
            candidate_id=str(data["candidate_id"]),
            status=ConceptStatus(data["status"]),
            scope=OperationalScope(data["scope"]),
            entity_type=str(data["entity_type"]),
            protocol=MeasurementProtocol.from_dict(data["protocol"]),
            fit=AdditiveInvariantFit.from_dict(data["fit"]),
            created_epoch=int(data.get("created_epoch", 0)),
            version=int(data.get("version", 1)),
            symbolic_name=data.get("symbolic_name"),
            applicable_experiment_families=tuple(
                data.get("applicable_experiment_families", ())
            ),
            evidence=[
                ValidationEvidence.from_dict(item)
                for item in data.get("evidence", ())
            ],
            rejection_reason=str(data.get("rejection_reason", "")),
        )


@dataclass(frozen=True)
class CertificationPolicy:
    minimum_score: float = 0.8
    max_fit_residual: float = 0.03
    min_fit_spectral_gap: float = 5.0
    required_counts: dict[EvidenceKind, int] = field(
        default_factory=lambda: {
            EvidenceKind.REPEATABILITY: 1,
            EvidenceKind.NUISANCE_INVARIANCE: 2,
            EvidenceKind.OBJECT_SENSITIVITY: 1,
            EvidenceKind.REFERENCE_TRANSITIVITY: 1,
            EvidenceKind.CROSS_APPARATUS_TRANSFER: 1,
            EvidenceKind.HELDOUT_PREDICTION: 1,
            EvidenceKind.PROVENANCE_AUDIT: 1,
        }
    )


@dataclass(frozen=True)
class ValidationTask:
    task_id: str
    candidate_id: str
    kind: ValidationTaskKind
    experiment_family: str | None
    intervention: str
    held_fixed: tuple[str, ...]
    required_channels: tuple[str, ...]
    priority: float
    question: str


_EVIDENCE_TO_TASK = {
    EvidenceKind.REPEATABILITY: ValidationTaskKind.REPEAT,
    EvidenceKind.NUISANCE_INVARIANCE: ValidationTaskKind.VARY_NUISANCE,
    EvidenceKind.OBJECT_SENSITIVITY: ValidationTaskKind.REPLACE_ARGUMENT,
    EvidenceKind.REFERENCE_TRANSITIVITY: ValidationTaskKind.REPLACE_REFERENCE,
    EvidenceKind.CROSS_APPARATUS_TRANSFER: ValidationTaskKind.CROSS_APPARATUS,
    EvidenceKind.HELDOUT_PREDICTION: ValidationTaskKind.HELDOUT_USE,
    EvidenceKind.PROVENANCE_AUDIT: ValidationTaskKind.AUDIT_PROVENANCE,
}


class OperationalConceptRegistry:
    """Lifecycle, evidence, and active-validation queue for trial concepts."""

    def __init__(self, policy: CertificationPolicy | None = None):
        self.policy = policy or CertificationPolicy()
        self.records: dict[str, OperationalConceptRecord] = {}
        self._next_id = 1

    def propose_additive_weight(
        self,
        fit: AdditiveInvariantFit,
        protocol: MeasurementProtocol,
        *,
        entity_type: str,
        created_epoch: int = 0,
        symbolic_name: str | None = None,
        applicable_experiment_families: Sequence[str] = (),
        begin_trial: bool = True,
    ) -> OperationalConceptRecord:
        if fit.observable != protocol.observable:
            raise ValueError("Fit and protocol refer to different observables")
        if protocol.reference_entity != fit.reference_entity:
            raise ValueError("Fit and protocol use different reference entities")
        candidate_id = f"OC_{self._next_id:04d}"
        self._next_id += 1
        record = OperationalConceptRecord(
            candidate_id=candidate_id,
            status=ConceptStatus.TRIAL if begin_trial else ConceptStatus.PROPOSED,
            scope=OperationalScope.OBJECT_INTRINSIC,
            entity_type=entity_type,
            protocol=protocol,
            fit=fit,
            created_epoch=created_epoch,
            symbolic_name=symbolic_name,
            applicable_experiment_families=tuple(applicable_experiment_families),
        )
        self.records[candidate_id] = record
        return record

    def begin_trial(self, candidate_id: str) -> OperationalConceptRecord:
        record = self.records[candidate_id]
        if record.status is ConceptStatus.PROPOSED:
            record.status = ConceptStatus.TRIAL
        return record

    def record_evidence(
        self, candidate_id: str, evidence: ValidationEvidence
    ) -> ConceptStatus:
        record = self.records[candidate_id]
        if record.status is ConceptStatus.REJECTED:
            return record.status
        record.evidence.append(evidence)
        return self.evaluate(candidate_id)

    def _qualifying_evidence(
        self, record: OperationalConceptRecord, kind: EvidenceKind
    ) -> list[ValidationEvidence]:
        result = []
        definition_experiments = set(record.fit.definition_experiments)
        definition_families = set(record.fit.definition_families)
        for item in record.evidence:
            if item.kind is not kind or item.score < self.policy.minimum_score:
                continue
            if kind is EvidenceKind.PROVENANCE_AUDIT:
                result.append(item)
                continue
            if not item.independent or item.experiment_id in definition_experiments:
                continue
            if (
                kind is EvidenceKind.CROSS_APPARATUS_TRANSFER
                and item.experiment_family in definition_families
            ):
                continue
            result.append(item)
        return result

    def evidence_counts(self, candidate_id: str) -> dict[EvidenceKind, int]:
        record = self.records[candidate_id]
        counts = {}
        for kind in self.policy.required_counts:
            evidence = self._qualifying_evidence(record, kind)
            if kind is EvidenceKind.NUISANCE_INVARIANCE:
                counts[kind] = len(
                    {item.intervention or item.experiment_id for item in evidence}
                )
            else:
                counts[kind] = len(evidence)
        return counts

    def evaluate(self, candidate_id: str) -> ConceptStatus:
        record = self.records[candidate_id]
        falsifications = [
            item
            for item in record.evidence
            if item.kind is EvidenceKind.FALSIFICATION
            and item.score >= self.policy.minimum_score
        ]
        if falsifications:
            record.status = ConceptStatus.REJECTED
            record.rejection_reason = falsifications[-1].notes or "controlled falsification"
            return record.status
        if record.status is ConceptStatus.PROPOSED:
            return record.status
        if record.fit.relative_residual > self.policy.max_fit_residual:
            return record.status
        if record.fit.spectral_gap < self.policy.min_fit_spectral_gap:
            return record.status
        counts = self.evidence_counts(candidate_id)
        if all(
            counts.get(kind, 0) >= required
            for kind, required in self.policy.required_counts.items()
        ):
            record.status = ConceptStatus.CERTIFIED
        return record.status

    def missing_requirements(self, candidate_id: str) -> dict[EvidenceKind, int]:
        counts = self.evidence_counts(candidate_id)
        return {
            kind: max(0, required - counts.get(kind, 0))
            for kind, required in self.policy.required_counts.items()
            if required > counts.get(kind, 0)
        }

    def validation_tasks(self, candidate_id: str) -> list[ValidationTask]:
        record = self.records[candidate_id]
        if record.status is not ConceptStatus.TRIAL:
            return []
        tasks = []
        missing = self.missing_requirements(candidate_id)
        for evidence_kind, count in missing.items():
            task_kind = _EVIDENCE_TO_TASK[evidence_kind]
            for offset in range(count):
                family: str | None = record.protocol.experiment_family
                intervention = task_kind.value
                if evidence_kind is EvidenceKind.NUISANCE_INVARIANCE:
                    controls = record.protocol.nuisance_controls
                    if controls:
                        intervention = f"vary:{controls[offset % len(controls)]}"
                elif evidence_kind is EvidenceKind.CROSS_APPARATUS_TRANSFER:
                    alternatives = record.protocol.alternate_experiment_families
                    family = alternatives[offset % len(alternatives)] if alternatives else None
                    intervention = "use_independent_apparatus"
                elif evidence_kind is EvidenceKind.OBJECT_SENSITIVITY:
                    intervention = "replace_argument_object"
                elif evidence_kind is EvidenceKind.REFERENCE_TRANSITIVITY:
                    intervention = "replace_reference_and_close_ratio_cycle"
                elif evidence_kind is EvidenceKind.HELDOUT_PREDICTION:
                    intervention = "use_frozen_measurement_in_heldout_law_search"
                elif evidence_kind is EvidenceKind.PROVENANCE_AUDIT:
                    family = None
                    intervention = "audit_channel_dependency_graph"
                tasks.append(
                    ValidationTask(
                        task_id=(
                            f"{candidate_id}:{task_kind.value}:"
                            f"{self.policy.required_counts[evidence_kind] - count + offset + 1}"
                        ),
                        candidate_id=candidate_id,
                        kind=task_kind,
                        experiment_family=family,
                        intervention=intervention,
                        held_fixed=("candidate_definition",),
                        required_channels=record.protocol.required_channels,
                        priority=1.0 + 0.25 * count,
                        question=(
                            f"Does {candidate_id} satisfy "
                            f"{evidence_kind.value.replace('_', ' ')}?"
                        ),
                    )
                )
        return tasks

    def all_validation_tasks(self) -> list[ValidationTask]:
        return [
            task
            for candidate_id in sorted(self.records)
            for task in self.validation_tasks(candidate_id)
        ]

    def eligible_records(
        self, experiment_family: str, *, purpose: str = "law_discovery"
    ) -> list[OperationalConceptRecord]:
        eligible_status = (
            {ConceptStatus.CERTIFIED}
            if purpose == "law_discovery"
            else {ConceptStatus.TRIAL, ConceptStatus.CERTIFIED}
        )
        return [
            record
            for record in self.records.values()
            if record.status in eligible_status
            and (
                not record.applicable_experiment_families
                or experiment_family in record.applicable_experiment_families
            )
        ]

    def to_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "records": {
                candidate_id: record.to_dict()
                for candidate_id, record in self.records.items()
            },
        }

    @classmethod
    def from_dict(
        cls, data: Mapping, policy: CertificationPolicy | None = None
    ) -> "OperationalConceptRegistry":
        registry = cls(policy=policy)
        registry._next_id = int(data.get("next_id", 1))
        registry.records = {
            str(candidate_id): OperationalConceptRecord.from_dict(record)
            for candidate_id, record in data.get("records", {}).items()
        }
        return registry

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(
        cls, path: str | Path, policy: CertificationPolicy | None = None
    ) -> "OperationalConceptRegistry":
        path = Path(path)
        if not path.exists():
            return cls(policy=policy)
        return cls.from_dict(json.loads(path.read_text()), policy=policy)


class OperationalPipelineBridge:
    """Small adapter between the registry and the existing concept bandit.

    The bridge never manufactures a DSL concept.  Once another component has
    registered a candidate's measurement program under ``symbolic_name``, this
    adapter ensures that only certified records enter the normal action pool.
    Trial records remain visible only to the validation-task selector.
    """

    def __init__(self, registry: OperationalConceptRegistry):
        self.registry = registry

    def validation_tasks(self) -> list[ValidationTask]:
        return self.registry.all_validation_tasks()

    def eligible_symbolic_names(
        self, experiment_family: str, *, purpose: str = "law_discovery"
    ) -> list[str]:
        return sorted(
            record.symbolic_name
            for record in self.registry.eligible_records(
                experiment_family, purpose=purpose
            )
            if record.symbolic_name is not None
        )

    def sync_certified_actions(
        self,
        register_action: Callable[[str, float], object],
        *,
        initial_value: float = 8.0,
    ) -> list[str]:
        synced = []
        for record in self.registry.records.values():
            if (
                record.status is ConceptStatus.CERTIFIED
                and record.symbolic_name is not None
            ):
                register_action(record.symbolic_name, initial_value)
                synced.append(record.symbolic_name)
        return sorted(set(synced))


def allowed_channels(names: Iterable[str]) -> tuple[ChannelRef, ...]:
    """Convenience constructor for values derived solely from raw sensors."""

    return tuple(
        ChannelRef(str(name), ChannelOrigin.DERIVED_FROM_ALLOWED) for name in names
    )
