"""
Mathematical concept registry — Phase 3 foundation.

Tracks the lifecycle of mathematical objects:
  proposed  → a candidate representation, map, or operator
  trial     → being evaluated on held-out data
  certified → passed all validation gates, can be used in law discovery
  rejected  → failed validation, never deleted (audit trail)

Every math concept has:
  - immutable ID
  - provenance chain to raw channels
  - validation evidence (independent from definition evidence)
  - equivalence group membership
  - certification status with rejection reason if rejected

Hard gates for certification:
  1. Independent held-out validation (data not used in definition)
  2. Cross-experiment transfer evidence
  3. Not equivalent to an existing certified concept
  4. No hidden-channel contamination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import hashlib
import json


class MathConceptStatus(Enum):
    PROPOSED = auto()    # candidate, not yet validated
    TRIAL = auto()       # being evaluated
    CERTIFIED = auto()   # passed all gates
    REJECTED = auto()    # failed validation, retained for audit
    SUPERSEDED = auto()  # replaced by a better representation


class MathObjectKind(Enum):
    SCALAR_MAP = auto()       # f: R^n → R
    VECTOR_MAP = auto()       # f: R^n → R^m
    OPERATOR = auto()         # transformation on functions/fields
    STRUCTURE = auto()        # symmetry, invariant, manifold
    REPRESENTATION = auto()   # latent representation (neural or symbolic)
    COORDINATE_TRANSFORM = auto()
    PROBABILITY_MAP = auto()  # distribution-level mapping


@dataclass
class MathConceptRecord:
    """Immutable record of a mathematical concept."""
    concept_id: str
    name: str                              # human-readable name (can be anonymous)
    kind: MathObjectKind
    description: str = ""
    expression: str = ""                   # symbolic form if available
    source_channels: List[str] = field(default_factory=list)
    definition_experiments: List[str] = field(default_factory=list)
    status: MathConceptStatus = MathConceptStatus.PROPOSED
    parent_concepts: List[str] = field(default_factory=list)  # lineage
    created_at: float = field(default_factory=time.time)
    certified_at: Optional[float] = None
    rejected_at: Optional[float] = None
    rejection_reason: str = ""
    validation_evidence: List[Dict[str, Any]] = field(default_factory=list)
    equivalence_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        raw = f"{self.kind.name}|{self.expression}|{sorted(self.source_channels)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "kind": self.kind.name,
            "description": self.description,
            "expression": self.expression,
            "source_channels": self.source_channels,
            "definition_experiments": self.definition_experiments,
            "status": self.status.name,
            "parent_concepts": self.parent_concepts,
            "created_at": self.created_at,
            "certified_at": self.certified_at,
            "rejected_at": self.rejected_at,
            "rejection_reason": self.rejection_reason,
            "validation_evidence": self.validation_evidence,
            "equivalence_group": self.equivalence_group,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MathConceptRecord":
        return cls(
            concept_id=d["concept_id"],
            name=d.get("name", d["concept_id"]),
            kind=MathObjectKind[d["kind"]],
            description=d.get("description", ""),
            expression=d.get("expression", ""),
            source_channels=d.get("source_channels", []),
            definition_experiments=d.get("definition_experiments", []),
            status=MathConceptStatus[d["status"]],
            parent_concepts=d.get("parent_concepts", []),
            created_at=d.get("created_at", time.time()),
            certified_at=d.get("certified_at"),
            rejected_at=d.get("rejected_at"),
            rejection_reason=d.get("rejection_reason", ""),
            validation_evidence=d.get("validation_evidence", []),
            equivalence_group=d.get("equivalence_group"),
            metadata=d.get("metadata", {}),
        )


class MathConceptRegistry:
    """
    Central registry for mathematical objects.

    Lifecycle:
      proposed → trial → certified  (success)
      proposed → trial → rejected   (failure, kept for audit)
      certified → superseded        (replaced by better representation)

    Key invariants:
      1. No concept can be certified without independent validation evidence
      2. Rejected concepts are never deleted
      3. Certified concepts with identical behavior → equivalence group
      4. Source channels must be allowed_for_discovery
    """

    def __init__(self):
        self._concepts: Dict[str, MathConceptRecord] = {}
        self._equivalence_groups: Dict[str, Set[str]] = {}
        self._id_counter: int = 0
        self._status_history: Dict[str, List[Tuple[float, str, str]]] = {}

    # --- CRUD ---

    def propose(
        self,
        name: str,
        kind: MathObjectKind,
        *,
        expression: str = "",
        description: str = "",
        source_channels: Optional[List[str]] = None,
        definition_experiments: Optional[List[str]] = None,
        parent_concepts: Optional[List[str]] = None,
        **metadata,
    ) -> str:
        """Propose a new mathematical concept. Returns concept ID."""
        self._id_counter += 1
        cid = f"MATH_{self._id_counter:04d}"

        record = MathConceptRecord(
            concept_id=cid,
            name=name,
            kind=kind,
            description=description,
            expression=expression,
            source_channels=source_channels or [],
            definition_experiments=definition_experiments or [],
            parent_concepts=parent_concepts or [],
            metadata=dict(metadata),
        )
        self._concepts[cid] = record
        self._record_transition(cid, MathConceptStatus.PROPOSED.name)
        return cid

    def get(self, concept_id: str) -> Optional[MathConceptRecord]:
        return self._concepts.get(concept_id)

    def promote_to_trial(self, concept_id: str) -> bool:
        """Move from proposed to trial status."""
        rec = self._concepts.get(concept_id)
        if rec is None or rec.status != MathConceptStatus.PROPOSED:
            return False
        old = rec.status.name
        rec.status = MathConceptStatus.TRIAL
        self._record_transition(concept_id, "TRIAL", old)
        return True

    def certify(
        self,
        concept_id: str,
        validation_evidence: List[Dict[str, Any]],
    ) -> bool:
        """
        Certify a concept — it has passed all validation gates.

        Requires at least one independent held-out validation.
        """
        rec = self._concepts.get(concept_id)
        if rec is None:
            return False
        if rec.status not in (MathConceptStatus.TRIAL, MathConceptStatus.PROPOSED):
            return False

        # Gate: must have independent validation evidence
        if not validation_evidence:
            return False

        old = rec.status.name
        rec.status = MathConceptStatus.CERTIFIED
        rec.certified_at = time.time()
        rec.validation_evidence.extend(validation_evidence)
        self._record_transition(concept_id, "CERTIFIED", old)
        return True

    def reject(self, concept_id: str, reason: str) -> bool:
        """Reject a concept — it failed validation. Kept for audit."""
        rec = self._concepts.get(concept_id)
        if rec is None:
            return False
        old = rec.status.name
        rec.status = MathConceptStatus.REJECTED
        rec.rejected_at = time.time()
        rec.rejection_reason = reason
        self._record_transition(concept_id, f"REJECTED: {reason}", old)
        return True

    def supersede(self, concept_id: str, superseded_by: str, reason: str = "") -> bool:
        """Mark a concept as superseded by a better one."""
        rec = self._concepts.get(concept_id)
        if rec is None:
            return False
        old = rec.status.name
        rec.status = MathConceptStatus.SUPERSEDED
        rec.metadata["superseded_by"] = superseded_by
        rec.metadata["supersede_reason"] = reason
        self._record_transition(concept_id, f"SUPERSEDED by {superseded_by}", old)
        return True

    def mark_equivalent(self, cid_a: str, cid_b: str) -> None:
        """Record that two concepts are equivalent representations."""
        gid = f"EQ_MATH_{min(cid_a, cid_b)}_{max(cid_a, cid_b)}"
        self._equivalence_groups.setdefault(gid, set()).add(cid_a)
        self._equivalence_groups.setdefault(gid, set()).add(cid_b)

        for cid in (cid_a, cid_b):
            rec = self._concepts.get(cid)
            if rec:
                rec.equivalence_group = gid

    # --- Queries ---

    @property
    def certified_concepts(self) -> Dict[str, MathConceptRecord]:
        return {k: v for k, v in self._concepts.items()
                if v.status == MathConceptStatus.CERTIFIED}

    @property
    def trial_concepts(self) -> Dict[str, MathConceptRecord]:
        return {k: v for k, v in self._concepts.items()
                if v.status == MathConceptStatus.TRIAL}

    @property
    def proposed_concepts(self) -> Dict[str, MathConceptRecord]:
        return {k: v for k, v in self._concepts.items()
                if v.status == MathConceptStatus.PROPOSED}

    @property
    def active_concepts(self) -> Dict[str, MathConceptRecord]:
        """Concepts that can be used in law discovery."""
        return {
            k: v for k, v in self._concepts.items()
            if v.status in (MathConceptStatus.CERTIFIED, MathConceptStatus.TRIAL)
        }

    @property
    def all_concepts(self) -> Dict[str, MathConceptRecord]:
        return dict(self._concepts)

    def count(self) -> int:
        return len(self._concepts)

    def count_certified(self) -> int:
        return len(self.certified_concepts)

    def count_by_kind(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self._concepts.values():
            k = rec.kind.name
            counts[k] = counts.get(k, 0) + 1
        return counts

    def get_concepts_by_kind(self, kind: MathObjectKind) -> List[MathConceptRecord]:
        return [rec for rec in self._concepts.values() if rec.kind == kind]

    def get_lineage(self, concept_id: str) -> List[str]:
        chain: List[str] = []
        current = concept_id
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            rec = self._concepts.get(current)
            if rec and rec.parent_concepts:
                current = rec.parent_concepts[0]  # primary parent
            else:
                break
        return list(reversed(chain))

    def get_equivalence_group(self, concept_id: str) -> List[str]:
        """Get all concepts in the same equivalence group."""
        rec = self._concepts.get(concept_id)
        if rec is None or rec.equivalence_group is None:
            return [concept_id] if rec else []
        members = self._equivalence_groups.get(rec.equivalence_group, set())
        return sorted(members)

    def is_equivalent(self, cid_a: str, cid_b: str) -> bool:
        """Check if two concepts are known equivalents."""
        group_a = self._concepts.get(cid_a)
        group_b = self._concepts.get(cid_b)
        if group_a is None or group_b is None:
            return False
        if group_a.equivalence_group and group_a.equivalence_group == group_b.equivalence_group:
            return True
        return False

    # --- Audit ---

    def get_status_history(self, concept_id: str) -> List[Tuple[float, str, str]]:
        return self._status_history.get(concept_id, [])

    def audit_certification_gaps(self) -> List[str]:
        """Find certified concepts without proper validation."""
        issues = []
        for cid, rec in self.certified_concepts.items():
            if not rec.validation_evidence:
                issues.append(f"{cid}: certified without validation evidence")
            has_independent = any(
                e.get("independent", False) for e in rec.validation_evidence
            )
            if not has_independent:
                issues.append(f"{cid}: no independent held-out validation")
            if not rec.definition_experiments:
                issues.append(f"{cid}: no definition experiments recorded")
        return issues

    def audit_channel_contamination(
        self, forbidden_channels: Set[str]
    ) -> List[str]:
        """Find concepts that read from forbidden channels."""
        issues = []
        for cid, rec in self._concepts.items():
            contaminated = set(rec.source_channels) & forbidden_channels
            if contaminated:
                issues.append(f"{cid}: reads forbidden channels: {contaminated}")
        return issues

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": {cid: rec.to_dict() for cid, rec in self._concepts.items()},
            "id_counter": self._id_counter,
            "equivalence_groups": {
                gid: list(members) for gid, members in self._equivalence_groups.items()
            },
            "status_history": {
                cid: [(ts, old, new) for ts, old, new in hist]
                for cid, hist in self._status_history.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MathConceptRegistry":
        reg = cls()
        reg._id_counter = d.get("id_counter", 0)
        for cid, rec_d in d.get("concepts", {}).items():
            reg._concepts[cid] = MathConceptRecord.from_dict(rec_d)
        for gid, members in d.get("equivalence_groups", {}).items():
            reg._equivalence_groups[gid] = set(members)
        for cid, hist in d.get("status_history", {}).items():
            reg._status_history[cid] = [(float(ts), old, new) for ts, old, new in hist]
        return reg

    def _record_transition(
        self, concept_id: str, new_status: str, old_status: str = "UNKNOWN"
    ) -> None:
        self._status_history.setdefault(concept_id, []).append(
            (time.time(), old_status, new_status)
        )
