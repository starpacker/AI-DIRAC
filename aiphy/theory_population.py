"""
Theory population and competing-hypothesis management — Phase 2.

Manages the lifecycle of physical theories:
  active      — currently accepted, makes predictions
  falsified   — contradicted by experiment, retained for audit
  restricted  — valid in a narrower domain than originally claimed
  archived    — superseded by a better theory, lineage preserved
  equivalent  — two theories make identical predictions in ALL observations

Each theory has an immutable ID. Prediction records are frozen
before experiment execution (pre-registered). When experiment results
contradict a theory, it is NOT deleted — it transitions to a
falsified/restricted state with full audit trail.

Key invariants:
  1. No pre-registered prediction → cannot enter general law
  2. Falsified theories remain in the database (never deleted)
  3. Two active theories with identical predictions → report equivalent
  4. Falsification earns POSITIVE information reward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from copy import deepcopy
import time
import json


class TheoryStatus(Enum):
    """Lifecycle status of a physical theory."""
    ACTIVE = auto()         # Currently accepted, validated
    FALSIFIED = auto()      # Contradicted by experiment
    RESTRICTED = auto()     # Valid in narrower domain
    ARCHIVED = auto()       # Superseded, lineage preserved
    CANDIDATE = auto()      # Proposed but not yet validated
    EQUIVALENT = auto()     # Makes same predictions as another theory


@dataclass
class TheoryRecord:
    """Immutable record of a single physical theory."""
    theory_id: str                                  # immutable ID
    name: str                                       # human-readable name
    expression: str                                 # symbolic expression
    status: TheoryStatus = TheoryStatus.CANDIDATE
    parent_id: Optional[str] = None                 # lineage
    valid_experiments: List[str] = field(default_factory=list)
    falsified_by: Optional[str] = None              # experiment that falsified it
    restricted_domain: Optional[str] = None          # domain restriction note
    superseded_by: Optional[str] = None             # theory ID that replaced it
    created_at: float = field(default_factory=time.time)
    falsified_at: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "name": self.name,
            "expression": self.expression,
            "status": self.status.name,
            "parent_id": self.parent_id,
            "valid_experiments": self.valid_experiments,
            "falsified_by": self.falsified_by,
            "restricted_domain": self.restricted_domain,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "falsified_at": self.falsified_at,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TheoryRecord":
        return cls(
            theory_id=d["theory_id"],
            name=d.get("name", d["theory_id"]),
            expression=d["expression"],
            status=TheoryStatus[d["status"]],
            parent_id=d.get("parent_id"),
            valid_experiments=d.get("valid_experiments", []),
            falsified_by=d.get("falsified_by"),
            restricted_domain=d.get("restricted_domain"),
            superseded_by=d.get("superseded_by"),
            created_at=d.get("created_at", time.time()),
            falsified_at=d.get("falsified_at"),
            extra=d.get("extra", {}),
        )


class TheoryPopulation:
    """
    Manages the population of competing physical theories.

    Tracks theory lineage, status transitions, and provides query
    methods for the experiment selector and falsification judge.
    """

    def __init__(self):
        self._theories: Dict[str, TheoryRecord] = {}
        self._id_counter: int = 0
        self._status_history: Dict[str, List[Tuple[float, str, str]]] = {}
        self._equivalent_groups: Dict[str, Set[str]] = {}

    @property
    def active_theories(self) -> Dict[str, TheoryRecord]:
        """Return all currently active theories."""
        return {
            tid: t for tid, t in self._theories.items()
            if t.status == TheoryStatus.ACTIVE
        }

    @property
    def falsified_theories(self) -> Dict[str, TheoryRecord]:
        """Return all falsified theories (never deleted)."""
        return {
            tid: t for tid, t in self._theories.items()
            if t.status == TheoryStatus.FALSIFIED
        }

    @property
    def all_theories(self) -> Dict[str, TheoryRecord]:
        return dict(self._theories)

    def count(self) -> int:
        return len(self._theories)

    def count_active(self) -> int:
        return sum(1 for t in self._theories.values()
                   if t.status == TheoryStatus.ACTIVE)

    def add_theory(
        self,
        name: str,
        expression: str,
        parent_id: Optional[str] = None,
        status: TheoryStatus = TheoryStatus.CANDIDATE,
        valid_experiments: Optional[List[str]] = None,
    ) -> str:
        """Register a new theory. Returns the immutable theory ID."""
        self._id_counter += 1
        tid = f"TH_{self._id_counter:04d}"

        record = TheoryRecord(
            theory_id=tid,
            name=name,
            expression=expression,
            status=status,
            parent_id=parent_id,
            valid_experiments=valid_experiments or [],
        )
        self._theories[tid] = record
        self._record_transition(tid, status.name)
        return tid

    def get_theory(self, theory_id: str) -> Optional[TheoryRecord]:
        return self._theories.get(theory_id)

    def activate(self, theory_id: str) -> bool:
        """Move a candidate to active status."""
        record = self._theories.get(theory_id)
        if record is None:
            return False
        old_status = record.status.name
        record.status = TheoryStatus.ACTIVE
        self._record_transition(theory_id, "ACTIVE", old_status)
        return True

    def falsify(
        self,
        theory_id: str,
        falsifying_experiment: str,
        reason: str = "",
    ) -> bool:
        """
        Falsify a theory — it is NOT deleted, only marked.
        Positive information reward; we learned something.
        """
        record = self._theories.get(theory_id)
        if record is None:
            return False
        old_status = record.status.name
        record.status = TheoryStatus.FALSIFIED
        record.falsified_by = falsifying_experiment
        record.falsified_at = time.time()
        if reason:
            record.extra["falsification_reason"] = reason
        self._record_transition(
            theory_id, "FALSIFIED", old_status,
            f"by {falsifying_experiment}: {reason}",
        )
        return True

    def restrict(
        self,
        theory_id: str,
        domain_note: str,
    ) -> bool:
        """Restrict a theory to a narrower domain."""
        record = self._theories.get(theory_id)
        if record is None:
            return False
        old_status = record.status.name
        record.status = TheoryStatus.RESTRICTED
        record.restricted_domain = domain_note
        self._record_transition(
            theory_id, "RESTRICTED", old_status,
            f"domain: {domain_note}",
        )
        return True

    def archive(
        self,
        theory_id: str,
        superseded_by: str,
        reason: str = "",
    ) -> bool:
        """Archive a theory superseded by a better one."""
        record = self._theories.get(theory_id)
        if record is None:
            return False
        old_status = record.status.name
        record.status = TheoryStatus.ARCHIVED
        record.superseded_by = superseded_by
        record.extra["archive_reason"] = reason or "superseded"
        self._record_transition(
            theory_id, "ARCHIVED", old_status,
            f"superseded by {superseded_by}: {reason}",
        )
        return True

    def mark_equivalent(self, theory_id_a: str, theory_id_b: str) -> None:
        """Record that two theories make identical predictions."""
        gid = f"EQ_{min(theory_id_a, theory_id_b)}_{max(theory_id_a, theory_id_b)}"
        self._equivalent_groups.setdefault(gid, set()).add(theory_id_a)
        self._equivalent_groups.setdefault(gid, set()).add(theory_id_b)

        for tid in (theory_id_a, theory_id_b):
            record = self._theories.get(tid)
            if record and record.status == TheoryStatus.ACTIVE:
                record.extra["has_equivalent"] = True
                record.extra["equivalent_group"] = gid

    def get_lineage(self, theory_id: str) -> List[str]:
        """Return the lineage chain from root to this theory."""
        chain = []
        current = theory_id
        visited: Set[str] = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            record = self._theories.get(current)
            if record and record.parent_id:
                current = record.parent_id
            else:
                break
        return list(reversed(chain))

    def get_status_history(self, theory_id: str) -> List[Tuple[float, str, str]]:
        """Return the status transition history for a theory."""
        return self._status_history.get(theory_id, [])

    def has_competing_theories(self, domain: Optional[str] = None) -> bool:
        """Check if there are ≥2 active theories in competition."""
        actives = list(self.active_theories.values())
        if domain:
            actives = [t for t in actives
                       if any(exp.startswith(domain) for exp in t.valid_experiments)]
        return len(actives) >= 2

    def theories_predicting_on(
        self, experiment_name: str
    ) -> List[TheoryRecord]:
        """Return all active theories that make predictions for an experiment."""
        return [
            t for t in self.active_theories.values()
            if experiment_name in t.valid_experiments
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theories": {
                tid: rec.to_dict() for tid, rec in self._theories.items()
            },
            "id_counter": self._id_counter,
            "status_history": {
                tid: [(ts, old, new) for ts, old, new in hist]
                for tid, hist in self._status_history.items()
            },
            "equivalent_groups": {
                gid: list(members)
                for gid, members in self._equivalent_groups.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TheoryPopulation":
        pop = cls()
        pop._id_counter = d.get("id_counter", 0)
        for tid, rec_d in d.get("theories", {}).items():
            pop._theories[tid] = TheoryRecord.from_dict(rec_d)
        for tid, hist in d.get("status_history", {}).items():
            pop._status_history[tid] = [(float(ts), old, new)
                                         for ts, old, new in hist]
        for gid, members in d.get("equivalent_groups", {}).items():
            pop._equivalent_groups[gid] = set(members)
        return pop

    def _record_transition(
        self, theory_id: str, new_status: str,
        old_status: str = "UNKNOWN", note: str = "",
    ) -> None:
        self._status_history.setdefault(theory_id, []).append(
            (time.time(), old_status, f"{new_status}: {note}" if note else new_status)
        )
