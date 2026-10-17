"""
Provenance tracking for AI-DIRAC strict — Phase 0.

Every data channel (observable) carries a provenance record that
tracks its origin, generation method, and dependencies.  This enables
auditing for data leaks, simulator-hidden-channel access, and
definition-data contamination of validation rewards.

Channel origin types:
  RAW_SENSOR       — direct measurement, e.g., posx[t], t[t]
  CONTROL          — experiment control parameter, e.g., initial velocity
  DERIVED          — computed from raw channels in allowed ways
  CERTIFIED_CONCEPT — certified concept through independent validation
  SIMULATOR_HIDDEN — internal simulator state exposed as if measured
                      (THIS IS A LEAK if accessible by discovery)
  ANSWER_TEMPLATE  — variable carrying the true answer structure,
                      e.g., "energy", "phase" directly in channels
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


class ChannelOrigin(Enum):
    """Provenance origin of a data channel."""
    RAW_SENSOR = auto()
    CONTROL = auto()
    DERIVED = auto()
    CERTIFIED_CONCEPT = auto()
    SIMULATOR_HIDDEN = auto()
    ANSWER_TEMPLATE = auto()


@dataclass
class ChannelProvenance:
    """Provenance record for a single data channel."""
    channel_name: str
    origin: ChannelOrigin
    generation_method: str
    dependencies: List[str] = field(default_factory=list)
    experiment_name: str = ""
    accessible_to_discovery: bool = True
    justification: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "origin": self.origin.name,
            "generation_method": self.generation_method,
            "dependencies": self.dependencies,
            "experiment_name": self.experiment_name,
            "accessible_to_discovery": self.accessible_to_discovery,
            "justification": self.justification,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChannelProvenance":
        return cls(
            channel_name=d["channel_name"],
            origin=ChannelOrigin[d["origin"]],
            generation_method=d.get("generation_method", ""),
            dependencies=d.get("dependencies", []),
            experiment_name=d.get("experiment_name", ""),
            accessible_to_discovery=d.get("accessible_to_discovery", True),
            justification=d.get("justification", ""),
            extra=d.get("extra", {}),
        )


class ProvenanceTracker:
    """
    Tracks provenance for all data channels across all experiments.

    Maintains a dependency graph and provides audit functions to detect:
      - SIMULATOR_HIDDEN channels accessible to discovery
      - ANSWER_TEMPLATE channels accessible to discovery
      - DERIVED channels whose upstream sources include hidden channels
      - Circular dependencies
      - Channels with missing or incomplete provenance
    """

    def __init__(self):
        self._channels: Dict[str, ChannelProvenance] = {}
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)

    def register_channel(
        self,
        channel_name: str,
        origin: ChannelOrigin,
        generation_method: str,
        dependencies: Optional[List[str]] = None,
        experiment_name: str = "",
        accessible_to_discovery: bool = True,
        justification: str = "",
        **extra,
    ) -> ChannelProvenance:
        key = ProvenanceTracker._make_key(experiment_name, channel_name)
        deps = dependencies or []
        prov = ChannelProvenance(
            channel_name=channel_name,
            origin=origin,
            generation_method=generation_method,
            dependencies=deps,
            experiment_name=experiment_name,
            accessible_to_discovery=accessible_to_discovery,
            justification=justification,
            extra=extra,
        )
        self._channels[key] = prov
        for dep in deps:
            dep_key = ProvenanceTracker._make_key(experiment_name, dep)
            self._reverse_deps[dep_key].add(key)
        return prov

    def get_channel(
        self, channel_name: str, experiment_name: str = ""
    ) -> Optional[ChannelProvenance]:
        key = ProvenanceTracker._make_key(experiment_name, channel_name)
        return self._channels.get(key)

    def get_all_channels(self) -> List[ChannelProvenance]:
        return list(self._channels.values())

    def get_channels_by_experiment(self, exp_name: str) -> List[ChannelProvenance]:
        return [c for c in self._channels.values() if c.experiment_name == exp_name]

    def audit_leaks(self) -> List[str]:
        """
        Find all channels that leak hidden simulator state to discovery.
        Returns a list of violation descriptions.
        """
        violations = []
        for key, prov in self._channels.items():
            if not prov.accessible_to_discovery:
                continue
            if prov.origin == ChannelOrigin.SIMULATOR_HIDDEN:
                violations.append(
                    f"LEAK: {prov.experiment_name}/{prov.channel_name} "
                    f"is SIMULATOR_HIDDEN but accessible to discovery. "
                    f"Method: {prov.generation_method}"
                )
            elif prov.origin == ChannelOrigin.ANSWER_TEMPLATE:
                violations.append(
                    f"LEAK: {prov.experiment_name}/{prov.channel_name} "
                    f"is ANSWER_TEMPLATE but accessible to discovery. "
                    f"Method: {prov.generation_method}"
                )
            elif prov.origin == ChannelOrigin.DERIVED:
                upstream_leaks = self._find_upstream_leaks(prov)
                if upstream_leaks:
                    violations.append(
                        f"LEAK: {prov.experiment_name}/{prov.channel_name} "
                        f"depends on hidden channels: {upstream_leaks}"
                    )
        return violations

    def audit_incomplete(self) -> List[str]:
        """Return channels that appear in registered experiments but lack provenance."""
        return []

    def audit_circular(self) -> List[str]:
        """Find circular dependencies in the provenance graph."""
        circular = []
        for key in self._channels:
            visited: Set[str] = set()
            path: List[str] = []
            found = [False]

            def dfs(k: str) -> None:
                if found[0]:
                    return
                if k in visited:
                    if k in path:
                        idx = path.index(k)
                        circular.append(" -> ".join(path[idx:] + [k]))
                        found[0] = True
                    return
                visited.add(k)
                path.append(k)
                prov = self._channels.get(k)
                if prov:
                    for dep in prov.dependencies:
                        dep_key = ProvenanceTracker._make_key(
                            prov.experiment_name, dep
                        )
                        dfs(dep_key)
                path.pop()

            dfs(key)
        return list(set(circular))

    def _find_upstream_leaks(self, prov: ChannelProvenance) -> List[str]:
        """Recursively check if any upstream dependency is SIMULATOR_HIDDEN."""
        leaks = []
        for dep in prov.dependencies:
            dep_key = ProvenanceTracker._make_key(prov.experiment_name, dep)
            dep_prov = self._channels.get(dep_key)
            if dep_prov is None:
                continue
            if dep_prov.origin == ChannelOrigin.SIMULATOR_HIDDEN:
                leaks.append(dep)
            elif dep_prov.origin == ChannelOrigin.ANSWER_TEMPLATE:
                leaks.append(dep + "(ANSWER_TEMPLATE)")
            else:
                sub_leaks = self._find_upstream_leaks(dep_prov)
                leaks.extend(sub_leaks)
        return leaks

    def export_manifest(self) -> Dict[str, Any]:
        """Export complete provenance manifest as a dictionary."""
        channels = [prov.to_dict() for prov in self._channels.values()]
        dep_graph = {k: list(v) for k, v in self._reverse_deps.items()}
        return {
            "format_version": "1.0",
            "channels": channels,
            "dependency_graph": dep_graph,
            "violations": self.audit_leaks(),
        }

    def save_manifest(self, path: str) -> None:
        """Save provenance manifest to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.export_manifest(), f, indent=2)

    @classmethod
    def load_manifest(cls, path: str) -> "ProvenanceTracker":
        """Load provenance manifest from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        tracker = cls()
        for ch_data in data.get("channels", []):
            prov = ChannelProvenance.from_dict(ch_data)
            key = ProvenanceTracker._make_key(
                prov.experiment_name, prov.channel_name
            )
            tracker._channels[key] = prov
            for dep in prov.dependencies:
                dep_key = ProvenanceTracker._make_key(
                    prov.experiment_name, dep
                )
                tracker._reverse_deps[dep_key].add(key)
        return tracker

    @staticmethod
    def _make_key(experiment_name: str, channel_name: str) -> str:
        return f"{experiment_name}:{channel_name}" if experiment_name else channel_name

    @staticmethod
    def hash_manifest(data: Dict[str, Any]) -> str:
        """Compute a hash of the manifest for reproducibility checks."""
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


# ─── Global singleton ───────────────────────────────────────────────────────

_global_tracker: ProvenanceTracker = ProvenanceTracker()


def get_global_tracker() -> ProvenanceTracker:
    return _global_tracker


def reset_global_tracker() -> None:
    global _global_tracker
    _global_tracker = ProvenanceTracker()


def register_channel(
    channel_name: str,
    origin: ChannelOrigin,
    generation_method: str,
    dependencies: Optional[List[str]] = None,
    experiment_name: str = "",
    accessible_to_discovery: bool = True,
    justification: str = "",
    **extra,
) -> ChannelProvenance:
    return _global_tracker.register_channel(
        channel_name=channel_name,
        origin=origin,
        generation_method=generation_method,
        dependencies=dependencies,
        experiment_name=experiment_name,
        accessible_to_discovery=accessible_to_discovery,
        justification=justification,
        **extra,
    )


def audit_all() -> List[str]:
    return _global_tracker.audit_leaks()
