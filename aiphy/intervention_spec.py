"""
Intervention specification — Phase 2.

Formal specification for designing discriminating experiments.
An InterventionSpec encodes:
  - Which observable/parameter to manipulate
  - What value(s) to set it to
  - What to measure
  - How it discriminates between competing theories

Integrates with:
  - DiscriminatingExperimentSelector (decides what to test)
  - FalsificationJudge (evaluates results)
  - MeasurementContext (observation_protocol.py)
  - ContextualBandit (rewards for good interventions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import json


class InterventionType(Enum):
    SET_PARAMETER = auto()      # Change a parameter value
    SWAP_OBJECT = auto()        # Replace one object with another
    CHANGE_ENVIRONMENT = auto() # Change environmental condition
    PROBE_STATE = auto()        # Measure without changing parameters
    EXPLORE_DOMAIN = auto()     # Explore a new region of domain


@dataclass
class InterventionSpec:
    """
    Formal specification of a discriminating intervention experiment.

    This is the bridge between the theory-selection logic and the
    experiment execution layer.
    """
    # --- Identity ---
    intervention_id: str
    target_experiment: str               # which experiment to modify
    intervention_type: InterventionType

    # --- What to change ---
    modified_parameter: str               # e.g., "mass", "initial_velocity"
    set_value: float                      # what value to set
    original_value: Optional[float] = None  # for audit trail

    # --- What to measure ---
    observed_quantity: str = ""           # e.g., "final_position", "energy"
    measurement_type: str = "conserved"   # "conserved", "zero", "relational"

    # --- Discrimination ---
    discriminates_theories: List[str] = field(default_factory=list)  # theory IDs
    expected_prediction_a: Optional[str] = None   # theory A's prediction
    expected_prediction_b: Optional[str] = None   # theory B's prediction
    discrimination_rationale: str = ""             # why this distinguishes them

    # --- Constraints ---
    validity_range: Optional[Tuple[float, float]] = None  # valid parameter range
    requires_pre_registration: bool = True
    priority: float = 0.5                # 0-1, higher = more important
    min_measurements: int = 3
    max_measurements: int = 100

    # --- Metadata ---
    created_at: float = 0.0
    designed_by: str = "discriminating_selector"
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "target_experiment": self.target_experiment,
            "intervention_type": self.intervention_type.name,
            "modified_parameter": self.modified_parameter,
            "set_value": self.set_value,
            "original_value": self.original_value,
            "observed_quantity": self.observed_quantity,
            "measurement_type": self.measurement_type,
            "discriminates_theories": self.discriminates_theories,
            "expected_prediction_a": self.expected_prediction_a,
            "expected_prediction_b": self.expected_prediction_b,
            "discrimination_rationale": self.discrimination_rationale,
            "validity_range": list(self.validity_range) if self.validity_range else None,
            "requires_pre_registration": self.requires_pre_registration,
            "priority": self.priority,
            "min_measurements": self.min_measurements,
            "max_measurements": self.max_measurements,
            "created_at": self.created_at,
            "designed_by": self.designed_by,
            "tags": self.tags,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterventionSpec":
        vr = d.get("validity_range")
        return cls(
            intervention_id=d["intervention_id"],
            target_experiment=d["target_experiment"],
            intervention_type=InterventionType[d["intervention_type"]],
            modified_parameter=d["modified_parameter"],
            set_value=d["set_value"],
            original_value=d.get("original_value"),
            observed_quantity=d.get("observed_quantity", ""),
            measurement_type=d.get("measurement_type", "conserved"),
            discriminates_theories=d.get("discriminates_theories", []),
            expected_prediction_a=d.get("expected_prediction_a"),
            expected_prediction_b=d.get("expected_prediction_b"),
            discrimination_rationale=d.get("discrimination_rationale", ""),
            validity_range=tuple(vr) if vr else None,
            requires_pre_registration=d.get("requires_pre_registration", True),
            priority=d.get("priority", 0.5),
            min_measurements=d.get("min_measurements", 3),
            max_measurements=d.get("max_measurements", 100),
            created_at=d.get("created_at", 0.0),
            designed_by=d.get("designed_by", "discriminating_selector"),
            tags=d.get("tags", []),
            extra=d.get("extra", {}),
        )


class InterventionDesigner:
    """
    Designs intervention specifications from theory competition analysis.

    Takes the output of the discriminating selector and the falsification
    judge to produce executable intervention specs.
    """

    def __init__(
        self,
        *,
        default_min_measurements: int = 3,
        default_max_measurements: int = 50,
        id_prefix: str = "INTRV",
    ):
        self.default_min_measurements = default_min_measurements
        self.default_max_measurements = default_max_measurements
        self.id_prefix = id_prefix
        self._id_counter = 0

    def design_from_competition(
        self,
        theory_id_a: str,
        theory_id_b: str,
        discriminating_experiment: str,
        prediction_a: Optional[str] = None,
        prediction_b: Optional[str] = None,
        *,
        parameter_to_modify: str = "parameter",
        set_value: float = 0.0,
        intervention_type: InterventionType = InterventionType.PROBE_STATE,
        priority: float = 0.8,
    ) -> InterventionSpec:
        """Design an intervention to discriminate between two competing theories."""
        self._id_counter += 1
        iid = f"{self.id_prefix}_{self._id_counter:04d}"

        import time
        return InterventionSpec(
            intervention_id=iid,
            target_experiment=discriminating_experiment,
            intervention_type=intervention_type,
            modified_parameter=parameter_to_modify,
            set_value=set_value,
            discriminates_theories=[theory_id_a, theory_id_b],
            expected_prediction_a=prediction_a,
            expected_prediction_b=prediction_b,
            discrimination_rationale=(
                f"Theories {theory_id_a} and {theory_id_b} make different "
                f"predictions for experiment {discriminating_experiment}"
            ),
            min_measurements=self.default_min_measurements,
            max_measurements=self.default_max_measurements,
            requires_pre_registration=True,
            priority=priority,
            created_at=time.time(),
        )

    def design_parameter_sweep(
        self,
        target_experiment: str,
        parameter_name: str,
        start_value: float,
        end_value: float,
        num_steps: int,
        *,
        theories_to_test: Optional[List[str]] = None,
    ) -> List[InterventionSpec]:
        """Design a parameter sweep to explore a domain region."""
        specs = []
        step_size = (end_value - start_value) / max(num_steps - 1, 1)

        for step in range(num_steps):
            value = start_value + step * step_size
            self._id_counter += 1
            iid = f"{self.id_prefix}_SWEEP_{self._id_counter:04d}"

            import time
            spec = InterventionSpec(
                intervention_id=iid,
                target_experiment=target_experiment,
                intervention_type=InterventionType.SET_PARAMETER,
                modified_parameter=parameter_name,
                set_value=value,
                discriminates_theories=theories_to_test or [],
                validity_range=(start_value, end_value),
                min_measurements=1,
                max_measurements=10,
                priority=0.3,  # Lower priority for sweeps vs. discriminating
                created_at=time.time(),
                tags=["parameter_sweep", parameter_name],
            )
            specs.append(spec)
        return specs

    def design_exploration(
        self,
        target_experiment: str,
        parameter_name: str,
        random_range: Tuple[float, float],
        num_exploratory: int = 5,
    ) -> List[InterventionSpec]:
        """Design exploratory interventions in a new domain region."""
        import random
        import time

        specs = []
        for _ in range(num_exploratory):
            value = random.uniform(*random_range)
            self._id_counter += 1
            iid = f"{self.id_prefix}_EXPL_{self._id_counter:04d}"

            spec = InterventionSpec(
                intervention_id=iid,
                target_experiment=target_experiment,
                intervention_type=InterventionType.EXPLORE_DOMAIN,
                modified_parameter=parameter_name,
                set_value=value,
                validity_range=random_range,
                min_measurements=1,
                max_measurements=20,
                priority=0.15,  # Low priority for exploration
                created_at=time.time(),
                tags=["exploration", parameter_name],
            )
            specs.append(spec)
        return specs

    def to_measurement_context(
        self, spec: InterventionSpec
    ) -> Dict[str, Any]:
        """
        Convert an InterventionSpec to a measurement context dict
        compatible with MeasurementContext in observation_protocol.py.
        """
        return {
            "apparatus_name": spec.target_experiment,
            "parameter_values": {spec.modified_parameter: spec.set_value},
            "intervention_spec": spec.intervention_id,
            "noise_model": "gaussian",
            "noise_std": 0.01,
            "extra": {
                "intervention_type": spec.intervention_type.name,
                "discriminates": spec.discriminates_theories,
                "priority": spec.priority,
                "validity_range": list(spec.validity_range) if spec.validity_range else None,
            },
        }
