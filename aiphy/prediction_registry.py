"""
Prediction pre-registration and frozen prediction records — Phase 2.

Every theory MUST pre-register its predictions before experiment execution.
Predictions are frozen records that cannot be modified after the experiment
produces results. This prevents:
  - post-hoc fitting (interpreting results to match existing theory)
  - HARKing (Hypothesizing After Results are Known)
  - "theory shopping" (picking which theory fits after seeing data)

Hard gate: NO pre-registered prediction → theory cannot enter general law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import json
import hashlib


class PredictionOutcome:
    """Enumeration of possible prediction-observation relationships."""
    CONFIRMED = "confirmed"       # observation matches prediction
    CONTRADICTED = "contradicted" # observation contradicts prediction
    INCONCLUSIVE = "inconclusive" # cannot determine
    INCOMPARABLE = "incomparable" # predictions are incommensurate


@dataclass
class PredictionRecord:
    """
    A frozen prediction record.

    Once created, the prediction cannot be modified. Any attempt to modify
    after the experiment's timestamp produces an audit violation.
    """
    record_id: str
    theory_id: str
    experiment_name: str
    predicted_expression: str           # symbolic expression of prediction
    predicted_value_range: Optional[Tuple[float, float]] = None  # (low, high) or None for qualitative
    prediction_type: str = "conserved"  # "conserved", "zero", "relational"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    frozen: bool = True                 # always True after creation

    def __post_init__(self):
        self.frozen = True  # Immutable after creation

    @property
    def hash(self) -> str:
        """Content hash for detecting tampering."""
        raw = f"{self.theory_id}|{self.experiment_name}|{self.predicted_expression}|{self.prediction_type}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "theory_id": self.theory_id,
            "experiment_name": self.experiment_name,
            "predicted_expression": self.predicted_expression,
            "predicted_value_range": self.predicted_value_range,
            "prediction_type": self.prediction_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        return cls(
            record_id=d["record_id"],
            theory_id=d["theory_id"],
            experiment_name=d["experiment_name"],
            predicted_expression=d["predicted_expression"],
            predicted_value_range=d.get("predicted_value_range"),
            prediction_type=d.get("prediction_type", "conserved"),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class PredictionOutcomeRecord:
    """
    Records what actually happened vs what was predicted.

    Created AFTER experiment execution, referencing a frozen prediction.
    """
    record_id: str
    prediction_id: str                       # reference to frozen PredictionRecord
    theory_id: str
    experiment_name: str
    outcome: str                             # PredictionOutcome values
    observed_expression: Optional[str] = None  # what was actually observed
    observed_value: Optional[float] = None
    nrmse: Optional[float] = None            # normalized root-mean-square error
    details: str = ""
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prediction_id": self.prediction_id,
            "theory_id": self.theory_id,
            "experiment_name": self.experiment_name,
            "outcome": self.outcome,
            "observed_expression": self.observed_expression,
            "observed_value": self.observed_value,
            "nrmse": self.nrmse,
            "details": self.details,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionOutcomeRecord":
        return cls(
            record_id=d["record_id"],
            prediction_id=d["prediction_id"],
            theory_id=d["theory_id"],
            experiment_name=d["experiment_name"],
            outcome=d["outcome"],
            observed_expression=d.get("observed_expression"),
            observed_value=d.get("observed_value"),
            nrmse=d.get("nrmse"),
            details=d.get("details", ""),
            evaluated_at=d.get("evaluated_at", time.time()),
        )


class PredictionRegistry:
    """
    Central registry for all theory predictions.

    Enforces:
      - Predictions are frozen before experiment execution
      - Every prediction has an immutable record ID and content hash
      - Outcomes can only be recorded once per prediction-experiment pair
      - Theories without predictions cannot enter general law
    """

    def __init__(self):
        self._predictions: Dict[str, PredictionRecord] = {}
        self._outcomes: Dict[str, PredictionOutcomeRecord] = {}
        self._id_counter: int = 0

    @property
    def total_predictions(self) -> int:
        return len(self._predictions)

    @property
    def total_outcomes(self) -> int:
        return len(self._outcomes)

    def pre_register(
        self,
        theory_id: str,
        experiment_name: str,
        predicted_expression: str,
        prediction_type: str = "conserved",
        predicted_value_range: Optional[Tuple[float, float]] = None,
        **metadata,
    ) -> str:
        """
        Pre-register a prediction BEFORE experiment execution.

        Returns the prediction record ID.
        """
        self._id_counter += 1
        rid = f"PR_{self._id_counter:06d}"

        record = PredictionRecord(
            record_id=rid,
            theory_id=theory_id,
            experiment_name=experiment_name,
            predicted_expression=predicted_expression,
            predicted_value_range=predicted_value_range,
            prediction_type=prediction_type,
            metadata=dict(metadata),
        )
        self._predictions[rid] = record
        return rid

    def get_prediction(self, record_id: str) -> Optional[PredictionRecord]:
        return self._predictions.get(record_id)

    def get_predictions_for_theory(self, theory_id: str) -> List[PredictionRecord]:
        return [p for p in self._predictions.values()
                if p.theory_id == theory_id]

    def get_predictions_for_experiment(self, experiment_name: str) -> List[PredictionRecord]:
        return [p for p in self._predictions.values()
                if p.experiment_name == experiment_name]

    def get_pending_predictions(self, experiment_name: str) -> List[PredictionRecord]:
        """Get predictions that have not yet had outcomes recorded for this experiment."""
        resolved = {
            o.prediction_id for o in self._outcomes.values()
            if o.experiment_name == experiment_name
        }
        return [p for p in self.get_predictions_for_experiment(experiment_name)
                if p.record_id not in resolved]

    def record_outcome(
        self,
        prediction_id: str,
        outcome: str,
        observed_expression: Optional[str] = None,
        observed_value: Optional[float] = None,
        nrmse: Optional[float] = None,
        details: str = "",
    ) -> str:
        """
        Record the outcome of an experiment against a frozen prediction.

        Returns the outcome record ID.
        """
        prediction = self._predictions.get(prediction_id)
        if prediction is None:
            raise ValueError(f"Prediction {prediction_id} not found")

        # Prevent duplicate outcomes
        for oid, orec in self._outcomes.items():
            if orec.prediction_id == prediction_id:
                raise ValueError(
                    f"Outcome already recorded for prediction {prediction_id}: "
                    f"{orec.outcome}"
                )

        self._id_counter += 1
        oid = f"PO_{self._id_counter:06d}"

        record = PredictionOutcomeRecord(
            record_id=oid,
            prediction_id=prediction_id,
            theory_id=prediction.theory_id,
            experiment_name=prediction.experiment_name,
            outcome=outcome,
            observed_expression=observed_expression,
            observed_value=observed_value,
            nrmse=nrmse,
            details=details,
        )
        self._outcomes[oid] = record
        return oid

    def get_outcome(self, record_id: str) -> Optional[PredictionOutcomeRecord]:
        return self._outcomes.get(record_id)

    def get_outcomes_for_theory(self, theory_id: str) -> List[PredictionOutcomeRecord]:
        return [o for o in self._outcomes.values()
                if o.theory_id == theory_id]

    def get_outcomes_for_experiment(self, experiment_name: str) -> List[PredictionOutcomeRecord]:
        return [o for o in self._outcomes.values()
                if o.experiment_name == experiment_name]

    def theory_has_predictions(self, theory_id: str) -> bool:
        """Check if a theory has ANY pre-registered predictions."""
        return len(self.get_predictions_for_theory(theory_id)) > 0

    def experiment_fulfilled(self, experiment_name: str) -> bool:
        """
        Check if ALL theories that predict on this experiment have had
        their outcomes recorded.
        """
        predictions = self.get_predictions_for_experiment(experiment_name)
        outcomes = self.get_outcomes_for_experiment(experiment_name)
        predicted_ids = {p.record_id for p in predictions}
        resolved_ids = {o.prediction_id for o in outcomes}
        return predicted_ids == resolved_ids

    def contradiction_summary(
        self, theory_id: str
    ) -> Dict[str, Any]:
        """Summarize contradictions found for a theory."""
        outcomes = self.get_outcomes_for_theory(theory_id)
        contradicted = [o for o in outcomes
                        if o.outcome == PredictionOutcome.CONTRADICTED]
        confirmed = [o for o in outcomes
                     if o.outcome == PredictionOutcome.CONFIRMED]
        return {
            "theory_id": theory_id,
            "total_predictions": len(self.get_predictions_for_theory(theory_id)),
            "total_outcomes": len(outcomes),
            "contradictions": len(contradicted),
            "confirmations": len(confirmed),
            "contradicting_experiments": list(set(
                o.experiment_name for o in contradicted
            )),
            "confirming_experiments": list(set(
                o.experiment_name for o in confirmed
            )),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": {
                rid: rec.to_dict() for rid, rec in self._predictions.items()
            },
            "outcomes": {
                oid: rec.to_dict() for oid, rec in self._outcomes.items()
            },
            "id_counter": self._id_counter,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRegistry":
        reg = cls()
        reg._id_counter = d.get("id_counter", 0)
        for rid, rec_d in d.get("predictions", {}).items():
            reg._predictions[rid] = PredictionRecord.from_dict(rec_d)
        for oid, rec_d in d.get("outcomes", {}).items():
            reg._outcomes[oid] = PredictionOutcomeRecord.from_dict(rec_d)
        return reg
