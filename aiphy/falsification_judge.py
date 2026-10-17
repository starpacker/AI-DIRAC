"""
Falsification judge and discriminating intervention designer — Phase 2.

Evaluates experiment results against pre-registered predictions to:
  1. Judge which theories are confirmed/falsified/inconclusive
  2. Compute information reward for falsification (POSITIVE reward)
  3. Detect equivalent theories (same predictions in ALL observations)
  4. Design discriminating interventions to distinguish competing theories

Key principle: falsification earns POSITIVE reward because we learned
something. A theory that survives 100 confirmations and 0 falsifications
earns more cumulative reward than one that survives nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import math

from .prediction_registry import (
    PredictionRegistry, PredictionRecord, PredictionOutcomeRecord,
    PredictionOutcome,
)
from .theory_population import (
    TheoryPopulation, TheoryRecord, TheoryStatus,
)


@dataclass
class FalsificationResult:
    """Result of comparing a prediction to experiment outcome."""
    theory_id: str
    prediction_id: str
    experiment_name: str
    outcome: str                     # PredictionOutcome value
    information_reward: float = 0.0  # positive for learning
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class FalsificationJudge:
    """
    Evaluates experiment results against pre-registered predictions.

    Decision logic:
      - confirmed: observation matches prediction within tolerance
      - contradicted: observation deviates beyond tolerance
      - inconclusive: noise or insufficient data
      - incomparable: predictions cannot be compared (different types/domains)
    """

    def __init__(
        self,
        theory_population: TheoryPopulation,
        prediction_registry: "PredictionRegistry",
        *,
        conservation_tolerance: float = 0.05,
        zero_tolerance: float = 0.01,
        contradiction_reward: float = 4.0,
        confirmation_reward: float = 1.0,
        inconclusive_reward: float = 0.1,
    ):
        self.population = theory_population
        self.registry = prediction_registry
        self.conservation_tolerance = conservation_tolerance
        self.zero_tolerance = zero_tolerance
        self.contradiction_reward = contradiction_reward
        self.confirmation_reward = confirmation_reward
        self.inconclusive_reward = inconclusive_reward

    def judge_experiment(
        self,
        experiment_name: str,
        observed_data: Dict[str, Any],
    ) -> List[FalsificationResult]:
        """
        Evaluate all pre-registered predictions against experiment results.

        Args:
            experiment_name: Name of the experiment
            observed_data: dict with keys 'is_conserved', 'is_zero', 'mean', 'std'

        Returns:
            List of FalsificationResult, one per active prediction
        """
        results = []
        predictions = self.registry.get_predictions_for_experiment(experiment_name)

        for pred in predictions:
            theory = self.population.get_theory(pred.theory_id)
            if theory is None or theory.status != TheoryStatus.ACTIVE:
                continue

            result = self._judge_single(pred, observed_data)
            results.append(result)

            # Record outcome in registry
            try:
                self.registry.record_outcome(
                    prediction_id=pred.record_id,
                    outcome=result.outcome,
                    observed_expression=str(observed_data.get("expression", "")),
                    observed_value=observed_data.get("mean"),
                    nrmse=observed_data.get("nrmse"),
                    details=result.reason,
                )
            except ValueError:
                pass  # Outcome already recorded

        return results

    def _judge_single(
        self,
        prediction: PredictionRecord,
        observed: Dict[str, Any],
    ) -> FalsificationResult:
        """Judge a single prediction against observed data."""
        pred_type = prediction.prediction_type

        if pred_type == "conserved":
            return self._judge_conserved(prediction, observed)
        elif pred_type == "zero":
            return self._judge_zero(prediction, observed)
        elif pred_type == "relational":
            return self._judge_relational(prediction, observed)
        else:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.INCOMPARABLE,
                reason=f"Unknown prediction type: {pred_type}",
            )

    def _judge_conserved(
        self,
        prediction: PredictionRecord,
        observed: Dict[str, Any],
    ) -> FalsificationResult:
        """Judge a conservation prediction."""
        is_conserved = observed.get("is_conserved", False)
        coeff_var = observed.get("coefficient_of_variation", None)  # std/mean

        if is_conserved:
            reward = self.confirmation_reward
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONFIRMED,
                information_reward=reward,
                reason=f"Conserved: cv={coeff_var:.4f}" if coeff_var else "Conserved",
            )
        elif coeff_var is not None and coeff_var > self.conservation_tolerance:
            # POSITIVE reward for falsification — we learned something
            reward = self.contradiction_reward * min(math.log1p(coeff_var), 5.0)
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONTRADICTED,
                information_reward=reward,
                reason=f"VARIATION: cv={coeff_var:.4f} > {self.conservation_tolerance}",
            )
        else:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.INCONCLUSIVE,
                information_reward=self.inconclusive_reward,
                reason="Borderline — within noise threshold",
            )

    def _judge_zero(
        self,
        prediction: PredictionRecord,
        observed: Dict[str, Any],
    ) -> FalsificationResult:
        """Judge a zero-valued prediction."""
        is_zero = observed.get("is_zero", False)
        mean_val = observed.get("mean", 0.0)
        std_val = observed.get("std", 0.0)

        if is_zero:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONFIRMED,
                information_reward=self.confirmation_reward,
                reason="Expression is zero",
            )
        elif std_val > 0 and abs(mean_val) / max(std_val, 1e-12) > 3.0:
            # Statistically significantly non-zero
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONTRADICTED,
                information_reward=self.contradiction_reward,
                reason=f"Non-zero: mean={mean_val:.4f}, std={std_val:.4f}",
            )
        else:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.INCONCLUSIVE,
                information_reward=self.inconclusive_reward,
                reason=f"Near zero within noise: mean={mean_val:.4f}, std={std_val:.4f}",
            )

    def _judge_relational(
        self,
        prediction: PredictionRecord,
        observed: Dict[str, Any],
    ) -> FalsificationResult:
        """Judge a relational prediction (e.g., PV = nRT)."""
        nrmse = observed.get("nrmse", 1.0)

        if nrmse < self.zero_tolerance:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONFIRMED,
                information_reward=self.confirmation_reward,
                reason=f"Relation holds: NRMSE={nrmse:.6f}",
            )
        elif nrmse > self.conservation_tolerance:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.CONTRADICTED,
                information_reward=self.contradiction_reward,
                reason=f"Relation violated: NRMSE={nrmse:.4f}",
            )
        else:
            return FalsificationResult(
                theory_id=prediction.theory_id,
                prediction_id=prediction.record_id,
                experiment_name=prediction.experiment_name,
                outcome=PredictionOutcome.INCONCLUSIVE,
                information_reward=self.inconclusive_reward,
                reason=f"Borderline: NRMSE={nrmse:.4f}",
            )

    def detect_equivalent_theories(self) -> List[Tuple[str, str]]:
        """
        Detect pairs of active theories that make identical predictions
        across ALL their valid experiments.

        Returns list of (theory_id_a, theory_id_b) pairs.
        """
        actives = self.population.active_theories
        active_ids = list(actives.keys())
        equivalents = []

        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                tid_a, tid_b = active_ids[i], active_ids[j]
                if self._are_equivalent(tid_a, tid_b):
                    equivalents.append((tid_a, tid_b))
                    self.population.mark_equivalent(tid_a, tid_b)

        return equivalents

    def _are_equivalent(
        self, theory_id_a: str, theory_id_b: str
    ) -> bool:
        """Check if two theories make identical predictions everywhere."""
        preds_a = self.registry.get_predictions_for_theory(theory_id_a)
        preds_b = self.registry.get_predictions_for_theory(theory_id_b)

        # Must predict on the same set of experiments
        exps_a = {p.experiment_name for p in preds_a}
        exps_b = {p.experiment_name for p in preds_b}

        if exps_a != exps_b:
            return False

        # For each experiment, predictions must be identical in form
        for exp_name in exps_a:
            pa = [p for p in preds_a if p.experiment_name == exp_name]
            pb = [p for p in preds_b if p.experiment_name == exp_name]

            if len(pa) != len(pb):
                return False

            for a, b in zip(sorted(pa, key=lambda x: x.predicted_expression),
                           sorted(pb, key=lambda x: x.predicted_expression)):
                if (a.prediction_type != b.prediction_type or
                        a.predicted_expression != b.predicted_expression):
                    return False

        return True

    def design_discriminating_intervention(
        self,
        theory_id_a: str,
        theory_id_b: str,
        available_experiments: List[str],
    ) -> Optional[str]:
        """
        Find an experiment where two competing theories make DIFFERENT
        predictions. This is the active experiment design step.

        Returns the best discriminating experiment name, or None.
        """
        best_experiment = None
        best_divergence = 0.0

        preds_a = {p.experiment_name: p
                   for p in self.registry.get_predictions_for_theory(theory_id_a)}
        preds_b = {p.experiment_name: p
                   for p in self.registry.get_predictions_for_theory(theory_id_b)}

        for exp_name in available_experiments:
            pa = preds_a.get(exp_name)
            pb = preds_b.get(exp_name)

            if pa is None or pb is None:
                # One theory has no prediction here — this is discriminating!
                divergence = 1.0
            elif pa.prediction_type != pb.prediction_type:
                divergence = 0.8
            elif pa.predicted_expression != pb.predicted_expression:
                divergence = 0.5
            else:
                continue  # Same prediction, not discriminating

            if divergence > best_divergence:
                best_divergence = divergence
                best_experiment = exp_name

        return best_experiment

    def apply_results_to_population(
        self,
        results: List[FalsificationResult],
    ) -> Dict[str, int]:
        """
        Apply falsification results to the theory population.

        Falsified theories are marked (not deleted).
        Confirmed theories remain active.
        Returns counts of status changes.
        """
        counts = {"falsified": 0, "confirmed": 0, "restricted": 0, "unchanged": 0}

        for result in results:
            if result.outcome == PredictionOutcome.CONTRADICTED:
                self.population.falsify(
                    result.theory_id,
                    result.experiment_name,
                    reason=result.reason,
                )
                counts["falsified"] += 1
            elif result.outcome == PredictionOutcome.CONFIRMED:
                theory = self.population.get_theory(result.theory_id)
                if theory and result.experiment_name not in theory.valid_experiments:
                    theory.valid_experiments.append(result.experiment_name)
                counts["confirmed"] += 1
            else:
                counts["unchanged"] += 1

        return counts

    def compute_law_qualification(
        self, theory_id: str
    ) -> Dict[str, Any]:
        """
        Determine if a theory qualifies for general law status.

        Requirements:
          1. Has at least 1 pre-registered prediction
          2. Has been tested on at least 2 independent experiments
          3. Has at least 1 confirmation
          4. Has no unresolved contradictions

        Returns a qualification report.
        """
        predictions = self.registry.get_predictions_for_theory(theory_id)
        outcomes = self.registry.get_outcomes_for_theory(theory_id)

        tested_experiments = list(set(o.experiment_name for o in outcomes))
        contradictions = [o for o in outcomes
                          if o.outcome == PredictionOutcome.CONTRADICTED]
        confirmations = [o for o in outcomes
                          if o.outcome == PredictionOutcome.CONFIRMED]

        has_predictions = len(predictions) > 0
        multi_experiment = len(tested_experiments) >= 2
        has_confirmations = len(confirmations) >= 1
        no_contradictions = len(contradictions) == 0

        qualified = all([
            has_predictions,
            multi_experiment,
            has_confirmations,
            no_contradictions,
        ])

        return {
            "theory_id": theory_id,
            "qualified": qualified,
            "has_predictions": has_predictions,
            "multi_experiment": multi_experiment,
            "tested_experiments": tested_experiments,
            "has_confirmations": has_confirmations,
            "confirmation_count": len(confirmations),
            "no_contradictions": no_contradictions,
            "contradiction_count": len(contradictions),
            "reason": _qualification_reason(
                has_predictions, multi_experiment,
                has_confirmations, no_contradictions,
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conservation_tolerance": self.conservation_tolerance,
            "zero_tolerance": self.zero_tolerance,
            "contradiction_reward": self.contradiction_reward,
            "confirmation_reward": self.confirmation_reward,
            "inconclusive_reward": self.inconclusive_reward,
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        theory_population: TheoryPopulation,
        prediction_registry: "PredictionRegistry",
    ) -> "FalsificationJudge":
        return cls(
            theory_population=theory_population,
            prediction_registry=prediction_registry,
            conservation_tolerance=d.get("conservation_tolerance", 0.05),
            zero_tolerance=d.get("zero_tolerance", 0.01),
            contradiction_reward=d.get("contradiction_reward", 4.0),
            confirmation_reward=d.get("confirmation_reward", 1.0),
            inconclusive_reward=d.get("inconclusive_reward", 0.1),
        )


def _qualification_reason(
    has_predictions: bool,
    multi_experiment: bool,
    has_confirmations: bool,
    no_contradictions: bool,
) -> str:
    if not has_predictions:
        return "No pre-registered predictions — cannot qualify"
    if not multi_experiment:
        return "Only tested on one experiment — need ≥2"
    if not has_confirmations:
        return "No confirmation on any experiment"
    if not no_contradictions:
        return "Has unresolved contradictions"
    return "Qualified for general law"
