"""
Discriminating experiment selector — Phase 2.

Selects experiments that best distinguish competing theories.
Integrates with the existing contextual bandit by adding a
theory-discrimination term to the experiment value function.

The selector:
  1. Identifies competing active theory pairs
  2. Finds experiments where their predictions differ most
  3. Prioritizes experiments that maximize information gain
  4. Ensures a minimum exploration rate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import random
from dataclasses import dataclass, field

from .theory_population import TheoryPopulation, TheoryStatus
from .prediction_registry import PredictionRegistry
from .falsification_judge import FalsificationJudge


@dataclass
class DiscriminatingExperimentScore:
    """Score for how well an experiment discriminates between theories."""
    experiment_name: str
    discrimination_value: float          # 0-1, how well it separates theories
    information_gain: float              # expected information gain
    completeness_score: float            # fraction of theories with predictions
    total_score: float = 0.0
    competing_pairs: List[Tuple[str, str]] = field(default_factory=list)
    details: str = ""

    def __post_init__(self):
        self.total_score = (
            self.discrimination_value * 0.5 +
            self.information_gain * 0.3 +
            self.completeness_score * 0.2
        )


class DiscriminatingExperimentSelector:
    """
    Selects experiments that best discriminate between competing theories.

    Weighted combination of:
      - discrimination_value: how differently theories predict
      - information_gain: expected learning from resolution
      - completeness: fraction of active theories making predictions
    """

    def __init__(
        self,
        theory_population: TheoryPopulation,
        prediction_registry: PredictionRegistry,
        falsification_judge: FalsificationJudge,
        *,
        exploration_rate: float = 0.15,
        minimum_discrimination: float = 0.01,
    ):
        self.population = theory_population
        self.registry = prediction_registry
        self.judge = falsification_judge
        self.exploration_rate = exploration_rate
        self.minimum_discrimination = minimum_discrimination

    def score_experiment(
        self, experiment_name: str
    ) -> DiscriminatingExperimentScore:
        """
        Score how well an experiment discriminates between active theories.

        High score = theories make DIFFERENT predictions here.
        """
        actives = self.population.active_theories
        active_ids = list(actives.keys())

        if len(active_ids) < 2:
            # Cannot discriminate with <2 theories
            return DiscriminatingExperimentScore(
                experiment_name=experiment_name,
                discrimination_value=0.0,
                information_gain=0.0,
                completeness_score=0.0,
                details="Need ≥2 active theories to discriminate",
            )

        # Find pairs with different predictions on this experiment
        predictions = self.registry.get_predictions_for_experiment(experiment_name)
        pred_map: Dict[str, str] = {}
        for p in predictions:
            if p.theory_id in active_ids:
                pred_map[p.theory_id] = p.predicted_expression

        # Count theories that have predictions
        predicted_count = len(pred_map)
        completeness = predicted_count / max(len(active_ids), 1)

        # Find competing pairs with different predictions
        competing = []
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                tid_a, tid_b = active_ids[i], active_ids[j]
                pa = pred_map.get(tid_a)
                pb = pred_map.get(tid_b)

                if pa is not None and pb is not None and pa != pb:
                    competing.append((tid_a, tid_b))
                elif pa is None or pb is None:
                    # One theory has no prediction — this gap is discriminating
                    competing.append((tid_a, tid_b))

        # Discrimination: fraction of pairs that differ
        total_pairs = len(active_ids) * (len(active_ids) - 1) // 2
        disc_value = len(competing) / max(total_pairs, 1)

        # Information gain: higher when many theories are undecided
        pending_count = len(self.registry.get_pending_predictions(experiment_name))
        info_gain = min(pending_count / max(len(active_ids), 1), 1.0)

        details = (
            f"{len(competing)} competing pairs, "
            f"{predicted_count}/{len(active_ids)} theories predict, "
            f"{pending_count} pending outcomes"
        )

        return DiscriminatingExperimentScore(
            experiment_name=experiment_name,
            discrimination_value=disc_value,
            information_gain=info_gain,
            completeness_score=completeness,
            competing_pairs=competing,
            details=details,
        )

    def select_experiment(
        self,
        available_experiments: List[str],
        current_experiment: Optional[str] = None,
    ) -> str:
        """
        Select the best discriminating experiment.

        If no discrimination is possible (all theories predict the same),
        falls back to random exploration.
        """
        if not available_experiments:
            raise ValueError("No experiments available")

        # Exploration: random choice
        if random.random() < self.exploration_rate:
            return random.choice(available_experiments)

        # Score all experiments
        scores: List[DiscriminatingExperimentScore] = []
        for exp_name in available_experiments:
            if exp_name == current_experiment:
                continue  # Don't repeat the same experiment
            score = self.score_experiment(exp_name)
            scores.append(score)

        if not scores:
            return random.choice(available_experiments)

        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)

        best = scores[0]
        if best.total_score < self.minimum_discrimination:
            # No discriminating experiment — explore randomly
            return random.choice(available_experiments)

        return best.experiment_name

    def select_top_discriminating(
        self,
        available_experiments: List[str],
        top_k: int = 3,
    ) -> List[Tuple[str, DiscriminatingExperimentScore]]:
        """Return top-k discriminating experiments with scores."""
        scores = [self.score_experiment(exp) for exp in available_experiments]
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return [(s.experiment_name, s) for s in scores[:top_k]]

    def report_competition_status(self) -> Dict[str, Any]:
        """Report the current state of theory competition."""
        actives = self.population.active_theories
        equivalents = self.judge.detect_equivalent_theories()

        return {
            "active_theories": len(actives),
            "competing_pairs": len(actives) * (len(actives) - 1) // 2,
            "equivalent_pairs": len(equivalents),
            "equivalent_pairs_detail": [
                {
                    "theory_a": tid_a,
                    "theory_b": tid_b,
                    "status": "equivalent — same predictions everywhere",
                }
                for tid_a, tid_b in equivalents
            ],
            "has_competition": len(actives) >= 2 and len(equivalents) < len(actives) * (len(actives) - 1) // 2,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_rate": self.exploration_rate,
            "minimum_discrimination": self.minimum_discrimination,
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        theory_population: TheoryPopulation,
        prediction_registry: PredictionRegistry,
        falsification_judge: FalsificationJudge,
    ) -> "DiscriminatingExperimentSelector":
        return cls(
            theory_population=theory_population,
            prediction_registry=prediction_registry,
            falsification_judge=falsification_judge,
            exploration_rate=d.get("exploration_rate", 0.15),
            minimum_discrimination=d.get("minimum_discrimination", 0.01),
        )
