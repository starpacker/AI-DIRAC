"""
Active Experiment Designer — Phase 4.

Composite experiment value function serving both:
  1. Physical theory falsification (Phase 2)
  2. Mathematical language uncertainty reduction (Phase 3)

Value function:
  value = theory_discrimination
        + math_structure_information_gain
        + falsification_potential
        + transfer_value
        - compute_cost
        - apparatus_cost
        - repetition_penalty

Provides comparison baselines: random, UCB, law-only, contextual,
fixed-language, growing-language.

Hard gate: experiment selection MUST NOT read simulator hidden truth
          or directly read target theory labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import random
import time

import numpy as np


# ---------------------------------------------------------------------------
# Value components
# ---------------------------------------------------------------------------

@dataclass
class ExperimentValueComponents:
    """Breakdown of experiment value for audit trail."""
    experiment_name: str
    theory_discrimination: float = 0.0        # how much theories differ here
    math_info_gain: float = 0.0              # new math structure info
    falsification_potential: float = 0.0     # chance of falsifying a theory
    transfer_value: float = 0.0             # cross-domain (apparatus/domain) value
    diversity_bonus: float = 0.0            # bonus for experimentation diversity
    compute_cost: float = 0.0               # estimated compute cost
    apparatus_cost: float = 0.0             # estimated apparatus cost
    repetition_penalty: float = 0.0          # penalty for repeating similar exp
    total_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "theory_discrimination": self.theory_discrimination,
            "math_info_gain": self.math_info_gain,
            "falsification_potential": self.falsification_potential,
            "transfer_value": self.transfer_value,
            "diversity_bonus": self.diversity_bonus,
            "compute_cost": self.compute_cost,
            "apparatus_cost": self.apparatus_cost,
            "repetition_penalty": self.repetition_penalty,
            "total_value": self.total_value,
        }


# ---------------------------------------------------------------------------
# Experiment types for diversity tracking
# ---------------------------------------------------------------------------

class ExperimentCategory(Enum):
    REPEAT = auto()           # same experiment again
    NUISANCE = auto()         # same experiment with nuisance variation
    REPLACEMENT = auto()      # different object/reagent/setup
    CROSS_APPARATUS = auto()  # different measurement device
    HELDOUT_VALIDATION = auto()  # held-out data, no new intervention
    EXPLORATION = auto()      # random exploration
    DISCRIMINATING = auto()   # designed to distinguish theories
    MATH_UNCERTAINTY = auto() # designed to reduce math representation uncertainty


@dataclass
class ExperimentDiversityTracker:
    """Tracks experiment history to penalize repetitive experiments."""
    experiment_usage: Dict[str, int] = field(default_factory=dict)
    category_usage: Dict[str, int] = field(default_factory=dict)
    apparatus_usage: Dict[str, int] = field(default_factory=dict)
    total_experiments: int = 0

    def record(self, exp_name: str, category: str, apparatus: str = "") -> None:
        self.experiment_usage[exp_name] = self.experiment_usage.get(exp_name, 0) + 1
        self.category_usage[category] = self.category_usage.get(category, 0) + 1
        if apparatus:
            self.apparatus_usage[apparatus] = self.apparatus_usage.get(apparatus, 0) + 1
        self.total_experiments += 1

    def repetition_penalty(self, exp_name: str, max_penalty: float = 0.3) -> float:
        """Penalty increases with repetition count."""
        count = self.experiment_usage.get(exp_name, 0)
        if count == 0:
            return 0.0
        return min(count * 0.1, max_penalty)

    def category_diversity_bonus(self, category: str) -> float:
        """Bonus for exploring under-sampled categories."""
        total = max(self.total_experiments, 1)
        cat_count = self.category_usage.get(category, 0)
        ratio = cat_count / total
        # Bonus is higher when category is under-explored
        return max(0.0, 0.1 - ratio) * 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_usage": dict(self.experiment_usage),
            "category_usage": dict(self.category_usage),
            "apparatus_usage": dict(self.apparatus_usage),
            "total_experiments": self.total_experiments,
        }


# ---------------------------------------------------------------------------
# Baseline selectors
# ---------------------------------------------------------------------------

class RandomSelector:
    """Pure random experiment selection — worst-case baseline."""
    def select(self, experiments: List[str]) -> str:
        return random.choice(experiments)


class UCBSelector:
    """Upper Confidence Bound experiment selection — legacy baseline."""
    def __init__(self, exploration_weight: float = 2.0):
        self.exploration_weight = exploration_weight
        self._counts: Dict[str, int] = {}
        self._values: Dict[str, float] = {}

    def select(self, experiments: List[str]) -> str:
        # Ensure all experiments are initialized
        for e in experiments:
            if e not in self._counts:
                self._counts[e] = 0
                self._values[e] = 0.0

        total = sum(self._counts.values()) + 1e-10

        best_exp = None
        best_score = -float("inf")
        for e in experiments:
            if self._counts[e] == 0:
                # Unexplored — select with high priority
                score = float("inf")
            else:
                exploitation = self._values[e]
                exploration = self.exploration_weight * math.sqrt(
                    math.log(total + 1) / (self._counts[e] + 1e-10)
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_exp = e

        if best_exp is None:
            return random.choice(experiments)

        self._counts[best_exp] = self._counts.get(best_exp, 0) + 1
        return best_exp

    def update(self, exp_name: str, reward: float) -> None:
        n = self._counts.get(exp_name, 0)
        old_val = self._values.get(exp_name, 0.0)
        self._values[exp_name] = (old_val * max(n - 1, 0) + reward) / max(n, 1)


class LawOnlySelector:
    """Selects experiments based only on law-reward history (no theory discrimination)."""
    def __init__(self):
        self._rewards: Dict[str, List[float]] = {}

    def select(self, experiments: List[str]) -> str:
        # Greedy: pick experiment with highest historical reward
        best = None
        best_mean = -float("inf")
        for e in experiments:
            rewards = self._rewards.get(e, [0.0])
            mean = sum(rewards) / len(rewards) if rewards else 0.0
            if mean > best_mean:
                best_mean = mean
                best = e
        return best if best else random.choice(experiments)

    def update(self, exp_name: str, reward: float) -> None:
        self._rewards.setdefault(exp_name, []).append(reward)


# ---------------------------------------------------------------------------
# Active Experiment Designer
# ---------------------------------------------------------------------------

@dataclass
class ActiveDesignerConfig:
    # Value weights
    weight_theory_discrimination: float = 1.0
    weight_math_info_gain: float = 0.7
    weight_falsification_potential: float = 1.2
    weight_transfer_value: float = 0.5
    weight_diversity: float = 0.3
    # Cost weights
    weight_compute_cost: float = -0.2
    weight_apparatus_cost: float = -0.3
    # Constraints
    exploration_rate: float = 0.12
    max_repetition_penalty: float = 0.3
    min_experiment_value: float = 0.01
    # Budget
    compute_budget: float = float("inf")
    apparatus_budget: float = float("inf")
    experiments_per_category_limit: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weight_theory_discrimination": self.weight_theory_discrimination,
            "weight_math_info_gain": self.weight_math_info_gain,
            "weight_falsification_potential": self.weight_falsification_potential,
            "weight_transfer_value": self.weight_transfer_value,
            "weight_diversity": self.weight_diversity,
            "weight_compute_cost": self.weight_compute_cost,
            "weight_apparatus_cost": self.weight_apparatus_cost,
            "exploration_rate": self.exploration_rate,
            "max_repetition_penalty": self.max_repetition_penalty,
            "min_experiment_value": self.min_experiment_value,
        }


class ActiveExperimentDesigner:
    """
    Composite experiment selector that jointly optimizes for
    theory falsification and math language growth.

    Provides comparison baselines: random, UCB, law-only.
    Tracks diversity and penalizes wasteful repetition.
    """

    def __init__(
        self,
        config: Optional[ActiveDesignerConfig] = None,
        *,
        seed: Optional[int] = None,
    ):
        self.config = config or ActiveDesignerConfig()
        if seed is not None:
            random.seed(seed)

        self.diversity = ExperimentDiversityTracker()
        self.compute_spent = 0.0
        self.apparatus_spent = 0.0
        self._value_history: List[ExperimentValueComponents] = []

        # Baselines for comparison
        self.random_selector = RandomSelector()
        self.ucb_selector = UCBSelector()
        self.law_selector = LawOnlySelector()

    # --- Main selection ---

    def evaluate_experiment(
        self,
        exp_name: str,
        *,
        theory_discrimination: float = 0.0,
        math_info_gain: float = 0.0,
        falsification_potential: float = 0.0,
        transfer_value: float = 0.0,
        category: str = "",
        apparatus: str = "",
        compute_cost: float = 0.1,
        apparatus_cost: float = 0.0,
    ) -> ExperimentValueComponents:
        """Evaluate the composite value of an experiment."""
        cfg = self.config

        rep_penalty = self.diversity.repetition_penalty(exp_name, cfg.max_repetition_penalty)
        div_bonus = self.diversity.category_diversity_bonus(category or "unknown")

        total = (
            cfg.weight_theory_discrimination * theory_discrimination +
            cfg.weight_math_info_gain * math_info_gain +
            cfg.weight_falsification_potential * falsification_potential +
            cfg.weight_transfer_value * transfer_value +
            cfg.weight_diversity * div_bonus +
            cfg.weight_compute_cost * compute_cost +
            cfg.weight_apparatus_cost * apparatus_cost -
            rep_penalty
        )

        result = ExperimentValueComponents(
            experiment_name=exp_name,
            theory_discrimination=theory_discrimination,
            math_info_gain=math_info_gain,
            falsification_potential=falsification_potential,
            transfer_value=transfer_value,
            diversity_bonus=div_bonus,
            compute_cost=compute_cost,
            apparatus_cost=apparatus_cost,
            repetition_penalty=rep_penalty,
            total_value=total,
        )
        self._value_history.append(result)
        return result

    def select(
        self,
        experiments: List[str],
        *,
        theory_disc_map: Optional[Dict[str, float]] = None,
        math_info_map: Optional[Dict[str, float]] = None,
        falsification_map: Optional[Dict[str, float]] = None,
        transfer_map: Optional[Dict[str, float]] = None,
        category_map: Optional[Dict[str, str]] = None,
        apparatus_map: Optional[Dict[str, str]] = None,
        compute_cost_map: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, ExperimentValueComponents]:
        """
        Select the best experiment based on composite value.

        Args:
            experiments: Available experiment names
            theory_disc_map: {exp_name: theory discrimination score}
            math_info_map: {exp_name: math information gain}
            falsification_map: {exp_name: falsification potential}
            transfer_map: {exp_name: transfer value}
            category_map: {exp_name: experiment category}
            apparatus_map: {exp_name: apparatus name}
            compute_cost_map: {exp_name: compute cost}
        """
        if not experiments:
            raise ValueError("No experiments available")

        # Always retain exploration
        if random.random() < self.config.exploration_rate:
            chosen = random.choice(experiments)
            result = self.evaluate_experiment(
                chosen,
                theory_discrimination=theory_disc_map.get(chosen, 0.0) if theory_disc_map else 0.0,
                math_info_gain=math_info_map.get(chosen, 0.0) if math_info_map else 0.0,
                falsification_potential=falsification_map.get(chosen, 0.0) if falsification_map else 0.0,
                category=category_map.get(chosen, "exploration") if category_map else "exploration",
                apparatus=apparatus_map.get(chosen, "") if apparatus_map else "",
                compute_cost=compute_cost_map.get(chosen, 0.1) if compute_cost_map else 0.1,
            )
            self.diversity.record(chosen, result.category if hasattr(result, 'category') else "exploration")
            return chosen, result

        # Score all experiments
        scored: List[Tuple[float, str, ExperimentValueComponents]] = []

        for exp_name in experiments:
            cat = category_map.get(exp_name, "") if category_map else ""
            # Check category limit
            if cat and self.diversity.category_usage.get(cat, 0) >= self.config.experiments_per_category_limit:
                continue

            result = self.evaluate_experiment(
                exp_name,
                theory_discrimination=theory_disc_map.get(exp_name, 0.0) if theory_disc_map else 0.0,
                math_info_gain=math_info_map.get(exp_name, 0.0) if math_info_map else 0.0,
                falsification_potential=falsification_map.get(exp_name, 0.0) if falsification_map else 0.0,
                transfer_value=transfer_map.get(exp_name, 0.0) if transfer_map else 0.0,
                category=cat,
                apparatus=apparatus_map.get(exp_name, "") if apparatus_map else "",
                compute_cost=compute_cost_map.get(exp_name, 0.1) if compute_cost_map else 0.1,
            )
            scored.append((result.total_value, exp_name, result))

        if not scored:
            chosen = random.choice(experiments)
            # Minimal evaluation
            result = self.evaluate_experiment(chosen)
            return chosen, result

        scored.sort(key=lambda x: x[0], reverse=True)
        best_value, best_name, best_result = scored[0]

        # Filter out experiments with too-low value
        if best_value < self.config.min_experiment_value:
            chosen = random.choice(experiments)
            result = self.evaluate_experiment(chosen)
        else:
            chosen = best_name
            result = best_result

        cat = category_map.get(chosen, "") if category_map else ""
        app = apparatus_map.get(chosen, "") if apparatus_map else ""
        self.diversity.record(chosen, cat, app)

        return chosen, result

    # --- Design validation tasks for math objects ---

    def design_validation_tasks(
        self,
        math_concept_id: str,
        available_experiments: List[str],
        *,
        experiment_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate validation task specifications for a trial math object.

        Task types:
          - repeat: same experiment, check reproducibility
          - nuisance: vary irrelevant parameters
          - replacement: change object/reagent
          - cross_apparatus: different measurement device
          - heldout: previously unseen data
        """
        tasks = []

        # Repeat task
        if available_experiments:
            tasks.append({
                "type": "repeat",
                "experiment": available_experiments[0],
                "concept_id": math_concept_id,
                "description": "Reproducibility check",
                "priority": 0.3,
            })

        # Nuisance task — use a different parameter value
        if len(available_experiments) > 1:
            tasks.append({
                "type": "nuisance",
                "experiment": available_experiments[1],
                "concept_id": math_concept_id,
                "description": "Test under parameter variation",
                "priority": 0.6,
            })

        # Held-out task
        if len(available_experiments) > 2:
            tasks.append({
                "type": "heldout_validation",
                "experiment": available_experiments[-1],
                "concept_id": math_concept_id,
                "description": "Independent held-out test",
                "priority": 0.9,
            })

        return tasks

    # --- Baseline comparison ---

    def compare_baselines(
        self,
        experiments: List[str],
        num_steps: int = 50,
        *,
        reward_fn: Callable[[str], float],
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Compare all baseline selectors on a simulated reward function.

        Returns cumulative regret curves for each selector.
        """
        max_possible = max(reward_fn(e) for e in experiments)
        curves: Dict[str, List[float]] = {
            "random": [], "ucb": [], "law_only": [], "active": [],
        }

        # Reset state
        self.ucb_selector = UCBSelector()
        self.law_selector = LawOnlySelector()

        for step in range(num_steps):
            # Random
            r_exp = self.random_selector.select(experiments)
            r_reward = reward_fn(r_exp)

            # UCB
            u_exp = self.ucb_selector.select(experiments)
            u_reward = reward_fn(u_exp)
            self.ucb_selector.update(u_exp, u_reward)

            # Law-only
            l_exp = self.law_selector.select(experiments)
            l_reward = reward_fn(l_exp)
            self.law_selector.update(l_exp, l_reward)

            # Active (this designer)
            a_exp, _ = self.select(experiments)
            a_reward = reward_fn(a_exp)

            curves["random"].append(r_reward / max(max_possible, 1e-10))
            curves["ucb"].append(u_reward / max(max_possible, 1e-10))
            curves["law_only"].append(l_reward / max(max_possible, 1e-10))
            curves["active"].append(a_reward / max(max_possible, 1e-10))

        return curves

    # --- Budget management ---

    def check_budget(self) -> bool:
        """Check if budget is exhausted."""
        return (
            self.compute_spent < self.config.compute_budget and
            self.apparatus_spent < self.config.apparatus_budget
        )

    def spend(self, compute: float = 0.0, apparatus: float = 0.0) -> None:
        self.compute_spent += compute
        self.apparatus_spent += apparatus

    # --- State ---

    def get_value_history(self) -> List[Dict[str, Any]]:
        return [v.to_dict() for v in self._value_history]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "diversity": self.diversity.to_dict(),
            "compute_spent": self.compute_spent,
            "apparatus_spent": self.apparatus_spent,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActiveExperimentDesigner":
        config = ActiveDesignerConfig()
        cfg_d = d.get("config", {})
        for k, v in cfg_d.items():
            if hasattr(config, k):
                setattr(config, k, v)

        designer = cls(config=config)
        div = d.get("diversity", {})
        designer.diversity.experiment_usage = div.get("experiment_usage", {})
        designer.diversity.category_usage = div.get("category_usage", {})
        designer.diversity.apparatus_usage = div.get("apparatus_usage", {})
        designer.diversity.total_experiments = div.get("total_experiments", 0)
        designer.compute_spent = d.get("compute_spent", 0.0)
        designer.apparatus_spent = d.get("apparatus_spent", 0.0)
        return designer
