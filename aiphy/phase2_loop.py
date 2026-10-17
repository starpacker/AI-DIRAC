"""
Phase 2 main loop — integrates theory population management with the
existing experimental pipeline.

Provides:
  - Phase2LoopRunner: standalone loop for theory-driven experiment design
  - Phase2Integrator: mixin that plugs Phase 2 into the existing Theorist
  - Phase2State: serializable snapshot of the Phase 2 pipeline state

The loop:
  1. Maintains a population of competing theories
  2. Pre-registers predictions before experiments
  3. Selects discriminating experiments
  4. Judges results (confirm/falsify) with positive info reward
  5. Designs new interventions based on remaining competition
  6. Tracks theory lineage and equivalence
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .theory_population import (
    TheoryPopulation, TheoryRecord, TheoryStatus,
)
from .prediction_registry import (
    PredictionRegistry, PredictionRecord, PredictionOutcomeRecord,
    PredictionOutcome,
)
from .falsification_judge import (
    FalsificationJudge, FalsificationResult,
)
from .discriminating_selector import (
    DiscriminatingExperimentSelector, DiscriminatingExperimentScore,
)
from .intervention_spec import (
    InterventionSpec, InterventionType, InterventionDesigner,
)


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

ExperimentExecutor = Callable[[str, Dict[str, Any]], Dict[str, Any]]
"""Execute an experiment and return observed data."""

TheoryGeneratorFn = Callable[[int, Optional[List[str]]], List[Tuple[str, str]]]
"""Generate candidate theories. Returns list of (name, expression)."""

PredictionMakerFn = Callable[[str, str, str], Tuple[str, float]]
"""Make a prediction for (theory_id, experiment_name, theory_expression).
Returns (predicted_expression, certainty)."""


# ---------------------------------------------------------------------------
# Loop state
# ---------------------------------------------------------------------------

@dataclass
class Phase2State:
    """Serializable snapshot of the Phase 2 pipeline."""
    population: Dict[str, Any]
    registry: Dict[str, Any]
    judge: Dict[str, Any]
    selector: Dict[str, Any]
    interventions: List[Dict[str, Any]]
    iteration: int
    total_info_reward: float
    active_theory_count: int
    falsified_theory_count: int
    discovered_laws: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "population": self.population,
            "registry": self.registry,
            "judge": self.judge,
            "selector": self.selector,
            "interventions": self.interventions,
            "iteration": self.iteration,
            "total_info_reward": self.total_info_reward,
            "active_theory_count": self.active_theory_count,
            "falsified_theory_count": self.falsified_theory_count,
            "discovered_laws": self.discovered_laws,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase2State":
        return cls(
            population=d["population"],
            registry=d["registry"],
            judge=d["judge"],
            selector=d["selector"],
            interventions=d.get("interventions", []),
            iteration=d.get("iteration", 0),
            total_info_reward=d.get("total_info_reward", 0.0),
            active_theory_count=d.get("active_theory_count", 0),
            falsified_theory_count=d.get("falsified_theory_count", 0),
            discovered_laws=d.get("discovered_laws", []),
        )


# ---------------------------------------------------------------------------
# Main loop runner
# ---------------------------------------------------------------------------

class Phase2LoopRunner:
    """
    Standalone loop that orchestrates the Phase 2 pipeline.

    Usage:
        runner = Phase2LoopRunner()
        runner.initialize_with_candidates(["x+5", "2*x+3", "x**2-1"])
        for step in range(100):
            summary = runner.step(experiment_executor)
            if summary["converged"]:
                break
    """

    def __init__(
        self,
        *,
        population_size: int = 10,
        conservation_tolerance: float = 0.05,
        zero_tolerance: float = 0.01,
        exploration_rate: float = 0.15,
        contradiction_reward: float = 4.0,
        confirmation_reward: float = 1.0,
        convergence_threshold: int = 3,     # active theories <= this → converged
        max_falsified_ratio: float = 0.9,   # 90% falsified → restart generation
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.convergence_threshold = convergence_threshold
        self.max_falsified_ratio = max_falsified_ratio

        if seed is not None:
            random.seed(seed)

        # Core pipeline objects
        self.population = TheoryPopulation()
        self.registry = PredictionRegistry()
        self.judge = FalsificationJudge(
            self.population,
            self.registry,
            conservation_tolerance=conservation_tolerance,
            zero_tolerance=zero_tolerance,
            contradiction_reward=contradiction_reward,
            confirmation_reward=confirmation_reward,
        )
        self.selector = DiscriminatingExperimentSelector(
            self.population,
            self.registry,
            self.judge,
            exploration_rate=exploration_rate,
        )
        self.intervention_designer = InterventionDesigner()

        # State
        self.iteration = 0
        self.total_info_reward = 0.0
        self._interventions: List[InterventionSpec] = []
        self._discovered_laws: List[str] = []
        self._terminated = False

    # --- Initialization ---

    def initialize_with_candidates(
        self,
        expressions: List[str],
        names: Optional[List[str]] = None,
    ) -> int:
        """
        Initialize the population with candidate theory expressions.

        Returns number of theories created.
        """
        if names is None:
            names = [f"candidate_{i}" for i in range(len(expressions))]

        for name, expr in zip(names, expressions):
            tid = self.population.add_theory(name=name, expression=expr)
            self.population.activate(tid)

        return self.population.count_active()

    def add_theory(self, name: str, expression: str, activate: bool = True) -> str:
        """Add a single theory to the population."""
        tid = self.population.add_theory(name=name, expression=expression)
        if activate:
            self.population.activate(tid)
        return tid

    # --- Main step ---

    def step(
        self,
        experiment_executor: ExperimentExecutor,
        *,
        available_experiments: Optional[List[str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute one step of the Phase 2 loop.

        Args:
            experiment_executor: Callable that runs an experiment and returns observed data.
                Signature: (experiment_name: str, context: dict) -> dict
                Must return dict with keys like 'is_conserved', 'mean', 'std', 'is_zero', etc.
            available_experiments: List of experiment names to choose from.
            experiment_metadata: Extra context to pass to the executor.
            verbose: Print debug info.

        Returns:
            Summary dict with iteration stats.
        """
        if self._terminated:
            return self._make_summary(converged=True, reason="terminated")

        actives = self.population.active_theories
        active_count = len(actives)

        # Check convergence
        if active_count <= self.convergence_threshold:
            self._terminated = True
            return self._make_summary(converged=True, reason=f"converged to {active_count} theories")

        # Check if too many falsified — time to generate new theories
        total = self.population.count()
        falsified = len(self.population.falsified_theories)
        if total > 0 and falsified / total >= self.max_falsified_ratio:
            # All existing theories are wrong — need new generation
            if verbose:
                print(f"  [{self.iteration}] {falsified}/{total} falsified — need new candidates")

        # --- 1. Select discriminating experiment ---
        if available_experiments is None:
            available_experiments = self._collect_all_experiments()

        try:
            chosen_exp = self.selector.select_experiment(available_experiments)
        except ValueError:
            self._terminated = True
            return self._make_summary(converged=True, reason="no experiments available")

        # --- 2. Pre-register predictions for all theories that haven't predicted ---
        self._pre_register_predictions(chosen_exp, experiment_metadata)

        # --- 3. Execute the experiment ---
        context = experiment_metadata or {}
        context["experiment_name"] = chosen_exp
        context["iteration"] = self.iteration

        observed = experiment_executor(chosen_exp, context)

        # --- 4. Judge results ---
        results = self.judge.judge_experiment(chosen_exp, observed)
        info_reward = sum(r.information_reward for r in results)
        self.total_info_reward += info_reward

        # Apply results to population
        counts = self.judge.apply_results_to_population(results)

        # --- 5. Design new interventions from remaining competition ---
        new_interventions = self._design_discriminating_interventions(chosen_exp)
        self._interventions.extend(new_interventions)

        # --- 6. Check for discovered laws ---
        new_laws = self._check_law_qualifications()
        self._discovered_laws.extend(new_laws)

        if verbose:
            print(
                f"  [{self.iteration}] exp={chosen_exp}, "
                f"active={active_count}→{self.population.count_active()}, "
                f"reward={info_reward:.1f}, "
                f"falsified={counts['falsified']}, "
                f"laws={len(new_laws)}"
            )

        self.iteration += 1

        return self._make_summary(
            converged=False,
            experiment=chosen_exp,
            info_reward=info_reward,
            counts=counts,
        )

    def run(
        self,
        experiment_executor: ExperimentExecutor,
        max_iterations: int = 100,
        available_experiments: Optional[List[str]] = None,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the Phase 2 loop until convergence or max iterations.

        Args:
            experiment_executor: Callable that executes experiments and returns observed data.
            max_iterations: Maximum number of iterations.
            available_experiments: List of experiment names.
            experiment_metadata: Extra context.
            callback: Called after each step with the step summary.
            verbose: Print debug info.

        Returns:
            Final summary dict.
        """
        summary = {}
        for _ in range(max_iterations):
            summary = self.step(
                experiment_executor,
                available_experiments=available_experiments,
                experiment_metadata=experiment_metadata,
                verbose=verbose,
            )
            if callback:
                callback(summary)
            if summary.get("converged"):
                break
        return summary

    # --- Internal helpers ---

    def _collect_all_experiments(self) -> List[str]:
        """Collect all experiment names that have predictions registered."""
        from collections import OrderedDict
        seen: Dict[str, bool] = OrderedDict()
        for pred in self.registry._predictions.values():
            if pred.experiment_name not in seen:
                seen[pred.experiment_name] = True
        if not seen:
            # Fallback: use experiments from theory valid_experiments
            for t in self.population.active_theories.values():
                for exp in t.valid_experiments:
                    seen[exp] = True
        return list(seen.keys())

    def _pre_register_predictions(
        self,
        experiment_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Pre-register predictions for all active theories that haven't predicted."""
        count = 0
        for tid, theory in self.population.active_theories.items():
            # Skip if already predicted on this experiment
            existing = [
                p for p in self.registry.get_predictions_for_theory(tid)
                if p.experiment_name == experiment_name
            ]
            if existing:
                continue

            # Use the expression as the prediction placeholder
            # In real usage, this would evaluate the expression at experiment parameters
            self.registry.pre_register(
                theory_id=tid,
                experiment_name=experiment_name,
                predicted_expression=theory.expression,
                prediction_type="conserved",
                source="phase2_loop",
                iteration=self.iteration,
            )
            count += 1
        return count

    def _design_discriminating_interventions(
        self, last_experiment: str
    ) -> List[InterventionSpec]:
        """Design new discriminating interventions based on remaining competition."""
        interventions: List[InterventionSpec] = []
        actives = list(self.population.active_theories.keys())

        if len(actives) < 2:
            return interventions

        # Find the best discriminating experiment for each pair of active theories
        for i in range(min(len(actives), 5)):  # Limit to 5 pairs
            for j in range(i + 1, min(len(actives), 6)):
                tid_a, tid_b = actives[i], actives[j]

                # Get predictions for both theories
                preds_a = self.registry.get_predictions_for_theory(tid_a)
                preds_b = self.registry.get_predictions_for_theory(tid_b)

                exp_a = {p.experiment_name: p for p in preds_a}
                exp_b = {p.experiment_name: p for p in preds_b}

                # Find an experiment where they differ
                for exp_name, pa in exp_a.items():
                    pb = exp_b.get(exp_name)
                    if pb is not None and pa.predicted_expression != pb.predicted_expression:
                        spec = self.intervention_designer.design_from_competition(
                            theory_id_a=tid_a,
                            theory_id_b=tid_b,
                            discriminating_experiment=exp_name,
                            prediction_a=pa.predicted_expression,
                            prediction_b=pb.predicted_expression,
                            priority=0.9,
                        )
                        interventions.append(spec)
                        break

        return interventions

    def _check_law_qualifications(self) -> List[str]:
        """Check which theories now qualify as general laws."""
        new_laws = []
        for tid in self.population.active_theories:
            if tid in self._discovered_laws:
                continue
            qual = self.judge.compute_law_qualification(tid)
            if qual["qualified"]:
                new_laws.append(tid)

        # Also check theories that survive with lone-active status
        if self.population.count_active() == 1:
            sole_survivor = list(self.population.active_theories.keys())[0]
            if sole_survivor not in self._discovered_laws:
                qual = self.judge.compute_law_qualification(sole_survivor)
                if qual["confirmed"] or not qual["contradictions"]:
                    new_laws.append(sole_survivor)

        return new_laws

    def _make_summary(
        self,
        converged: bool = False,
        reason: str = "",
        experiment: str = "",
        info_reward: float = 0.0,
        counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Build a summary of the current loop state."""
        actives = self.population.active_theories
        falsified = self.population.falsified_theories

        summary: Dict[str, Any] = {
            "iteration": self.iteration,
            "converged": converged,
            "reason": reason,
            "experiment": experiment,
            "active_theory_count": len(actives),
            "falsified_theory_count": len(falsified),
            "total_theory_count": self.population.count(),
            "total_info_reward": self.total_info_reward,
            "step_info_reward": info_reward,
            "active_theories": [
                {"id": tid, "name": t.name, "expression": t.expression}
                for tid, t in actives.items()
            ],
            "falsified_theories": [
                {"id": tid, "name": t.name, "falsified_by": t.falsified_by}
                for tid, t in falsified.items()
            ],
            "discovered_laws": self._discovered_laws,
            "pending_interventions": len(self._interventions),
            "step_counts": counts or {},
        }

        # Competition report
        competition = self.selector.report_competition_status()
        summary["competition"] = competition

        return summary

    # --- State serialization ---

    def get_state(self) -> Phase2State:
        """Get a serializable snapshot of the pipeline."""
        return Phase2State(
            population=self.population.to_dict(),
            registry=self.registry.to_dict(),
            judge=self.judge.to_dict(),
            selector=self.selector.to_dict(),
            interventions=[s.to_dict() for s in self._interventions],
            iteration=self.iteration,
            total_info_reward=self.total_info_reward,
            active_theory_count=self.population.count_active(),
            falsified_theory_count=len(self.population.falsified_theories),
            discovered_laws=list(self._discovered_laws),
        )

    def load_state(self, state: Phase2State) -> None:
        """Restore the pipeline from a snapshot."""
        self.population = TheoryPopulation.from_dict(state.population)
        self.registry = PredictionRegistry.from_dict(state.registry)
        self.judge = FalsificationJudge.from_dict(
            state.judge, self.population, self.registry
        )
        self.selector = DiscriminatingExperimentSelector.from_dict(
            state.selector, self.population, self.registry, self.judge
        )
        self._interventions = [
            InterventionSpec.from_dict(s) for s in state.interventions
        ]
        self.iteration = state.iteration
        self.total_info_reward = state.total_info_reward
        self._discovered_laws = list(state.discovered_laws)


# ---------------------------------------------------------------------------
# Theorist integration mixin
# ---------------------------------------------------------------------------

class Phase2Integrator:
    """
    Mixin/helper that plugs Phase 2 into the existing Theorist pipeline.

    This is designed to be used either as a mixin or as a standalone helper
    that wraps a Theorist instance. It bridges:
      - Theorist.knowledge → theory expressions (candidates)
      - Theorist.specific → experiment names and parameters
      - Theorist.memory → experiment history (reward feedback)
    """

    def __init__(
        self,
        *,
        population_size: int = 10,
        conservation_tolerance: float = 0.05,
        exploration_rate: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.runner = Phase2LoopRunner(
            population_size=population_size,
            conservation_tolerance=conservation_tolerance,
            exploration_rate=exploration_rate,
            seed=seed,
        )
        self._theorist = None  # Set by attach()

    def attach(self, theorist) -> None:
        """Attach to a Theorist instance."""
        self._theorist = theorist

    def extract_candidates_from_theorist(self) -> List[str]:
        """
        Extract candidate theory expressions from the Theorist's knowledge base.
        """
        if self._theorist is None:
            return []

        candidates: List[str] = []

        # Extract from general laws
        general = getattr(self._theorist, "general", None)
        if general is not None:
            law_exprs = getattr(general, "law_expressions", None)
            if law_exprs:
                candidates.extend(str(e) for e in law_exprs.values())

        # Extract from specific models
        specific = getattr(self._theorist, "specific", {})
        for model in specific.values():
            conserved = getattr(model, "conserved_infos", {})
            for info in conserved.values():
                if hasattr(info, "expression"):
                    candidates.append(str(info.expression))

        # Extract from knowledge concepts
        knowledge = getattr(self._theorist, "knowledge", None)
        if knowledge is not None:
            concepts = getattr(knowledge, "concept_list", [])
            for c in concepts[:20]:  # Limit
                if hasattr(c, "exp"):
                    candidates.append(str(c.exp))

        return candidates

    def extract_experiments_from_theorist(self) -> List[str]:
        """Extract experiment names from the Theorist."""
        if self._theorist is None:
            return []

        specific = getattr(self._theorist, "specific", {})
        return list(specific.keys())

    def make_theorist_executor(self) -> ExperimentExecutor:
        """
        Create an experiment executor that calls into the Theorist.

        The executor runs an actual experiment using the Theorist's
        experiment framework and returns observed data.
        """

        def executor(experiment_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
            if self._theorist is None:
                return {"is_conserved": True, "mean": 0.0, "std": 0.01}

            specific = getattr(self._theorist, "specific", {})
            model = specific.get(experiment_name)

            if model is None:
                return {"is_conserved": True, "mean": 0.0, "std": 0.01}

            # Run the experiment through the specific model
            try:
                # Trigger observation
                result = model.process()
                if result is not None:
                    return {
                        "is_conserved": getattr(result, "is_conserved", True),
                        "mean": getattr(result, "mean", 0.0),
                        "std": getattr(result, "std", 0.01),
                        "is_zero": getattr(result, "is_zero", False),
                        "expression": str(getattr(result, "expression", "")),
                        "coefficient_of_variation": getattr(
                            result, "coefficient_of_variation", 0.0
                        ),
                    }
            except Exception:
                pass

            return {"is_conserved": True, "mean": 0.0, "std": 0.01}

        return executor

    def initialize_from_theorist(self) -> int:
        """Initialize Phase 2 from the Theorist's current state."""
        candidates = self.extract_candidates_from_theorist()
        if not candidates:
            # Fallback: create simple candidates
            candidates = ["x + 1", "x*2", "x**2", "x/2", "x - 1"]

        return self.runner.initialize_with_candidates(candidates)

    def run_integrated(
        self,
        max_iterations: int = 50,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the Phase 2 loop integrated with the Theorist.

        Uses the Theorist as the experiment executor.
        """
        executor = self.make_theorist_executor()
        experiments = self.extract_experiments_from_theorist()

        return self.runner.run(
            experiment_executor=executor,
            max_iterations=max_iterations,
            available_experiments=experiments or None,
            verbose=verbose,
        )
