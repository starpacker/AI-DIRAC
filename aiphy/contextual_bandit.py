"""Theory-aware experiment selection for AI-Newton.

The original experiment recommender is an arm-wise UCB: it remembers which
experiment produced rewards in the past, but it cannot condition that choice
on the theory that exists *now*.  This module provides a small, CPU-only
contextual bandit and the state/reward boundary needed to train it.

The exact (JSON serialisable) snapshots are intentionally kept separate from
the fixed-size feature vector.  The former makes a discovery trajectory
auditable; the latter is only a lossy representation used by LinUCB.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence
import zlib

import numpy as np


def _items(value: Any) -> list[tuple[Any, Any]]:
    if isinstance(value, Mapping):
        return list(value.items())
    try:
        return list(value.items())
    except (AttributeError, TypeError):
        return []


def _keys(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.keys())
    try:
        return list(value.keys())
    except (AttributeError, TypeError):
        return []


def _safe(getter, default):
    """Call into the Python/Rust boundary without hiding user interrupts."""
    try:
        return getter()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # pyo3_runtime.PanicException derives directly from BaseException.
        return default


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class TheorySnapshot:
    """An exact, compact-enough description of the knowledge state."""

    epoch: int
    concepts: dict[str, str]
    intrinsic_concepts: dict[str, str]
    conclusions: dict[str, str]
    general_laws: dict[str, dict[str, Any]]
    completion_exclusions: dict[str, list[str]]
    specific_conclusions: dict[str, list[str]]
    marked_general_laws: list[str]
    experiment_history_sizes: dict[str, int]
    law_complexity: float
    derived_laws: dict[str, dict[str, Any]] = field(default_factory=dict)
    derived_concepts: list[str] = field(default_factory=list)

    @property
    def state_hash(self) -> str:
        return _canonical_hash(self.to_json())

    @property
    def valid_pairs(self) -> set[tuple[str, str]]:
        return {
            (law_name, exp_name)
            for law_name, law in self.general_laws.items()
            for exp_name in law.get("valid_experiments", [])
        }

    @staticmethod
    def _independent_scope(law: Mapping[str, Any]) -> list[str]:
        certificate = law.get("independence") or {}
        independent = certificate.get("independent_experiments")
        if certificate.get("status") == "independent" and independent is not None:
            return list(independent)
        # Legacy and branch-safe symbolic laws have no probabilistic marginal
        # certificate; their validated scope remains the conservative support.
        return list(law.get("valid_experiments", []))

    @property
    def independent_pairs(self) -> set[tuple[str, str]]:
        return {
            (law_name, exp_name)
            for law_name, law in self.general_laws.items()
            for exp_name in self._independent_scope(law)
        }

    @property
    def active_concept_names(self) -> set[str]:
        return set(self.concepts) - set(self.derived_concepts)

    @property
    def exclusion_pairs(self) -> set[tuple[str, str]]:
        return {
            (law_name, exp_name)
            for law_name, experiments in self.completion_exclusions.items()
            for exp_name in experiments
        }

    @property
    def maturity(self) -> float:
        # Only the independent law basis lives in general_laws.  Validation
        # scope is logarithmically saturated per law so algebraic copies or a
        # single very broad law cannot manufacture maturity by pair counting.
        coverage_evidence = 1.5 * sum(
            math.log1p(len(self._independent_scope(law)))
            for law in self.general_laws.values()
        )
        evidence = (
            6.0 * len(self.general_laws)
            + coverage_evidence
            + 2.0 * len(self.marked_general_laws)
            + 0.05 * min(len(self.active_concept_names), 40)
        )
        return float(1.0 - math.exp(-evidence / 100.0))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "TheorySnapshot":
        return TheorySnapshot(**dict(data))


@dataclass
class ExperimentDescriptor:
    """The action-side state for one candidate experiment."""

    name: str
    spatial_dimension: int
    objects: list[tuple[str, str, int]]
    original_data: list[str]
    original_concepts: list[str]
    known_valid_laws: list[str]
    known_excluded_laws: list[str]
    testable_unknown_laws: list[str]
    non_applicable_laws: list[str]
    selection_count: int
    computational_limit: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class TheoryStateEncoder:
    """Extract exact theory and experiment states from a ``Theorist``."""

    @staticmethod
    def capture(theorist: Any) -> TheorySnapshot:
        knowledge = theorist.knowledge
        memory = theorist.memory
        general = theorist.general

        concepts = {
            str(name): str(value)
            for name, value in _items(_safe(lambda: knowledge.fetch_concepts, {}))
        }
        intrinsic = {
            str(name): str(value)
            for name, value in _items(_safe(lambda: knowledge.fetch_intrinsic_concepts, {}))
        }
        conclusions = {
            str(name): str(value)
            for name, value in _items(_safe(lambda: knowledge.fetch_conclusions, {}))
        }

        laws: dict[str, dict[str, Any]] = {}
        law_complexity = 0.0
        for name, law in sorted(general.general_laws.items()):
            prop = str(law.prop)
            complexity = _safe(lambda law=law: float(knowledge.raw_complexity(law.prop)), None)
            if complexity is None:
                complexity = float(len(prop))
            law_complexity += complexity
            laws[str(name)] = {
                "prop": prop,
                "type": str(law.type),
                "indication": str(law.indication),
                "valid_experiments": sorted(str(x) for x in law.valid_experiments),
                "father": getattr(law, "father_name", None),
                "complexity": complexity,
                "independence": dict(
                    general.service.independence_certificates.get(str(name), {})
                ),
            }

        specific_conclusions = {
            str(exp_name): sorted(str(name) for name in _keys(model.conclusions))
            for exp_name, model in sorted(theorist.specific.items())
        }
        experiment_history_sizes = {
            str(exp_name): len(records)
            for exp_name, records in sorted(memory.history.items())
        }
        exclusions = {
            str(name): sorted(str(exp) for exp in experiments)
            for name, experiments in sorted(general.completion_exclude.items())
        }
        marked = sorted(str(x) for x in general.service.marked_general_laws)

        return TheorySnapshot(
            epoch=int(memory.epoch),
            concepts=dict(sorted(concepts.items())),
            intrinsic_concepts=dict(sorted(intrinsic.items())),
            conclusions=dict(sorted(conclusions.items())),
            general_laws=laws,
            completion_exclusions=exclusions,
            specific_conclusions=specific_conclusions,
            marked_general_laws=marked,
            experiment_history_sizes=experiment_history_sizes,
            law_complexity=float(law_complexity),
            derived_laws=dict(sorted(general.service.derived_general_laws.items())),
            derived_concepts=sorted(
                set(general.service.derived_concepts)
                | set(getattr(memory, "redundant_concepts", set()))
            ),
        )

    @staticmethod
    def describe_experiment(theorist: Any, exp_name: str) -> ExperimentDescriptor:
        model = theorist.specific[exp_name]
        experiment = model.experiment
        snapshot_laws = theorist.general.general_laws

        obj_info = _safe(lambda: experiment.obj_info, {})
        objects = sorted(
            (str(obj_name), str(obj_type), int(obj_id))
            for obj_name, (obj_type, obj_id) in _items(obj_info)
        )
        original_data = sorted(str(x) for x in _safe(lambda: experiment.original_data, []))
        original_concepts = sorted(str(x) for x in _safe(lambda: experiment.original_concept, []))

        valid: list[str] = []
        excluded: list[str] = []
        testable: list[str] = []
        non_applicable: list[str] = []
        for law_name, law in sorted(snapshot_laws.items()):
            if exp_name in law.valid_experiments:
                valid.append(str(law_name))
            elif exp_name in theorist.general.completion_exclude.get(law_name, []):
                excluded.append(str(law_name))
            else:
                specialised = _safe(
                    lambda law=law: theorist.knowledge.specialize(law.relevant_concept, exp_name),
                    [],
                )
                if specialised:
                    testable.append(str(law_name))
                else:
                    non_applicable.append(str(law_name))

        old_action = theorist.memory.experiment_bandit.actions.get(exp_name)
        old_count = int(round(old_action.count)) if old_action is not None else 0
        contextual = getattr(theorist.memory, "contextual_bandit", None)
        contextual_count = contextual.action_counts.get(exp_name, 0) if contextual is not None else 0

        return ExperimentDescriptor(
            name=exp_name,
            spatial_dimension=int(_safe(lambda: experiment.spdim, 0)),
            objects=objects,
            original_data=original_data,
            original_concepts=original_concepts,
            known_valid_laws=valid,
            known_excluded_laws=excluded,
            testable_unknown_laws=testable,
            non_applicable_laws=non_applicable,
            selection_count=max(old_count, contextual_count),
            computational_limit=float(theorist.memory.computational_limit),
        )


class ContextFeatureEncoder:
    """Turn ``(theory state, experiment)`` into a fixed-size vector."""

    def __init__(self, hash_dim: int = 24):
        self.hash_dim = int(hash_dim)
        self.numeric_dim = 25
        self.dimension = self.numeric_dim + self.hash_dim

    @staticmethod
    def _log(value: float) -> float:
        return math.log1p(max(float(value), 0.0))

    def encode(self, state: TheorySnapshot, exp: ExperimentDescriptor) -> np.ndarray:
        n_laws = len(state.general_laws)
        denom = max(n_laws, 1)
        specific_total = sum(len(x) for x in state.specific_conclusions.values())
        avg_coverage = len(state.valid_pairs) / denom
        distinct_types = len({obj_type for _, obj_type, _ in exp.objects})
        testable_ratio = len(exp.testable_unknown_laws) / denom
        novelty_weight = 1.0 - state.maturity

        numeric = np.asarray([
            1.0,
            state.maturity,
            self._log(len(state.concepts)),
            self._log(n_laws),
            self._log(len(state.intrinsic_concepts)),
            self._log(len(state.conclusions)),
            self._log(specific_total),
            self._log(len(state.valid_pairs)),
            self._log(len(state.exclusion_pairs)),
            self._log(state.epoch),
            self._log(state.law_complexity) / 5.0,
            avg_coverage / max(len(state.specific_conclusions), 1),
            exp.spatial_dimension / 3.0,
            self._log(len(exp.objects)),
            self._log(distinct_types),
            self._log(len(exp.original_data)),
            self._log(len(exp.original_concepts)),
            self._log(exp.selection_count),
            len(exp.known_valid_laws) / denom,
            len(exp.known_excluded_laws) / denom,
            testable_ratio,
            novelty_weight * self._log(len(exp.original_concepts)),
            state.maturity * testable_ratio,
            len(exp.non_applicable_laws) / denom,
            self._log(exp.computational_limit) / 5.0,
        ], dtype=np.float64)

        hashed = np.zeros(self.hash_dim, dtype=np.float64)
        tokens = [f"experiment:{exp.name}"]
        tokens.extend(f"object:{obj_type}" for _, obj_type, _ in exp.objects)
        tokens.extend(f"data:{item}" for item in exp.original_data)
        tokens.extend(f"concept:{item}" for item in exp.original_concepts)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.hash_dim
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            hashed[index] += sign
        norm = np.linalg.norm(hashed)
        if norm > 0:
            hashed /= norm
        return np.concatenate([numeric, hashed])


@dataclass
class SelectionDecision:
    experiment: str
    feature: list[float]
    state_hash: str
    descriptor: dict[str, Any]
    predicted_reward: float
    uncertainty: float
    score: float
    forced_exploration: bool
    empirical_reward: float = 0.0
    selection_reason: str = "policy"


@dataclass
class AdaptiveRewardConfig:
    bootstrap_concept_reward: float = 0.45
    concept_budget: float = 40.0
    unproductive_concept_penalty: float = 0.035
    productive_concept_penalty: float = 0.005
    redundant_concept_penalty: float = 0.10
    supported_law_reward: float = 4.0
    unsupported_law_penalty: float = 1.0
    revision_reward: float = 8.0
    contradiction_reward: float = 4.0
    validation_reward: float = 2.0
    useful_removal_reward: float = 1.0
    complexity_reduction_reward: float = 0.01
    no_change_penalty: float = 0.35
    duration_penalty: float = 0.03
    timeout_penalty: float = 2.0
    error_penalty: float = 2.5


@dataclass
class RewardBreakdown:
    total: float
    maturity: float
    outcome: str
    added_concepts: int = 0
    added_derived_concepts: int = 0
    added_laws: int = 0
    removed_laws: int = 0
    revised_laws: int = 0
    new_contradictions: int = 0
    new_validations: int = 0
    backfilled_validations: int = 0
    complexity_reduction: float = 0.0
    novelty_term: float = 0.0
    revision_term: float = 0.0
    validation_term: float = 0.0
    parsimony_term: float = 0.0
    cost_term: float = 0.0
    vocabulary_scarcity: float = 0.0
    growth_penalty: float = 0.0
    legacy_reward: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveTheoryReward:
    """Reward discovery early and theory revision/parsimoniousness later."""

    def __init__(self, config: AdaptiveRewardConfig | None = None):
        self.config = config or AdaptiveRewardConfig()

    @staticmethod
    def _blend(early: float, mature: float, maturity: float) -> float:
        return early * (1.0 - maturity) + mature * maturity

    def evaluate(
        self,
        before: TheorySnapshot,
        after: TheorySnapshot,
        *,
        outcome: str = "success",
        duration_seconds: float = 0.0,
        legacy_reward: float = 0.0,
    ) -> RewardBreakdown:
        cfg = self.config
        maturity = before.maturity
        if outcome != "success":
            penalty = cfg.timeout_penalty if outcome == "timeout" else cfg.error_penalty
            cost = -cfg.duration_penalty * math.log1p(max(duration_seconds, 0.0))
            return RewardBreakdown(
                total=-penalty + cost,
                maturity=maturity,
                outcome=outcome,
                cost_term=cost,
                legacy_reward=float(legacy_reward),
            )

        before_laws = set(before.general_laws)
        after_laws = set(after.general_laws)
        added_law_names = after_laws - before_laws
        removed_law_names = before_laws - after_laws
        added_concepts = len(
            after.active_concept_names - before.active_concept_names
        )
        added_derived_concepts = len(
            set(after.derived_concepts) - set(before.derived_concepts)
        )

        revised = sum(
            1
            for name in added_law_names
            if after.general_laws[name].get("father") in before_laws
        )
        revised += sum(
            1
            for name in before_laws & after_laws
            if before.general_laws[name].get("prop") != after.general_laws[name].get("prop")
        )
        # A failed scope check for a brand-new hypothesis is useful filtering,
        # but it did not overturn the theory that existed before the action.
        # Only new exclusions of pre-existing laws count as falsification.
        contradictions = sum(
            1
            for law_name, _ in after.exclusion_pairs - before.exclusion_pairs
            if law_name in before_laws
        )
        new_validation_pairs = after.valid_pairs - before.valid_pairs
        # Cross-validation performed while a new law is created reuses cached
        # specific models.  It supports accepting that law, but it is not new
        # evidence caused by the selected experiment and must not be rewarded
        # a second time.  Only newly tested pairs of pre-existing laws count.
        validations = sum(
            law_name in before_laws
            for law_name, _ in new_validation_pairs
        )
        backfilled_validations = len(new_validation_pairs) - validations
        complexity_reduction = before.law_complexity - after.law_complexity

        supported_laws = sum(
            bool(after.general_laws[name].get("valid_experiments"))
            for name in added_law_names
        )
        unsupported_laws = len(added_law_names) - supported_laws
        vocabulary_scarcity = math.exp(
            -len(before.active_concept_names) / max(cfg.concept_budget, 1.0)
        )
        theory_progress = bool(
            supported_laws or revised or contradictions or validations
        )
        concept_bonus = (
            cfg.bootstrap_concept_reward
            * vocabulary_scarcity
            * math.log1p(added_concepts)
        )
        growth_rate = (
            cfg.productive_concept_penalty
            if theory_progress
            else cfg.unproductive_concept_penalty
        )
        growth_penalty = (
            growth_rate * added_concepts * (1.0 - vocabulary_scarcity)
            + cfg.redundant_concept_penalty * added_derived_concepts
        )
        law_term = (
            cfg.supported_law_reward * supported_laws
            - cfg.unsupported_law_penalty * unsupported_laws
        )
        novelty_term = concept_bonus - growth_penalty + law_term
        # Evidence counts can be large because a law is checked in many
        # experiments at once.  Log saturation prevents scope size alone from
        # overwhelming every other reward component.
        revision_term = (
            cfg.revision_reward * revised
            + cfg.contradiction_reward * math.log1p(contradictions)
        )
        validation_term = cfg.validation_reward * math.log1p(validations)
        parsimony_term = (
            cfg.useful_removal_reward * len(removed_law_names)
            + cfg.complexity_reduction_reward * complexity_reduction
        )
        cost_term = -cfg.duration_penalty * math.log1p(max(duration_seconds, 0.0))

        changed = any((
            added_concepts,
            added_derived_concepts,
            added_law_names,
            removed_law_names,
            revised,
            contradictions,
            validations,
        ))
        no_change = 0.0 if changed else -cfg.no_change_penalty
        total = novelty_term + revision_term + validation_term + parsimony_term + cost_term + no_change
        return RewardBreakdown(
            total=float(total),
            maturity=maturity,
            outcome=outcome,
            added_concepts=added_concepts,
            added_derived_concepts=added_derived_concepts,
            added_laws=len(added_law_names),
            removed_laws=len(removed_law_names),
            revised_laws=revised,
            new_contradictions=contradictions,
            new_validations=validations,
            backfilled_validations=backfilled_validations,
            complexity_reduction=float(complexity_reduction),
            novelty_term=float(novelty_term),
            revision_term=float(revision_term),
            validation_term=float(validation_term),
            parsimony_term=float(parsimony_term),
            cost_term=float(cost_term),
            vocabulary_scarcity=float(vocabulary_scarcity),
            growth_penalty=float(growth_penalty),
            legacy_reward=float(legacy_reward),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self.config)

    @staticmethod
    def from_json(data: Mapping[str, Any] | None) -> "AdaptiveTheoryReward":
        if not data:
            return AdaptiveTheoryReward()
        # Old checkpoints used linear early/mature concept/law coefficients.
        # Ignore unknown legacy fields while retaining any still-valid knobs.
        valid = AdaptiveRewardConfig.__dataclass_fields__
        return AdaptiveTheoryReward(AdaptiveRewardConfig(**{
            key: value for key, value in data.items() if key in valid
        }))


class LinUCBExperimentSelector:
    """Hybrid contextual/per-arm UCB with coverage and repetition control."""

    def __init__(
        self,
        alpha: float = 1.25,
        regularization: float = 1.0,
        hash_dim: int = 24,
        reward_clip: float = 20.0,
        max_history: int = 512,
        max_state_archive: int = 128,
        contextual_weight: float = 0.35,
        empirical_weight: float = 1.0,
        arm_exploration: float = 0.75,
        coverage_interval: int = 4,
        max_consecutive: int = 3,
        repeat_penalty: float = 0.40,
        no_progress_penalty: float = 0.12,
        ema_rate: float = 0.20,
    ):
        self.alpha = float(alpha)
        self.regularization = float(regularization)
        self.reward_clip = float(reward_clip)
        self.max_history = int(max_history)
        self.max_state_archive = int(max_state_archive)
        self.contextual_weight = float(contextual_weight)
        self.empirical_weight = float(empirical_weight)
        self.arm_exploration = float(arm_exploration)
        self.coverage_interval = max(int(coverage_interval), 1)
        self.max_consecutive = max(int(max_consecutive), 1)
        self.repeat_penalty = float(repeat_penalty)
        self.no_progress_penalty = float(no_progress_penalty)
        self.ema_rate = float(ema_rate)
        self.encoder = ContextFeatureEncoder(hash_dim=hash_dim)
        self.a = np.eye(self.encoder.dimension, dtype=np.float64) * self.regularization
        self.b = np.zeros(self.encoder.dimension, dtype=np.float64)
        self.action_counts: dict[str, int] = {}
        self.action_reward_ema: dict[str, float] = {}
        self.action_no_progress: dict[str, int] = {}
        self.action_last_selected: dict[str, int] = {}
        self.action_compute_limits: dict[str, float] = {}
        self.action_timeouts: dict[str, int] = {}
        self.last_experiment: str | None = None
        self.consecutive_selections = 0
        self.total_updates = 0
        self.last_limit_round = 0
        self.history: list[dict[str, Any]] = []
        self.state_archive: dict[str, dict[str, Any]] = {}

    def remember_state(self, state: TheorySnapshot) -> str:
        state_hash = state.state_hash
        raw = json.dumps(
            state.to_json(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        self.state_archive[state_hash] = {
            "encoding": "zlib+base64",
            "payload": base64.b64encode(zlib.compress(raw, level=6)).decode("ascii"),
        }
        while len(self.state_archive) > self.max_state_archive:
            oldest = next(iter(self.state_archive))
            del self.state_archive[oldest]
        return state_hash

    def load_state(self, state_hash: str) -> TheorySnapshot:
        record = self.state_archive[state_hash]
        if record.get("encoding") == "zlib+base64":
            raw = zlib.decompress(base64.b64decode(record["payload"]))
            return TheorySnapshot.from_json(json.loads(raw.decode("utf-8")))
        # Compatibility with early, uncompressed development checkpoints.
        return TheorySnapshot.from_json(record)

    def select(
        self,
        theorist: Any,
        candidates: Sequence[str],
        state: TheorySnapshot | None = None,
        *,
        verbose: bool = False,
    ) -> SelectionDecision:
        candidates = sorted(set(str(x) for x in candidates))
        if not candidates:
            raise ValueError("The contextual selector received an empty experiment pool")
        state = state or TheoryStateEncoder.capture(theorist)
        state_hash = self.remember_state(state)
        inv_a = np.linalg.inv(self.a)
        theta = inv_a @ self.b
        cold_start = [name for name in candidates if self.action_counts.get(name, 0) == 0]
        selection_reason = "policy"
        if cold_start:
            pool = cold_start
            selection_reason = "cold_start"
        elif self.total_updates % self.coverage_interval == 0:
            least_count = min(self.action_counts.get(name, 0) for name in candidates)
            pool = [name for name in candidates
                    if self.action_counts.get(name, 0) == least_count]
            selection_reason = "coverage"
        else:
            pool = list(candidates)

        # A hard cooldown is deliberately separate from learned reward.  It
        # guarantees that model misspecification cannot select one cheap arm
        # hundreds of times in a row.
        if (self.last_experiment in pool
                and self.consecutive_selections >= self.max_consecutive
                and len(pool) > 1):
            pool = [name for name in pool if name != self.last_experiment]
            selection_reason = "cooldown" if selection_reason == "policy" else selection_reason

        scored: list[tuple[float, str, np.ndarray, ExperimentDescriptor,
                           float, float, float]] = []
        for name in pool:
            descriptor = TheoryStateEncoder.describe_experiment(theorist, name)
            feature = self.encoder.encode(state, descriptor)
            prediction = float(theta @ feature)
            uncertainty = float(math.sqrt(max(feature @ inv_a @ feature, 0.0)))
            empirical = self.action_reward_ema.get(name, 0.0)
            count = self.action_counts.get(name, 0)
            arm_bonus = self.arm_exploration * math.sqrt(
                math.log(self.total_updates + 2.0) / (count + 1.0)
            )
            last_selected = self.action_last_selected.get(name, -1)
            recency = max(self.total_updates - last_selected, 0)
            recency_bonus = 0.05 * math.sqrt(math.log1p(recency))
            repeat_cost = (
                self.repeat_penalty * self.consecutive_selections
                if name == self.last_experiment else 0.0
            )
            stagnation_cost = (
                self.no_progress_penalty
                * min(self.action_no_progress.get(name, 0), 10)
            )
            contextual_score = prediction + self.alpha * uncertainty
            score = (
                self.contextual_weight * contextual_score
                + self.empirical_weight * empirical
                + arm_bonus
                + recency_bonus
                - repeat_cost
                - stagnation_cost
            )
            scored.append((score, name, feature, descriptor,
                           prediction, uncertainty, empirical))
        score, name, feature, descriptor, prediction, uncertainty, empirical = max(
            scored, key=lambda row: (row[0], row[1])
        )
        if verbose:
            print(
                "Contextual scores: "
                + str([(item[1], round(item[0], 4)) for item in sorted(scored, reverse=True)]),
                flush=True,
            )
        return SelectionDecision(
            experiment=name,
            feature=feature.tolist(),
            state_hash=state_hash,
            descriptor=descriptor.to_json(),
            predicted_reward=prediction,
            uncertainty=uncertainty,
            score=score,
            forced_exploration=selection_reason != "policy",
            empirical_reward=empirical,
            selection_reason=selection_reason,
        )

    def update(self, decision: SelectionDecision, reward: RewardBreakdown,
               next_state: TheorySnapshot | None = None) -> None:
        feature = np.asarray(decision.feature, dtype=np.float64)
        if feature.shape != (self.encoder.dimension,):
            raise ValueError(
                f"Expected a {self.encoder.dimension}-dimensional feature, got {feature.shape}"
            )
        target = float(np.clip(reward.total, -self.reward_clip, self.reward_clip))
        self.a += np.outer(feature, feature)
        self.b += target * feature
        exp_name = decision.experiment
        self.action_counts[exp_name] = self.action_counts.get(exp_name, 0) + 1
        old_ema = self.action_reward_ema.get(exp_name)
        self.action_reward_ema[exp_name] = (
            target if old_ema is None
            else (1.0 - self.ema_rate) * old_ema + self.ema_rate * target
        )
        scientific_progress = bool(
            reward.added_laws
            or reward.revised_laws
            or reward.new_contradictions
            or reward.new_validations
            or reward.total > 0.0
        )
        self.action_no_progress[exp_name] = (
            0 if scientific_progress
            else self.action_no_progress.get(exp_name, 0) + 1
        )
        if self.last_experiment == exp_name:
            self.consecutive_selections += 1
        else:
            self.last_experiment = exp_name
            self.consecutive_selections = 1
        self.action_last_selected[exp_name] = self.total_updates
        self.total_updates += 1
        next_state_hash = self.remember_state(next_state) if next_state is not None else decision.state_hash
        self.history.append({
            "experiment": decision.experiment,
            "state_hash": decision.state_hash,
            "next_state_hash": next_state_hash,
            "experiment_descriptor": decision.descriptor,
            "predicted_reward": decision.predicted_reward,
            "uncertainty": decision.uncertainty,
            "score": decision.score,
            "forced_exploration": decision.forced_exploration,
            "empirical_reward": decision.empirical_reward,
            "selection_reason": decision.selection_reason,
            "reward": reward.to_json(),
        })
        if len(self.history) > self.max_history:
            del self.history[:-self.max_history]

    def completed_rounds(self, candidates: Iterable[str]) -> int:
        candidates = list(candidates)
        if not candidates:
            return 0
        return min(self.action_counts.get(str(name), 0) for name in candidates)

    def compute_limit_for(self, experiment: str, base_limit: float,
                          max_limit: float) -> float:
        return min(
            self.action_compute_limits.get(experiment, float(base_limit)),
            float(max_limit),
        )

    def record_compute_outcome(self, experiment: str, outcome: str,
                               base_limit: float, max_limit: float,
                               multiplier: float) -> None:
        current = self.compute_limit_for(experiment, base_limit, max_limit)
        if outcome == "timeout":
            self.action_timeouts[experiment] = self.action_timeouts.get(experiment, 0) + 1
            self.action_compute_limits[experiment] = min(
                max(current * multiplier, current + base_limit),
                max_limit,
            )
        else:
            self.action_compute_limits.setdefault(experiment, current)

    def to_json(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "regularization": self.regularization,
            "reward_clip": self.reward_clip,
            "max_history": self.max_history,
            "max_state_archive": self.max_state_archive,
            "contextual_weight": self.contextual_weight,
            "empirical_weight": self.empirical_weight,
            "arm_exploration": self.arm_exploration,
            "coverage_interval": self.coverage_interval,
            "max_consecutive": self.max_consecutive,
            "repeat_penalty": self.repeat_penalty,
            "no_progress_penalty": self.no_progress_penalty,
            "ema_rate": self.ema_rate,
            "hash_dim": self.encoder.hash_dim,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
            "action_counts": self.action_counts,
            "action_reward_ema": self.action_reward_ema,
            "action_no_progress": self.action_no_progress,
            "action_last_selected": self.action_last_selected,
            "action_compute_limits": self.action_compute_limits,
            "action_timeouts": self.action_timeouts,
            "last_experiment": self.last_experiment,
            "consecutive_selections": self.consecutive_selections,
            "total_updates": self.total_updates,
            "last_limit_round": self.last_limit_round,
            "history": self.history,
            "state_archive": self.state_archive,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any] | None) -> "LinUCBExperimentSelector":
        if not data:
            return LinUCBExperimentSelector()
        obj = LinUCBExperimentSelector(
            alpha=float(data.get("alpha", 1.25)),
            regularization=float(data.get("regularization", 1.0)),
            hash_dim=int(data.get("hash_dim", 24)),
            reward_clip=float(data.get("reward_clip", 20.0)),
            max_history=int(data.get("max_history", 512)),
            max_state_archive=int(data.get("max_state_archive", 128)),
            contextual_weight=float(data.get("contextual_weight", 0.35)),
            empirical_weight=float(data.get("empirical_weight", 1.0)),
            arm_exploration=float(data.get("arm_exploration", 0.75)),
            coverage_interval=int(data.get("coverage_interval", 4)),
            max_consecutive=int(data.get("max_consecutive", 3)),
            repeat_penalty=float(data.get("repeat_penalty", 0.40)),
            no_progress_penalty=float(data.get("no_progress_penalty", 0.12)),
            ema_rate=float(data.get("ema_rate", 0.20)),
        )
        a = np.asarray(data.get("a", obj.a), dtype=np.float64)
        b = np.asarray(data.get("b", obj.b), dtype=np.float64)
        if a.shape == obj.a.shape and b.shape == obj.b.shape:
            obj.a = a
            obj.b = b
        obj.action_counts = {str(k): int(v) for k, v in data.get("action_counts", {}).items()}
        obj.action_reward_ema = {
            str(k): float(v) for k, v in data.get("action_reward_ema", {}).items()
        }
        obj.action_no_progress = {
            str(k): int(v) for k, v in data.get("action_no_progress", {}).items()
        }
        obj.action_last_selected = {
            str(k): int(v) for k, v in data.get("action_last_selected", {}).items()
        }
        obj.action_compute_limits = {
            str(k): float(v) for k, v in data.get("action_compute_limits", {}).items()
        }
        obj.action_timeouts = {
            str(k): int(v) for k, v in data.get("action_timeouts", {}).items()
        }
        obj.last_experiment = data.get("last_experiment")
        obj.consecutive_selections = int(data.get("consecutive_selections", 0))
        obj.total_updates = int(data.get("total_updates", 0))
        obj.last_limit_round = int(data.get("last_limit_round", 0))
        obj.history = list(data.get("history", []))[-obj.max_history:]
        obj.state_archive = dict(data.get("state_archive", {}))
        if len(obj.state_archive) > obj.max_state_archive:
            obj.state_archive = dict(list(obj.state_archive.items())[-obj.max_state_archive:])
        return obj
