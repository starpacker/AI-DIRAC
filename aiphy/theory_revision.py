"""End-to-end fixed-experiment theory revision controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Sequence

import numpy as np
import sympy as sp

from .core import DataStruct
from .probabilistic_reducer import (
    ProbabilisticReducerConfig,
    RandomizedKahlerRankOracle,
)
from .pysr_regression import (
    GenericSearchSpec,
    PySRDiscovery,
    TheoryUpdateResult,
    _mean_trace,
)


@dataclass(frozen=True)
class RevisionControllerConfig:
    contradiction_nrmse: float = 0.03
    acceptance_nrmse: float = 1e-5
    minimum_relative_improvement: float = 0.90
    contradiction_reward: float = 4.0
    revision_reward: float = 8.0
    fit_gain_reward: float = 2.0
    validation_reward: float = 2.0
    complexity_penalty: float = 0.05
    duration_penalty: float = 0.03
    redundant_penalty: float = 2.0


@dataclass
class ActiveTheoryLaw:
    name: str
    formula: str
    target: str
    father: str | None = None
    valid_experiments: list[str] = field(default_factory=list)


@dataclass
class RevisionTheoryState:
    active_laws: dict[str, ActiveTheoryLaw]
    archived_laws: dict[str, dict[str, Any]] = field(default_factory=dict)
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_law(cls, law: ActiveTheoryLaw) -> "RevisionTheoryState":
        return cls(active_laws={law.name: law})


@dataclass(frozen=True)
class FormulaEvaluation:
    nrmse: float
    bias: float
    sample_count: int


@dataclass(frozen=True)
class RevisionRewardBreakdown:
    total: float
    contradiction_term: float = 0.0
    revision_term: float = 0.0
    fit_gain_term: float = 0.0
    validation_term: float = 0.0
    parsimony_term: float = 0.0
    redundancy_term: float = 0.0
    cost_term: float = 0.0


@dataclass
class TheoryRevisionOutcome:
    status: str
    old_law: ActiveTheoryLaw
    new_law: ActiveTheoryLaw | None
    old_training: FormulaEvaluation
    old_validation: list[FormulaEvaluation]
    new_training: FormulaEvaluation | None
    new_validation: list[FormulaEvaluation]
    strategy: str | None
    replaced_variable: str | None
    reducer_certificate: dict[str, Any] | None
    reward: RevisionRewardBreakdown
    duration_seconds: float
    search_audit: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["old_law"] = asdict(self.old_law)
        result["new_law"] = asdict(self.new_law) if self.new_law else None
        return result


@dataclass
class KnowledgeBaseRevisionOutcome:
    """Combined SR, real-knowledge-base, and adaptive-reward result."""

    search: TheoryRevisionOutcome
    commit: Any | None
    adaptive_reward: Any
    before_state_hash: str
    after_state_hash: str

    @property
    def status(self) -> str:
        if self.search.status != "revised":
            return self.search.status
        return self.commit.status if self.commit is not None else "commit_missing"

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "search": self.search.to_json(),
            "commit": asdict(self.commit) if self.commit is not None else None,
            "adaptive_reward": asdict(self.adaptive_reward),
            "before_state_hash": self.before_state_hash,
            "after_state_hash": self.after_state_hash,
        }


class TheoryRevisionController:
    """Falsify, repair, validate, reduce, and atomically replace one law."""

    def __init__(
        self,
        discovery: PySRDiscovery,
        config: RevisionControllerConfig | None = None,
        reducer_config: ProbabilisticReducerConfig | None = None,
    ) -> None:
        self.discovery = discovery
        self.config = config or RevisionControllerConfig()
        self.reducer = RandomizedKahlerRankOracle(reducer_config)

    @staticmethod
    def _sympy_formula(formula: str, specification: GenericSearchSpec) -> sp.Expr:
        replacements = {
            expression: sp.Symbol(f"q{index}")
            for index, expression in enumerate(specification.scalars)
        }
        safe = formula
        for expression in sorted(replacements, key=len, reverse=True):
            safe = safe.replace(expression, str(replacements[expression]))
        parsed = sp.sympify(safe.replace("^", "**"), locals={
            str(symbol): symbol for symbol in replacements.values()
        })
        if not parsed.free_symbols <= set(replacements.values()):
            raise ValueError(f"Formula uses undeclared observations: {formula}")
        return parsed

    @staticmethod
    def _values(data: DataStruct, specification: GenericSearchSpec):
        target = _mean_trace(data.fetch_data_by_str(specification.target))
        if target is None:
            raise ValueError(f"Missing observations for {specification.target}")
        values = {specification.target: target}
        for expression in specification.scalars:
            if expression == specification.target:
                continue
            trace = _mean_trace(
                data.fetch_data_by_str(expression), target.size
            )
            if trace is None:
                raise ValueError(f"Missing observations for {expression}")
            values[expression] = trace
        return values

    def evaluate_formula(
        self,
        formula: str,
        data: DataStruct,
        specification: GenericSearchSpec,
    ) -> FormulaEvaluation:
        parsed = self._sympy_formula(formula, specification)
        replacements = {
            expression: sp.Symbol(f"q{index}")
            for index, expression in enumerate(specification.scalars)
        }
        values = self._values(data, specification)
        ordered_expressions = list(specification.scalars)
        ordered_symbols = [replacements[expression] for expression in ordered_expressions]
        function = sp.lambdify(ordered_symbols, parsed, modules="numpy")
        predicted = np.asarray(
            function(*[values[expression] for expression in ordered_expressions]),
            dtype=np.float64,
        )
        target = values[specification.target]
        if predicted.ndim == 0:
            predicted = np.full(target.shape, float(predicted))
        if predicted.shape != target.shape or not np.all(np.isfinite(predicted)):
            raise ValueError(f"Formula produced invalid predictions: {formula}")
        residual = target - predicted
        scale = max(float(np.std(target)), 1e-12)
        return FormulaEvaluation(
            nrmse=float(np.sqrt(np.mean(residual**2)) / scale),
            bias=float(abs(np.mean(residual)) / scale),
            sample_count=int(target.size),
        )

    @staticmethod
    def _complexity(formula: str, specification: GenericSearchSpec) -> int:
        return int(sp.count_ops(
            TheoryRevisionController._sympy_formula(formula, specification),
            visual=False,
        ))

    def _reward(
        self,
        *,
        contradicted: bool,
        revised: bool,
        old_error: float,
        new_error: float | None,
        validations: int,
        old_complexity: int,
        new_complexity: int | None,
        redundant: bool,
        duration: float,
    ) -> RevisionRewardBreakdown:
        cfg = self.config
        contradiction_term = cfg.contradiction_reward if contradicted else 0.0
        revision_term = cfg.revision_reward if revised else 0.0
        fit_gain_term = 0.0
        if revised and new_error is not None:
            ratio = old_error / max(new_error, 1e-12)
            fit_gain_term = cfg.fit_gain_reward * min(math.log1p(ratio), 5.0)
        validation_term = (
            cfg.validation_reward * math.log1p(validations) if revised else 0.0
        )
        parsimony_term = 0.0
        if revised and new_complexity is not None:
            parsimony_term = -cfg.complexity_penalty * max(
                new_complexity - old_complexity, 0
            )
        redundancy_term = -cfg.redundant_penalty if redundant else 0.0
        cost_term = -cfg.duration_penalty * math.log1p(max(duration, 0.0))
        total = sum((
            contradiction_term,
            revision_term,
            fit_gain_term,
            validation_term,
            parsimony_term,
            redundancy_term,
            cost_term,
        ))
        return RevisionRewardBreakdown(
            total=float(total),
            contradiction_term=float(contradiction_term),
            revision_term=float(revision_term),
            fit_gain_term=float(fit_gain_term),
            validation_term=float(validation_term),
            parsimony_term=float(parsimony_term),
            redundancy_term=float(redundancy_term),
            cost_term=float(cost_term),
        )

    def revise(
        self,
        state: RevisionTheoryState,
        law_name: str,
        training_data: DataStruct,
        validation_data: Sequence[DataStruct],
        specification: GenericSearchSpec,
        *,
        strategies: Sequence[str] = (
            "additive", "multiplicative", "internal", "skeleton"
        ),
        experiment_names: Sequence[str] | None = None,
        debug: bool = False,
    ) -> TheoryRevisionOutcome:
        started = time.monotonic()
        old_law = state.active_laws[law_name]
        old_training = self.evaluate_formula(
            old_law.formula, training_data, specification
        )
        old_validation = [
            self.evaluate_formula(old_law.formula, data, specification)
            for data in validation_data
        ]
        contradicted = max(
            [old_training.nrmse, *(item.nrmse for item in old_validation)]
        ) > self.config.contradiction_nrmse
        if not contradicted:
            duration = time.monotonic() - started
            reward = self._reward(
                contradicted=False,
                revised=False,
                old_error=old_training.nrmse,
                new_error=None,
                validations=0,
                old_complexity=self._complexity(old_law.formula, specification),
                new_complexity=None,
                redundant=False,
                duration=duration,
            )
            return TheoryRevisionOutcome(
                "not_contradicted", old_law, None, old_training, old_validation,
                None, [], None, None, None, reward, duration, {}
            )

        updates = self.discovery.discover_theory_updates(
            training_data,
            specification,
            old_law.formula,
            strategies=strategies,
            debug=debug,
        )
        accepted: list[
            tuple[TheoryUpdateResult, FormulaEvaluation, list[FormulaEvaluation]]
        ] = []
        for update in updates:
            formula = str(update.expression)
            train_evaluation = self.evaluate_formula(
                formula, training_data, specification
            )
            external = [
                self.evaluate_formula(formula, data, specification)
                for data in validation_data
            ]
            maximum_error = max(
                [train_evaluation.nrmse, *(item.nrmse for item in external)]
            )
            old_maximum = max(
                [old_training.nrmse, *(item.nrmse for item in old_validation)]
            )
            relative_improvement = 1.0 - maximum_error / max(old_maximum, 1e-12)
            if (
                maximum_error <= self.config.acceptance_nrmse
                and relative_improvement >= self.config.minimum_relative_improvement
            ):
                accepted.append((update, train_evaluation, external))

        duration = time.monotonic() - started
        if not accepted:
            reward = self._reward(
                contradicted=True,
                revised=False,
                old_error=old_training.nrmse,
                new_error=None,
                validations=0,
                old_complexity=self._complexity(old_law.formula, specification),
                new_complexity=None,
                redundant=False,
                duration=duration,
            )
            return TheoryRevisionOutcome(
                "contradicted_unrepaired", old_law, None, old_training,
                old_validation, None, [], None, None, None, reward, duration,
                dict(self.discovery.last_search_audit),
            )

        update, new_training, new_validation = min(
            accepted,
            key=lambda item: (
                max(evaluation.nrmse for evaluation in item[2]),
                self._complexity(str(item[0].expression), specification),
            ),
        )
        new_formula = str(update.expression)
        retained_formulas = [
            self._sympy_formula(law.formula, specification)
            for name, law in state.active_laws.items()
            if name != law_name and law.target == specification.target
        ]
        candidate_sympy = self._sympy_formula(new_formula, specification)
        certificate = self.reducer.compare(
            retained_formulas, candidate_sympy,
            key=f"revision:{law_name}:{new_formula}",
        )
        redundant = certificate.status == "dependent"
        if redundant:
            reward = self._reward(
                contradicted=True,
                revised=False,
                old_error=old_training.nrmse,
                new_error=new_training.nrmse,
                validations=0,
                old_complexity=self._complexity(old_law.formula, specification),
                new_complexity=self._complexity(new_formula, specification),
                redundant=True,
                duration=duration,
            )
            return TheoryRevisionOutcome(
                "rejected_redundant", old_law, None, old_training,
                old_validation, new_training, new_validation, update.strategy,
                update.replaced_variable, certificate.to_json(), reward, duration,
                dict(self.discovery.last_search_audit),
            )

        revision_index = len(state.revision_history) + 1
        new_law = ActiveTheoryLaw(
            name=f"{old_law.name}_revision_{revision_index}",
            formula=new_formula,
            target=old_law.target,
            father=old_law.name,
            valid_experiments=list(experiment_names or [
                f"validation_{index}" for index in range(len(validation_data))
            ]),
        )
        state.active_laws.pop(old_law.name)
        state.active_laws[new_law.name] = new_law
        state.archived_laws[old_law.name] = {
            "formula": old_law.formula,
            "reason": "falsified_and_revised",
            "replaced_by": new_law.name,
            "contradiction_nrmse": max(item.nrmse for item in old_validation),
        }
        history_record = {
            "old_law": old_law.name,
            "new_law": new_law.name,
            "strategy": update.strategy,
            "replaced_variable": update.replaced_variable,
            "certificate": certificate.to_json(),
        }
        state.revision_history.append(history_record)
        reward = self._reward(
            contradicted=True,
            revised=True,
            old_error=max(item.nrmse for item in old_validation),
            new_error=max(item.nrmse for item in new_validation),
            validations=len(new_validation),
            old_complexity=self._complexity(old_law.formula, specification),
            new_complexity=self._complexity(new_formula, specification),
            redundant=False,
            duration=duration,
        )
        return TheoryRevisionOutcome(
            "revised", old_law, new_law, old_training, old_validation,
            new_training, new_validation, update.strategy,
            update.replaced_variable, certificate.to_json(), reward, duration,
            dict(self.discovery.last_search_audit),
        )


class GeneralLawRevisionAdapter:
    """Connect numerical theory repair to AI-Newton's real general-law basis.

    The controller deliberately remains independent of AI-Newton's DSL.  This
    adapter performs the missing boundary operations: recover the predictive
    form of an existing zero law, generalize the revised residual into a real
    ``Concept``/``Proposition``, commit through ``GeneralLawService``, and
    calculate the same adaptive reward consumed by experiment selection.
    """

    def __init__(self, theorist: Any, controller: TheoryRevisionController):
        self.theorist = theorist
        self.controller = controller

    def old_formula_from_general_law(
        self,
        law_name: str,
        experiment_name: str,
        specification: GenericSearchSpec,
    ) -> str:
        from .core import Exp

        law = self.theorist.general.general_laws[law_name]
        if law.type != "IsZero":
            raise ValueError("predictive revision currently requires an IsZero law")
        specialised = self.theorist.knowledge.specialize(
            law.relevant_concept, experiment_name
        )
        if len(specialised) != 1:
            raise ValueError(
                "predictive revision requires exactly one specialised expression"
            )
        model = self.theorist.specific[experiment_name]
        residual = sp.expand(
            model._sympy_of_raw_defi(model.expand_exp(specialised[0])).doit()
        )
        symbols = {
            expression: sp.Symbol(f"q{index}")
            for index, expression in enumerate(specification.scalars)
        }
        # Replace the most structured observations first.  ``subs`` works on
        # raw differential expressions, whereas textual replacement would be
        # unsafe for indexed variables such as x[1] and x[10].
        raw_observations: list[tuple[sp.Expr, sp.Symbol]] = []
        for expression, symbol in symbols.items():
            raw = model._sympy_of_raw_defi(
                model.expand_exp(Exp(expression))
            ).doit()
            raw_observations.append((raw, symbol))
        raw_observations.sort(
            key=lambda item: int(sp.count_ops(item[0], visual=False)),
            reverse=True,
        )
        symbolic_residual = residual
        for raw, symbol in raw_observations:
            symbolic_residual = symbolic_residual.subs(raw, symbol)
        target_symbol = symbols[specification.target]
        solutions = sp.solve(
            sp.Eq(symbolic_residual, 0), target_symbol,
            dict=False, simplify=False,
        )
        if not solutions:
            raise ValueError(
                f"cannot isolate target {specification.target} in {law_name}"
            )
        formula = str(solutions[0])
        for expression, symbol in sorted(
            symbols.items(), key=lambda item: len(str(item[1])), reverse=True
        ):
            formula = re_sub_symbol(formula, str(symbol), expression)
        # Reuse the controller's parser as a strict undeclared-symbol check.
        self.controller._sympy_formula(formula, specification)
        return formula

    def proposition_from_formula(
        self,
        formula: str,
        experiment_name: str,
        specification: GenericSearchSpec,
        observation_mapping: dict[str, str] | None = None,
    ):
        from .core import Exp, Proposition

        # ``Exp`` accepts the same generic operator syntax emitted by the SR
        # renderer (including sin/cos/exp/log and integer powers).
        mapping = observation_mapping or {}
        rendered_formula = formula
        for source in sorted(mapping, key=len, reverse=True):
            rendered_formula = rendered_formula.replace(
                source, f"({mapping[source]})"
            )
        rendered_target = mapping.get(
            specification.target, specification.target
        )
        residual = Exp(f"({rendered_target}) - ({rendered_formula})")
        concept = self.theorist.knowledge.generalize_to_normal_concept(
            experiment_name, residual
        )
        if concept is None:
            concept = self.theorist.knowledge.generalize(
                experiment_name, residual
            )
        if concept is None:
            raise ValueError(
                f"revised formula cannot be generalized in {experiment_name}: {formula}"
            )
        if not self.theorist.knowledge.specialize(concept, experiment_name):
            raise ValueError("generalized revision cannot be specialized back")
        return Proposition.Zero(concept)

    def revise_and_commit(
        self,
        law_name: str,
        experiment_name: str,
        training_data: DataStruct,
        validation_data: Sequence[DataStruct],
        specification: GenericSearchSpec,
        *,
        old_formula: str | None = None,
        strategies: Sequence[str] = (
            "additive", "multiplicative", "internal", "skeleton"
        ),
        evidence_names: Sequence[str] | None = None,
        scope_experiments: Sequence[str] | None = None,
        validate_against_specific_models: bool = True,
        observation_mapping: dict[str, str] | None = None,
        debug: bool = False,
    ) -> KnowledgeBaseRevisionOutcome:
        from .contextual_bandit import AdaptiveTheoryReward, TheoryStateEncoder

        before = TheoryStateEncoder.capture(self.theorist)
        real_law = self.theorist.general.general_laws[law_name]
        formula = old_formula or self.old_formula_from_general_law(
            law_name, experiment_name, specification
        )
        state = RevisionTheoryState.from_law(ActiveTheoryLaw(
            name=law_name,
            formula=formula,
            target=specification.target,
            father=real_law.father_name,
            valid_experiments=list(real_law.valid_experiments),
        ))
        search = self.controller.revise(
            state,
            law_name,
            training_data,
            validation_data,
            specification,
            strategies=strategies,
            experiment_names=evidence_names,
            debug=debug,
        )
        commit = None
        if search.status == "revised" and search.new_law is not None:
            proposition = self.proposition_from_formula(
                search.new_law.formula, experiment_name, specification,
                observation_mapping,
            )
            commit = self.theorist.general.service.replace_general_law_atomically(
                law_name,
                proposition,
                validation_experiments=scope_experiments or [experiment_name],
                contradiction_experiments=[experiment_name],
                validate_against_specific_models=validate_against_specific_models,
            )
        after = TheoryStateEncoder.capture(self.theorist)
        reward = AdaptiveTheoryReward().evaluate(
            before,
            after,
            duration_seconds=search.duration_seconds,
        )
        return KnowledgeBaseRevisionOutcome(
            search, commit, reward, before.state_hash, after.state_hash
        )


def re_sub_symbol(text: str, symbol: str, replacement: str) -> str:
    """Replace a generated scalar symbol without corrupting q1 inside q10."""

    import re
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                  f"({replacement})", text)
