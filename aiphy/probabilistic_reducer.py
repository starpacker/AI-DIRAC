"""Fast probabilistic tests for local functional independence of laws.

The expensive reducer in :mod:`aiphy.general_model` proves differential
consequences experiment by experiment.  That is still the final authority,
but it is a poor first line of defence against algebraic closure such as
``P``, ``P**2`` and ``P**3`` all being proposed as separate conserved laws.

This module implements the inexpensive part of a Sedoglavic-style workflow:

* expand DSL concepts to local SymPy jet expressions;
* regard functions and their derivatives at a generic time as jet variables;
* form the Kahler/Jacobian matrix of the existing quantities and candidate;
* compute generic ranks at several random points over large finite fields.

If adding a conserved quantity never increases the generic rank, it is a
locally algebraically dependent first integral and should be stored as a
derived theorem rather than as a new member of the theory basis.  A result of
``inconclusive`` deliberately falls back to the existing exact differential
algebra reducer.  Small rational zero constraints use a bounded exact
localized-radical ideal check.  Any generic nonzero chart assumptions are
stored in the certificate; ambiguous unions such as ``x*y == 0`` are not
collapsed to an arbitrary branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import random
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import sympy as sp
import numpy as np
from sympy.core.function import AppliedUndef


RankStatus = Literal["independent", "dependent", "inconclusive"]


@dataclass(frozen=True)
class ProbabilisticReducerConfig:
    """Configuration for the generic-rank oracle.

    Large prime fields make rank decisions exact after random specialization;
    randomness only affects whether a point accidentally lies on a vanishing
    minor.  Independent primes and points make that probability negligible.
    """

    primes: tuple[int, ...] = (1_000_000_007, 1_000_000_009, 998_244_353)
    trials_per_prime: int = 2
    max_basis_laws: int = 12
    # Maximum number of structural scope classes.  Experiments with the same
    # available basis, object signature and spatial dimension share a generic
    # local chart and are represented by the cheapest member of that class.
    max_experiments: int = 20
    seed: int = 20_260_716
    enabled: bool = True

    @staticmethod
    def from_json(data: Mapping[str, Any] | None) -> "ProbabilisticReducerConfig":
        if not data:
            return ProbabilisticReducerConfig()
        valid = ProbabilisticReducerConfig.__dataclass_fields__
        values = {key: value for key, value in data.items() if key in valid}
        if "primes" in values:
            values["primes"] = tuple(int(item) for item in values["primes"])
        return ProbabilisticReducerConfig(**values)


@dataclass
class RankCertificate:
    status: RankStatus
    method: str = "finite_field_local_kahler_rank"
    reason: str = ""
    basis_laws: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    ranks_before: list[int] = field(default_factory=list)
    ranks_after: list[int] = field(default_factory=list)
    trials: int = 0
    confidence_lower_bound: float = 0.0
    candidate_scope_size: int = 0
    candidate_experiments: list[str] = field(default_factory=list)
    independent_experiments: list[str] = field(default_factory=list)
    dependent_experiments: list[str] = field(default_factory=list)
    inconclusive_experiments: list[str] = field(default_factory=list)
    generic_nonzero_assumptions: list[str] = field(default_factory=list)

    @property
    def rank_gain(self) -> int:
        return max(
            (after - before for before, after in zip(self.ranks_before, self.ranks_after)),
            default=0,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class UnsupportedExpression(ValueError):
    """Raised when an expression is outside the rational jet fragment."""


def _localized_squarefree_numerator(
    expression: sp.Expr,
    variables: Sequence[sp.Symbol],
) -> tuple[sp.Expr, list[str], list[sp.Expr]]:
    """Return a conservative local radical representative of ``expr == 0``.

    Repeated factors do not change a zero set and are therefore made
    square-free.  A monomial factor is cancelled only when another,
    non-monomial factor remains.  This is the generic local chart used by
    Sedoglavic-style tests: for example ``y * (c - 1) == 0`` becomes
    ``c - 1 == 0`` under the explicit assumption ``y != 0``.  In contrast,
    ``x*y == 0`` is left intact because choosing either branch would be
    arbitrary.
    """

    numerator, denominator = sp.fraction(sp.cancel(expression))
    numerator = sp.expand(numerator)
    assumptions: list[str] = []
    localizers: list[sp.Expr] = []
    if denominator != 1:
        assumptions.append(f"({sp.sstr(denominator)}) != 0")
        localizers.append(denominator)
    if numerator == 0 or not variables:
        return numerator, assumptions, localizers
    try:
        coefficient, factors = sp.factor_list(numerator, *variables)
    except (sp.PolynomialError, TypeError, ValueError):
        return numerator, assumptions, localizers
    if not factors:
        return sp.sympify(coefficient), assumptions, localizers

    def is_monomial(factor: sp.Expr) -> bool:
        try:
            return len(sp.Poly(factor, *variables, domain=sp.QQ).terms()) == 1
        except (sp.PolynomialError, TypeError, ValueError):
            return False

    has_non_monomial_core = any(not is_monomial(factor) for factor, _ in factors)
    kept: list[sp.Expr] = []
    for factor, _multiplicity in factors:
        if has_non_monomial_core and is_monomial(factor):
            assumptions.append(f"({sp.sstr(factor)}) != 0")
            localizers.append(factor)
        else:
            # Multiplicity is deliberately discarded: radical(f**k) = radical(f).
            kept.append(factor)
    if not kept:
        # Defensive fallback.  The all-monomial case is intentionally not
        # localized, so this branch should only be reachable for constants.
        return numerator, assumptions, localizers
    return canonical_rational(sp.Mul(*kept)), assumptions, localizers


def compare_localized_zero_ideals(
    basis: Sequence[sp.Expr],
    candidate: Sequence[sp.Expr] | sp.Expr,
    *,
    max_variables: int = 64,
    max_terms: int = 512,
) -> RankCertificate:
    """Test local rational-ideal implication with exact arithmetic.

    This complements the Jacobian rank oracle for zero constraints.  It is
    deliberately bounded and returns ``inconclusive`` outside the small
    rational-polynomial fragment rather than invoking unbounded symbolic
    elimination.
    """

    if isinstance(candidate, sp.Expr):
        candidate = [candidate]
    if not candidate:
        return RankCertificate(status="inconclusive", reason="empty candidate")
    try:
        atomized, variables, base_state_symbols = _atomize_jets_with_base_symbols([
            *(canonical_rational(item) for item in basis),
            *(canonical_rational(item) for item in candidate),
        ])
        if len(variables) > max_variables:
            return RankCertificate(
                status="inconclusive",
                method="exact_localized_rational_ideal",
                reason=f"zero ideal has {len(variables)} variables (limit {max_variables})",
            )
        split = len(basis)
        normalized_basis: list[sp.Expr] = []
        normalized_candidate: list[sp.Expr] = []
        assumptions: list[str] = []
        localizers: list[sp.Expr] = []
        for index, expression in enumerate(atomized):
            normalized, local_assumptions, local_nonzero = _localized_squarefree_numerator(
                expression, variables
            )
            assumptions.extend(local_assumptions)
            localizers.extend(local_nonzero)
            target = normalized_basis if index < split else normalized_candidate
            if normalized != 0:
                target.append(normalized)
        # Sedoglavic's local calculation is performed at a generic physical
        # state.  For a non-monomial state relation such as PV-nRT=0, its
        # zero-order state coordinates are generically nonzero; explicitly
        # invert them so differential consequences may divide by n and R.
        # Do not apply this to monomial unions such as x*y=0, whose coordinate
        # branches must remain distinct.
        generic_state_localizers: set[sp.Symbol] = set()
        for normalized in normalized_basis:
            try:
                polynomial = sp.Poly(normalized, *variables, domain=sp.QQ)
                if len(polynomial.terms()) > 1:
                    generic_state_localizers.update(
                        normalized.free_symbols & base_state_symbols
                    )
            except (sp.PolynomialError, TypeError, ValueError):
                continue
        for symbol in sorted(generic_state_localizers, key=str):
            if symbol not in localizers:
                localizers.append(symbol)
                assumptions.append(f"({sp.sstr(symbol)}) != 0")
        term_count = sum(
            len(sp.Poly(item, *variables, domain=sp.QQ).terms())
            for item in [*normalized_basis, *normalized_candidate]
        ) if variables else 0
        if term_count > max_terms:
            return RankCertificate(
                status="inconclusive",
                method="exact_localized_rational_ideal",
                reason=f"zero ideal has {term_count} terms (limit {max_terms})",
                generic_nonzero_assumptions=sorted(set(assumptions)),
            )
        if not normalized_candidate:
            return RankCertificate(
                status="dependent",
                method="exact_localized_rational_ideal",
                reason="candidate is identically zero",
                trials=1,
                confidence_lower_bound=1.0,
                generic_nonzero_assumptions=sorted(set(assumptions)),
            )
        if not normalized_basis:
            return RankCertificate(
                status="independent",
                method="exact_localized_rational_ideal",
                reason="no local zero constraints in basis",
                trials=1,
                confidence_lower_bound=1.0,
                generic_nonzero_assumptions=sorted(set(assumptions)),
            )
        ideal_basis = list(normalized_basis)
        ideal_variables = list(variables)
        if localizers:
            # Membership in I localized at D is equivalent to membership in
            # <I, z*D-1> after adjoining one inverse variable.  Previously the
            # certificate merely *reported* D != 0 without imposing it, so
            # rational consequences such as d(nT/PV)=0 from PV=nRT and dR=0
            # were incorrectly declared independent.
            inverse = sp.Symbol("__localizer_inverse")
            while inverse in ideal_variables:
                inverse = sp.Symbol("_" + str(inverse))
            localizer_product = canonical_rational(sp.Mul(*localizers))
            ideal_basis.append(sp.expand(inverse * localizer_product - 1))
            ideal_variables.insert(0, inverse)
        ideal = sp.groebner(
            ideal_basis,
            *ideal_variables,
            order="grevlex",
            domain=sp.QQ,
        )
        remainders = [ideal.reduce(item)[1] for item in normalized_candidate]
        dependent = all(remainder == 0 for remainder in remainders)
        return RankCertificate(
            status="dependent" if dependent else "independent",
            method="exact_localized_rational_ideal",
            reason=(
                "candidate belongs to the localized radical ideal"
                if dependent else
                "candidate is not implied by the localized zero ideal"
            ),
            trials=1,
            confidence_lower_bound=1.0,
            generic_nonzero_assumptions=sorted(set(assumptions)),
        )
    except (sp.PolynomialError, TypeError, ValueError, ZeroDivisionError) as exc:
        return RankCertificate(
            status="inconclusive",
            method="exact_localized_rational_ideal",
            reason=f"unsupported zero ideal: {exc}",
        )


def _stable_seed(*parts: str, base: int) -> int:
    payload = "\x1f".join(parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return base ^ int.from_bytes(digest, "little")


def _atomize_jets_with_base_symbols(
    expressions: Sequence[sp.Expr],
) -> tuple[list[sp.Expr], list[sp.Symbol], set[sp.Symbol]]:
    """Atomize jets and retain which symbols represent zero-order states."""

    atoms: set[sp.Expr] = set()
    for expression in expressions:
        atoms.update(expression.atoms(sp.Derivative))
        atoms.update(expression.atoms(AppliedUndef))
    ordered_atoms = sorted(atoms, key=sp.srepr)
    replacements = {
        atom: sp.Symbol(f"__jet_{index}")
        for index, atom in enumerate(ordered_atoms)
    }
    atomized = [sp.sympify(expression).xreplace(replacements) for expression in expressions]
    variables = sorted(
        set().union(*(expression.free_symbols for expression in atomized)),
        key=str,
    )
    base_symbols = {
        replacements[atom]
        for atom in ordered_atoms
        if isinstance(atom, AppliedUndef)
    }
    return atomized, variables, base_symbols


def atomize_jets(expressions: Sequence[sp.Expr]) -> tuple[list[sp.Expr], list[sp.Symbol]]:
    """Replace applied functions and derivatives by deterministic jet symbols."""

    atomized, variables, _ = _atomize_jets_with_base_symbols(expressions)
    return atomized, variables


def canonical_rational(expression: sp.Expr) -> sp.Expr:
    """Return a conservative canonical form without aggressive expansion."""

    expression = sp.sympify(expression)
    if expression.has(sp.sin, sp.cos):
        try:
            expression = sp.trigsimp(expression)
        except Exception:
            pass
    try:
        return sp.factor_terms(sp.cancel(expression))
    except Exception:
        return expression


def rational_jet_signature(expressions: Sequence[sp.Expr]) -> str | None:
    """Semantic-preserving canonical signature for exact concept deduplication.

    Unlike the rank atomizer, jet symbols retain a digest of their original
    function/derivative name.  Thus ``vx`` and ``vy`` do not collide merely
    because each expression contains one jet variable.
    """

    expressions = [sp.sympify(item) for item in expressions]
    atoms: set[sp.Expr] = set()
    for expression in expressions:
        atoms.update(expression.atoms(sp.Derivative))
        atoms.update(expression.atoms(AppliedUndef))
    replacements = {}
    for atom in atoms:
        digest = hashlib.blake2b(sp.srepr(atom).encode("utf-8"), digest_size=10).hexdigest()
        replacements[atom] = sp.Symbol(f"__semantic_jet_{digest}")
    try:
        normalized = sorted(
            sp.srepr(canonical_rational(item.xreplace(replacements)))
            for item in expressions
        )
    except Exception:
        return None
    return hashlib.sha256("\x1e".join(normalized).encode("utf-8")).hexdigest()


def _modular_number(value: sp.Expr, prime: int) -> int:
    if value.is_Integer:
        return int(value) % prime
    if value.is_Rational:
        numerator = int(value.p) % prime
        denominator = int(value.q) % prime
        if denominator == 0:
            raise ZeroDivisionError
        return numerator * pow(denominator, prime - 2, prime) % prime
    if value.is_Float:
        rational = sp.Rational(str(value))
        return _modular_number(rational, prime)
    if value.is_Number:
        rational = sp.Rational(value)
        return _modular_number(rational, prime)
    raise UnsupportedExpression(f"Unsupported coefficient: {value!r}")


def evaluate_mod_prime(expression: sp.Expr, values: Mapping[sp.Symbol, int], prime: int) -> int:
    """Evaluate a rational SymPy expression exactly in ``GF(prime)``."""

    if expression.is_Number:
        return _modular_number(expression, prime)
    if expression.is_Symbol:
        try:
            return int(values[expression]) % prime
        except KeyError as exc:
            raise UnsupportedExpression(f"Missing value for {expression}") from exc
    if expression.is_Add:
        return sum(evaluate_mod_prime(arg, values, prime) for arg in expression.args) % prime
    if expression.is_Mul:
        result = 1
        for arg in expression.args:
            result = result * evaluate_mod_prime(arg, values, prime) % prime
        return result
    if expression.is_Pow:
        base, exponent = expression.args
        if not exponent.is_Integer:
            raise UnsupportedExpression(f"Non-integral power: {expression}")
        base_value = evaluate_mod_prime(base, values, prime)
        exponent_value = int(exponent)
        if exponent_value < 0:
            if base_value == 0:
                raise ZeroDivisionError
            base_value = pow(base_value, prime - 2, prime)
            exponent_value = -exponent_value
        return pow(base_value, exponent_value, prime)
    raise UnsupportedExpression(f"Non-rational operation: {expression.func}")


def rank_mod_prime(matrix: Sequence[Sequence[int]], prime: int) -> int:
    """Gaussian-elimination rank over a prime field."""

    if not matrix:
        return 0
    work = [[int(value) % prime for value in row] for row in matrix]
    rows = len(work)
    columns = len(work[0]) if rows else 0
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], prime - 2, prime)
        work[rank] = [(value * inverse) % prime for value in work[rank]]
        for row in range(rows):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                (left - factor * right) % prime
                for left, right in zip(work[row], work[rank])
            ]
        rank += 1
        if rank == rows:
            break
    return rank


class RandomizedKahlerRankOracle:
    """Generic local functional-independence oracle for rational jets."""

    def __init__(self, config: ProbabilisticReducerConfig | None = None):
        self.config = config or ProbabilisticReducerConfig()

    @staticmethod
    def _degree_bound(expressions: Sequence[sp.Expr], variables: Sequence[sp.Symbol]) -> int:
        degree = 1
        for expression in expressions:
            numerator, denominator = sp.fraction(sp.cancel(expression))
            for item in (numerator, denominator):
                try:
                    degree = max(degree, int(sp.Poly(item, *variables).total_degree()))
                except (sp.PolynomialError, TypeError, ValueError):
                    continue
        return max(1, degree * max(1, len(expressions)))

    def compare(
        self,
        basis: Sequence[sp.Expr],
        candidate: Sequence[sp.Expr] | sp.Expr,
        *,
        key: str = "",
    ) -> RankCertificate:
        """Compare generic Jacobian ranks before and after a candidate.

        ``dependent`` is returned only when every successful random trial has
        zero rank gain.  Any observed rank gain proves local independence for
        the sampled rational model.  Unsupported expressions are explicitly
        inconclusive so callers can use a symbolic fallback.
        """

        if isinstance(candidate, sp.Expr):
            candidate = [candidate]
        basis = [canonical_rational(item) for item in basis]
        candidate = [canonical_rational(item) for item in candidate]
        if not candidate:
            return RankCertificate(status="inconclusive", reason="empty candidate")

        all_expressions, variables = atomize_jets([*basis, *candidate])
        basis_count = len(basis)
        basis_atomized = all_expressions[:basis_count]
        candidate_atomized = all_expressions[basis_count:]
        if not variables:
            zero = all(sp.simplify(item) == 0 for item in candidate_atomized)
            return RankCertificate(
                status="dependent" if zero else "independent",
                reason="constant candidate",
                ranks_before=[0],
                ranks_after=[0 if zero else 1],
                trials=1,
                confidence_lower_bound=1.0,
            )

        try:
            jacobian_basis = [
                [canonical_rational(sp.diff(expression, variable)) for variable in variables]
                for expression in basis_atomized
            ]
            jacobian_candidate = [
                [canonical_rational(sp.diff(expression, variable)) for variable in variables]
                for expression in candidate_atomized
            ]
        except Exception as exc:
            return RankCertificate(status="inconclusive", reason=f"jacobian failed: {exc}")

        if any(
            entry.has(sp.sin, sp.cos, sp.exp, sp.log)
            or any(
                power.exp.is_integer is not True
                for power in entry.atoms(sp.Pow)
            )
            for row in [*jacobian_basis, *jacobian_candidate]
            for entry in row
        ):
            return self._compare_transcendental_numeric(
                jacobian_basis,
                jacobian_candidate,
                variables,
                key=key,
            )

        degree_bound = self._degree_bound(all_expressions, variables)
        rng = random.Random(_stable_seed(key, *(sp.srepr(x) for x in all_expressions), base=self.config.seed))
        ranks_before: list[int] = []
        ranks_after: list[int] = []
        attempted = 0
        successful = 0
        for prime in self.config.primes:
            for _ in range(max(1, self.config.trials_per_prime)):
                attempted += 1
                values = {variable: rng.randrange(1, prime) for variable in variables}
                try:
                    before_matrix = [
                        [evaluate_mod_prime(entry, values, prime) for entry in row]
                        for row in jacobian_basis
                    ]
                    candidate_matrix = [
                        [evaluate_mod_prime(entry, values, prime) for entry in row]
                        for row in jacobian_candidate
                    ]
                except ZeroDivisionError:
                    continue
                except UnsupportedExpression as exc:
                    return RankCertificate(
                        status="inconclusive",
                        reason=str(exc),
                        trials=successful,
                    )
                before_rank = rank_mod_prime(before_matrix, prime)
                after_rank = rank_mod_prime([*before_matrix, *candidate_matrix], prime)
                successful += 1
                ranks_before.append(before_rank)
                ranks_after.append(after_rank)
                if after_rank > before_rank:
                    return RankCertificate(
                        status="independent",
                        reason="candidate increased generic rank",
                        ranks_before=ranks_before,
                        ranks_after=ranks_after,
                        trials=successful,
                        confidence_lower_bound=1.0,
                    )

        if successful == 0:
            return RankCertificate(
                status="inconclusive",
                reason=f"all {attempted} random points hit a pole",
            )
        smallest_prime = min(self.config.primes)
        failure_bound = min(1.0, (degree_bound / smallest_prime) ** successful)
        return RankCertificate(
            status="dependent",
            reason="candidate did not increase generic rank",
            ranks_before=ranks_before,
            ranks_after=ranks_after,
            trials=successful,
            confidence_lower_bound=max(0.0, 1.0 - failure_bound),
        )

    def _compare_transcendental_numeric(
        self,
        jacobian_basis: Sequence[Sequence[sp.Expr]],
        jacobian_candidate: Sequence[Sequence[sp.Expr]],
        variables: Sequence[sp.Symbol],
        *,
        key: str,
    ) -> RankCertificate:
        """Deterministic floating-rank fallback for sin/cos/exp/log jets.

        Finite fields have no canonical transcendental evaluation.  We retain
        the same Kahler/Jacobian criterion but evaluate at several positive
        generic real points, keeping logarithms in-domain.  Any observed rank
        gain proves local independence; dependence requires agreement at all
        successful points and is explicitly labelled numeric in its
        certificate.
        """

        try:
            basis_functions = [
                [sp.lambdify(variables, entry, modules="numpy") for entry in row]
                for row in jacobian_basis
            ]
            candidate_functions = [
                [sp.lambdify(variables, entry, modules="numpy") for entry in row]
                for row in jacobian_candidate
            ]
        except Exception as exc:
            return RankCertificate(
                status="inconclusive",
                method="numeric_transcendental_kahler_rank",
                reason=f"transcendental lambdify failed: {exc}",
            )

        rng = random.Random(
            _stable_seed(
                key,
                *(sp.srepr(entry) for row in [*jacobian_basis, *jacobian_candidate] for entry in row),
                base=self.config.seed,
            )
        )
        ranks_before: list[int] = []
        ranks_after: list[int] = []
        requested_trials = max(6, len(self.config.primes) * self.config.trials_per_prime)
        for _ in range(requested_trials * 2):
            values = [rng.uniform(0.25, 1.75) for _variable in variables]
            try:
                before_matrix = np.asarray(
                    [[function(*values) for function in row] for row in basis_functions],
                    dtype=np.complex128,
                ).reshape((len(basis_functions), len(variables)))
                candidate_matrix = np.asarray(
                    [[function(*values) for function in row] for row in candidate_functions],
                    dtype=np.complex128,
                ).reshape((len(candidate_functions), len(variables)))
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                continue
            combined = np.vstack([before_matrix, candidate_matrix])
            if not np.all(np.isfinite(combined)):
                continue
            before_rank = int(np.linalg.matrix_rank(before_matrix, tol=1e-10))
            after_rank = int(np.linalg.matrix_rank(combined, tol=1e-10))
            ranks_before.append(before_rank)
            ranks_after.append(after_rank)
            if after_rank > before_rank:
                return RankCertificate(
                    status="independent",
                    method="numeric_transcendental_kahler_rank",
                    reason="candidate increased generic transcendental rank",
                    ranks_before=ranks_before,
                    ranks_after=ranks_after,
                    trials=len(ranks_before),
                    confidence_lower_bound=1.0,
                )
            if len(ranks_before) >= requested_trials:
                break
        if len(ranks_before) < requested_trials:
            return RankCertificate(
                status="inconclusive",
                method="numeric_transcendental_kahler_rank",
                reason="too few finite transcendental rank trials",
                ranks_before=ranks_before,
                ranks_after=ranks_after,
                trials=len(ranks_before),
            )
        return RankCertificate(
            status="dependent",
            method="numeric_transcendental_kahler_rank",
            reason="candidate did not increase generic transcendental rank",
            ranks_before=ranks_before,
            ranks_after=ranks_after,
            trials=len(ranks_before),
            confidence_lower_bound=0.999,
        )


class GeneralLawRankReducer:
    """Adapter from an AI-Newton ``GeneralLawBase`` to the rank oracle."""

    def __init__(self, general_law_base: Any,
                 config: ProbabilisticReducerConfig | None = None):
        self.general_law_base = general_law_base
        self.config = config or ProbabilisticReducerConfig()
        self.oracle = RandomizedKahlerRankOracle(self.config)
        self._complexity_cache: dict[str, tuple[int, int]] = {}

    def _raw_expressions(self, law_name: str, exp_name: str) -> list[sp.Expr]:
        base = self.general_law_base
        law = base.general_laws[law_name]
        model = base.specific[exp_name]
        specialised = base.knowledge.specialize(law.relevant_concept, exp_name)
        return [
            canonical_rational(
                model._sympy_of_raw_defi(model.expand_exp(expression)).doit()
            )
            for expression in specialised
        ]

    @staticmethod
    def _time_derivatives(expressions: Sequence[sp.Expr]) -> list[sp.Expr]:
        """Return total time derivatives in the local one-time jet chart.

        AI-Newton's specialised observations are functions of ``t_0``.  A
        conserved quantity ``f`` represents the differential constraint
        ``D_t(f) = 0``; keeping ``f`` itself would incorrectly treat the
        unknown integration constant as if its value were fixed.
        """

        derivatives: list[sp.Expr] = []
        for expression in expressions:
            time_symbols = sorted(
                (symbol for symbol in expression.free_symbols
                 if re.fullmatch(r"t(?:_\d+)?", str(symbol))),
                key=str,
            )
            if not time_symbols:
                derivatives.append(sp.Integer(0))
                continue
            derivatives.append(canonical_rational(sum(
                (sp.diff(expression, symbol) for symbol in time_symbols),
                sp.Integer(0),
            )))
        return derivatives

    def _constraint_expressions(
        self,
        law_name: str,
        exp_name: str,
        *,
        include_zero_derivatives: bool = True,
    ) -> list[sp.Expr]:
        """Expand a law into equations defining its local differential ideal."""

        raw = self._raw_expressions(law_name, exp_name)
        law = self.general_law_base.general_laws[law_name]
        if law.type == "IsConserved":
            return self._time_derivatives(raw)
        if law.type == "IsZero":
            if include_zero_derivatives:
                return [*raw, *self._time_derivatives(raw)]
            return raw
        raise UnsupportedExpression(f"unsupported law type: {law.type}")

    def expanded_complexity(self, law_name: str) -> tuple[int, int]:
        """Complexity of the expanded quantity, not merely its ``C_x`` atom."""

        if law_name in self._complexity_cache:
            return self._complexity_cache[law_name]
        law = self.general_law_base.general_laws[law_name]
        experiments = sorted(law.valid_experiments)
        try:
            expressions = self._raw_expressions(law_name, experiments[0])
            result = (
                sum(int(sp.count_ops(item, visual=False)) for item in expressions),
                sum(len(sp.srepr(item)) for item in expressions),
            )
        except Exception:
            result = (
                int(self.general_law_base.knowledge.raw_complexity(law.prop)),
                len(str(law.prop)),
            )
        self._complexity_cache[law_name] = result
        return result

    def priority_key(self, law_name: str) -> tuple[Any, ...]:
        law = self.general_law_base.general_laws[law_name]
        return (
            # A zero law fixes an algebraic value, whereas conservation only
            # fixes its time derivative.  Admit the stronger constraint first
            # so e.g. F-R=0 and D_t(R)=0 can make D_t(F)=0 derived, never the
            # other way around.
            0 if law.type == "IsZero" else 1,
            *self.expanded_complexity(law_name),
            -len(law.valid_experiments),
            law_name,
        )

    def relevant_objtypes(self, law_name: str) -> set[str]:
        """Return object types occurring in a law without unsafe atom walks."""

        law = self.general_law_base.general_laws[law_name]
        pending = list(set(re.findall(r"C_\d+", str(law.prop))))
        result: set[str] = set()
        visited: set[str] = set()
        try:
            result.update(str(item) for item in law.relevant_concept.objtype_id_map)
        except Exception:
            pass
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            try:
                concept = self.general_law_base.knowledge.fetch_concept_by_name(name)
            except Exception:
                concept = None
            if concept is None:
                continue
            result.update(str(item) for item in concept.objtype_id_map)
            pending.extend(
                child for child in re.findall(r"C_\d+", str(concept.exp))
                if child not in visited
            )
        return result

    def classify(self, law_name: str, basis_names: Iterable[str]) -> RankCertificate:
        """Classify a law, refining globally covered scopes only when needed."""

        return self._classify(
            law_name,
            list(basis_names),
            merge_local_basis=False,
        )

    def _classify(
        self,
        law_name: str,
        basis_names: Iterable[str],
        *,
        merge_local_basis: bool,
    ) -> RankCertificate:
        base = self.general_law_base
        law = base.general_laws[law_name]
        candidate_scope = set(law.valid_experiments)
        certificate = RankCertificate(
            status="inconclusive",
            candidate_scope_size=len(candidate_scope),
            candidate_experiments=sorted(candidate_scope),
        )
        if not self.config.enabled:
            certificate.reason = "probabilistic reducer disabled"
            return certificate
        if law.type not in {"IsConserved", "IsZero"}:
            certificate.reason = f"unsupported law type: {law.type}"
            return certificate
        if not candidate_scope:
            certificate.reason = "candidate has no validated scope"
            return certificate

        # Scope safety matters: f(P) may be redundant where P is known yet add
        # information in a different experiment.  Compute marginal rank in
        # every claimed experiment instead of making one all-or-nothing scope
        # comparison.
        candidate_complexity = self.expanded_complexity(law_name)
        candidate_objtypes = self.relevant_objtypes(law_name)
        eligible = [
            name for name in set(basis_names)
            if name != law_name
            and name in base.general_laws
            # Zero constraints may imply conservation after differentiation.
            # Conserved quantities cannot, however, fix the value of a zero
            # constraint, so an IsZero candidate only sees IsZero generators.
            and (
                base.general_laws[name].type == law.type
                or (
                    law.type == "IsConserved"
                    and base.general_laws[name].type == "IsZero"
                )
            )
            and base.general_laws[name].indication == law.indication
            and (
                self.expanded_complexity(name) <= candidate_complexity
                # A zero-order equation is strictly stronger than a
                # conservation statement.  It may therefore eliminate a
                # shorter conserved consequence and reduce total basis
                # cardinality even when its own syntax tree is larger.
                or (
                    law.type == "IsConserved"
                    and base.general_laws[name].type == "IsZero"
                )
            )
            and (
                not candidate_objtypes
                or not self.relevant_objtypes(name)
                or bool(candidate_objtypes & self.relevant_objtypes(name))
            )
        ]
        eligible.sort(key=self.priority_key)
        universal_basis = [
            name for name in eligible
            if candidate_scope <= set(base.general_laws[name].valid_experiments)
        ][: self.config.max_basis_laws]

        scope_classes: dict[tuple[Any, ...], list[str]] = {}
        local_basis_by_class: dict[tuple[Any, ...], list[str]] = {}
        relevant_types = candidate_objtypes
        for experiment in sorted(candidate_scope):
            experiment_basis = [
                name for name in eligible
                if experiment in base.general_laws[name].valid_experiments
            ]
            # A globally covering generator stabilizes exact-scope closure,
            # but it must not hide additional laws that are available in this
            # experiment.  Those laws determine the true marginal scope used
            # by the reward model.
            if universal_basis and not merge_local_basis:
                local_basis = universal_basis
            else:
                local_basis = sorted(
                    set(universal_basis) | set(experiment_basis),
                    key=self.priority_key,
                )[: self.config.max_basis_laws]
            try:
                structure = base.knowledge.fetch_expstruct(experiment)
                object_signature = tuple(sorted(
                    str(obj_type) for obj_type, _ in structure.obj_info.values()
                    if not relevant_types or str(obj_type) in relevant_types
                ))
                spatial_dimension = int(structure.spdim)
            except Exception:
                object_signature = ()
                spatial_dimension = -1
            class_key = (tuple(local_basis), object_signature, spatial_dimension)
            scope_classes.setdefault(class_key, []).append(experiment)
            local_basis_by_class[class_key] = local_basis

        ordered_classes = sorted(
            scope_classes.items(),
            key=lambda item: (
                min(
                    len(base.knowledge.fetch_expstruct(name).original_data)
                    for name in item[1]
                ),
                item[1][0],
            ),
        )
        empty_basis_classes = [
            item for item in ordered_classes
            if not local_basis_by_class[item[0]]
        ]
        rank_test_classes = [
            item for item in ordered_classes
            if local_basis_by_class[item[0]]
        ]
        selected_classes = rank_test_classes[: self.config.max_experiments]
        skipped_classes = rank_test_classes[self.config.max_experiments:]
        aggregate_before: list[int] = []
        aggregate_after: list[int] = []
        confidence = 1.0
        tested: list[str] = []
        used_basis: set[str] = set()
        independent_experiments: list[str] = []
        dependent_experiments: list[str] = []
        inconclusive_experiments: list[str] = []
        generic_nonzero_assumptions: set[str] = set()
        used_methods: set[str] = set()
        # Even with an empty basis, expand and classify the candidate.  The
        # previous shortcut labelled an identically-zero expression as an
        # independent generator merely because there was no prior law.
        for _, class_experiments in empty_basis_classes:
            experiment = min(
                class_experiments,
                key=lambda name: (
                    len(base.knowledge.fetch_expstruct(name).original_data),
                    name,
                ),
            )
            try:
                candidate_exprs = self._constraint_expressions(
                    law_name,
                    experiment,
                    include_zero_derivatives=False,
                )
                if law.type == "IsZero":
                    result = compare_localized_zero_ideals([], candidate_exprs)
                    if result.status == "inconclusive":
                        result = self.oracle.compare(
                            [], candidate_exprs,
                            key=f"{law_name}:{experiment}:zero-transcendental",
                        )
                else:
                    result = compare_localized_zero_ideals([], candidate_exprs)
            except Exception as exc:
                certificate.reason = f"failed to expand {experiment}: {exc}"
                inconclusive_experiments.extend(class_experiments)
                continue
            generic_nonzero_assumptions.update(
                result.generic_nonzero_assumptions
            )
            used_methods.add(result.method)
            if result.status == "inconclusive":
                certificate.reason = f"{experiment}: {result.reason}"
                inconclusive_experiments.extend(class_experiments)
                continue
            tested.append(experiment)
            aggregate_before.extend(result.ranks_before)
            aggregate_after.extend(result.ranks_after)
            confidence = min(confidence, result.confidence_lower_bound)
            if result.status == "independent":
                independent_experiments.extend(class_experiments)
            else:
                dependent_experiments.extend(class_experiments)
        for class_key, class_experiments in selected_classes:
            local_basis = local_basis_by_class[class_key]
            used_basis.update(local_basis)
            experiment = min(
                class_experiments,
                key=lambda name: (
                    len(base.knowledge.fetch_expstruct(name).original_data),
                    name,
                ),
            )
            try:
                candidate_exprs = self._constraint_expressions(
                    law_name,
                    experiment,
                    include_zero_derivatives=False,
                )
                basis_exprs = [
                    expression
                    for basis_name in local_basis
                    for expression in self._constraint_expressions(
                        basis_name,
                        experiment,
                        include_zero_derivatives=True,
                    )
                ]
            except Exception as exc:
                certificate.reason = f"failed to expand {experiment}: {exc}"
                inconclusive_experiments.extend(class_experiments)
                continue
            if law.type == "IsZero":
                result = compare_localized_zero_ideals(basis_exprs, candidate_exprs)
                if result.status == "inconclusive":
                    result = self.oracle.compare(
                        basis_exprs,
                        candidate_exprs,
                        key=f"{law_name}:{experiment}:zero-transcendental",
                    )
            else:
                result = compare_localized_zero_ideals(
                    basis_exprs,
                    candidate_exprs,
                )
                # Exact localized ideals deliberately support only the
                # rational fragment.  Preserve the old functional-rank path
                # for all-conserved transcendental closures such as sin(f).
                if (
                    result.status == "inconclusive"
                    and all(
                        base.general_laws[name].type == "IsConserved"
                        for name in local_basis
                    )
                ):
                    result = self.oracle.compare(
                        [
                            expression
                            for basis_name in local_basis
                            for expression in self._raw_expressions(
                                basis_name, experiment
                            )
                        ],
                        self._raw_expressions(law_name, experiment),
                        key=f"{law_name}:{experiment}:transcendental",
                    )
            generic_nonzero_assumptions.update(result.generic_nonzero_assumptions)
            used_methods.add(result.method)
            if result.status == "inconclusive":
                certificate.reason = f"{experiment}: {result.reason}"
                inconclusive_experiments.extend(class_experiments)
                continue
            tested.append(experiment)
            aggregate_before.extend(result.ranks_before)
            aggregate_after.extend(result.ranks_after)
            confidence = min(confidence, result.confidence_lower_bound)
            if result.status == "independent":
                independent_experiments.extend(class_experiments)
            else:
                dependent_experiments.extend(class_experiments)

        for _, class_experiments in skipped_classes:
            inconclusive_experiments.extend(class_experiments)

        common = dict(
            basis_laws=sorted(used_basis, key=self.priority_key),
            experiments=tested,
            ranks_before=aggregate_before,
            ranks_after=aggregate_after,
            trials=len(aggregate_before),
            confidence_lower_bound=confidence if tested else 0.0,
            candidate_scope_size=len(candidate_scope),
            candidate_experiments=sorted(candidate_scope),
            independent_experiments=sorted(independent_experiments),
            dependent_experiments=sorted(dependent_experiments),
            inconclusive_experiments=sorted(inconclusive_experiments),
            generic_nonzero_assumptions=sorted(generic_nonzero_assumptions),
            method=(
                "+".join(sorted(method for method in used_methods if method))
                or (
                    "exact_localized_rational_ideal"
                    if law.type == "IsZero" else
                    "finite_field_local_kahler_rank"
                )
            ),
        )
        if independent_experiments:
            # A full-scope basis is the cheapest way to reject obvious
            # closure.  If it does not suffice, repeat with all laws available
            # in each experiment so reward/maturity see the true marginal
            # scope rather than an overestimate.
            if universal_basis and not merge_local_basis:
                return self._classify(
                    law_name,
                    basis_names,
                    merge_local_basis=True,
                )
            return RankCertificate(
                status="independent",
                reason=(
                    f"local information increased in {len(independent_experiments)}/"
                    f"{len(candidate_scope)} scope experiments"
                ),
                **common,
            )
        if inconclusive_experiments:
            return RankCertificate(
                status="inconclusive",
                reason=(
                    f"no rank gain observed; {len(inconclusive_experiments)} "
                    "scope experiments were inconclusive"
                ),
                **common,
            )
        return RankCertificate(
            status="dependent",
            reason="no generic rank gain anywhere in the candidate scope",
            **common,
        )
