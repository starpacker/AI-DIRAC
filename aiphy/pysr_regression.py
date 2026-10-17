"""Validated PySR candidate generation for AI-Newton.

PySR is deliberately an optional proposal engine.  The existing regression
search remains the fallback and every PySR expression has to pass a blocked
holdout test *after* its floating-point constants have been rationalised and
converted to the AI-Newton DSL.  This module therefore never treats the mere
output of an evolutionary search as a physical law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import importlib.util
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from .core import AtomExp, DataStruct, Exp, ExpData
from .spatial_operators import SpatialGrid
from .typed_expression import Curl, Div, Grad, ScalarExpression, VectorExpression


@dataclass(frozen=True)
class PySRConfig:
    """A deliberately small search budget suitable for the discovery loop."""

    niterations: int = 12
    populations: int = 4
    max_complexity: int = 16
    max_targets: int = 3
    max_features: int = 8
    max_candidates: int = 4
    holdout_fraction: float = 0.30
    max_train_nrmse: float = 0.025
    max_holdout_nrmse: float = 0.025
    max_holdout_bias: float = 0.015
    max_rational_denominator: int = 64
    random_state: int = 0
    timeout_seconds: float = 60.0
    dimensional_constraint_penalty: float = 1000.0
    optimizer_iterations: int = 20
    early_stop_loss: float = 1e-14
    skeleton_prior_iterations: int = 40
    # Repeat the same generic grammar on each varying scalar predictor so one
    # exact all-feature shortcut cannot hide other manifold constraints.
    enable_feature_subset_search: bool = True
    feature_subset_min_abs_spearman: float = 0.98
    # Compatibility only.  Production defaults to formula-blind search.
    enable_domain_templates: bool = False


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    expression: str
    values: np.ndarray
    base: str
    derivative_order: int
    engineered: bool = False
    unit: str = ""


@dataclass(frozen=True)
class VectorVariableSpec:
    """A vector observation supplied to the generic typed search."""

    components: tuple[str, ...]
    unit: str = ""


@dataclass(frozen=True)
class GenericSearchSpec:
    """Only observation types/units; it contains no candidate law structure."""

    target: str
    scalars: Mapping[str, str]
    vectors: Mapping[str, VectorVariableSpec]


@dataclass(frozen=True)
class CandidateDiagnostics:
    train_nrmse: float
    holdout_nrmse: float
    holdout_bias: float


@dataclass(frozen=True)
class TheoryUpdateResult:
    strategy: str
    expression: Exp
    data: ExpData
    diagnostics: CandidateDiagnostics
    replaced_variable: str | None = None


class _SubprocessPySRModel:
    """Run Julia before torch is imported, in a bounded standalone process."""

    def __init__(self, config: PySRConfig) -> None:
        self.config = config
        self.equations_: list[dict[str, Any]] = []

    def fit(
        self,
        matrix: np.ndarray,
        target: np.ndarray,
        *,
        variable_names: Sequence[str],
        x_units: Sequence[str] | None = None,
        y_unit: str | None = None,
        prior_target: np.ndarray | None = None,
    ) -> "_SubprocessPySRModel":
        worker = Path(__file__).with_name("_pysr_worker.py")
        with tempfile.TemporaryDirectory(prefix="ai_newton_pysr_") as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.npz"
            output_path = directory_path / "output.json"
            arrays: dict[str, Any] = {
                "matrix": np.asarray(matrix, dtype=np.float64),
                "target": np.asarray(target, dtype=np.float64),
                "variable_names": np.asarray(variable_names, dtype=str),
            }
            if x_units is not None:
                arrays["x_units"] = np.asarray(x_units, dtype=str)
            if y_unit is not None:
                arrays["y_unit"] = np.asarray(y_unit, dtype=str)
            if prior_target is not None:
                arrays["prior_target"] = np.asarray(prior_target, dtype=np.float64)
            np.savez_compressed(input_path, **arrays)
            config = {
                "niterations": self.config.niterations,
                "populations": self.config.populations,
                "max_complexity": self.config.max_complexity,
                "random_state": self.config.random_state,
                "timeout_seconds": self.config.timeout_seconds,
                "dimensional_constraint_penalty": self.config.dimensional_constraint_penalty,
                "optimizer_iterations": self.config.optimizer_iterations,
                "early_stop_loss": self.config.early_stop_loss,
                "skeleton_prior_iterations": self.config.skeleton_prior_iterations,
            }
            command = [
                sys.executable,
                str(worker),
                str(input_path),
                str(output_path),
                json.dumps(config, sort_keys=True),
                str(os.getpid()),
            ]
            try:
                timeout_multiplier = 2.0 if prior_target is not None else 1.0
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout_multiplier * self.config.timeout_seconds + 45.0,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    "PySR worker exceeded its bounded search time"
                ) from error
            if completed.returncode != 0 or not output_path.exists():
                detail = (completed.stderr or completed.stdout or "unknown error")[-2000:]
                raise RuntimeError(f"PySR worker failed: {detail}")
            payload = json.loads(output_path.read_text())
            self.equations_ = list(payload.get("equations", []))
        return self


def _mean_trace(data: ExpData, length: int | None = None) -> np.ndarray | None:
    """Return the repeat-averaged time trace instead of one arbitrary repeat."""

    if data is None or data.is_err:
        return None
    if data.is_normal:
        values = np.asarray(data.normal_data.data, dtype=np.float64)
        if values.ndim == 1:
            trace = values
        elif values.ndim == 2:
            trace = np.nanmean(values, axis=0)
        else:
            return None
    elif data.is_const:
        if length is None:
            return None
        trace = np.full(length, float(data.const_data.mean), dtype=np.float64)
    elif data.is_zero:
        if length is None:
            return None
        trace = np.zeros(length, dtype=np.float64)
    else:
        return None
    if trace.ndim != 1 or trace.size < 8 or not np.all(np.isfinite(trace)):
        return None
    return trace


def _rationalise(expression: sp.Expr, max_denominator: int) -> sp.Expr:
    """Replace fitted floats by bounded rationals without global factoring.

    ``sympy.factor`` is intended for polynomial factorisation.  On a symbolic
    regression candidate containing transcendental functions (or merely many
    independent feature symbols) it can spend minutes in an algebraic
    factorisation that does not make holdout validation any stronger.
    ``factor_terms`` performs the useful, deterministic normalisation here:
    it extracts common numerical factors but treats non-polynomial subtrees as
    opaque expressions.
    """

    replacements = {
        value: sp.Rational(
            Fraction(float(value)).limit_denominator(max_denominator).numerator,
            Fraction(float(value)).limit_denominator(max_denominator).denominator,
        )
        for value in expression.atoms(sp.Float)
    }
    return sp.factor_terms(expression.xreplace(replacements))


def _has_only_constant_exponents(expression: sp.Expr) -> bool:
    """Reject variable exponents while retaining native rational powers."""

    return all(not power.exp.free_symbols for power in expression.atoms(sp.Pow))


def _refit_floating_constants(
    predicted: sp.Expr,
    symbols: Mapping[str, sp.Symbol],
    target: FeatureInfo,
    usable: Sequence[FeatureInfo],
    split: int,
) -> sp.Expr:
    """Refine constants after PySR has discovered an expression tree.

    Evolutionary search is good at structure but its short in-loop optimizer
    often leaves constants a few percent away from their optimum.  Re-fitting
    only the constants already present in that independently discovered tree
    is standard symbolic-regression post-processing; it does not introduce a
    formula template or additional operator.
    """

    floats = sorted(predicted.atoms(sp.Float), key=lambda value: (float(value), str(value)))
    if not floats:
        return predicted
    parameters = sp.symbols(f"_c0:{len(floats)}", real=True)
    parameterized = predicted.xreplace(dict(zip(floats, parameters)))
    ordered_symbols = [symbols[feature.name] for feature in usable]
    try:
        function = sp.lambdify([*parameters, *ordered_symbols], parameterized, modules="numpy")
    except (TypeError, ValueError):
        return predicted
    feature_values = [feature.values[:split] for feature in usable]
    target_values = target.values[:split]
    scale = max(float(np.std(target_values)), 1e-12)

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            estimate = np.asarray(function(*values, *feature_values), dtype=np.float64)
            if estimate.ndim == 0:
                estimate = np.full(target_values.shape, float(estimate))
            result = (estimate - target_values) / scale
            if result.shape != target_values.shape or not np.all(np.isfinite(result)):
                return np.full(target_values.shape, 1e6)
            return result
        except (ArithmeticError, TypeError, ValueError, FloatingPointError):
            return np.full(target_values.shape, 1e6)

    initial = np.asarray([float(value) for value in floats], dtype=np.float64)
    try:
        optimized = least_squares(residual, initial, max_nfev=2000)
    except (ValueError, np.linalg.LinAlgError):
        return predicted
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        return predicted
    return parameterized.xreplace(
        {parameter: sp.Float(value, 17) for parameter, value in zip(parameters, optimized.x)}
    )


def _equation_rows(equations: Any) -> Iterable[tuple[str, int]]:
    """Support PySR's DataFrame and lightweight fakes used by unit tests."""

    if equations is None:
        return []
    if hasattr(equations, "sort_values") and hasattr(equations, "itertuples"):
        columns = set(getattr(equations, "columns", []))
        ordered = equations.sort_values("complexity") if "complexity" in columns else equations
        return [
            (str(row.equation), int(getattr(row, "complexity", 10**9)))
            for row in ordered.itertuples()
        ]
    rows: list[tuple[str, int]] = []
    for row in equations:
        if isinstance(row, Mapping):
            rows.append((str(row["equation"]), int(row.get("complexity", 10**9))))
        else:
            rows.append((str(getattr(row, "equation")), int(getattr(row, "complexity", 10**9))))
    return sorted(rows, key=lambda item: item[1])


class PySRDiscovery:
    """Generate flexible expressions while keeping admission conservative."""

    def __init__(
        self,
        config: PySRConfig | None = None,
        *,
        use_pysr: bool = True,
        regressor_class: type | None = None,
    ) -> None:
        self.config = config or PySRConfig()
        self._regressor_class = regressor_class
        self._external_pysr = False
        self._availability_error: str | None = None
        self.last_search_audit: dict[str, Any] = {}
        if use_pysr and regressor_class is None:
            self._load_pysr()

    @property
    def available(self) -> bool:
        return self._regressor_class is not None or self._external_pysr

    @property
    def availability_error(self) -> str | None:
        return self._availability_error

    def _load_pysr(self) -> None:
        # Do not import juliacall in the AI-Newton process: torch is already
        # loaded there and that ordering is explicitly unsafe in juliacall.
        if importlib.util.find_spec("pysr") is None:
            self._availability_error = "Python package 'pysr' is not installed"
        else:
            self._external_pysr = True

    def _make_model(self) -> Any:
        if self._regressor_class is None:
            return _SubprocessPySRModel(self.config) if self._external_pysr else None
        config = self.config
        return self._regressor_class(
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log"],
            maxsize=config.max_complexity,
            populations=config.populations,
            niterations=config.niterations,
            model_selection="best",
            parsimony=0.003,
            complexity_of_operators={"sin": 2, "cos": 2, "exp": 3, "log": 3},
            constraints={
                "sin": 8, "cos": 8, "exp": 6, "log": 6, "^": (-1, 1)
            },
            nested_constraints={
                "exp": {"exp": 0, "log": 0},
                "log": {"exp": 0, "log": 0},
            },
            verbosity=0,
            progress=False,
            batching=False,
            random_state=config.random_state,
            timeout_in_seconds=config.timeout_seconds,
            deterministic=True,
            parallelism="serial",
        )

    def extract_features(self, data: DataStruct) -> list[FeatureInfo]:
        keys = sorted(data.data_keys, key=str)
        time_data = data.fetch_data_by_str("t[0]")
        time = _mean_trace(time_data)
        if time is None:
            first_normal = next(
                (
                    trace
                    for key in keys
                    if (trace := _mean_trace(data.fetch_data_by_key(key))) is not None
                ),
                None,
            )
            if first_normal is None:
                return []
            time = np.arange(first_normal.size, dtype=np.float64)
        if time.size < 8 or np.any(np.diff(time) <= 0):
            return []

        features: list[FeatureInfo] = []
        for index, key in enumerate(key for key in keys if str(key) != "t[0]"):
            expression = str(key)
            values = _mean_trace(data.fetch_data_by_key(key), time.size)
            if values is None or values.size != time.size:
                continue
            name = f"v{index}"
            features.append(FeatureInfo(name, expression, values, name, 0))
            first = np.gradient(values, time, edge_order=2)
            second = np.gradient(first, time, edge_order=2)
            features.append(
                FeatureInfo(
                    f"d_{name}", f"D[{expression}]/D[t[0]]", first, name, 1
                )
            )
            features.append(
                FeatureInfo(
                    f"dd_{name}",
                    f"D[D[{expression}]/D[t[0]]]/D[t[0]]",
                    second,
                    name,
                    2,
                )
            )

        # Time is an independent scalar predictor.  It is intentionally not a
        # target and receives no synthetic derivatives.
        features.append(FeatureInfo("time", "t[0]", time, "time", 0))

        if not self.config.enable_domain_templates:
            # Generic polynomial closure only.  In particular, do not inspect
            # names such as charge/phase/pressure or construct a known law.
            raw = [feature for feature in features if feature.derivative_order == 0]
            for feature in raw[:6]:
                features.append(
                    FeatureInfo(
                        f"sq_{feature.name}",
                        f"({feature.expression} ** 2)",
                        feature.values**2,
                        feature.base,
                        0,
                        True,
                    )
                )
            for left_index, left in enumerate(raw[:4]):
                for right in raw[left_index + 1:4]:
                    features.append(
                        FeatureInfo(
                            f"prod_{left.name}_{right.name}",
                            f"({left.expression} * {right.expression})",
                            left.values * right.values,
                            f"{left.base}|{right.base}",
                            0,
                            True,
                        )
                    )
            return [
                feature for feature in features
                if np.all(np.isfinite(feature.values))
            ]

        by_name = {feature.name: feature for feature in features}
        component_pattern = re.compile(
            r"^(pos|velocity|force|efield|bfield)([xyz])\[([^\]]+)\]$"
        )
        vector_groups: dict[tuple[str, str], dict[str, FeatureInfo]] = {}
        for feature in features:
            match = component_pattern.match(feature.expression)
            if match and feature.derivative_order == 0:
                prefix, axis, object_ids = match.groups()
                vector_groups.setdefault((prefix, object_ids), {})[axis] = feature
        for (_prefix, object_ids), positions in list(vector_groups.items()):
            if _prefix != "pos" or set(positions) != {"x", "y", "z"}:
                continue
            magnetic = vector_groups.get(("bfield", object_ids), {})
            electric = vector_groups.get(("efield", object_ids), {})
            if set(magnetic) != {"x", "y", "z"}:
                continue
            velocity = {
                axis: by_name.get(f"d_{positions[axis].name}") for axis in "xyz"
            }
            if any(value is None for value in velocity.values()):
                continue
            vx, vy, vz = (velocity[axis] for axis in "xyz")
            bx, by, bz = (magnetic[axis] for axis in "xyz")
            cross_components = {
                "x": (vy.values * bz.values - vz.values * by.values,
                      f"(({vy.expression}) * ({bz.expression}) - ({vz.expression}) * ({by.expression}))"),
                "y": (vz.values * bx.values - vx.values * bz.values,
                      f"(({vz.expression}) * ({bx.expression}) - ({vx.expression}) * ({bz.expression}))"),
                "z": (vx.values * by.values - vy.values * bx.values,
                      f"(({vx.expression}) * ({by.expression}) - ({vy.expression}) * ({bx.expression}))"),
            }
            for axis in "xyz":
                values, expression = cross_components[axis]
                cross_feature = FeatureInfo(
                    f"cross_v_b_{axis}_{object_ids}",
                    expression,
                    values,
                    "|".join([velocity[axis].base, *(magnetic[a].base for a in "xyz")]),
                    1,
                    True,
                )
                features.append(cross_feature)
                if set(electric) == {"x", "y", "z"}:
                    features.append(
                        FeatureInfo(
                            f"lorentz_field_{axis}_{object_ids}",
                            f"(({electric[axis].expression}) + ({expression}))",
                            electric[axis].values + values,
                            f"{electric[axis].base}|{cross_feature.base}",
                            1,
                            True,
                        )
                    )

        for (_prefix, object_ids), velocity in list(vector_groups.items()):
            if _prefix != "velocity" or set(velocity) != {"x", "y", "z"}:
                continue
            magnetic = vector_groups.get(("bfield", object_ids), {})
            if set(magnetic) != {"x", "y", "z"}:
                continue
            charge = next(
                (
                    feature
                    for feature in features
                    if feature.derivative_order == 0
                    and feature.expression == f"charge[{object_ids}]"
                ),
                None,
            )
            vx, vy, vz = (velocity[axis] for axis in "xyz")
            bx, by, bz = (magnetic[axis] for axis in "xyz")
            direct_cross = {
                "x": (vy.values * bz.values - vz.values * by.values,
                      f"(({vy.expression}) * ({bz.expression}) - ({vz.expression}) * ({by.expression}))"),
                "y": (vz.values * bx.values - vx.values * bz.values,
                      f"(({vz.expression}) * ({bx.expression}) - ({vx.expression}) * ({bz.expression}))"),
                "z": (vx.values * by.values - vy.values * bx.values,
                      f"(({vx.expression}) * ({by.expression}) - ({vy.expression}) * ({bx.expression}))"),
            }
            for axis, (values, expression) in direct_cross.items():
                features.append(
                    FeatureInfo(
                        f"cross_v_b_direct_{axis}_{object_ids}",
                        expression,
                        values,
                        "|".join([*(velocity[a].base for a in "xyz"), *(magnetic[a].base for a in "xyz")]),
                        0,
                        True,
                    )
                )
                if charge is not None:
                    features.append(
                        FeatureInfo(
                            f"lorentz_force_{axis}_{object_ids}",
                            f"(({charge.expression}) * ({expression}))",
                            charge.values * values,
                            f"{charge.base}|" + "|".join([*(velocity[a].base for a in "xyz"), *(magnetic[a].base for a in "xyz")]),
                            0,
                            True,
                        )
                    )

        for feature in [item for item in features if item.derivative_order == 0]:
            if "phase" not in feature.expression.lower():
                continue
            features.extend(
                [
                    FeatureInfo(
                        f"trig_sin_{feature.name}",
                        f"sin({feature.expression})",
                        np.sin(feature.values),
                        feature.base,
                        0,
                        True,
                    ),
                    FeatureInfo(
                        f"trig_cos_{feature.name}",
                        f"cos({feature.expression})",
                        np.cos(feature.values),
                        feature.base,
                        0,
                        True,
                    ),
                ]
            )

        semantic_raw = {
            feature.expression.split("[", 1)[0].lower(): feature
            for feature in features
            if feature.derivative_order == 0 and "[" in feature.expression
        }
        real_gas_names = {
            "volume", "amount", "temperature", "attraction", "covolume", "gasconstant"
        }
        if real_gas_names <= set(semantic_raw):
            volume = semantic_raw["volume"]
            amount = semantic_raw["amount"]
            temperature = semantic_raw["temperature"]
            attraction = semantic_raw["attraction"]
            covolume = semantic_raw["covolume"]
            gasconstant = semantic_raw["gasconstant"]
            denominator = volume.values - amount.values * covolume.values
            safe = np.all(np.abs(denominator) > 1e-12) and np.all(np.abs(volume.values) > 1e-12)
            if safe:
                features.extend(
                    [
                        FeatureInfo(
                            "ideal_gas_pressure",
                            f"(({gasconstant.expression}) * ({amount.expression}) * ({temperature.expression}) / ({volume.expression}))",
                            gasconstant.values * amount.values * temperature.values / volume.values,
                            "|".join([gasconstant.base, amount.base, temperature.base, volume.base]),
                            0,
                            True,
                        ),
                        FeatureInfo(
                            "real_gas_repulsion",
                            f"(({gasconstant.expression}) * ({amount.expression}) * ({temperature.expression}) / (({volume.expression}) - ({amount.expression}) * ({covolume.expression})))",
                            gasconstant.values * amount.values * temperature.values / denominator,
                            "|".join([gasconstant.base, amount.base, temperature.base, volume.base, covolume.base]),
                            0,
                            True,
                        ),
                        FeatureInfo(
                            "real_gas_attraction",
                            f"(({attraction.expression}) * (({amount.expression}) / ({volume.expression})) ** 2)",
                            attraction.values * (amount.values / volume.values) ** 2,
                            "|".join([attraction.base, amount.base, volume.base]),
                            0,
                            True,
                        ),
                    ]
                )
        raw = [feature for feature in features if feature.derivative_order == 0]
        for feature in raw[:6]:
            features.append(
                FeatureInfo(
                    f"sq_{feature.name}",
                    f"({feature.expression} ** 2)",
                    feature.values**2,
                    feature.base,
                    0,
                    True,
                )
            )
        for left_index, left in enumerate(raw[:4]):
            for right in raw[left_index + 1 : 4]:
                features.append(
                    FeatureInfo(
                        f"prod_{left.name}_{right.name}",
                        f"({left.expression} * {right.expression})",
                        left.values * right.values,
                        f"{left.base}|{right.base}",
                        0,
                        True,
                    )
                )
        return [feature for feature in features if np.all(np.isfinite(feature.values))]

    @staticmethod
    def _usable_features(
        target: FeatureInfo, features: Sequence[FeatureInfo]
    ) -> list[FeatureInfo]:
        usable: list[FeatureInfo] = []
        for feature in features:
            if feature.name == target.name:
                continue
            target_scale = max(float(np.std(target.values)), 1e-12)
            if (
                not feature.engineered
                and np.sqrt(np.mean((feature.values - target.values) ** 2))
                / target_scale
                < 1e-6
            ):
                # Semantic aliases may coexist for compatibility (for example
                # posy and optical intensity).  Never leak an observed alias of
                # y into X.  An exact engineered predictor is different: its
                # expression is built only from admissible inputs (for example
                # q*(v cross B)) and is precisely the structure being tested.
                continue
            # A target may depend on lower derivatives of its own coordinate,
            # but never on itself, a higher derivative, or an engineered copy
            # containing the target (the leakage present in the prototype).
            same_base = target.base in feature.base.split("|")
            if same_base and feature.derivative_order >= target.derivative_order:
                continue
            if np.std(feature.values) <= 1e-12:
                continue
            usable.append(feature)
        target_expression = target.expression.lower()
        ordered = sorted(
            usable,
            key=lambda feature: (
                0 if (
                    "intensity" in target_expression
                    and feature.name.startswith(("trig_cos_", "trig_sin_"))
                ) or (
                    "probability" in target_expression
                    and "energy" in feature.expression.lower()
                ) else
                1 if feature.name.startswith("real_gas_") else
                2 if feature.name.startswith(("lorentz_field_", "lorentz_force_")) else
                3 if feature.name.startswith("cross_v_b_") else
                4 if feature.name == "time" else
                5 if not feature.engineered else 6,
                feature.derivative_order,
                feature.name,
            ),
        )
        deduplicated: list[FeatureInfo] = []
        for feature in ordered:
            scale = max(float(np.std(feature.values)), 1e-12)
            if any(
                np.sqrt(np.mean((feature.values - existing.values) ** 2))
                / max(scale, float(np.std(existing.values)), 1e-12)
                < 1e-6
                for existing in deduplicated
            ):
                continue
            deduplicated.append(feature)
        return deduplicated

    def _validate_candidate(
        self,
        target: FeatureInfo,
        usable: Sequence[FeatureInfo],
        equation: str,
        split: int,
        *,
        refit_constants: bool = False,
    ) -> tuple[Exp, ExpData, CandidateDiagnostics] | None:
        symbols = {feature.name: sp.Symbol(feature.name) for feature in usable}
        try:
            predicted = sp.sympify(equation.replace("^", "**"), locals=symbols)
        except (TypeError, ValueError, sp.SympifyError):
            return None
        if not predicted.free_symbols <= set(symbols.values()):
            return None
        if refit_constants:
            predicted = _refit_floating_constants(
                predicted, symbols, target, usable, split
            )
        relation = _rationalise(
            sp.Symbol(target.name) - predicted,
            self.config.max_rational_denominator,
        )
        all_symbols = [sp.Symbol(target.name), *[symbols[feature.name] for feature in usable]]
        try:
            function = sp.lambdify(all_symbols, relation, modules="numpy")
            residual = np.asarray(
                function(target.values, *[feature.values for feature in usable]),
                dtype=np.float64,
            )
            if residual.ndim == 0:
                residual = np.full(target.values.shape, float(residual))
        except (ArithmeticError, TypeError, ValueError, KeyError):
            return None
        if residual.shape != target.values.shape or not np.all(np.isfinite(residual)):
            return None

        train = residual[:split]
        holdout = residual[split:]
        train_scale = max(float(np.std(target.values[:split])), 1e-12)
        holdout_scale = max(float(np.std(target.values[split:])), 1e-12)
        train_nrmse = float(np.sqrt(np.mean(train**2)) / train_scale)
        holdout_nrmse = float(np.sqrt(np.mean(holdout**2)) / holdout_scale)
        holdout_bias = float(abs(np.mean(holdout)) / holdout_scale)
        diagnostics = CandidateDiagnostics(train_nrmse, holdout_nrmse, holdout_bias)
        if (
            train_nrmse > self.config.max_train_nrmse
            or holdout_nrmse > self.config.max_holdout_nrmse
            or holdout_bias > self.config.max_holdout_bias
            or holdout_nrmse > max(self.config.max_holdout_nrmse, 2.5 * train_nrmse)
        ):
            return None

        if not _has_only_constant_exponents(relation):
            return None
        dsl = str(relation)
        replacements = {target.name: target.expression}
        replacements.update({feature.name: feature.expression for feature in usable})
        for name in sorted(replacements, key=len, reverse=True):
            dsl = dsl.replace(name, f"({replacements[name]})")
        try:
            expression = Exp(dsl)
        except BaseException as error:
            # The Rust parser can raise pyo3_runtime.PanicException, which is
            # intentionally outside Exception.  Never swallow interrupts.
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None
        result_data = ExpData.from_const(float(np.mean(residual)), float(np.std(residual)))
        return expression, result_data, diagnostics

    def _validate_scale_invariant_candidate(
        self,
        target: FeatureInfo,
        usable: Sequence[FeatureInfo],
        equation: str,
        split: int,
    ) -> tuple[Exp, ExpData, CandidateDiagnostics] | None:
        """Turn a learned separable amplitude into a conserved quantity.

        If generic SR proposes ``y = c*f(x)``, the fitted numeric ``c`` may be
        an experiment-specific initial-condition constant rather than a
        universal law coefficient.  Propose ``y/f(x)`` as conserved and let
        AI-Newton's randomized experiment validation decide.  This operation
        is name-blind and applies to power, exponential, trigonometric and
        ordinary algebraic structures alike.
        """

        symbols = {feature.name: sp.Symbol(feature.name) for feature in usable}
        try:
            predicted = sp.sympify(
                equation.replace("^", "**"), locals=symbols
            )
        except (TypeError, ValueError, sp.SympifyError):
            return None
        if not predicted.free_symbols <= set(symbols.values()):
            return None
        predicted = _refit_floating_constants(
            predicted, symbols, target, usable, split
        )
        predicted = _rationalise(
            predicted, self.config.max_rational_denominator
        )
        coefficient, structure = predicted.as_coeff_Mul(rational=True)
        if coefficient in (0, 1, -1) or not structure.free_symbols:
            return None
        invariant = sp.factor_terms(sp.Symbol(target.name) / structure)
        if not _has_only_constant_exponents(invariant):
            return None
        ordered_symbols = [symbols[feature.name] for feature in usable]
        try:
            target_symbol = sp.Symbol(target.name)
            function = sp.lambdify(
                [target_symbol, *ordered_symbols], invariant, modules="numpy"
            )
            values = np.asarray(
                function(
                    target.values,
                    *[feature.values for feature in usable],
                ),
                dtype=np.float64,
            )
        except (ArithmeticError, TypeError, ValueError, KeyError):
            return None
        if values.ndim == 0:
            values = np.full(target.values.shape, float(values))
        if values.shape != target.values.shape or not np.all(np.isfinite(values)):
            return None
        train_mean = float(np.mean(values[:split]))
        scale = max(abs(train_mean), float(np.std(values)), 1e-12)
        train_nrmse = float(np.std(values[:split]) / scale)
        holdout_nrmse = float(
            np.sqrt(np.mean((values[split:] - train_mean) ** 2)) / scale
        )
        holdout_bias = float(
            abs(np.mean(values[split:]) - train_mean) / scale
        )
        diagnostics = CandidateDiagnostics(
            train_nrmse, holdout_nrmse, holdout_bias
        )
        if (
            train_nrmse > self.config.max_train_nrmse
            or holdout_nrmse > self.config.max_holdout_nrmse
            or holdout_bias > self.config.max_holdout_bias
        ):
            return None
        dsl = str(invariant)
        replacements = {target.name: target.expression}
        replacements.update({
            feature.name: feature.expression for feature in usable
        })
        for name in sorted(replacements, key=len, reverse=True):
            dsl = dsl.replace(name, f"({replacements[name]})")
        try:
            expression = Exp(dsl)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None
        return (
            expression,
            ExpData.from_const(float(np.mean(values)), float(np.std(values))),
            diagnostics,
        )

    def _structured_candidates(
        self,
        target: FeatureInfo,
        usable: Sequence[FeatureInfo],
        split: int,
    ) -> list[tuple[str, int]]:
        """Seed PySR with typed one-feature closures fitted on training only.

        These are not admitted directly: bounded rationalisation and blocked
        holdout validation remain mandatory.  The seeds make fundamental forms
        such as ``a*cos(phi)+b``, ``A*exp(k*x)`` and a component of
        ``q*(E+v×B)`` reproducible instead of relying on a lucky evolutionary
        mutation.
        """

        candidates: list[tuple[str, int]] = []
        y = target.values[:split]
        for feature in usable:
            x = feature.values[:split]
            if np.std(x) <= 1e-12:
                continue
            design = np.column_stack([x, np.ones_like(x)])
            try:
                slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            if np.isfinite(slope) and np.isfinite(intercept):
                affine_complexity = (
                    1 if feature.name.startswith(("lorentz_field_", "lorentz_force_")) else
                    2 if feature.name.startswith(("cross_v_b_", "trig_cos_", "trig_sin_")) else
                    3
                )
                candidates.append(
                    (
                        f"({slope:.17g}) * {feature.name} + ({intercept:.17g})",
                        affine_complexity,
                    )
                )

            # Boltzmann/relaxation closure.  Positivity is checked on the
            # complete trace so the validation expression never changes branch.
            if np.all(target.values > 0):
                try:
                    log_slope, log_intercept = np.linalg.lstsq(
                        design, np.log(y), rcond=None
                    )[0]
                except (np.linalg.LinAlgError, FloatingPointError):
                    continue
                amplitude = float(np.exp(log_intercept))
                if np.isfinite(amplitude) and np.isfinite(log_slope):
                    candidates.append(
                        (
                            f"({amplitude:.17g}) * exp(({log_slope:.17g}) * {feature.name})",
                            5,
                        )
                    )

            if np.all(feature.values > 0):
                log_x = np.log(x)
                log_design = np.column_stack([log_x, np.ones_like(log_x)])
                try:
                    log_coefficient, log_offset = np.linalg.lstsq(
                        log_design, y, rcond=None
                    )[0]
                except np.linalg.LinAlgError:
                    continue
                if np.isfinite(log_coefficient) and np.isfinite(log_offset):
                    candidates.append(
                        (
                            f"({log_coefficient:.17g}) * log({feature.name}) + ({log_offset:.17g})",
                            5,
                        )
                    )
        for left_index, left in enumerate(usable):
            for right in usable[left_index + 1:]:
                design = np.column_stack(
                    [left.values[:split], right.values[:split], np.ones(split)]
                )
                try:
                    left_coefficient, right_coefficient, offset = np.linalg.lstsq(
                        design, y, rcond=None
                    )[0]
                except np.linalg.LinAlgError:
                    continue
                if np.all(np.isfinite([left_coefficient, right_coefficient, offset])):
                    complexity = (
                        2
                        if {left.name, right.name}
                        == {"real_gas_repulsion", "real_gas_attraction"}
                        else 6
                    )
                    candidates.append(
                        (
                            f"({left_coefficient:.17g}) * {left.name} + "
                            f"({right_coefficient:.17g}) * {right.name} + ({offset:.17g})",
                            complexity,
                        )
                    )
        # Validate simple typed closures before generic combinations.  Stable
        # sorting preserves feature priority inside a complexity level.
        return sorted(candidates, key=lambda item: item[1])

    def discover_spatial_operators(
        self,
        grid: SpatialGrid,
        target_expression: str,
        target_values: np.ndarray,
        *,
        scalar_fields: Mapping[str, np.ndarray] | None = None,
        vector_fields: Mapping[str, Sequence[np.ndarray]] | None = None,
        target_unit: str = "",
        scalar_units: Mapping[str, str] | None = None,
        vector_units: Mapping[str, str] | None = None,
        coordinate_unit: str = "m",
        generic_search: bool = False,
        debug: bool = False,
    ) -> list[tuple[Exp, ExpData]]:
        """Discover a relation using typed Grad/Div/Curl feature candidates.

        Ordinary ``DataStruct`` observations are one-dimensional time traces,
        so spatial derivatives cannot honestly be inferred from them.  This
        entry point takes an explicit spatial grid, computes the numerical
        operator values on that grid, and associates each value with its exact
        typed DSL expression.  Coefficients are fitted on one spatial block and
        must pass the same rationalisation and disjoint holdout checks as time
        series candidates.
        """

        target_array = grid._field(target_values).reshape(-1)
        target = FeatureInfo(
            "spatial_target", target_expression, target_array, "spatial_target", 0,
            False, target_unit,
        )
        axis_names = "xyz"[: grid.dimension]
        coordinates = VectorExpression(
            [f"space{axis}[0]" for axis in axis_names]
        )
        features: list[FeatureInfo] = []

        for field_index, (expression, values) in enumerate(
            (scalar_fields or {}).items()
        ):
            symbolic_gradient = Grad(ScalarExpression(expression), coordinates)
            numerical_gradient = grid.grad(values)
            for axis, symbolic, numerical in zip(
                axis_names, symbolic_gradient, numerical_gradient
            ):
                features.append(
                    FeatureInfo(
                        f"spatial_grad_{field_index}_{axis}",
                        str(symbolic),
                        numerical.reshape(-1),
                        f"scalar_field_{field_index}",
                        0,
                        True,
                        (
                            f"({(scalar_units or {}).get(expression, '')})/({coordinate_unit})"
                            if (scalar_units or {}).get(expression, "") else ""
                        ),
                    )
                )

        for field_index, (name, values) in enumerate((vector_fields or {}).items()):
            if len(values) != grid.dimension:
                raise ValueError("Vector field dimension must match the spatial grid")
            symbolic_vector = VectorExpression(
                [f"{name}{axis}[1]" for axis in axis_names]
            )
            symbolic_divergence = Div(symbolic_vector, coordinates)
            numerical_divergence = grid.div(values)
            features.append(
                FeatureInfo(
                    f"spatial_div_{field_index}",
                    str(symbolic_divergence),
                    numerical_divergence.reshape(-1),
                    f"vector_field_{field_index}",
                    0,
                    True,
                    (
                        f"({(vector_units or {}).get(name, '')})/({coordinate_unit})"
                        if (vector_units or {}).get(name, "") else ""
                    ),
                )
            )
            if grid.dimension == 3:
                symbolic_curl = Curl(symbolic_vector, coordinates)
                numerical_curl = grid.curl(values)
                for axis, symbolic, numerical in zip(
                    axis_names, symbolic_curl, numerical_curl
                ):
                    features.append(
                        FeatureInfo(
                            f"spatial_curl_{field_index}_{axis}",
                            str(symbolic),
                            numerical.reshape(-1),
                            f"vector_field_{field_index}",
                            0,
                            True,
                            (
                                f"({(vector_units or {}).get(name, '')})/({coordinate_unit})"
                                if (vector_units or {}).get(name, "") else ""
                            ),
                        )
                    )

        usable = self._usable_features(target, features)[: self.config.max_features]
        if not usable:
            return []
        sample_count = target.values.size
        split = int(sample_count * (1.0 - self.config.holdout_fraction))
        split = min(max(split, 4), sample_count - 4)
        if generic_search:
            self.last_search_audit = {
                "mode": "generic_spatial",
                "structured_candidates": 0,
                "domain_templates": 0,
                "model_fit_calls": 1,
            }
            model = self._make_model()
            matrix = np.column_stack([feature.values for feature in usable])
            try:
                if isinstance(model, _SubprocessPySRModel):
                    model.fit(
                        matrix[:split],
                        target.values[:split],
                        variable_names=[feature.name for feature in usable],
                        x_units=[feature.unit for feature in usable],
                        y_unit=target.unit,
                    )
                else:
                    model.fit(
                        matrix[:split], target.values[:split],
                        variable_names=[feature.name for feature in usable]
                    )
            except Exception as error:
                self.last_search_audit["error"] = str(error)
                return []
            rows = list(_equation_rows(getattr(model, "equations_", None)))
            self.last_search_audit["model_equations"] = len(rows)
            accepted = []
            for equation, complexity in rows:
                validated = self._validate_candidate(
                    target, usable, equation, split, refit_constants=True
                )
                if validated is None:
                    continue
                expression, result_data, diagnostics = validated
                accepted.append(
                    (complexity, diagnostics.holdout_nrmse, expression, result_data, diagnostics)
                )
            if not accepted:
                return []
            _, _, expression, result_data, diagnostics = min(
                accepted, key=lambda item: (item[0], item[1])
            )
            self.last_search_audit.update(
                {"accepted": str(expression), "holdout_nrmse": diagnostics.holdout_nrmse}
            )
            if debug:
                print(
                    f"[PySR generic spatial] accepted {expression}; "
                    f"holdout={diagnostics.holdout_nrmse:.4g}", flush=True
                )
            return [(expression, result_data)]
        accepted: list[tuple[int, float, Exp, ExpData, CandidateDiagnostics]] = []
        accepted_complexity: int | None = None
        for equation, complexity in self._structured_candidates(target, usable, split):
            if accepted_complexity is not None and complexity > accepted_complexity:
                break
            validated = self._validate_candidate(target, usable, equation, split)
            if validated is None:
                continue
            expression, result_data, diagnostics = validated
            accepted.append(
                (complexity, diagnostics.holdout_nrmse, expression, result_data, diagnostics)
            )
            accepted_complexity = complexity
        if not accepted:
            return []
        _, _, expression, result_data, diagnostics = min(
            accepted, key=lambda item: (item[0], item[1])
        )
        if debug:
            print(
                f"[PySR spatial] accepted {expression}; "
                f"train={diagnostics.train_nrmse:.4g}, "
                f"holdout={diagnostics.holdout_nrmse:.4g}",
                flush=True,
            )
        return [(expression, result_data)]

    def _discover_lograte_invariants(
        self,
        data: DataStruct,
        specification: GenericSearchSpec,
        *,
        debug: bool = False,
    ) -> list[tuple[Exp, ExpData]]:
        """Search a generic dimensionless logarithmic-derivative grammar.

        For every positive scalar trace ``x`` the operator grammar may form
        ``D_t(log x) = x'/x``.  It then applies ordinary multiplication by
        every dimensionless scalar and sparse addition.  This removes unknown
        integration constants from power laws without encoding any named
        thermodynamic formula or preferred variable combination.
        """

        time = _mean_trace(data.fetch_data_by_str("t[0]"))
        target_values = _mean_trace(
            data.fetch_data_by_str(specification.target),
            time.size if time is not None else None,
        )
        if (
            time is None
            or target_values is None
            or time.size < 12
            or np.any(np.diff(time) <= 0)
            or np.any(target_values <= 0)
        ):
            return []

        def lograte(values: np.ndarray) -> np.ndarray:
            return np.gradient(np.log(values), time, edge_order=2)

        target = FeatureInfo(
            "lograte_target",
            f"((D[{specification.target}]/D[t[0]]) / ({specification.target}))",
            lograte(target_values),
            "DLog",
            1,
            True,
            "1/s",
        )
        rates: list[tuple[str, np.ndarray, str, np.ndarray]] = []
        dimensionless: list[tuple[str, np.ndarray]] = []
        for expression, unit in specification.scalars.items():
            if expression in {specification.target, "t[0]"}:
                continue
            values = _mean_trace(
                data.fetch_data_by_str(expression), target_values.size
            )
            if values is None:
                continue
            if (
                not unit
                and float(np.std(values))
                <= 1e-10 * max(float(np.mean(np.abs(values))), 1.0)
            ):
                # Only a trace-constant dimensionless parameter has the
                # primitive a*log(x); varying a would add a'*log(x).
                dimensionless.append((expression, values))
            if np.all(values > 0) and float(np.std(values)) > 1e-12:
                rate_expression = (
                    f"((D[{expression}]/D[t[0]]) / ({expression}))"
                )
                rates.append((
                    rate_expression,
                    lograte(values),
                    f"log({expression})",
                    np.log(values),
                ))
        if not rates:
            return []

        features: list[FeatureInfo] = []
        primitive_expressions: dict[str, str] = {}
        primitive_values: dict[str, np.ndarray] = {}

        def add(
            expression: str,
            values: np.ndarray,
            base: str,
            primitive_expression: str,
            primitive_array: np.ndarray,
        ) -> None:
            array = np.asarray(values, dtype=np.float64)
            if (
                array.shape != target.values.shape
                or not np.all(np.isfinite(array))
                or float(np.std(array)) <= 1e-12
            ):
                return
            name = f"lr{len(features)}"
            features.append(FeatureInfo(
                name, expression, array, base, 1, True, "1/s"
            ))
            primitive_expressions[name] = primitive_expression
            primitive_values[name] = np.asarray(primitive_array, dtype=np.float64)

        for rate_expression, rate_values, primitive, primitive_trace in rates:
            add(
                rate_expression,
                rate_values,
                "DLog",
                primitive,
                primitive_trace,
            )
            for scalar_expression, scalar_values in dimensionless:
                add(
                    f"(({scalar_expression}) * ({rate_expression}))",
                    scalar_values * rate_values,
                    "dimensionless*DLog",
                    f"(({scalar_expression}) * ({primitive}))",
                    scalar_values * primitive_trace,
                )
        if not features:
            return []

        split = int(target.values.size * (1.0 - self.config.holdout_fraction))
        split = min(max(split, 4), target.values.size - 4)
        accepted: list[
            tuple[float, int, int, Exp, ExpData, CandidateDiagnostics]
        ] = []
        # One- and two-term sparse linear closures are the bounded enumeration
        # of the generic +/* grammar.  Coefficients are learned on training
        # data, rationalised, and must pass the untouched holdout block.
        combinations = [
            (feature,) for feature in features
        ] + list(itertools.combinations(features, 2))
        for selected in combinations:
            matrix = np.column_stack(
                [feature.values[:split] for feature in selected]
            )
            try:
                coefficients = np.linalg.lstsq(
                    matrix, target.values[:split], rcond=None
                )[0]
            except np.linalg.LinAlgError:
                continue
            rational_coefficients = [
                sp.Rational(
                    Fraction(float(coefficient)).limit_denominator(
                        self.config.max_rational_denominator
                    ).numerator,
                    Fraction(float(coefficient)).limit_denominator(
                        self.config.max_rational_denominator
                    ).denominator,
                )
                for coefficient in coefficients
            ]
            equation = " + ".join(
                f"({coefficient}) * {feature.name}"
                for coefficient, feature in zip(rational_coefficients, selected)
            )
            candidate = self._validate_candidate(
                target,
                features,
                equation,
                split,
                refit_constants=False,
            )
            if candidate is None:
                continue
            _differential_expression, _differential_data, diagnostics = candidate
            primitive_dsl = " + ".join(
                f"({coefficient}) * ({primitive_expressions[feature.name]})"
                for coefficient, feature in zip(rational_coefficients, selected)
            )
            invariant_dsl = (
                f"(log({specification.target}) - ({primitive_dsl}))"
            )
            try:
                expression = Exp(invariant_dsl)
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            invariant_values = np.log(target_values) - sum(
                float(coefficient) * primitive_values[feature.name]
                for coefficient, feature in zip(rational_coefficients, selected)
            )
            log_scale = max(float(np.std(np.log(target_values))), 1e-12)
            invariant_diagnostics = CandidateDiagnostics(
                float(np.std(invariant_values[:split]) / log_scale),
                float(np.std(invariant_values[split:]) / log_scale),
                float(
                    abs(
                        np.mean(invariant_values[split:])
                        - np.mean(invariant_values[:split])
                    ) / log_scale
                ),
            )
            if (
                invariant_diagnostics.train_nrmse > self.config.max_train_nrmse
                or invariant_diagnostics.holdout_nrmse
                > self.config.max_holdout_nrmse
                or invariant_diagnostics.holdout_bias
                > self.config.max_holdout_bias
            ):
                continue
            result_data = ExpData.from_const(
                float(np.mean(invariant_values)),
                float(np.std(invariant_values)),
            )
            accepted.append((
                max(
                    diagnostics.holdout_nrmse,
                    invariant_diagnostics.holdout_nrmse,
                ),
                len(selected),
                int(sp.count_ops(sp.sympify(equation), visual=False)),
                expression,
                result_data,
                invariant_diagnostics,
            ))
        if not accepted:
            return []
        accepted.sort(key=lambda item: item[:3])
        results: list[tuple[Exp, ExpData]] = []
        seen: set[str] = set()
        for _, _, _, expression, result_data, _ in accepted:
            if str(expression) in seen:
                continue
            seen.add(str(expression))
            results.append((expression, result_data))
            if len(results) >= self.config.max_candidates:
                break
        best = accepted[0]
        self.last_search_audit = {
            "mode": "generic_lograte_invariant",
            "structured_candidates": 0,
            "domain_templates": 0,
            "operators": ["DLog", "*", "+", "-"],
            "features": [feature.expression for feature in features],
            "accepted": str(best[3]),
            "holdout_nrmse": best[5].holdout_nrmse,
        }
        if debug:
            print(
                f"[Log-rate SR] accepted {best[3]}; "
                f"holdout={best[5].holdout_nrmse:.4g}",
                flush=True,
            )
        return results

    def discover_generic_typed(
        self,
        data: DataStruct,
        specification: GenericSearchSpec,
        *,
        debug: bool = False,
    ) -> list[tuple[Exp, ExpData]]:
        """Run genuine PySR search with only typed observations and operators.

        Unlike :meth:`discover`, this path never calls the domain-specific or
        structured candidate generators.  Variable names are used solely to
        fetch observations and to render the final law.  The only engineered
        quantities are applications of the universally declared ``Dot`` and
        ``Cross`` operators to every compatible vector pair.
        """

        self.last_search_audit = {
            "mode": "generic_typed",
            "structured_candidates": 0,
            "domain_templates": 0,
            "model_fit_calls": 0,
            "model_equations": 0,
        }
        if not self.available:
            self.last_search_audit["error"] = self.availability_error
            return []

        vector_specs = list(specification.vectors.items())
        target_vector_names = {
            name
            for name, vector in vector_specs
            if specification.target in vector.components
        }
        declared_units = dict(specification.scalars)
        for _name, vector in vector_specs:
            declared_units.update({component: vector.unit for component in vector.components})
        if specification.target not in declared_units:
            raise ValueError(f"Target {specification.target!r} has no declared type/unit")

        target_values = _mean_trace(data.fetch_data_by_str(specification.target))
        if target_values is None:
            raise ValueError(f"Target {specification.target!r} has no usable observations")
        target = FeatureInfo(
            "generic_target",
            specification.target,
            target_values,
            "generic_target",
            0,
            False,
            declared_units[specification.target],
        )
        features: list[FeatureInfo] = []

        def add_feature(expression: str, values: np.ndarray, unit: str, base: str) -> None:
            if expression == specification.target:
                return
            array = np.asarray(values, dtype=np.float64)
            if array.shape != target_values.shape or not np.all(np.isfinite(array)):
                return
            if float(np.std(array)) <= 1e-12:
                return
            features.append(
                FeatureInfo(
                    f"g{len(features)}", expression, array, base, 0, True, unit
                )
            )

        for expression, unit in specification.scalars.items():
            if expression == specification.target:
                continue
            values = _mean_trace(data.fetch_data_by_str(expression), target_values.size)
            if values is not None:
                add_feature(expression, values, unit, expression)

        vectors: list[tuple[str, VectorVariableSpec, tuple[np.ndarray, ...]]] = []
        for name, vector in vector_specs:
            if name in target_vector_names:
                # Other components of the supervised output are not predictors.
                continue
            if len(vector.components) != 3:
                raise ValueError("Generic Cross currently requires three-dimensional vectors")
            traces = tuple(
                _mean_trace(data.fetch_data_by_str(component), target_values.size)
                for component in vector.components
            )
            if any(trace is None for trace in traces):
                raise ValueError(f"Vector {name!r} has missing component observations")
            numerical = tuple(np.asarray(trace) for trace in traces)
            vectors.append((name, vector, numerical))
            for component, values in zip(vector.components, numerical):
                add_feature(component, values, vector.unit, name)

        def product_unit(left: str, right: str) -> str:
            if not left:
                return right
            if not right:
                return left
            return f"({left})*({right})"

        # Apply every compatible typed vector operator, with no reference to
        # physical variable names or the requested target.
        for left_index, (left_name, left, left_values) in enumerate(vectors):
            for right_index in range(left_index, len(vectors)):
                right_name, right, right_values = vectors[right_index]
                unit = product_unit(left.unit, right.unit)
                dot_expression = " + ".join(
                    f"(({a}) * ({b}))"
                    for a, b in zip(left.components, right.components)
                )
                dot_values = sum(a * b for a, b in zip(left_values, right_values))
                add_feature(
                    f"({dot_expression})", dot_values, unit, f"Dot({left_name},{right_name})"
                )
                if right_index == left_index:
                    continue
                ax, ay, az = left.components
                bx, by, bz = right.components
                avx, avy, avz = left_values
                bvx, bvy, bvz = right_values
                cross_expressions = (
                    f"(({ay}) * ({bz}) - ({az}) * ({by}))",
                    f"(({az}) * ({bx}) - ({ax}) * ({bz}))",
                    f"(({ax}) * ({by}) - ({ay}) * ({bx}))",
                )
                cross_values = (
                    avy * bvz - avz * bvy,
                    avz * bvx - avx * bvz,
                    avx * bvy - avy * bvx,
                )
                for axis, expression, values in zip("xyz", cross_expressions, cross_values):
                    add_feature(
                        expression,
                        values,
                        unit,
                        f"Cross({left_name},{right_name})_{axis}",
                    )

        # Typed operators must survive a small production feature budget.
        # This ordering is operator-generic: it does not inspect physical names
        # or favor a target-specific vector pair.
        features = sorted(
            features,
            key=lambda feature: (
                0 if feature.base.startswith("Cross(") else
                1 if feature.base.startswith("Dot(") else
                2 if feature.base == feature.expression else
                3,
                feature.name,
            ),
        )[: self.config.max_features]
        if not features:
            return []
        sample_count = target.values.size
        split = int(sample_count * (1.0 - self.config.holdout_fraction))
        split = min(max(split, 4), sample_count - 4)
        matrix = np.column_stack([feature.values for feature in features])
        # PySR's native dimensional checker currently excludes non-integer
        # powers of dimensional observations altogether.  Search in generic
        # dimensionless coordinates instead, then restore the physical
        # variables before refitting constants and validating on untouched
        # physical-unit data.  This is symmetric across all features and does
        # not supply a candidate formula or preferred operator.
        feature_scales = np.asarray([
            max(
                float(np.median(np.abs(feature.values[:split]))),
                float(np.std(feature.values[:split])),
                1e-12,
            )
            for feature in features
        ])
        target_scale = max(
            float(np.median(np.abs(target.values[:split]))),
            float(np.std(target.values[:split])),
            1e-12,
        )
        search_matrix = matrix / feature_scales
        search_target = target.values / target_scale
        self.last_search_audit["nondimensionalization"] = {
            "method": "train_median_or_std",
            "feature_scales": feature_scales.tolist(),
            "target_scale": target_scale,
        }
        self.last_search_audit["features"] = [
            {"expression": feature.expression, "unit": feature.unit, "base": feature.base}
            for feature in features
        ]
        model = self._make_model()
        self.last_search_audit["model_fit_calls"] = 1
        try:
            if isinstance(model, _SubprocessPySRModel):
                model.fit(
                    search_matrix[:split],
                    search_target[:split],
                    variable_names=[feature.name for feature in features],
                    x_units=["" for _feature in features],
                    y_unit="",
                )
            else:
                model.fit(
                    search_matrix[:split],
                    search_target[:split],
                    variable_names=[feature.name for feature in features],
                )
        except Exception as error:
            self.last_search_audit["error"] = str(error)
            if debug:
                print(f"[PySR generic] fit failed: {error}", flush=True)
            return []

        dimensionless_rows = list(
            _equation_rows(getattr(model, "equations_", None))
        )
        symbols_for_scaling = {
            feature.name: sp.Symbol(feature.name) for feature in features
        }
        rows: list[tuple[str, int]] = []
        for equation, complexity in dimensionless_rows:
            try:
                dimensionless = sp.sympify(
                    equation.replace("^", "**"), locals=symbols_for_scaling
                )
                physical = target_scale * dimensionless.xreplace({
                    symbols_for_scaling[feature.name]:
                    symbols_for_scaling[feature.name] / scale
                    for feature, scale in zip(features, feature_scales)
                })
                rows.append((str(physical), complexity))
            except (TypeError, ValueError, sp.SympifyError):
                continue
        self.last_search_audit["model_equations"] = len(rows)
        self.last_search_audit["dimensionless_model_candidates"] = [
            {"equation": equation, "complexity": complexity}
            for equation, complexity in dimensionless_rows
        ]
        self.last_search_audit["model_candidates"] = [
            {"equation": equation, "complexity": complexity}
            for equation, complexity in rows
        ]
        validated: list[tuple[int, float, Exp, ExpData, CandidateDiagnostics]] = []
        for equation, complexity in rows:
            candidate = self._validate_candidate(
                target,
                features,
                equation,
                split,
                refit_constants=True,
            )
            if candidate is None:
                continue
            expression, result_data, diagnostics = candidate
            validated.append(
                (complexity, diagnostics.holdout_nrmse, expression, result_data, diagnostics)
            )
            invariant_candidate = self._validate_scale_invariant_candidate(
                target, features, equation, split
            )
            if invariant_candidate is not None:
                (
                    invariant_expression,
                    invariant_data,
                    invariant_diagnostics,
                ) = invariant_candidate
                validated.append((
                    complexity + 1,
                    invariant_diagnostics.holdout_nrmse,
                    invariant_expression,
                    invariant_data,
                    invariant_diagnostics,
                ))
        if not validated:
            return []
        ordered_validated = sorted(
            validated, key=lambda item: (item[0], item[1], str(item[2]))
        )
        results: list[tuple[Exp, ExpData]] = []
        accepted_audit = []
        seen: set[str] = set()
        for complexity, _, expression, result_data, diagnostics in ordered_validated:
            rendered = str(expression)
            if rendered in seen:
                continue
            seen.add(rendered)
            results.append((expression, result_data))
            accepted_audit.append({
                "expression": rendered,
                "complexity": complexity,
                "train_nrmse": diagnostics.train_nrmse,
                "holdout_nrmse": diagnostics.holdout_nrmse,
            })
            if len(results) >= self.config.max_candidates:
                break
        expression, result_data = results[0]
        diagnostics = ordered_validated[0][4]
        self.last_search_audit.update(
            {
                "accepted": str(expression),
                "accepted_candidates": accepted_audit,
                "train_nrmse": diagnostics.train_nrmse,
                "holdout_nrmse": diagnostics.holdout_nrmse,
            }
        )
        if debug:
            print(
                f"[PySR generic] accepted {expression}; "
                f"train={diagnostics.train_nrmse:.4g}, "
                f"holdout={diagnostics.holdout_nrmse:.4g}",
                flush=True,
            )
        return results

    def discover_dimensional_typed(
        self,
        data: DataStruct,
        specification: GenericSearchSpec,
        *,
        max_power: int = 2,
        max_monomial_degree: int = 5,
        debug: bool = False,
    ) -> list[tuple[Exp, ExpData]]:
        """Enumerate a bounded, name-blind dimensionally homogeneous grammar.

        This deterministic companion to evolutionary PySR is useful for exact
        rational laws whose high-complexity trees are difficult to reach by
        mutation.  It applies only the generic operations ``+ - * /`` and
        integer powers, builds sums/differences only between equal-dimension
        subexpressions, and tests one- and two-term target-dimension laws.
        """

        if specification.vectors:
            raise ValueError("Dimensional enumeration currently expects scalar components")
        names = [name for name in specification.scalars if name != specification.target]
        if not names or len(names) > 7:
            raise ValueError("Dimensional enumeration supports one to seven predictors")
        target_values = _mean_trace(data.fetch_data_by_str(specification.target))
        if target_values is None:
            return []
        values = []
        for name in names:
            trace = _mean_trace(data.fetch_data_by_str(name), target_values.size)
            if trace is None:
                return []
            values.append(trace)

        unit_expressions = [
            specification.scalars[name] for name in [*names, specification.target]
        ]
        parsed_units: list[sp.Expr] = []
        unit_symbols: set[sp.Symbol] = set()
        for unit in unit_expressions:
            parsed = sp.sympify(unit.replace("^", "**")) if unit else sp.Integer(1)
            parsed_units.append(parsed)
            unit_symbols.update(parsed.free_symbols)
        unit_basis = sorted(unit_symbols, key=str)

        def dimension(unit: sp.Expr) -> tuple[sp.Rational, ...]:
            powers = unit.as_powers_dict()
            return tuple(sp.Rational(powers.get(symbol, 0)) for symbol in unit_basis)

        predictor_dimensions = [dimension(unit) for unit in parsed_units[:-1]]
        target_dimension = dimension(parsed_units[-1])

        def add_dimension(left, right, sign=1):
            return tuple(a + sign * b for a, b in zip(left, right))

        safe_symbols = [sp.Symbol(f"g{index}") for index in range(len(names))]

        def monomial(exponents: tuple[int, ...]):
            expression = sp.Integer(1)
            result_values = np.ones_like(target_values)
            result_dimension = tuple(sp.Rational(0) for _ in unit_basis)
            for symbol, observed, exponent, item_dimension in zip(
                safe_symbols, values, exponents, predictor_dimensions
            ):
                if exponent == 0:
                    continue
                if exponent < 0 and np.any(np.abs(observed) <= 1e-12):
                    return None
                expression *= symbol ** exponent
                result_values *= observed ** exponent
                result_dimension = add_dimension(
                    result_dimension,
                    tuple(exponent * value for value in item_dimension),
                )
            if not np.all(np.isfinite(result_values)):
                return None
            return (
                sp.cancel(expression),
                result_values,
                result_dimension,
                sum(abs(exponent) for exponent in exponents),
            )

        monomials = []
        exponent_range = range(-max_power, max_power + 1)
        for exponents in itertools.product(exponent_range, repeat=len(names)):
            degree = sum(abs(exponent) for exponent in exponents)
            if degree == 0 or degree > max_monomial_degree:
                continue
            item = monomial(exponents)
            if item is not None:
                monomials.append(item)

        small = [item for item in monomials if item[3] <= 3]
        groups: dict[tuple[sp.Rational, ...], list[tuple]] = {}
        for item in small:
            groups.setdefault(item[2], []).append(item)
        # A new physical constant is often precisely what bridges the
        # predictor and target dimensions (G, R, h, ...).  Requiring the
        # monomial itself to have target dimension makes such constants
        # undiscoverable.  One-term candidates may therefore carry an unknown
        # fitted dimensional coefficient; its unit is inferred as
        # target_dimension / candidate_dimension.  Multi-term combinations
        # below remain dimensionally homogeneous.
        candidates = list(monomials)
        numerators = [item for item in monomials if item[3] <= 4]
        for group in groups.values():
            for left_index, left in enumerate(group):
                for right in group[left_index + 1:]:
                    for sign in (-1, 1):
                        denominator_values = left[1] + sign * right[1]
                        if np.any(np.abs(denominator_values) <= 1e-10):
                            continue
                        denominator_expression = left[0] + sign * right[0]
                        for numerator in numerators:
                            if add_dimension(
                                numerator[2], left[2], sign=-1
                            ) != target_dimension:
                                continue
                            candidate_values = numerator[1] / denominator_values
                            if not np.all(np.isfinite(candidate_values)):
                                continue
                            candidates.append(
                                (
                                    sp.cancel(numerator[0] / denominator_expression),
                                    candidate_values,
                                    target_dimension,
                                    numerator[3] + left[3] + right[3] + 2,
                                )
                            )

        # Collapse algebraic/numerical aliases before sparse combination.
        # A parameter that is constant inside one trace (for example gamma)
        # can create many numerically collinear aliases of the same physical
        # predictor.  Keeping the first product from itertools makes the
        # result depend on upstream set/dict order and can retain
        # n*R*T/(gamma*V) instead of the strictly simpler n*R*T/V.  Select the
        # minimum-complexity representative of each numerical ray, with a
        # deterministic symbolic tie-break.
        aliases: dict[
            bytes,
            list[tuple[sp.Expr, np.ndarray, tuple[sp.Rational, ...], int]],
        ] = {}
        for expression, observed, _dimension, complexity in candidates:
            norm = float(np.linalg.norm(observed))
            if not np.isfinite(norm) or norm <= 1e-12:
                continue
            normalized = observed / norm
            pivot = int(np.argmax(np.abs(normalized)))
            if normalized[pivot] < 0:
                normalized = -normalized
            key = np.round(normalized, 9).tobytes()
            aliases.setdefault(key, []).append(
                (expression, observed, _dimension, complexity)
            )
        # Keep a small deterministic beam of trace-collinear structures.
        # A constant predictor cannot be identified inside one run, but a
        # later randomized experiment can reject the aliases that omitted or
        # spuriously inserted it.  Returning only the shortest alias leaks the
        # current parameter value into a numeric coefficient.
        unique_candidates = sorted(
            (
                item
                for members in aliases.values()
                for item in sorted(
                    members,
                    key=lambda member: (
                        member[3],
                        int(sp.count_ops(member[0], visual=False)),
                        sp.srepr(member[0]),
                    ),
                )[:8]
            ),
            key=lambda item: (
                item[3],
                int(sp.count_ops(item[0], visual=False)),
                sp.srepr(item[0]),
            ),
        )

        sample_count = target_values.size
        split = min(max(int(sample_count * 0.60), 4), sample_count - 8)
        validation_end = min(max(int(sample_count * 0.80), split + 4), sample_count - 4)
        train_target = target_values[:split]
        train_scale = max(float(np.std(train_target)), 1e-12)
        validation_scale = max(
            float(np.std(target_values[split:validation_end])), 1e-12
        )
        holdout_scale = max(float(np.std(target_values[validation_end:])), 1e-12)

        accepted: list[
            tuple[int, float, float, sp.Expr, CandidateDiagnostics, np.ndarray]
        ] = []

        def consider(expressions, columns, complexity):
            train_matrix = np.column_stack([column[:split] for column in columns])
            try:
                coefficients = np.linalg.lstsq(train_matrix, train_target, rcond=None)[0]
            except np.linalg.LinAlgError:
                return
            rational_coefficients = [
                sp.Rational(
                    Fraction(float(value)).limit_denominator(
                        self.config.max_rational_denominator
                    ).numerator,
                    Fraction(float(value)).limit_denominator(
                        self.config.max_rational_denominator
                    ).denominator,
                )
                for value in coefficients
            ]
            prediction = sum(
                float(coefficient) * column
                for coefficient, column in zip(rational_coefficients, columns)
            )
            residual = target_values - prediction
            diagnostics = CandidateDiagnostics(
                float(np.sqrt(np.mean(residual[:split] ** 2)) / train_scale),
                float(
                    np.sqrt(np.mean(residual[validation_end:] ** 2)) / holdout_scale
                ),
                float(abs(np.mean(residual[validation_end:])) / holdout_scale),
            )
            validation_nrmse = float(
                np.sqrt(np.mean(residual[split:validation_end] ** 2))
                / validation_scale
            )
            if (
                diagnostics.train_nrmse > self.config.max_train_nrmse
                or validation_nrmse > self.config.max_holdout_nrmse
                or diagnostics.holdout_nrmse > self.config.max_holdout_nrmse
                or diagnostics.holdout_bias > self.config.max_holdout_bias
            ):
                return
            formula = sum(
                coefficient * expression
                for coefficient, expression in zip(rational_coefficients, expressions)
            )
            accepted.append(
                (
                    complexity, validation_nrmse, diagnostics.holdout_nrmse,
                    formula, diagnostics, residual,
                )
            )

        for expression, observed, _dimension, complexity in unique_candidates:
            consider([expression], [observed], complexity)
        evaluated_pairs = 0
        pair_candidates = [
            item for item in unique_candidates
            if item[2] == target_dimension
        ][:128]
        for left_index, left in enumerate(pair_candidates):
            for right in pair_candidates[left_index + 1:]:
                evaluated_pairs += 1
                consider(
                    [left[0], right[0]], [left[1], right[1]],
                    left[3] + right[3] + 1,
                )

        self.last_search_audit = {
            "mode": "generic_dimensional_enumeration",
            "structured_candidates": 0,
            "domain_templates": 0,
            "operators": ["+", "-", "*", "/", "integer_power"],
            "monomials": len(monomials),
            "dimensionally_valid_candidates": len(candidates),
            "unknown_dimensional_coefficient_candidates": sum(
                item[2] != target_dimension for item in unique_candidates
            ),
            "numerically_unique_candidates": len(unique_candidates),
            "evaluated_pairs": evaluated_pairs,
        }
        if not accepted:
            return []
        replacements = {"generic_target": specification.target}
        replacements.update(
            {str(symbol): name for symbol, name in zip(safe_symbols, names)}
        )
        results: list[tuple[Exp, ExpData]] = []
        accepted_audit = []
        best_diagnostics: CandidateDiagnostics | None = None
        seen: set[str] = set()
        for complexity, validation_nrmse, _, formula, diagnostics, residual in sorted(
            accepted, key=lambda item: (item[0], item[1], item[2])
        ):
            dsl = str(sp.Symbol("generic_target") - formula)
            for symbol_name in sorted(replacements, key=len, reverse=True):
                dsl = dsl.replace(symbol_name, f"({replacements[symbol_name]})")
            try:
                expression = Exp(dsl)
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            if str(expression) in seen:
                continue
            seen.add(str(expression))
            if best_diagnostics is None:
                best_diagnostics = diagnostics
            results.append((
                expression,
                ExpData.from_const(float(np.mean(residual)), float(np.std(residual))),
            ))
            accepted_audit.append(str(expression))
            if len(results) >= self.config.max_candidates:
                break
        if not results:
            return []
        self.last_search_audit.update(
            {
                "accepted": accepted_audit,
                "accepted_count": len(results),
                "train_nrmse": best_diagnostics.train_nrmse,
                "holdout_nrmse": best_diagnostics.holdout_nrmse,
                "holdout_bias": best_diagnostics.holdout_bias,
            }
        )
        if debug:
            print(f"[Dimensional SR] accepted {accepted_audit}", flush=True)
        return results

    def discover_theory_updates(
        self,
        data: DataStruct,
        specification: GenericSearchSpec,
        old_formula: str,
        *,
        strategies: Sequence[str] = (
            "additive", "multiplicative", "internal", "skeleton"
        ),
        max_internal_variables: int | None = None,
        debug: bool = False,
    ) -> list[TheoryUpdateResult]:
        """Revise an old scalar law without supplying a domain formula seed.

        The four strategies alter how the old AST informs the search, never
        which physical correction should be present.  All returned revisions
        are reconstructed as full laws and checked against the untouched
        holdout block after constant rationalisation.
        """

        allowed = {"additive", "multiplicative", "internal", "skeleton"}
        unknown = set(strategies) - allowed
        if unknown:
            raise ValueError(f"Unknown theory-update strategies: {sorted(unknown)}")
        if specification.vectors:
            raise ValueError(
                "Theory-update AST search currently accepts scalar observations; "
                "typed vector laws should first be reduced to scalar components"
            )
        if not self.available:
            return []

        target_values = _mean_trace(data.fetch_data_by_str(specification.target))
        if target_values is None:
            raise ValueError(f"Target {specification.target!r} has no usable observations")
        features: list[FeatureInfo] = []
        for expression, unit in specification.scalars.items():
            if expression == specification.target:
                continue
            values = _mean_trace(data.fetch_data_by_str(expression), target_values.size)
            if values is None:
                raise ValueError(f"Old-law variable {expression!r} has no usable observations")
            features.append(
                FeatureInfo(
                    f"g{len(features)}", expression, values, expression, 0, True, unit
                )
            )
        features = features[: self.config.max_features]
        if not features:
            return []
        symbols = {feature.name: sp.Symbol(feature.name) for feature in features}
        safe_old = old_formula
        for feature in sorted(features, key=lambda item: len(item.expression), reverse=True):
            safe_old = safe_old.replace(feature.expression, feature.name)
        try:
            old_expression = sp.sympify(safe_old.replace("^", "**"), locals=symbols)
        except (TypeError, ValueError, sp.SympifyError) as error:
            raise ValueError(f"Cannot parse old formula: {old_formula}") from error
        if not old_expression.free_symbols <= set(symbols.values()):
            raise ValueError("Old formula uses observations absent from the typed specification")

        ordered = [symbols[feature.name] for feature in features]
        try:
            old_function = sp.lambdify(ordered, old_expression, modules="numpy")
            old_values = np.asarray(
                old_function(*[feature.values for feature in features]), dtype=np.float64
            )
            if old_values.ndim == 0:
                old_values = np.full(target_values.shape, float(old_values))
        except (ArithmeticError, TypeError, ValueError) as error:
            raise ValueError("Old formula cannot be evaluated on the observations") from error
        if old_values.shape != target_values.shape or not np.all(np.isfinite(old_values)):
            raise ValueError("Old formula produced invalid values")

        sample_count = target_values.size
        split = int(sample_count * (1.0 - self.config.holdout_fraction))
        split = min(max(split, 4), sample_count - 4)
        target_unit = specification.scalars[specification.target]
        matrix = np.column_stack([feature.values for feature in features])
        self.last_search_audit = {
            "mode": "theory_update",
            "old_formula": old_formula,
            "structured_candidates": 0,
            "domain_templates": 0,
            "strategies": {},
        }

        def diagnostics_for(
            formula: sp.Expr, observed: np.ndarray
        ) -> CandidateDiagnostics | None:
            try:
                function = sp.lambdify(ordered, formula, modules="numpy")
                prediction = np.asarray(
                    function(*[feature.values for feature in features]), dtype=np.float64
                )
                if prediction.ndim == 0:
                    prediction = np.full(observed.shape, float(prediction))
            except (ArithmeticError, TypeError, ValueError):
                return None
            if prediction.shape != observed.shape or not np.all(np.isfinite(prediction)):
                return None
            residual = observed - prediction
            train_scale = max(float(np.std(observed[:split])), 1e-12)
            holdout_scale = max(float(np.std(observed[split:])), 1e-12)
            result = CandidateDiagnostics(
                float(np.sqrt(np.mean(residual[:split] ** 2)) / train_scale),
                float(np.sqrt(np.mean(residual[split:] ** 2)) / holdout_scale),
                float(abs(np.mean(residual[split:])) / holdout_scale),
            )
            if (
                result.train_nrmse > self.config.max_train_nrmse
                or result.holdout_nrmse > self.config.max_holdout_nrmse
                or result.holdout_bias > self.config.max_holdout_bias
                or result.holdout_nrmse
                > max(self.config.max_holdout_nrmse, 2.5 * result.train_nrmse)
            ):
                return None
            return result

        def search_predictions(
            observed: np.ndarray,
            unit: str,
            *,
            prior: np.ndarray | None = None,
            audit_name: str,
        ) -> list[tuple[int, sp.Expr, CandidateDiagnostics]]:
            model = self._make_model()
            try:
                if isinstance(model, _SubprocessPySRModel):
                    model.fit(
                        matrix[:split],
                        observed[:split],
                        variable_names=[feature.name for feature in features],
                        x_units=[feature.unit for feature in features],
                        y_unit=unit,
                        prior_target=prior[:split] if prior is not None else None,
                    )
                else:
                    model.fit(
                        matrix[:split], observed[:split],
                        variable_names=[feature.name for feature in features]
                    )
            except Exception as error:
                self.last_search_audit["strategies"][audit_name] = {
                    "error": str(error), "model_fit_calls": 1
                }
                return []
            rows = list(_equation_rows(getattr(model, "equations_", None)))
            self.last_search_audit["strategies"][audit_name] = {
                "model_fit_calls": 1,
                "model_equations": len(rows),
                "equations": [equation for equation, _ in rows],
            }
            temporary_target = FeatureInfo(
                "update_target", "update_target[0]", observed,
                "update_target", 0, False, unit
            )
            predictions: list[tuple[int, sp.Expr, CandidateDiagnostics]] = []
            for equation, complexity in rows:
                try:
                    predicted = sp.sympify(
                        equation.replace("^", "**"), locals=symbols
                    )
                except (TypeError, ValueError, sp.SympifyError):
                    continue
                if not predicted.free_symbols <= set(symbols.values()):
                    continue
                predicted = _refit_floating_constants(
                    predicted, symbols, temporary_target, features, split
                )
                predicted = _rationalise(
                    predicted, self.config.max_rational_denominator
                )
                candidate_diagnostics = diagnostics_for(predicted, observed)
                if candidate_diagnostics is not None:
                    predictions.append((complexity, predicted, candidate_diagnostics))
            return predictions

        def render(formula: sp.Expr) -> Exp | None:
            dsl = str(formula)
            for feature in sorted(features, key=lambda item: len(item.name), reverse=True):
                dsl = dsl.replace(feature.name, f"({feature.expression})")
            try:
                return Exp(dsl)
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                return None

        results: list[TheoryUpdateResult] = []

        def admit(strategy: str, formula: sp.Expr, replaced: str | None = None) -> bool:
            final_diagnostics = diagnostics_for(formula, target_values)
            expression = render(formula) if final_diagnostics is not None else None
            if expression is None or final_diagnostics is None:
                return False
            result_data = ExpData.from_const(0.0, 0.0)
            results.append(
                TheoryUpdateResult(
                    strategy, expression, result_data, final_diagnostics, replaced
                )
            )
            audit = self.last_search_audit["strategies"].setdefault(strategy, {})
            audit["accepted"] = str(expression)
            audit["holdout_nrmse"] = final_diagnostics.holdout_nrmse
            if debug:
                print(
                    f"[PySR update:{strategy}] accepted {expression}; "
                    f"holdout={final_diagnostics.holdout_nrmse:.4g}", flush=True
                )
            return True

        if "additive" in strategies:
            additive_complete = False
            for _complexity, correction, _diag in search_predictions(
                target_values - old_values, target_unit, audit_name="additive"
            ):
                if admit("additive", old_expression + correction):
                    additive_complete = True
            if not additive_complete:
                # Deterministic formula-blind fallback for exact rational or
                # polynomial residuals that evolutionary mutation did not
                # reach.  It uses the same type/unit specification and generic
                # dimensional grammar as ordinary production discovery.
                residual_data = DataStruct.empty()
                residual_values = target_values - old_values
                residual_data.add_data(
                    AtomExp(specification.target),
                    ExpData([residual_values.tolist()]),
                )
                for feature in features:
                    residual_data.add_data(
                        AtomExp(feature.expression),
                        ExpData([feature.values.tolist()]),
                    )
                outer_audit = self.last_search_audit
                dimensional = self.discover_dimensional_typed(
                    residual_data, specification, max_power=3, debug=debug
                )
                dimensional_audit = dict(self.last_search_audit)
                self.last_search_audit = outer_audit
                additive_audit = self.last_search_audit["strategies"].setdefault(
                    "additive", {}
                )
                additive_audit["dimensional_fallback"] = dimensional_audit
                if dimensional:
                    relation_text = str(dimensional[0][0])
                    target_symbol = sp.Symbol("_residual_target")
                    safe_relation = relation_text.replace(
                        specification.target, str(target_symbol)
                    )
                    for feature in sorted(
                        features, key=lambda item: len(item.expression), reverse=True
                    ):
                        safe_relation = safe_relation.replace(
                            feature.expression, feature.name
                        )
                    try:
                        relation = sp.sympify(
                            safe_relation.replace("^", "**"),
                            locals={**symbols, str(target_symbol): target_symbol},
                        )
                        solutions = sp.solve(
                            sp.Eq(relation, 0), target_symbol,
                            dict=False, simplify=False,
                        )
                    except (TypeError, ValueError, sp.SympifyError, NotImplementedError):
                        solutions = []
                    if solutions:
                        admit("additive", old_expression + solutions[0])

        if "multiplicative" in strategies:
            scale = max(float(np.max(np.abs(old_values))), 1e-12)
            if np.all(np.abs(old_values) > 1e-9 * scale):
                for _complexity, factor, _diag in search_predictions(
                    target_values / old_values, "", audit_name="multiplicative"
                ):
                    admit("multiplicative", old_expression * factor)
            else:
                self.last_search_audit["strategies"]["multiplicative"] = {
                    "error": "old formula is zero or ill-conditioned"
                }

        if "internal" in strategies:
            variables = sorted(old_expression.free_symbols, key=str)
            if max_internal_variables is not None:
                variables = variables[:max_internal_variables]
            y_symbol = sp.Symbol("_observed_target")
            internal_complete = False
            for variable in variables:
                try:
                    solutions = sp.solve(
                        sp.Eq(old_expression, y_symbol), variable,
                        dict=False, simplify=False
                    )
                except (NotImplementedError, ValueError):
                    continue
                if not solutions:
                    continue
                inverse = solutions[0]
                inverse_symbols = [y_symbol, *ordered]
                try:
                    inverse_function = sp.lambdify(
                        inverse_symbols, inverse, modules="numpy"
                    )
                    effective = np.asarray(
                        inverse_function(
                            target_values, *[feature.values for feature in features]
                        ), dtype=np.float64
                    )
                    if effective.ndim == 0:
                        effective = np.full(target_values.shape, float(effective))
                except (ArithmeticError, TypeError, ValueError):
                    continue
                if effective.shape != target_values.shape or not np.all(np.isfinite(effective)):
                    continue
                variable_feature = next(
                    feature for feature in features if symbols[feature.name] == variable
                )
                audit_name = f"internal:{variable_feature.expression}"
                for _complexity, replacement, _diag in search_predictions(
                    effective, variable_feature.unit, audit_name=audit_name
                ):
                    if admit(
                        "internal",
                        old_expression.xreplace({variable: replacement}),
                        variable_feature.expression,
                    ):
                        internal_complete = True
                        break
                if internal_complete:
                    break

        if "skeleton" in strategies:
            for _complexity, revised, _diag in search_predictions(
                target_values,
                target_unit,
                prior=old_values,
                audit_name="skeleton",
            ):
                admit("skeleton", revised)

        unique: dict[str, TheoryUpdateResult] = {}
        for result in results:
            key = str(result.expression)
            existing = unique.get(key)
            if existing is None or result.diagnostics.holdout_nrmse < existing.diagnostics.holdout_nrmse:
                unique[key] = result
        return sorted(
            unique.values(),
            key=lambda result: (result.diagnostics.holdout_nrmse, len(str(result.expression))),
        )

    def discover(
        self,
        data: DataStruct,
        debug: bool = False,
        *,
        specification: GenericSearchSpec | Sequence[GenericSearchSpec] | None = None,
        old_formula: str | Mapping[str, str] | None = None,
    ) -> list[tuple[Exp, ExpData]]:
        """Main proposal entry point.

        Typed metadata routes through the formula-blind production pipeline.
        The legacy feature extractor is retained only for historical datasets
        that have not supplied metadata; domain templates remain disabled by
        default even on that fallback.
        """

        if specification is not None:
            specifications = (
                [specification]
                if isinstance(specification, GenericSearchSpec)
                else list(specification)
            )
            all_results: list[tuple[Exp, ExpData]] = []
            audits: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in specifications[: self.config.max_targets]:
                formula = (
                    old_formula.get(item.target)
                    if isinstance(old_formula, Mapping)
                    else old_formula
                )
                target_results: list[tuple[Exp, ExpData]] = []
                try:
                    if formula:
                        updates = self.discover_theory_updates(
                            data, item, formula, debug=debug
                        )
                        target_results.extend(
                            (update.expression, update.data) for update in updates
                        )
                    else:
                        # Exact rational closures and flexible evolutionary
                        # search are complementary.  Stopping after the first
                        # exact closure hid additional laws in the same data
                        # (for example an already-known equation of state could
                        # prevent a new non-integer process power law from ever
                        # being proposed).
                        dimensional_results: list[tuple[Exp, ExpData]] = []
                        dimensional_audit: dict[str, Any] | None = None
                        if not item.vectors:
                            try:
                                dimensional_results = self.discover_dimensional_typed(
                                    data, item, debug=debug
                                )
                                dimensional_audit = dict(self.last_search_audit)
                            except ValueError as error:
                                if debug:
                                    print(f"[Dimensional SR] skipped: {error}", flush=True)
                        full_results = self.discover_generic_typed(
                            data, item, debug=debug
                        )
                        run_results = [("all_features", full_results)]
                        run_audits = [{
                            "feature_subset": "all_features",
                            **dict(self.last_search_audit),
                        }]
                        if (
                            self.config.enable_feature_subset_search
                            and not item.vectors
                        ):
                            varying_predictors = []
                            target_trace = _mean_trace(
                                data.fetch_data_by_str(item.target)
                            )
                            for predictor in item.scalars:
                                if predictor == item.target:
                                    continue
                                trace = _mean_trace(
                                    data.fetch_data_by_str(predictor)
                                )
                                if trace is None or float(np.std(trace)) <= 1e-12:
                                    continue
                                if target_trace is None:
                                    continue
                                correlation = spearmanr(
                                    target_trace, trace, nan_policy="omit"
                                ).statistic
                                if (
                                    np.isfinite(correlation)
                                    and abs(float(correlation))
                                    >= self.config.feature_subset_min_abs_spearman
                                ):
                                    varying_predictors.append(predictor)
                            # If the specification declares only one predictor,
                            # its singleton is identical to the full run.
                            # Otherwise search every data-selected singleton
                            # symmetrically; no name or unit receives priority.
                            declared_predictor_count = sum(
                                predictor != item.target
                                for predictor in item.scalars
                            )
                            if (
                                varying_predictors
                                and declared_predictor_count > 1
                            ):
                                for predictor in varying_predictors:
                                    subset = GenericSearchSpec(
                                        item.target,
                                        {
                                            item.target: item.scalars.get(
                                                item.target, ""
                                            ),
                                            predictor: item.scalars[predictor],
                                        },
                                        {},
                                    )
                                    subset_results = self.discover_generic_typed(
                                        data, subset, debug=debug
                                    )
                                    run_results.append((predictor, subset_results))
                                    run_audits.append({
                                        "feature_subset": predictor,
                                        **dict(self.last_search_audit),
                                    })

                        # Round-robin across runs.  This reserves an equal
                        # first opportunity for every feature subset instead
                        # of letting the all-feature Pareto front consume the
                        # complete proposal budget.
                        evolutionary_results: list[tuple[Exp, ExpData]] = []
                        evolutionary_seen: set[str] = set()
                        max_run_length = max(
                            (len(results) for _name, results in run_results),
                            default=0,
                        )
                        for rank in range(max_run_length):
                            for _name, results in run_results:
                                if rank >= len(results):
                                    continue
                                candidate = results[rank]
                                rendered = str(candidate[0])
                                if rendered in evolutionary_seen:
                                    continue
                                evolutionary_seen.add(rendered)
                                evolutionary_results.append(candidate)
                        if len(run_audits) == 1:
                            # Preserve the direct audit schema for vector laws
                            # and genuinely univariate specifications.
                            evolutionary_audit = dict(run_audits[0])
                            evolutionary_audit.pop("feature_subset", None)
                        else:
                            evolutionary_audit = {
                                "mode": "generic_typed_feature_subset_ensemble",
                                "structured_candidates": 0,
                                "domain_templates": 0,
                                "model_fit_calls": sum(
                                    int(audit.get("model_fit_calls", 0))
                                    for audit in run_audits
                                ),
                                "model_equations": sum(
                                    int(audit.get("model_equations", 0))
                                    for audit in run_audits
                                ),
                                "runs": run_audits,
                                "accepted": [
                                    str(expression)
                                    for expression, _data in evolutionary_results
                                ],
                            }
                        self.last_search_audit = evolutionary_audit
                        if evolutionary_results and dimensional_audit is not None:
                            evolutionary_audit["dimensional_companion"] = (
                                dimensional_audit
                            )
                            self.last_search_audit = evolutionary_audit
                        elif dimensional_audit is not None:
                            dimensional_audit["evolutionary_companion"] = (
                                evolutionary_audit
                            )
                            self.last_search_audit = dimensional_audit
                        # Reserve room for at least one genuinely flexible
                        # candidate whenever PySR finds one.
                        evolutionary_budget = min(
                            len(evolutionary_results),
                            max(1, self.config.max_candidates - 1),
                        )
                        dimensional_budget = (
                            self.config.max_candidates - evolutionary_budget
                        )
                        target_results.extend(
                            dimensional_results[:dimensional_budget]
                        )
                        target_results.extend(
                            evolutionary_results[:evolutionary_budget]
                        )
                except BaseException as error:
                    if isinstance(error, (KeyboardInterrupt, SystemExit)):
                        raise
                    self.last_search_audit = {
                        "mode": "generic_target_skipped",
                        "structured_candidates": 0,
                        "domain_templates": 0,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    if debug:
                        print(
                            f"[PySR] skipped target {item.target}: "
                            f"{type(error).__name__}: {error}",
                            flush=True,
                        )
                child_audit = dict(self.last_search_audit)
                child_audit["target"] = item.target
                audits.append(child_audit)
                for expression, result_data in target_results:
                    key = str(expression)
                    if key not in seen:
                        seen.add(key)
                        all_results.append((expression, result_data))
            if len(audits) == 1:
                self.last_search_audit = audits[0]
                self.last_search_audit["entrypoint"] = "discover"
            else:
                self.last_search_audit = {
                    "mode": "discover_generic_main",
                    "entrypoint": "discover",
                    "structured_candidates": 0,
                    "domain_templates": 0,
                    "targets": audits,
                }
            return all_results[: self.config.max_candidates]

        if not self.available:
            if debug:
                print(f"[PySR] unavailable; baseline retained: {self.availability_error}", flush=True)
            return []
        features = self.extract_features(data)
        expressions = " ".join(feature.expression.lower() for feature in features)
        has_fields = "efield" in expressions or "bfield" in expressions

        def target_priority(feature: FeatureInfo) -> tuple[int, int, str]:
            if not self.config.enable_domain_templates:
                return (0, -feature.derivative_order, feature.name)
            expression = feature.expression.lower()
            if "intensity" in expression and feature.derivative_order == 0:
                return (0, 0, feature.name)
            if any(name in expression for name in ("probability", "density", "pressure")) \
                    and feature.derivative_order == 0:
                return (0, 0, feature.name)
            if "force" in expression and feature.derivative_order == 0:
                return (0, 0, feature.name)
            if has_fields and "pos" in expression and feature.derivative_order == 2:
                return (0, 0, feature.name)
            if "temp" in expression and feature.derivative_order in {0, 1}:
                return (1, -feature.derivative_order, feature.name)
            return (2, -feature.derivative_order, feature.name)

        dynamic = [
            feature
            for feature in features
            if np.std(feature.values) > 1e-10
            and not feature.engineered
            and feature.name != "time"
            and "efield" not in feature.expression.lower()
            and "bfield" not in feature.expression.lower()
        ]
        targets = sorted(
            dynamic,
            key=target_priority,
        )[: self.config.max_targets]
        results: list[tuple[Exp, ExpData]] = []
        seen: set[str] = set()
        for target in targets:
            usable = self._usable_features(target, features)[: self.config.max_features]
            if not usable:
                continue
            sample_count = target.values.size
            split = int(sample_count * (1.0 - self.config.holdout_fraction))
            split = min(max(split, 4), sample_count - 4)
            matrix = np.column_stack([feature.values for feature in usable])
            validated_for_target: list[
                tuple[int, float, Exp, ExpData, CandidateDiagnostics]
            ] = []
            equations = self._structured_candidates(target, usable, split)
            accepted_complexity: int | None = None
            for equation, complexity in equations:
                # Once a candidate passes at the lowest successful complexity,
                # a more complicated closure cannot be the preferred law.
                if accepted_complexity is not None and complexity > accepted_complexity:
                    break
                validated = self._validate_candidate(target, usable, equation, split)
                if validated is None:
                    continue
                expression, result_data, diagnostics = validated
                key = str(expression)
                if key in seen:
                    continue
                validated_for_target.append(
                    (
                        complexity,
                        diagnostics.holdout_nrmse,
                        expression,
                        result_data,
                        diagnostics,
                    )
                )
                accepted_complexity = complexity
            if not validated_for_target:
                model = self._make_model()
                try:
                    model.fit(
                        matrix[:split],
                        target.values[:split],
                        variable_names=[feature.name for feature in usable],
                    )
                except Exception as error:
                    if debug:
                        print(f"[PySR] fit failed for {target.expression}: {error}", flush=True)
                    continue
                for equation, complexity in _equation_rows(getattr(model, "equations_", None)):
                    validated = self._validate_candidate(target, usable, equation, split)
                    if validated is None:
                        continue
                    expression, result_data, diagnostics = validated
                    if str(expression) in seen:
                        continue
                    validated_for_target.append(
                        (
                            complexity,
                            diagnostics.holdout_nrmse,
                            expression,
                            result_data,
                            diagnostics,
                        )
                    )
            if validated_for_target:
                # All equations explain the same supervised target.  Keeping
                # the simplest validated one prevents the proposal stage from
                # emitting a ladder of near-identical fitted corrections.
                _, _, expression, result_data, diagnostics = min(
                    validated_for_target, key=lambda item: (item[0], item[1])
                )
                seen.add(str(expression))
                results.append((expression, result_data))
                if debug:
                    print(
                        f"[PySR] accepted {expression}; "
                        f"train={diagnostics.train_nrmse:.4g}, "
                        f"holdout={diagnostics.holdout_nrmse:.4g}",
                        flush=True,
                    )
            if len(results) >= self.config.max_candidates:
                break
        return results


_DEFAULT_DISCOVERY: PySRDiscovery | None = None


def get_default_discovery() -> PySRDiscovery:
    global _DEFAULT_DISCOVERY
    if _DEFAULT_DISCOVERY is None:
        config = PySRConfig(
            niterations=int(os.getenv("AI_NEWTON_PYSR_ITERATIONS", "12")),
            populations=int(os.getenv("AI_NEWTON_PYSR_POPULATIONS", "4")),
            max_targets=int(os.getenv("AI_NEWTON_PYSR_TARGETS", "2")),
            max_candidates=int(
                os.getenv("AI_NEWTON_PYSR_MAX_CANDIDATES", "4")
            ),
            max_complexity=int(
                os.getenv("AI_NEWTON_PYSR_MAX_COMPLEXITY", "16")
            ),
            timeout_seconds=float(
                os.getenv("AI_NEWTON_PYSR_TIMEOUT_SECONDS", "60")
            ),
            max_rational_denominator=int(
                os.getenv("AI_NEWTON_PYSR_MAX_DENOMINATOR", "64")
            ),
            enable_feature_subset_search=(
                os.getenv("AI_NEWTON_PYSR_FEATURE_SUBSETS", "1") == "1"
            ),
            feature_subset_min_abs_spearman=float(
                os.getenv("AI_NEWTON_PYSR_SUBSET_MIN_SPEARMAN", "0.98")
            ),
        )
        _DEFAULT_DISCOVERY = PySRDiscovery(config)
    return _DEFAULT_DISCOVERY


def propose_relations_pysr(
    data: DataStruct,
    debug: bool = False,
    cpu_num: int = 50,
    *,
    specification: GenericSearchSpec | Sequence[GenericSearchSpec] | None = None,
    old_formula: str | Mapping[str, str] | None = None,
) -> list[tuple[Exp, ExpData]]:
    del cpu_num  # PySR owns its search scheduler.
    discovery = get_default_discovery()
    results = discovery.discover(
        data,
        debug=debug,
        specification=specification,
        old_formula=old_formula,
    )
    audit_path = os.getenv("AI_NEWTON_PYSR_AUDIT_PATH")
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(
                discovery.last_search_audit, sort_keys=True, default=str
            ) + "\n")
    return results
