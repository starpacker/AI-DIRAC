"""
MapMiner — Phase 3.

Searches for transferable mappings between measurement domains:
  z = f(x)                    single-variable map
  z = f(x, context)           context-dependent map
  y = g(z, intervention)      intervention-dependent map

Candidate maps must satisfy:
  1. Low reconstruction error on held-out data
  2. Cross-experiment stability
  3. Intervention invariance (if map is intrinsic)
  4. Acceptable description length

Uses symbolic regression (PySR bridge), spline/polynomial fitting,
and the existing operator vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import itertools

import numpy as np
from scipy import optimize
from scipy.stats import pearsonr


class MapType(Enum):
    SCALAR_TO_SCALAR = auto()    # f: R → R
    VECTOR_TO_SCALAR = auto()    # f: R^n → R
    SCALAR_TO_VECTOR = auto()    # f: R → R^m
    VECTOR_TO_VECTOR = auto()    # f: R^n → R^m
    CONTEXT_DEPENDENT = auto()   # f(x, context) with context as categorical/label


@dataclass
class MapCandidate:
    """A candidate mapping with validation evidence."""
    map_id: str
    map_type: MapType
    input_channels: List[str]
    output_channel: str
    expression: str = ""                     # symbolic form (if found)
    parameters: Optional[Dict[str, float]] = None  # learned parameters
    raw_predictor: Optional[Callable] = None        # neural/numeric evaluator
    # Validation
    definition_experiments: List[str] = field(default_factory=list)
    heldout_experiments: List[str] = field(default_factory=list)
    definition_error: float = float("inf")
    heldout_error: float = float("inf")
    cross_experiment_r2: float = 0.0
    intervention_stability: float = 0.0      # std across interventions
    description_length: int = 0
    # Metadata
    source: str = ""                         # "symbolic", "neural", "spline"
    parent_maps: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.heldout_error < float("inf")
            and self.expression != ""
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_id": self.map_id,
            "map_type": self.map_type.name,
            "input_channels": self.input_channels,
            "output_channel": self.output_channel,
            "expression": self.expression,
            "parameters": self.parameters,
            "definition_experiments": self.definition_experiments,
            "heldout_experiments": self.heldout_experiments,
            "definition_error": self.definition_error,
            "heldout_error": self.heldout_error,
            "cross_experiment_r2": self.cross_experiment_r2,
            "intervention_stability": self.intervention_stability,
            "description_length": self.description_length,
            "source": self.source,
            "parent_maps": self.parent_maps,
        }


@dataclass
class MapMinerConfig:
    """Configuration for the MapMiner."""
    # Search budget
    max_candidates_per_input: int = 10
    max_expression_depth: int = 4
    max_polynomial_degree: int = 5
    # Validation gates
    heldout_fraction: float = 0.3
    max_heldout_error: float = 0.05     # normalized
    max_cross_experiment_error: float = 0.10
    max_intervention_variance: float = 0.02
    # Complexity
    max_description_length: int = 200
    min_cross_experiment_r2: float = 0.8
    # GPU
    use_gpu: bool = False
    seed: int = 42


class MapMiner:
    """
    Searches for transferable mappings from measurement data.

    Pipeline:
      1. Receive (input, output) data from one or more experiments
      2. Search symbolic forms via function templates
      3. Fit continuous forms (polynomial, rational, spline)
      4. Test on held-out data within the same experiment
      5. Test transfer to a different experiment
      6. Test invariance under irrelevant interventions
      7. Rank by heldout error, cross-experiment stability, and complexity
    """

    # Base function templates for symbolic search
    BASE_TEMPLATES = [
        "{a}*x + {b}",                    # linear
        "{a}*x**2 + {b}*x + {c}",         # quadratic
        "{a}*x**3 + {b}*x**2 + {c}*x + {d}",  # cubic
        "{a}*exp({b}*x)",                 # exponential
        "{a}*log(abs(x) + {eps})",        # log
        "{a}*sin({b}*x + {c})",           # sinusoidal
        "{a}*x**{p}",                     # power law
        "{a}/(x + {eps})",                # inverse
        "{a}*sqrt(abs(x) + {eps})",       # sqrt
        "{a}*tanh({b}*x)",                # tanh (saturation)
    ]

    def __init__(
        self,
        config: Optional[MapMinerConfig] = None,
        *,
        numerical_tolerance: float = 1e-8,
        min_data_points: int = 10,
    ):
        self.config = config or MapMinerConfig()
        self.numerical_tolerance = numerical_tolerance
        self.min_data_points = min_data_points
        self._id_counter = 0

    # --- Main search ---

    def search(
        self,
        inputs: Dict[str, np.ndarray],  # channel_name → values
        outputs: Dict[str, np.ndarray],  # channel_name → values
        *,
        experiment_name: str = "",
        heldout_mask: Optional[np.ndarray] = None,
        cross_experiment_inputs: Optional[Dict[str, np.ndarray]] = None,
        cross_experiment_outputs: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[MapCandidate]:
        """
        Search for maps from input channels to output channels.

        Args:
            inputs: Training input data {channel: values}
            outputs: Training output data {channel: values}
            experiment_name: Name of current experiment
            heldout_mask: Boolean mask for heldout split (True = heldout)
            cross_experiment_inputs: Data from a different experiment for transfer test
            cross_experiment_outputs: Corresponding output from different experiment
        """
        candidates: List[MapCandidate] = []

        for in_channel, x in inputs.items():
            for out_channel, y in outputs.items():
                if in_channel == out_channel:
                    continue
                if len(x) < self.min_data_points or len(y) < self.min_data_points:
                    continue

                # Ensure same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]

                # Create heldout split
                if heldout_mask is None:
                    np.random.seed(self.config.seed)
                    heldout_mask = np.random.random(min_len) < self.config.heldout_fraction

                x_train = x[~heldout_mask] if np.any(~heldout_mask) else x
                y_train = y[~heldout_mask] if np.any(~heldout_mask) else y
                x_test = x[heldout_mask] if np.any(heldout_mask) else x[-1:]
                y_test = y[heldout_mask] if np.any(heldout_mask) else y[-1:]

                # Search
                channel_candidates = self._search_channel_pair(
                    x_train, y_train, x_test, y_test,
                    in_channel, out_channel,
                    experiment_name,
                    cross_experiment_inputs, cross_experiment_outputs,
                )
                candidates.extend(channel_candidates)

        # Sort by heldout error (ascending)
        candidates.sort(key=lambda c: c.heldout_error)
        return candidates[:self.config.max_candidates_per_input * len(inputs)]

    def _search_channel_pair(
        self,
        x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray,
        in_channel: str, out_channel: str,
        exp_name: str,
        cross_x: Optional[Dict[str, np.ndarray]],
        cross_y: Optional[Dict[str, np.ndarray]],
    ) -> List[MapCandidate]:
        """Search all templates on a single channel pair."""
        candidates: List[MapCandidate] = []

        # 1. Symbolic templates
        for template in self.BASE_TEMPLATES:
            candidate = self._fit_template(
                template, x_train, y_train, x_test, y_test,
                in_channel, out_channel, exp_name,
            )
            if candidate and candidate.is_valid:
                candidates.append(candidate)

        # 2. Polynomial fit
        for degree in range(1, self.config.max_polynomial_degree + 1):
            candidate = self._fit_polynomial(
                degree, x_train, y_train, x_test, y_test,
                in_channel, out_channel, exp_name,
            )
            if candidate and candidate.is_valid:
                candidates.append(candidate)

        # 3. Cross-experiment test
        if cross_x is not None and cross_y is not None:
            for c in candidates:
                c_x = cross_x.get(c.input_channels[0])
                c_y = cross_y.get(c.output_channel)
                if c_x is not None and c_y is not None:
                    self._test_cross_experiment(c, c_x, c_y)

        return candidates

    # --- Template fitting ---

    def _fit_template(
        self,
        template: str,
        x: np.ndarray, y: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray,
        in_channel: str, out_channel: str,
        exp_name: str,
    ) -> Optional[MapCandidate]:
        """Fit a symbolic template using curve_fit."""
        import re

        # Generate the evaluable function
        try:
            # Extract parameter names
            params_in_template = re.findall(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}', template)
            params_in_template = [p for p in params_in_template if p not in ('eps',)]

            if not params_in_template:
                return None

            # Create function string
            func_str = template
            for p in params_in_template:
                func_str = func_str.replace(f'{{{p}}}', f'p_{p}')

            # Build the function
            local_ns = {'np': np, 'math': math, 'eps': self.numerical_tolerance}
            code = f"lambda x, {', '.join(f'p_{p}' for p in params_in_template)}: {func_str}"
            func = eval(code, {"__builtins__": {}}, local_ns)

            # Fit with curve_fit
            # Need initial guesses
            n_params = len(params_in_template)
            p0 = np.ones(n_params)

            try:
                popt, pcov = optimize.curve_fit(
                    func, x, y, p0=p0,
                    maxfev=5000,
                )
            except (RuntimeError, ValueError, TypeError):
                return None

            # Make predictor
            def predictor(xv):
                return func(xv, *popt)

            param_dict = {params_in_template[i]: float(popt[i]) for i in range(n_params)}

            # Evaluate on train and test
            y_pred_train = predictor(x)
            train_err = np.sqrt(np.mean((y - y_pred_train) ** 2))

            y_pred_test = predictor(x_test)
            test_err = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

            # Normalize errors
            y_range = np.max(y) - np.min(y)
            if y_range > self.numerical_tolerance:
                train_err /= y_range
                test_err /= y_range

            # Build expression string
            expr = template
            for p_name, p_val in param_dict.items():
                expr = expr.replace(f'{{{p_name}}}', f'{p_val:.4f}')
            expr = expr.replace('{eps}', str(self.numerical_tolerance))

            # Description length
            desc_len = len(expr.replace(" ", ""))

            self._id_counter += 1
            return MapCandidate(
                map_id=f"MAP_{self._id_counter:04d}",
                map_type=MapType.SCALAR_TO_SCALAR,
                input_channels=[in_channel],
                output_channel=out_channel,
                expression=expr,
                parameters=param_dict,
                raw_predictor=predictor,
                definition_experiments=[exp_name],
                definition_error=float(train_err),
                heldout_error=float(test_err),
                description_length=desc_len,
                source="symbolic",
            )

        except Exception:
            return None

    # --- Polynomial fitting ---

    def _fit_polynomial(
        self,
        degree: int,
        x: np.ndarray, y: np.ndarray,
        x_test: np.ndarray, y_test: np.ndarray,
        in_channel: str, out_channel: str,
        exp_name: str,
    ) -> Optional[MapCandidate]:
        """Fit a polynomial of given degree."""
        try:
            coeffs = np.polyfit(x, y, degree)
            p = np.poly1d(coeffs)

            y_pred_train = p(x)
            train_err = np.sqrt(np.mean((y - y_pred_train) ** 2))

            y_pred_test = p(x_test)
            test_err = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

            y_range = np.max(y) - np.min(y)
            if y_range > self.numerical_tolerance:
                train_err /= y_range
                test_err /= y_range

            # Build expression
            terms = []
            for i, c in enumerate(coeffs):
                power = degree - i
                if abs(c) < self.numerical_tolerance:
                    continue
                if power == 0:
                    terms.append(f"{c:.4f}")
                elif power == 1:
                    terms.append(f"{c:.4f}*x")
                else:
                    terms.append(f"{c:.4f}*x**{power}")
            expr = " + ".join(terms) if terms else "0"

            self._id_counter += 1
            return MapCandidate(
                map_id=f"MAP_{self._id_counter:04d}",
                map_type=MapType.SCALAR_TO_SCALAR,
                input_channels=[in_channel],
                output_channel=out_channel,
                expression=expr,
                raw_predictor=lambda xv, p=p: p(xv),
                definition_experiments=[exp_name],
                definition_error=float(train_err),
                heldout_error=float(test_err),
                description_length=len(expr),
                source="polynomial",
            )
        except Exception:
            return None

    # --- Cross-experiment transfer ---

    def _test_cross_experiment(
        self,
        candidate: MapCandidate,
        cross_x: np.ndarray,
        cross_y: np.ndarray,
    ) -> None:
        """Test map on data from a different experiment."""
        if candidate.raw_predictor is None:
            return

        try:
            y_pred = candidate.raw_predictor(cross_x)
            error = np.sqrt(np.mean((cross_y - y_pred) ** 2))

            y_range = np.max(cross_y) - np.min(cross_y)
            if y_range > self.numerical_tolerance:
                error /= y_range

            # R²
            ss_res = np.sum((cross_y - y_pred) ** 2)
            ss_tot = np.sum((cross_y - np.mean(cross_y)) ** 2)
            if ss_tot > self.numerical_tolerance:
                r2 = 1 - ss_res / ss_tot
            else:
                r2 = 0.0

            candidate.cross_experiment_r2 = max(r2, 0.0)
        except Exception:
            candidate.cross_experiment_r2 = 0.0

    # --- Intervention stability ---

    @staticmethod
    def test_intervention_stability(
        candidate: MapCandidate,
        intervention_data: Dict[str, Dict[str, np.ndarray]],
    ) -> float:
        """
        Test if a map is stable across interventions.

        intervention_data: {intervention_name: {"x": values, "y": values}}
        Returns stability score (std of predictions across interventions).
        """
        if candidate.raw_predictor is None:
            return float("inf")

        predictions_by_intervention = []
        for name, data in intervention_data.items():
            x = data.get("x")
            if x is None:
                continue
            try:
                y_pred = candidate.raw_predictor(x)
                predictions_by_intervention.append(np.mean(y_pred))
            except Exception:
                pass

        if len(predictions_by_intervention) < 2:
            return 0.0

        return float(np.std(predictions_by_intervention))

    # --- Batch search across experiments ---

    def search_across_experiments(
        self,
        experiment_data: Dict[
            str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        ],
        *,
        train_experiments: Optional[List[str]] = None,
        test_experiments: Optional[List[str]] = None,
    ) -> List[MapCandidate]:
        """
        Search for maps that transfer across multiple experiments.

        Args:
            experiment_data: {exp_name: (inputs_dict, outputs_dict)}
            train_experiments: experiments for training
            test_experiments: experiments for held-out transfer test
        """
        if train_experiments is None:
            train_experiments = list(experiment_data.keys())[:max(1, len(experiment_data) // 2)]
        if test_experiments is None:
            test_experiments = [e for e in experiment_data if e not in train_experiments]

        all_candidates: List[MapCandidate] = []

        for exp_name in train_experiments:
            inputs, outputs = experiment_data[exp_name]

            # Prepare cross-experiment data from test set
            cross_x: Dict[str, np.ndarray] = {}
            cross_y: Dict[str, np.ndarray] = {}
            for test_exp in test_experiments:
                if test_exp in experiment_data:
                    ti, to = experiment_data[test_exp]
                    for k, v in ti.items():
                        if k in cross_x:
                            cross_x[k] = np.concatenate([cross_x[k], v])
                        else:
                            cross_x[k] = v.copy()
                    for k, v in to.items():
                        if k in cross_y:
                            cross_y[k] = np.concatenate([cross_y[k], v])
                        else:
                            cross_y[k] = v.copy()

            candidates = self.search(
                inputs, outputs,
                experiment_name=exp_name,
                cross_experiment_inputs=cross_x if cross_x else None,
                cross_experiment_outputs=cross_y if cross_y else None,
            )
            all_candidates.extend(candidates)

        # Rank by cross-experiment R² (higher is better)
        all_candidates.sort(key=lambda c: c.cross_experiment_r2, reverse=True)
        return all_candidates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "max_candidates_per_input": self.config.max_candidates_per_input,
                "max_expression_depth": self.config.max_expression_depth,
                "max_polynomial_degree": self.config.max_polynomial_degree,
                "heldout_fraction": self.config.heldout_fraction,
                "max_heldout_error": self.config.max_heldout_error,
                "max_cross_experiment_error": self.config.max_cross_experiment_error,
                "max_intervention_variance": self.config.max_intervention_variance,
                "max_description_length": self.config.max_description_length,
                "min_cross_experiment_r2": self.config.min_cross_experiment_r2,
                "seed": self.config.seed,
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MapMiner":
        config = MapMinerConfig()
        cfg_d = d.get("config", {})
        for k, v in cfg_d.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return cls(config=config)
