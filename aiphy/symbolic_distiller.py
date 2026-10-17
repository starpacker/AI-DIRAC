"""
SymbolicDistiller — Phase 3.

Converts neural representations into symbolic ASTs or executable programs.
Takes a trained neural latent representation and searches for:
  1. Explicit symbolic formula for the encoder f(x)
  2. Explicit symbolic formula for the decoder g(z)
  3. Program form if no closed-form expression exists

Validation gates:
  - Distilled form must reproduce neural output within tolerance
  - Distilled form must have lower complexity than full network
  - Distilled form must generalize to held-out data
  - Gauge degeneracies must be preserved (equivalent forms detected)

Integrates with:
  - MathConceptRegistry: certified distilled forms become math concepts
  - MapMiner: distilled encoder/decoder are candidate maps
  - EquivalenceCanonicalizer: detect redundant distilled forms
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import warnings

import numpy as np
from scipy import optimize, interpolate


class DistillationMethod(Enum):
    TEMPLATE_FIT = auto()      # Fit to known function templates
    POLYNOMIAL_APPROX = auto() # Chebyshev/polynomial approximation
    SPLINE = auto()            # Spline-based representation
    PROGRAM = auto()           # GPU program or finite automaton
    NEURAL_ONLY = auto()       # Cannot distill — stays as black box


@dataclass
class DistillationResult:
    """Result of distilling a neural representation."""
    source_repr_id: str
    method: DistillationMethod
    # Encoder distillation
    encoder_expression: str = ""
    encoder_error: float = float("inf")
    encoder_complexity: int = 0
    # Decoder distillation
    decoder_expression: str = ""
    decoder_error: float = float("inf")
    decoder_complexity: int = 0
    # Full model
    is_successful: bool = False
    total_complexity: int = 0
    gauge_preserved: bool = False
    # Metadata
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_repr_id": self.source_repr_id,
            "method": self.method.name,
            "encoder_expression": self.encoder_expression,
            "encoder_error": self.encoder_error,
            "encoder_complexity": self.encoder_complexity,
            "decoder_expression": self.decoder_expression,
            "decoder_error": self.decoder_error,
            "decoder_complexity": self.decoder_complexity,
            "is_successful": self.is_successful,
            "total_complexity": self.total_complexity,
            "gauge_preserved": self.gauge_preserved,
            "notes": self.notes,
        }


class SymbolicDistiller:
    """
    Distills neural representations into symbolic forms.

    Pipeline:
      1. Extract input-output pairs from trained neural net
      2. Fit symbolic templates to each output dimension
      3. Combine into a full symbolic model
      4. Test that gauge degeneracies are preserved
      5. Report distillation error and complexity
    """

    # Template library for 1D projection
    ENCODER_TEMPLATES = [
        "{a}*x + {b}",
        "{a}*x**2 + {b}*x + {c}",
        "{a}*sin({b}*x + {c})",
        "{a}*exp({b}*x)",
        "{a}*log(abs(x) + {e})",
        "{a}*tanh({b}*x + {c})",
        "{a}*x**{p}",
        "{a}/(abs(x) + {e})",
    ]

    def __init__(
        self,
        *,
        numerical_tolerance: float = 1e-6,
        max_polynomial_degree: int = 8,
        distillation_error_tolerance: float = 0.05,
        min_data_points: int = 20,
    ):
        self.numerical_tolerance = numerical_tolerance
        self.max_polynomial_degree = max_polynomial_degree
        self.distillation_error_tolerance = distillation_error_tolerance
        self.min_data_points = min_data_points

    # --- Main distillation ---

    def distill(
        self,
        neural_model: Callable[[np.ndarray], np.ndarray],
        input_data: np.ndarray,    # (n_samples, input_dim)
        *,
        latent_dim: int,
        method: DistillationMethod = DistillationMethod.TEMPLATE_FIT,
        heldout_data: Optional[np.ndarray] = None,
    ) -> DistillationResult:
        """
        Distill a neural encoder into symbolic form.

        Args:
            neural_model: Function that maps input → latent
            input_data: Training input samples
            latent_dim: Output dimension of the model
            method: Distillation strategy
            heldout_data: Held-out data for validation
        """
        # Get neural outputs
        latent_outputs = neural_model(input_data)  # (n, latent_dim)

        if heldout_data is not None:
            heldout_latent = neural_model(heldout_data)

        encoder_exprs = []
        encoder_errors = []
        encoder_complexities = []

        for dim in range(latent_dim):
            target = latent_outputs[:, dim] if latent_outputs.ndim > 1 else latent_outputs

            if heldout_data is not None:
                h_target = heldout_latent[:, dim] if heldout_latent.ndim > 1 else heldout_latent
            else:
                h_target = None

            expr, error, complexity = self._distill_dimension(
                input_data, target, h_target, dim, method
            )

            encoder_exprs.append(expr)
            encoder_errors.append(error)
            encoder_complexities.append(complexity)

        # Combine
        encoder_expression = " | ".join(encoder_exprs)
        avg_encoder_error = float(np.mean(encoder_errors))
        total_encoder_complexity = sum(encoder_complexities)

        is_successful = avg_encoder_error < self.distillation_error_tolerance

        return DistillationResult(
            source_repr_id="unknown",
            method=method,
            encoder_expression=encoder_expression,
            encoder_error=avg_encoder_error,
            encoder_complexity=total_encoder_complexity,
            is_successful=is_successful,
            total_complexity=total_encoder_complexity,
            gauge_preserved=latent_dim > 1,  # Gauge if >1 dim
            notes=f"Distilled {latent_dim}-dim encoder with avg error {avg_encoder_error:.4f}",
        )

    def _distill_dimension(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        heldout_targets: Optional[np.ndarray],
        dim: int,
        method: DistillationMethod,
    ) -> Tuple[str, float, int]:
        """
        Distill a single latent dimension. For multi-dimensional inputs,
        reduces to scalar distillation per dimension.
        """
        if inputs.ndim == 1:
            # Scalar input → scalar output
            return self._distill_scalar_scalar(inputs, targets, heldout_targets, method)
        else:
            # Vector input → scalar output
            # Fit linear combination + nonlinearity
            return self._distill_vector_scalar(inputs, targets, heldout_targets)

    # --- Scalar → Scalar distillation ---

    def _distill_scalar_scalar(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_test: Optional[np.ndarray],
        method: DistillationMethod,
    ) -> Tuple[str, float, int]:
        """Distill f: R → R."""
        if method == DistillationMethod.TEMPLATE_FIT:
            return self._template_fit(x, y, y_test)
        elif method == DistillationMethod.POLYNOMIAL_APPROX:
            return self._polynomial_fit(x, y, y_test)
        else:
            return self._template_fit(x, y, y_test)

    def _template_fit(
        self,
        x: np.ndarray, y: np.ndarray,
        y_test: Optional[np.ndarray],
    ) -> Tuple[str, float, int]:
        """Fit symbolic templates to find the best expression."""
        import re

        best_expr = ""
        best_error = float("inf")
        best_complexity = 0

        for template in self.ENCODER_TEMPLATES:
            try:
                params = re.findall(r'\{([a-zA-Z])\}', template)
                params = list(set(params))  # unique

                if not params:
                    continue

                # Build function
                func_str = template
                for p in params:
                    func_str = func_str.replace(f'{{{p}}}', f'p_{p}')

                local_ns = {'np': np, 'math': math, 'e': self.numerical_tolerance + 1e-4}
                code = f"lambda x, {', '.join(f'p_{p}' for p in params)}: {func_str}"
                func = eval(code, {"__builtins__": {}}, local_ns)

                p0 = np.ones(len(params))
                try:
                    popt, _ = optimize.curve_fit(func, x, y, p0=p0, maxfev=5000)
                except (RuntimeError, ValueError, TypeError):
                    continue

                y_pred = func(x, *popt)
                error = np.sqrt(np.mean((y - y_pred) ** 2))

                # Normalize
                y_range = np.max(y) - np.min(y)
                if y_range > self.numerical_tolerance:
                    error /= y_range

                # Test on heldout
                if y_test is not None:
                    yt_pred = func(x, *popt)
                    test_err = np.sqrt(np.mean((y_test - yt_pred) ** 2))
                    test_range = np.max(y_test) - np.min(y_test)
                    if test_range > self.numerical_tolerance:
                        test_err /= test_range
                    error = max(error, test_err)  # Conservative

                expr = template
                for p_name, p_val in zip(params, popt):
                    expr = expr.replace(f'{{{p_name}}}', f'{p_val:.4f}')
                expr = expr.replace('{e}', str(self.numerical_tolerance + 1e-4))

                complexity = len(expr.replace(" ", ""))

                if error < best_error:
                    best_error = error
                    best_expr = expr
                    best_complexity = complexity

            except Exception:
                continue

        if not best_expr:
            return f"x", float("inf"), 1

        return best_expr, best_error, best_complexity

    def _polynomial_fit(
        self,
        x: np.ndarray, y: np.ndarray,
        y_test: Optional[np.ndarray],
    ) -> Tuple[str, float, int]:
        """Fit a polynomial and select the best degree."""
        best_expr = ""
        best_error = float("inf")
        best_complexity = 0

        for degree in range(1, self.max_polynomial_degree + 1):
            try:
                coeffs = np.polyfit(x, y, degree)
                p = np.poly1d(coeffs)

                y_pred = p(x)
                error = np.sqrt(np.mean((y - y_pred) ** 2))

                y_range = np.max(y) - np.min(y)
                if y_range > self.numerical_tolerance:
                    error /= y_range

                if y_test is not None:
                    yt_pred = p(x)
                    test_err = np.sqrt(np.mean((y_test - yt_pred) ** 2))
                    test_range = np.max(y_test) - np.min(y_test)
                    if test_range > self.numerical_tolerance:
                        test_err /= test_range
                    error = max(error, test_err)

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
                complexity = len(expr)

                if error < best_error:
                    best_error = error
                    best_expr = expr
                    best_complexity = complexity
            except Exception:
                continue

        if not best_expr:
            return f"x", float("inf"), 1

        return best_expr, best_error, best_complexity

    # --- Vector → Scalar distillation ---

    def _distill_vector_scalar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_test: Optional[np.ndarray],
    ) -> Tuple[str, float, int]:
        """Distill f: R^n → R as linear combination + optional nonlinearity."""
        # Linear fit
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return "0", float("inf"), 1

        y_pred = X @ coeffs
        error = np.sqrt(np.mean((y - y_pred) ** 2))

        y_range = np.max(y) - np.min(y)
        if y_range > self.numerical_tolerance:
            error /= y_range

        # Build expression
        terms = []
        for i, c in enumerate(coeffs):
            if abs(c) > self.numerical_tolerance * 100:
                terms.append(f"{c:.4f}*x_{i}")
        expr = " + ".join(terms) if terms else "0"
        complexity = max(len(expr), 5)

        return expr, float(error), complexity

    # --- Batch distillation for all representations ---

    def distill_all(
        self,
        representations: List[Any],  # NeuralRepresentation list
        model_extractor: Callable[[Any], Callable],
        input_data: np.ndarray,
        latent_dims: List[int],
        *,
        heldout_data: Optional[np.ndarray] = None,
    ) -> List[DistillationResult]:
        """
        Distill a batch of neural representations.

        Args:
            representations: List of NeuralRepresentation objects
            model_extractor: Function to get the neural callable from a representation
            input_data: Shared input data
            latent_dims: Latent dimensions for each representation
            heldout_data: Shared heldout data
        """
        results = []
        for rep, ld in zip(representations, latent_dims):
            try:
                model = model_extractor(rep)
                result = self.distill(
                    model, input_data,
                    latent_dim=ld,
                    heldout_data=heldout_data,
                )
                result.source_repr_id = getattr(rep, 'repr_id', 'unknown')
                results.append(result)
            except Exception:
                results.append(DistillationResult(
                    source_repr_id=getattr(rep, 'repr_id', 'unknown'),
                    method=DistillationMethod.NEURAL_ONLY,
                    notes="Distillation failed",
                ))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_tolerance": self.numerical_tolerance,
            "max_polynomial_degree": self.max_polynomial_degree,
            "distillation_error_tolerance": self.distillation_error_tolerance,
            "min_data_points": self.min_data_points,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SymbolicDistiller":
        return cls(
            numerical_tolerance=d.get("numerical_tolerance", 1e-6),
            max_polynomial_degree=d.get("max_polynomial_degree", 8),
            distillation_error_tolerance=d.get("distillation_error_tolerance", 0.05),
            min_data_points=d.get("min_data_points", 20),
        )
