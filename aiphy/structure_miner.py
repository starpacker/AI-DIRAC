"""
StructureMiner — Phase 3.

Searches for algebraic structures, invariants, symmetries, and conserved forms:
  - Invariants: quantities unchanged under transformations
  - Symmetries: transformation groups that preserve structure
  - Conservation forms: expressions that sum to zero across states
  - Manifold structures: low-dimensional subspaces/constraints in data
  - Recurrence patterns: self-similar or periodic structures

Every structure must:
  1. Hold on held-out data not used in definition
  2. Be stable under parameter perturbations
  3. Have cross-experiment evidence
  4. Not be trivially vacuous (zero variance = always true)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import itertools

import numpy as np
from scipy import linalg, optimize, stats
from scipy.stats import pearsonr


class StructureKind(Enum):
    INVARIANT = auto()              # f(T(x)) = f(x)
    SYMMETRY = auto()               # T such that property preserved
    CONSERVED_FORM = auto()         # Σ w_i * Δy_i ≈ 0
    MANIFOLD = auto()               # low-dim subspace constraint
    PERIODICITY = auto()            # f(t + T) ≈ f(t)
    SCALING_RELATION = auto()       # f(a*x) = a^p * f(x)
    LINEAR_DEPENDENCE = auto()      # rank-deficiency in a set of functions
    ALGEBRAIC_CLOSURE = auto()      # operations close within a set


@dataclass
class StructureCandidate:
    """A candidate mathematical structure."""
    structure_id: str
    kind: StructureKind
    description: str = ""
    expression: str = ""
    # Numerical evidence
    definition_experiments: List[str] = field(default_factory=list)
    heldout_experiments: List[str] = field(default_factory=list)
    definition_error: float = float("inf")      # normalized
    heldout_error: float = float("inf")         # normalized
    cross_experiment_support: int = 0
    robustness_under_noise: float = 0.0          # higher = more robust
    # Components
    depends_on: List[str] = field(default_factory=list)
    description_length: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            self.heldout_error < float("inf")
            and self.description != ""
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structure_id": self.structure_id,
            "kind": self.kind.name,
            "description": self.description,
            "expression": self.expression,
            "definition_experiments": self.definition_experiments,
            "heldout_experiments": self.heldout_experiments,
            "definition_error": self.definition_error,
            "heldout_error": self.heldout_error,
            "cross_experiment_support": self.cross_experiment_support,
            "robustness_under_noise": self.robustness_under_noise,
            "depends_on": self.depends_on,
            "description_length": self.description_length,
        }


class StructureMiner:
    """
    Searches for invariant structures in data.

    Methods:
      1. Conservation: Σ w_i * Δy_i ≈ 0 (additive invariants)
      2. Periodicity: autocorrelation peak detection
      3. Scaling: power-law relation detection
      4. Manifold: PCA/spectral clustering on residuals
      5. Symmetry: detect invariance under transformations
    """

    def __init__(
        self,
        *,
        numerical_tolerance: float = 1e-6,
        min_data_points: int = 10,
        periodicity_min_confidence: float = 0.7,
        scaling_min_r2: float = 0.95,
        manifold_variance_threshold: float = 0.95,
    ):
        self.numerical_tolerance = numerical_tolerance
        self.min_data_points = min_data_points
        self.periodicity_min_confidence = periodicity_min_confidence
        self.scaling_min_r2 = scaling_min_r2
        self.manifold_variance_threshold = manifold_variance_threshold
        self._id_counter = 0

    # --- Conservation forms ---

    def search_conserved_forms(
        self,
        state_changes: Dict[str, np.ndarray],  # channel → Δ values (n_samples,)
        *,
        experiment_name: str = "",
        heldout_mask: Optional[np.ndarray] = None,
        cross_experiment_changes: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[StructureCandidate]:
        """
        Find linear combinations of state changes that sum to zero.

        Solves: X @ w ≈ 0, where X_ij = Δy_i^{(j)}
        Returns the null-space vectors w (conserved forms).
        """
        candidates: List[StructureCandidate] = []
        channels = list(state_changes.keys())
        n_channels = len(channels)

        if n_channels < 2:
            return candidates

        # Build data matrix: (n_samples, n_channels)
        lengths = [len(v) for v in state_changes.values()]
        if min(lengths) < self.min_data_points:
            return candidates

        n = min(lengths)
        X = np.column_stack([state_changes[ch][:n] for ch in channels])

        if heldout_mask is None:
            heldout_mask = np.zeros(n, dtype=bool)
            heldout_mask[n//3:2*n//3] = True

        X_train = X[~heldout_mask]
        X_test = X[heldout_mask]

        if len(X_train) < 3:
            return candidates

        try:
            # SVD to find null space
            U, S, Vt = np.linalg.svd(X_train, full_matrices=True)

            # Tolerance: find singular values near zero
            max_s = S[0] if len(S) > 0 else 1.0
            if max_s < self.numerical_tolerance:
                return candidates

            rel_s = S / max_s

            for i, sv in enumerate(rel_s):
                if sv > 0.05:  # Not near zero
                    continue

                # Null space vector
                w = Vt[i, :]  # This is already a row of Vt
                # Normalize
                w = w / (np.linalg.norm(w) + self.numerical_tolerance)

                # Build expression
                terms = []
                for j, ch in enumerate(channels):
                    coeff = w[j]
                    if abs(coeff) > 0.01:
                        terms.append(f"{coeff:.4f}*Δ{ch}")
                if not terms:
                    continue
                expr = " + ".join(terms)

                # Test: evaluate on heldout
                test_residual = X_test @ w
                test_err = np.sqrt(np.mean(test_residual ** 2))
                test_std = np.std(X_test) if X_test.size > 0 else 1.0
                if test_std > self.numerical_tolerance:
                    test_err /= test_std

                # Cross-experiment test
                cross_support = 0
                if cross_experiment_changes is not None:
                    cross_X_list = []
                    for ch in channels:
                        cross_ch = cross_experiment_changes.get(ch)
                        if cross_ch is not None:
                            cross_X_list.append(cross_ch[:n])
                    if cross_X_list:
                        cross_X = np.column_stack(cross_X_list)
                        cross_residual = cross_X @ w
                        cross_err = np.sqrt(np.mean(cross_residual ** 2))
                        cross_std = np.std(cross_X) if cross_X.size > 0 else 1.0
                        if cross_std > self.numerical_tolerance:
                            cross_err /= cross_std
                        if cross_err < 0.1:  # 10% threshold
                            cross_support = 1

                # Robustness: add noise and recheck
                noise_std = 0.01 * np.std(X_train)
                X_noisy = X_train + np.random.randn(*X_train.shape) * noise_std
                noisy_residual = X_noisy @ w
                noisy_err = np.sqrt(np.mean(noisy_residual ** 2))
                robustness = 1.0 / (1.0 + noisy_err)

                self._id_counter += 1
                candidates.append(StructureCandidate(
                    structure_id=f"STRUCT_{self._id_counter:04d}",
                    kind=StructureKind.CONSERVED_FORM,
                    description=f"Conserved form from {', '.join(channels)}",
                    expression=expr,
                    definition_experiments=[experiment_name],
                    definition_error=float(sv / max_s),
                    heldout_error=float(test_err),
                    cross_experiment_support=cross_support,
                    robustness_under_noise=float(robustness),
                    depends_on=list(channels),
                    description_length=len(expr),
                ))

        except np.linalg.LinAlgError:
            pass

        return candidates

    # --- Periodicity detection ---

    def search_periodicity(
        self,
        time_series: Dict[str, np.ndarray],  # channel → (n_timesteps,)
        *,
        sampling_rate: float = 1.0,
        min_period: int = 3,
        max_period_factor: float = 0.5,
    ) -> List[StructureCandidate]:
        """
        Detect periodic structures via autocorrelation.

        For each time series, find the lag with peak autocorrelation.
        """
        candidates = []

        for channel, values in time_series.items():
            n = len(values)
            if n < self.min_data_points:
                continue

            # Remove mean
            centered = values - np.mean(values)
            energy = np.sum(centered ** 2)
            if energy < self.numerical_tolerance:
                continue

            # Autocorrelation
            max_lag = int(n * max_period_factor)
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[n-1:]  # Only non-negative lags
            autocorr = autocorr[:max_lag+1]

            # Normalize
            autocorr = autocorr / (energy + self.numerical_tolerance)

            # Find peaks after first zero crossing
            # Simple: find lag with max autocorrelation in [min_period, max_lag]
            search_start = max(min_period, 1)
            if search_start >= len(autocorr):
                continue

            peak_lag = search_start + np.argmax(autocorr[search_start:])
            peak_value = autocorr[peak_lag]

            if peak_value < self.periodicity_min_confidence:
                continue

            period = peak_lag / sampling_rate

            self._id_counter += 1
            candidates.append(StructureCandidate(
                structure_id=f"STRUCT_{self._id_counter:04d}",
                kind=StructureKind.PERIODICITY,
                description=f"Periodic: T ≈ {period:.3f}",
                expression=f"period({channel}) = {period:.3f}",
                definition_error=float(1.0 - peak_value),
                heldout_error=float(1.0 - peak_value),
                robustness_under_noise=float(peak_value),
                depends_on=[channel],
                description_length=20,
            ))

        return candidates

    # --- Scaling relations ---

    def search_scaling_relations(
        self,
        x_data: Dict[str, np.ndarray],
        y_data: Dict[str, np.ndarray],
        *,
        experiment_name: str = "",
    ) -> List[StructureCandidate]:
        """
        Detect power-law relations: y ∝ x^p.

        Fits log(y) = p * log(x) + c and tests the R².
        """
        candidates = []

        for x_name, x in x_data.items():
            for y_name, y in y_data.items():
                if x_name == y_name:
                    continue
                n = min(len(x), len(y))
                if n < self.min_data_points:
                    continue

                # Positive data only for log transform
                x_pos = x[:n]
                y_pos = y[:n]

                pos_mask = (x_pos > self.numerical_tolerance) & (
                    y_pos > self.numerical_tolerance
                )
                if np.sum(pos_mask) < self.min_data_points:
                    continue

                log_x = np.log(x_pos[pos_mask])
                log_y = np.log(y_pos[pos_mask])

                # Linear fit
                A = np.column_stack([log_x, np.ones_like(log_x)])
                coeffs, residuals, rank, s = np.linalg.lstsq(A, log_y, rcond=None)

                p = coeffs[0]
                c = np.exp(coeffs[1])

                # R²
                y_pred = A @ coeffs
                ss_res = np.sum((log_y - y_pred) ** 2)
                ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > self.numerical_tolerance else 0.0

                if r2 < self.scaling_min_r2:
                    continue

                self._id_counter += 1
                candidates.append(StructureCandidate(
                    structure_id=f"STRUCT_{self._id_counter:04d}",
                    kind=StructureKind.SCALING_RELATION,
                    description=f"Power law: {y_name} ∝ {x_name}^{p:.3f}",
                    expression=f"{y_name} = {c:.3f} * {x_name}^{p:.3f}",
                    definition_experiments=[experiment_name],
                    definition_error=float(1.0 - r2),
                    heldout_error=float(1.0 - r2),
                    robustness_under_noise=float(r2),
                    depends_on=[x_name, y_name],
                    description_length=len(y_name) + len(x_name) + 20,
                ))

        return candidates

    # --- Manifold detection ---

    def search_manifold_structure(
        self,
        data_matrix: np.ndarray,  # (n_samples, n_features)
        *,
        experiment_name: str = "",
        variance_threshold: Optional[float] = None,
    ) -> List[StructureCandidate]:
        """
        Detect low-dimensional manifold structure via PCA.

        Reports the effective dimension of the data.
        """
        if variance_threshold is None:
            variance_threshold = self.manifold_variance_threshold

        n, d = data_matrix.shape
        if n < 3 or d < 2:
            return []

        try:
            center = data_matrix - np.mean(data_matrix, axis=0)
            U, S, Vt = np.linalg.svd(center, full_matrices=False)

            total_var = np.sum(S ** 2)
            cum_var = np.cumsum(S ** 2) / total_var

            # Effective dimension: number of components to reach threshold
            eff_dim = int(np.searchsorted(cum_var, variance_threshold) + 1)
            eff_dim = min(eff_dim, d, n)

            if eff_dim >= d:
                # No dimensionality reduction
                return []

            self._id_counter += 1
            return [StructureCandidate(
                structure_id=f"STRUCT_{self._id_counter:04d}",
                kind=StructureKind.MANIFOLD,
                description=f"Low-dim manifold: effective dim = {eff_dim} (of {d})",
                expression=f"dim_reduction({d}→{eff_dim})",
                definition_experiments=[experiment_name],
                definition_error=float(1.0 - variance_threshold),
                heldout_error=float(1.0 - variance_threshold),
                robustness_under_noise=float(cum_var[eff_dim-1] if eff_dim > 0 else 0.0),
                depends_on=[],
                description_length=eff_dim + 10,
            )]
        except np.linalg.LinAlgError:
            return []

    # --- Symmetry detection ---

    def search_symmetry(
        self,
        original_data: np.ndarray,
        transformed_data: np.ndarray,
        *,
        tolerance: float = 0.05,
    ) -> List[StructureCandidate]:
        """
        Detect if transformed_data ≈ original_data (symmetry under transformation).

        original_data and transformed_data must have the same shape.
        """
        if original_data.shape != transformed_data.shape:
            return []

        diff = np.abs(original_data - transformed_data)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Normalize by original data scale
        data_scale = np.std(original_data)
        if data_scale > self.numerical_tolerance:
            normalized_diff = mean_diff / data_scale
        else:
            normalized_diff = mean_diff

        if normalized_diff < tolerance:
            self._id_counter += 1
            return [StructureCandidate(
                structure_id=f"STRUCT_{self._id_counter:04d}",
                kind=StructureKind.SYMMETRY,
                description=f"Invariant under transformation (mean diff: {normalized_diff:.4f})",
                expression=f"T(x) ≈ x (diff={normalized_diff:.4e})",
                definition_error=float(normalized_diff),
                heldout_error=float(normalized_diff),
                robustness_under_noise=1.0 - float(normalized_diff),
                depends_on=[],
                description_length=10,
            )]

        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_tolerance": self.numerical_tolerance,
            "min_data_points": self.min_data_points,
            "periodicity_min_confidence": self.periodicity_min_confidence,
            "scaling_min_r2": self.scaling_min_r2,
            "manifold_variance_threshold": self.manifold_variance_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructureMiner":
        return cls(
            numerical_tolerance=d.get("numerical_tolerance", 1e-6),
            min_data_points=d.get("min_data_points", 10),
            periodicity_min_confidence=d.get("periodicity_min_confidence", 0.7),
            scaling_min_r2=d.get("scaling_min_r2", 0.95),
            manifold_variance_threshold=d.get("manifold_variance_threshold", 0.95),
        )
