"""
Equivalence canonicalizer — Phase 3.

Detects when two mathematical objects are the same up to:
  - affine scaling: f and a*f + b
  - variable renaming: f(x) and f(y)
  - algebraic equivalence: x*y and y*x, (x+1)^2 and x^2 + 2x + 1
  - gauge freedom: representations related by invertible transforms
  - latent rotation equivalence: neural representations up to orthogonal transforms

For each equivalence class, selects one canonical representative.
Integrates with MathConceptRegistry to automatically merge equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import math
import hashlib
import itertools

import numpy as np


class EquivalenceKind(Enum):
    AFFINE_SCALING = auto()      # f and a*f + b
    VARIABLE_RENAMING = auto()   # f(x,y) and f(y,x)
    ALGEBRAIC = auto()           # (x+1)^2 == x^2 + 2x + 1
    GAUGE_FREEDOM = auto()       # representations under invertible group action
    LATENT_ROTATION = auto()     # neural latents under orthogonal transform
    NUMERICAL_EQUIVALENT = auto()  # same numerical output everywhere


@dataclass
class EquivalenceResult:
    """Result of comparing two mathematical objects."""
    object_a: str            # concept_id or expression
    object_b: str
    are_equivalent: bool
    kind: Optional[EquivalenceKind] = None
    transform_description: str = ""
    confidence: float = 0.0  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)


class EquivalenceCanonicalizer:
    """
    Detects and merges equivalent mathematical representations.

    Two objects are equivalent if:
      1. They produce identical numerical output on all test points
      2. They can be transformed into each other by a known equivalence
      3. Their latent representations span the same subspace

    Canonicalization strategy:
      - For each equivalence group, select the simplest canonical form
      - "Simplest" = lowest description length (fewest ops + simplest constants)
    """

    def __init__(
        self,
        *,
        numerical_tolerance: float = 1e-6,
        min_test_points: int = 20,
        affine_scaling_threshold: float = 0.9999,  # Pearson r for affine detection
        max_affine_attempts: int = 5,
    ):
        self.numerical_tolerance = numerical_tolerance
        self.min_test_points = min_test_points
        self.affine_scaling_threshold = affine_scaling_threshold
        self.max_affine_attempts = max_affine_attempts

    # --- Numerical equivalence ---

    def check_numerical_equivalence(
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
    ) -> EquivalenceResult:
        """
        Check if two sets of values are numerically equivalent.
        Values should be evaluated at the same test points.
        """
        if len(values_a) != len(values_b):
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=False,
                confidence=0.0,
                evidence={"reason": "different number of test points"},
            )

        if len(values_a) < self.min_test_points:
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=False,
                confidence=0.0,
                evidence={"reason": f"too few test points: {len(values_a)}"},
            )

        # Check if values are identical within tolerance
        max_diff = np.max(np.abs(values_a - values_b))
        if max_diff < self.numerical_tolerance:
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=True,
                kind=EquivalenceKind.NUMERICAL_EQUIVALENT,
                confidence=1.0,
                evidence={"max_diff": float(max_diff), "identical": True},
            )

        # Check affine scaling: values_b ≈ a * values_a + b
        r = self._check_affine_scaling(values_a, values_b)
        if r is not None:
            a, b_coeff, pearson_r = r
            if pearson_r > self.affine_scaling_threshold:
                return EquivalenceResult(
                    object_a="a", object_b="b",
                    are_equivalent=True,
                    kind=EquivalenceKind.AFFINE_SCALING,
                    confidence=float(pearson_r),
                    transform_description=f"b = {a:.6f} * a + {b_coeff:.6f}",
                    evidence={"a_coeff": a, "b_coeff": b_coeff, "pearson_r": float(pearson_r)},
                )

        return EquivalenceResult(
            object_a="a", object_b="b",
            are_equivalent=False,
            confidence=0.0,
            evidence={"max_diff": float(max_diff)},
        )

    @staticmethod
    def _check_affine_scaling(
        x: np.ndarray, y: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Check if y ≈ a*x + b. Returns (a, b, pearson_r) or None."""
        if len(x) < 3:
            return None

        # Linear regression: y = a*x + b
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        den = np.sum((x - x_mean) ** 2)

        if abs(den) < 1e-16:
            return None

        a = num / den
        b_coeff = y_mean - a * x_mean

        # Pearson correlation
        r_num = np.sum((x - x_mean) * (y - y_mean))
        r_den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
        if abs(r_den) < 1e-16:
            return 0.0, b_coeff, 0.0

        r = r_num / r_den
        return a, b_coeff, abs(r)

    # --- Latent space equivalence ---

    def check_latent_equivalence(
        self,
        latent_a: np.ndarray,  # (n_points, dim_a)
        latent_b: np.ndarray,  # (n_points, dim_b)
        *,
        tolerance: float = 1e-4,
    ) -> EquivalenceResult:
        """
        Check if two latent representations span the same subspace.

        Uses SVD to find a linear map between the two spaces.
        If the reconstruction error is small, they are equivalent
        up to a linear transformation (gauge freedom).
        """
        if latent_a.shape[0] != latent_b.shape[0]:
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=False,
                evidence={"reason": "different number of samples"},
            )

        n, dim_a = latent_a.shape
        dim_b = latent_b.shape[1]

        if min(n, dim_a, dim_b) < 3:
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=False,
                evidence={"reason": "too few samples"},
            )

        # Center the data
        a_centered = latent_a - latent_a.mean(axis=0)
        b_centered = latent_b - latent_b.mean(axis=0)

        # Find linear map: b ≈ a @ M
        # M = (a^T a)^{-1} a^T b  (least squares)
        try:
            M, residuals, rank, s = np.linalg.lstsq(a_centered, b_centered, rcond=None)
        except np.linalg.LinAlgError:
            return EquivalenceResult(
                object_a="a", object_b="b",
                are_equivalent=False,
                evidence={"reason": "numerical instability"},
            )

        # Reconstruction error
        b_pred = a_centered @ M
        reconstruction_error = np.mean((b_centered - b_pred) ** 2)

        # Normalize by data variance
        b_var = np.mean(np.var(b_centered, axis=0))
        if b_var > 1e-16:
            normalized_error = reconstruction_error / b_var
        else:
            normalized_error = reconstruction_error

        is_equivalent = normalized_error < tolerance

        return EquivalenceResult(
            object_a="a", object_b="b",
            are_equivalent=is_equivalent,
            kind=EquivalenceKind.LATENT_ROTATION if is_equivalent else None,
            confidence=1.0 - min(normalized_error, 1.0),
            transform_description=(
                f"Linear map with reconstruction error {normalized_error:.2e}"
            ),
            evidence={
                "reconstruction_error": float(reconstruction_error),
                "normalized_error": float(normalized_error),
                "map_rank": int(rank),
            },
        )

    # --- Batch canonicalization ---

    def canonicalize_batch(
        self,
        concept_values: Dict[str, np.ndarray],  # concept_id → values array
    ) -> List[Set[str]]:
        """
        Find all equivalence classes in a batch of concepts.

        Returns list of sets, each containing equivalent concept IDs.
        Each set will be merged into one canonical concept.
        """
        ids = list(concept_values.keys())
        n = len(ids)

        # Union-find
        parent = {cid: cid for cid in ids}

        def find(cid: str) -> str:
            while parent[cid] != cid:
                parent[cid] = parent[parent[cid]]
                cid = parent[cid]
            return cid

        def union(cid_a: str, cid_b: str) -> None:
            ra, rb = find(cid_a), find(cid_b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                cid_a, cid_b = ids[i], ids[j]
                if find(cid_a) == find(cid_b):
                    continue

                result = self.check_numerical_equivalence(
                    concept_values[cid_a], concept_values[cid_b]
                )
                if result.are_equivalent:
                    union(cid_a, cid_b)

        # Collect groups
        groups: Dict[str, Set[str]] = {}
        for cid in ids:
            root = find(cid)
            groups.setdefault(root, set()).add(cid)

        # Only return groups with >1 member (actual equivalences)
        return [members for members in groups.values() if len(members) > 1]

    def select_canonical(
        self,
        group: Set[str],
        expression_map: Dict[str, str],  # concept_id → expression
    ) -> str:
        """
        Select the canonical representative from an equivalence group.

        Chooses the simplest expression (fewest operations, simplest constants).
        """
        if len(group) == 1:
            return next(iter(group))

        def complexity(cid: str) -> float:
            expr = expression_map.get(cid, "")
            # Count operations: +, -, *, /, **
            ops = sum(1 for c in expr if c in "+-*/")
            # Prefer shorter expressions
            return ops * 10 + len(expr)

        return min(group, key=complexity)

    # --- Description length ---

    @staticmethod
    def description_length(expression: str) -> int:
        """
        Compute rough description length of an expression.
        Used for selecting the simplest canonical form.

        Counts: number of unique operations + number of tokens.
        """
        # Count distinct operators
        ops = set()
        for c in expression:
            if c in "+-*/^":
                ops.add(c)
            elif expression.count("**") > 0:
                ops.add("**")

        # Count tokens (numbers, variables, operators)
        tokens = len(expression.replace(" ", "").split("+"))  # simple heuristic
        return len(ops) * 100 + tokens * 10 + len(expression)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numerical_tolerance": self.numerical_tolerance,
            "min_test_points": self.min_test_points,
            "affine_scaling_threshold": self.affine_scaling_threshold,
            "max_affine_attempts": self.max_affine_attempts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EquivalenceCanonicalizer":
        return cls(
            numerical_tolerance=d.get("numerical_tolerance", 1e-6),
            min_test_points=d.get("min_test_points", 20),
            affine_scaling_threshold=d.get("affine_scaling_threshold", 0.9999),
            max_affine_attempts=d.get("max_affine_attempts", 5),
        )
