"""Numerical spatial observations for validating Grad/Div/Curl semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SpatialGrid:
    axes: tuple[np.ndarray, ...]

    def __init__(self, axes: Sequence[Sequence[float]]):
        converted = tuple(np.asarray(axis, dtype=float) for axis in axes)
        if not 1 <= len(converted) <= 3:
            raise ValueError("Spatial grids must have one, two, or three axes")
        if any(axis.ndim != 1 or axis.size < 3 or np.any(np.diff(axis) <= 0) for axis in converted):
            raise ValueError("Each spatial axis must be a strictly increasing 1-D array")
        object.__setattr__(self, "axes", converted)

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.axes)

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        return tuple(np.meshgrid(*self.axes, indexing="ij"))

    def _field(self, values) -> np.ndarray:
        field = np.asarray(values, dtype=float)
        if field.shape != self.shape:
            raise ValueError(f"Field shape {field.shape} does not match grid {self.shape}")
        return field

    def grad(self, scalar) -> tuple[np.ndarray, ...]:
        scalar = self._field(scalar)
        return tuple(np.gradient(scalar, *self.axes, edge_order=2))

    def div(self, vector: Sequence[np.ndarray]) -> np.ndarray:
        if len(vector) != self.dimension:
            raise ValueError("Vector dimension must match the spatial grid")
        components = [self._field(component) for component in vector]
        return sum(
            np.gradient(component, self.axes[index], axis=index, edge_order=2)
            for index, component in enumerate(components)
        )

    def curl(self, vector: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        if self.dimension != 3 or len(vector) != 3:
            raise ValueError("Curl requires a three-dimensional vector field and grid")
        fx, fy, fz = (self._field(component) for component in vector)
        x, y, z = self.axes
        return (
            np.gradient(fz, y, axis=1, edge_order=2)
            - np.gradient(fy, z, axis=2, edge_order=2),
            np.gradient(fx, z, axis=2, edge_order=2)
            - np.gradient(fz, x, axis=0, edge_order=2),
            np.gradient(fy, x, axis=0, edge_order=2)
            - np.gradient(fx, y, axis=1, edge_order=2),
        )

