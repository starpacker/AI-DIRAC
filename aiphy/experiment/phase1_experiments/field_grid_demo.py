"""
Phase 1 — Field Grid Demonstration Experiment.

A 2D heat equation steady-state solution: temperature field T(x,y)
on a rectangular plate with fixed boundary conditions.

  - Left edge: T = T_hot
  - Right edge: T = T_cold
  - Top/bottom edges: insulated (dT/dy = 0)

The steady-state solution is linear in x: T(x,y) = T_hot - (T_hot - T_cold) * x / L

Observable: T(x,y) — temperature at each grid point
Control: T_hot, T_cold (boundary temperatures, known to apparatus)
Context: Lx, Ly (plate dimensions, known)

The discovery engine should find:
  - d²T/dx² ≈ 0 (Laplace equation for steady state)
  - dT/dy ≈ 0 (uniform in y due to insulated boundaries)
  - T(x) is approximately linear in x

This demonstrates:
  1. FieldGridData construction
  2. Spatial held-out validation (split along x-axis)
  3. Provenance pass for field data
  4. Same discovery API (observation_to_datastruct) works

RAW_SENSOR: T — thermometer reading at each grid point
CONTROL: T_hot, T_cold — known boundary temperatures
"""

import numpy as np


def generate_heat_equation_field(
    Lx: float = 1.0,
    Ly: float = 0.5,
    nx: int = 50,
    ny: int = 25,
    T_hot: float = 100.0,
    T_cold: float = 20.0,
    noise_std: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Generate a steady-state heat equation field on a 2D grid.

    Returns:
      - 'T': 2D array of shape (ny, nx) — temperature field
      - 'x': 1D array of grid x-coordinates
      - 'y': 1D array of grid y-coordinates
      - 'T_hot': hot boundary temperature
      - 'T_cold': cold boundary temperature
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Analytical steady-state solution: linear in x
    T = T_hot - (T_hot - T_cold) * x / Lx
    T = np.tile(T, (ny, 1))  # repeat in y-direction

    # Add small measurement noise
    T += rng.normal(0, noise_std, T.shape)

    return {
        'T': T,
        'x': x,
        'y': y,
        'T_hot': T_hot,
        'T_cold': T_cold,
    }


def generate_electrostatic_potential_field(
    Lx: float = 2.0,
    Ly: float = 2.0,
    nx: int = 60,
    ny: int = 60,
    charge_pos: tuple = (0.3, 0.0),
    charge_strength: float = 1.0,
    noise_std: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Generate a 2D electrostatic potential field for a point charge.

    V(x,y) = k * q / sqrt((x - x0)^2 + (y - y0)^2 + epsilon)

    The discovery engine should find that d²V/dx² + d²V/dy² ≈ 0
    (Laplace equation) everywhere except near the charge.

    Returns:
      - 'V': 2D array of shape (ny, nx) — potential field
      - 'x': 1D array of grid x-coordinates
      - 'y': 1D array of grid y-coordinates
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(-Lx/2, Lx/2, nx)
    y = np.linspace(-Ly/2, Ly/2, ny)

    X, Y = np.meshgrid(x, y, indexing='ij')
    epsilon = 0.01  # regularization to avoid singularity
    dx = X - charge_pos[0]
    dy = Y - charge_pos[1]
    r = np.sqrt(dx**2 + dy**2 + epsilon**2)

    V = charge_strength / r

    # Add measurement noise
    V += rng.normal(0, noise_std, V.shape)

    return {
        'V': V,
        'x': x,
        'y': y,
    }
