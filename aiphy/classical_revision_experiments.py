"""Fixed classical-mechanics datasets for end-to-end theory revision."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .core import AtomExp, DataStruct, ExpData
from .pysr_regression import GenericSearchSpec


def _observations(
    series: dict[str, np.ndarray],
    *,
    repeats: int = 8,
    noise: float = 1e-9,
    seed: int,
) -> DataStruct:
    rng = np.random.default_rng(seed)
    length = len(next(iter(series.values())))
    data = DataStruct.empty()
    sample_coordinate = np.linspace(0.0, 1.0, length)
    data.add_data(
        AtomExp("t[0]"),
        ExpData(np.repeat(sample_coordinate[None, :], repeats, axis=0).tolist()),
    )
    for name, values in series.items():
        values = np.asarray(values, dtype=np.float64)
        samples = np.repeat(values[None, :], repeats, axis=0)
        if noise:
            samples += rng.normal(0.0, noise, samples.shape)
        data.add_data(AtomExp(f"{name}[1]"), ExpData(samples.tolist()))
    return data


def _pendulum_trajectories(
    configurations: Sequence[tuple[float, float, float, float]],
    *,
    points_per_trajectory: int = 90,
) -> dict[str, np.ndarray]:
    collected = {
        "angle": [],
        "angularacceleration": [],
        "gravity": [],
        "length": [],
    }
    for gravity, length, initial_angle, initial_velocity in configurations:
        time = np.linspace(0.0, 4.0, points_per_trajectory)

        def dynamics(_time, state):
            angle, velocity = state
            return velocity, -(gravity / length) * np.sin(angle)

        solution = solve_ivp(
            dynamics,
            (time[0], time[-1]),
            (initial_angle, initial_velocity),
            t_eval=time,
            rtol=1e-11,
            atol=1e-12,
        )
        angle = solution.y[0]
        acceleration = -(gravity / length) * np.sin(angle)
        collected["angle"].append(angle)
        collected["angularacceleration"].append(acceleration)
        collected["gravity"].append(np.full_like(angle, gravity))
        collected["length"].append(np.full_like(angle, length))
    return {name: np.concatenate(blocks) for name, blocks in collected.items()}


def pendulum_revision_training_data() -> DataStruct:
    return _observations(
        _pendulum_trajectories(
            [
                (9.2, 0.90, 0.70, 0.10),
                (10.4, 1.30, 1.00, -0.15),
                (8.5, 1.70, 1.25, 0.20),
                (11.0, 1.10, 1.40, 0.00),
            ]
        ),
        seed=2026071701,
    )


def pendulum_revision_validation_data() -> list[DataStruct]:
    configurations = [
        (9.81, 1.45, 0.82, 0.05),
        (7.8, 1.00, 1.15, -0.10),
        (11.6, 1.90, 1.35, 0.12),
    ]
    return [
        _observations(
            _pendulum_trajectories([configuration]),
            seed=2026071710 + index,
        )
        for index, configuration in enumerate(configurations)
    ]


def pendulum_revision_spec() -> GenericSearchSpec:
    return GenericSearchSpec(
        target="angularacceleration[1]",
        scalars={
            "angularacceleration[1]": "1/s^2",
            "angle[1]": "",
            "gravity[1]": "m/s^2",
            "length[1]": "m",
        },
        vectors={},
    )


PENDULUM_LINEAR_LAW = "-(gravity[1] * angle[1]) / length[1]"


def _duffing_trajectories(
    configurations: Sequence[tuple[float, float, float, float, float]],
    *,
    points_per_trajectory: int = 90,
) -> dict[str, np.ndarray]:
    collected = {
        "position": [],
        "acceleration": [],
        "mass": [],
        "stiffness": [],
        "cubic": [],
    }
    for mass, stiffness, cubic, initial_position, initial_velocity in configurations:
        time = np.linspace(0.0, 5.0, points_per_trajectory)

        def dynamics(_time, state):
            position, velocity = state
            acceleration = -(stiffness * position + cubic * position**3) / mass
            return velocity, acceleration

        solution = solve_ivp(
            dynamics,
            (time[0], time[-1]),
            (initial_position, initial_velocity),
            t_eval=time,
            rtol=1e-11,
            atol=1e-12,
        )
        position = solution.y[0]
        acceleration = -(stiffness * position + cubic * position**3) / mass
        collected["position"].append(position)
        collected["acceleration"].append(acceleration)
        collected["mass"].append(np.full_like(position, mass))
        collected["stiffness"].append(np.full_like(position, stiffness))
        collected["cubic"].append(np.full_like(position, cubic))
    return {name: np.concatenate(blocks) for name, blocks in collected.items()}


def duffing_revision_training_data() -> DataStruct:
    return _observations(
        _duffing_trajectories(
            [
                (0.8, 1.3, 0.45, 1.10, 0.00),
                (1.1, 0.9, 0.70, 0.85, 0.15),
                (1.4, 1.7, 0.35, 1.25, -0.10),
                (0.9, 1.1, 0.60, 1.00, 0.20),
            ]
        ),
        seed=2026071721,
    )


def duffing_revision_validation_data() -> list[DataStruct]:
    configurations = [
        (1.25, 1.45, 0.52, 0.92, 0.08),
        (0.75, 0.80, 0.38, 1.18, -0.12),
        (1.55, 1.90, 0.66, 1.05, 0.05),
    ]
    return [
        _observations(
            _duffing_trajectories([configuration]),
            seed=2026071730 + index,
        )
        for index, configuration in enumerate(configurations)
    ]


def duffing_revision_spec() -> GenericSearchSpec:
    return GenericSearchSpec(
        target="acceleration[1]",
        scalars={
            "acceleration[1]": "m/s^2",
            "position[1]": "m",
            "mass[1]": "kg",
            "stiffness[1]": "kg/s^2",
            "cubic[1]": "kg/(m^2*s^2)",
        },
        vectors={},
    )


HOOKE_LINEAR_LAW = "-(stiffness[1] * position[1]) / mass[1]"
