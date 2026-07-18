"""Small diagnostic experiments for the extended symbolic language.

These are capability benchmarks, not additions to the historical source-38
suite.  Each isolates one requested language feature so a failure has a clear
meaning.
"""

from __future__ import annotations

import numpy as np

from .core import AtomExp, DataStruct, ExpData
from .spatial_operators import SpatialGrid
from .pysr_regression import GenericSearchSpec, VectorVariableSpec
from .sr_metadata import register_experiment_targets


register_experiment_targets("capability_lorentz", ["forcex"])
register_experiment_targets("capability_boltzmann", ["probability"])
register_experiment_targets("capability_real_gas", ["pressure"])


def _series_data(series: dict[str, np.ndarray], *, repeats: int = 12,
                 noise: float = 1e-9, seed: int = 20260716) -> DataStruct:
    rng = np.random.default_rng(seed)
    length = len(next(iter(series.values())))
    data = DataStruct.empty()
    time = np.linspace(0.0, 1.0, length)
    data.add_data(
        AtomExp("t[0]"),
        ExpData(np.repeat(time[None, :], repeats, axis=0).tolist()),
    )
    for name, values in series.items():
        values = np.asarray(values, dtype=float)
        samples = np.repeat(values[None, :], repeats, axis=0)
        if noise:
            samples = samples + rng.normal(0.0, noise, samples.shape)
        data.add_data(AtomExp(f"{name}[1]"), ExpData(samples.tolist()))
    return data


def lorentz_force_capability_data(samples: int = 240) -> DataStruct:
    parameter = np.linspace(-2.0, 2.0, samples)
    velocity = np.column_stack(
        [np.sin(parameter), np.cos(0.7 * parameter), 0.4 * parameter]
    )
    magnetic = np.column_stack(
        [0.3 + 0.1 * np.cos(parameter), np.full(samples, -0.4), np.full(samples, 1.2)]
    )
    # Vary q independently enough that it cannot be absorbed into a fitted
    # numerical coefficient.  The benchmark must identify q*(v cross B), not
    # merely a constant multiple of the cross product.
    charge = 1.05 + 0.25 * np.sin(1.3 * parameter + 0.2)
    force = charge[:, None] * np.cross(velocity, magnetic)
    return _series_data(
        {
            "velocityx": velocity[:, 0], "velocityy": velocity[:, 1], "velocityz": velocity[:, 2],
            "bfieldx": magnetic[:, 0], "bfieldy": magnetic[:, 1], "bfieldz": magnetic[:, 2],
            "charge": charge,
            "forcex": force[:, 0], "forcey": force[:, 1], "forcez": force[:, 2],
        }
    )


def lorentz_generic_spec() -> GenericSearchSpec:
    return GenericSearchSpec(
        target="forcex[1]",
        scalars={"charge[1]": "A*s"},
        vectors={
            "velocity": VectorVariableSpec(
                ("velocityx[1]", "velocityy[1]", "velocityz[1]"), "m/s"
            ),
            "magnetic": VectorVariableSpec(
                ("bfieldx[1]", "bfieldy[1]", "bfieldz[1]"), "kg/(A*s^2)"
            ),
            "force": VectorVariableSpec(
                ("forcex[1]", "forcey[1]", "forcez[1]"), "kg*m/s^2"
            ),
        },
    )
def boltzmann_capability_data(samples: int = 240) -> DataStruct:
    energy = np.linspace(0.1, 4.0, samples)
    beta = 1.5
    normalization = 1.25
    probability = normalization * np.exp(-beta * energy)
    return _series_data({"energy": energy, "probability": probability}, noise=1e-10)


def boltzmann_generic_spec() -> GenericSearchSpec:
    return GenericSearchSpec(
        target="probability[1]",
        scalars={"energy[1]": "kg*m^2/s^2", "probability[1]": ""},
        vectors={},
    )


def real_gas_capability_data(samples: int = 300) -> DataStruct:
    rng = np.random.default_rng(314159)
    amount = rng.uniform(0.6, 1.8, samples)
    temperature = rng.uniform(1.0, 4.0, samples)
    covolume = rng.uniform(0.12, 0.28, samples)
    volume = rng.uniform(1.4, 4.5, samples) + amount * covolume
    attraction = rng.uniform(0.5, 1.3, samples)
    gasconstant = rng.uniform(0.8, 1.2, samples)
    pressure = (
        gasconstant * amount * temperature / (volume - amount * covolume)
        - attraction * (amount / volume) ** 2
    )
    return _series_data(
        {
            "pressure": pressure,
            "volume": volume,
            "amount": amount,
            "temperature": temperature,
            "attraction": attraction,
            "covolume": covolume,
            "gasconstant": gasconstant,
        },
        noise=1e-10,
    )


def real_gas_generic_spec() -> GenericSearchSpec:
    return GenericSearchSpec(
        target="pressure[1]",
        scalars={
            "pressure[1]": "kg/(m*s^2)",
            "volume[1]": "m^3",
            "amount[1]": "mol",
            "temperature[1]": "K",
            "attraction[1]": "kg*m^5/(s^2*mol^2)",
            "covolume[1]": "m^3/mol",
            "gasconstant[1]": "kg*m^2/(s^2*mol*K)",
        },
        vectors={},
    )


def nabla_capability_fields(points: int = 17):
    axis = np.linspace(-1.0, 1.0, points)
    grid = SpatialGrid([axis, axis, axis])
    x, y, z = grid.mesh
    scalar = x**2 * y + np.sin(z)
    scalar_gradient = (2.0 * x * y, x**2, np.cos(z))
    vector = (-0.5 * y, 0.5 * x, z**2)
    divergence = 2.0 * z
    curl = (np.zeros_like(x), np.zeros_like(x), np.ones_like(x))
    return grid, scalar, scalar_gradient, vector, divergence, curl
