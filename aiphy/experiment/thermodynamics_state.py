"""Formula-blind thermodynamic state/process experiments — STRICT EDITION.

Observables exposed to the discovery engine (per experiment):
  - pressure, volume, temperature, t  — direct sensor readings
  - gasconstant and gamma are HIDDEN (not in data_info).

amount (n) is deliberately NOT exposed.  The engine must discover that
PV/T is conserved across processes for the same gas sample, thereby
inventing the concept "nR" (the product of amount and gas constant).

The gasconstant is a hidden simulation parameter; gamma is a hidden
material property.  Both may be discoverable as conserved quantities
from the data.
"""

from __future__ import annotations

import numpy as np

from aiphy.experiment import (
    Concept,
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    concept_t,
)
from aiphy.experiment.challenge_thermodynamics.common import (
    fixed_parastructure,
    ranged_parastructure,
)
from aiphy.sr_metadata import register_experiment_targets


concept_pressure = Concept("(1->Particle) |- pressure[1]")
concept_volume = Concept("(1->Particle) |- volume[1]")
concept_temperature = Concept("(1->Particle) |- temperature[1]")
# NOTE: amount and gasconstant are NOT registered as observables below.
# amount is a hidden simulation parameter — the engine must discover nR = PV/T.


_objects = {
    "clock": Objstructure.clock(),
    "gas": Objstructure.make_particle(1.0, 2.0),
}
_data_info = [
    (concept_pressure, ["gas"]),
    (concept_volume, ["gas"]),
    (concept_temperature, ["gas"]),
    (concept_t, ["clock"]),
]
_base_parameters = {
    # Hidden simulation parameters — NOT exposed to the engine as data
    "gasconstant_hidden": fixed_parastructure(8.314462618),
    "amount_hidden": ranged_parastructure(0.8, 2.2),
    # Observable-varying parameters
    "temperature": ranged_parastructure(1.4, 3.8),
    "volume": ranged_parastructure(1.3, 3.6),
    "pressure": ranged_parastructure(0.8, 3.0),
    # Hidden material property (may be discovered as conserved)
    "gamma_hidden": ranged_parastructure(1.35, 1.67),
    "phase1": ranged_parastructure(-0.7, 0.7),
    "phase2": ranged_parastructure(-0.7, 0.7),
}


def _noise(rng: np.random.Generator, error: float, values: np.ndarray):
    scale = max(float(np.std(values)), float(np.mean(np.abs(values))), 1.0)
    return values + rng.normal(0.0, error * scale, values.shape)


def do_experiment(t_end: float, t_num: int, error: float,
                  exp_config: ExpConfig):
    mode = int(round(exp_config.para("mode")))
    gasconstant = exp_config.para("gasconstant_hidden")
    amount0 = exp_config.para("amount_hidden")
    temperature0 = exp_config.para("temperature")
    volume0 = exp_config.para("volume")
    pressure0 = exp_config.para("pressure")
    gamma = exp_config.para("gamma_hidden")
    phase1 = exp_config.para("phase1")
    phase2 = exp_config.para("phase2")
    time = np.linspace(0.0, t_end, t_num, endpoint=False)
    coordinate = 2.0 * np.pi * time / max(t_end, 1e-12)
    nR = gasconstant * amount0  # The conserved quantity the engine must discover

    if mode == 1:  # General state scan, including slow gas injection.
        volume = volume0 * (1.0 + 0.22 * np.sin(coordinate + phase1))
        temperature = temperature0 * (
            1.0 + 0.18 * np.cos(1.7 * coordinate + phase2)
        )
        amount = amount0 * (1.0 + 0.12 * np.sin(0.63 * coordinate - phase2))
        pressure = nR * amount / amount0 * temperature / volume
    elif mode == 2:  # Isothermal compression/expansion.
        volume = volume0 * (1.0 + 0.32 * np.sin(coordinate + phase1))
        temperature = np.full_like(time, temperature0)
        pressure = nR * temperature / volume
    elif mode == 3:  # Isochoric heating/cooling.
        volume = np.full_like(time, volume0)
        temperature = temperature0 * (
            1.0 + 0.28 * np.sin(coordinate + phase1)
        )
        pressure = nR * temperature / volume
    elif mode == 4:  # Isobaric heating/cooling.
        pressure = np.full_like(time, pressure0)
        temperature = temperature0 * (
            1.0 + 0.25 * np.sin(coordinate + phase1)
        )
        volume = nR * temperature / pressure
    elif mode == 5:  # Reversible adiabatic expansion/compression.
        volume = volume0 * (1.0 + 0.28 * np.sin(coordinate + phase1))
        adiabatic_constant = pressure0 * volume0**gamma
        pressure = adiabatic_constant / volume**gamma
        temperature = pressure * volume / nR
    else:
        raise ValueError(f"Unsupported thermodynamic process mode: {mode}")

    rng = np.random.default_rng(20260717 + mode)
    data = exp_config.new_datastruct_of_doexperiment(t_num)
    # ONLY pressure, volume, temperature, time are in the data stream.
    # amount, gasconstant, and gamma are NEVER inserted.
    data.insert_data((concept_pressure, ["gas"]), _noise(rng, error, pressure))
    data.insert_data((concept_volume, ["gas"]), _noise(rng, error, volume))
    data.insert_data(
        (concept_temperature, ["gas"]), _noise(rng, error, temperature)
    )
    data.insert_data((concept_t, ["clock"]), time)
    return data


def _config(name: str, mode: int) -> ExpStructure:
    parameters = dict(_base_parameters)
    parameters["mode"] = fixed_parastructure(float(mode))
    config = ExpConfig(name, 1, parameters, _objects, _data_info)
    return ExpStructure(config, DoExpType(__file__))


def ideal_gas_state_scan_config() -> ExpStructure:
    return _config("ideal_gas_state_scan", 1)


def ideal_gas_isothermal_config() -> ExpStructure:
    return _config("ideal_gas_isothermal", 2)


def ideal_gas_isochoric_config() -> ExpStructure:
    return _config("ideal_gas_isochoric", 3)


def ideal_gas_isobaric_config() -> ExpStructure:
    return _config("ideal_gas_isobaric", 4)


def ideal_gas_adiabatic_config() -> ExpStructure:
    return _config("ideal_gas_adiabatic", 5)


for _experiment_name in (
    "ideal_gas_state_scan",
    "ideal_gas_isothermal",
    "ideal_gas_isochoric",
    "ideal_gas_isobaric",
    "ideal_gas_adiabatic",
):
    register_experiment_targets(_experiment_name, ["pressure"])
