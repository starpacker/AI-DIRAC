"""Formula-blind thermodynamic state/process experiments.

The experiment structures expose measured state variables and their generic
types only.  No equation is registered as geometry information or supplied to
the symbolic-regression engine.  Multiple processes share the same gas object
schema so a law proposed in one epoch can be validated in the others.
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
concept_amount = Concept("(1->Particle) |- amount[1]")
concept_temperature = Concept("(1->Particle) |- temperature[1]")
concept_gasconstant = Concept("(1->Particle) |- gasconstant[1]")
concept_gamma = Concept("(1->Particle) |- gamma[1]")


_objects = {
    "clock": Objstructure.clock(),
    "gas": Objstructure.make_particle(1.0, 2.0),
}
_data_info = [
    (concept_pressure, ["gas"]),
    (concept_volume, ["gas"]),
    (concept_amount, ["gas"]),
    (concept_temperature, ["gas"]),
    (concept_gasconstant, ["gas"]),
    (concept_gamma, ["gas"]),
    (concept_t, ["clock"]),
]
_base_parameters = {
    "gasconstant": ranged_parastructure(0.85, 1.20),
    "amount": ranged_parastructure(0.8, 2.2),
    "temperature": ranged_parastructure(1.4, 3.8),
    "volume": ranged_parastructure(1.3, 3.6),
    "pressure": ranged_parastructure(0.8, 3.0),
    "gamma": ranged_parastructure(1.35, 1.67),
    "phase1": ranged_parastructure(-0.7, 0.7),
    "phase2": ranged_parastructure(-0.7, 0.7),
}


def _noise(rng: np.random.Generator, error: float, values: np.ndarray):
    # Scale the framework's dimensionless error to each measured quantity.
    scale = max(float(np.std(values)), float(np.mean(np.abs(values))), 1.0)
    return values + rng.normal(0.0, error * scale, values.shape)


def do_experiment(t_end: float, t_num: int, error: float,
                  exp_config: ExpConfig):
    mode = int(round(exp_config.para("mode")))
    gasconstant = exp_config.para("gasconstant")
    amount0 = exp_config.para("amount")
    temperature0 = exp_config.para("temperature")
    volume0 = exp_config.para("volume")
    pressure0 = exp_config.para("pressure")
    gamma = exp_config.para("gamma")
    phase1 = exp_config.para("phase1")
    phase2 = exp_config.para("phase2")
    time = np.linspace(0.0, t_end, t_num, endpoint=False)
    coordinate = 2.0 * np.pi * time / max(t_end, 1e-12)

    if mode == 1:  # General state scan, including slow gas injection.
        volume = volume0 * (1.0 + 0.22 * np.sin(coordinate + phase1))
        temperature = temperature0 * (
            1.0 + 0.18 * np.cos(1.7 * coordinate + phase2)
        )
        amount = amount0 * (1.0 + 0.12 * np.sin(0.63 * coordinate - phase2))
        pressure = gasconstant * amount * temperature / volume
    elif mode == 2:  # Isothermal compression/expansion.
        volume = volume0 * (1.0 + 0.32 * np.sin(coordinate + phase1))
        temperature = np.full_like(time, temperature0)
        amount = np.full_like(time, amount0)
        pressure = gasconstant * amount * temperature / volume
    elif mode == 3:  # Isochoric heating/cooling.
        volume = np.full_like(time, volume0)
        temperature = temperature0 * (
            1.0 + 0.28 * np.sin(coordinate + phase1)
        )
        amount = np.full_like(time, amount0)
        pressure = gasconstant * amount * temperature / volume
    elif mode == 4:  # Isobaric heating/cooling.
        pressure = np.full_like(time, pressure0)
        temperature = temperature0 * (
            1.0 + 0.25 * np.sin(coordinate + phase1)
        )
        amount = np.full_like(time, amount0)
        volume = gasconstant * amount * temperature / pressure
    elif mode == 5:  # Reversible adiabatic expansion/compression.
        volume = volume0 * (1.0 + 0.28 * np.sin(coordinate + phase1))
        amount = np.full_like(time, amount0)
        adiabatic_constant = pressure0 * volume0**gamma
        pressure = adiabatic_constant / volume**gamma
        temperature = pressure * volume / (gasconstant * amount)
    else:
        raise ValueError(f"Unsupported thermodynamic process mode: {mode}")

    rng = np.random.default_rng(20260717 + mode)
    data = exp_config.new_datastruct_of_doexperiment(t_num)
    data.insert_data((concept_pressure, ["gas"]), _noise(rng, error, pressure))
    data.insert_data((concept_volume, ["gas"]), _noise(rng, error, volume))
    data.insert_data((concept_amount, ["gas"]), _noise(rng, error, amount))
    data.insert_data(
        (concept_temperature, ["gas"]), _noise(rng, error, temperature)
    )
    data.insert_data(
        (concept_gasconstant, ["gas"]), np.full_like(time, gasconstant)
    )
    data.insert_data((concept_gamma, ["gas"]), np.full_like(time, gamma))
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

