"""Thermodynamics with an epistemically honest measurement boundary.

Only quantities with an explicit initial measurement protocol are exposed:
pressure, volume, temperature, sample mass, molar mass and time.  Amount of
substance, the gas constant and the adiabatic exponent remain simulator-side
latent variables.  The equilibrium scan represents a sequence of independent
samples, so all five measured state/sample coordinates vary and are
identifiable from one dataset.
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
concept_samplemass = Concept("(1->Particle) |- samplemass[1]")
concept_molarmass = Concept("(1->Particle) |- molarmass[1]")


_objects = {
    "clock": Objstructure.clock(),
    "gas": Objstructure.make_particle(1.0, 2.0),
}
_measured_data = [
    (concept_pressure, ["gas"]),
    (concept_volume, ["gas"]),
    (concept_temperature, ["gas"]),
    (concept_samplemass, ["gas"]),
    (concept_molarmass, ["gas"]),
    (concept_t, ["clock"]),
]
_parameters = {
    # R and gamma are latent.  R is universal across every registered
    # experiment.  Gamma is a material intrinsic: it stays fixed when the same
    # gas experiment is repeated, but is never supplied as an observation.
    # Separate material configurations may use different hidden gamma values.
    # Use SI scaling.  The previous epistemic-v1 run deliberately used the
    # normalized value 1.2, which SR correctly rendered as 6/5; it was not an
    # estimate of the SI molar gas constant.
    "gasconstant_hidden": fixed_parastructure(8.314),
    "gamma_hidden": fixed_parastructure(1.40),
    "samplemass": ranged_parastructure(0.8, 2.4),
    "molarmass": ranged_parastructure(0.55, 1.45),
    "temperature": ranged_parastructure(1.4, 3.8),
    "volume": ranged_parastructure(1.3, 3.6),
    "pressure": ranged_parastructure(0.8, 3.0),
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
    gamma = exp_config.para("gamma_hidden")
    mass0 = exp_config.para("samplemass")
    molarmass0 = exp_config.para("molarmass")
    temperature0 = exp_config.para("temperature")
    volume0 = exp_config.para("volume")
    pressure0 = exp_config.para("pressure")
    phase1 = exp_config.para("phase1")
    phase2 = exp_config.para("phase2")
    time = np.linspace(0.0, t_end, t_num, endpoint=False)
    coordinate = 2.0 * np.pi * time / max(t_end, 1e-12)

    if mode == 1:  # Independent equilibrium-state/sample sweep.
        volume = volume0 * (1.0 + 0.22 * np.sin(coordinate + phase1))
        temperature = temperature0 * (
            1.0 + 0.18 * np.cos(1.7 * coordinate + phase2)
        )
        samplemass = mass0 * (
            1.0 + 0.15 * np.sin(0.63 * coordinate - phase2)
        )
        molarmass = molarmass0 * (
            1.0 + 0.11 * np.cos(0.83 * coordinate + phase1 - phase2)
        )
        amount_hidden = samplemass / molarmass
        pressure = gasconstant * amount_hidden * temperature / volume
    else:
        samplemass = np.full_like(time, mass0)
        molarmass = np.full_like(time, molarmass0)
        amount_hidden = samplemass / molarmass
        if mode == 2:  # Isothermal compression/expansion.
            volume = volume0 * (1.0 + 0.32 * np.sin(coordinate + phase1))
            temperature = np.full_like(time, temperature0)
            pressure = gasconstant * amount_hidden * temperature / volume
        elif mode == 3:  # Isochoric heating/cooling.
            volume = np.full_like(time, volume0)
            temperature = temperature0 * (
                1.0 + 0.28 * np.sin(coordinate + phase1)
            )
            pressure = gasconstant * amount_hidden * temperature / volume
        elif mode == 4:  # Isobaric heating/cooling.
            pressure = np.full_like(time, pressure0)
            temperature = temperature0 * (
                1.0 + 0.25 * np.sin(coordinate + phase1)
            )
            volume = gasconstant * amount_hidden * temperature / pressure
        elif mode == 5:  # Reversible adiabatic process; gamma stays latent.
            volume = volume0 * (1.0 + 0.28 * np.sin(coordinate + phase1))
            adiabatic_constant = pressure0 * volume0 ** gamma
            pressure = adiabatic_constant / volume ** gamma
            temperature = pressure * volume / (gasconstant * amount_hidden)
        else:
            raise ValueError(f"Unsupported epistemic thermodynamic mode: {mode}")

    rng = np.random.default_rng(20260730 + mode)
    data = exp_config.new_datastruct_of_doexperiment(t_num)
    data.insert_data((concept_pressure, ["gas"]), _noise(rng, error, pressure))
    data.insert_data((concept_volume, ["gas"]), _noise(rng, error, volume))
    data.insert_data(
        (concept_temperature, ["gas"]), _noise(rng, error, temperature)
    )
    data.insert_data(
        (concept_samplemass, ["gas"]), _noise(rng, error, samplemass)
    )
    data.insert_data(
        (concept_molarmass, ["gas"]), _noise(rng, error, molarmass)
    )
    data.insert_data((concept_t, ["clock"]), time)
    return data


def _config(name: str, mode: int) -> ExpStructure:
    parameters = dict(_parameters)
    parameters["mode"] = fixed_parastructure(float(mode))
    return ExpStructure(
        ExpConfig(name, 1, parameters, _objects, _measured_data),
        DoExpType(__file__),
    )


def epistemic_gas_state_scan_config() -> ExpStructure:
    return _config("epistemic_gas_state_scan", 1)


def epistemic_gas_isothermal_config() -> ExpStructure:
    return _config("epistemic_gas_isothermal", 2)


def epistemic_gas_isochoric_config() -> ExpStructure:
    return _config("epistemic_gas_isochoric", 3)


def epistemic_gas_isobaric_config() -> ExpStructure:
    return _config("epistemic_gas_isobaric", 4)


def epistemic_gas_adiabatic_config() -> ExpStructure:
    return _config("epistemic_gas_adiabatic", 5)


for _name in (
    "epistemic_gas_state_scan",
    "epistemic_gas_isothermal",
    "epistemic_gas_isochoric",
    "epistemic_gas_isobaric",
    "epistemic_gas_adiabatic",
):
    register_experiment_targets(_name, ["pressure"])
