"""Strict ladder experiments with explicit open/hidden column policies.

Each config in this module exposes only the columns declared by its ladder
stage.  Hidden simulator quantities may be used internally to generate data,
but they are omitted from both ``data_info`` and the returned data struct.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

from aiphy.experiment import (
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_cx,
    concept_cy,
    concept_cz,
    concept_dist,
    concept_length,
    concept_mass,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_t,
    default_parastructure,
)
from aiphy.experiment.column_policy import (
    ColumnRole,
    ColumnSpec,
    audit_specs,
    concept,
    insert_open_columns,
    key,
    open_data_info,
    particle_concept,
    resolve_column_specs,
)


MODE = {
    "strict_1_1_collision_pair": 101,
    "strict_1_2_collision_reference": 102,
    "strict_2_1_free_fall_time": 201,
    "strict_2_2_incline_slide": 202,
    "strict_2_3_atwood": 203,
    "strict_3_1_mass_under_force": 301,
    "strict_3_2_hanging_spring": 302,
    "strict_4_1_drop_repeat": 401,
    "strict_4_2_orbit_two_body": 402,
    "strict_5_1_volume_by_displacement": 501,
    "strict_5_2_density_emergence": 502,
    "strict_5_3_pressure_discovery": 503,
    "strict_5_4_hydrostatic_pressure": 504,
    "strict_5_5_sphere_terminal_fall": 505,
    "strict_5_6_pipe_flow": 506,
    "strict_6_1_spring_oscillator": 601,
    "strict_6_2_rolling_shapes": 602,
    "strict_6_3_pendulum_small": 603,
    "strict_6_4_damped_oscillator": 604,
    "strict_x_1_elastic_contact": 650,
    "strict_7_1_projectile_no_drag": 701,
    "strict_7_2_projectile_linear_drag": 702,
    "strict_7_3_vertical_fall_quadratic": 703,
    "strict_7_4_forced_oscillator": 704,
    "strict_y_1_high_speed_kinematics": 750,
    "strict_y_2_relativistic_rocket": 751,
    "strict_t_1_temperature_sensor": 801,
    "strict_t_2_ideal_gas_state": 802,
    "strict_t_3_adiabatic_gas": 803,
    "strict_t_4_heat_capacity_calibration": 804,
    "strict_t_5_cooling_and_conduction": 805,
    "strict_t_6_van_der_waals_gas": 806,
    "strict_t_7_solid_heat_capacity": 807,
    "strict_t_8_entropy_expansion": 808,
    "strict_o_1_light_time_of_flight": 901,
    "strict_o_2_interference_wavelength": 902,
    "strict_o_3_frequency_relation": 903,
    "strict_o_4_diffraction": 904,
    "strict_o_5_photoelectric_effect": 905,
    "strict_o_6_blackbody_spectrum": 906,
}


concept_area = particle_concept("area")
concept_particle_length = particle_concept("length")
concept_radius = particle_concept("radius")
concept_depth = particle_concept("depth")
concept_volume = particle_concept("volume")
concept_pressure = particle_concept("pressure")
concept_temperature = particle_concept("temperature")
concept_samplemass = particle_concept("samplemass")
concept_molarmass = particle_concept("molarmass")
concept_power = particle_concept("power")
concept_heat = particle_concept("heat")
concept_position = particle_concept("xpos")
concept_intensity = particle_concept("intensity")
concept_wavelength = particle_concept("wavelength")
concept_frequency = particle_concept("frequency")
concept_stop_voltage = particle_concept("stop_voltage")
concept_spectral = particle_concept("spectral_intensity")
concept_control_u = particle_concept("control_u")
concept_drive_f0 = particle_concept("drive_f0")
concept_drive_omega = particle_concept("drive_omega")
concept_path_length = particle_concept("path_length")
concept_arrival_time = particle_concept("arrival_time")
concept_energy_input = particle_concept("energy_input")
concept_fuel_mass = particle_concept("fuel_mass")
concept_atom_count = particle_concept("atom_count")
concept_entropy = particle_concept("entropy")
concept_kspring = concept("(1->Spring) |- kspring[1]")


def _fixed(value: float):
    return default_parastructure(value, value)


def _base_para(mode: int, **extra):
    params = {"mode": _fixed(float(mode))}
    params.update(extra)
    return params


def _objects(*names: str, spring: bool = False, slope: bool = False) -> Dict[str, Objstructure]:
    info = {name: Objstructure.make_particle(1.0, 5.0) for name in names}
    if spring:
        info["s1"] = Objstructure.make_spring(1.5, 2.5, 4.0, 6.0)
    if slope:
        info["slope"] = Objstructure.make_slope()
    info["clock"] = Objstructure.clock()
    return info


STRICT_EXPERIMENT_OBJECTS = {
    "strict_1_1_collision_pair": (("a", "b"), {}, 1),
    "strict_1_2_collision_reference": (("a", "b", "c"), {}, 1),
    "strict_2_1_free_fall_time": (("o1",), {}, 1),
    "strict_2_2_incline_slide": (("o1",), {"slope": True}, 2),
    "strict_2_3_atwood": (("a", "b"), {}, 1),
    "strict_3_1_mass_under_force": (("o1", "ctrl"), {}, 1),
    "strict_3_2_hanging_spring": (("o1",), {"spring": True}, 1),
    "strict_4_1_drop_repeat": (("o1",), {}, 1),
    "strict_4_2_orbit_two_body": (("a", "b"), {}, 2),
    "strict_5_1_volume_by_displacement": (("surface", "container", "body"), {}, 1),
    "strict_5_2_density_emergence": (("surface", "container", "body"), {}, 1),
    "strict_5_3_pressure_discovery": (("piston", "gas"), {}, 1),
    "strict_5_4_hydrostatic_pressure": (("probe",), {}, 1),
    "strict_5_5_sphere_terminal_fall": (("ball",), {}, 1),
    "strict_5_6_pipe_flow": (("pipe", "outflow", "pin", "pout"), {}, 1),
    "strict_6_1_spring_oscillator": (("o1",), {"spring": True}, 1),
    "strict_6_2_rolling_shapes": (("o1",), {"slope": True}, 1),
    "strict_6_3_pendulum_small": (("o1",), {"spring": True}, 1),
    "strict_6_4_damped_oscillator": (("o1",), {"spring": True}, 1),
    "strict_x_1_elastic_contact": (("probe", "ctrl"), {}, 1),
    "strict_7_1_projectile_no_drag": (("o1",), {}, 2),
    "strict_7_2_projectile_linear_drag": (("o1",), {}, 2),
    "strict_7_3_vertical_fall_quadratic": (("o1",), {}, 1),
    "strict_7_4_forced_oscillator": (("o1", "ctrl"), {"spring": True}, 1),
    "strict_y_1_high_speed_kinematics": (("o1", "ctrl"), {}, 1),
    "strict_y_2_relativistic_rocket": (("o1",), {}, 1),
    "strict_t_1_temperature_sensor": (("a", "b"), {}, 1),
    "strict_t_2_ideal_gas_state": (("gas",), {}, 1),
    "strict_t_3_adiabatic_gas": (("gas",), {}, 1),
    "strict_t_4_heat_capacity_calibration": (("sample", "heater"), {}, 1),
    "strict_t_5_cooling_and_conduction": (("a", "b", "env", "rod"), {}, 1),
    "strict_t_6_van_der_waals_gas": (("gas",), {}, 1),
    "strict_t_7_solid_heat_capacity": (("sample",), {}, 1),
    "strict_t_8_entropy_expansion": (("gas",), {}, 1),
    "strict_o_1_light_time_of_flight": (("beam",), {}, 1),
    "strict_o_2_interference_wavelength": (("detector", "screen", "slit1", "slit2", "beam"), {}, 1),
    "strict_o_3_frequency_relation": (("beam",), {}, 1),
    "strict_o_4_diffraction": (("detector", "screen", "aperture", "beam"), {}, 1),
    "strict_o_5_photoelectric_effect": (("beam", "detector"), {}, 1),
    "strict_o_6_blackbody_spectrum": (("beam", "detector", "body"), {}, 1),
}


def _specs(mode: int) -> List[ColumnSpec]:
    D = ColumnRole.DEFAULT
    U = ColumnRole.UNLOCKED
    C = ColumnRole.CONTROL
    H = ColumnRole.HIDDEN
    if mode == MODE["strict_1_1_collision_pair"]:
        return [
            ColumnSpec("posx_a", key(concept_posx, "a"), D),
            ColumnSpec("posx_b", key(concept_posx, "b"), D),
            ColumnSpec("posy_a", key(concept_posy, "a"), D),
            ColumnSpec("posy_b", key(concept_posy, "b"), D),
            ColumnSpec("posz_a", key(concept_posz, "a"), D),
            ColumnSpec("posz_b", key(concept_posz, "b"), D),
            ColumnSpec("dist_ab", key(concept_dist, "a", "b"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
            ColumnSpec("mass_a", key(concept_mass, "a"), H),
            ColumnSpec("mass_b", key(concept_mass, "b"), H),
        ]
    if mode == MODE["strict_1_2_collision_reference"]:
        specs = [
            ColumnSpec(f"posx_{o}", key(concept_posx, o), D) for o in ("a", "b", "c")
        ]
        specs += [ColumnSpec(f"posy_{o}", key(concept_posy, o), D) for o in ("a", "b", "c")]
        specs += [ColumnSpec(f"posz_{o}", key(concept_posz, o), D) for o in ("a", "b", "c")]
        specs += [
            ColumnSpec("dist_ab", key(concept_dist, "a", "b"), D),
            ColumnSpec("dist_bc", key(concept_dist, "b", "c"), D),
            ColumnSpec("dist_ac", key(concept_dist, "a", "c"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
            ColumnSpec("mass_a", key(concept_mass, "a"), U),
            ColumnSpec("mass_b", key(concept_mass, "b"), H),
            ColumnSpec("mass_c", key(concept_mass, "c"), H),
        ]
        return specs
    if mode in (MODE["strict_2_1_free_fall_time"], MODE["strict_4_1_drop_repeat"]):
        role = U if mode == MODE["strict_4_1_drop_repeat"] else H
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("posy_o1", key(concept_posy, "o1"), D),
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), role),
        ]
    if mode == MODE["strict_2_2_incline_slide"]:
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("posy_o1", key(concept_posy, "o1"), D),
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("cx_slope", key(concept_cx, "slope"), D),
            ColumnSpec("cy_slope", key(concept_cy, "slope"), D),
            ColumnSpec("cz_slope", key(concept_cz, "slope"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_2_3_atwood"]:
        return [
            ColumnSpec("posz_a", key(concept_posz, "a"), D),
            ColumnSpec("posz_b", key(concept_posz, "b"), D),
            ColumnSpec("mass_a", key(concept_mass, "a"), U),
            ColumnSpec("mass_b", key(concept_mass, "b"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_3_1_mass_under_force"]:
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("control_u", key(concept_control_u, "ctrl"), C),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_3_2_hanging_spring"]:
        return [
            ColumnSpec("length_s1", key(concept_length, "s1"), D),
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
            ColumnSpec("kspring_s1", key(concept_kspring, "s1"), H),
        ]
    if mode == MODE["strict_4_2_orbit_two_body"]:
        return [
            ColumnSpec("posx_a", key(concept_posx, "a"), D),
            ColumnSpec("posy_a", key(concept_posy, "a"), D),
            ColumnSpec("posx_b", key(concept_posx, "b"), D),
            ColumnSpec("posy_b", key(concept_posy, "b"), D),
            ColumnSpec("dist_ab", key(concept_dist, "a", "b"), D),
            ColumnSpec("mass_a", key(concept_mass, "a"), U),
            ColumnSpec("mass_b", key(concept_mass, "b"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode in (MODE["strict_5_1_volume_by_displacement"], MODE["strict_5_2_density_emergence"]):
        volume_role = H if mode == MODE["strict_5_1_volume_by_displacement"] else U
        return [
            ColumnSpec("posz_surface", key(concept_posz, "surface"), D),
            ColumnSpec("area_container", key(concept_area, "container"), D),
            ColumnSpec("radius_body", key(concept_radius, "body"), D),
            ColumnSpec("volume_body", key(concept_volume, "body"), volume_role),
            ColumnSpec("mass_body", key(concept_mass, "body"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_5_3_pressure_discovery"]:
        return [
            ColumnSpec("posz_piston", key(concept_posz, "piston"), D),
            ColumnSpec("area_piston", key(concept_area, "piston"), D),
            ColumnSpec("mass_piston", key(concept_mass, "piston"), U),
            ColumnSpec("pressure_gas", key(concept_pressure, "gas"), H),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_5_4_hydrostatic_pressure"]:
        return [
            ColumnSpec("depth_probe", key(concept_depth, "probe"), D),
            ColumnSpec("pressure_probe", key(concept_pressure, "probe"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_5_5_sphere_terminal_fall"]:
        return [
            ColumnSpec("posz_ball", key(concept_posz, "ball"), D),
            ColumnSpec("radius_ball", key(concept_radius, "ball"), D),
            ColumnSpec("mass_ball", key(concept_mass, "ball"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_5_6_pipe_flow"]:
        return [
            ColumnSpec("radius_pipe", key(concept_radius, "pipe"), D),
            ColumnSpec("length_pipe", key(concept_particle_length, "pipe"), D),
            ColumnSpec("volume_out", key(concept_volume, "outflow"), U),
            ColumnSpec("pressure_in", key(concept_pressure, "pin"), U),
            ColumnSpec("pressure_out", key(concept_pressure, "pout"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_6_1_spring_oscillator"]:
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("kspring_s1", key(concept_kspring, "s1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_6_2_rolling_shapes"]:
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("radius_o1", key(concept_radius, "o1"), D),
            ColumnSpec("cx_slope", key(concept_cx, "slope"), D),
            ColumnSpec("cy_slope", key(concept_cy, "slope"), D),
            ColumnSpec("cz_slope", key(concept_cz, "slope"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode in (MODE["strict_6_3_pendulum_small"], MODE["strict_6_4_damped_oscillator"]):
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("length_s1", key(concept_length, "s1"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_x_1_elastic_contact"]:
        return [
            ColumnSpec("posz_probe", key(concept_posz, "probe"), D),
            ColumnSpec("radius_probe", key(concept_radius, "probe"), D),
            ColumnSpec("control_u", key(concept_control_u, "ctrl"), C),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode in (MODE["strict_7_1_projectile_no_drag"], MODE["strict_7_2_projectile_linear_drag"], MODE["strict_7_3_vertical_fall_quadratic"]):
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("posy_o1", key(concept_posy, "o1"), D),
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("radius_o1", key(concept_radius, "o1"), D),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_7_4_forced_oscillator"]:
        return [
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("length_s1", key(concept_length, "s1"), D),
            ColumnSpec("drive_f0", key(concept_drive_f0, "ctrl"), C),
            ColumnSpec("drive_omega", key(concept_drive_omega, "ctrl"), C),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("kspring_s1", key(concept_kspring, "s1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_y_1_high_speed_kinematics"]:
        return [
            ColumnSpec("posx_o1", key(concept_posx, "o1"), D),
            ColumnSpec("energy_input", key(concept_energy_input, "ctrl"), C),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_y_2_relativistic_rocket"]:
        return [
            ColumnSpec("posz_o1", key(concept_posz, "o1"), D),
            ColumnSpec("fuel_mass", key(concept_fuel_mass, "o1"), C),
            ColumnSpec("mass_o1", key(concept_mass, "o1"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if 800 <= mode < 900:
        return _thermal_specs(mode)
    if 900 <= mode < 1000:
        return _optics_specs(mode)
    raise ValueError(f"Unsupported strict ladder mode: {mode}")


def _thermal_specs(mode: int) -> List[ColumnSpec]:
    D = ColumnRole.DEFAULT
    U = ColumnRole.UNLOCKED
    C = ColumnRole.CONTROL
    H = ColumnRole.HIDDEN
    if mode == MODE["strict_t_1_temperature_sensor"]:
        return [
            ColumnSpec("temperature_a", key(concept_temperature, "a"), D),
            ColumnSpec("temperature_b", key(concept_temperature, "b"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode in (MODE["strict_t_2_ideal_gas_state"], MODE["strict_t_3_adiabatic_gas"], MODE["strict_t_6_van_der_waals_gas"]):
        return [
            ColumnSpec("pressure_gas", key(concept_pressure, "gas"), U),
            ColumnSpec("volume_gas", key(concept_volume, "gas"), U),
            ColumnSpec("temperature_gas", key(concept_temperature, "gas"), U),
            ColumnSpec("samplemass_gas", key(concept_samplemass, "gas"), D),
            ColumnSpec("molarmass_gas", key(concept_molarmass, "gas"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_t_4_heat_capacity_calibration"]:
        return [
            ColumnSpec("temperature_sample", key(concept_temperature, "sample"), U),
            ColumnSpec("power_heater", key(concept_power, "heater"), C),
            ColumnSpec("mass_sample", key(concept_mass, "sample"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_t_5_cooling_and_conduction"]:
        return [
            ColumnSpec("temperature_a", key(concept_temperature, "a"), U),
            ColumnSpec("temperature_b", key(concept_temperature, "b"), U),
            ColumnSpec("temperature_env", key(concept_temperature, "env"), D),
            ColumnSpec("length_rod", key(concept_particle_length, "rod"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_t_7_solid_heat_capacity"]:
        return [
            ColumnSpec("temperature_sample", key(concept_temperature, "sample"), U),
            ColumnSpec("heat_sample", key(concept_heat, "sample"), U),
            ColumnSpec("atom_count", key(concept_atom_count, "sample"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_t_8_entropy_expansion"]:
        return [
            ColumnSpec("pressure_gas", key(concept_pressure, "gas"), U),
            ColumnSpec("volume_gas", key(concept_volume, "gas"), U),
            ColumnSpec("temperature_gas", key(concept_temperature, "gas"), U),
            ColumnSpec("heat_gas", key(concept_heat, "gas"), U),
            ColumnSpec("entropy_gas", key(concept_entropy, "gas"), H),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    raise ValueError(f"Unsupported thermal mode: {mode}")


def _optics_specs(mode: int) -> List[ColumnSpec]:
    D = ColumnRole.DEFAULT
    U = ColumnRole.UNLOCKED
    C = ColumnRole.CONTROL
    H = ColumnRole.HIDDEN
    if mode == MODE["strict_o_1_light_time_of_flight"]:
        return [
            ColumnSpec("path_length", key(concept_path_length, "beam"), D),
            ColumnSpec("arrival_time", key(concept_arrival_time, "beam"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_o_2_interference_wavelength"]:
        return [
            ColumnSpec("x_detector", key(concept_position, "detector"), D),
            ColumnSpec("intensity_detector", key(concept_intensity, "detector"), D),
            ColumnSpec("length_path", key(concept_particle_length, "screen"), D),
            ColumnSpec("dist_slit", key(concept_dist, "slit1", "slit2"), D),
            ColumnSpec("wavelength", key(concept_wavelength, "beam"), H),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_o_3_frequency_relation"]:
        return [
            ColumnSpec("frequency_beam", key(concept_frequency, "beam"), C),
            ColumnSpec("wavelength", key(concept_wavelength, "beam"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_o_4_diffraction"]:
        return [
            ColumnSpec("x_detector", key(concept_position, "detector"), D),
            ColumnSpec("intensity_detector", key(concept_intensity, "detector"), D),
            ColumnSpec("length_path", key(concept_particle_length, "screen"), D),
            ColumnSpec("area_aperture", key(concept_area, "aperture"), D),
            ColumnSpec("wavelength", key(concept_wavelength, "beam"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_o_5_photoelectric_effect"]:
        return [
            ColumnSpec("frequency_beam", key(concept_frequency, "beam"), C),
            ColumnSpec("intensity_beam", key(concept_intensity, "beam"), C),
            ColumnSpec("stop_voltage", key(concept_stop_voltage, "detector"), D),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    if mode == MODE["strict_o_6_blackbody_spectrum"]:
        return [
            ColumnSpec("wavelength", key(concept_wavelength, "beam"), U),
            ColumnSpec("spectral_intensity", key(concept_spectral, "detector"), D),
            ColumnSpec("temperature_body", key(concept_temperature, "body"), U),
            ColumnSpec("t", key(concept_t, "clock"), D),
        ]
    raise ValueError(f"Unsupported optics mode: {mode}")


def _config(
    name: str,
    objects: Dict[str, Objstructure],
    specs: Sequence[ColumnSpec],
    spdim: int = 1,
    unlocked_concepts: Iterable[str] = (),
    column_roles: Mapping[str, ColumnRole | str] | None = None,
) -> ExpStructure:
    specs = resolve_column_specs(specs, unlocked_concepts, column_roles)
    audit_specs(specs)
    open_flags = {
        f"open__{spec.name}": _fixed(1.0 if spec.is_open else 0.0)
        for spec in specs
    }
    expconfig = ExpConfig(name, spdim, _base_para(MODE[name], **open_flags), objects, open_data_info(specs))
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)


def _noise(error: float, n: int) -> np.ndarray:
    return np.random.normal(0, error, n)


def _time(t_end: float, t_num: int) -> np.ndarray:
    return np.linspace(0.0, t_end, t_num, endpoint=False)


def _insert(exp_config, t_num: int, specs: Sequence[ColumnSpec], values: Dict[str, np.ndarray]):
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    insert_open_columns(ds, specs, values)
    return ds


def _with_common_zeros(values: Dict[str, np.ndarray], objects: Sequence[str], t_num: int, error: float) -> None:
    for obj in objects:
        values.setdefault(f"posy_{obj}", _noise(error, t_num))
        values.setdefault(f"posz_{obj}", _noise(error, t_num))


def do_experiment(t_end: float, t_num: int, error: float, exp_config: ExpConfig):
    mode = int(round(exp_config.para("mode")))
    specs = _runtime_specs(mode, exp_config)
    t = _time(t_end, t_num)
    values = _generate(mode, t, error, exp_config)
    values["t"] = t + _noise(error, t_num)
    return _insert(exp_config, t_num, specs, values)


def _runtime_specs(mode: int, exp_config: ExpConfig) -> List[ColumnSpec]:
    specs = _specs(mode)
    resolved = []
    for spec in specs:
        is_open = exp_config.para(f"open__{spec.name}") > 0.5
        role = spec.role if is_open else ColumnRole.HIDDEN
        if is_open and spec.role is ColumnRole.HIDDEN:
            role = ColumnRole.UNLOCKED
        resolved.append(ColumnSpec(spec.name, spec.key, role, spec.provenance, spec.unlock))
    return resolved


def _obj_mass(exp_config: ExpConfig, name: str, fallback: float) -> float:
    """Read the simulated object's randomized inertial mass.

    The simulator may randomize the particle mass on every ``random_settings``
    call; the collision must respond to it so the additive weight is
    identifiable.  Hidden from data_info; used only to generate traces.
    """
    if exp_config is None:
        return float(fallback)
    try:
        return float(exp_config.get_obj_para(name, "m"))
    except Exception:
        return float(fallback)


def _generate(mode: int, t: np.ndarray, error: float, exp_config: ExpConfig | None = None) -> Dict[str, np.ndarray]:
    n = len(t)
    g = 9.801234567
    values: Dict[str, np.ndarray] = {}
    if mode == MODE["strict_1_1_collision_pair"]:
        m1 = _obj_mass(exp_config, "a", 2.0)
        m2 = _obj_mass(exp_config, "b", 4.0)
        x10, x20, v10, v20 = -2.0, 2.0, 1.8, -0.5
        tc = (x20 - x10) / (v10 - v20)
        v1p = ((m1 - m2) * v10 + 2 * m2 * v20) / (m1 + m2)
        v2p = ((m2 - m1) * v20 + 2 * m1 * v10) / (m1 + m2)
        x1 = np.where(t < tc, x10 + v10 * t, x10 + v10 * tc + v1p * (t - tc))
        x2 = np.where(t < tc, x20 + v20 * t, x20 + v20 * tc + v2p * (t - tc))
        values.update(posx_a=x1 + _noise(error, n), posx_b=x2 + _noise(error, n), dist_ab=x2 - x1 + _noise(error, n), mass_a=np.full(n, m1), mass_b=np.full(n, m2))
        _with_common_zeros(values, ("a", "b"), n, error)
    elif mode == MODE["strict_1_2_collision_reference"]:
        # Three-body 1D elastic collisions with randomized inertial masses.
        m1 = _obj_mass(exp_config, "a", 1.0)
        m2 = _obj_mass(exp_config, "b", 2.0)
        m3 = _obj_mass(exp_config, "c", 3.0)
        x10, x20, x30 = -4.0, -0.5, 3.0
        v10, v20, v30 = 1.5, 0.2, -1.0
        # Resolve pairwise elastic collisions in time order (simple event model).
        positions = {
            "a": [x10], "b": [x20], "c": [x30],
        }
        masses = {"a": m1, "b": m2, "c": m3}
        pos = {"a": x10, "b": x20, "c": x30}
        vel = {"a": v10, "b": v20, "c": v30}
        prev_t = 0.0
        traj = {"a": [], "b": [], "c": []}
        order = ("a", "b", "c")
        # Integrate piecewise, applying the next pairwise collision in time.
        remaining = list(t)
        idx = 0
        cur_t = 0.0
        while idx < len(t):
            # time to next collision among approaching pairs
            next_tc = None
            pair = None
            for i in range(3):
                for j in range(i + 1, 3):
                    oi, oj = order[i], order[j]
                    if vel[oi] - vel[oj] <= 1e-12:
                        continue
                    dtc = (pos[oj] - pos[oi]) / (vel[oi] - vel[oj])
                    if dtc < 0:
                        continue
                    if next_tc is None or dtc < next_tc:
                        next_tc, pair = dtc, (oi, oj)
            if next_tc is None:
                stop = t[-1]
            else:
                stop = cur_t + next_tc
            while idx < len(t) and t[idx] <= stop + 1e-12:
                for o in order:
                    traj[o].append(pos[o] + vel[o] * (t[idx] - cur_t))
                idx += 1
            if next_tc is None:
                break
            for o in order:
                pos[o] = pos[o] + vel[o] * (stop - cur_t)
            oi, oj = pair
            mi, mj = masses[oi], masses[oj]
            vi, vj = vel[oi], vel[oj]
            vel[oi] = ((mi - mj) * vi + 2 * mj * vj) / (mi + mj)
            vel[oj] = ((mj - mi) * vj + 2 * mi * vi) / (mi + mj)
            cur_t = stop
        x1 = np.asarray(traj["a"]); x2 = np.asarray(traj["b"]); x3 = np.asarray(traj["c"])
        values.update(posx_a=x1, posx_b=x2, posx_c=x3, dist_ab=x2 - x1, dist_bc=x3 - x2, dist_ac=x3 - x1)
        _with_common_zeros(values, ("a", "b", "c"), n, error)
        values["mass_a"] = np.full(n, m1)
        values["mass_b"] = np.full(n, m2)
        values["mass_c"] = np.full(n, m3)
    elif mode in (MODE["strict_2_1_free_fall_time"], MODE["strict_4_1_drop_repeat"]):
        z = 5.0 - 0.5 * g * t**2
        values.update(posx_o1=_noise(error, n), posy_o1=_noise(error, n), posz_o1=z + _noise(error, n), mass_o1=np.full(n, 3.0))
    elif mode == MODE["strict_2_2_incline_slide"]:
        theta = math.pi / 6
        a = g * math.sin(theta)
        s = 0.5 * a * t**2
        values.update(
            posx_o1=s * math.cos(theta) + _noise(error, n),
            posy_o1=_noise(error, n),
            posz_o1=-s * math.sin(theta) + _noise(error, n),
            cx_slope=np.full(n, math.sin(theta)),
            cy_slope=np.zeros(n),
            cz_slope=np.full(n, math.cos(theta)),
        )
    elif mode == MODE["strict_2_3_atwood"]:
        m1, m2 = 2.0, 3.0
        a = (m2 - m1) * g / (m1 + m2)
        z1 = 0.5 * a * t**2
        z2 = -z1
        values.update(posz_a=z1, posz_b=z2, mass_a=np.full(n, m1), mass_b=np.full(n, m2))
    elif mode == MODE["strict_3_1_mass_under_force"]:
        m = 2.0
        u = 3.0
        pos = 0.5 * (u / m) * t**2
        values.update(posx_o1=pos, control_u=np.full(n, u), mass_o1=np.full(n, m))
    elif mode == MODE["strict_3_2_hanging_spring"]:
        m, k, l0 = 2.0, 12.0, 1.0
        length = l0 + m * g / k + 0.05 * np.cos(np.sqrt(k / m) * t)
        values.update(length_s1=length, posz_o1=2.0 - length, mass_o1=np.full(n, m), kspring_s1=np.full(n, k))
    elif mode == MODE["strict_4_2_orbit_two_body"]:
        r, omega, m1, m2 = 3.0, 0.7, 5.0, 2.0
        x1, y1 = r * np.cos(omega * t), r * np.sin(omega * t)
        x2, y2 = -(m1 / m2) * x1, -(m1 / m2) * y1
        values.update(posx_a=x1, posy_a=y1, posx_b=x2, posy_b=y2, dist_ab=np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), mass_a=np.full(n, m1), mass_b=np.full(n, m2))
    elif mode in (MODE["strict_5_1_volume_by_displacement"], MODE["strict_5_2_density_emergence"]):
        area, radius = 2.0, 0.5
        volume = 4.0 / 3.0 * math.pi * radius**3
        surface = 1.0 + volume / area * (t > t[len(t) // 4])
        values.update(posz_surface=surface, area_container=np.full(n, area), radius_body=np.full(n, radius), volume_body=np.full(n, volume), mass_body=np.full(n, 1.6))
    elif mode == MODE["strict_5_3_pressure_discovery"]:
        area, m = 0.4, 1.5
        pressure = 14.0 + 0.8 * np.sin(1.2 * t)
        a = (pressure * area - m * g) / m
        pos = 0.5 * cumulative_trapezoid(cumulative_trapezoid(a, t, initial=0), t, initial=0)
        values.update(posz_piston=pos, area_piston=np.full(n, area), mass_piston=np.full(n, m), pressure_gas=pressure)
    elif mode == MODE["strict_5_4_hydrostatic_pressure"]:
        rho = 1.2
        depth = 0.2 + 0.1 * t
        values.update(depth_probe=depth, pressure_probe=rho * g * depth)
    elif mode == MODE["strict_5_5_sphere_terminal_fall"]:
        vt = 1.5
        z = 5.0 - vt * t - 0.5 * np.exp(-t)
        values.update(posz_ball=z, radius_ball=np.full(n, 0.2), mass_ball=np.full(n, 0.4))
    elif mode == MODE["strict_5_6_pipe_flow"]:
        r, L, mu, dp = 0.2, 2.0, 0.05, 4.0
        q = math.pi * dp * r**4 / (8 * mu * L)
        values.update(radius_pipe=np.full(n, r), length_pipe=np.full(n, L), volume_out=q * t, pressure_in=np.full(n, 5.0), pressure_out=np.full(n, 1.0))
    elif mode == MODE["strict_6_1_spring_oscillator"]:
        m, k = 2.0, 8.0
        values.update(posx_o1=np.cos(np.sqrt(k / m) * t), mass_o1=np.full(n, m), kspring_s1=np.full(n, k))
    elif mode == MODE["strict_6_2_rolling_shapes"]:
        theta, beta = math.pi / 6, 0.4
        a = g * math.sin(theta) / (1 + beta)
        values.update(posx_o1=0.5 * a * t**2, radius_o1=np.full(n, 0.4), cx_slope=np.full(n, math.sin(theta)), cy_slope=np.zeros(n), cz_slope=np.full(n, math.cos(theta)), mass_o1=np.full(n, 1.0))
    elif mode in (MODE["strict_6_3_pendulum_small"], MODE["strict_6_4_damped_oscillator"]):
        L, gamma = 1.5, 0.25 if mode == MODE["strict_6_4_damped_oscillator"] else 0.0
        theta = 0.25 * np.exp(-gamma * t) * np.cos(np.sqrt(g / L) * t)
        values.update(posx_o1=L * np.sin(theta), posz_o1=-L * np.cos(theta), length_s1=np.full(n, L), mass_o1=np.full(n, 1.0))
    elif mode == MODE["strict_x_1_elastic_contact"]:
        delta = 0.01 + 0.002 * t
        values.update(posz_probe=-delta, radius_probe=np.full(n, 0.2), control_u=delta ** 1.5)
    elif mode == MODE["strict_7_1_projectile_no_drag"]:
        v0, theta = 5.0, math.pi / 4
        values.update(posx_o1=v0 * math.cos(theta) * t, posy_o1=v0 * math.sin(theta) * t - 0.5 * g * t**2, posz_o1=_noise(error, n), radius_o1=np.full(n, 0.1), mass_o1=np.full(n, 1.0))
    elif mode == MODE["strict_7_2_projectile_linear_drag"]:
        v0, theta, kdrag = 5.0, math.pi / 4, 0.25
        vx0, vy0 = v0 * math.cos(theta), v0 * math.sin(theta)
        x = vx0 * (1 - np.exp(-kdrag * t)) / kdrag
        y = (vy0 + g / kdrag) * (1 - np.exp(-kdrag * t)) / kdrag - g * t / kdrag
        values.update(posx_o1=x, posy_o1=y, posz_o1=_noise(error, n), radius_o1=np.full(n, 0.1), mass_o1=np.full(n, 1.0))
    elif mode == MODE["strict_7_3_vertical_fall_quadratic"]:
        vt = 3.0
        z = 8.0 - (vt**2 / g) * np.log(np.cosh(g * t / vt))
        values.update(posx_o1=_noise(error, n), posy_o1=_noise(error, n), posz_o1=z, radius_o1=np.full(n, 0.15), mass_o1=np.full(n, 0.5))
    elif mode == MODE["strict_7_4_forced_oscillator"]:
        m, k, gamma, f0, omega = 2.0, 8.0, 0.3, 1.5, 1.8
        amp = (f0 / m) / math.sqrt((k / m - omega**2) ** 2 + (gamma * omega) ** 2)
        z = amp * np.cos(omega * t - 0.4)
        values.update(posz_o1=z, length_s1=1.0 + z, drive_f0=np.full(n, f0), drive_omega=np.full(n, omega), mass_o1=np.full(n, m), kspring_s1=np.full(n, k))
    elif mode == MODE["strict_y_1_high_speed_kinematics"]:
        c = 10.0
        energy = np.linspace(0.1, 8.0, n)
        v = c * np.sqrt(1 - 1 / (1 + energy) ** 2)
        values.update(posx_o1=cumulative_trapezoid(v, t, initial=0), energy_input=energy, mass_o1=np.full(n, 1.0))
    elif mode == MODE["strict_y_2_relativistic_rocket"]:
        c, ve = 10.0, 1.5
        fuel = np.maximum(0.2, 2.0 - 0.2 * t)
        m0 = 3.0
        v = c * np.tanh((ve / c) * np.log(m0 / (1.0 + fuel)))
        values.update(posz_o1=cumulative_trapezoid(v, t, initial=0), fuel_mass=fuel, mass_o1=1.0 + fuel)
    elif 800 <= mode < 900:
        values.update(_generate_thermal(mode, t, error))
    elif 900 <= mode < 1000:
        values.update(_generate_optics(mode, t, error))
    else:
        raise ValueError(f"Unsupported strict ladder mode: {mode}")
    return {name: np.asarray(vals, dtype=float) + (0 if name == "t" else 0) for name, vals in values.items()}


def _generate_thermal(mode: int, t: np.ndarray, error: float) -> Dict[str, np.ndarray]:
    n = len(t)
    R = 8.314
    if mode == MODE["strict_t_1_temperature_sensor"]:
        eq = 300.0
        return {"temperature_a": eq + 30 * np.exp(-0.5 * t), "temperature_b": eq - 20 * np.exp(-0.5 * t)}
    if mode == MODE["strict_t_2_ideal_gas_state"]:
        amount, T = 1.2, 280 + 20 * np.sin(0.5 * t)
        V = 1.0 + 0.1 * np.cos(0.7 * t)
        return {"pressure_gas": amount * R * T / V, "volume_gas": V, "temperature_gas": T, "samplemass_gas": np.full(n, 0.0348), "molarmass_gas": np.full(n, 0.029)}
    if mode == MODE["strict_t_3_adiabatic_gas"]:
        gamma, V0, P0 = 1.4, 1.0, 1000.0
        V = V0 * (1 + 0.15 * np.sin(0.5 * t))
        P = P0 * (V0 / V) ** gamma
        T = P * V / (1.0 * R)
        return {"pressure_gas": P, "volume_gas": V, "temperature_gas": T, "samplemass_gas": np.full(n, 0.029), "molarmass_gas": np.full(n, 0.029)}
    if mode == MODE["strict_t_4_heat_capacity_calibration"]:
        power = np.full(n, 8.0)
        C = 40.0
        return {"temperature_sample": 290 + cumulative_trapezoid(power, t, initial=0) / C, "power_heater": power, "mass_sample": np.full(n, 0.5)}
    if mode == MODE["strict_t_5_cooling_and_conduction"]:
        env = np.full(n, 295.0)
        a = 330 - 35 * (1 - np.exp(-0.4 * t))
        b = 280 + 15 * (1 - np.exp(-0.4 * t))
        return {"temperature_a": a, "temperature_b": b, "temperature_env": env, "length_rod": np.full(n, 1.0)}
    if mode == MODE["strict_t_6_van_der_waals_gas"]:
        amount, a_vdw, b_vdw, T = 1.0, 0.5, 0.04, 310 + 10 * np.sin(0.3 * t)
        V = 1.0 + 0.2 * np.sin(0.5 * t)
        P = amount * R * T / (V - amount * b_vdw) - a_vdw * amount**2 / V**2
        return {"pressure_gas": P, "volume_gas": V, "temperature_gas": T, "samplemass_gas": np.full(n, 0.029), "molarmass_gas": np.full(n, 0.029)}
    if mode == MODE["strict_t_7_solid_heat_capacity"]:
        T = np.linspace(5, 300, n)
        theta = 120.0
        Cv = 3 * (T / (T + theta)) ** 3
        Q = cumulative_trapezoid(Cv, T, initial=0)
        return {"temperature_sample": T, "heat_sample": Q, "atom_count": np.full(n, 1.0)}
    if mode == MODE["strict_t_8_entropy_expansion"]:
        amount, T = 1.0, np.full(n, 300.0)
        V = 1.0 + 0.1 * t
        P = amount * R * T / V
        Q = amount * R * T * np.log(V / V[0])
        S = amount * R * np.log(V / V[0])
        return {"pressure_gas": P, "volume_gas": V, "temperature_gas": T, "heat_gas": Q, "entropy_gas": S}
    raise ValueError(f"Unsupported thermal mode: {mode}")


def _generate_optics(mode: int, t: np.ndarray, error: float) -> Dict[str, np.ndarray]:
    n = len(t)
    c = 3.0e8
    if mode == MODE["strict_o_1_light_time_of_flight"]:
        L = np.linspace(1.0, 5.0, n)
        return {"path_length": L, "arrival_time": L / c}
    if mode == MODE["strict_o_2_interference_wavelength"]:
        x = np.linspace(-0.02, 0.02, n)
        lam, L, d = 600e-9, 1.0, 50e-6
        I = 0.5 * (1 + np.cos(2 * math.pi * d * x / (lam * L)))
        return {"x_detector": x, "intensity_detector": I, "length_path": np.full(n, L), "dist_slit": np.full(n, d), "wavelength": np.full(n, lam)}
    if mode == MODE["strict_o_3_frequency_relation"]:
        nu = np.linspace(4.0e14, 7.0e14, n)
        return {"frequency_beam": nu, "wavelength": c / nu}
    if mode == MODE["strict_o_4_diffraction"]:
        x = np.linspace(-0.02, 0.02, n)
        lam, L, a = 550e-9, 1.0, 60e-6
        beta = math.pi * a * x / (lam * L)
        I = np.sinc(beta / math.pi) ** 2
        return {"x_detector": x, "intensity_detector": I, "length_path": np.full(n, L), "area_aperture": np.full(n, a), "wavelength": np.full(n, lam)}
    if mode == MODE["strict_o_5_photoelectric_effect"]:
        h_over_e, phi_over_e = 4.1357e-15, 1.8
        nu = np.linspace(5e14, 9e14, n)
        V = np.maximum(0, h_over_e * nu - phi_over_e)
        return {"frequency_beam": nu, "intensity_beam": np.linspace(0.5, 2.0, n), "stop_voltage": V}
    if mode == MODE["strict_o_6_blackbody_spectrum"]:
        h, kb, temp = 6.626e-34, 1.381e-23, 4200.0
        lam = np.linspace(300e-9, 2500e-9, n)
        B = (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * kb * temp)) - 1)
        return {"wavelength": lam, "spectral_intensity": B / np.max(B), "temperature_body": np.full(n, temp)}
    raise ValueError(f"Unsupported optics mode: {mode}")


def strict_1_1_collision_pair_config():
    return _config("strict_1_1_collision_pair", _objects("a", "b"), _specs(MODE["strict_1_1_collision_pair"]), 1)


def strict_1_2_collision_reference_config():
    return _config("strict_1_2_collision_reference", _objects("a", "b", "c"), _specs(MODE["strict_1_2_collision_reference"]), 1)


def strict_2_1_free_fall_time_config():
    return _config("strict_2_1_free_fall_time", _objects("o1"), _specs(MODE["strict_2_1_free_fall_time"]), 1)


def strict_2_2_incline_slide_config():
    return _config("strict_2_2_incline_slide", _objects("o1", slope=True), _specs(MODE["strict_2_2_incline_slide"]), 2)


def strict_2_3_atwood_config():
    return _config("strict_2_3_atwood", _objects("a", "b"), _specs(MODE["strict_2_3_atwood"]), 1)


def strict_3_1_mass_under_force_config():
    return _config("strict_3_1_mass_under_force", _objects("o1", "ctrl"), _specs(MODE["strict_3_1_mass_under_force"]), 1)


def strict_3_2_hanging_spring_config():
    return _config("strict_3_2_hanging_spring", _objects("o1", spring=True), _specs(MODE["strict_3_2_hanging_spring"]), 1)


def strict_4_1_drop_repeat_config():
    return _config("strict_4_1_drop_repeat", _objects("o1"), _specs(MODE["strict_4_1_drop_repeat"]), 1)


def strict_4_2_orbit_two_body_config():
    return _config("strict_4_2_orbit_two_body", _objects("a", "b"), _specs(MODE["strict_4_2_orbit_two_body"]), 2)


def _simple_factory(
    name: str,
    objs: Sequence[str],
    *,
    spring: bool = False,
    slope: bool = False,
    spdim: int = 1,
    unlocked_concepts: Iterable[str] = (),
    column_roles: Mapping[str, ColumnRole | str] | None = None,
):
    return _config(
        name,
        _objects(*objs, spring=spring, slope=slope),
        _specs(MODE[name]),
        spdim,
        unlocked_concepts,
        column_roles,
    )


def make_ladder_strict_config(
    name: str,
    unlocked_concepts: Iterable[str] = (),
    column_roles: Mapping[str, ColumnRole | str] | None = None,
) -> ExpStructure:
    """Build any strict ladder experiment from certified framework state."""
    if name not in STRICT_EXPERIMENT_OBJECTS:
        raise KeyError(f"Unknown strict ladder experiment: {name}")
    objs, kwargs, spdim = STRICT_EXPERIMENT_OBJECTS[name]
    return _simple_factory(
        name,
        objs,
        spdim=spdim,
        unlocked_concepts=unlocked_concepts,
        column_roles=column_roles,
        **kwargs,
    )


def strict_5_1_volume_by_displacement_config():
    return _simple_factory("strict_5_1_volume_by_displacement", ("surface", "container", "body"))


def strict_5_2_density_emergence_config():
    return _simple_factory("strict_5_2_density_emergence", ("surface", "container", "body"))


def strict_5_3_pressure_discovery_config():
    return _simple_factory("strict_5_3_pressure_discovery", ("piston", "gas"))


def strict_5_4_hydrostatic_pressure_config():
    return _simple_factory("strict_5_4_hydrostatic_pressure", ("probe",))


def strict_5_5_sphere_terminal_fall_config():
    return _simple_factory("strict_5_5_sphere_terminal_fall", ("ball",))


def strict_5_6_pipe_flow_config():
    return _simple_factory("strict_5_6_pipe_flow", ("pipe", "outflow", "pin", "pout"))


def strict_6_1_spring_oscillator_config():
    return _simple_factory("strict_6_1_spring_oscillator", ("o1",), spring=True)


def strict_6_2_rolling_shapes_config():
    return _simple_factory("strict_6_2_rolling_shapes", ("o1",), slope=True)


def strict_6_3_pendulum_small_config():
    return _simple_factory("strict_6_3_pendulum_small", ("o1",), spring=True)


def strict_6_4_damped_oscillator_config():
    return _simple_factory("strict_6_4_damped_oscillator", ("o1",), spring=True)


def strict_x_1_elastic_contact_config():
    return _simple_factory("strict_x_1_elastic_contact", ("probe", "ctrl"))


def strict_7_1_projectile_no_drag_config():
    return _simple_factory("strict_7_1_projectile_no_drag", ("o1",), spdim=2)


def strict_7_2_projectile_linear_drag_config():
    return _simple_factory("strict_7_2_projectile_linear_drag", ("o1",), spdim=2)


def strict_7_3_vertical_fall_quadratic_config():
    return _simple_factory("strict_7_3_vertical_fall_quadratic", ("o1",), spdim=1)


def strict_7_4_forced_oscillator_config():
    return _simple_factory("strict_7_4_forced_oscillator", ("o1", "ctrl"), spring=True)


def strict_y_1_high_speed_kinematics_config():
    return _simple_factory("strict_y_1_high_speed_kinematics", ("o1", "ctrl"))


def strict_y_2_relativistic_rocket_config():
    return _simple_factory("strict_y_2_relativistic_rocket", ("o1",))


def strict_t_1_temperature_sensor_config():
    return _simple_factory("strict_t_1_temperature_sensor", ("a", "b"))


def strict_t_2_ideal_gas_state_config():
    return _simple_factory("strict_t_2_ideal_gas_state", ("gas",))


def strict_t_3_adiabatic_gas_config():
    return _simple_factory("strict_t_3_adiabatic_gas", ("gas",))


def strict_t_4_heat_capacity_calibration_config():
    return _simple_factory("strict_t_4_heat_capacity_calibration", ("sample", "heater"))


def strict_t_5_cooling_and_conduction_config():
    return _simple_factory("strict_t_5_cooling_and_conduction", ("a", "b", "env", "rod"))


def strict_t_6_van_der_waals_gas_config():
    return _simple_factory("strict_t_6_van_der_waals_gas", ("gas",))


def strict_t_7_solid_heat_capacity_config():
    return _simple_factory("strict_t_7_solid_heat_capacity", ("sample",))


def strict_t_8_entropy_expansion_config():
    return _simple_factory("strict_t_8_entropy_expansion", ("gas",))


def strict_o_1_light_time_of_flight_config():
    return _simple_factory("strict_o_1_light_time_of_flight", ("beam",))


def strict_o_2_interference_wavelength_config():
    return _simple_factory("strict_o_2_interference_wavelength", ("detector", "screen", "slit1", "slit2", "beam"))


def strict_o_3_frequency_relation_config():
    return _simple_factory("strict_o_3_frequency_relation", ("beam",))


def strict_o_4_diffraction_config():
    return _simple_factory("strict_o_4_diffraction", ("detector", "screen", "aperture", "beam"))


def strict_o_5_photoelectric_effect_config():
    return _simple_factory("strict_o_5_photoelectric_effect", ("beam", "detector"))


def strict_o_6_blackbody_spectrum_config():
    return _simple_factory("strict_o_6_blackbody_spectrum", ("beam", "detector", "body"))


LADDER_STRICT_FACTORIES = [
    strict_1_1_collision_pair_config,
    strict_1_2_collision_reference_config,
    strict_2_1_free_fall_time_config,
    strict_2_2_incline_slide_config,
    strict_2_3_atwood_config,
    strict_3_1_mass_under_force_config,
    strict_3_2_hanging_spring_config,
    strict_4_1_drop_repeat_config,
    strict_4_2_orbit_two_body_config,
    strict_5_1_volume_by_displacement_config,
    strict_5_2_density_emergence_config,
    strict_5_3_pressure_discovery_config,
    strict_5_4_hydrostatic_pressure_config,
    strict_5_5_sphere_terminal_fall_config,
    strict_5_6_pipe_flow_config,
    strict_6_1_spring_oscillator_config,
    strict_6_2_rolling_shapes_config,
    strict_6_3_pendulum_small_config,
    strict_6_4_damped_oscillator_config,
    strict_x_1_elastic_contact_config,
    strict_7_1_projectile_no_drag_config,
    strict_7_2_projectile_linear_drag_config,
    strict_7_3_vertical_fall_quadratic_config,
    strict_7_4_forced_oscillator_config,
    strict_y_1_high_speed_kinematics_config,
    strict_y_2_relativistic_rocket_config,
    strict_t_1_temperature_sensor_config,
    strict_t_2_ideal_gas_state_config,
    strict_t_3_adiabatic_gas_config,
    strict_t_4_heat_capacity_calibration_config,
    strict_t_5_cooling_and_conduction_config,
    strict_t_6_van_der_waals_gas_config,
    strict_t_7_solid_heat_capacity_config,
    strict_t_8_entropy_expansion_config,
    strict_o_1_light_time_of_flight_config,
    strict_o_2_interference_wavelength_config,
    strict_o_3_frequency_relation_config,
    strict_o_4_diffraction_config,
    strict_o_5_photoelectric_effect_config,
    strict_o_6_blackbody_spectrum_config,
]
