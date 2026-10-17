"""Radiative cooling experiment — Reviewer Section 6, thermodynamics experiment 12.

dT/dt = -k_linear*(T - Tenv) - k_rad*(T^4 - Tenv^4)
Tests: Newton linear cooling only valid for small delta-T;
at large delta-T, linear is falsified and quartic nonlinear emerges.

Reuses the lumped_relaxation infrastructure (mode=4) with different parameter ranges.
"""
from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_thermodynamics.common import (
    finalize_structure, fixed_parastructure, insert_temperature_data,
    integrate_thermal_system, make_thermal_expconfig,
    ranged_parastructure, time_grid,
)

def do_experiment(t_end, t_num, error, exp_config):
    mode = exp_config.para("mode")
    T0 = exp_config.para("T0")
    Tenv = exp_config.para("Tenv")
    k = exp_config.para("k")
    P = exp_config.para("P")
    Tamb_amp = exp_config.para("Tamb_amp")
    omega = exp_config.para("omega")
    krad = exp_config.para("krad")
    capacity = exp_config.get_obj_para("o1", "m")
    t = time_grid(t_end, t_num)

    if int(round(mode)) == 4:
        def rhs(y, time_value):
            return [
                -(k / capacity) * (y[0] - Tenv)
                - (krad / capacity) * (y[0] ** 4 - Tenv ** 4)
            ]
    else:
        def rhs(y, time_value):
            return [-(k / capacity) * (y[0] - Tenv)]

    temps = integrate_thermal_system(t, [T0], rhs)
    return insert_temperature_data(exp_config, t, error, temps, tenv=Tenv)

def radiative_cooling_config():
    expconfig = make_thermal_expconfig(
        "radiative_cooling",
        {
            "mode": fixed_parastructure(4.0),
            "T0": ranged_parastructure(60.0, 150.0),
            "Tenv": ranged_parastructure(0.0, 30.0),
            "k": ranged_parastructure(0.05, 0.3),
            "P": fixed_parastructure(0.0),
            "Tamb_amp": fixed_parastructure(0.0),
            "omega": fixed_parastructure(0.0),
            "krad": ranged_parastructure(1e-6, 1e-4),
        },
        [(0.5, 2.0)],
    )
    return finalize_structure(expconfig, __file__)
