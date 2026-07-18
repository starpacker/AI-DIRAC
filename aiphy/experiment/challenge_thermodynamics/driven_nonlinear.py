import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_thermodynamics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_temperature_data,
    integrate_thermal_system,
    make_thermal_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "T0": ranged_parastructure(0.0, 120.0),
    "Tenv": ranged_parastructure(0.0, 80.0),
    "k": ranged_parastructure(0.4, 1.4),
    "A": ranged_parastructure(4.0, 18.0),
    "omega": ranged_parastructure(1.0, 3.0),
    "P": ranged_parastructure(8.0, 28.0),
    "delta": ranged_parastructure(4.0, 10.0),
    "beta": ranged_parastructure(0.01, 0.03),
    "switch_low": ranged_parastructure(25.0, 40.0),
    "switch_high": ranged_parastructure(45.0, 65.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    T0 = exp_config.para("T0")
    Tenv = exp_config.para("Tenv")
    k = exp_config.para("k")
    A = exp_config.para("A")
    omega = exp_config.para("omega")
    P = exp_config.para("P")
    delta = exp_config.para("delta")
    beta = exp_config.para("beta")
    switch_low = exp_config.para("switch_low")
    switch_high = exp_config.para("switch_high")
    capacity = exp_config.get_obj_para("o1", "m")
    t = time_grid(t_end, t_num)

    if mode == 1:
        def rhs(y, time_value):
            Tenv_t = Tenv + A * np.sin(omega * time_value)
            return [-(k / capacity) * (y[0] - Tenv_t)]

        temps = integrate_thermal_system(t, [T0], rhs)
    elif mode == 2:
        def rhs(y, time_value):
            heater = P * (0.5 + 0.5 * np.cos(omega * time_value)) ** 2
            return [-(k / capacity) * (y[0] - Tenv) + heater / capacity]

        temps = integrate_thermal_system(t, [T0], rhs)
    elif mode == 3:
        step = t_end / t_num
        heater_on = True
        values = np.zeros(t_num)
        values[0] = T0
        for idx in range(1, t_num):
            prev = values[idx - 1]
            if prev >= switch_high:
                heater_on = False
            elif prev <= switch_low:
                heater_on = True
            heater = P if heater_on else 0.0
            dT = -(k / capacity) * (prev - Tenv) + heater / capacity
            values[idx] = prev + step * dT
        temps = [values]
    elif mode == 4:
        def rhs(y, time_value):
            source = P * np.exp(beta * max(y[0] - Tenv, 0.0))
            return [-(k / capacity) * (y[0] - Tenv) + source / capacity]

        temps = integrate_thermal_system(t, [T0], rhs)
    elif mode == 5:
        def rhs(y, time_value):
            Tenv_t = Tenv - delta if time_value < 0.4 * t_end else Tenv + delta
            return [-(k / capacity) * (y[0] - Tenv_t)]

        temps = integrate_thermal_system(t, [T0], rhs)
    else:
        raise ValueError(f"Unsupported driven thermal mode: {mode}")

    return insert_temperature_data(exp_config, t, error, temps)


def driven_ambient_oscillation_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "driven_ambient_oscillation",
        make_exp_para(
            mode=fixed_parastructure(1.0),
            T0=ranged_parastructure(20.0, 60.0),
            Tenv=ranged_parastructure(30.0, 50.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def periodic_power_heating_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "periodic_power_heating",
        make_exp_para(
            mode=fixed_parastructure(2.0),
            T0=ranged_parastructure(5.0, 20.0),
            Tenv=ranged_parastructure(0.0, 10.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def thermostat_limit_cycle_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "thermostat_limit_cycle",
        make_exp_para(
            mode=fixed_parastructure(3.0),
            T0=ranged_parastructure(25.0, 40.0),
            Tenv=ranged_parastructure(0.0, 15.0),
            P=ranged_parastructure(18.0, 30.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def exothermic_self_heating_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "exothermic_self_heating",
        make_exp_para(
            mode=fixed_parastructure(4.0),
            T0=ranged_parastructure(10.0, 30.0),
            Tenv=ranged_parastructure(0.0, 10.0),
            P=ranged_parastructure(6.0, 14.0),
            beta=ranged_parastructure(0.015, 0.03),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def quench_recovery_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "quench_recovery",
        make_exp_para(
            mode=fixed_parastructure(5.0),
            T0=ranged_parastructure(50.0, 90.0),
            Tenv=ranged_parastructure(35.0, 55.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)
