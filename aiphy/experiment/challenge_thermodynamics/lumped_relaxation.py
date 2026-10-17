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
    "T0": ranged_parastructure(20.0, 120.0),
    "Tenv": ranged_parastructure(0.0, 80.0),
    "k": ranged_parastructure(0.4, 1.4),
    "P": ranged_parastructure(0.0, 40.0),
    "Tamb_amp": ranged_parastructure(4.0, 16.0),
    "omega": ranged_parastructure(1.0, 3.0),
    "krad": ranged_parastructure(1e-5, 4e-5),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
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

    if int(round(mode)) == 1:
        def rhs(y, time_value):
            return [-(k / capacity) * (y[0] - Tenv)]
    elif int(round(mode)) == 2:
        def rhs(y, time_value):
            return [-(k / capacity) * (y[0] - Tenv) + P / capacity]
    elif int(round(mode)) == 3:
        def rhs(y, time_value):
            Tenv_t = Tenv + Tamb_amp * np.sin(omega * time_value)
            return [-(k / capacity) * (y[0] - Tenv_t)]
    elif int(round(mode)) == 4:
        def rhs(y, time_value):
            return [
                -(k / capacity) * (y[0] - Tenv)
                - (krad / capacity) * (y[0] ** 4 - Tenv ** 4)
            ]
    elif int(round(mode)) == 5:
        def rhs(y, time_value):
            heater = P if time_value < 0.7 * t_end else 0.0
            return [-(k / capacity) * (y[0] - Tenv) + heater / capacity]
    else:
        raise ValueError(f"Unsupported lumped thermal mode: {mode}")

    temps = integrate_thermal_system(t, [T0], rhs)
    return insert_temperature_data(exp_config, t, error, temps)


def lumped_cooling_static_env_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "lumped_cooling_static_env",
        make_exp_para(
            mode=fixed_parastructure(1.0),
            T0=ranged_parastructure(80.0, 120.0),
            Tenv=ranged_parastructure(5.0, 25.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def lumped_heating_static_env_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "lumped_heating_static_env",
        make_exp_para(
            mode=fixed_parastructure(1.0),
            T0=ranged_parastructure(0.0, 20.0),
            Tenv=ranged_parastructure(60.0, 100.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def lumped_cooling_dynamic_env_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "lumped_cooling_dynamic_env",
        make_exp_para(
            mode=fixed_parastructure(3.0),
            T0=ranged_parastructure(40.0, 90.0),
            Tenv=ranged_parastructure(20.0, 40.0),
            Tamb_amp=ranged_parastructure(6.0, 18.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def lumped_joule_heating_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "lumped_joule_heating",
        make_exp_para(
            mode=fixed_parastructure(2.0),
            T0=ranged_parastructure(5.0, 20.0),
            Tenv=ranged_parastructure(0.0, 10.0),
            P=ranged_parastructure(12.0, 36.0),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def lumped_radiative_cooling_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "lumped_radiative_cooling",
        make_exp_para(
            mode=fixed_parastructure(4.0),
            T0=ranged_parastructure(70.0, 120.0),
            Tenv=ranged_parastructure(0.0, 20.0),
            krad=ranged_parastructure(1e-5, 5e-5),
        ),
        [(1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)
