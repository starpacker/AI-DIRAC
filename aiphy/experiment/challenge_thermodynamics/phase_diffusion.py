from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_thermodynamics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_temperature_data,
    integrate_thermal_system,
    make_thermal_expconfig,
    ranged_parastructure,
    smooth_heat_capacity,
    time_grid,
)


BASE_EXP_PARA = {
    "T10": ranged_parastructure(0.0, 140.0),
    "T20": ranged_parastructure(0.0, 140.0),
    "T30": ranged_parastructure(0.0, 140.0),
    "T40": ranged_parastructure(0.0, 140.0),
    "Tenv": ranged_parastructure(0.0, 100.0),
    "k": ranged_parastructure(0.4, 1.8),
    "k12": ranged_parastructure(0.4, 1.8),
    "k23": ranged_parastructure(0.4, 1.8),
    "k34": ranged_parastructure(0.4, 1.8),
    "Tm": ranged_parastructure(40.0, 70.0),
    "latent": ranged_parastructure(3.0, 8.0),
    "width": ranged_parastructure(2.0, 6.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    if mode == 1:
        c1 = exp_config.get_obj_para("o1", "m")
        T10 = exp_config.para("T10")
        Tenv = exp_config.para("Tenv")
        k = exp_config.para("k")
        Tm = exp_config.para("Tm")
        latent = exp_config.para("latent")
        width = exp_config.para("width")

        def rhs(y, _time_value):
            ceff = smooth_heat_capacity(y[0], c1, latent, Tm, width)
            return [k * (Tenv - y[0]) / ceff]

        temps = integrate_thermal_system(t, [T10], rhs)
    elif mode == 2:
        c1 = exp_config.get_obj_para("o1", "m")
        T10 = exp_config.para("T10")
        Tenv = exp_config.para("Tenv")
        k = exp_config.para("k")
        Tm = exp_config.para("Tm")
        latent = exp_config.para("latent")
        width = exp_config.para("width")

        def rhs(y, _time_value):
            ceff = smooth_heat_capacity(y[0], c1, latent, Tm, width)
            return [k * (Tenv - y[0]) / ceff]

        temps = integrate_thermal_system(t, [T10], rhs)
    elif mode == 3:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        k12 = exp_config.para("k12")
        Tm = exp_config.para("Tm")
        latent = exp_config.para("latent")
        width = exp_config.para("width")

        def rhs(y, _time_value):
            T1, T2 = y
            ceff1 = smooth_heat_capacity(T1, c1, latent, Tm, width)
            flow = k12 * (T2 - T1)
            return [flow / ceff1, -flow / c2]

        temps = integrate_thermal_system(t, [T10, T20], rhs)
    elif mode == 4:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        c3 = exp_config.get_obj_para("o3", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        T30 = exp_config.para("T30")
        k12 = exp_config.para("k12")
        k23 = exp_config.para("k23")

        def rhs(y, _time_value):
            T1, T2, T3 = y
            flow12 = k12 * (T2 - T1)
            flow23 = k23 * (T3 - T2)
            return [
                flow12 / c1,
                (-flow12 + flow23) / c2,
                -flow23 / c3,
            ]

        temps = integrate_thermal_system(t, [T10, T20, T30], rhs)
    elif mode == 5:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        c3 = exp_config.get_obj_para("o3", "m")
        c4 = exp_config.get_obj_para("o4", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        T30 = exp_config.para("T30")
        T40 = exp_config.para("T40")
        k12 = exp_config.para("k12")
        k23 = exp_config.para("k23")
        k34 = exp_config.para("k34")

        def rhs(y, _time_value):
            T1, T2, T3, T4 = y
            flow12 = k12 * (T2 - T1)
            flow23 = k23 * (T3 - T2)
            flow34 = k34 * (T4 - T3)
            return [
                flow12 / c1,
                (-flow12 + flow23) / c2,
                (-flow23 + flow34) / c3,
                -flow34 / c4,
            ]

        temps = integrate_thermal_system(t, [T10, T20, T30, T40], rhs)
    else:
        raise ValueError(f"Unsupported phase/diffusion thermal mode: {mode}")

    return insert_temperature_data(exp_config, t, error, temps)


def latent_heat_heating_plateau_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "latent_heat_heating_plateau",
        make_exp_para(
            mode=fixed_parastructure(1.0),
            T10=ranged_parastructure(0.0, 20.0),
            Tenv=ranged_parastructure(90.0, 120.0),
        ),
        [(1.0, 1.5)],
    )
    return finalize_structure(expconfig, __file__)


def latent_heat_cooling_plateau_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "latent_heat_cooling_plateau",
        make_exp_para(
            mode=fixed_parastructure(2.0),
            T10=ranged_parastructure(90.0, 120.0),
            Tenv=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 1.5)],
    )
    return finalize_structure(expconfig, __file__)


def latent_heat_contact_two_body_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "latent_heat_contact_two_body",
        make_exp_para(
            mode=fixed_parastructure(3.0),
            T10=ranged_parastructure(20.0, 35.0),
            T20=ranged_parastructure(90.0, 120.0),
        ),
        [(1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


def rod_three_node_diffusion_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "rod_three_node_diffusion",
        make_exp_para(
            mode=fixed_parastructure(4.0),
            T10=ranged_parastructure(90.0, 120.0),
            T20=ranged_parastructure(20.0, 40.0),
            T30=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


def rod_four_node_diffusion_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "rod_four_node_diffusion",
        make_exp_para(
            mode=fixed_parastructure(5.0),
            T10=ranged_parastructure(0.0, 10.0),
            T20=ranged_parastructure(90.0, 120.0),
            T30=ranged_parastructure(20.0, 35.0),
            T40=ranged_parastructure(0.0, 10.0),
        ),
        [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)
