import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_optics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_optics_data,
    linear_scan,
    make_single_particle_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "mode": fixed_parastructure(1.0),
    "xmin": ranged_parastructure(-2.0, -1.0),
    "xmax": ranged_parastructure(1.0, 2.0),
    "L": ranged_parastructure(3.0, 6.0),
    "n1": ranged_parastructure(1.0, 1.1),
    "n2": ranged_parastructure(1.3, 1.8),
    "d1": ranged_parastructure(1.0, 2.0),
    "d2": ranged_parastructure(1.0, 2.0),
    "theta_min": ranged_parastructure(0.05, 0.15),
    "theta_max": ranged_parastructure(0.55, 0.8),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    if mode == 1:
        x = linear_scan(exp_config.para("xmin"), exp_config.para("xmax"), t_num)
        L = exp_config.para("L")
        n1 = exp_config.para("n1")
        path = np.sqrt(L ** 2 + x ** 2)
        delay = n1 * path
        posx, posy, posz = x, path, delay
    elif mode == 2:
        theta = linear_scan(exp_config.para("theta_min"), exp_config.para("theta_max"), t_num)
        d1 = exp_config.para("d1")
        path = 2.0 * d1 / np.cos(theta)
        posx, posy, posz = theta, theta, path
    elif mode == 3:
        theta_i = linear_scan(exp_config.para("theta_min"), exp_config.para("theta_max"), t_num)
        n1 = exp_config.para("n1")
        n2 = exp_config.para("n2")
        d1 = exp_config.para("d1")
        d2 = exp_config.para("d2")
        theta_t = np.arcsin(np.clip((n1 / n2) * np.sin(theta_i), -0.999999, 0.999999))
        delay = n1 * d1 / np.cos(theta_i) + n2 * d2 / np.cos(theta_t)
        posx, posy, posz = theta_i, theta_t, delay
    elif mode == 4:
        theta_i = linear_scan(0.05, 0.7, t_num)
        n1 = exp_config.para("n2")
        n2 = exp_config.para("n1")
        d1 = exp_config.para("d2")
        d2 = exp_config.para("d1")
        critical = np.arcsin(min(0.999999, n2 / n1))
        theta_i = np.minimum(theta_i, critical - 0.05)
        theta_t = np.arcsin(np.clip((n1 / n2) * np.sin(theta_i), -0.999999, 0.999999))
        delay = n1 * d1 / np.cos(theta_i) + n2 * d2 / np.cos(theta_t)
        posx, posy, posz = theta_i, theta_t, delay
    elif mode == 5:
        n1 = exp_config.para("n2")
        n2 = exp_config.para("n1")
        critical = np.arcsin(min(0.999999, n2 / n1))
        theta_i = linear_scan(critical + 0.03, min(1.1, critical + 0.35), t_num)
        ratio = np.sqrt(np.sin(theta_i) ** 2 - (n2 / n1) ** 2)
        phase_shift = 2.0 * np.arctan2(ratio, np.maximum(np.cos(theta_i), 1e-6))
        posx, posy, posz = theta_i, theta_i, phase_shift
    else:
        raise ValueError(f"Unsupported optics interface mode: {mode}")

    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def homogeneous_path_scan_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "homogeneous_path_scan",
        2,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__)


def plane_mirror_equal_angle_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "plane_mirror_equal_angle",
        2,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posx[o1] - posy[o1]")])


def snell_air_to_glass_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "snell_air_to_glass",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__)


def snell_glass_to_air_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "snell_glass_to_air",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__)


def total_internal_reflection_phase_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "total_internal_reflection_phase",
        2,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posx[o1] - posy[o1]")])
