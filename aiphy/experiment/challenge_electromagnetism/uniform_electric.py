import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_electromagnetism.common import (
    finalize_structure,
    fixed_parastructure,
    insert_motion_data,
    make_single_particle_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "x0": ranged_parastructure(-1.5, 1.5),
    "y0": ranged_parastructure(-1.5, 1.5),
    "z0": ranged_parastructure(-1.5, 1.5),
    "vx0": ranged_parastructure(-1.0, 1.0),
    "vy0": ranged_parastructure(-1.0, 1.0),
    "vz0": ranged_parastructure(-1.0, 1.0),
    "Ex": ranged_parastructure(-0.8, 0.8),
    "Ey": ranged_parastructure(-0.8, 0.8),
    "Ez": ranged_parastructure(-0.8, 0.8),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    x0 = exp_config.para("x0")
    y0 = exp_config.para("y0")
    z0 = exp_config.para("z0")
    vx0 = exp_config.para("vx0")
    vy0 = exp_config.para("vy0")
    vz0 = exp_config.para("vz0")
    ex = exp_config.para("Ex")
    ey = exp_config.para("Ey")
    ez = exp_config.para("Ez")
    charge = exp_config.get_obj_para("o1", "e")
    mass = exp_config.get_obj_para("o1", "m")

    t = time_grid(t_end, t_num)
    ax = charge * ex / mass
    ay = charge * ey / mass
    az = charge * ez / mass

    posx = x0 + vx0 * t + 0.5 * ax * t ** 2
    posy = y0 + vy0 * t + 0.5 * ay * t ** 2
    posz = z0 + vz0 * t + 0.5 * az * t ** 2
    return insert_motion_data(
        exp_config,
        t,
        error,
        posx,
        posy,
        posz,
        electric_field=np.array([ex, ey, ez], dtype=float),
    )


def charged_particle_electric_x_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electric_x",
        1,
        make_exp_para(
            y0=fixed_parastructure(0.0),
            z0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.4, 1.0),
            Ey=fixed_parastructure(0.0),
            Ez=fixed_parastructure(0.0),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("zero", "posy[o1]"), ("zero", "posz[o1]")],
    )


def charged_particle_electric_z_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electric_z",
        1,
        make_exp_para(
            x0=fixed_parastructure(0.0),
            y0=fixed_parastructure(0.0),
            vx0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            Ex=fixed_parastructure(0.0),
            Ey=fixed_parastructure(0.0),
            Ez=ranged_parastructure(0.4, 1.0),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("zero", "posx[o1]"), ("zero", "posy[o1]")],
    )


def charged_particle_electric_xy_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electric_xy",
        2,
        make_exp_para(
            z0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.3, 0.8),
            Ey=ranged_parastructure(-0.9, -0.3),
            Ez=fixed_parastructure(0.0),
        ),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1]")])


def charged_particle_electric_xz_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electric_xz",
        2,
        make_exp_para(
            y0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.3, 0.8),
            Ey=fixed_parastructure(0.0),
            Ez=ranged_parastructure(-0.9, -0.3),
        ),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posy[o1]")])


def charged_particle_electric_3d_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electric_3d",
        3,
        make_exp_para(
            Ex=ranged_parastructure(0.3, 0.9),
            Ey=ranged_parastructure(-0.8, -0.2),
            Ez=ranged_parastructure(0.2, 0.7),
        ),
    )
    return finalize_structure(expconfig, __file__)
