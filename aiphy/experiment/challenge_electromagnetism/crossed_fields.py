import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_electromagnetism.common import (
    drift_velocity,
    finalize_structure,
    fixed_parastructure,
    insert_motion_data,
    integrate_lorentz,
    make_single_particle_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "ic_mode": fixed_parastructure(0.0),
    "x0": ranged_parastructure(-0.8, 0.8),
    "y0": ranged_parastructure(-0.8, 0.8),
    "z0": ranged_parastructure(-0.8, 0.8),
    "vx0": ranged_parastructure(-0.8, 0.8),
    "vy0": ranged_parastructure(-0.8, 0.8),
    "vz0": ranged_parastructure(-0.8, 0.8),
    "Ex": ranged_parastructure(-0.8, 0.8),
    "Ey": ranged_parastructure(-0.8, 0.8),
    "Ez": ranged_parastructure(-0.8, 0.8),
    "Bx": ranged_parastructure(-1.0, 1.0),
    "By": ranged_parastructure(-1.0, 1.0),
    "Bz": ranged_parastructure(-1.0, 1.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    ex = exp_config.para("Ex")
    ey = exp_config.para("Ey")
    ez = exp_config.para("Ez")
    bx = exp_config.para("Bx")
    by = exp_config.para("By")
    bz = exp_config.para("Bz")
    evec = np.array([ex, ey, ez], dtype=float)
    bvec = np.array([bx, by, bz], dtype=float)
    ic_mode = int(round(exp_config.para("ic_mode")))

    if ic_mode == 0:
        state0 = [
            exp_config.para("x0"),
            exp_config.para("y0"),
            exp_config.para("z0"),
            exp_config.para("vx0"),
            exp_config.para("vy0"),
            exp_config.para("vz0"),
        ]
    elif ic_mode == 1:
        v_drift = drift_velocity(evec, bvec)
        state0 = [
            exp_config.para("x0"),
            exp_config.para("y0"),
            exp_config.para("z0"),
            v_drift[0],
            v_drift[1],
            v_drift[2],
        ]
    else:
        raise ValueError(f"Unsupported crossed-field initial-condition mode: {ic_mode}")

    t = time_grid(t_end, t_num)
    posx, posy, posz = integrate_lorentz(
        t,
        state0,
        exp_config.get_obj_para("o1", "e"),
        exp_config.get_obj_para("o1", "m"),
        lambda _time_value, _pos, _vel: evec,
        lambda _time_value, _pos, _vel: bvec,
    )
    return insert_motion_data(
        exp_config,
        t,
        error,
        posx,
        posy,
        posz,
        electric_field=evec,
        magnetic_field=bvec,
    )


def charged_particle_crossed_EB_trochoid_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_crossed_EB_trochoid",
        2,
        make_exp_para(
            z0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.4, 0.9),
            Ey=fixed_parastructure(0.0),
            Ez=fixed_parastructure(0.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.9, 1.4),
        ),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1]")])


def charged_particle_crossed_EB_drift_line_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_crossed_EB_drift_line",
        1,
        make_exp_para(
            ic_mode=fixed_parastructure(1.0),
            x0=fixed_parastructure(0.0),
            z0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.4, 0.9),
            Ey=fixed_parastructure(0.0),
            Ez=fixed_parastructure(0.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.9, 1.4),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("zero", "posx[o1]"), ("zero", "posz[o1]")],
    )


def charged_particle_crossed_EB_helix_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_crossed_EB_helix",
        3,
        make_exp_para(
            Ex=ranged_parastructure(0.4, 0.9),
            Ey=fixed_parastructure(0.0),
            Ez=fixed_parastructure(0.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.9, 1.4),
            vz0=ranged_parastructure(0.3, 1.0),
        ),
    )
    return finalize_structure(expconfig, __file__)


def charged_particle_crossed_EB_xz_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_crossed_EB_xz",
        2,
        make_exp_para(
            y0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            Ex=ranged_parastructure(0.4, 0.9),
            Ey=fixed_parastructure(0.0),
            Ez=fixed_parastructure(0.0),
            Bx=fixed_parastructure(0.0),
            By=ranged_parastructure(0.9, 1.4),
            Bz=fixed_parastructure(0.0),
        ),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posy[o1]")])


def charged_particle_crossed_EB_tilted_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_crossed_EB_tilted",
        3,
        make_exp_para(
            Ex=ranged_parastructure(0.3, 0.7),
            Ey=ranged_parastructure(-0.5, -0.2),
            Ez=ranged_parastructure(0.2, 0.6),
            Bx=ranged_parastructure(0.3, 0.8),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.8, 1.2),
        ),
    )
    return finalize_structure(expconfig, __file__)
