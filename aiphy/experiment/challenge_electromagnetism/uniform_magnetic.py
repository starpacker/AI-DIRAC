import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_electromagnetism.common import (
    circular_state_from_magnetic_field,
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
    "x0": ranged_parastructure(-1.0, 1.0),
    "y0": ranged_parastructure(-1.0, 1.0),
    "z0": ranged_parastructure(-1.0, 1.0),
    "vx0": ranged_parastructure(-1.2, 1.2),
    "vy0": ranged_parastructure(-1.2, 1.2),
    "vz0": ranged_parastructure(-1.2, 1.2),
    "radius": ranged_parastructure(0.5, 1.2),
    "v_parallel": ranged_parastructure(-0.8, 0.8),
    "Bx": ranged_parastructure(-1.0, 1.0),
    "By": ranged_parastructure(-1.0, 1.0),
    "Bz": ranged_parastructure(-1.0, 1.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    q = exp_config.get_obj_para("o1", "e")
    m = exp_config.get_obj_para("o1", "m")
    bx = exp_config.para("Bx")
    by = exp_config.para("By")
    bz = exp_config.para("Bz")
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
        state0 = circular_state_from_magnetic_field(
            q,
            m,
            bvec,
            exp_config.para("radius"),
            0.0,
        )
    elif ic_mode == 2:
        state0 = circular_state_from_magnetic_field(
            q,
            m,
            bvec,
            exp_config.para("radius"),
            exp_config.para("v_parallel"),
        )
    else:
        raise ValueError(f"Unsupported magnetic initial-condition mode: {ic_mode}")

    t = time_grid(t_end, t_num)
    posx, posy, posz = integrate_lorentz(
        t,
        state0,
        q,
        m,
        lambda _time_value, _pos, _vel: np.zeros(3),
        lambda _time_value, _pos, _vel: bvec,
    )
    return insert_motion_data(
        exp_config, t, error, posx, posy, posz, magnetic_field=bvec
    )


def charged_particle_magnetic_xy_circle_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_magnetic_xy_circle",
        2,
        make_exp_para(
            ic_mode=fixed_parastructure(1.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.8, 1.5),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [
            ("zero", "posz[o1]"),
            ("conserved", "posx[o1] ** 2 + posy[o1] ** 2"),
        ],
    )


def charged_particle_magnetic_xy_orbit_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_magnetic_xy_orbit",
        2,
        make_exp_para(
            z0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.8, 1.5),
        ),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1]")])


def charged_particle_magnetic_helix_z_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_magnetic_helix_z",
        3,
        make_exp_para(
            ic_mode=fixed_parastructure(2.0),
            Bx=fixed_parastructure(0.0),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.8, 1.5),
            v_parallel=ranged_parastructure(0.3, 1.0),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("conserved", "posx[o1] ** 2 + posy[o1] ** 2")],
    )


def charged_particle_magnetic_yz_circle_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_magnetic_yz_circle",
        2,
        make_exp_para(
            ic_mode=fixed_parastructure(1.0),
            Bx=ranged_parastructure(0.8, 1.5),
            By=fixed_parastructure(0.0),
            Bz=fixed_parastructure(0.0),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [
            ("zero", "posx[o1]"),
            ("conserved", "posy[o1] ** 2 + posz[o1] ** 2"),
        ],
    )


def charged_particle_magnetic_tilted_helix_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_magnetic_tilted_helix",
        3,
        make_exp_para(
            ic_mode=fixed_parastructure(2.0),
            Bx=ranged_parastructure(0.4, 0.9),
            By=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.8, 1.4),
            v_parallel=ranged_parastructure(0.3, 1.0),
        ),
    )
    return finalize_structure(expconfig, __file__)
