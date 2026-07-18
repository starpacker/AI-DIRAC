import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_electromagnetism.common import (
    finalize_structure,
    fixed_parastructure,
    insert_motion_data,
    integrate_lorentz,
    make_single_particle_expconfig,
    ranged_parastructure,
    time_grid,
)


BASE_EXP_PARA = {
    "x0": ranged_parastructure(-0.8, 0.8),
    "y0": ranged_parastructure(-0.8, 0.8),
    "z0": ranged_parastructure(-0.8, 0.8),
    "vx0": ranged_parastructure(-0.8, 0.8),
    "vy0": ranged_parastructure(-0.8, 0.8),
    "vz0": ranged_parastructure(-0.8, 0.8),
    "Ex0": fixed_parastructure(0.0),
    "Ey0": fixed_parastructure(0.0),
    "Ez0": fixed_parastructure(0.0),
    "Bx": fixed_parastructure(0.0),
    "By": fixed_parastructure(0.0),
    "Bz": fixed_parastructure(0.0),
    "gamma_drag": fixed_parastructure(0.0),
    "kx_static": fixed_parastructure(0.0),
    "ky_static": fixed_parastructure(0.0),
    "kz_static": fixed_parastructure(0.0),
    "kx_rf": fixed_parastructure(0.0),
    "ky_rf": fixed_parastructure(0.0),
    "kz_rf": fixed_parastructure(0.0),
    "omega_rf": fixed_parastructure(1.0),
    "Ex_drive": fixed_parastructure(0.0),
    "Ey_drive": fixed_parastructure(0.0),
    "Ez_drive": fixed_parastructure(0.0),
    "omega_drive": fixed_parastructure(1.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    q = exp_config.get_obj_para("o1", "e")
    m = exp_config.get_obj_para("o1", "m")
    state0 = [
        exp_config.para("x0"),
        exp_config.para("y0"),
        exp_config.para("z0"),
        exp_config.para("vx0"),
        exp_config.para("vy0"),
        exp_config.para("vz0"),
    ]
    ex0 = exp_config.para("Ex0")
    ey0 = exp_config.para("Ey0")
    ez0 = exp_config.para("Ez0")
    bx = exp_config.para("Bx")
    by = exp_config.para("By")
    bz = exp_config.para("Bz")
    gamma_drag = exp_config.para("gamma_drag")
    kx_static = exp_config.para("kx_static")
    ky_static = exp_config.para("ky_static")
    kz_static = exp_config.para("kz_static")
    kx_rf = exp_config.para("kx_rf")
    ky_rf = exp_config.para("ky_rf")
    kz_rf = exp_config.para("kz_rf")
    omega_rf = exp_config.para("omega_rf")
    ex_drive = exp_config.para("Ex_drive")
    ey_drive = exp_config.para("Ey_drive")
    ez_drive = exp_config.para("Ez_drive")
    omega_drive = exp_config.para("omega_drive")

    def electric_field(time_value, pos, _vel):
        x, y, z = pos
        rf = np.cos(omega_rf * time_value)
        drive = np.cos(omega_drive * time_value)
        return np.array(
            [
                ex0 + kx_static * x + kx_rf * x * rf + ex_drive * drive,
                ey0 + ky_static * y + ky_rf * y * rf + ey_drive * drive,
                ez0 + kz_static * z + kz_rf * z * rf + ez_drive * drive,
            ],
            dtype=float,
        )

    bvec = np.array([bx, by, bz], dtype=float)
    t = time_grid(t_end, t_num)
    posx, posy, posz = integrate_lorentz(
        t,
        state0,
        q,
        m,
        electric_field,
        lambda _time_value, _pos, _vel: bvec,
        lambda _time_value, _pos, _vel: gamma_drag,
    )
    positions = np.column_stack([posx, posy, posz])
    efield = np.asarray(
        [electric_field(time_value, position, np.zeros(3)) for time_value, position in zip(t, positions)]
    )
    return insert_motion_data(
        exp_config,
        t,
        error,
        posx,
        posy,
        posz,
        electric_field=efield,
        magnetic_field=bvec,
    )


def charged_particle_damped_cyclotron_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_damped_cyclotron",
        2,
        make_exp_para(
            z0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            x0=ranged_parastructure(0.5, 1.0),
            y0=fixed_parastructure(0.0),
            vx0=fixed_parastructure(0.0),
            vy0=ranged_parastructure(0.8, 1.2),
            Bz=ranged_parastructure(0.9, 1.4),
            gamma_drag=ranged_parastructure(0.08, 0.18),
        ),
        mass_range=(1.0, 1.4),
        charge_range=(0.9, 1.2),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1]")])


def charged_particle_driven_cyclotron_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_driven_cyclotron",
        2,
        make_exp_para(
            z0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            x0=fixed_parastructure(0.0),
            y0=fixed_parastructure(0.0),
            vx0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            Bz=ranged_parastructure(0.9, 1.1),
            gamma_drag=ranged_parastructure(0.03, 0.08),
            Ex_drive=ranged_parastructure(0.3, 0.6),
            omega_drive=ranged_parastructure(0.9, 1.1),
        ),
        mass_range=(1.0, 1.2),
        charge_range=(0.9, 1.1),
    )
    return finalize_structure(expconfig, __file__, [("zero", "posz[o1]")])


def charged_particle_electrostatic_oscillator_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_electrostatic_oscillator",
        1,
        make_exp_para(
            y0=fixed_parastructure(0.0),
            z0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            x0=ranged_parastructure(0.5, 1.2),
            vx0=ranged_parastructure(-0.2, 0.2),
            kx_static=ranged_parastructure(-0.8, -0.3),
        ),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("zero", "posy[o1]"), ("zero", "posz[o1]")],
    )


def charged_particle_paul_trap_1d_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_paul_trap_1d",
        1,
        make_exp_para(
            y0=fixed_parastructure(0.0),
            z0=fixed_parastructure(0.0),
            vy0=fixed_parastructure(0.0),
            vz0=fixed_parastructure(0.0),
            x0=ranged_parastructure(0.2, 0.7),
            vx0=ranged_parastructure(-0.2, 0.2),
            kx_rf=ranged_parastructure(-0.35, -0.15),
            omega_rf=ranged_parastructure(1.6, 2.4),
        ),
        mass_range=(1.0, 1.4),
        charge_range=(0.9, 1.3),
    )
    return finalize_structure(
        expconfig,
        __file__,
        [("zero", "posy[o1]"), ("zero", "posz[o1]")],
    )


def charged_particle_penning_like_trap_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "charged_particle_penning_like_trap",
        3,
        make_exp_para(
            x0=ranged_parastructure(0.2, 0.6),
            y0=ranged_parastructure(0.2, 0.6),
            z0=ranged_parastructure(0.2, 0.6),
            vx0=ranged_parastructure(-0.2, 0.2),
            vy0=ranged_parastructure(-0.2, 0.2),
            vz0=ranged_parastructure(-0.2, 0.2),
            Bz=ranged_parastructure(1.3, 1.8),
            kx_static=ranged_parastructure(0.08, 0.16),
            ky_static=ranged_parastructure(0.08, 0.16),
            kz_static=ranged_parastructure(-0.32, -0.16),
        ),
        mass_range=(1.0, 1.5),
        charge_range=(0.9, 1.3),
    )
    return finalize_structure(expconfig, __file__)
