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
    "A": ranged_parastructure(0.35, 0.7),
    "n": ranged_parastructure(1.3, 1.8),
    "g": ranged_parastructure(0.6, 1.4),
    "x0": ranged_parastructure(0.6, 1.2),
    "v0": ranged_parastructure(-0.4, 0.4),
    "f1": ranged_parastructure(3.0, 6.0),
    "f2": ranged_parastructure(1.2, 2.4),
    "f0": ranged_parastructure(3.0, 5.0),
    "alpha": ranged_parastructure(0.08, 0.2),
    "beta": ranged_parastructure(0.05, 0.15),
    "d": ranged_parastructure(0.3, 0.8),
    "Cmin": ranged_parastructure(0.08, 0.15),
    "Cmax": ranged_parastructure(0.22, 0.35),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    if mode == 1:
        A = exp_config.para("A")
        n = exp_config.para("n")
        inc = linear_scan(A / 2.0 + 0.05, 0.95, t_num)
        r1 = np.arcsin(np.clip(np.sin(inc) / n, -0.999999, 0.999999))
        r2 = A - r1
        out = np.arcsin(np.clip(n * np.sin(r2), -0.999999, 0.999999))
        dev = inc + out - A
        posx, posy, posz = inc, dev, out
    elif mode == 2:
        z = t
        g = exp_config.para("g")
        x0 = exp_config.para("x0")
        v0 = exp_config.para("v0")
        x = x0 * np.cos(g * z) + (v0 / g) * np.sin(g * z)
        vx = -g * x0 * np.sin(g * z) + v0 * np.cos(g * z)
        invariant = x ** 2 + (vx / g) ** 2
        posx, posy, posz = z, x, invariant
    elif mode == 3:
        h = linear_scan(-1.2, 1.2, t_num)
        f0 = exp_config.para("f0")
        alpha = exp_config.para("alpha")
        beta = exp_config.para("beta")
        focus_shift = f0 + alpha * h ** 2
        spot = beta * np.abs(h) ** 3
        posx, posy, posz = h, focus_shift, spot
    elif mode == 4:
        inc = linear_scan(-0.25, 0.25, t_num)
        f1 = exp_config.para("f1")
        f2 = exp_config.para("f2")
        mag = -f1 / f2
        out = mag * inc
        exit_shift = (f1 + f2) * inc
        posx, posy, posz = inc, out, exit_shift
    elif mode == 5:
        C = linear_scan(exp_config.para("Cmin"), exp_config.para("Cmax"), t_num)
        n = exp_config.para("n")
        d = exp_config.para("d")
        power = (n - 1.0) * (2.0 * C - (d * (n - 1.0) / n) * C ** 2)
        f_eff = 1.0 / np.maximum(power, 1e-6)
        posx, posy, posz = C, f_eff, power
    else:
        raise ValueError(f"Unsupported advanced ray mode: {mode}")

    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def prism_deviation_scan_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "prism_deviation_scan",
        3,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__)


def grin_ray_oscillation_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "grin_ray_oscillation",
        3,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__, [("conserved", "posz[o1]")])


def spherical_aberration_focus_shift_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "spherical_aberration_focus_shift",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__)


def two_lens_telescope_scan_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "two_lens_telescope_scan",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__)


def thick_lens_lensmaker_scan_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "thick_lens_lensmaker_scan",
        3,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__)
