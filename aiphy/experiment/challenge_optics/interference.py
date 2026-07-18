import numpy as np

from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_optics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_optics_data,
    linear_scan,
    make_single_particle_expconfig,
    ranged_parastructure,
    safe_sinc,
    time_grid,
)


BASE_EXP_PARA = {
    "mode": fixed_parastructure(1.0),
    "xmin": ranged_parastructure(-4.0, -2.0),
    "xmax": ranged_parastructure(2.0, 4.0),
    "L": ranged_parastructure(6.0, 10.0),
    "lam": ranged_parastructure(0.4, 0.8),
    "d": ranged_parastructure(2.0, 4.0),
    "a": ranged_parastructure(0.4, 0.8),
    "F": ranged_parastructure(8.0, 20.0),
    "I1": ranged_parastructure(0.4, 0.9),
    "I2": ranged_parastructure(0.8, 1.4),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)
    x = linear_scan(exp_config.para("xmin"), exp_config.para("xmax"), t_num)
    lam = exp_config.para("lam")
    L = exp_config.para("L")
    d = exp_config.para("d")
    phase = 2.0 * np.pi * d * x / (lam * L)

    if mode == 1:
        intensity = 0.5 * (1.0 + np.cos(phase))
        posx, posy, posz = x, intensity, phase
    elif mode == 2:
        a = exp_config.para("a")
        envelope = safe_sinc(np.pi * a * x / (lam * L)) ** 2
        intensity = envelope * 0.5 * (1.0 + np.cos(phase))
        posx, posy, posz = x, intensity, phase
    elif mode == 3:
        disp = linear_scan(-2.5, 2.5, t_num)
        phase = 4.0 * np.pi * disp / lam
        intensity = 0.5 * (1.0 + np.cos(phase))
        posx, posy, posz = disp, intensity, phase
    elif mode == 4:
        detune = linear_scan(-3.0 * np.pi, 3.0 * np.pi, t_num)
        finesse = exp_config.para("F")
        intensity = 1.0 / (1.0 + finesse * np.sin(detune / 2.0) ** 2)
        posx, posy, posz = detune, intensity, np.arctan2(np.sin(detune), 1.0 + finesse)
    elif mode == 5:
        phase = linear_scan(-3.0 * np.pi, 3.0 * np.pi, t_num)
        I1 = exp_config.para("I1")
        I2 = exp_config.para("I2")
        intensity = I1 + I2 + 2.0 * np.sqrt(I1 * I2) * np.cos(phase)
        visibility = 2.0 * np.sqrt(I1 * I2) / (I1 + I2)
        posx, posy, posz = phase, intensity, np.full(t_num, visibility)
    else:
        raise ValueError(f"Unsupported interference mode: {mode}")

    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def young_double_slit_basic_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "young_double_slit_basic",
        3,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__)


def young_double_slit_envelope_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "young_double_slit_envelope",
        3,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__)


def michelson_fringes_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "michelson_fringes",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__)


def fabry_perot_airy_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "fabry_perot_airy",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__)


def unequal_amplitude_interference_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "unequal_amplitude_interference",
        3,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__, [("conserved", "posz[o1]")])
