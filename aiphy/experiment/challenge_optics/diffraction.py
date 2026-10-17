import numpy as np
from scipy.special import fresnel, j1

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
    "xmin": ranged_parastructure(-5.0, -2.5),
    "xmax": ranged_parastructure(2.5, 5.0),
    "lam": ranged_parastructure(0.4, 0.8),
    "L": ranged_parastructure(6.0, 10.0),
    "a": ranged_parastructure(0.3, 0.8),
    "d": ranged_parastructure(1.5, 3.0),
    "N": ranged_parastructure(4.0, 8.0),
    "D": ranged_parastructure(1.5, 3.0),
    "sigma": ranged_parastructure(1.0, 2.5),
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

    if mode == 1:
        a = exp_config.para("a")
        beta = np.pi * a * x / (lam * L)
        intensity = safe_sinc(beta) ** 2
        posx, posy, posz = x, intensity, beta
    elif mode == 2:
        a = exp_config.para("a")
        d = exp_config.para("d")
        N = int(round(exp_config.para("N")))
        alpha = np.pi * d * x / (lam * L)
        beta = np.pi * a * x / (lam * L)
        denom = np.sin(alpha)
        ratio = np.where(np.abs(denom) < 1e-6, 1.0, np.sin(N * alpha) / (N * denom))
        intensity = (safe_sinc(beta) ** 2) * (ratio ** 2)
        posx, posy, posz = x, intensity, alpha
    elif mode == 3:
        D = exp_config.para("D")
        alpha = np.pi * D * np.abs(x) / (lam * L)
        airy = np.ones_like(alpha)
        mask = np.abs(alpha) > 1e-6
        airy[mask] = (2.0 * j1(alpha[mask]) / alpha[mask]) ** 2
        posx, posy, posz = x, airy, alpha
    elif mode == 4:
        u = linear_scan(-3.0, 3.0, t_num)
        S, C = fresnel(u / np.sqrt(np.pi))
        intensity = 0.5 * ((0.5 + C) ** 2 + (0.5 + S) ** 2)
        posx, posy, posz = u, intensity, u
    elif mode == 5:
        sigma = exp_config.para("sigma")
        a = exp_config.para("a")
        beta = np.pi * a * x / (lam * L)
        visibility = np.exp(-(x / sigma) ** 2)
        intensity = (safe_sinc(beta) ** 2) * (0.5 + 0.5 * visibility * np.cos(2.5 * beta))
        posx, posy, posz = x, intensity, visibility
    else:
        raise ValueError(f"Unsupported diffraction mode: {mode}")

    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def single_slit_fraunhofer_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "single_slit_fraunhofer",
        3,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__)


def diffraction_grating_multiorder_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "diffraction_grating_multiorder",
        3,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__)


def airy_disk_cross_section_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "airy_disk_cross_section",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__)


def fresnel_edge_diffraction_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "fresnel_edge_diffraction",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__)


def partial_coherence_diffraction_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "partial_coherence_diffraction",
        3,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__)
