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
    "h": ranged_parastructure(0.8, 1.2),
    "phi": ranged_parastructure(1.5, 2.5),
    "Imax": ranged_parastructure(1.0, 2.0),
    "alpha": ranged_parastructure(1.0, 2.0),
    "tau0": ranged_parastructure(0.6, 1.2),
    "lam": ranged_parastructure(0.4, 0.8),
    "L": ranged_parastructure(5.0, 8.0),
    "d": ranged_parastructure(1.5, 3.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    if mode == 1:
        nu = linear_scan(0.6, 3.5, t_num)
        h = exp_config.para("h")
        phi = exp_config.para("phi")
        kinetic = np.maximum(h * nu - phi, 0.0)
        counts = np.where(kinetic > 0.0, 1.0, 0.0)
        posx, posy, posz = nu, kinetic, counts
    elif mode == 2:
        intensity = linear_scan(0.0, exp_config.para("Imax"), t_num)
        alpha = exp_config.para("alpha")
        kinetic = np.full(t_num, 1.0)
        counts = alpha * intensity
        posx, posy, posz = intensity, counts, kinetic
    elif mode == 3:
        tau = linear_scan(-3.0, 3.0, t_num)
        tau0 = exp_config.para("tau0")
        g2 = 1.0 - np.exp(-(tau / tau0) ** 2)
        visibility = 1.0 - g2
        posx, posy, posz = tau, g2, visibility
    elif mode == 4:
        x = linear_scan(-4.0, 4.0, t_num)
        lam = exp_config.para("lam")
        L = exp_config.para("L")
        d = exp_config.para("d")
        phase = 2.0 * np.pi * d * x / (lam * L)
        count_expectation = 0.5 * (1.0 + np.cos(phase))
        posx, posy, posz = x, count_expectation, phase
    elif mode == 5:
        D = linear_scan(0.0, 0.999, t_num)
        V = np.sqrt(np.maximum(0.0, 1.0 - D ** 2))
        center_signal = 1.0 + V
        posx, posy, posz = D, V, center_signal
    else:
        raise ValueError(f"Unsupported wave-particle mode: {mode}")

    return insert_optics_data(exp_config, t, error, posx, posy, posz)


def photoelectric_threshold_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "photoelectric_threshold",
        3,
        make_exp_para(mode=fixed_parastructure(1.0)),
    )
    return finalize_structure(expconfig, __file__)


def photoelectric_intensity_scan_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "photoelectric_intensity_scan",
        3,
        make_exp_para(mode=fixed_parastructure(2.0)),
    )
    return finalize_structure(expconfig, __file__, [("conserved", "posz[o1]")])


def photon_antibunching_g2_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "photon_antibunching_g2",
        3,
        make_exp_para(mode=fixed_parastructure(3.0)),
    )
    return finalize_structure(expconfig, __file__)


def single_photon_double_slit_counts_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "single_photon_double_slit_counts",
        3,
        make_exp_para(mode=fixed_parastructure(4.0)),
    )
    return finalize_structure(expconfig, __file__)


def complementarity_tradeoff_config() -> ExpStructure:
    expconfig = make_single_particle_expconfig(
        "complementarity_tradeoff",
        3,
        make_exp_para(mode=fixed_parastructure(5.0)),
    )
    return finalize_structure(expconfig, __file__)
