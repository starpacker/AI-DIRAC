"""Nonlinear spring do_experiment function — separate module to override linear spring.

F = -k*x - alpha*x^3. At low amplitude: linear model works; at high amplitude,
linear model is falsified. The system should distinguish between them.
"""
import numpy as np
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_length, concept_t

exp_para = {
    "zl": default_parastructure(8, 12),
    "zr": default_parastructure(-5, -1),
    "alpha": default_parastructure(0.5, 3.0),
}
obj_info = {
    "o1": Objstructure.make_particle(1, 5),
    "s1": Objstructure.make_spring(1.5, 2.5, 8, 20),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]), (concept_posy, ["o1"]), (concept_posz, ["o1"]),
    (concept_length, ["s1"]), (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    zl = exp_config.para('zl'); zr = exp_config.para('zr')
    L0 = exp_config.get_obj_para('s1', 'freel')
    k = exp_config.get_obj_para('s1', 'thickness') ** 3
    alpha = exp_config.para('alpha')
    m = exp_config.get_obj_para('o1', 'm')
    step = t_end / t_num; t = np.arange(0, t_end, step)
    g = 9.801234567

    # Nonlinear ODE: equilibrium at z_eq where k*z_eq + alpha*z_eq^3 = m*g
    # For perturbative solution, use RK45 directly.
    def f(y, t):
        z, v = y
        # Total extension from free length
        # Force: -k*z - alpha*z^3 + m*g (positive = stretching)
        a = (m*g - k*z - alpha*z**3) / m
        return [v, a]

    # Initial displacement
    z0 = L0 + zr - zl  # initial stretch
    sol = odeint(f, [z0, 0.0], t)
    z = sol[:, 0]
    posz = zl - L0 - z
    length = L0 + z
    zeros = np.zeros(t_num)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ['o1']), zeros + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ['o1']), zeros + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posz, ['o1']), posz + np.random.normal(0, error, t_num))
    ds.insert_data((concept_length, ['s1']), length + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def nonlinear_spring_do_config():
    expconfig = ExpConfig("nonlinear_spring", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posx[o1]")))
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    return ExpStructure(expconfig, doexp)
