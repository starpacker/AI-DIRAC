"""Variable-length pendulum — Reviewer Section 6, pendulum_1 Variant C.

Pendulum with time-varying length l = l(t).
Tests: x^2 + z^2 = l(t)^2 form is recoverable;
fixed constant length relations should NOT survive.
"""
import numpy as np
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t

pi = 3.1415926

exp_para = {
    'l0': default_parastructure(3, 6),
    'v_l': default_parastructure(-0.5, 0.5),
    'theta0': default_parastructure(-pi/8, pi/8),
    'omega0': default_parastructure(-0.5, 0.5),
}
obj_info = {
    "o1": Objstructure.make_particle(1, 5),
    "clock": Objstructure.clock()
}
data_info = [
    (concept_posx, ["o1"]), (concept_posy, ["o1"]),
    (concept_posz, ["o1"]), (concept_t, ["clock"]),
]

g = 9.801234567

def do_experiment(t_end, t_num, error, exp_config):
    theta0 = exp_config.para('theta0')
    omega0 = exp_config.para('omega0')
    l0 = exp_config.para('l0')
    v_l = exp_config.para('v_l')
    step = t_end / t_num; t = np.arange(0, t_end, step)

    def f(y, t):
        theta, omega = y
        Lt = l0 + v_l * t
        return [omega, -(g/Lt)*np.sin(theta) - 2*v_l*omega/Lt]

    sol = odeint(f, [theta0, omega0], t)
    theta = sol[:, 0]
    L = l0 + v_l * t
    posx = L * np.sin(theta)
    posz = -L * np.cos(theta)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ['o1']), posx + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    ds.insert_data((concept_posz, ['o1']), posz + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def variable_pendulum_config():
    expconfig = ExpConfig("variable_pendulum", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    return ExpStructure(expconfig, doexp)
