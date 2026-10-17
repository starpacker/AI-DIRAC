"""Damped pendulum experiment — Reviewer Section 6, pendulum_1 Variant D.

Pendulum with viscous damping: theta'' + gamma*theta' + (g/l)*sin(theta) = 0.
Tests: geometric constraints still hold; mechanical energy is NOT conserved;
system can discover or fit damping-related relations.
"""
import numpy as np
from scipy.integrate import odeint
from aiphy.experiment import default_parastructure, ExpConfig, Objstructure, DoExpType, ExpStructure, Proposition
from aiphy.experiment import concept_posx, concept_posy, concept_posz, concept_t

pi = 3.1415926

exp_para = {
    'l': default_parastructure(2, 5),
    'theta0': default_parastructure(-pi/6, pi/6),
    'omega0': default_parastructure(-pi/4, pi/4),
    'gamma': default_parastructure(0.05, 0.3),
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
    omega0 = exp_config.para('omega0')
    theta0 = exp_config.para('theta0')
    L = exp_config.para('l')
    gamma = exp_config.para('gamma')
    step = t_end / t_num; t = np.arange(0, t_end, step)

    def f(y, t):
        theta, omega = y
        return [omega, -gamma*omega - (g/L)*np.sin(theta)]

    sol = odeint(f, [theta0, omega0], t)
    theta = sol[:, 0]
    posx = L * np.sin(theta)
    posz = -L * np.cos(theta)
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ['o1']), posx + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ['o1']), np.random.normal(0, error, t_num))
    ds.insert_data((concept_posz, ['o1']), posz + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ['clock']), t + np.random.normal(0, error, t_num))
    return ds

def damped_pendulum_config():
    expconfig = ExpConfig("damped_pendulum", 2, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp("posy[o1]")))
    return ExpStructure(expconfig, doexp)
