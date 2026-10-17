"""
2D central force motion experiment — STRICT EDITION.

A single particle moves in 2D (posx, posy) under an inverse-square
central force field. Both ax and ay are non-zero and time-varying.

Observables: posx, posy, t
Hidden: central mass parameter GM
Physics: a = -GM * r / |r|^3  (Kepler-like central force)
Key discovery targets:
  - Angular momentum: x*vy - y*vx = const
  - Energy: 0.5*(vx^2+vy^2) - GM/r = const
  - Both ax and ay are non-zero → falsifies R_03, R_04
"""
import numpy as np
from aiphy.experiment import (
    DoExpType, ExpConfig, ExpStructure, Objstructure, Proposition,
    concept_posx, concept_posy, concept_t,
    default_parastructure,
)

exp_para = {
    "GM": default_parastructure(8.0, 15.0),
    "r0": default_parastructure(1.5, 3.0),
    "v_perp0": default_parastructure(1.5, 4.0),
}

obj_info = {
    "o1": Objstructure.make_particle(0.5, 1.5),
    "clock": Objstructure.clock(),
}

data_info = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_t, ["clock"]),
]

def do_experiment(t_end, t_num, error, exp_config):
    GM = exp_config.para("GM")
    r0 = exp_config.para("r0")
    v_perp0 = exp_config.para("v_perp0")
    step = t_end / t_num
    t = np.arange(0, t_end, step)
    posx = np.zeros(t_num); posy = np.zeros(t_num)
    # Initial conditions: start at (r0, 0) with perpendicular velocity
    posx[0] = r0; posy[0] = 0.0
    vx = 0.0; vy = v_perp0
    for i in range(1, t_num):
        r = np.sqrt(posx[i-1]**2 + posy[i-1]**2)
        r = max(r, 0.1)
        ax_val = -GM * posx[i-1] / r**3
        ay_val = -GM * posy[i-1] / r**3
        vx += ax_val * step
        vy += ay_val * step
        posx[i] = posx[i-1] + vx * step
        posy[i] = posy[i-1] + vy * step
    ds = exp_config.new_datastruct_of_doexperiment(t_num)
    ds.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, t_num))
    ds.insert_data((concept_posy, ["o1"]), posy + np.random.normal(0, error, t_num))
    ds.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, t_num))
    return ds

def central_force_2d_config():
    expconfig = ExpConfig("central_force_2d", 1, exp_para, obj_info, data_info)
    doexp = DoExpType(__file__)
    return ExpStructure(expconfig, doexp)
