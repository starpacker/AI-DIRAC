import numpy as np

from aiphy.experiment import (
    Concept,
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    Proposition,
    concept_posx,
    concept_posy,
    concept_posz,
    concept_t,
    default_parastructure,
)


# Phase 0 provenance fix: intensity, phase, and scan were SIMULATOR_HIDDEN channels
# directly exposing the answer structure.  posy now carries the legitimate detector
# reading (intensity IS what a photodetector measures — RAW_SENSOR).
# posz is now background noise, not the optical phase.
# Redundant channels (scan, intensity, phase) are removed entirely.
# The internal simulator state (phase, visibility) is NOT exposed to discovery.

DATA_INFO = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_t, ["clock"]),
]


def ranged_parastructure(low: float, high: float):
    return default_parastructure(low, high)


def fixed_parastructure(value: float):
    return default_parastructure(value, value)


def make_single_particle_expconfig(exp_name: str, dim: int, exp_para: dict):
    return ExpConfig(
        exp_name,
        dim,
        exp_para,
        {
            "o1": Objstructure.make_particle(1.0, 1.0),
            "clock": Objstructure.clock(),
        },
        DATA_INFO,
    )


def register_geometry_specs(expconfig: ExpConfig, geometry_specs):
    for geom_type, expr in geometry_specs:
        if geom_type == "zero":
            expconfig.register_geometry_info(Proposition.IsZero(expconfig.gen_exp(expr)))
        elif geom_type == "conserved":
            expconfig.register_geometry_info(
                Proposition.IsConserved(expconfig.gen_exp(expr))
            )
        else:
            raise ValueError(f"Unsupported geometry spec: {geom_type}")


def finalize_structure(expconfig: ExpConfig, file_path: str, geometry_specs=None):
    register_geometry_specs(expconfig, geometry_specs or [])
    return ExpStructure(expconfig, DoExpType(file_path))


def time_grid(t_end: float, t_num: int):
    return np.linspace(0.0, t_end, t_num, endpoint=False)


def linear_scan(low: float, high: float, n: int):
    return np.linspace(low, high, n, endpoint=False)


def insert_optics_data(exp_config: ExpConfig, t, error: float, posx, posy, posz):
    """Insert optical data with posx=detector_position, posy=detector_reading (intensity).

    posy = intensity IS a legitimate RAW_SENSOR — a photodetector output.
    posz = 0 (background noise only) — no longer leaks phase/visibility.
    """
    data_struct = exp_config.new_datastruct_of_doexperiment(len(t))
    data_struct.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, len(t)))
    data_struct.insert_data((concept_posy, ["o1"]), posy + np.random.normal(0, error, len(t)))
    # posz is now JUST background noise — not phase, not visibility, not intensity.
    data_struct.insert_data((concept_posz, ["o1"]), np.random.normal(0, error, len(t)))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, len(t)))
    return data_struct


def safe_sinc(x):
    return np.sinc(x / np.pi)
