"""Fixed linear/nonlinear spring regimes for a real Duffing-law revision."""

import numpy as np

from aiphy.experiment import (
    Concept,
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    default_parastructure,
)


concept_position = Concept("(1->Particle) |- revisionposition[1]")
concept_acceleration = Concept("(1->Particle) |- revisionacceleration[1]")
concept_mass = Concept("(1->Particle) |- revisionmass[1]")
concept_stiffness = Concept("(1->Particle) |- revisionstiffness[1]")
concept_cubic = Concept("(1->Particle) |- revisioncubic[1]")

_base_parameters = {
    "mass": default_parastructure(0.7, 1.6),
    "stiffness": default_parastructure(0.8, 2.0),
    "amplitude": default_parastructure(0.7, 1.4),
    "phase": default_parastructure(-0.3, 0.3),
}
_objects = {
    "clock": Objstructure.clock(),
    "body": Objstructure.make_particle(1.0, 2.0),
}
_data = [
    (concept_position, ["body"]),
    (concept_acceleration, ["body"]),
    (concept_mass, ["body"]),
    (concept_stiffness, ["body"]),
    (concept_cubic, ["body"]),
]


def do_experiment(t_end: float, t_num: int, error: float,
                  exp_config: ExpConfig):
    del error
    mass = exp_config.para("mass")
    stiffness = exp_config.para("stiffness")
    cubic = exp_config.para("cubic")
    amplitude = exp_config.para("amplitude")
    phase = exp_config.para("phase")
    time = np.linspace(0.0, t_end, t_num, endpoint=False)
    position = amplitude * np.sin(0.91 * time + phase)
    acceleration = -(stiffness * position + cubic * position**3) / mass
    data = exp_config.new_datastruct_of_doexperiment(t_num)
    data.insert_data((concept_position, ["body"]), position)
    data.insert_data((concept_acceleration, ["body"]), acceleration)
    data.insert_data((concept_mass, ["body"]), np.full(t_num, mass))
    data.insert_data(
        (concept_stiffness, ["body"]), np.full(t_num, stiffness)
    )
    data.insert_data((concept_cubic, ["body"]), np.full(t_num, cubic))
    return data


def _config(name: str, cubic_range: tuple[float, float]) -> ExpStructure:
    parameters = dict(_base_parameters)
    parameters["cubic"] = default_parastructure(*cubic_range)
    return ExpStructure(
        ExpConfig(name, 1, parameters, _objects, _data),
        DoExpType(__file__),
    )


def revision_duffing_linear_config() -> ExpStructure:
    return _config("revision_duffing_linear", (1e-12, 2e-12))


def revision_duffing_critical_config() -> ExpStructure:
    return _config("revision_duffing_critical", (0.25, 0.75))

