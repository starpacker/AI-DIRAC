"""Two fixed pendulum regimes used to exercise theory revision end to end.

The small-angle experiment is deliberately confined to a regime where the
linear law is within numerical tolerance.  The critical experiment covers
large angles and therefore falsifies that approximation.  Both expose only
typed observations; no equation is registered as geometry information.
"""

import numpy as np

from aiphy.experiment import (
    Concept,
    DoExpType,
    ExpConfig,
    ExpStructure,
    Objstructure,
    default_parastructure,
)


concept_angle = Concept("(1->Particle) |- angle[1]")
concept_angular_acceleration = Concept(
    "(1->Particle) |- angularacceleration[1]"
)
concept_gravity = Concept("(1->Particle) |- gravity[1]")
concept_revision_length = Concept("(1->Particle) |- revisionlength[1]")

_parameters = {
    "gravity": default_parastructure(7.5, 11.5),
    "length": default_parastructure(0.8, 2.0),
    "phase": default_parastructure(-0.4, 0.4),
}
# Keep the bob at object id 1 so the same typed observations can be fed
# directly to the generic SR datasets used by the revision controller.
_objects = {
    "clock": Objstructure.clock(),
    "bob": Objstructure.make_particle(1.0, 2.0),
}
_data = [
    (concept_angle, ["bob"]),
    (concept_angular_acceleration, ["bob"]),
    (concept_gravity, ["bob"]),
    (concept_revision_length, ["bob"]),
]


def do_experiment(t_end: float, t_num: int, error: float,
                  exp_config: ExpConfig):
    del error  # Exact simulator data isolates model error from sensor noise.
    gravity = exp_config.para("gravity")
    length = exp_config.para("length")
    phase = exp_config.para("phase")
    time = np.linspace(0.0, t_end, t_num, endpoint=False)
    amplitude = exp_config.para("amplitude")
    angle = amplitude * np.sin(1.17 * time + phase)
    angular_acceleration = -(gravity / length) * np.sin(angle)
    data = exp_config.new_datastruct_of_doexperiment(t_num)
    data.insert_data((concept_angle, ["bob"]), angle)
    data.insert_data(
        (concept_angular_acceleration, ["bob"]), angular_acceleration
    )
    data.insert_data((concept_gravity, ["bob"]), np.full(t_num, gravity))
    data.insert_data(
        (concept_revision_length, ["bob"]), np.full(t_num, length)
    )
    return data


def _config(name: str, amplitude_range: tuple[float, float]) -> ExpStructure:
    parameters = dict(_parameters)
    parameters["amplitude"] = default_parastructure(*amplitude_range)
    config = ExpConfig(name, 1, parameters, _objects, _data)
    return ExpStructure(config, DoExpType(__file__))


def revision_pendulum_small_config() -> ExpStructure:
    return _config("revision_pendulum_small", (8e-5, 1.2e-4))


def revision_pendulum_critical_config() -> ExpStructure:
    return _config("revision_pendulum_critical", (1.15, 1.45))
