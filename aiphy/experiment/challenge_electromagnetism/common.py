import numpy as np
from scipy.integrate import odeint

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


concept_efieldx = Concept("(1->Particle) |- efieldx[1]")
concept_efieldy = Concept("(1->Particle) |- efieldy[1]")
concept_efieldz = Concept("(1->Particle) |- efieldz[1]")
concept_bfieldx = Concept("(1->Particle) |- bfieldx[1]")
concept_bfieldy = Concept("(1->Particle) |- bfieldy[1]")
concept_bfieldz = Concept("(1->Particle) |- bfieldz[1]")


DATA_INFO = [
    (concept_posx, ["o1"]),
    (concept_posy, ["o1"]),
    (concept_posz, ["o1"]),
    (concept_efieldx, ["o1"]),
    (concept_efieldy, ["o1"]),
    (concept_efieldz, ["o1"]),
    (concept_bfieldx, ["o1"]),
    (concept_bfieldy, ["o1"]),
    (concept_bfieldz, ["o1"]),
    (concept_t, ["clock"]),
]


def ranged_parastructure(low: float, high: float):
    return default_parastructure(low, high)


def fixed_parastructure(value: float):
    return default_parastructure(value, value)


def make_single_particle_obj_info(
    mass_range=(1.0, 2.0),
    charge_range=(0.8, 1.6),
):
    return {
        "o1": Objstructure.make_elec_particle(
            mass_range[0],
            mass_range[1],
            charge_range[0],
            charge_range[1],
        ),
        "clock": Objstructure.clock(),
    }


def make_single_particle_expconfig(
    exp_name: str,
    dim: int,
    exp_para: dict,
    mass_range=(1.0, 2.0),
    charge_range=(0.8, 1.6),
):
    return ExpConfig(
        exp_name,
        dim,
        exp_para,
        make_single_particle_obj_info(mass_range, charge_range),
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


def insert_motion_data(
    exp_config: ExpConfig,
    t,
    error: float,
    posx,
    posy,
    posz,
    electric_field=None,
    magnetic_field=None,
):
    def field_components(field):
        if field is None:
            return [np.zeros(len(t)) for _ in range(3)]
        array = np.asarray(field, dtype=float)
        if array.shape == (3,):
            array = np.repeat(array[None, :], len(t), axis=0)
        if array.shape != (len(t), 3):
            raise ValueError(f"Field data must have shape (3,) or ({len(t)}, 3), got {array.shape}")
        return [array[:, index] for index in range(3)]

    efield = field_components(electric_field)
    bfield = field_components(magnetic_field)
    data_struct = exp_config.new_datastruct_of_doexperiment(len(t))
    data_struct.insert_data((concept_posx, ["o1"]), posx + np.random.normal(0, error, len(t)))
    data_struct.insert_data((concept_posy, ["o1"]), posy + np.random.normal(0, error, len(t)))
    data_struct.insert_data((concept_posz, ["o1"]), posz + np.random.normal(0, error, len(t)))
    for concept, values in zip(
        (concept_efieldx, concept_efieldy, concept_efieldz), efield
    ):
        data_struct.insert_data((concept, ["o1"]), values + np.random.normal(0, error, len(t)))
    for concept, values in zip(
        (concept_bfieldx, concept_bfieldy, concept_bfieldz), bfield
    ):
        data_struct.insert_data((concept, ["o1"]), values + np.random.normal(0, error, len(t)))
    data_struct.insert_data((concept_t, ["clock"]), t + np.random.normal(0, error, len(t)))
    return data_struct


def orthonormal_basis_from_axis(axis):
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        raise ValueError("Axis must be nonzero.")
    bhat = axis / axis_norm
    ref = np.array([1.0, 0.0, 0.0]) if abs(bhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, bhat) * bhat
    e1 /= np.linalg.norm(e1)
    e2 = -np.cross(bhat, e1)
    e2 /= np.linalg.norm(e2)
    return bhat, e1, e2


def circular_state_from_magnetic_field(q, m, bvec, radius, v_parallel=0.0):
    bvec = np.asarray(bvec, dtype=float)
    bmag = np.linalg.norm(bvec)
    if bmag < 1e-8:
        raise ValueError("Magnetic field magnitude must be nonzero for circular initial conditions.")
    bhat, e1, e2 = orthonormal_basis_from_axis(bvec)
    omega = q * bmag / m
    position = radius * e1
    velocity = omega * radius * e2 + v_parallel * bhat
    return np.concatenate([position, velocity])


def drift_velocity(evec, bvec):
    evec = np.asarray(evec, dtype=float)
    bvec = np.asarray(bvec, dtype=float)
    bnorm_sq = float(np.dot(bvec, bvec))
    if bnorm_sq < 1e-8:
        raise ValueError("Magnetic field magnitude must be nonzero for drift initial conditions.")
    return np.cross(evec, bvec) / bnorm_sq


def lorentz_rhs(q, m, electric_field_fn, magnetic_field_fn, drag_coeff_fn=None):
    def rhs(state, time_value):
        pos = np.asarray(state[:3], dtype=float)
        vel = np.asarray(state[3:], dtype=float)
        evec = np.asarray(electric_field_fn(time_value, pos, vel), dtype=float)
        bvec = np.asarray(magnetic_field_fn(time_value, pos, vel), dtype=float)
        gamma = float(drag_coeff_fn(time_value, pos, vel)) if drag_coeff_fn else 0.0
        accel = q * (evec + np.cross(vel, bvec)) / m - gamma * vel
        return [vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]]

    return rhs


def integrate_lorentz(
    t,
    state0,
    q,
    m,
    electric_field_fn,
    magnetic_field_fn,
    drag_coeff_fn=None,
):
    sol = odeint(
        lorentz_rhs(q, m, electric_field_fn, magnetic_field_fn, drag_coeff_fn),
        state0,
        t,
    )
    return sol[:, 0], sol[:, 1], sol[:, 2]
