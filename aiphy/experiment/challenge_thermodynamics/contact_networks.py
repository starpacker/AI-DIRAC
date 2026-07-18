from aiphy.experiment import ExpStructure
from aiphy.experiment.challenge_thermodynamics.common import (
    finalize_structure,
    fixed_parastructure,
    insert_temperature_data,
    integrate_thermal_system,
    make_thermal_expconfig,
    ranged_parastructure,
    time_grid,
)
from aiphy.operational_pipeline import (
    AdditiveOperationalExperimentSpec,
    register_additive_operational_experiment,
)


BASE_EXP_PARA = {
    "T10": ranged_parastructure(0.0, 120.0),
    "T20": ranged_parastructure(0.0, 120.0),
    "T30": ranged_parastructure(0.0, 120.0),
    "k12": ranged_parastructure(0.5, 2.0),
    "k23": ranged_parastructure(0.4, 1.8),
    "k31": ranged_parastructure(0.4, 1.8),
    "k1e": ranged_parastructure(0.2, 0.8),
    "k2e": ranged_parastructure(0.2, 0.8),
    "Tenv": ranged_parastructure(0.0, 80.0),
}


def make_exp_para(**overrides):
    exp_para = dict(BASE_EXP_PARA)
    exp_para.update(overrides)
    return exp_para


def do_experiment(t_end: float, t_num: int, error: float, exp_config):
    mode = int(round(exp_config.para("mode")))
    t = time_grid(t_end, t_num)

    if mode == 1:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        k12 = exp_config.para("k12")

        def rhs(y, _time_value):
            T1, T2 = y
            flow = k12 * (T2 - T1)
            return [flow / c1, -flow / c2]

        temps = integrate_thermal_system(t, [T10, T20], rhs)
    elif mode == 2:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        k12 = exp_config.para("k12")

        def rhs(y, _time_value):
            T1, T2 = y
            flow = k12 * (T2 - T1)
            return [flow / c1, -flow / c2]

        temps = integrate_thermal_system(t, [T10, T20], rhs)
    elif mode == 3:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        k12 = exp_config.para("k12")
        k1e = exp_config.para("k1e")
        k2e = exp_config.para("k2e")
        Tenv = exp_config.para("Tenv")

        def rhs(y, _time_value):
            T1, T2 = y
            flow = k12 * (T2 - T1)
            return [
                (flow + k1e * (Tenv - T1)) / c1,
                (-flow + k2e * (Tenv - T2)) / c2,
            ]

        temps = integrate_thermal_system(t, [T10, T20], rhs)
    elif mode == 4:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        c3 = exp_config.get_obj_para("o3", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        T30 = exp_config.para("T30")
        k12 = exp_config.para("k12")
        k23 = exp_config.para("k23")

        def rhs(y, _time_value):
            T1, T2, T3 = y
            flow12 = k12 * (T2 - T1)
            flow23 = k23 * (T3 - T2)
            return [
                flow12 / c1,
                (-flow12 + flow23) / c2,
                -flow23 / c3,
            ]

        temps = integrate_thermal_system(t, [T10, T20, T30], rhs)
    elif mode == 5:
        c1 = exp_config.get_obj_para("o1", "m")
        c2 = exp_config.get_obj_para("o2", "m")
        c3 = exp_config.get_obj_para("o3", "m")
        T10 = exp_config.para("T10")
        T20 = exp_config.para("T20")
        T30 = exp_config.para("T30")
        k12 = exp_config.para("k12")
        k23 = exp_config.para("k23")
        k31 = exp_config.para("k31")

        def rhs(y, _time_value):
            T1, T2, T3 = y
            flow12 = k12 * (T2 - T1)
            flow23 = k23 * (T3 - T2)
            flow31 = k31 * (T1 - T3)
            return [
                (flow12 - flow31) / c1,
                (-flow12 + flow23) / c2,
                (-flow23 + flow31) / c3,
            ]

        temps = integrate_thermal_system(t, [T10, T20, T30], rhs)
    else:
        raise ValueError(f"Unsupported thermal contact mode: {mode}")

    return insert_temperature_data(exp_config, t, error, temps)


def two_body_contact_equal_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "two_body_contact_equal",
        make_exp_para(
            mode=fixed_parastructure(1.0),
            T10=ranged_parastructure(80.0, 120.0),
            T20=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


def two_body_contact_unequal_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "two_body_contact_unequal",
        make_exp_para(
            mode=fixed_parastructure(2.0),
            T10=ranged_parastructure(80.0, 120.0),
            T20=ranged_parastructure(0.0, 20.0),
        ),
        [(2.0, 2.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


def two_body_contact_with_ambient_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "two_body_contact_with_ambient",
        make_exp_para(
            mode=fixed_parastructure(3.0),
            T10=ranged_parastructure(60.0, 110.0),
            T20=ranged_parastructure(10.0, 40.0),
            Tenv=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 2.0), (1.0, 2.0)],
    )
    return finalize_structure(expconfig, __file__)


def three_body_chain_relaxation_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "three_body_chain_relaxation",
        make_exp_para(
            mode=fixed_parastructure(4.0),
            T10=ranged_parastructure(80.0, 120.0),
            T20=ranged_parastructure(20.0, 40.0),
            T30=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


def three_body_ring_relaxation_config() -> ExpStructure:
    expconfig = make_thermal_expconfig(
        "three_body_ring_relaxation",
        make_exp_para(
            mode=fixed_parastructure(5.0),
            T10=ranged_parastructure(80.0, 120.0),
            T20=ranged_parastructure(20.0, 40.0),
            T30=ranged_parastructure(0.0, 20.0),
        ),
        [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
    )
    return finalize_structure(expconfig, __file__)


# Apparatus affordance only: body 2 is a concrete reference and these controls
# may be varied independently.  No heat-capacity name, value, or target law is
# provided to the operational miner.
register_additive_operational_experiment(
    AdditiveOperationalExperimentSpec(
        experiment_name="two_body_contact_unequal",
        observable="temp",
        reference_object_id=2,
        entity_type="Particle",
        nuisance_controls=("T10", "T20", "k12"),
        alternate_experiment_families=("three_body_chain_relaxation",),
    )
)
