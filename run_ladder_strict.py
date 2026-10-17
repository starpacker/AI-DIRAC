"""Strict ladder runner with dynamic concept unlocking.

Experiments declare candidate UNLOCKED columns, but those columns are exposed
only when the runner's certified registry already contains the corresponding
concept.  The registry is updated after each experiment, so experiment order
controls open/hidden state.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aiphy import MeasureType, Theorist
from aiphy.experiment.column_policy import ColumnRole
from aiphy.experiment.ladder_strict import MODE, _specs, make_ladder_strict_config
from aiphy.operational_concepts import ConceptStatus


@dataclass(frozen=True)
class LadderExperiment:
    label: str
    config_name: str
    certifies: tuple[str, ...] = ()


@dataclass(frozen=True)
class LadderLevel:
    name: str
    experiments: tuple[LadderExperiment, ...]

    @property
    def certifies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for experiment in self.experiments:
            for concept in experiment.certifies:
                if concept not in seen:
                    seen.append(concept)
        return tuple(seen)


LADDER_LEVELS: tuple[LadderLevel, ...] = (
    LadderLevel("L1", (
        LadderExperiment("1.1", "strict_1_1_collision_pair", ("mass", "velocity")),
        LadderExperiment("1.2", "strict_1_2_collision_reference"),
    )),
    LadderLevel("L2", (
        LadderExperiment("2.1", "strict_2_1_free_fall_time", ("acceleration",)),
        LadderExperiment("2.2", "strict_2_2_incline_slide"),
        LadderExperiment("2.3", "strict_2_3_atwood"),
    )),
    LadderLevel("L3", (
        LadderExperiment("3.1", "strict_3_1_mass_under_force", ("force",)),
        LadderExperiment("3.2", "strict_3_2_hanging_spring", ("kspring", "delta_length")),
    )),
    LadderLevel("L4", (
        LadderExperiment("4.1", "strict_4_1_drop_repeat", ("gravity",)),
        LadderExperiment("4.2", "strict_4_2_orbit_two_body", ("GM", "orbit_radius", "orbit_period")),
    )),
    LadderLevel("L5", (
        LadderExperiment("5.1", "strict_5_1_volume_by_displacement", ("volume",)),
        LadderExperiment("5.2", "strict_5_2_density_emergence", ("density",)),
        LadderExperiment("5.3", "strict_5_3_pressure_discovery", ("pressure",)),
        LadderExperiment("5.4", "strict_5_4_hydrostatic_pressure", ("fluid_density", "depth")),
        LadderExperiment("5.5", "strict_5_5_sphere_terminal_fall", ("viscosity", "terminal_velocity")),
        LadderExperiment("5.6", "strict_5_6_pipe_flow", ("flow_rate", "pressure_delta")),
    )),
    LadderLevel("L6", (
        LadderExperiment("6.1", "strict_6_1_spring_oscillator", ("energy",)),
        LadderExperiment("6.2", "strict_6_2_rolling_shapes", ("inertia_ratio",)),
        LadderExperiment("6.3", "strict_6_3_pendulum_small"),
        LadderExperiment("6.4", "strict_6_4_damped_oscillator", ("damping",)),
    )),
    LadderLevel("L6.5", (
        LadderExperiment("X.1", "strict_x_1_elastic_contact", ("elastic_modulus_star",)),
    )),
    LadderLevel("L7", (
        LadderExperiment("7.1", "strict_7_1_projectile_no_drag"),
        LadderExperiment("7.2", "strict_7_2_projectile_linear_drag", ("linear_drag", "ln")),
        LadderExperiment("7.3", "strict_7_3_vertical_fall_quadratic", ("drag_coefficient", "tanh")),
        LadderExperiment("7.4", "strict_7_4_forced_oscillator", ("phase",)),
    )),
    LadderLevel("L7.5", (
        LadderExperiment("Y.1", "strict_y_1_high_speed_kinematics", ("speed_limit",)),
        LadderExperiment("Y.2", "strict_y_2_relativistic_rocket", ("exhaust_velocity",)),
    )),
    LadderLevel("L5T", (
        LadderExperiment("T.1", "strict_t_1_temperature_sensor", ("temperature",)),
        LadderExperiment("T.2", "strict_t_2_ideal_gas_state", ("amount", "gas_constant")),
        LadderExperiment("T.3", "strict_t_3_adiabatic_gas", ("gamma_gas",)),
        LadderExperiment("T.4", "strict_t_4_heat_capacity_calibration", ("heat", "heat_capacity")),
        LadderExperiment("T.5", "strict_t_5_cooling_and_conduction", ("thermal_diffusivity", "thermal_conductivity")),
        LadderExperiment("T.6", "strict_t_6_van_der_waals_gas", ("vdw_a", "vdw_b")),
        LadderExperiment("T.7", "strict_t_7_solid_heat_capacity", ("debye_temperature",)),
        LadderExperiment("T.8", "strict_t_8_entropy_expansion", ("entropy", "boltzmann_constant")),
    )),
    LadderLevel("L5O", (
        LadderExperiment("O.1", "strict_o_1_light_time_of_flight", ("light_speed",)),
        LadderExperiment("O.2", "strict_o_2_interference_wavelength", ("wavelength",)),
        LadderExperiment("O.3", "strict_o_3_frequency_relation", ("frequency",)),
        LadderExperiment("O.4", "strict_o_4_diffraction", ("sinc",)),
        LadderExperiment("O.5", "strict_o_5_photoelectric_effect", ("planck_over_charge", "work_function")),
        LadderExperiment("O.6", "strict_o_6_blackbody_spectrum", ("exp", "wien_constant", "stefan_boltzmann")),
    )),
)


LEVEL_BY_NAME = {level.name.upper(): level for level in LADDER_LEVELS}


def _smoke_collect(exp) -> set[str]:
    exp.random_settings()
    data = exp.collect_expdata(MeasureType.default())
    data_keys = {str(key) for key in data.datastruct.data_keys}
    original_keys = {str(key) for key in exp.original_data}
    if data_keys != original_keys:
        raise RuntimeError(
            f"{exp.exp_name}: collected columns differ from data_info: "
            f"data={sorted(data_keys)}, data_info={sorted(original_keys)}"
        )
    return data_keys


def _atom_for_spec(exp, spec) -> str:
    ids = [exp.obj_info[obj_name][1] for obj_name in spec.key[1]]
    return str(spec.key[0].subst_by_vec(ids))


def _verify_dynamic_unlocking(exp, certified: set[str]) -> None:
    specs = _specs(MODE[exp.exp_name])
    open_atoms = {str(atom) for atom in exp.original_data}
    for spec in specs:
        if spec.role is not ColumnRole.UNLOCKED:
            continue
        atom = _atom_for_spec(exp, spec)
        should_open = spec.required_unlock in certified
        if should_open and atom not in open_atoms:
            raise RuntimeError(
                f"{exp.exp_name}: `{spec.name}` requires certified `{spec.required_unlock}` "
                "but is not open"
            )
        if not should_open and atom in open_atoms:
            raise RuntimeError(
                f"{exp.exp_name}: `{spec.name}` opened before `{spec.required_unlock}` was certified"
            )


def select_levels(levels: str | None) -> list[LadderLevel]:
    if not levels:
        return list(LADDER_LEVELS)
    requested = [item.strip().upper() for item in levels.split(",") if item.strip()]
    unknown = [item for item in requested if item not in LEVEL_BY_NAME]
    if unknown:
        valid = ", ".join(level.name for level in LADDER_LEVELS)
        raise SystemExit(f"Unknown level(s): {', '.join(unknown)}. Valid levels: {valid}")
    return [LEVEL_BY_NAME[item] for item in requested]


def run_ladder(
    levels: Iterable[LadderLevel],
    *,
    theorist: Theorist | None,
    smoke: bool,
    trust_declared_certifications: bool,
) -> tuple[int, set[str]]:
    registered = 0
    certified: set[str] = set()

    for level in levels:
        level_before = set(certified)
        print(f"[{level.name}] {len(level.experiments)} experiment(s)")
        for item in level.experiments:
            if theorist is not None and not trust_declared_certifications:
                certified = _framework_certified_concepts(theorist)
            exp = make_ladder_strict_config(item.config_name, unlocked_concepts=certified)
            _verify_dynamic_unlocking(exp, certified)
            data_keys = _smoke_collect(exp.copy()) if smoke else {str(key) for key in exp.original_data}
            if theorist is not None:
                theorist.register_experiment(exp)
                registered += 1
            print(
                f"  + {item.label} {exp.exp_name}: "
                f"{len(data_keys)} open column(s), certified before={sorted(certified)}"
            )
            if item.certifies:
                if trust_declared_certifications:
                    certified.update(item.certifies)
                    print(f"    trusts declared certification: {', '.join(item.certifies)}")
                else:
                    print(f"    declares target certification: {', '.join(item.certifies)}")

        if theorist is not None and not trust_declared_certifications:
            certified = _framework_certified_concepts(theorist)
        missing = set(level.certifies) - certified
        if missing and trust_declared_certifications:
            raise RuntimeError(f"{level.name}: missing declared certification(s): {sorted(missing)}")
        gained = sorted(certified - level_before)
        if missing and not trust_declared_certifications:
            print(f"  framework has not certified yet: {', '.join(sorted(missing))}")
        print(f"  level gained from framework: {', '.join(gained) if gained else 'none'}")

    return registered, certified


def _framework_certified_concepts(theorist: Theorist) -> set[str]:
    certified: set[str] = set()
    for record in theorist.operational_registry.records.values():
        if record.status is ConceptStatus.CERTIFIED and record.symbolic_name is not None:
            certified.add(record.symbolic_name)
    return certified


def _assert_strict_only_theorist(theorist: Theorist) -> None:
    experiments = sorted(theorist.knowledge.fetch_exps)
    legacy = [name for name in experiments if not name.startswith("strict_")]
    if legacy:
        raise RuntimeError(f"Non-strict experiment(s) leaked into strict ladder: {legacy}")
    pool = sorted(theorist.memory.experiments_pool)
    leaked_pool = [name for name in pool if not name.startswith("strict_")]
    if leaked_pool:
        raise RuntimeError(f"Non-strict experiment(s) leaked into discovery pool: {leaked_pool}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register and optionally run the strict experiment ladder.")
    parser.add_argument("--levels", type=str, default="", help="Comma-separated levels, e.g. L1,L2,L3 or L5T,L5O.")
    parser.add_argument("--init-time-limit", type=float, default=5.0)
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument("--discover-epochs", type=int, default=0)
    parser.add_argument("--cpt-lim-mul", type=float, default=3.0)
    parser.add_argument("--max-actions", type=int, default=7)
    parser.add_argument("--cpt-lim-bias", type=float, default=5.0)
    parser.add_argument("--max-cpt-lim", type=float, default=3600.0)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument(
        "--trust-declared-certifications",
        action="store_true",
        help=(
            "Scaffold mode: unlock columns from LADDER_LEVELS declarations. "
            "Default discovery mode only unlocks framework-certified operational concepts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    args = parse_args()
    levels = select_levels(args.levels)
    if args.skip_register and args.discover_epochs > 0:
        raise SystemExit("--discover-epochs requires registration; remove --skip-register")

    theorist = None if args.skip_register else Theorist(
        init_time_limit=args.init_time_limit,
        num_threads=args.num_threads,
        include_default_experiments=False,
    )

    registered, certified = run_ladder(
        levels,
        theorist=theorist,
        smoke=not args.skip_smoke,
        trust_declared_certifications=args.trust_declared_certifications or theorist is None,
    )
    if theorist is not None:
        _assert_strict_only_theorist(theorist)
        print(f"strict-only experiment pool: {sorted(theorist.knowledge.fetch_exps)}")

    if theorist is not None and args.save_path:
        save_path = str(root / args.save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        theorist.save_to_file(save_path)
        print(f"saved registered ladder to {save_path}")

    if args.discover_epochs <= 0:
        action = "registered" if theorist is not None else "validated"
        count = registered if theorist is not None else sum(len(level.experiments) for level in levels)
        print(f"{action} {count} strict ladder experiments; certified={sorted(certified)}; discovery not started")
        return

    assert theorist is not None
    _assert_strict_only_theorist(theorist)
    theorist.perform_discovery(
        exp_select_mode="contextual",
        cpt_lim_mul=args.cpt_lim_mul,
        max_actions=args.max_actions,
        max_cpt_lim=args.max_cpt_lim,
        cpt_lim_bias=args.cpt_lim_bias,
        max_epochs=args.discover_epochs,
        save_path=str(root / args.save_path) if args.save_path else None,
    )


if __name__ == "__main__":
    main()
