"""Machine-readable electromagnetic discovery curriculum.

The entries here are experiment contracts, not formula hints for symbolic
regression.  Benchmark relations are hidden from the regressor and used only
for scoring or human audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EMExperimentSpec:
    experiment_id: str
    stage: str
    raw_observations: tuple[str, ...]
    control_settings: tuple[str, ...]
    required_unlocked_concepts: tuple[str, ...]
    forbidden_locked_concepts: tuple[str, ...]
    candidate_concepts: tuple[str, ...]
    benchmark_relations: tuple[str, ...]
    validation_holdouts: tuple[str, ...]

    def assert_unlock_contract(self) -> None:
        overlap = set(self.required_unlocked_concepts) & set(
            self.forbidden_locked_concepts
        )
        if overlap:
            raise ValueError(
                f"{self.experiment_id} has concepts both required and forbidden: "
                f"{sorted(overlap)}"
            )


COMMON_PRE_FIELD_FORBIDDEN = (
    "charge",
    "electric_field",
    "magnetic_field",
    "electric_potential",
    "electric_flux",
    "magnetic_flux",
    "emf",
    "capacitance",
    "inductance",
    "electric_dipole",
    "magnetic_dipole",
    "epsilon0",
    "mu0",
)


CURRICULUM_SPECS: tuple[EMExperimentSpec, ...] = (
    EMExperimentSpec(
        experiment_id="kinematics_free_motion",
        stage="A1",
        raw_observations=("t", "posx", "posy", "posz"),
        control_settings=("initial_position", "launch_setting"),
        required_unlocked_concepts=(),
        forbidden_locked_concepts=COMMON_PRE_FIELD_FORBIDDEN
        + ("mass", "force", "momentum", "energy"),
        candidate_concepts=("velocity", "acceleration"),
        benchmark_relations=(
            "d(pos)/d(t) = velocity",
            "d(velocity)/d(t) = acceleration",
            "free isolated motion: acceleration = 0",
        ),
        validation_holdouts=("dimension", "launch_speed", "initial_position"),
    ),
    EMExperimentSpec(
        experiment_id="charge_accumulation_by_current",
        stage="B1",
        raw_observations=("t", "current", "switch_state", "conductor_id"),
        control_settings=("current_source_setting", "transfer_duration"),
        required_unlocked_concepts=(),
        forbidden_locked_concepts=tuple(
            item for item in COMMON_PRE_FIELD_FORBIDDEN if item != "charge"
        ),
        candidate_concepts=("charge",),
        benchmark_relations=(
            "Delta(charge) = integral(current dt)",
            "d(charge)/d(t) = current",
        ),
        validation_holdouts=("current_waveform", "current_sign", "conductor_id"),
    ),
    EMExperimentSpec(
        experiment_id="charge_transfer_conservation",
        stage="B2",
        raw_observations=("t", "current", "charge", "switch_state", "conductor_id"),
        control_settings=("switch_topology",),
        required_unlocked_concepts=("charge",),
        forbidden_locked_concepts=tuple(
            item for item in COMMON_PRE_FIELD_FORBIDDEN if item != "charge"
        ),
        candidate_concepts=("charge_sign",),
        benchmark_relations=(
            "isolated system: sum(charge_i) = constant",
            "current_ab = -current_ba",
        ),
        validation_holdouts=("three_conductor_cycle", "switch_sequence"),
    ),
    EMExperimentSpec(
        experiment_id="coulomb_two_charge_force",
        stage="C1",
        raw_observations=("t", "posx", "posy", "posz", "charge", "force"),
        control_settings=("source_charge", "test_charge", "initial_separation"),
        required_unlocked_concepts=("charge", "force", "mass", "acceleration"),
        forbidden_locked_concepts=(
            "electric_field",
            "electric_potential",
            "electric_flux",
            "epsilon0",
        ),
        candidate_concepts=("coulomb_constant", "electric_interaction"),
        benchmark_relations=(
            "force_12 = -force_21",
            "force_12 = k_e*q1*q2*r_12/|r_12|^3",
        ),
        validation_holdouts=("charge_ratio", "distance", "geometry_dimension"),
    ),
    EMExperimentSpec(
        experiment_id="electric_field_from_test_charge",
        stage="D1",
        raw_observations=("t", "posx", "posy", "posz", "mass", "charge"),
        control_settings=("source_configuration", "test_charge", "test_mass"),
        required_unlocked_concepts=("charge", "mass", "acceleration"),
        forbidden_locked_concepts=("force", "electric_field", "electric_potential"),
        candidate_concepts=("electric_field",),
        benchmark_relations=(
            "mass_test*acceleration_test/charge_test is invariant",
            "acceleration = charge*electric_field/mass",
            "electric_field_total = sum(electric_field_i)",
        ),
        validation_holdouts=("test_particle", "source_count", "spatial_grid_point"),
    ),
    EMExperimentSpec(
        experiment_id="magnetic_field_from_moving_charge",
        stage="G1",
        raw_observations=(
            "t",
            "posx",
            "posy",
            "posz",
            "velocity",
            "acceleration",
            "mass",
            "charge",
            "electric_field",
        ),
        control_settings=("source_apparatus", "test_velocity", "test_charge"),
        required_unlocked_concepts=(
            "charge",
            "mass",
            "velocity",
            "acceleration",
            "electric_field",
        ),
        forbidden_locked_concepts=("force", "magnetic_field", "magnetic_flux", "mu0"),
        candidate_concepts=("magnetic_field",),
        benchmark_relations=(
            "acceleration depends on charge/mass, electric_field, velocity, and source_apparatus",
            "acceleration = (charge/mass)*(electric_field + velocity cross magnetic_field)",
        ),
        validation_holdouts=("velocity_direction", "charge_sign", "source_orientation"),
    ),
    EMExperimentSpec(
        experiment_id="lorentz_dynamics_crossed_fields",
        stage="I",
        raw_observations=(
            "t",
            "posx",
            "posy",
            "posz",
            "velocity",
            "acceleration",
            "mass",
            "charge",
            "electric_field",
            "magnetic_field",
        ),
        control_settings=("field_orientation", "charge_mass_ratio"),
        required_unlocked_concepts=(
            "charge",
            "mass",
            "velocity",
            "acceleration",
            "electric_field",
            "magnetic_field",
        ),
        forbidden_locked_concepts=("force",),
        candidate_concepts=("lorentz_dynamics_law",),
        benchmark_relations=(
            "acceleration = (charge/mass)*(electric_field + velocity cross magnetic_field)",
        ),
        validation_holdouts=("field_angle", "charge_sign", "coordinate_plane"),
    ),
    EMExperimentSpec(
        experiment_id="faraday_changing_magnetic_flux",
        stage="J2",
        raw_observations=("t", "current", "electric_field", "magnetic_flux", "emf"),
        control_settings=("loop_geometry", "source_motion", "orientation"),
        required_unlocked_concepts=("electric_field", "magnetic_field", "magnetic_flux", "emf"),
        forbidden_locked_concepts=("inductance",),
        candidate_concepts=("faraday_law", "inductance"),
        benchmark_relations=(
            "emf = -d(magnetic_flux)/d(t)",
            "curl(electric_field) = -partial(magnetic_field)/partial(t)",
        ),
        validation_holdouts=("loop_shape", "source_motion", "orientation"),
    ),
    EMExperimentSpec(
        experiment_id="charging_capacitor_displacement_current",
        stage="K2",
        raw_observations=("t", "current", "electric_flux", "magnetic_field"),
        control_settings=("plate_area", "plate_separation", "loop_geometry"),
        required_unlocked_concepts=(
            "electric_field",
            "electric_flux",
            "magnetic_field",
            "current_density",
            "epsilon0",
            "mu0",
        ),
        forbidden_locked_concepts=(),
        candidate_concepts=("displacement_current", "maxwell_ampere_law"),
        benchmark_relations=(
            "line_integral(B dl) = mu0*I + mu0*epsilon0*d(Phi_E)/d(t)",
            "curl(B) = mu0*J + mu0*epsilon0*partial(E)/partial(t)",
        ),
        validation_holdouts=("spanning_surface", "charging_waveform", "plate_separation"),
    ),
    EMExperimentSpec(
        experiment_id="electric_dipole_torque_and_field",
        stage="M1",
        raw_observations=("charge", "separation_vector", "electric_field", "torque"),
        control_settings=("dipole_orientation", "external_field_strength"),
        required_unlocked_concepts=("charge", "electric_field", "torque"),
        forbidden_locked_concepts=("electric_dipole",),
        candidate_concepts=("electric_dipole",),
        benchmark_relations=(
            "electric_dipole = charge*separation_vector",
            "torque = electric_dipole cross electric_field",
        ),
        validation_holdouts=("charge_separation_product", "orientation"),
    ),
    EMExperimentSpec(
        experiment_id="magnetic_dipole_current_loop",
        stage="M2",
        raw_observations=("current", "loop_area", "normal_vector", "magnetic_field", "torque"),
        control_settings=("loop_orientation", "external_field_strength"),
        required_unlocked_concepts=("magnetic_field", "torque"),
        forbidden_locked_concepts=("magnetic_dipole",),
        candidate_concepts=("magnetic_dipole",),
        benchmark_relations=(
            "magnetic_dipole = current*loop_area*normal_vector",
            "torque = magnetic_dipole cross magnetic_field",
        ),
        validation_holdouts=("loop_area", "current_sign", "orientation"),
    ),
)


def specs_by_stage(stage: str) -> tuple[EMExperimentSpec, ...]:
    return tuple(spec for spec in CURRICULUM_SPECS if spec.stage == stage)


def spec_by_id(experiment_id: str) -> EMExperimentSpec:
    for spec in CURRICULUM_SPECS:
        if spec.experiment_id == experiment_id:
            return spec
    raise KeyError(f"Unknown electromagnetic curriculum experiment: {experiment_id}")


def validate_curriculum_specs(specs: Iterable[EMExperimentSpec] = CURRICULUM_SPECS) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.experiment_id in seen:
            raise ValueError(f"Duplicate experiment id: {spec.experiment_id}")
        seen.add(spec.experiment_id)
        spec.assert_unlock_contract()
