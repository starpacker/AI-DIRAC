#!/usr/bin/env python3
"""Strict low-prior discovery: mechanics + thermal, PySR mandatory, no physics leaks."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from aiphy import Theorist
from aiphy.experiment.basic import (
    motion0_3d_config, motion_3d_config, collision_elastic_3body_config,
    collision_y_config, collision_2d_config, collision_nh_config,
    circle_config, collision_circle_config, collision_circle_2_config,
)
from aiphy.experiment.ground_laboratory import (
    gravity_config, free_fall_config, projectile_config, oblique_projectile_x_config,
    oblique_projectile_y_config, oblique_projectile_config, vertical_projectile_config,
    hanging_spring_1_config, hanging_spring_2_config, hanging_spring_3_config,
    pendulum_oscillation_1_config, pendulum_oscillation_2_config,
    slope_oscillation_1_config, slope_oscillation_2_config,
    projectile_oscillation_1_config, projectile_oscillation_2_config,
    damped_pendulum_config, dynamic_spring_config, nonlinear_spring_config,
    inelastic_collision_config, pendulum_1_config, pendulum_2_config, pendulum_3_config,
    pulley_1_config, pulley_2_config, pulley_3_config,
    large_angle_pendulum_config, variable_pendulum_config,
    masspulley_oscillation_1_config, masspulley_oscillation_2_config,
    nonlinear_spring_do_config, pulley_oscillation_1_config, pulley_oscillation_2_config,
    surface_gravity_relation_config,
)
from aiphy.experiment.oscillation import (
    oscillation_x_1_config, oscillation_x_2_config, oscillation_x_3_config,
    oscillation_2d_1_config, oscillation_2d_2_config, oscillation_2d_3_config,
    oscillation_r_3_config, oscillation_r_4_config,
    oscillation_rot_1_config, oscillation_rot_2_config, oscillation_rot_3_config,
)
from aiphy.experiment.cosmos import (
    celestial_2_xy_config, celestial_2_xz_config, celestial_2_yz_config,
    celestial_3_config, celestial_4_config,
)
from aiphy.experiment.challenge_mechanics import (
    collision_1d_elastic_config, damped_elastic_pendulum_1_config,
    damped_hanging_spring_1_config, damped_hanging_spring_2_config,
    damped_pendulum_1_config, damped_pendulum_2_config,
    driven_damped_hanging_spring_1_config, driven_damped_hanging_spring_2_config,
    free_fall_1_config, pendulum_undamped_1_config, spring_undamped_1_config,
    two_stage_rocket_1_config, variable_mass_rocket_1_config, variable_mass_rocket_2_config,
)
from aiphy.experiment.thermodynamics_epistemic import (
    epistemic_gas_state_scan_config, epistemic_gas_isothermal_config,
    epistemic_gas_isochoric_config, epistemic_gas_isobaric_config, epistemic_gas_adiabatic_config,
)
from aiphy.experiment.thermodynamics_state import (
    ideal_gas_state_scan_config, ideal_gas_isothermal_config,
    ideal_gas_isochoric_config, ideal_gas_isobaric_config, ideal_gas_adiabatic_config,
)
from aiphy.experiment.challenge_thermodynamics.contact_networks import (
    two_body_contact_equal_config, two_body_contact_unequal_config,
    two_body_contact_with_ambient_config, three_body_chain_relaxation_config,
    three_body_ring_relaxation_config,
)
from aiphy.experiment.challenge_thermodynamics.lumped_relaxation import (
    lumped_cooling_static_env_config, lumped_heating_static_env_config,
    lumped_cooling_dynamic_env_config, lumped_joule_heating_config,
    lumped_radiative_cooling_config,
)
from aiphy.experiment.challenge_thermodynamics.driven_nonlinear import (
    driven_ambient_oscillation_config, periodic_power_heating_config,
    thermostat_limit_cycle_config, exothermic_self_heating_config, quench_recovery_config,
)
from aiphy.experiment.challenge_thermodynamics.phase_diffusion import (
    latent_heat_heating_plateau_config, latent_heat_cooling_plateau_config,
    latent_heat_contact_two_body_config, rod_three_node_diffusion_config,
    rod_four_node_diffusion_config,
)

N_EPOCHS = 600
SAVE = os.path.join(os.path.dirname(__file__), 'data', 'strict_run')

print(f"=== AI-Newton Strict Discovery ===", flush=True)
print(f"  Epochs: {N_EPOCHS} | Backend: PySR | Time: {time.strftime('%H:%M:%S')}", flush=True)

theorist = Theorist(init_time_limit=5, num_threads=32)

configs = [
    motion0_3d_config(), motion_3d_config(), collision_elastic_3body_config(),
    collision_y_config(), collision_2d_config(), collision_nh_config(),
    circle_config(), collision_circle_config(), collision_circle_2_config(),
    gravity_config(), free_fall_config(), projectile_config(),
    oblique_projectile_config(), oblique_projectile_x_config(), oblique_projectile_y_config(),
    vertical_projectile_config(), hanging_spring_1_config(), hanging_spring_2_config(),
    hanging_spring_3_config(), pendulum_oscillation_1_config(), pendulum_oscillation_2_config(),
    slope_oscillation_1_config(), slope_oscillation_2_config(),
    projectile_oscillation_1_config(), projectile_oscillation_2_config(),
    damped_pendulum_config(), dynamic_spring_config(), nonlinear_spring_config(),
    inelastic_collision_config(), pendulum_1_config(), pendulum_2_config(), pendulum_3_config(),
    pulley_1_config(), pulley_2_config(), pulley_3_config(),
    large_angle_pendulum_config(), variable_pendulum_config(),
    masspulley_oscillation_1_config(), masspulley_oscillation_2_config(),
    nonlinear_spring_do_config(), pulley_oscillation_1_config(), pulley_oscillation_2_config(),
    surface_gravity_relation_config(),
    oscillation_x_1_config(), oscillation_x_2_config(), oscillation_x_3_config(),
    oscillation_2d_1_config(), oscillation_2d_2_config(), oscillation_2d_3_config(),
    oscillation_r_3_config(), oscillation_r_4_config(),
    oscillation_rot_1_config(), oscillation_rot_2_config(), oscillation_rot_3_config(),
    celestial_2_xy_config(), celestial_2_xz_config(), celestial_2_yz_config(),
    celestial_3_config(), celestial_4_config(),
    collision_1d_elastic_config(), damped_elastic_pendulum_1_config(),
    damped_hanging_spring_1_config(), damped_hanging_spring_2_config(),
    damped_pendulum_1_config(), damped_pendulum_2_config(),
    driven_damped_hanging_spring_1_config(), driven_damped_hanging_spring_2_config(),
    free_fall_1_config(), pendulum_undamped_1_config(), spring_undamped_1_config(),
    two_stage_rocket_1_config(), variable_mass_rocket_1_config(), variable_mass_rocket_2_config(),
    epistemic_gas_state_scan_config(), epistemic_gas_isothermal_config(),
    epistemic_gas_isochoric_config(), epistemic_gas_isobaric_config(), epistemic_gas_adiabatic_config(),
    ideal_gas_state_scan_config(), ideal_gas_isothermal_config(),
    ideal_gas_isochoric_config(), ideal_gas_isobaric_config(), ideal_gas_adiabatic_config(),
    two_body_contact_equal_config(), two_body_contact_unequal_config(),
    two_body_contact_with_ambient_config(), three_body_chain_relaxation_config(),
    three_body_ring_relaxation_config(),
    lumped_cooling_static_env_config(), lumped_heating_static_env_config(),
    lumped_cooling_dynamic_env_config(), lumped_joule_heating_config(),
    lumped_radiative_cooling_config(),
    driven_ambient_oscillation_config(), periodic_power_heating_config(),
    thermostat_limit_cycle_config(), exothermic_self_heating_config(), quench_recovery_config(),
    latent_heat_heating_plateau_config(), latent_heat_cooling_plateau_config(),
    latent_heat_contact_two_body_config(), rod_three_node_diffusion_config(),
    rod_four_node_diffusion_config(),
]

for cfg in configs:
    theorist.register_experiment(cfg)

total = len(theorist.knowledge.fetch_exps)
print(f"  {total} experiments registered", flush=True)
theorist.save_to_file(SAVE)

print(f"  Starting discovery loop...", flush=True)
theorist.perform_discovery(
    exp_select_mode='contextual', regression_backend='pysr',
    cpt_lim_mul=2, max_actions=5, max_cpt_lim=600., cpt_lim_bias=4,
    max_epochs=N_EPOCHS, save_path=SAVE,
)

print(f"\n=== DONE ({time.strftime('%H:%M:%S')}) ===", flush=True)
theorist.save_to_file(SAVE)
print(f"Concepts: {len(theorist.knowledge.fetch_concepts)}")
print(f"Conclusions: {len(theorist.knowledge.fetch_conclusions)}")
print(f"General laws: {len(theorist.general.general_laws)}")
for name, law in theorist.general.general_laws.items():
    vc = len(law.valid_experiments) if hasattr(law, 'valid_experiments') else '?'
    print(f"  {name}: {law} (valid in {vc}/{total})")
for name, cpt in sorted(theorist.knowledge.fetch_concepts.items()):
    print(f"  {name}: {cpt}")
