#!/usr/bin/env python3
"""Strict discovery: all thermal + fast mechanics, PySR mandatory, 45s timeout, dim check, 4-arity."""
import os, sys, time, threading
sys.path.insert(0, os.path.dirname(__file__))
os.environ["AI_NEWTON_PYSR_TIMEOUT_SECONDS"] = "45"
from aiphy import Theorist

def register_with_timeout(theorist, cfg, name, timeout_s=45):
    result = [False]
    def target():
        try: theorist.register_experiment(cfg()); result[0] = True
        except: pass
    t = threading.Thread(target=target); t.daemon = True; t.start()
    t.join(timeout_s)
    return result[0]

from aiphy.experiment.basic import *
from aiphy.experiment.ground_laboratory import *
from aiphy.experiment.oscillation import *
from aiphy.experiment.cosmos import *
from aiphy.experiment.challenge_mechanics import *
from aiphy.experiment.thermodynamics_epistemic import *
from aiphy.experiment.thermodynamics_state import *
from aiphy.experiment.challenge_thermodynamics.contact_networks import *
from aiphy.experiment.challenge_thermodynamics.lumped_relaxation import *
from aiphy.experiment.challenge_thermodynamics.driven_nonlinear import *
from aiphy.experiment.challenge_thermodynamics.phase_diffusion import *

N_EPOCHS = 600
SAVE = os.path.join(os.path.dirname(__file__), 'data', 'strict_run')

print(f"=== Strict Discovery: {N_EPOCHS}e, PySR, {time.strftime('%H:%M:%S')} ===", flush=True)
theorist = Theorist(init_time_limit=45, num_threads=32)

configs = [
    (motion0_3d_config,'m0'),(motion_3d_config,'m3'),(collision_elastic_3body_config,'c3'),
    (collision_y_config,'cy'),(collision_2d_config,'c2'),(collision_nh_config,'cn'),
    (circle_config,'ci'),(collision_circle_config,'cc'),(collision_circle_2_config,'cc2'),
    (gravity_config,'g'),(free_fall_config,'ff'),(projectile_config,'pr'),
    (oblique_projectile_config,'op'),(oblique_projectile_x_config,'opx'),(oblique_projectile_y_config,'opy'),
    (vertical_projectile_config,'vp'),(hanging_spring_1_config,'hs1'),(hanging_spring_2_config,'hs2'),
    (pendulum_oscillation_1_config,'po1'),(pendulum_oscillation_2_config,'po2'),
    (damped_pendulum_config,'dp'),(dynamic_spring_config,'ds'),(nonlinear_spring_config,'ns'),
    (inelastic_collision_config,'ic'),(pendulum_1_config,'p1'),
    (pulley_1_config,'pu1'),(pulley_2_config,'pu2'),(pulley_3_config,'pu3'),
    (large_angle_pendulum_config,'lap'),(variable_pendulum_config,'vp2'),
    (nonlinear_spring_do_config,'nsd'),
    (surface_gravity_relation_config,'sg'),
    (oscillation_x_1_config,'ox1'),(oscillation_x_2_config,'ox2'),(oscillation_x_3_config,'ox3'),
    (oscillation_2d_1_config,'o21'),(oscillation_2d_2_config,'o22'),(oscillation_2d_3_config,'o23'),
    (celestial_2_xy_config,'cxy'),(celestial_2_xz_config,'cxz'),(celestial_2_yz_config,'cyz'),
    (celestial_3_config,'c3'),(celestial_4_config,'c4'),
    (collision_1d_elastic_config,'c1e'),(damped_elastic_pendulum_1_config,'dep'),
    (damped_hanging_spring_1_config,'dhs1'),(damped_hanging_spring_2_config,'dhs2'),
    (damped_pendulum_1_config,'dp1'),(damped_pendulum_2_config,'dp2'),
    (driven_damped_hanging_spring_1_config,'ddhs1'),(driven_damped_hanging_spring_2_config,'ddhs2'),
    (free_fall_1_config,'ff1'),(pendulum_undamped_1_config,'pu1'),(spring_undamped_1_config,'su1'),
    (two_stage_rocket_1_config,'r1'),(variable_mass_rocket_1_config,'vm1'),(variable_mass_rocket_2_config,'vm2'),
    # ALL THERMAL
    (epistemic_gas_state_scan_config,'te1'),(epistemic_gas_isothermal_config,'te2'),
    (epistemic_gas_isochoric_config,'te3'),(epistemic_gas_isobaric_config,'te4'),(epistemic_gas_adiabatic_config,'te5'),
    (ideal_gas_state_scan_config,'ti1'),(ideal_gas_isothermal_config,'ti2'),
    (ideal_gas_isochoric_config,'ti3'),(ideal_gas_isobaric_config,'ti4'),(ideal_gas_adiabatic_config,'ti5'),
    (two_body_contact_equal_config,'tc1'),(two_body_contact_unequal_config,'tc2'),
    (two_body_contact_with_ambient_config,'tc3'),(three_body_chain_relaxation_config,'tc4'),
    (three_body_ring_relaxation_config,'tc5'),
    (lumped_cooling_static_env_config,'tl1'),(lumped_heating_static_env_config,'tl2'),
    (lumped_cooling_dynamic_env_config,'tl3'),(lumped_joule_heating_config,'tl4'),(lumped_radiative_cooling_config,'tl5'),
    (driven_ambient_oscillation_config,'td1'),(periodic_power_heating_config,'td2'),
    (thermostat_limit_cycle_config,'td3'),(exothermic_self_heating_config,'td4'),(quench_recovery_config,'td5'),
    (latent_heat_heating_plateau_config,'tp1'),(latent_heat_cooling_plateau_config,'tp2'),
    (latent_heat_contact_two_body_config,'tp3'),(rod_three_node_diffusion_config,'tp4'),(rod_four_node_diffusion_config,'tp5'),
    (pressure_discovery_config,'pd'),
]

reg = skip = 0
for cfg, name in configs:
    if register_with_timeout(theorist, cfg, name, 45):
        reg += 1
    else:
        skip += 1
        print(f"  SKIP: {name}", flush=True)

total = len(theorist.knowledge.fetch_exps)
print(f"\n  {reg} registered, {skip} skipped, {total} in pool", flush=True)
theorist.save_to_file(SAVE)
print(f"  Starting discovery ({time.strftime('%H:%M:%S')})...", flush=True)

theorist.perform_discovery(
    exp_select_mode='contextual', regression_backend='pysr',
    cpt_lim_mul=2, max_actions=5, max_cpt_lim=900., cpt_lim_bias=8,
    max_epochs=N_EPOCHS, save_path=SAVE,
)

print(f"\n=== DONE {time.strftime('%H:%M:%S')} ===", flush=True)
theorist.save_to_file(SAVE)
print(f"Concepts: {len(theorist.knowledge.fetch_concepts)}")
print(f"Conclusions: {len(theorist.knowledge.fetch_conclusions)}")
print(f"General laws: {len(theorist.general.general_laws)}")
for name, law in theorist.general.general_laws.items():
    vc = len(law.valid_experiments) if hasattr(law, 'valid_experiments') else '?'
    print(f"  {name}: {law} (valid in {vc}/{total})")
for name, cpt in sorted(theorist.knowledge.fetch_concepts.items()):
    print(f"  {name}: {cpt}")
