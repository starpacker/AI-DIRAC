"""Run the original AI-Newton source suite with contextual experiment selection.

The 38 factories below are copied exactly from ``test/examples/test_example_1.py``.
No challenge-suite experiment is imported or registered here.
"""

from pathlib import Path
import os
import time

from aiphy import Theorist
from aiphy.experiment.basic import (
    motion0_3d_config,
    motion_3d_config,
    collision_elastic_3body_config,
    collision_y_config,
    collision_2d_config,
    collision_nh_config,
)
from aiphy.experiment.ground_laboratory import (
    gravity_config,
    free_fall_config,
    projectile_config,
    oblique_projectile_x_config,
    oblique_projectile_y_config,
    oblique_projectile_config,
    vertical_projectile_config,
    hanging_spring_1_config,
    hanging_spring_2_config,
    hanging_spring_3_config,
    pendulum_oscillation_1_config,
    pendulum_oscillation_2_config,
    slope_oscillation_1_config,
    slope_oscillation_2_config,
    projectile_oscillation_1_config,
    projectile_oscillation_2_config,
)
from aiphy.experiment.oscillation import (
    oscillation_x_1_config,
    oscillation_x_2_config,
    oscillation_x_3_config,
    oscillation_2d_1_config,
    oscillation_2d_2_config,
    oscillation_2d_3_config,
    oscillation_r_3_config,
    oscillation_r_4_config,
    oscillation_rot_1_config,
    oscillation_rot_2_config,
    oscillation_rot_3_config,
)
from aiphy.experiment.cosmos import (
    celestial_2_xy_config,
    celestial_2_xz_config,
    celestial_2_yz_config,
    celestial_3_config,
    celestial_4_config,
)


NUMBER_OF_ITERATIONS = 1200
NUM_THREADS = 32
INIT_TIME_LIMIT = 5
COMPUTE_LIMIT_MULTIPLIER = 3
MAX_ACTIONS = 7
COMPUTE_LIMIT_BIAS = 5
MAX_COMPUTE_LIMIT = 3600.0

SOURCE_EXPERIMENT_FACTORIES = (
    motion0_3d_config,
    motion_3d_config,
    collision_elastic_3body_config,
    collision_y_config,
    collision_2d_config,
    collision_nh_config,
    gravity_config,
    free_fall_config,
    projectile_config,
    oblique_projectile_config,
    oblique_projectile_x_config,
    oblique_projectile_y_config,
    vertical_projectile_config,
    hanging_spring_1_config,
    hanging_spring_2_config,
    hanging_spring_3_config,
    pendulum_oscillation_1_config,
    pendulum_oscillation_2_config,
    slope_oscillation_1_config,
    slope_oscillation_2_config,
    projectile_oscillation_1_config,
    projectile_oscillation_2_config,
    oscillation_x_1_config,
    oscillation_x_2_config,
    oscillation_x_3_config,
    oscillation_2d_1_config,
    oscillation_2d_2_config,
    oscillation_2d_3_config,
    oscillation_r_3_config,
    oscillation_r_4_config,
    oscillation_rot_1_config,
    oscillation_rot_2_config,
    oscillation_rot_3_config,
    celestial_2_xy_config,
    celestial_2_xz_config,
    celestial_2_yz_config,
    celestial_3_config,
    celestial_4_config,
)

OUTPUT_DIR = Path("data/test_cases/contextual_source38")
CHECKPOINT = OUTPUT_DIR / "contextual_source38_run1"


def main():
    if len(SOURCE_EXPERIMENT_FACTORIES) != 38:
        raise RuntimeError(
            f"Source-suite registration changed: expected 38, got {len(SOURCE_EXPERIMENT_FACTORIES)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72, flush=True)
    selection_mode = globals().get(
        "EXP_SELECT_MODE", os.getenv("AI_NEWTON_EXP_SELECT_MODE", "contextual")
    )
    regression_backend = globals().get(
        "REGRESSION_BACKEND", os.getenv("AI_NEWTON_REGRESSION_BACKEND", "ver3")
    )
    print(f"AI-Newton source-suite run ({selection_mode} selection)", flush=True)
    print(f"PID: {os.getpid()}", flush=True)
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Source registrations: 38", flush=True)
    print("Challenge registrations: 0", flush=True)
    print(f"Epochs: {NUMBER_OF_ITERATIONS}; threads: {NUM_THREADS}", flush=True)
    print(f"Experiment selection: {selection_mode}", flush=True)
    print(f"Regression backend: {regression_backend}", flush=True)
    print("General-law reducer: enabled", flush=True)
    print(f"Checkpoint: {CHECKPOINT}", flush=True)
    print("=" * 72, flush=True)

    theorist = Theorist(init_time_limit=INIT_TIME_LIMIT, num_threads=NUM_THREADS)
    built_in_experiments = set(theorist.knowledge.fetch_exps)
    registered_names = []
    for factory in SOURCE_EXPERIMENT_FACTORIES:
        experiment = factory()
        name = experiment.exp_name
        if "challenge" in name.lower():
            raise RuntimeError(f"Challenge experiment is forbidden in this run: {name}")
        theorist.register_experiment(experiment)
        registered_names.append(name)

    all_experiments = set(theorist.knowledge.fetch_exps)
    print(f"Built-in experiments: {len(built_in_experiments)}", flush=True)
    print(f"Source registration calls: {len(registered_names)}", flush=True)
    print(f"Unique experiments available to selector: {len(all_experiments)}", flush=True)
    print(f"Registered source names: {registered_names}", flush=True)
    print(f"All selector names: {sorted(all_experiments)}", flush=True)

    theorist.save_to_file(str(CHECKPOINT))
    theorist.perform_discovery(
        exp_select_mode=selection_mode,
        regression_backend=regression_backend,
        cpt_lim_mul=COMPUTE_LIMIT_MULTIPLIER,
        max_actions=MAX_ACTIONS,
        max_cpt_lim=MAX_COMPUTE_LIMIT,
        cpt_lim_bias=COMPUTE_LIMIT_BIAS,
        max_epochs=NUMBER_OF_ITERATIONS,
        save_path=str(CHECKPOINT),
    )
    theorist.save_to_file(str(CHECKPOINT))
    print(f"Finish: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
