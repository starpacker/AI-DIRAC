"""Run one resumable full-discovery arm for experiment-selector comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import socket
import time

import numpy as np
import torch

from aiphy import Theorist
from aiphy.contextual_bandit import TheoryStateEncoder
from aiphy.experiment.basic import (
    collision_2d_config,
    collision_elastic_3body_config,
    collision_nh_config,
    collision_y_config,
    motion0_3d_config,
    motion_3d_config,
)
from aiphy.experiment.cosmos import (
    celestial_2_xy_config,
    celestial_2_xz_config,
    celestial_2_yz_config,
    celestial_3_config,
    celestial_4_config,
)
from aiphy.experiment.ground_laboratory import (
    free_fall_config,
    gravity_config,
    hanging_spring_1_config,
    hanging_spring_2_config,
    hanging_spring_3_config,
    oblique_projectile_config,
    oblique_projectile_x_config,
    oblique_projectile_y_config,
    pendulum_oscillation_1_config,
    pendulum_oscillation_2_config,
    projectile_config,
    projectile_oscillation_1_config,
    projectile_oscillation_2_config,
    slope_oscillation_1_config,
    slope_oscillation_2_config,
    vertical_projectile_config,
)
from aiphy.experiment.oscillation import (
    oscillation_2d_1_config,
    oscillation_2d_2_config,
    oscillation_2d_3_config,
    oscillation_r_3_config,
    oscillation_r_4_config,
    oscillation_rot_1_config,
    oscillation_rot_2_config,
    oscillation_rot_3_config,
    oscillation_x_1_config,
    oscillation_x_2_config,
    oscillation_x_3_config,
)


EXPERIMENT_CONFIGS = [
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
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contextual", "ucb", "random"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--init-time-limit", type=float, default=5.0)
    parser.add_argument("--time-limit-multiplier", type=float, default=3.0)
    parser.add_argument("--max-time-limit", type=float, default=3600.0)
    parser.add_argument("--time-limit-bias", type=float, default=5.0)
    parser.add_argument("--max-actions", type=int, default=7)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def checkpoint_exists(prefix: Path) -> bool:
    suffixes = ["_knowledge.txt", "_memory.json", "_general_model.json", "_specific_model.json"]
    return all(Path(str(prefix) + suffix).exists() for suffix in suffixes)


def trim_metrics_to_checkpoint(metrics_path: Path, epoch: int):
    if not metrics_path.exists():
        return
    kept = []
    for line in metrics_path.read_text().splitlines():
        try:
            if int(json.loads(line)["epoch"]) <= epoch:
                kept.append(line)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))


def create_theorist(args, prefix: Path) -> Theorist:
    theorist = Theorist(init_time_limit=args.init_time_limit, num_threads=args.threads)
    for config_fn in EXPERIMENT_CONFIGS:
        exp = config_fn()
        if exp.exp_name not in theorist.specific:
            theorist.register_experiment(exp)
    theorist.save_to_file(str(prefix))
    return theorist


def summary(theorist: Theorist, args, started_at: float, completed_at: float) -> dict:
    state = TheoryStateEncoder.capture(theorist)
    total_experiments = len(theorist.knowledge.fetch_exps)
    laws = {
        name: {
            "prop": str(law.prop),
            "type": law.type,
            "indication": law.indication,
            "valid_experiments": sorted(law.valid_experiments),
            "coverage": len(law.valid_experiments) / max(total_experiments, 1),
            "father": law.father.name if law.father is not None else None,
        }
        for name, law in theorist.general.general_laws.items()
    }
    return {
        "status": "complete",
        "mode": args.mode,
        "seed": args.seed,
        "epochs": theorist.memory.epoch,
        "experiments": total_experiments,
        "concepts": len(state.concepts),
        "intrinsic_concepts": len(state.intrinsic_concepts),
        "conclusions": len(state.conclusions),
        "general_laws": len(state.general_laws),
        "valid_law_experiment_pairs": len(state.valid_pairs),
        "law_complexity": state.law_complexity,
        "theory_maturity": state.maturity,
        "state_hash": state.state_hash,
        "started_at_unix": started_at,
        "completed_at_unix": completed_at,
        "wall_seconds": completed_at - started_at,
        "laws": laws,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "checkpoint"
    metrics_path = output_dir / "metrics.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"

    now = time.time()
    previous_manifest = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            previous_manifest = {}
    started_at = float(previous_manifest.get("started_at_unix", now))
    manifest = {
        "status": "running",
        "pid": os.getpid(),
        "mode": args.mode,
        "seed": args.seed,
        "target_epochs": args.epochs,
        "threads": args.threads,
        "init_time_limit": args.init_time_limit,
        "time_limit_multiplier": args.time_limit_multiplier,
        "max_time_limit": args.max_time_limit,
        "time_limit_bias": args.time_limit_bias,
        "max_actions": args.max_actions,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "started_at_unix": started_at,
        "last_resumed_at_unix": now,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if checkpoint_exists(prefix):
        print(f"Resume checkpoint: {prefix}", flush=True)
        theorist = Theorist.read_from_file(
            str(prefix), reset_experiment=True, num_threads=args.threads
        )
        trim_metrics_to_checkpoint(metrics_path, theorist.memory.epoch)
    else:
        print(f"Create fresh checkpoint: {prefix}", flush=True)
        theorist = create_theorist(args, prefix)

    print(
        f"Start {args.mode}: epoch={theorist.memory.epoch} -> {args.epochs}, "
        f"experiments={len(theorist.knowledge.fetch_exps)}",
        flush=True,
    )
    theorist.perform_discovery(
        exp_select_mode=args.mode,
        cpt_lim_mul=args.time_limit_multiplier,
        max_actions=args.max_actions,
        max_cpt_lim=args.max_time_limit,
        cpt_lim_bias=args.time_limit_bias,
        max_epochs=args.epochs,
        save_path=str(prefix),
        metrics_path=str(metrics_path) if args.mode == "contextual" else None,
    )
    theorist.save_to_file(str(prefix))

    completed_at = time.time()
    result = summary(theorist, args, started_at, completed_at)
    summary_path.write_text(json.dumps(result, indent=2))
    manifest.update({
        "status": "complete",
        "completed_at_unix": completed_at,
        "wall_seconds": completed_at - started_at,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
