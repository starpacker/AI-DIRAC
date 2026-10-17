#!/usr/bin/env python3
"""Pre-experiment runner: register ONE strict experiment, run discovery,
then dump human-readable candidates for manual verdict on intrinsic quantities."""
import argparse, json, os, sys
from pathlib import Path

from aiphy import MeasureType, Theorist
from aiphy.experiment.ladder_strict import _specs, make_ladder_strict_config, MODE


def dump_candidates(theorist, exp_name, unlocked, out_dir: Path):
    """Write a human-readable candidate report for manual verdict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Preflight report: {exp_name}")
    lines.append(f"# unlocked_concepts passed in: {sorted(unlocked)}")
    lines.append("")
    lines.append("## Discovered concepts (name := expression)")
    for name, cpt in sorted(theorist.knowledge.fetch_concepts.items()):
        lines.append(f"{name} := {cpt}")
    lines.append("")
    lines.append("## General laws")
    for name, law in theorist.general.general_laws.items():
        vc = len(law.valid_experiments) if hasattr(law, "valid_experiments") else "?"
        lines.append(f"{name}: {law}  (valid in {vc})")
    report = out_dir / f"preflight_{exp_name}_report.txt"
    report.write_text("\n".join(lines))
    print(f"wrote {report}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="strict experiment config name")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--unlocked", default="", help="comma-separated already-unlocked concepts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-threads", type=int, default=32)
    args = ap.parse_args()

    unlocked = {s.strip() for s in args.unlocked.split(",") if s.strip()}
    out_dir = Path(args.out)

    theorist = Theorist(init_time_limit=45, num_threads=args.num_threads,
                        include_default_experiments=False)
    exp = make_ladder_strict_config(args.exp, unlocked_concepts=unlocked)
    open_cols = sorted(str(k) for k in exp.original_data)
    print(f"[{args.exp}] open columns ({len(open_cols)}): {open_cols}")
    print(f"[{args.exp}] unlocked_concepts={sorted(unlocked)}")
    theorist.register_experiment(exp)

    print(f"starting discovery: {args.epochs} epochs")
    theorist.perform_discovery(
        exp_select_mode="contextual",
        cpt_lim_mul=2, max_actions=5, max_cpt_lim=900, cpt_lim_bias=8,
        max_epochs=args.epochs,
        save_path=str(out_dir / args.exp),
    )
    theorist.save_to_file(str(out_dir / args.exp))
    dump_candidates(theorist, args.exp, unlocked, out_dir)
    print("\n=== discovery done ===")
    print(f"concepts: {len(theorist.knowledge.fetch_concepts)}")
    print(f"general laws: {len(theorist.general.general_laws)}")


if __name__ == "__main__":
    main()
