"""Summarize the original UCB log and contextual JSONL on a common schema."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+): (\S+)")
FINISHED_RE = re.compile(r"Discovery for (\S+) is finished in ([0-9.]+)s")
TIMEOUT_RE = re.compile(r"computational limit\(([0-9.]+)s\)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--baseline-memory", required=True)
    parser.add_argument("--baseline-general", required=True)
    parser.add_argument("--contextual-metrics", required=True)
    parser.add_argument("--contextual-memory", required=True)
    parser.add_argument("--contextual-general", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def finish(records: list[dict]) -> dict:
    successful = [row for row in records if row.get("outcome") == "success"]
    durations = [float(row["duration_seconds"]) for row in records if row.get("duration_seconds") is not None]
    rewards = [float(row["reward"]) for row in records if row.get("reward") is not None]
    return {
        "epochs_observed": len(records),
        "first_epoch": min((row["epoch"] for row in records), default=None),
        "last_epoch": max((row["epoch"] for row in records), default=None),
        "successes": len(successful),
        "timeouts_or_errors": len(records) - len(successful),
        "success_rate": len(successful) / max(len(records), 1),
        "analysis_seconds_observed": sum(durations),
        "mean_analysis_seconds": statistics.fmean(durations) if durations else None,
        "median_analysis_seconds": statistics.median(durations) if durations else None,
        "total_reward_observed": sum(rewards),
        "mean_reward": statistics.fmean(rewards) if rewards else None,
        "experiment_counts": dict(Counter(row["experiment"] for row in records)),
        "records": records,
    }


def parse_baseline(path: Path) -> dict:
    records: list[dict] = []
    current = None
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw).strip()
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            if current is not None:
                records.append(current)
            current = {
                "epoch": int(epoch_match.group(1)),
                "experiment": epoch_match.group(3),
                "outcome": "unknown",
                "duration_seconds": None,
                "reward": None,
                "new_concepts": 0,
                "new_laws": 0,
            }
            continue
        if current is None:
            continue
        finished = FINISHED_RE.search(line)
        if finished:
            current["outcome"] = "success"
            current["duration_seconds"] = float(finished.group(2))
        elif "terminated due to computational limit" in line:
            current["outcome"] = "timeout"
            timeout = TIMEOUT_RE.search(line)
            if timeout:
                current["duration_seconds"] = float(timeout.group(1))
        elif line.startswith("Total rewards:"):
            try:
                current["reward"] = float(line.rsplit(":", 1)[1])
            except ValueError:
                pass
        elif "Registered New Concept:" in line:
            current["new_concepts"] += 1
        elif "Registered new general law:" in line:
            current["new_laws"] += 1
    if current is not None:
        records.append(current)
    return finish(records)


def parse_contextual(path: Path) -> dict:
    by_epoch = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reward = row["reward"]
            by_epoch[int(row["epoch"])] = {
                "epoch": int(row["epoch"]),
                "experiment": row["experiment"],
                "outcome": reward["outcome"],
                "duration_seconds": float(row["duration_seconds"]),
                "reward": float(reward["total"]),
                "legacy_reward": float(reward["legacy_reward"]),
                "new_concepts": int(reward["added_concepts"]),
                "new_laws": int(reward["added_laws"]),
                "revised_laws": int(reward["revised_laws"]),
                "contradictions": int(reward["new_contradictions"]),
                "validations": int(reward["new_validations"]),
                "maturity": float(reward["maturity"]),
            }
    return finish([by_epoch[key] for key in sorted(by_epoch)])


def checkpoint_summary(memory_path: Path, general_path: Path) -> dict:
    if not memory_path.exists() or not general_path.exists():
        return {"available": False}
    memory = json.loads(memory_path.read_text())
    general = json.loads(general_path.read_text())
    laws = general.get("general_laws", {})
    valid_pairs = sum(len(law.get("valid_experiments", [])) for law in laws.values())
    return {
        "available": True,
        "epoch": int(memory.get("epoch", 0)),
        "general_laws": len(laws),
        "valid_law_experiment_pairs": valid_pairs,
        "history_experiments": len(memory.get("history", {})),
        "history_relations": sum(len(items) for items in memory.get("history", {}).values()),
        "contextual_updates": memory.get("contextual_bandit", {}).get("total_updates"),
    }


def comparable_prefix(baseline: dict, contextual: dict) -> dict:
    common = min(baseline["last_epoch"] or 0, contextual["last_epoch"] or 0)
    if common == 0:
        return {"epochs": 0}
    base = finish([row for row in baseline["records"] if row["epoch"] <= common])
    ctx = finish([row for row in contextual["records"] if row["epoch"] <= common])
    return {"epochs": common, "baseline": base, "contextual": ctx}


def markdown(report: dict) -> str:
    base = report["baseline"]
    ctx = report["contextual"]
    prefix = report["comparable_prefix"]
    lines = [
        "# AI-Newton full comparison",
        "",
        "| Arm | Epoch | Success rate | Observed analysis time | General laws | Valid law-exp pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, run in (("Original UCB", base), ("Contextual", ctx)):
        checkpoint = run["checkpoint"]
        lines.append(
            f"| {name} | {checkpoint.get('epoch', 0)} | {run['success_rate']:.1%} | "
            f"{run['analysis_seconds_observed']:.1f}s | {checkpoint.get('general_laws', 0)} | "
            f"{checkpoint.get('valid_law_experiment_pairs', 0)} |"
        )
    lines.extend(["", f"Comparable prefix: first {prefix['epochs']} epochs."])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = parse_baseline(Path(args.baseline_log))
    contextual = parse_contextual(Path(args.contextual_metrics))
    baseline["checkpoint"] = checkpoint_summary(
        Path(args.baseline_memory), Path(args.baseline_general)
    )
    contextual["checkpoint"] = checkpoint_summary(
        Path(args.contextual_memory), Path(args.contextual_general)
    )
    report = {
        "baseline": baseline,
        "contextual": contextual,
        "comparable_prefix": comparable_prefix(baseline, contextual),
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    (output / "comparison.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
