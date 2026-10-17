"""Summarize v3 PySR/reducer progress against the historical baseline log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def summarize_log(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = ANSI.sub("", path.read_text(errors="replace"))
    epochs = [int(value) for value in re.findall(r"Epoch (\d+)/\d+", text)]
    discovery = text
    first_epoch = re.search(r"Epoch 1/\d+", text)
    if first_epoch is not None:
        discovery = text[first_epoch.start():]
    rewards = [int(value) for value in re.findall(r"Total rewards: (-?\d+)", discovery)]
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "last_epoch": max(epochs, default=0),
        "completed_epochs": len(rewards),
        "reward_sum": sum(rewards),
        "positive_reward_epochs": sum(value > 0 for value in rewards),
        "registered_law_events": discovery.count("Registered new general law:"),
        "pysr_accept_events": sum(
            discovery.count(marker) for marker in (
                "[PySR] accepted", "[Generic SR] accepted",
                "[Dimensional SR] accepted",
            )
        ),
        "derived_law_events": discovery.count("archive as a derived theorem"),
        "basis_exchange_events": discovery.count("basis exchange:"),
        "timeout_events": discovery.count("terminated due to computational limit"),
        "analysis_error_events": discovery.count("Error occurred in analyzing"),
        "contextual_reward_events": discovery.count("Contextual reward:"),
    }


def checkpoint_summary(prefix: Path) -> dict:
    memory_path = Path(str(prefix) + "_memory.json")
    general_path = Path(str(prefix) + "_general_model.json")
    if not memory_path.exists() or not general_path.exists():
        return {"available": False}
    memory = json.loads(memory_path.read_text())
    general = json.loads(general_path.read_text())
    service = general.get("general_service", {})
    contextual = memory.get("contextual_bandit", {})
    return {
        "available": True,
        "epoch": memory.get("epoch", 0),
        "active_laws": len(general.get("general_laws", {})),
        "marked_laws": len(service.get("marked_general_laws", [])),
        "derived_laws": len(service.get("derived_general_laws", {})),
        "independence_certificates": len(
            service.get("independence_certificates", {})
        ),
        "redundant_concepts": len(memory.get("redundant_concepts", [])),
        "contextual_updates": contextual.get("total_updates", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v3-log",
        default="logs/source38_v3_pysr_reducer/full_discovery.log",
    )
    parser.add_argument(
        "--v3-checkpoint",
        default=(
            "data/test_cases/source38_v3_pysr_reducer/"
            "source38_v3_pysr_reducer_run1"
        ),
    )
    parser.add_argument(
        "--baseline-log", default="logs/test_example_1/full_discovery.log"
    )
    args = parser.parse_args()
    report = {
        "v3": summarize_log(Path(args.v3_log)),
        "v3_checkpoint": checkpoint_summary(Path(args.v3_checkpoint)),
        "historical_baseline": summarize_log(Path(args.baseline_log)),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
