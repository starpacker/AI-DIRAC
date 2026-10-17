"""Audit or migrate an AI-Newton checkpoint to a local independent-law basis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiphy import Theorist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Checkpoint prefix without _memory.json")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="If provided, prune dependent laws and save a new checkpoint here.",
    )
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.checkpoint.resolve()
    output = args.output_prefix.resolve() if args.output_prefix else None
    if output == source:
        raise ValueError("Refusing to overwrite the source checkpoint; choose a new prefix")

    theorist = Theorist.read_from_file(
        str(source),
        reset_experiment=False,
        num_threads=args.threads,
    )
    before = len(theorist.general.general_laws)
    audit = theorist.general.service.audit_independence()
    dependent = sorted(name for name, item in audit.items() if item["status"] == "dependent")
    independent = sorted(name for name, item in audit.items() if item["status"] == "independent")
    inconclusive = sorted(name for name, item in audit.items() if item["status"] == "inconclusive")
    report = {
        "source": str(source),
        "laws_before": before,
        "predicted_basis_size": before - len(dependent),
        "dependent": dependent,
        "independent": independent,
        "inconclusive": inconclusive,
        "certificates": audit,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        theorist.general.service.prune_dependent_basis()
        theorist.save_to_file(str(output))
        print(
            f"Saved migrated checkpoint with {len(theorist.general.general_laws)} "
            f"basis laws and {len(theorist.general.service.derived_general_laws)} "
            f"derived laws to {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
