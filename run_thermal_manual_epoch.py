"""Run and audit exactly one manually selected thermodynamics epoch."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import queue
import time

import sympy as sp

from aiphy.contextual_bandit import AdaptiveTheoryReward, TheoryStateEncoder
from aiphy.experiment.thermodynamics_state import (
    ideal_gas_adiabatic_config,
    ideal_gas_isobaric_config,
    ideal_gas_isochoric_config,
    ideal_gas_isothermal_config,
    ideal_gas_state_scan_config,
)
from aiphy.main import Theorist, run_method_in_process


ROOT = Path("data/test_cases/thermal_manual_v1")
CHECKPOINT = ROOT / "thermal_manual_v1"
AUDIT_DIR = ROOT / "audits"

FACTORIES = (
    ideal_gas_state_scan_config,
    ideal_gas_isothermal_config,
    ideal_gas_isochoric_config,
    ideal_gas_isobaric_config,
    ideal_gas_adiabatic_config,
)

OBSERVATIONS = {
    "ideal_gas_state_scan": [
        "pressure", "volume", "amount", "temperature", "gasconstant", "gamma", "t"
    ],
    "ideal_gas_isothermal": [
        "pressure", "volume", "amount", "temperature", "gasconstant", "t"
    ],
    "ideal_gas_isochoric": [
        "pressure", "volume", "amount", "temperature", "gasconstant", "t"
    ],
    "ideal_gas_isobaric": [
        "pressure", "volume", "amount", "temperature", "gasconstant", "t"
    ],
    "ideal_gas_adiabatic": ["pressure", "volume", "gamma", "t"],
}


def checkpoint_exists() -> bool:
    return Path(str(CHECKPOINT) + "_memory.json").exists()


def new_theorist() -> Theorist:
    theorist = Theorist(init_time_limit=180, num_threads=8)
    for factory in FACTORIES:
        theorist.register_experiment(factory())
    return theorist


def raw_proposition(theorist: Theorist, prop, experiment: str) -> list[str]:
    model = theorist.specific[experiment]
    result = []
    concept = prop.unwrap_concept
    for expression in theorist.knowledge.specialize(concept, experiment) or []:
        try:
            raw = model._sympy_of_raw_defi(model.expand_exp(expression)).doit()
            result.append(str(sp.factor_terms(raw)))
        except BaseException as error:
            result.append(f"<expansion failed: {type(error).__name__}: {error}>")
    return result


def raw_specific_proposition(theorist: Theorist, prop, experiment: str) -> list[str]:
    model = theorist.specific[experiment]
    try:
        raw = model._sympy_of_raw_defi(model.expand_exp(prop.unwrap_exp)).doit()
        return [str(sp.factor_terms(raw))]
    except BaseException as error:
        return [f"<expansion failed: {type(error).__name__}: {error}>"]


def theory_audit(theorist: Theorist, experiment: str) -> dict:
    laws = {}
    for name, law in sorted(theorist.general.general_laws.items()):
        expansion_experiment = (
            experiment if experiment in law.valid_experiments
            else law.valid_experiments[0] if law.valid_experiments else experiment
        )
        laws[name] = {
            "prop": str(law.prop),
            "raw": raw_proposition(theorist, law.prop, expansion_experiment),
            "type": law.type,
            "valid_experiments": sorted(law.valid_experiments),
            "father": law.father_name,
            "scope_kind": law.scope_kind,
            "certificate": theorist.general.service.independence_certificates.get(name),
        }
    model = theorist.specific[experiment]
    derived = {}
    for name, archived in sorted(
        theorist.general.service.derived_general_laws.items()
    ):
        record = dict(archived)
        try:
            from aiphy.interface import Proposition
            record["raw"] = raw_proposition(
                theorist,
                Proposition(archived["prop"]),
                experiment,
            )
        except BaseException as error:
            record["raw"] = [
                f"<expansion failed: {type(error).__name__}: {error}>"
            ]
        derived[name] = record
    specific = {}
    for name in sorted(model.conclusions.keys()):
        prop = model.conclusions.get(name)
        specific[name] = {
            "prop": str(prop),
            "raw": raw_specific_proposition(theorist, prop, experiment),
        }
    return {
        "active_general_laws": laws,
        "derived_general_laws": derived,
        "specific_conclusions": specific,
        "active_concepts": sorted(theorist.knowledge.fetch_concepts.keys()),
        "derived_concepts": sorted(theorist.general.service.derived_concepts),
        "selectable_concepts": sorted(theorist.memory.action_bandit.actions),
        "redundant_concepts": sorted(theorist.memory.redundant_concepts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(OBSERVATIONS), required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    # One pressure target keeps a manual epoch bounded and does not encode its
    # formula.  Types/units and generic operators remain the only SR priors.
    os.environ.setdefault("AI_NEWTON_PYSR_TARGETS", "1")
    ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh or not checkpoint_exists():
        theorist = new_theorist()
        theorist.save_to_file(str(CHECKPOINT))
    else:
        theorist = Theorist.read_from_file(
            str(CHECKPOINT), reset_experiment=True, num_threads=8
        )

    before = TheoryStateEncoder.capture(theorist)
    theorist.memory.epoch += 1
    epoch = theorist.memory.epoch
    result_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    shared = manager.list([])
    process = multiprocessing.Process(
        target=run_method_in_process,
        args=(
            theorist, "theoretical_analysis", result_queue, shared,
            args.experiment, "pysr", OBSERVATIONS[args.experiment], 8,
        ),
    )
    started = time.monotonic()
    process.start()
    process.join(args.timeout)
    duration = time.monotonic() - started
    status = "success"
    legacy_reward = 0
    if process.is_alive():
        process.terminate()
        process.join()
        status = "timeout"
    else:
        try:
            message = result_queue.get(timeout=5.0)
            updated, legacy_reward = message[0]
            if updated is None:
                status = "error"
            else:
                theorist = updated
        except (queue.Empty, EOFError, BrokenPipeError, IndexError, TypeError):
            status = "error"
    manager.shutdown()

    after = TheoryStateEncoder.capture(theorist)
    reward = AdaptiveTheoryReward().evaluate(
        before, after,
        outcome="success" if status == "success" else status,
        duration_seconds=duration,
        legacy_reward=float(legacy_reward),
    )
    if status == "success":
        theorist.save_to_file(str(CHECKPOINT))
    report = {
        "epoch": epoch,
        "experiment": args.experiment,
        "status": status,
        "duration_seconds": duration,
        "selected_observations": OBSERVATIONS[args.experiment],
        "legacy_reward": legacy_reward,
        "adaptive_audit_reward": asdict(reward),
        "new_concepts": sorted(set(after.concepts) - set(before.concepts)),
        "new_general_laws": sorted(
            set(after.general_laws) - set(before.general_laws)
        ),
        "removed_general_laws": sorted(
            set(before.general_laws) - set(after.general_laws)
        ),
        "theory": theory_audit(theorist, args.experiment),
    }
    audit_path = AUDIT_DIR / f"epoch_{epoch:03d}_{args.experiment}.json"
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"AUDIT_PATH={audit_path}", flush=True)


if __name__ == "__main__":
    main()
