"""Run one thermodynamics epoch with a strict measurement boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import queue
import time

from aiphy.contextual_bandit import AdaptiveTheoryReward, TheoryStateEncoder
from aiphy.experiment.thermodynamics_epistemic import (
    epistemic_gas_adiabatic_config,
    epistemic_gas_isobaric_config,
    epistemic_gas_isochoric_config,
    epistemic_gas_isothermal_config,
    epistemic_gas_state_scan_config,
)
from aiphy.experiment.challenge_thermodynamics import (
    driven_ambient_oscillation_config,
    exothermic_self_heating_config,
    latent_heat_contact_two_body_config,
    latent_heat_cooling_plateau_config,
    latent_heat_heating_plateau_config,
    lumped_cooling_dynamic_env_config,
    lumped_cooling_static_env_config,
    lumped_heating_static_env_config,
    lumped_joule_heating_config,
    lumped_radiative_cooling_config,
    periodic_power_heating_config,
    quench_recovery_config,
    rod_four_node_diffusion_config,
    rod_three_node_diffusion_config,
    thermostat_limit_cycle_config,
    three_body_chain_relaxation_config,
    three_body_ring_relaxation_config,
    two_body_contact_equal_config,
    two_body_contact_unequal_config,
    two_body_contact_with_ambient_config,
)
from aiphy.main import Theorist, run_method_in_process
from aiphy.operational_vocabulary import (
    load_vocabulary as load_operational_vocabulary,
    update_operational_vocabulary,
)
from run_thermal_manual_epoch import theory_audit


RUN_NAME = os.getenv("AI_NEWTON_THERMAL_RUN", "thermal_epistemic_si_v2")
ROOT = Path("data/test_cases") / RUN_NAME
CHECKPOINT = ROOT / RUN_NAME
AUDIT_DIR = ROOT / "audits"
VOCABULARY = ROOT / "operational_vocabulary.json"

FACTORIES = (
    epistemic_gas_state_scan_config,
    epistemic_gas_isothermal_config,
    epistemic_gas_isochoric_config,
    epistemic_gas_isobaric_config,
    epistemic_gas_adiabatic_config,
    lumped_cooling_static_env_config,
    lumped_heating_static_env_config,
    lumped_cooling_dynamic_env_config,
    lumped_joule_heating_config,
    lumped_radiative_cooling_config,
    two_body_contact_equal_config,
    two_body_contact_unequal_config,
    two_body_contact_with_ambient_config,
    three_body_chain_relaxation_config,
    three_body_ring_relaxation_config,
    driven_ambient_oscillation_config,
    periodic_power_heating_config,
    thermostat_limit_cycle_config,
    exothermic_self_heating_config,
    quench_recovery_config,
    latent_heat_heating_plateau_config,
    latent_heat_cooling_plateau_config,
    latent_heat_contact_two_body_config,
    rod_three_node_diffusion_config,
    rod_four_node_diffusion_config,
)
RAW_OBSERVATIONS = {
    factory().exp_name: sorted({
        concept.atomexp_name for concept in factory().original_concept
    })
    for factory in FACTORIES
}


def checkpoint_exists() -> bool:
    return Path(str(CHECKPOINT) + "_memory.json").exists()


def new_theorist() -> Theorist:
    theorist = Theorist(init_time_limit=180, num_threads=8)
    for factory in FACTORIES:
        theorist.register_experiment(factory())
    return theorist


def load_vocabulary() -> dict:
    return load_operational_vocabulary(VOCABULARY)


def selected_observations(theorist: Theorist, experiment: str) -> list[str]:
    selected = list(RAW_OBSERVATIONS[experiment])
    vocabulary = load_vocabulary()
    for name in sorted({
        *vocabulary.get("trial", {}),
        *vocabulary.get("promoted", {}),
    }):
        if name not in theorist.knowledge.fetch_concepts:
            continue
        try:
            if theorist.knowledge.specialize_concept(name, experiment):
                selected.append(name)
        except BaseException:
            continue
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(RAW_OBSERVATIONS), required=True)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--iterations", type=int, default=48)
    parser.add_argument("--populations", type=int, default=6)
    parser.add_argument("--sr-timeout", type=float, default=120.0)
    parser.add_argument("--max-complexity", type=int, default=18)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--no-feature-subsets", action="store_true")
    args = parser.parse_args()

    os.environ["AI_NEWTON_PYSR_TARGETS"] = "1"
    os.environ["AI_NEWTON_PYSR_ITERATIONS"] = str(args.iterations)
    os.environ["AI_NEWTON_PYSR_POPULATIONS"] = str(args.populations)
    os.environ["AI_NEWTON_PYSR_TIMEOUT_SECONDS"] = str(args.sr_timeout)
    os.environ["AI_NEWTON_PYSR_MAX_COMPLEXITY"] = str(args.max_complexity)
    os.environ["AI_NEWTON_PYSR_MAX_CANDIDATES"] = str(args.max_candidates)
    os.environ["AI_NEWTON_PYSR_FEATURE_SUBSETS"] = (
        "0" if args.no_feature_subsets else "1"
    )
    # 8.314 = 4157/500.  This affects only post-search constant
    # rationalisation, not the candidate structures available to PySR.
    os.environ["AI_NEWTON_PYSR_MAX_DENOMINATOR"] = "1000"
    ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh or not checkpoint_exists():
        theorist = new_theorist()
        VOCABULARY.write_text(json.dumps(
            {"trial": {}, "promoted": {}, "rejected": {}}, indent=2
        ))
        theorist.save_to_file(str(CHECKPOINT))
    else:
        theorist = Theorist.read_from_file(
            str(CHECKPOINT), reset_experiment=True, num_threads=8
        )

    observations = selected_observations(theorist, args.experiment)
    before = TheoryStateEncoder.capture(theorist)
    theorist.memory.epoch += 1
    epoch = theorist.memory.epoch
    sr_audit_path = AUDIT_DIR / (
        f"epoch_{epoch:03d}_{args.experiment}_sr.jsonl"
    )
    if sr_audit_path.exists():
        sr_audit_path.unlink()
    os.environ["AI_NEWTON_PYSR_AUDIT_PATH"] = str(sr_audit_path)
    result_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    shared = manager.list([])
    process = multiprocessing.Process(
        target=run_method_in_process,
        args=(
            theorist, "theoretical_analysis", result_queue, shared,
            args.experiment, "pysr", observations, 8,
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

    vocabulary_changes = {"trial_added": [], "promoted": []}
    if status == "success":
        current_laws = sorted(
            set(theorist.general.general_laws) - set(before.general_laws)
        )
        try:
            vocabulary_changes = update_operational_vocabulary(
                theorist, VOCABULARY, args.experiment, current_laws
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            vocabulary_changes = {
                "trial_added": [],
                "promoted": [],
                "error": f"{type(error).__name__}: {error}",
            }
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
        "raw_observations": RAW_OBSERVATIONS[args.experiment],
        "selected_observations": observations,
        "latent_quantities": ["amount", "gasconstant", "gamma"],
        "legacy_reward": legacy_reward,
        "adaptive_audit_reward": asdict(reward),
        "new_concepts": sorted(set(after.concepts) - set(before.concepts)),
        "new_general_laws": sorted(set(after.general_laws) - set(before.general_laws)),
        "removed_general_laws": sorted(set(before.general_laws) - set(after.general_laws)),
        "operational_vocabulary": load_vocabulary(),
        "operational_vocabulary_changes": vocabulary_changes,
        "symbolic_regression_audit": str(sr_audit_path),
        "theory": theory_audit(theorist, args.experiment),
    }
    audit_path = AUDIT_DIR / f"epoch_{epoch:03d}_{args.experiment}.json"
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"AUDIT_PATH={audit_path}", flush=True)


if __name__ == "__main__":
    main()
