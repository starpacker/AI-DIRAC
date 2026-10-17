"""Run a fixed, unattended thermodynamics-only discovery benchmark.

The schedule, observations, operator library and search budget are frozen
before epoch one.  The driver never inspects a discovered formula to decide
what to inject next; it only records each child epoch's exit status and audit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


FUNDAMENTAL_EXPERIMENTS = (
    "epistemic_gas_state_scan",
    "epistemic_gas_isothermal",
    "epistemic_gas_isochoric",
    "epistemic_gas_isobaric",
    "epistemic_gas_adiabatic",
)

DIFFICULT_EXPERIMENTS = (
    "lumped_cooling_static_env",
    "lumped_heating_static_env",
    "lumped_cooling_dynamic_env",
    "lumped_joule_heating",
    "lumped_radiative_cooling",
    "two_body_contact_equal",
    "two_body_contact_unequal",
    "two_body_contact_with_ambient",
    "three_body_chain_relaxation",
    "three_body_ring_relaxation",
    "driven_ambient_oscillation",
    "periodic_power_heating",
    "thermostat_limit_cycle",
    "exothermic_self_heating",
    "quench_recovery",
    "latent_heat_heating_plateau",
    "latent_heat_cooling_plateau",
    "latent_heat_contact_two_body",
    "rod_three_node_diffusion",
    "rod_four_node_diffusion",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name", default="thermal_autonomous_clean_v1"
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--populations", type=int, default=10)
    parser.add_argument("--sr-timeout", type=float, default=120.0)
    parser.add_argument("--epoch-timeout", type=float, default=1200.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / "data" / "test_cases" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "autonomous_progress.json"
    log_path = output / "autonomous_suite.log"
    schedule = [
        *FUNDAMENTAL_EXPERIMENTS,
        *(DIFFICULT_EXPERIMENTS * args.rounds),
    ]
    environment = dict(os.environ)
    environment.update({
        "AI_NEWTON_THERMAL_RUN": args.run_name,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    records = []
    suite_started = time.time()
    with log_path.open("a", buffering=1) as log:
        log.write(json.dumps({
            "event": "suite_start",
            "run_name": args.run_name,
            "schedule": schedule,
            "iterations": args.iterations,
            "populations": args.populations,
            "sr_timeout": args.sr_timeout,
            "epoch_timeout": args.epoch_timeout,
            "time": suite_started,
        }, sort_keys=True) + "\n")
        for index, experiment in enumerate(schedule):
            command = [
                sys.executable,
                str(root / "run_thermal_epistemic_epoch.py"),
                "--experiment", experiment,
                "--iterations", str(args.iterations),
                "--populations", str(args.populations),
                "--sr-timeout", str(args.sr_timeout),
                "--max-complexity", "18",
                "--max-candidates", "8",
                "--timeout", str(args.epoch_timeout),
            ]
            if index == 0:
                command.append("--fresh")
            started = time.time()
            log.write(json.dumps({
                "event": "epoch_start",
                "index": index + 1,
                "experiment": experiment,
                "time": started,
            }, sort_keys=True) + "\n")
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.epoch_timeout + 180.0,
                check=False,
            )
            record = {
                "index": index + 1,
                "experiment": experiment,
                "returncode": completed.returncode,
                "duration_seconds": time.time() - started,
            }
            records.append(record)
            progress_path.write_text(json.dumps({
                "run_name": args.run_name,
                "status": "running",
                "completed": len(records),
                "total": len(schedule),
                "records": records,
                "started": suite_started,
                "updated": time.time(),
            }, indent=2, sort_keys=True))
            log.write(json.dumps({"event": "epoch_end", **record}, sort_keys=True) + "\n")
        progress_path.write_text(json.dumps({
            "run_name": args.run_name,
            "status": "complete",
            "completed": len(records),
            "total": len(schedule),
            "records": records,
            "started": suite_started,
            "updated": time.time(),
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
