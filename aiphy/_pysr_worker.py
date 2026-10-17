"""Standalone PySR worker.

This file is executed by path, not imported as an ``aiphy`` submodule.  PySR
and juliacall therefore initialise before AI-Newton's torch dependency enters
the process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys


def _bind_lifetime_to_parent(expected_parent: int) -> None:
    if sys.platform == "linux":
        import ctypes

        ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
        if os.getppid() != expected_parent:
            raise SystemExit("AI-Newton parent exited before PySR worker startup")


_bind_lifetime_to_parent(int(sys.argv[4]))

from pysr import PySRRegressor  # Must precede any possible torch import.  noqa: E402
import numpy as np


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    config = json.loads(sys.argv[3])
    payload = np.load(input_path)
    matrix = payload["matrix"]
    target = payload["target"]
    variable_names = payload["variable_names"].tolist()
    x_units = payload["x_units"].tolist() if "x_units" in payload else None
    y_unit = str(payload["y_unit"].item()) if "y_unit" in payload else None
    prior_target = payload["prior_target"] if "prior_target" in payload else None

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "exp", "log"],
        maxsize=int(config["max_complexity"]),
        populations=int(config["populations"]),
        niterations=int(config["niterations"]),
        timeout_in_seconds=float(config["timeout_seconds"]),
        model_selection="best",
        parsimony=0.003,
        complexity_of_operators={"sin": 2, "cos": 2, "exp": 3, "log": 3},
        # The exponent branch is kept to one node.  This is PySR's standard
        # interpretable power-law grammar: the base remains unrestricted while
        # the exponent can only be a fitted constant (or a single variable,
        # which AI-Newton rejects during candidate admission).
        constraints={
            "sin": 8, "cos": 8, "exp": 6, "log": 6, "^": (-1, 1)
        },
        nested_constraints={
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
        },
        verbosity=0,
        progress=False,
        batching=False,
        random_state=int(config["random_state"]),
        deterministic=True,
        parallelism="serial",
        output_directory=str(input_path.parent / "pysr_output"),
        temp_equation_file=False,
        delete_tempfiles=True,
        dimensional_constraint_penalty=float(
            config.get("dimensional_constraint_penalty", 1000.0)
        ),
        dimensionless_constants_only=False,
        optimizer_iterations=int(config.get("optimizer_iterations", 20)),
        early_stop_condition=(
            f"(loss, complexity) -> loss < {float(config.get('early_stop_loss', 1e-14))}"
        ),
        warm_start=prior_target is not None,
    )
    fit_kwargs = dict(
        variable_names=variable_names,
        X_units=x_units,
        y_units=y_unit,
    )
    if prior_target is not None:
        final_iterations = model.niterations
        model.niterations = int(config.get("skeleton_prior_iterations", 40))
        model.fit(matrix, prior_target, **fit_kwargs)
        # PySR's supported warm-start path preserves and mutates the existing
        # populations.  Changing only y turns the old law trees into a genuine
        # skeleton prior for the revised observations.
        model.niterations = final_iterations
        model.fit(matrix, target, **fit_kwargs)
    else:
        model.fit(matrix, target, **fit_kwargs)
    equations = [
        {
            "equation": str(row.equation),
            "complexity": int(row.complexity),
            "loss": float(row.loss),
        }
        for row in model.equations_.itertuples()
    ]
    output_path.write_text(json.dumps({"equations": equations}, sort_keys=True))


if __name__ == "__main__":
    main()
