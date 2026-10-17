"""Run the fixed classical-mechanics end-to-end revision benchmarks."""

from __future__ import annotations

import json

from aiphy.classical_revision_experiments import (
    HOOKE_LINEAR_LAW,
    PENDULUM_LINEAR_LAW,
    duffing_revision_spec,
    duffing_revision_training_data,
    duffing_revision_validation_data,
    pendulum_revision_spec,
    pendulum_revision_training_data,
    pendulum_revision_validation_data,
)
from aiphy.pysr_regression import PySRConfig, PySRDiscovery
from aiphy.theory_revision import (
    ActiveTheoryLaw,
    RevisionControllerConfig,
    RevisionTheoryState,
    TheoryRevisionController,
)


def run_case(name, old_formula, target, train, validation, spec, strategies, config):
    old = ActiveTheoryLaw(name, old_formula, target)
    state = RevisionTheoryState.from_law(old)
    outcome = TheoryRevisionController(
        PySRDiscovery(config),
        RevisionControllerConfig(acceptance_nrmse=1e-6),
    ).revise(
        state,
        old.name,
        train,
        validation,
        spec,
        strategies=strategies,
        debug=True,
    )
    return outcome.to_json()


def main():
    pendulum = run_case(
        "small_angle_pendulum",
        PENDULUM_LINEAR_LAW,
        "angularacceleration[1]",
        pendulum_revision_training_data(),
        pendulum_revision_validation_data(),
        pendulum_revision_spec(),
        ("internal",),
        PySRConfig(
            niterations=50,
            populations=6,
            max_complexity=10,
            max_features=6,
            timeout_seconds=50,
            random_state=17,
            max_train_nrmse=1e-6,
            max_holdout_nrmse=1e-6,
            max_holdout_bias=1e-6,
        ),
    )
    duffing = run_case(
        "linear_hooke",
        HOOKE_LINEAR_LAW,
        "acceleration[1]",
        duffing_revision_training_data(),
        duffing_revision_validation_data(),
        duffing_revision_spec(),
        ("additive",),
        PySRConfig(
            niterations=1,
            populations=1,
            max_complexity=12,
            max_features=8,
            timeout_seconds=5,
            random_state=23,
            max_train_nrmse=1e-6,
            max_holdout_nrmse=1e-6,
            max_holdout_bias=1e-6,
        ),
    )
    print(json.dumps({"pendulum": pendulum, "duffing": duffing}, indent=2))


if __name__ == "__main__":
    main()
