"""Run v3 on the source-38 suite with the original UCB selection policy.

The only active algorithmic changes relative to the baseline arm are:
1. PySR-backed symbolic-regression proposals.
2. Probabilistic/exact minimum-basis filtering of concepts and general laws.

Contextual experiment selection, adaptive theory reward, and the fixed
revision benchmark experiments are deliberately not enabled here.
"""

from pathlib import Path

import run_contextual_source_suite as source_suite


source_suite.OUTPUT_DIR = Path("data/test_cases/source38_v3_pysr_reducer")
source_suite.CHECKPOINT = (
    source_suite.OUTPUT_DIR / "source38_v3_pysr_reducer_run1"
)
source_suite.EXP_SELECT_MODE = "ucb"
source_suite.REGRESSION_BACKEND = "pysr"
# PySR normally needs tens of seconds after Julia starts.  All budget growth,
# feasible-experiment filtering, UCB selection, concept-group selection, and
# checkpoint cadence otherwise remain on the baseline path.
source_suite.INIT_TIME_LIMIT = 90


if __name__ == "__main__":
    source_suite.main()
