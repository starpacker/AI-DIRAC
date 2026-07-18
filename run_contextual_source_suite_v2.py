"""Launch the improved contextual selector on the same original source suite."""

from pathlib import Path

import run_contextual_source_suite as source_suite


source_suite.OUTPUT_DIR = Path("data/test_cases/contextual_source38_v2")
source_suite.CHECKPOINT = source_suite.OUTPUT_DIR / "contextual_source38_v2_run1"


if __name__ == "__main__":
    source_suite.main()
