import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_detection_posttrain.train.action_verifier_posttrain import main_for_experiment


OBJECTIVE = "dpo"
VERIFIER = "manifold"
DEFAULT_RUN_NAME = "round2123_dpo_manifold_s42"


def main() -> None:
    main_for_experiment(OBJECTIVE, VERIFIER, default_run_name=DEFAULT_RUN_NAME)


if __name__ == "__main__":
    main()
