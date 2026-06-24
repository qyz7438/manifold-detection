from __future__ import annotations

import sys

from spectral_detection_posttrain.eval.eval_rerank import main


if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "oracle_ramp"])
    main()
