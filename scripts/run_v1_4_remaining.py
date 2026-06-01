"""Run V1.5 evaluation."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.benchmark import V1BenchmarkRunner
from crypto_spot_v1.metrics import compute_score
from crypto_spot_v1.evaluation import RESEARCH_MODE, VALID_MODES, normalize_mode, save_evaluation_run

CANDIDATES = ["v2_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(VALID_MODES),
        default=RESEARCH_MODE,
        help="research for fast candidate screening; complete for full diagnostics/audit.",
    )
    return parser.parse_args()


def main(mode: str = RESEARCH_MODE) -> None:
    import time as tmod

    mode = normalize_mode(mode)
    config_path = PROJECT_ROOT / "configs" / "backtest_v1.json"
    output_dir = PROJECT_ROOT / "results"
    runner = V1BenchmarkRunner(str(config_path), str(output_dir))

    for candidate_name in CANDIDATES:
        print(f"\n====== {candidate_name} ======", flush=True)
        t0 = tmod.time()

        results = runner.run_all(candidate_name=candidate_name)
        print(f"  run_all: {tmod.time() - t0:.1f}s", flush=True)

        bh_summary = results.get("buy_hold")
        scores = {name: float(compute_score(s, bh_summary)) for name, s in results.items()}
        print(f"  scores: {scores}", flush=True)

        verdict = runner.check_promotion(results, candidate_name)
        print(f"  verdict score: {verdict.get('candidate_score', 'N/A')}", flush=True)

        timestamp = tmod.strftime("%Y%m%d_%H%M%S")
        save_t0 = tmod.time()
        run_dir = save_evaluation_run(
            runner=runner,
            results=results,
            scores=scores,
            verdict=verdict,
            candidate_name=candidate_name,
            mode=mode,
            timestamp=timestamp,
            config_path=config_path,
            output_root=output_dir,
            diagnostics_enabled=(mode != RESEARCH_MODE),
        )
        print(f"  save_evaluation_run[{mode}]: {tmod.time() - save_t0:.1f}s", flush=True)
        print(f"  run_dir={run_dir}", flush=True)
        print(f"  DONE", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args.mode)
