#!/usr/bin/env python3
"""Run the clean V1 baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1.benchmark import V1BenchmarkRunner
from crypto_spot_v1.report import generate_html
from crypto_spot_v1.results_io import save_structured_results


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "backtest_v1.json"
    output_dir = PROJECT_ROOT / "results"
    runner = V1BenchmarkRunner(str(config_path), output_dir=str(output_dir))

    results = runner.run_all("v1")
    scores = runner.score_all(results)
    verdict = runner.check_promotion(results, "v1")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    structured_dir = save_structured_results(
        results=results,
        scores=scores,
        verdict=verdict,
        config=runner.config,
        output_dir=output_dir,
        candidate_name="v1",
        timestamp=ts,
        artifacts=runner.artifacts,
    )

    html = generate_html(
        results=results,
        scores=scores,
        verdict=verdict,
        candidate_name="v1",
        config=runner.config,
    )
    report_path = output_dir / f"backtest_report_v1_{ts}.html"
    report_path.write_text(html, encoding="utf-8")

    print("V1 baseline complete")
    for name, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: score={score:.4f}")
    print(f"structured_dir={structured_dir}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
