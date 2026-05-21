"""HTML report generation for V1 backtest summaries."""

from __future__ import annotations

from html import escape

from .metrics import StrategySummary, compute_score_components


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _row(cells: list[str], tag: str = "td") -> str:
    body = "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
    return f"<tr>{body}</tr>"


def generate_html(
    results: dict[str, StrategySummary],
    scores: dict[str, float],
    verdict: dict,
    candidate_name: str | None = None,
    config: dict | None = None,
) -> str:
    candidate_name = candidate_name or "v1"
    candidate = results.get(candidate_name)
    buy_hold = results.get("buy_hold")
    config = config or {}

    sections = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Crypto Spot V1 Backtest Report</title>",
        _style(),
        "</head><body><main>",
        "<h1>Crypto Spot V1 Backtest Report</h1>",
        _config_section(config),
        _summary_section(results, scores),
    ]

    if candidate:
        sections.append(_candidate_section(candidate, buy_hold, scores.get(candidate_name)))
        sections.append(_score_section(candidate, buy_hold))
        sections.append(_window_section(candidate))

    sections.extend(["</main></body></html>"])
    return "\n".join(section for section in sections if section)


def _style() -> str:
    return """
<style>
body { margin: 0; font-family: Segoe UI, Arial, sans-serif; color: #172033; background: #f6f8fb; }
main { max-width: 1180px; margin: 0 auto; padding: 28px; }
h1 { margin: 0 0 18px; font-size: 28px; }
h2 { margin: 28px 0 12px; font-size: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }
.stat { background: white; border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; }
.label { color: #66758c; font-size: 12px; margin-bottom: 5px; }
.value { font-weight: 700; font-size: 17px; }
table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #dbe3ef; }
th, td { padding: 9px 10px; border-bottom: 1px solid #e8eef6; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #edf3fb; color: #42536b; font-size: 12px; text-transform: uppercase; }
tr:last-child td { border-bottom: 0; }
</style>
"""


def _config_section(config: dict) -> str:
    execution = config.get("execution", {}).get("mode", "next_open")
    symbols = ", ".join(config.get("symbols", []))
    windows = ", ".join(f"{w.get('name', w.get('days'))}" for w in config.get("windows", []))
    fee = config.get("cost", {}).get("fee_rate", 0.0)
    warmup = config.get("warmup_bars", "")
    return f"""
<h2>Configuration</h2>
<div class='grid'>
  <div class='stat'><div class='label'>Symbols</div><div class='value'>{escape(symbols)}</div></div>
  <div class='stat'><div class='label'>Timeframe</div><div class='value'>{escape(str(config.get('timeframe', '')))}</div></div>
  <div class='stat'><div class='label'>Execution</div><div class='value'>{escape(str(execution))}</div></div>
  <div class='stat'><div class='label'>Warmup Bars</div><div class='value'>{escape(str(warmup))}</div></div>
  <div class='stat'><div class='label'>Fee Rate</div><div class='value'>{_pct(fee)}</div></div>
  <div class='stat'><div class='label'>Windows</div><div class='value'>{escape(windows)}</div></div>
</div>
"""


def _summary_section(results: dict[str, StrategySummary], scores: dict[str, float]) -> str:
    rows = [
        _row([
            "Strategy",
            "Score",
            "Windows",
            "Mean Return",
            "Mean Excess",
            "Win vs BH",
            "Max Drawdown",
            "Trades",
            "Exposure",
            "Turnover",
        ], "th")
    ]
    for name, summary in results.items():
        rows.append(_row([
            escape(name),
            _score(scores.get(name)),
            str(summary.total_window_count()),
            _pct(summary.mean_return()),
            _pct(summary.mean_excess_return()),
            _pct(summary.win_rate_vs_bh()),
            _pct(summary.mean_max_drawdown()),
            _num(summary.mean_trade_count(), 2),
            _pct(summary.mean_exposure()),
            _num(summary.mean_turnover(), 2),
        ]))
    return "<h2>Summary</h2><table>" + "".join(rows) + "</table>"


def _candidate_section(
    candidate: StrategySummary,
    buy_hold: StrategySummary | None,
    score: float | None,
) -> str:
    return f"""
<h2>V1 vs Buy & Hold</h2>
<div class='grid'>
  <div class='stat'><div class='label'>Score</div><div class='value'>{_score(score)}</div></div>
  <div class='stat'><div class='label'>Retention Ratio</div><div class='value'>{_num(candidate.retention_ratio(buy_hold), 3)}</div></div>
  <div class='stat'><div class='label'>Drawdown Reduction</div><div class='value'>{_pct(candidate.drawdown_reduction(buy_hold))}</div></div>
  <div class='stat'><div class='label'>Median Excess</div><div class='value'>{_pct(candidate.median_excess_return())}</div></div>
</div>
"""


def _score_section(candidate: StrategySummary, buy_hold: StrategySummary | None) -> str:
    components = compute_score_components(candidate, buy_hold)
    rows = [_row(["Component", "Score", "Weight", "Weighted"], "th")]
    for name, component in components.items():
        rows.append(_row([
            escape(name),
            _num(component["score"], 4),
            _pct(component["weight"]),
            _num(component["weighted"], 4),
        ]))
    return "<h2>Score Components</h2><table>" + "".join(rows) + "</table>"


def _window_section(candidate: StrategySummary) -> str:
    rows = [_row([
        "Symbol",
        "Window Set",
        "Window",
        "Return",
        "Buy & Hold",
        "Excess",
        "Max Drawdown",
        "Trades",
        "Exposure",
        "Turnover",
    ], "th")]
    for perf in candidate.perfs:
        for window in perf.windows:
            rows.append(_row([
                escape(perf.symbol),
                escape(perf.window_label),
                escape(window.window_label),
                _pct(window.total_return),
                _pct(window.buy_hold_return),
                _pct(window.excess_return),
                _pct(window.max_drawdown),
                str(window.trade_count),
                _pct(window.avg_exposure),
                _num(window.turnover, 2),
            ]))
    return "<h2>Rolling Windows</h2><table>" + "".join(rows) + "</table>"
