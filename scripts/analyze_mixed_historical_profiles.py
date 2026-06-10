#!/usr/bin/env python3
"""Profile historical MIXED regimes from DB data before changing rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crypto_spot_v1 import strategy_utils  # noqa: E402
from crypto_spot_v1.database import load_candles_from_db  # noqa: E402


PAIRS = ("BTC/USDT", "ETH/USDT", "BNB/USDT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument(
        "--test-years",
        type=int,
        default=2,
        help="Hold out the latest N years as the test period.",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--run-id", default="mixed_historical_profiles_db_20260608")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed = build_dataset(args.exchange, args.timeframe)
    train, test, split_ts = split_train_test(mixed, args.test_years)
    profile = summarize(mixed, ["profile"])
    btc_profile = summarize(mixed, ["profile", "btc_regime"])
    rank_profile = summarize(mixed, ["profile", "rank_band"])
    momentum_profile = summarize(mixed, ["profile", "momentum_band"])
    volume_profile = summarize(mixed, ["profile", "volume_band"])
    btc_signal = summarize(mixed, ["btc_signal"])
    universal = build_universal_table(mixed)
    universal_train = build_universal_table(train)
    universal_test = build_universal_table(test)
    universal_oos = compare_universal_train_test(universal_train, universal_test)
    profile_oos = compare_summary_train_test(summarize(train, ["profile"]), summarize(test, ["profile"]), ["profile"])
    btc_signal_oos = compare_summary_train_test(
        summarize(train, ["btc_signal"]),
        summarize(test, ["btc_signal"]),
        ["btc_signal"],
    )

    mixed.to_csv(output_dir / "mixed_historical_bars.csv", index=False)
    train.to_csv(output_dir / "mixed_historical_train_bars.csv", index=False)
    test.to_csv(output_dir / "mixed_historical_test_bars.csv", index=False)
    profile.to_csv(output_dir / "mixed_profile_summary.csv", index=False)
    btc_profile.to_csv(output_dir / "mixed_profile_by_btc_regime.csv", index=False)
    rank_profile.to_csv(output_dir / "mixed_profile_by_rank.csv", index=False)
    momentum_profile.to_csv(output_dir / "mixed_profile_by_momentum.csv", index=False)
    volume_profile.to_csv(output_dir / "mixed_profile_by_volume.csv", index=False)
    btc_signal.to_csv(output_dir / "mixed_btc_signal_summary.csv", index=False)
    universal.to_csv(output_dir / "mixed_universal_candidates.csv", index=False)
    universal_train.to_csv(output_dir / "mixed_universal_candidates_train.csv", index=False)
    universal_test.to_csv(output_dir / "mixed_universal_candidates_test.csv", index=False)
    universal_oos.to_csv(output_dir / "mixed_universal_candidates_oos.csv", index=False)
    profile_oos.to_csv(output_dir / "mixed_profile_oos.csv", index=False)
    btc_signal_oos.to_csv(output_dir / "mixed_btc_signal_oos.csv", index=False)
    report = render_report(
        output_dir,
        mixed,
        profile,
        btc_profile,
        rank_profile,
        volume_profile,
        btc_signal,
        universal,
        split_ts=split_ts,
        train=train,
        test=test,
        universal_oos=universal_oos,
        profile_oos=profile_oos,
        btc_signal_oos=btc_signal_oos,
    )
    (output_dir / "mixed_historical_profiles_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Wrote {output_dir}")


def split_train_test(frame: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    end_ts = frame["timestamp"].max()
    split_ts = end_ts - pd.DateOffset(years=test_years)
    train = frame[frame["timestamp"] < split_ts].copy()
    test = frame[frame["timestamp"] >= split_ts].copy()
    return train, test, split_ts


def build_dataset(exchange: str, timeframe: str) -> pd.DataFrame:
    frames = {pair: load_pair(exchange, pair, timeframe) for pair in PAIRS}
    btc = frames["BTC/USDT"].copy()
    btc["btc_regime"] = strategy_utils.compute_btc_regime(btc)
    btc["btc_price_vs_ema72"] = btc["close"] / btc["ema72"] - 1.0
    btc["btc_price_vs_ema168"] = btc["close"] / btc["ema168"] - 1.0
    btc_features = btc[
        [
            "timestamp",
            "btc_regime",
            "btc_price_vs_ema72",
            "btc_price_vs_ema168",
            "ema24_slope",
            "ema168_slope",
            "rolling_365d_pos",
            "donchian_pos",
            "roc_20",
            "volume_strength",
        ]
    ].rename(
        columns={
            "ema24_slope": "btc_ema24_slope",
            "ema168_slope": "btc_ema168_slope",
            "rolling_365d_pos": "btc_rolling_365d_pos",
            "donchian_pos": "btc_donchian_pos",
            "roc_20": "btc_roc_20",
            "volume_strength": "btc_volume_strength",
        }
    )

    rows = []
    for pair, frame in frames.items():
        merged = pd.merge_asof(
            frame.sort_values("timestamp"),
            btc_features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged["pair"] = pair
        merged["market_state"] = merged.apply(strategy_utils.detect_market_state, axis=1)
        merged = add_features(merged)
        merged = add_forward_returns(merged)
        mixed = merged[merged["market_state"].eq("MIXED") & merged["fwd_ret_60d"].notna()].copy()
        mixed["profile"] = mixed.apply(label_profile, axis=1)
        mixed["btc_signal"] = mixed.apply(label_btc_signal, axis=1)
        mixed["rank_band"] = pd.cut(
            mixed["rolling_365d_pos"],
            [-np.inf, 0.25, 0.55, 0.75, np.inf],
            labels=["low_rank", "mid_rank", "high_rank", "top_rank"],
        )
        mixed["momentum_band"] = pd.cut(
            mixed["roc_20"],
            [-np.inf, -0.08, 0.0, 0.08, np.inf],
            labels=["sharp_down", "weak_down", "weak_up", "strong_up"],
        )
        mixed["volume_band"] = pd.cut(
            mixed["volume_strength"],
            [-np.inf, 0.85, 1.15, np.inf],
            labels=["low_vol", "normal_vol", "high_vol"],
        )
        rows.append(mixed)
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "pair"]).reset_index(drop=True)


def load_pair(exchange: str, pair: str, timeframe: str) -> pd.DataFrame:
    frame = load_candles_from_db(exchange, pair, timeframe).sort_values("timestamp").reset_index(drop=True)
    return strategy_utils.compute_indicators(frame)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["price_vs_ema24"] = out["close"] / out["ema24"] - 1.0
    out["price_vs_ema72"] = out["close"] / out["ema72"] - 1.0
    out["price_vs_ema168"] = out["close"] / out["ema168"] - 1.0
    out["ema24_vs_ema72"] = out["ema24"] / out["ema72"] - 1.0
    out["ema72_vs_ema168"] = out["ema72"] / out["ema168"] - 1.0
    out["volume_strength"] = out.get("volume_strength", pd.Series(1.0, index=out.index)).fillna(1.0)
    return out


def add_forward_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["close"].to_numpy(dtype=float)
    for horizon in (30, 60, 90):
        out[f"fwd_ret_{horizon}d"] = out["close"].shift(-horizon) / out["close"] - 1.0
        down = []
        for idx, price in enumerate(close):
            end = min(len(close), idx + horizon + 1)
            if end <= idx + 1:
                down.append(np.nan)
            else:
                down.append(np.nanmin(close[idx + 1 : end]) / price - 1.0)
        out[f"fwd_down_{horizon}d"] = down
    return out


def label_profile(row: pd.Series) -> str:
    close = row["close"]
    ema72 = row["ema72"]
    ema168 = row["ema168"]
    ema168_slope = row["ema168_slope"]
    if close > ema72 and ema72 <= ema168:
        return "RECLAIM_EMA72_LONG_DOWN"
    if close < ema168 and ema72 > ema168:
        return "LOST_EMA168_LONG_UP"
    if close > ema72 and ema72 > ema168:
        return "ABOVE_EMA72_LONG_UP"
    if ema168 < close <= ema72 and ema72 > ema168:
        return "PULLBACK_ABOVE_EMA168"
    if close < ema168 and ema72 <= ema168:
        return "BELOW_EMA168_LONG_DOWN"
    if close > ema168 and ema72 <= ema168 and ema168_slope >= 0:
        return "EARLY_REPAIR"
    return "OTHER_MIXED"


def label_btc_signal(row: pd.Series) -> str:
    regime = str(row.get("btc_regime", "RANGE"))
    if regime in {"STRONG_BULL", "BULL"}:
        if row.get("btc_price_vs_ema72", 0.0) >= 0 and row.get("btc_ema168_slope", 0.0) >= 0:
            return "BTC_SUPPORTIVE"
        return "BTC_MIXED_UP"
    if regime == "BEAR":
        return "BTC_BEAR"
    if row.get("btc_price_vs_ema72", 0.0) >= 0 and row.get("btc_roc_20", 0.0) >= 0:
        return "BTC_RANGE_IMPROVING"
    return "BTC_RANGE_WEAK"


def summarize(frame: pd.DataFrame, group_cols: list[str], min_count: int = 12) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)
    result = grouped.agg(
        count=("pair", "size"),
        ret30_med=("fwd_ret_30d", "median"),
        ret60_med=("fwd_ret_60d", "median"),
        ret90_med=("fwd_ret_90d", "median"),
        pos60_rate=("fwd_ret_60d", lambda s: float((s > 0).mean())),
        down60_med=("fwd_down_60d", "median"),
        btc_bear_rate=("btc_regime", lambda s: float((s == "BEAR").mean())),
        rank_med=("rolling_365d_pos", "median"),
        volume_med=("volume_strength", "median"),
    ).reset_index()
    result = result[result["count"] >= min_count].copy()
    return result.sort_values(["ret60_med", "pos60_rate"], ascending=[False, False]).reset_index(drop=True)


def build_universal_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = {
        "supportive_reclaim": (
            frame["profile"].eq("RECLAIM_EMA72_LONG_DOWN")
            & frame["btc_signal"].isin(["BTC_SUPPORTIVE", "BTC_RANGE_IMPROVING"])
        ),
        "bear_reclaim_trap": (
            frame["profile"].eq("RECLAIM_EMA72_LONG_DOWN")
            & frame["btc_signal"].eq("BTC_BEAR")
        ),
        "top_chase_risk": (
            frame["profile"].eq("ABOVE_EMA72_LONG_UP")
            & frame["rolling_365d_pos"].ge(0.75)
        ),
        "pullback_risk": frame["profile"].eq("PULLBACK_ABOVE_EMA168"),
        "low_rank_repair": (
            frame["rolling_365d_pos"].le(0.25)
            & frame["btc_signal"].isin(["BTC_SUPPORTIVE", "BTC_RANGE_IMPROVING"])
            & frame["price_vs_ema72"].ge(0)
        ),
        "panic_repair_candidate": (
            frame["profile"].eq("LOST_EMA168_LONG_UP")
            & frame["roc_20"].ge(0)
            & frame["btc_signal"].ne("BTC_BEAR")
        ),
        "downtrend_no_reclaim": (
            frame["profile"].eq("BELOW_EMA168_LONG_DOWN")
            & frame["btc_signal"].isin(["BTC_BEAR", "BTC_RANGE_WEAK"])
        ),
        "high_volume_reclaim": (
            frame["profile"].eq("RECLAIM_EMA72_LONG_DOWN")
            & frame["volume_strength"].ge(1.15)
        ),
        "high_volume_pullback": (
            frame["profile"].eq("PULLBACK_ABOVE_EMA168")
            & frame["volume_strength"].ge(1.15)
        ),
    }
    for name, mask in rules.items():
        group = frame[mask].copy()
        if group.empty:
            continue
        rows.append({
            "candidate_label": name,
            "count": len(group),
            "ret30_med": group["fwd_ret_30d"].median(),
            "ret60_med": group["fwd_ret_60d"].median(),
            "ret90_med": group["fwd_ret_90d"].median(),
            "pos60_rate": float((group["fwd_ret_60d"] > 0).mean()),
            "down60_med": group["fwd_down_60d"].median(),
            "pairs": ",".join(sorted(group["pair"].unique())),
            "btc_bear_rate": float((group["btc_regime"] == "BEAR").mean()),
            "rank_med": group["rolling_365d_pos"].median(),
            "volume_med": group["volume_strength"].median(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["ret60_med", "pos60_rate"], ascending=[False, False]).reset_index(drop=True)


def compare_universal_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    if train.empty or test.empty:
        return pd.DataFrame()
    cols = [
        "candidate_label",
        "count",
        "ret60_med",
        "pos60_rate",
        "down60_med",
        "btc_bear_rate",
        "rank_med",
        "volume_med",
    ]
    merged = train[cols].merge(test[cols], on="candidate_label", suffixes=("_train", "_test"))
    merged["ret60_med_delta"] = merged["ret60_med_test"] - merged["ret60_med_train"]
    merged["pos60_rate_delta"] = merged["pos60_rate_test"] - merged["pos60_rate_train"]
    merged["direction_stable"] = (
        ((merged["ret60_med_train"] > 0) & (merged["ret60_med_test"] > 0))
        | ((merged["ret60_med_train"] < 0) & (merged["ret60_med_test"] < 0))
    )
    return merged.sort_values(["direction_stable", "ret60_med_test"], ascending=[False, False]).reset_index(drop=True)


def compare_summary_train_test(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if train.empty or test.empty:
        return pd.DataFrame()
    cols = [*keys, "count", "ret60_med", "pos60_rate", "down60_med", "btc_bear_rate", "rank_med", "volume_med"]
    merged = train[cols].merge(test[cols], on=keys, suffixes=("_train", "_test"))
    merged["ret60_med_delta"] = merged["ret60_med_test"] - merged["ret60_med_train"]
    merged["pos60_rate_delta"] = merged["pos60_rate_test"] - merged["pos60_rate_train"]
    merged["direction_stable"] = (
        ((merged["ret60_med_train"] > 0) & (merged["ret60_med_test"] > 0))
        | ((merged["ret60_med_train"] < 0) & (merged["ret60_med_test"] < 0))
    )
    return merged.sort_values(["direction_stable", "ret60_med_test"], ascending=[False, False]).reset_index(drop=True)


def render_report(
    output_dir: Path,
    mixed: pd.DataFrame,
    profile: pd.DataFrame,
    btc_profile: pd.DataFrame,
    rank_profile: pd.DataFrame,
    volume_profile: pd.DataFrame,
    btc_signal: pd.DataFrame,
    universal: pd.DataFrame,
    *,
    split_ts: pd.Timestamp,
    train: pd.DataFrame,
    test: pd.DataFrame,
    universal_oos: pd.DataFrame,
    profile_oos: pd.DataFrame,
    btc_signal_oos: pd.DataFrame,
) -> str:
    cols = ["count", "ret30_med", "ret60_med", "ret90_med", "pos60_rate", "down60_med", "btc_bear_rate", "rank_med", "volume_med"]
    return "\n".join([
        "# MIXED Historical Profiles",
        "",
        f"- Output: `{output_dir}`",
        f"- Rows: `{len(mixed)}`",
        f"- Date range: `{mixed['timestamp'].min()}` to `{mixed['timestamp'].max()}`",
        f"- Train rows: `{len(train)}` before `{split_ts}`",
        f"- Test rows: `{len(test)}` from `{split_ts}`",
        "",
        "## Profile Summary",
        "",
        profile[["profile", *cols]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## BTC Signal Summary",
        "",
        btc_signal[["btc_signal", *cols]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Universal Candidate Labels",
        "",
        universal.to_markdown(index=False, floatfmt=".4f") if not universal.empty else "No universal labels.",
        "",
        "## Universal Train/Test Check",
        "",
        universal_oos.to_markdown(index=False, floatfmt=".4f") if not universal_oos.empty else "No train/test labels.",
        "",
        "## Profile Train/Test Check",
        "",
        profile_oos.to_markdown(index=False, floatfmt=".4f") if not profile_oos.empty else "No profile train/test comparison.",
        "",
        "## BTC Signal Train/Test Check",
        "",
        btc_signal_oos.to_markdown(index=False, floatfmt=".4f") if not btc_signal_oos.empty else "No BTC signal train/test comparison.",
        "",
        "## Best Profile x BTC Regime",
        "",
        btc_profile[["profile", "btc_regime", *cols]].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Profile x Rank",
        "",
        rank_profile[["profile", "rank_band", *cols]].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Profile x Volume",
        "",
        volume_profile[["profile", "volume_band", *cols]].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
    ])


if __name__ == "__main__":
    main()
