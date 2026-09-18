from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml


@dataclass
class ChildResult:
    exposure: pd.Series
    signal_count: pd.Series
    ma_states: pd.DataFrame


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # yfinance may return a ticker level even for one ticker
        out.columns = out.columns.get_level_values(0)
    out.columns = [str(c).title() for c in out.columns]
    need = ["Open", "High", "Low", "Close"]
    missing = [c for c in need if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    return out[need + (["Volume"] if "Volume" in out.columns else [])].dropna(subset=["Close"])


def fetch_yfinance(ticker: str, lookback_years: int = 12, auto_adjust: bool = False) -> pd.DataFrame:
    import yfinance as yf

    period = f"{max(lookback_years, 3)}y"
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return normalize_ohlc(df)


def load_local_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return normalize_ohlc(df)


def completed_weekly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    weekly = daily.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        **({"Volume": "sum"} if "Volume" in daily.columns else {}),
    }).dropna(subset=["Open", "Close"])
    # On Mon-Thu the current partial week is labeled with a future Friday. Exclude it.
    weekly = weekly[weekly.index <= daily.index.max()]
    return weekly


def hysteresis_states(close: pd.Series, mas: List[int], bands: List[Tuple[float, float]]) -> pd.DataFrame:
    if len(mas) != len(bands):
        raise ValueError("mas and bands length mismatch")
    result = {}
    for ma, (buy_band, sell_band) in zip(mas, bands):
        mean = close.rolling(ma, min_periods=ma).mean()
        state = np.zeros(len(close), dtype=np.int8)
        cur = 0
        for i, (px, avg) in enumerate(zip(close.to_numpy(), mean.to_numpy())):
            if np.isnan(avg):
                state[i] = cur
                continue
            if px > avg * (1.0 + buy_band):
                cur = 1
            elif px < avg * (1.0 - sell_band):
                cur = 0
            state[i] = cur
        result[f"MA{ma}"] = state
    return pd.DataFrame(result, index=close.index)


def exposure_from_counts(counts: pd.Series, mode: str) -> pd.Series:
    mode = mode.upper()
    arr = counts.astype(int).to_numpy()
    exp = np.zeros(len(arr), dtype=float)
    if len(arr) == 0:
        return pd.Series(exp, index=counts.index)

    if mode == "A":
        mapping = {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.0}
        exp = np.array([mapping[int(x)] for x in arr], dtype=float)
    elif mode == "B":
        mapping = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
        exp = np.array([mapping[int(x)] for x in arr], dtype=float)
    elif mode == "C":
        exp[0] = min(arr[0] * 0.50, 1.0)
        for i in range(1, len(arr)):
            exp[i] = np.clip(exp[i - 1] + 0.50 * (arr[i] - arr[i - 1]), 0.0, 1.0)
    elif mode == "D":
        # User-defined strict D:
        # only when signal count changes, move exposure by at most 50%p for that day;
        # if count is unchanged, keep exposure unchanged.
        exp[0] = min(arr[0] * 0.50, 1.0)
        for i in range(1, len(arr)):
            delta = arr[i] - arr[i - 1]
            if delta > 0:
                exp[i] = min(1.0, exp[i - 1] + 0.50)
            elif delta < 0:
                exp[i] = max(0.0, exp[i - 1] - 0.50)
            else:
                exp[i] = exp[i - 1]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pd.Series(exp, index=counts.index, name="exposure")


def run_child(frame: pd.DataFrame, spec: dict) -> ChildResult:
    states = hysteresis_states(
        frame["Close"],
        [int(x) for x in spec["mas"]],
        [tuple(map(float, x)) for x in spec["bands"]],
    )
    counts = states.sum(axis=1).astype(int)
    exposure = exposure_from_counts(counts, spec["mode"])
    return ChildResult(exposure=exposure, signal_count=counts, ma_states=states)


def child_to_daily(daily_index: pd.DatetimeIndex, daily: pd.DataFrame, spec: dict) -> ChildResult:
    if spec["timeframe"] == "daily":
        return run_child(daily, spec)
    if spec["timeframe"] != "weekly":
        raise ValueError(f"Unknown timeframe: {spec['timeframe']}")

    weekly = completed_weekly_from_daily(daily)
    wr = run_child(weekly, spec)
    # A completed Friday signal is held until the next completed weekly signal.
    exp = wr.exposure.reindex(daily_index, method="ffill").fillna(0.0)
    cnt = wr.signal_count.reindex(daily_index, method="ffill").fillna(0).astype(int)
    states = wr.ma_states.reindex(daily_index, method="ffill").fillna(0).astype(int)
    return ChildResult(exp, cnt, states)


def combine_children(children: List[ChildResult], how: str) -> pd.Series:
    frame = pd.concat([c.exposure for c in children], axis=1)
    how = how.lower()
    if how == "single":
        return frame.iloc[:, 0]
    if how == "mean":
        return frame.mean(axis=1)
    if how == "min":
        return frame.min(axis=1)
    if how == "max":
        return frame.max(axis=1)
    raise ValueError(f"Unknown combine rule: {how}")


def compute_all_exposures(data: Dict[str, pd.DataFrame], cfg: dict):
    common = None
    for ticker, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    common = pd.DatetimeIndex(common).sort_values()

    asset_exp = {}
    details = {}
    for ticker, scfg in cfg["strategies"].items():
        daily = data[ticker].reindex(common).dropna(subset=["Close"])
        children = [child_to_daily(daily.index, daily, child) for child in scfg["children"]]
        exp = combine_children(children, scfg["combine"])
        asset_exp[ticker] = exp.reindex(common).ffill().fillna(0.0)
        details[ticker] = children
    return pd.DataFrame(asset_exp, index=common), details


def apply_caps(exposure: pd.DataFrame, caps: Dict[str, float]) -> pd.DataFrame:
    caps_s = pd.Series(caps, dtype=float)
    raw = exposure.mul(caps_s, axis=1)
    total = raw.sum(axis=1)
    scale = pd.Series(1.0, index=raw.index)
    mask = total > 1.0
    scale.loc[mask] = 1.0 / total.loc[mask]
    weights = raw.mul(scale, axis=0)
    weights["CASH"] = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    return weights


def format_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def build_message(weights: pd.DataFrame, exposure: pd.DataFrame, cfg: dict) -> str:
    latest = weights.index[-1]
    today = weights.iloc[-1]
    prev = weights.iloc[-2] if len(weights) > 1 else today * 0
    ex = exposure.iloc[-1]
    lines = [
        f"[Quant A] {latest.date()} 미국장 종가 신호",
        "",
        "목표 비중",
    ]
    for ticker in ["QQQ", "TLT", "GLD", "XLE"]:
        delta = today[ticker] - prev[ticker]
        action = "유지"
        if delta > 0.0025:
            action = f"매수 +{format_pct(delta)}"
        elif delta < -0.0025:
            action = f"매도 {format_pct(delta)}"
        lines.append(
            f"- {ticker}: {format_pct(today[ticker])} | 신호강도 {format_pct(ex[ticker])} | {action}"
        )
    lines.append(f"- 현금: {format_pct(today['CASH'])}")
    lines.append("")
    lines.append("Cap: QQQ 90% / TLT 40% / GLD 80% / XLE 30%")
    lines.append("실행 가정: 미국 종가 신호 → 한국장 당일 오후 매매(미국 익일 시초가 근사)")
    return "\n".join(lines)


def send_telegram(message: str, token: str, chat_id: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=30)
    r.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--local-dir", default=None, help="Optional directory containing qqq_us_d.csv etc. for offline validation")
    parser.add_argument("--send", action="store_true", help="Send Telegram message using TELEGRAM_BOT_TOKEN/CHAT_ID")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tickers = ["QQQ", "TLT", "GLD", "XLE"]
    data = {}
    for ticker in tickers:
        if args.local_dir:
            path = Path(args.local_dir) / f"{ticker.lower()}_us_d.csv"
            data[ticker] = load_local_csv(path)
        else:
            data[ticker] = fetch_yfinance(
                ticker,
                int(cfg["data"].get("lookback_years", 12)),
                bool(cfg["data"].get("auto_adjust", False)),
            )

    exposure, _ = compute_all_exposures(data, cfg)
    weights = apply_caps(exposure, cfg["portfolio"]["caps"])
    message = build_message(weights, exposure, cfg)
    print(message)

    if args.send:
        import os
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        chat_id = os.environ["TELEGRAM_CHAT_ID"]
        send_telegram(message, token, chat_id)


if __name__ == "__main__":
    main()
