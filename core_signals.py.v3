"""
코어 포트폴리오(QQQ/TLT/GLD/XLE) 일일 신호 계산 + 텔레그램 발송  (v2, 2026-09 최종 확정)
=================================================================================
v1 대비 주요 변경점:
  - QQQ/GLD/XLE: 일봉 이동평균 -> "진짜 주봉 이동평균"(주봉 캔들로 집계한 20/50/100주선)으로 교체
  - TLT: 주1회(금요일) 체크 -> 매일 체크로 변경 (일봉 20/120/200일선은 그대로)
  - 코어 예산 80% -> 100%로 변경
  - 캡: QQQ 60% / TLT 40% / GLD 20% / XLE 20%
  - TLT가 매일 바뀔 수 있으므로, 화~금요일이라도 TLT 비중이 실제로 바뀐 날은 상세 메시지로 전환

- 매일 KST 06:30(월~토) GitHub Actions로 실행
- yfinance로 최신 데이터를 받아 처음부터 다시 계산(무상태 설계)
- 토요일/월요일 또는 TLT 변경일: 상세 리밸런싱 메시지
- 그 외 화~금요일: 보유 현황만(매매 없음)
- 일요일: 미실행

필요 환경변수(GitHub Secrets):
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""
import os
import sys
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# 1. 자산별 확정 파라미터 (2026-09 최종 확정안)
# ============================================================
ASSET_CONFIG = {
    "QQQ": {
        "ma_type": "weekly", "ma": [20, 50, 100],
        "checkpoint": "weekly_friday",
        "buy_band": 0.010, "sell_band": 0.010,
        "scheme": "C",
        "cap": 0.60,
    },
    "TLT": {
        "ma_type": "daily", "ma": [20, 120, 200],
        "checkpoint": "daily",
        "buy_band": 0.040, "sell_band": 0.005,
        "scheme": "C",
        "cap": 0.40,
    },
    "GLD": {
        "ma_type": "weekly", "ma": [20, 50, 100],
        "checkpoint": "weekly_friday",
        "buy_band": 0.030, "sell_band": 0.020,
        "scheme": "C",
        "cap": 0.20,
    },
    "XLE": {
        "ma_type": "weekly", "ma": [20, 50, 100],
        "checkpoint": "weekly_friday",
        "buy_band": 0.030, "sell_band": 0.020,
        "scheme": "A",
        "cap": 0.20,
    },
}

CORE_BUDGET = 1.00
TRADE_COST = 0.0005
REBALANCE_THRESHOLD = 0.005

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

KST = datetime.timezone(datetime.timedelta(hours=9))


def download_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"{ticker} 다운로드 실패")
    return df


def build_weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """일봉 -> 주봉 리샘플. 아직 끝나지 않은 이번 주(마지막 미완성 봉)는 제거."""
    d = daily.set_index("date")
    weekly = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    weekly = weekly.reset_index()
    today = daily["date"].iloc[-1]
    if today.dayofweek != 4:
        weekly = weekly.iloc[:-1].reset_index(drop=True)
    return weekly


def ma_hysteresis_state(close: np.ndarray, sma_arr: np.ndarray, buy_b: float, sell_b: float) -> np.ndarray:
    n = len(close)
    state = np.zeros(n, dtype=int)
    cur = 0
    upper = sma_arr * (1 + buy_b)
    lower = sma_arr * (1 - sell_b)
    for i in range(n):
        if cur == 0 and close[i] > upper[i]:
            cur = 1
        elif cur == 1 and close[i] < lower[i]:
            cur = 0
        state[i] = cur
    return state


def scheme_A(cnt: np.ndarray) -> np.ndarray:
    table = np.array([0.0, 0.5, 0.75, 1.0])
    return table[cnt]


def scheme_C(cnt: np.ndarray) -> np.ndarray:
    n = len(cnt)
    w = np.zeros(n)
    w[0] = cnt[0] / 3
    for i in range(1, n):
        diff = cnt[i] - cnt[i - 1]
        if diff > 0:
            w[i] = min(1.0, w[i - 1] + 0.5)
        elif diff < 0:
            w[i] = max(0.0, w[i - 1] - 0.5)
        else:
            w[i] = w[i - 1]
    return w


SCHEME_FUNCS = {"A": scheme_A, "C": scheme_C}


def compute_asset(ticker: str, cfg: dict) -> dict:
    daily = download_daily(ticker)
    for m in cfg["ma"]:
        daily[f"sma{m}"] = daily["close"].rolling(m).mean()

    if cfg["ma_type"] == "weekly":
        bars = build_weekly_bars(daily)
        for m in cfg["ma"]:
            bars[f"sma{m}"] = bars["close"].rolling(m).mean()
        bars = bars.dropna().reset_index(drop=True)
        close = bars["close"].to_numpy()
        dates = bars["date"]
        lookback_label = "전주비"
    else:
        bars = daily.dropna(subset=[f"sma{m}" for m in cfg["ma"]]).reset_index(drop=True)
        close = bars["close"].to_numpy()
        dates = bars["date"]
        lookback_label = "전일비"

    ma_states = {}
    cnt = np.zeros(len(bars), dtype=int)
    for m in cfg["ma"]:
        st = ma_hysteresis_state(close, bars[f"sma{m}"].to_numpy(), cfg["buy_band"], cfg["sell_band"])
        ma_states[m] = st
        cnt += st

    w = SCHEME_FUNCS[cfg["scheme"]](cnt)

    last_close = close[-1]
    last_date = dates.iloc[-1]
    ma_dev = {m: (last_close / bars[f"sma{m}"].iloc[-1] - 1) for m in cfg["ma"]}

    new_w = w[-1]
    prev_w = w[-2] if len(w) >= 2 else w[-1]
    price_change = last_close / close[-2] - 1 if len(close) >= 2 else 0.0

    return {
        "ticker": ticker, "cfg": cfg,
        "last_close": last_close, "last_date": last_date,
        "ma_states": {m: int(ma_states[m][-1]) for m in cfg["ma"]},
        "ma_dev": ma_dev,
        "prev_w": float(prev_w), "new_w": float(new_w),
        "lookback_label": lookback_label,
        "price_change": price_change,
        "changed_today": abs(new_w - prev_w) > 1e-9,
    }


def apply_core_normalization(asset_results: dict, weight_key: str) -> dict:
    raw = {t: r[weight_key] * r["cfg"]["cap"] for t, r in asset_results.items()}
    raw_sum = sum(raw.values())
    scale = min(1.0, CORE_BUDGET / raw_sum) if raw_sum > 0 else 1.0
    final = {t: v * scale for t, v in raw.items()}
    return {"weights": final, "raw_sum": raw_sum, "scale": scale, "normalized": scale < 0.999}


ARROW_UP, ARROW_DOWN = "🔴", "🔵"
DOT_ON, DOT_OFF = "●", "○"


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


def build_message(results: dict, prev_alloc: dict, new_alloc: dict, is_detail_day: bool,
                   day_label: str, ref_date: str) -> str:
    lines = []
    if is_detail_day:
        lines.append(f"📊 [{day_label}] 리밸런싱 신호")
    else:
        lines.append(f"📋 [{day_label}] 보유 현황 — 매매 없음")
    lines.append(f"{ref_date} 종가 기준")
    lines.append("")

    changes = []
    for t in ASSET_CONFIG:
        pv = prev_alloc["weights"][t] * 100
        nv = new_alloc["weights"][t] * 100
        if abs(nv - pv) > 0.5:
            changes.append((t, pv, nv))

    if is_detail_day:
        if changes:
            header = f"🔔 리밸런싱 필요 — {len(changes)}건"
            if new_alloc["normalized"]:
                header += f" (정규화 적용 {new_alloc['raw_sum']*100:.0f}%→100%)"
            lines.append(header)
            for t, pv, nv in changes:
                arrow = ARROW_UP if nv > pv else ARROW_DOWN
                r = results[t]
                ma_arrows = [f"{m}{'주' if r['cfg']['ma_type']=='weekly' else '일'}선{'▲' if r['ma_states'][m]==1 else '▼'}"
                             for m in r["cfg"]["ma"]]
                lines.append(f"{arrow} {t} {pv:.0f}% → {nv:.0f}% ({', '.join(ma_arrows)})")
        else:
            lines.append("🔔 리밸런싱 필요 없음 (모든 비중 유지)")
        lines.append("")

    lines.append(f"{DOT_ON} = 이동평균선 ON")
    lines.append("")

    for t, cfg in ASSET_CONFIG.items():
        r = results[t]
        nv = new_alloc["weights"][t] * 100
        pv = prev_alloc["weights"][t] * 100
        dots = "".join(DOT_ON if r["ma_states"][m] == 1 else DOT_OFF for m in cfg["ma"])
        norm_note = f" (정규화 전 {pv:.0f}%)" if abs(nv - r["new_w"] * cfg["cap"] * 100) > 0.5 else ""
        ma_unit = "주" if cfg["ma_type"] == "weekly" else "일"
        lines.append(f"{dots} {nv:.0f}%{norm_note} {t} (최대 {cfg['cap']*100:.0f}%)")
        sell_disp = cfg['sell_band']*100
        band_str = (f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도-{sell_disp:.1f}%" if sell_disp > 0
                    else f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도0%")
        tf_str = "매일체크" if cfg['checkpoint'] == 'daily' else "주봉(금요일체크)"
        lines.append(f"   {r['lookback_label']} {fmt_pct(r['price_change'])} ({tf_str}, {band_str})")
        dev_str = " / ".join(f"{m}{ma_unit}:{fmt_pct(r['ma_dev'][m])}" for m in cfg["ma"])
        lines.append(f"    MA 대비 {dev_str}")
        lines.append("")

    cash = 1.0 - sum(new_alloc["weights"].values())
    lines.append(f"💰 코어 합계 : {sum(new_alloc['weights'].values())*100:.0f}%")
    lines.append(f"💵 현금 : {cash*100:.0f}%")

    if not is_detail_day:
        lines.append("")
        lines.append("ℹ️ 다음 정기판정: 금요일(QQQ·GLD·XLE) · TLT는 매일 재판정")

    return "\n".join(lines)


def send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 - 콘솔에만 출력합니다.\n")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    if resp.status_code != 200:
        print(f"[에러] 텔레그램 발송 실패: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    print("텔레그램 발송 완료")


def main():
    now_kst = datetime.datetime.now(KST)
    weekday = now_kst.weekday()

    if weekday == 6:
        print("일요일 - 실행하지 않습니다.")
        return

    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_label = day_names[weekday]

    results = {}
    for ticker, cfg in ASSET_CONFIG.items():
        print(f"{ticker} 계산 중...")
        results[ticker] = compute_asset(ticker, cfg)

    is_detail_day = (weekday in (5, 0)) or results["TLT"]["changed_today"]

    prev_alloc = apply_core_normalization(results, "prev_w")
    new_alloc = apply_core_normalization(results, "new_w")

    ref_date = max(r["last_date"] for r in results.values()).strftime("%Y-%m-%d (%a)")

    msg = build_message(results, prev_alloc, new_alloc, is_detail_day, day_label, ref_date)
    send_telegram(msg)


if __name__ == "__main__":
    main()
