"""
코어 포트폴리오(QQQ/TLT/GLD/XLE) 일일 신호 계산 + 텔레그램 발송
=================================================================
- 매일 KST 06:30(월~토) GitHub Actions로 실행
- yfinance로 최신 데이터를 받아 20/120/200일선 히스테리시스 상태를 처음부터 다시 계산
  (상태를 별도 저장하지 않고 매번 전체 재계산 -> 무상태(stateless) 설계, 버그 발생시 자동 복구됨)
- 토요일/월요일: 상세 리밸런싱 메시지
- 화~금요일: 보유 현황만(매매 없음)
- 일요일: 미실행 (Actions 크론에서 제외)

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
# 1. 자산별 확정 파라미터 (2026-09 기준 최종 확정안)
# ============================================================
ASSET_CONFIG = {
    "QQQ": {
        "ma": [20, 120, 200],
        "buy_band": 0.005, "sell_band": 0.010,
        "checkpoint": "weekly_friday",       # 매주 금요일 종가
        "scheme": "C",
        "cap": 0.60,                          # 코어 내 최대비중
    },
    "TLT": {
        "ma": [20, 120, 200],
        "buy_band": 0.035, "sell_band": 0.000,
        "checkpoint": "weekly_friday",
        "scheme": "C",
        "cap": 0.40,
    },
    "GLD": {
        "ma": [20, 120, 200],
        "buy_band": 0.010, "sell_band": 0.010,
        "checkpoint": "month_last_friday",    # 그 달의 마지막 금요일 종가
        "scheme": "C",
        "cap": 0.35,
    },
    "XLE": {
        "ma": [20, 120, 200],
        "buy_band": 0.0425, "sell_band": 0.0275,
        "checkpoint": "month_last_friday",
        "scheme": "A",                        # 0/50/75/100%
        "cap": 0.20,
    },
}

CORE_BUDGET = 1.00      # 코어 예산 100%
TRADE_COST = 0.0005     # 참고용(메시지엔 표기 안 함, 백테스트 정합용)
CASH_ANNUAL_RATE = 0.03  # 현금 3% (근사, 실제로는 단기금리 연동 가능)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

KST = datetime.timezone(datetime.timedelta(hours=9))


# ============================================================
# 2. 데이터 다운로드 + 상태 계산
# ============================================================
def download(ticker: str) -> pd.DataFrame:
    """전체 히스토리를 받아 SMA까지 계산해서 반환."""
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"{ticker} 다운로드 실패")
    for m in (20, 120, 200):
        df[f"sma{m}"] = df["close"].rolling(m).mean()
    df = df.dropna().reset_index(drop=True)
    return df


def build_checkpoint_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    """판정 시점(체크포인트) 불리언 마스크 생성."""
    n = len(df)
    if mode == "weekly_friday":
        # 그 주의 마지막 거래일(보통 금요일, 공휴일이면 그 전날)
        wid = df["date"].dt.isocalendar().year.astype(str) + "-" + df["date"].dt.isocalendar().week.astype(str)
        mask = (wid != wid.shift(-1)).to_numpy().copy()
        mask[-1] = True
        return mask
    elif mode == "month_last_friday":
        mid = df["date"].dt.year.astype(str) + "-" + df["date"].dt.month.astype(str)
        mask = np.zeros(n, dtype=bool)
        tmp = pd.Series(range(n))
        for _, grp in tmp.groupby(mid):
            fridays = grp[df.loc[grp, "date"].dt.dayofweek == 4]
            if len(fridays) > 0:
                mask[fridays.iloc[-1]] = True
            else:
                mask[grp.iloc[-1]] = True
        return mask
    else:
        raise ValueError(f"unknown checkpoint mode: {mode}")


def ma_hysteresis_state(close: np.ndarray, sma: np.ndarray, buy_b: float, sell_b: float,
                         mask: np.ndarray) -> np.ndarray:
    """단일 이동평균선에 대한 0/1 히스테리시스 상태."""
    n = len(close)
    state = np.zeros(n, dtype=int)
    cur = 0
    upper = sma * (1 + buy_b)
    lower = sma * (1 - sell_b)
    for i in range(n):
        if mask[i]:
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
    """자산 하나에 대한 전체 계산 결과(현재/직전 신호, MA 상태, 이격도 등)."""
    df = download(ticker)
    close = df["close"].to_numpy()
    mask = build_checkpoint_mask(df, cfg["checkpoint"])

    ma_states = {}
    cnt = np.zeros(len(df), dtype=int)
    for m in cfg["ma"]:
        st = ma_hysteresis_state(close, df[f"sma{m}"].to_numpy(), cfg["buy_band"], cfg["sell_band"], mask)
        ma_states[m] = st
        cnt += st

    w = SCHEME_FUNCS[cfg["scheme"]](cnt)

    last_close = close[-1]
    last_date = df["date"].iloc[-1]
    ma_dev = {m: (last_close / df[f"sma{m}"].iloc[-1] - 1) for m in cfg["ma"]}

    # 체크포인트 기준 직전 값(=현재 보유중인 목표비중)과 최신 값(=새 목표비중)
    checkpoint_idx = np.where(mask)[0]
    is_today_checkpoint = mask[-1]
    if len(checkpoint_idx) >= 2:
        prev_w = w[checkpoint_idx[-2]] if is_today_checkpoint else w[checkpoint_idx[-1]]
    else:
        prev_w = w[0]
    new_w = w[-1]

    # 전주/전월 대비 가격 등락률
    if cfg["checkpoint"] == "weekly_friday":
        lookback_label = "전주비"
        prior_idx = checkpoint_idx[-2] if is_today_checkpoint and len(checkpoint_idx) >= 2 else (
            checkpoint_idx[-1] if len(checkpoint_idx) >= 1 else 0)
    else:
        lookback_label = "전월비"
        prior_idx = checkpoint_idx[-2] if is_today_checkpoint and len(checkpoint_idx) >= 2 else (
            checkpoint_idx[-1] if len(checkpoint_idx) >= 1 else 0)
    price_change = last_close / close[prior_idx] - 1 if close[prior_idx] else 0.0

    return {
        "ticker": ticker,
        "cfg": cfg,
        "last_close": last_close,
        "last_date": last_date,
        "ma_states": {m: int(ma_states[m][-1]) for m in cfg["ma"]},
        "ma_dev": ma_dev,
        "prev_w": float(prev_w),   # 0~1, 자산 자체 신호값(캡 곱하기 전)
        "new_w": float(new_w),
        "lookback_label": lookback_label,
        "price_change": price_change,
        "is_today_checkpoint": bool(is_today_checkpoint),
    }


# ============================================================
# 3. 코어 정규화(비례축소) 적용
# ============================================================
def apply_core_normalization(asset_results: dict, weight_key: str) -> dict:
    """weight_key: 'prev_w' 또는 'new_w' — 자산 신호값에 캡 곱하고 정규화."""
    raw = {t: r[weight_key] * r["cfg"]["cap"] for t, r in asset_results.items()}
    raw_sum = sum(raw.values())
    scale = min(1.0, CORE_BUDGET / raw_sum) if raw_sum > 0 else 1.0
    final = {t: v * scale for t, v in raw.items()}
    return {"weights": final, "raw_sum": raw_sum, "scale": scale, "normalized": scale < 0.999}


# ============================================================
# 4. 메시지 생성
# ============================================================
ARROW_UP, ARROW_DOWN = "🔴", "🔵"
DOT_ON, DOT_OFF = "●", "○"


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


def build_message(results: dict, prev_alloc: dict, new_alloc: dict, is_detail_day: bool,
                   day_label: str, ref_date: str) -> str:
    lines = []
    if is_detail_day:
        lines.append(f"📊 [{day_label}] 주간 리밸런싱 신호")
    else:
        lines.append(f"📋 [{day_label}] 보유 현황 — 매매 없음")
    lines.append(f"{ref_date} 종가 기준")
    lines.append("")

    changes = []
    for t in ASSET_CONFIG:
        pv = prev_alloc["weights"][t] * 100
        nv = new_alloc["weights"][t] * 100
        if abs(nv - pv) > 0.5:  # 0.5%p 미만 변화는 무시
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
                ma_arrows = []
                for m in r["cfg"]["ma"]:
                    ma_arrows.append(f"{m}일선{'▲' if r['ma_states'][m]==1 else '▼'}")
                lines.append(f"{arrow} {t} {pv:.0f}% → {nv:.0f}% ({', '.join(ma_arrows)})")
        else:
            lines.append("🔔 리밸런싱 필요 없음 (모든 비중 유지)")
        lines.append("")

    lines.append(f"{DOT_ON} = 20/120/200일선 ON")
    lines.append("")

    for t, cfg in ASSET_CONFIG.items():
        r = results[t]
        nv = new_alloc["weights"][t] * 100
        pv = prev_alloc["weights"][t] * 100
        dots = "".join(DOT_ON if r["ma_states"][m] == 1 else DOT_OFF for m in cfg["ma"])
        norm_note = f" (정규화 전 {pv:.0f}%)" if abs(nv - r["new_w"] * cfg["cap"] * 100) > 0.5 else ""
        lines.append(f"{dots} {nv:.0f}%{norm_note} {t} (최대 {cfg['cap']*100:.0f}%)")
        sell_disp = cfg['sell_band']*100
        band_str = f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도-{sell_disp:.1f}%" if sell_disp > 0 else f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도0%"
        tf_str = "주봉" if "weekly" in cfg["checkpoint"] else "월봉"
        lines.append(f"   {r['lookback_label']} {fmt_pct(r['price_change'])} ({tf_str}, {band_str})")
        dev_str = " / ".join(fmt_pct(r["ma_dev"][m]) for m in cfg["ma"])
        lines.append(f"    MA 대비 {dev_str}")
        lines.append("")

    cash = 1.0 - sum(new_alloc["weights"].values())
    lines.append(f"💰 코어 합계 : {sum(new_alloc['weights'].values())*100:.0f}%")
    lines.append(f"💵 현금 : {cash*100:.0f}%")

    if not is_detail_day:
        lines.append("")
        lines.append("ℹ️ 다음 판정: " + next_checkpoint_summary(results))

    return "\n".join(lines)


def next_checkpoint_summary(results: dict) -> str:
    parts = []
    seen_weekly = False
    for t, r in results.items():
        cfg = r["cfg"]
        if cfg["checkpoint"] == "weekly_friday" and not seen_weekly:
            parts.append("금요일(QQQ·TLT)")
            seen_weekly = True
        elif cfg["checkpoint"] == "month_last_friday":
            parts.append(f"이번달 마지막 금요일({t})")
    return " · ".join(parts)


# ============================================================
# 5. 텔레그램 발송
# ============================================================
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


# ============================================================
# 6. 메인
# ============================================================
def main():
    now_kst = datetime.datetime.now(KST)
    weekday = now_kst.weekday()  # 0=월 ... 5=토 6=일

    if weekday == 6:
        print("일요일 - 실행하지 않습니다.")
        return

    is_detail_day = weekday in (5, 0)  # 토(5), 월(0)
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_label = day_names[weekday]

    results = {}
    for ticker, cfg in ASSET_CONFIG.items():
        print(f"{ticker} 계산 중...")
        results[ticker] = compute_asset(ticker, cfg)

    prev_alloc = apply_core_normalization(results, "prev_w")
    new_alloc = apply_core_normalization(results, "new_w")

    ref_date = max(r["last_date"] for r in results.values()).strftime("%Y-%m-%d (%a)")

    msg = build_message(results, prev_alloc, new_alloc, is_detail_day, day_label, ref_date)
    send_telegram(msg)


if __name__ == "__main__":
    main()
