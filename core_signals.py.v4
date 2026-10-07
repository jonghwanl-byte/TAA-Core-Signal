"""
코어 포트폴리오(QQQ/TLT/GLD/XLE) 일일 신호 계산 + 텔레그램 발송  (v4, 2026-09)
=================================================================================
v3 대비 수정사항
  1) [버그] TLT 변경일(화~금) 메시지에 이미 매매한 QQQ/GLD/XLE 변경이 또 표시되던 문제 수정
       - 주봉 자산의 '이전 비중'은 주봉이 방금 완성된 날(토/월)에만 직전 주봉 비중을 쓰고,
         그 외(화~금)에는 현재 비중과 같게 취급
  2) [버그] 금요일이 휴장인 주(성금요일 등)에 마지막 주봉이 통째로 버려져 신호가 1주 늦던 문제 수정
       - 이번 주 금요일 장 마감(미국 동부 16:00) 이후이면 그 주 마지막 거래일 봉으로 주봉 확정
  3) [표시] '정규화 전' 비중이 이전 비중으로 잘못 표시되던 부분 수정 (정규화 전 목표비중 표시)
  4) [옵션] 총 노출 배율(EXPOSURE_SCALE) 추가. 기본 1.0 = 기존 P와 동일. 0.9로 바꾸면 모든 목표비중에 0.9 곱함
     [옵션] 방식 C 정의 선택(C_JUMP). 기본 False = 기존 P. True = 변경 신호 개수 x 50%p (백테스트상 체결 지연에 더 민감)
  5) [안전] 자산별 데이터 기준일이 서로 다르면 메시지 상단에 경고 표시, 다운로드 3회 재시도
  6) [테스트] 환경변수 FORCE_RUN=1 이면 일요일에도 실행(수동 테스트용, Actions 수동 실행의 force 옵션)

변경하지 않은 것: 자산별 이평선/밴드/방식/Cap, 코어 예산 100% 정규화, 스케줄(KST 월~토 06:30)

필요 환경변수(GitHub Secrets):  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import time
import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# 1. 자산별 확정 파라미터 (P 최종안, 변경 없음)
# ============================================================
ASSET_CONFIG = {
    "QQQ": {"ma_type": "weekly", "ma": [20, 50, 100], "checkpoint": "weekly_friday",
            "buy_band": 0.010, "sell_band": 0.010, "scheme": "C", "cap": 0.60},
    "TLT": {"ma_type": "daily", "ma": [20, 120, 200], "checkpoint": "daily",
            "buy_band": 0.040, "sell_band": 0.005, "scheme": "C", "cap": 0.40},
    "GLD": {"ma_type": "weekly", "ma": [20, 50, 100], "checkpoint": "weekly_friday",
            "buy_band": 0.030, "sell_band": 0.020, "scheme": "C", "cap": 0.20},
    "XLE": {"ma_type": "weekly", "ma": [20, 50, 100], "checkpoint": "weekly_friday",
            "buy_band": 0.030, "sell_band": 0.020, "scheme": "A", "cap": 0.20},
}

CORE_BUDGET = 1.00        # 코어 예산(합계 상한)
EXPOSURE_SCALE = 1.0      # 총 노출 배율 (1.0 = 기존 P, 0.9 = 전체 비중 10% 축소 후 현금)
C_JUMP = False            # 방식 C 정의. False = 기존(하루 ±50%p 한도, 검증된 P), True = 변경된 신호 개수 x 50%p

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")


def download_daily(ticker: str) -> pd.DataFrame:
    last_err = None
    for _ in range(3):
        try:
            df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index().rename(columns={"Date": "date"})
            df.columns = [c.lower() for c in df.columns]
            if not df.empty:
                return df
        except Exception as e:  # noqa
            last_err = e
        time.sleep(5)
    raise RuntimeError(f"{ticker} 다운로드 실패 {last_err or ''}")


def build_weekly_bars(daily: pd.DataFrame, now_et: datetime.datetime) -> pd.DataFrame:
    """일봉 -> 주봉(W-FRI). 아직 끝나지 않은 이번 주 봉은 제거.
    '완성' 판정: 마지막 일봉이 금요일이거나, 이번 주 금요일 장 마감(ET 16:00)이 이미 지난 경우
    (금요일 휴장 주 포함). 각 주봉에 그 주의 마지막 거래일(last_date)도 기록한다."""
    d = daily.set_index("date").copy()
    d["_d"] = d.index
    weekly = d.resample("W-FRI").agg({"open": "first", "high": "max", "low": "min", "close": "last", "_d": "last"}).dropna()
    weekly = weekly.rename(columns={"_d": "last_date"}).reset_index()
    last_label = pd.Timestamp(weekly["date"].iloc[-1])
    last_daily = pd.Timestamp(daily["date"].iloc[-1])
    friday_close_et = datetime.datetime.combine(last_label.date(), datetime.time(16, 0), tzinfo=ET)
    complete = (last_daily.dayofweek == 4) or (now_et >= friday_close_et)
    if not complete:
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
        if C_JUMP:
            w[i] = min(1.0, max(0.0, w[i - 1] + 0.5 * diff))
        elif diff > 0:
            w[i] = min(1.0, w[i - 1] + 0.5)
        elif diff < 0:
            w[i] = max(0.0, w[i - 1] - 0.5)
        else:
            w[i] = w[i - 1]
    return w


SCHEME_FUNCS = {"A": scheme_A, "C": scheme_C}


def compute_asset(ticker: str, cfg: dict, now_et: datetime.datetime, daily: pd.DataFrame = None) -> dict:
    if daily is None:
        daily = download_daily(ticker)
    daily = daily.copy()
    for m in cfg["ma"]:
        daily[f"sma{m}"] = daily["close"].rolling(m).mean()
    latest_daily_date = pd.Timestamp(daily["date"].iloc[-1])

    if cfg["ma_type"] == "weekly":
        bars = build_weekly_bars(daily, now_et)
        for m in cfg["ma"]:
            bars[f"sma{m}"] = bars["close"].rolling(m).mean()
        bars = bars.dropna().reset_index(drop=True)
        lookback_label = "전주비"
    else:
        bars = daily.dropna(subset=[f"sma{m}" for m in cfg["ma"]]).reset_index(drop=True)
        lookback_label = "전일비"

    close = bars["close"].to_numpy()
    dates = bars["date"]

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
    if cfg["ma_type"] == "weekly":
        # 주봉이 '가장 최근 일봉'에서 막 완성된 경우에만 직전 주봉 비중이 이전 비중.
        # (그 외 화~금에는 이미 지난주에 반영/매매된 비중이므로 이전 = 현재)
        just_completed = pd.Timestamp(bars["last_date"].iloc[-1]) == latest_daily_date
        prev_w = w[-2] if (just_completed and len(w) >= 2) else w[-1]
    else:
        prev_w = w[-2] if len(w) >= 2 else w[-1]
    price_change = last_close / close[-2] - 1 if len(close) >= 2 else 0.0

    return {
        "ticker": ticker, "cfg": cfg,
        "last_close": last_close, "last_date": last_date, "latest_daily_date": latest_daily_date,
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
    final = {t: v * scale * EXPOSURE_SCALE for t, v in raw.items()}
    return {"weights": final, "raw": raw, "raw_sum": raw_sum, "scale": scale, "normalized": scale < 0.999}


ARROW_UP, ARROW_DOWN = "🔴", "🔵"
DOT_ON, DOT_OFF = "●", "○"


def fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.1f}%"


def build_message(results: dict, prev_alloc: dict, new_alloc: dict, is_detail_day: bool,
                  day_label: str, ref_date: str, warn: str = "") -> str:
    lines = []
    if is_detail_day:
        lines.append(f"📊 [{day_label}] 리밸런싱 신호")
    else:
        lines.append(f"📋 [{day_label}] 보유 현황 — 매매 없음")
    lines.append(f"{ref_date} 종가 기준")
    if EXPOSURE_SCALE != 1.0:
        lines.append(f"(총 노출 배율 ×{EXPOSURE_SCALE:g} 적용)")
    if warn:
        lines.append(warn)
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
        dots = "".join(DOT_ON if r["ma_states"][m] == 1 else DOT_OFF for m in cfg["ma"])
        raw_pct = new_alloc["raw"][t] * 100
        norm_note = f" (정규화 전 {raw_pct:.0f}%)" if new_alloc["normalized"] and abs(nv - raw_pct * EXPOSURE_SCALE) > 0.5 else ""
        ma_unit = "주" if cfg["ma_type"] == "weekly" else "일"
        lines.append(f"{dots} {nv:.0f}%{norm_note} {t} (최대 {cfg['cap']*100:.0f}%)")
        sell_disp = cfg["sell_band"] * 100
        band_str = (f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도-{sell_disp:.1f}%" if sell_disp > 0
                    else f"밴드 매수+{cfg['buy_band']*100:.1f}% / 매도0%")
        tf_str = "매일체크" if cfg["checkpoint"] == "daily" else "주봉(금요일체크)"
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


def run(now_kst: datetime.datetime, downloader=None, sender=None):
    """한 번 실행. downloader/sender를 바꿔 끼우면 테스트 가능."""
    sender = sender or send_telegram
    weekday = now_kst.weekday()
    if weekday == 6 and not os.environ.get("FORCE_RUN"):
        print("일요일 - 실행하지 않습니다. (테스트로 실행하려면 FORCE_RUN=1)")
        return None

    now_et = now_kst.astimezone(ET)
    day_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    day_label = day_names[weekday]

    results = {}
    for ticker, cfg in ASSET_CONFIG.items():
        print(f"{ticker} 계산 중...")
        daily = downloader(ticker) if downloader else None
        results[ticker] = compute_asset(ticker, cfg, now_et, daily)

    is_detail_day = (weekday in (5, 0)) or results["TLT"]["changed_today"]

    prev_alloc = apply_core_normalization(results, "prev_w")
    new_alloc = apply_core_normalization(results, "new_w")

    dates = {t: r["latest_daily_date"].strftime("%Y-%m-%d") for t, r in results.items()}
    warn = ""
    if len(set(dates.values())) > 1:
        warn = "⚠️ 자산별 데이터 기준일 불일치: " + ", ".join(f"{t} {d}" for t, d in dates.items())
    ref_date = max(r["latest_daily_date"] for r in results.values()).strftime("%Y-%m-%d (%a)")

    msg = build_message(results, prev_alloc, new_alloc, is_detail_day, day_label, ref_date, warn)
    sender(msg)
    return {"results": results, "prev": prev_alloc, "new": new_alloc, "detail": is_detail_day, "msg": msg}


def main():
    run(datetime.datetime.now(KST))


if __name__ == "__main__":
    main()
