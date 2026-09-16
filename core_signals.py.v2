"""
자산배분 프로젝트 - "코어" 포트폴리오(QQQ/TLT/GLD) 비중 계산 + 텔레그램 전송
========================================================================
운영 원칙 (사용자 확정):
  1. 코어(QQQ/TLT/GLD) + 새터라이트(추후 확정)로 구분.
  2. 코어의 목표 비중 합계는 전체 포트폴리오의 80%.
  3. 새터라이트는 코어에서 남는 현금(20% + 코어 내 미배분분) 중 일부로 운영 예정
     (최대 비중 미정, 점진 확대 예정) - 이 스크립트에서는 아직 계산하지 않음.
  4. 코어 내부(코어=100% 기준)에서 자산별 최대 비중: QQQ 80%, TLT 80%, GLD 40%.

각 자산의 신호(0~100%, 각 자산별로 이미 검증된 개별 프레임워크)를 해당 자산의
코어 내부 최대비중까지 선형으로 매핑해서 "코어 내부 비중"을 구하고,
세 자산의 코어 내부 비중 합이 100%를 넘으면 비례 축소(정규화)한다.
  예) QQQ 신호 100% -> 코어 내부 80%(=최대), GLD 신호 67% -> 코어 내부 40%*0.67=26.7%

이 결합 방식은 이번에 사용자가 제시한 "자산별 최대 비중"만으로 도출한 가장 단순한
해석이며, 실제 원하는 배분 로직과 다르면 언제든 combine_core() 함수만 고치면 됨.

필요한 환경변수(GitHub Actions Secrets로 설정):
  TELEGRAM_TOKEN - @BotFather에서 발급받은 토큰
  TELEGRAM_TO    - 메시지를 받을 채팅 ID
"""
import os
import traceback
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_TO = os.environ.get("TELEGRAM_TO", "")
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------- 운영 원칙 상수
CORE_TOTAL = 0.80  # 코어(QQQ+TLT+GLD) 목표 비중 합계(전체 포트폴리오 대비)
MAX_CORE_INTERNAL = {"QQQ": 0.80, "TLT": 0.80, "GLD": 0.40}  # 코어 내부(=100%) 기준 자산별 최대 비중


# ---------------------------------------------------------------- 공통 유틸

def get_close(ticker, period="5y"):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"{ticker} 데이터를 받지 못함")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.set_index("date")[series_id].dropna()


def rolling_percentile(s, window):
    mp = max(60, int(window / 3))
    return s.rolling(window, min_periods=mp).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)


def hysteresis_series(fav, enter_th, exit_th):
    v = fav.to_numpy()
    n = len(v)
    state = np.zeros(n)
    cur = 0.0
    for i in range(n):
        if np.isnan(v[i]):
            state[i] = np.nan
            continue
        if cur == 0.0 and v[i] > enter_th:
            cur = 1.0
        elif cur == 1.0 and v[i] < exit_th:
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=fav.index)


def tier_signal(fav, boundaries, band):
    v = fav.to_numpy()
    n = len(v)
    K = len(boundaries) + 1
    tier = np.full(n, np.nan)
    cur = 0
    for i in range(n):
        x = v[i]
        if np.isnan(x):
            tier[i] = np.nan
            continue
        moved = True
        while moved:
            moved = False
            if cur < K - 1 and x > boundaries[cur] + band:
                cur += 1
                moved = True
            elif cur > 0 and x < boundaries[cur - 1] - band:
                cur -= 1
                moved = True
        tier[i] = cur
    return pd.Series(tier, index=fav.index) / (K - 1)


# ---------------------------------------------------------------- QQQ
# MA 계단식 A: 20/120/200일선 각각 개별 히스테리시스(매수+1.0%/매도-1.5%),
# 위에 있는 선 개수(0~3)를 0/33/67/100% 신호값에 매핑 (project: qqq-백테스트-결과.md 확정값)

def qqq_signal():
    px = get_close("QQQ", period="3y")
    sigs = {}
    for w in (20, 120, 200):
        ma = px.rolling(w).mean()
        upper = ma * 1.010
        lower = ma * 0.985
        p, u, l = px.to_numpy(), upper.to_numpy(), lower.to_numpy()
        state = np.zeros(len(p))
        cur = 0.0
        for i in range(len(p)):
            if np.isnan(u[i]) or np.isnan(l[i]):
                state[i] = np.nan
                continue
            if p[i] > u[i]:
                cur = 1.0
            elif p[i] < l[i]:
                cur = 0.0
            state[i] = cur
        sigs[w] = pd.Series(state, index=px.index)
    sig_df = pd.DataFrame(sigs)
    count = sig_df.sum(axis=1).where(sig_df.notna().all(axis=1))
    weight = (count / 3.0).dropna()
    last_date = weight.index[-1]
    detail = {w: int(sigs[w].dropna().iloc[-1]) for w in (20, 120, 200)}
    arrow = lambda v: "▲" if v else "▼"
    return {
        "asset": "QQQ",
        "date": last_date,
        "price": float(px.loc[last_date]),
        "weight_pct": round(weight.loc[last_date] * 100, 1),
        "detail": f"20일선{arrow(detail[20])} 120일선{arrow(detail[120])} 200일선{arrow(detail[200])} (신호 {int(count.loc[last_date])}/3개)",
    }


# ---------------------------------------------------------------- TLT
# QQQ-TLT 120일 상관계수의 756일 백분위, favorability=1-백분위(반전),
# 히스테리시스 진입0.65/이탈0.40, 이진 0/100% (tlt-지표-상관성-분석.md 확정값)

def tlt_signal():
    qqq = get_close("QQQ", period="4y")
    tlt = get_close("TLT", period="4y")
    df = pd.concat([qqq.rename("qqq"), tlt.rename("tlt")], axis=1).dropna()
    corr120 = df["qqq"].pct_change().rolling(120).corr(df["tlt"].pct_change())
    pctl = rolling_percentile(corr120, 756)
    fav = 1.0 - pctl
    sig = hysteresis_series(fav, 0.65, 0.40).dropna()
    last_date = sig.index[-1]
    return {
        "asset": "TLT",
        "date": last_date,
        "price": float(tlt.loc[last_date]),
        "weight_pct": round(sig.loc[last_date] * 100, 1),
        "detail": f"QQQ-TLT상관 favorability={fav.loc[last_date]:.2f} (진입>0.65/이탈<0.40)",
    }


# ---------------------------------------------------------------- GLD
# 기대인플레이션(T10YIE) 504일 백분위, 4단계(경계 0.25/0.50/0.75, 대칭밴드±0.20)
# (qqq-백테스트-결과.md 확정값)

def gld_signal():
    gld = get_close("GLD", period="4y")
    t10yie = fred_series("T10YIE")
    df = pd.DataFrame({"gld": gld})
    df["t10yie"] = t10yie.reindex(df.index).ffill()
    df = df.dropna()
    pctl = rolling_percentile(df["t10yie"], 504)
    fav = 1.0 - pctl  # 확정 공식(qqq-백테스트-결과.md): 백분위가 낮을수록(기대인플레↓) GLD에 유리
    weight = tier_signal(fav, [0.25, 0.50, 0.75], 0.20).dropna()
    last_date = weight.index[-1]
    return {
        "asset": "GLD",
        "date": last_date,
        "price": float(gld.loc[last_date]),
        "weight_pct": round(weight.loc[last_date] * 100, 1),
        "detail": f"T10YIE={df['t10yie'].loc[last_date]:.2f}%, 504일 백분위={pctl.loc[last_date]:.2f}, favorability={fav.loc[last_date]:.2f}",
    }


# ---------------------------------------------------------------- 코어 결합(운영 원칙 2~4)

def combine_core(qqq_r, tlt_r, gld_r):
    raw = {
        "QQQ": qqq_r["weight_pct"] / 100.0,
        "TLT": tlt_r["weight_pct"] / 100.0,
        "GLD": gld_r["weight_pct"] / 100.0,
    }
    # 신호값(0~1) -> 자산별 코어 내부 최대비중까지 선형 매핑
    core_internal_raw = {k: raw[k] * MAX_CORE_INTERNAL[k] for k in raw}
    total_internal_raw = sum(core_internal_raw.values())

    normalized = total_internal_raw > 1.0
    if normalized:
        scale = 1.0 / total_internal_raw
        core_internal_final = {k: v * scale for k, v in core_internal_raw.items()}
        core_cash_internal = 0.0
    else:
        core_internal_final = dict(core_internal_raw)
        core_cash_internal = 1.0 - total_internal_raw

    overall = {k: v * CORE_TOTAL for k, v in core_internal_final.items()}
    overall_core_cash = core_cash_internal * CORE_TOTAL
    satellite_reserve = 1.0 - CORE_TOTAL  # 새터라이트 확정 전에는 전액 현금 대기

    return {
        "raw": raw,
        "core_internal_raw": core_internal_raw,
        "total_internal_raw": total_internal_raw,
        "normalized": normalized,
        "core_internal_final": core_internal_final,
        "overall": overall,
        "overall_core_cash": overall_core_cash,
        "satellite_reserve": satellite_reserve,
    }


# ---------------------------------------------------------------- 리포트 조립 + 전송

def format_message(results, combo, errors):
    today_kst = datetime.now(KST).strftime("%Y-%m-%d")
    lines = [f"📊 코어 포트폴리오 신호 리포트 — {today_kst}", ""]

    for r in results:
        a = r["asset"]
        lines.append(f"● {a}: 신호 {r['weight_pct']}%  (기준일 {r['date'].date()}, 종가 {r['price']:.2f})")
        lines.append(f"   {r['detail']}")
        lines.append(
            f"   → 코어 내부 비중 {combo['core_internal_final'][a] * 100:.1f}%"
            f" (최대 {MAX_CORE_INTERNAL[a] * 100:.0f}%)"
            f"  → 전체 비중 {combo['overall'][a] * 100:.1f}%"
        )

    lines.append("")
    lines.append(f"코어 내부 합계(정규화 전): {combo['total_internal_raw'] * 100:.1f}%")
    if combo["normalized"]:
        lines.append("⚠️ 100% 초과 → 비례 축소(정규화) 적용됨 (코어 내 현금 없음)")
    else:
        lines.append(f"코어 내 미배분 현금: {combo['overall_core_cash'] * 100:.1f}%p (전체 포트폴리오 기준)")

    lines.append("")
    lines.append("전체 포트폴리오 요약:")
    lines.append(
        f"   QQQ {combo['overall']['QQQ']*100:.1f}% + TLT {combo['overall']['TLT']*100:.1f}%"
        f" + GLD {combo['overall']['GLD']*100:.1f}% + 코어 내 현금 {combo['overall_core_cash']*100:.1f}%"
        f" = 코어 합계 {CORE_TOTAL*100:.0f}%"
    )
    lines.append(f"   위성/현금 예비분: {combo['satellite_reserve']*100:.0f}% (새터라이트 전략 확정 전, 전액 현금 대기)")

    if errors:
        lines.append("")
        lines.append("⚠️ 계산 실패한 자산:")
        for name, err in errors:
            lines.append(f"   - {name}: {err}")

    return "\n".join(lines)


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_TO:
        print("[경고] TELEGRAM_TOKEN/TELEGRAM_TO 미설정 - 콘솔에만 출력합니다")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_TO, "text": text}, timeout=15)
    print("텔레그램 전송 응답:", resp.status_code, resp.text[:300])
    resp.raise_for_status()


def main():
    signal_fns = {"QQQ": qqq_signal, "TLT": tlt_signal, "GLD": gld_signal}
    results = {}
    errors = []
    for name, fn in signal_fns.items():
        try:
            results[name] = fn()
        except Exception as e:
            errors.append((name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    if len(results) < 3:
        # 신호 중 하나라도 실패하면 코어 결합 계산이 불가능하므로, 실패 내역만 알림
        msg_lines = [f"⚠️ 코어 신호 계산 실패 - 일부 자산 데이터를 받지 못했습니다.", ""]
        for name, r in results.items():
            msg_lines.append(f"● {name}: 신호 {r['weight_pct']}% (정상)")
        for name, err in errors:
            msg_lines.append(f"● {name}: 실패 - {err}")
        msg = "\n".join(msg_lines)
    else:
        combo = combine_core(results["QQQ"], results["TLT"], results["GLD"])
        msg = format_message([results["QQQ"], results["TLT"], results["GLD"]], combo, errors)

    print(msg)
    send_telegram(msg)


if __name__ == "__main__":
    main()
