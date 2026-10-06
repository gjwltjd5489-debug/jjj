"""바구니: 규칙상 보유 종목 중 3일 평균 점수 상위 k개를 1/k씩, 매월 15일(휴장이면 직전 거래일)과
마지막 거래일 종가에 교체 (월 2회).

검증: docs/topk.md (scripts/eval_topk.py 의 '월2회' 주기와 같은 규칙)
- 후보: 매수·매도 규칙상 보유 구간이고 3일 평균 ≥ n
- 상위 k개를 1/k씩. 경계에서 점수가 같으면 최근 60거래일 수익률이 높은 종목. 후보가 k개보다 적으면 빈자리는 현금.
- 교체 사이에 규칙 매도 신호가 나면 그날 종가에 팔아 현금 (scoring/checklist.py _simulate).
- 중복 제거: IWM·VTV·SCHD 는 다른 주식 종목과 겹쳐 바구니 후보에서 뺀다 (메일 표에는 그대로 나온다).
- SPY 200일선 필터: SPY 종가가 200일선 아래면 주식 종목(지수·섹터·해외 증시·빅테크)은 목표 비중의 절반.
  교체일에는 절반으로 짜고, 교체 사이에 SPY 가 200일선 아래로 내려간 첫날에는 그날 종가에 절반을 판다
  (교체 기간마다 한 번).
- QLD 하락 매수: SPY 가 200일선보다 5% 아래면 QLD 5%, 5% 더 내려갈 때마다 5%씩 (최대 50%).
  현금부터 쓰고 모자라면 바구니 하위 순위부터 줄인다. SPY 가 200일선을 회복하면 전량 팔아 그 돈으로 지금 바구니
  종목을 목표 비중까지 채운다. QLD 묶음이 산 금액 대비 −25%면 손절하고, 200일선을 회복할 때까지 다시 사지 않는다.
검증: docs/topk.md 8장 (scripts/eval_overlay.py)
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .market_calendar import is_trading_day

BASKET_N = 60.0
BASKET_K = 5
MID_DAY = 15  # 월중 교체 기준일


MOM_DAYS = 60  # 점수가 같을 때 비교하는 최근 수익률 기간 (거래일)

BASKET_EXCLUDE = ("IWM", "VTV", "SCHD")   # 다른 주식 종목과 겹쳐 바구니 후보에서 뺌
EQUITY_GROUPS = ("지수", "섹터·스타일", "해외 증시", "빅테크")   # SPY 200일선 필터를 받는 주식 종목
RISK_OFF_CUT = 0.5      # SPY 200일선 아래에서 주식 종목 비중 배수
DIP_TICKER = "QLD"
DIP_STEP, DIP_SIZE, DIP_CAP, DIP_STOP = 0.05, 0.05, 0.50, -0.25


def dip_target(dist: float) -> float:
    """SPY 가 200일선보다 dist(0.12 = 12%) 아래일 때 QLD 누적 매수 목표 (포트폴리오 대비)."""
    if not dist > 0:
        return 0.0
    return min(DIP_CAP, int(dist / DIP_STEP + 1e-9) * DIP_SIZE)


def trim_bottom(w: dict[str, float], need: float) -> float:
    """w(순위 순 dict)에서 하위 순위부터 need 만큼 덜어낸다. 덜어낸 양을 돌려준다 (w 를 바꾼다)."""
    got = 0.0
    for t in reversed(list(w)):
        if got >= need - 1e-12:
            break
        x = min(w[t], need - got)
        w[t] -= x
        got += x
        if w[t] <= 1e-12:
            del w[t]
    return got


def basket_weights(scores: dict[str, float], held: dict[str, bool],
                   n: float = BASKET_N, k: int = BASKET_K, mom: dict[str, float] | None = None,
                   exclude=BASKET_EXCLUDE) -> dict[str, float]:
    """종목 → 비중 (합이 1보다 작으면 나머지는 현금). 언제나 k개 이하, 1/k씩, 순위 순서.

    점수가 같으면 최근 MOM_DAYS 거래일 수익률(mom)이 높은 종목, 그것도 같으면 티커 알파벳 순.
    exclude 종목은 후보에서 뺀다 (SPY 200일선 필터·QLD 몫은 scoring/checklist.py 에서 따로)."""
    elig = {t: s for t, s in scores.items() if held.get(t) and pd.notna(s) and s >= n and t not in exclude}
    mom = mom or {}

    def rank(t: str) -> tuple:
        m = mom.get(t)
        return (-elig[t], -(m if m is not None and pd.notna(m) else -np.inf), t)
    return {t: 1.0 / k for t in sorted(elig, key=rank)[:k]}


def month_end(d: date) -> date:
    """d 가 속한 달의 마지막 거래일."""
    x = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x


def mid_month(d: date) -> date:
    """d 가 속한 달의 15일 이하 마지막 거래일."""
    x = d.replace(day=MID_DAY)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x


def rebalance_days(d: date) -> list[date]:
    """d 가 속한 달의 교체일 (15일 기준일, 마지막 거래일)."""
    return [mid_month(d), month_end(d)]


def prev_rebalance(d: date) -> date:
    """d 보다 앞선 가장 최근 교체일."""
    days = [x for x in rebalance_days(d) if x < d]
    return days[-1] if days else prev_month_end(d)


def next_rebalance(d: date) -> date:
    """d 보다 뒤의 가장 가까운 교체일."""
    days = [x for x in rebalance_days(d) if x > d]
    return days[0] if days else mid_month((d.replace(day=28) + timedelta(days=4)).replace(day=1))


def prev_month_end(d: date) -> date:
    return month_end(d.replace(day=1) - timedelta(days=1))


def next_month_end(d: date) -> date:
    return month_end((d.replace(day=28) + timedelta(days=4)).replace(day=1))
