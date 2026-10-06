"""바구니: 규칙상 보유 종목 중 3일 평균 점수 상위 k개를 1/k씩, 매월 15일(휴장이면 직전 거래일)과
마지막 거래일 종가에 교체 (월 2회).

검증: docs/topk.md (scripts/eval_topk.py 의 '월2회' 주기와 같은 규칙)
- 후보: 매수·매도 규칙상 보유 구간이고 3일 평균 ≥ n
- 상위 k개를 1/k씩. 경계에서 점수가 같으면 최근 60거래일 수익률이 높은 종목. 후보가 k개보다 적으면 빈자리는 현금.
- 교체 사이에는 규칙 매도 신호가 나도 다음 교체일까지 보유한다 (검증도 그렇게 했다).
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


def basket_weights(scores: dict[str, float], held: dict[str, bool],
                   n: float = BASKET_N, k: int = BASKET_K, mom: dict[str, float] | None = None) -> dict[str, float]:
    """종목 → 비중 (합이 1보다 작으면 나머지는 현금). 언제나 k개 이하, 1/k씩.

    점수가 같으면 최근 MOM_DAYS 거래일 수익률(mom)이 높은 종목, 그것도 같으면 티커 알파벳 순."""
    elig = {t: s for t, s in scores.items() if held.get(t) and pd.notna(s) and s >= n}
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
