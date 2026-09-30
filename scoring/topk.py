"""이번 달 바구니: 규칙상 보유 종목 중 3일 평균 점수 상위 k개를 1/k씩, 매월 마지막 거래일 종가에 교체.

검증: docs/topk.md (scripts/eval_topk.py 의 '월말' 주기와 같은 규칙)
- 후보: 매수·매도 규칙상 보유 구간이고 3일 평균 ≥ n
- 상위 k개를 1/k씩. 경계에서 점수가 같은 종목들은 남은 자리를 나눠 갖는다. 후보가 k개보다 적으면 빈자리는 현금.
- 교체 사이에는 규칙 매도 신호가 나도 월말까지 보유한다 (검증도 그렇게 했다).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .market_calendar import is_trading_day

BASKET_N = 60.0
BASKET_K = 10


def basket_weights(scores: dict[str, float], held: dict[str, bool],
                   n: float = BASKET_N, k: int = BASKET_K) -> dict[str, float]:
    """종목 → 비중 (합이 1보다 작으면 나머지는 현금)."""
    elig = {t: s for t, s in scores.items() if held.get(t) and pd.notna(s) and s >= n}
    if len(elig) <= k:
        return {t: 1.0 / k for t in elig}
    cut = sorted(elig.values(), reverse=True)[k - 1]
    above = [t for t, s in elig.items() if s > cut]
    tie = [t for t, s in elig.items() if s == cut]
    w = {t: 1.0 / k for t in above}
    w.update({t: (k - len(above)) / (len(tie) * k) for t in tie})
    return w


def month_end(d: date) -> date:
    """d 가 속한 달의 마지막 거래일."""
    x = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x


def prev_month_end(d: date) -> date:
    return month_end(d.replace(day=1) - timedelta(days=1))


def next_month_end(d: date) -> date:
    return month_end((d.replace(day=28) + timedelta(days=4)).replace(day=1))
