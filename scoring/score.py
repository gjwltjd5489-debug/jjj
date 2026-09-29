"""100점 만점 매수/매도 점수 계산. 기준표는 docs/scorecards.md 참고."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from . import indicators as ind
from .cards import COMBOS, OPTIONAL, V0, Card, Combo, get_card
from .profiles import Profile

LABELS = ("강한 매도", "매도", "중립", "매수", "강한 매수")
VOLUME_CATEGORIES = ("volume", "capitulation", "internals")

# v0 호환용 (기존 코드/테스트에서 사용)
ITEMS = [(it.category, it.key, it.max) for it in V0.items]
CATEGORIES = tuple(V0.categories)
CATEGORY_MAX = {c: V0.category_max(c) for c in CATEGORIES}


def _align(series: pd.Series | None, index: pd.Index, lag: int = 0) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index)
    s = pd.to_numeric(series, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if lag:
        s = s.shift(lag)
    # 휴장일이 달라도 직전 값으로 채운다 (최대 5영업일)
    return s.reindex(s.index.union(index)).ffill(limit=5).reindex(index)


def compute_indicators(df: pd.DataFrame, p: Profile, bench: pd.Series | None = None,
                       ext: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """모든 카드가 쓰는 지표를 한 번에 계산.

    bench: 벤치마크 종가 (상대강도용, 예: QQQ 에 대해 SPY)
    ext:   외부 시계열 {"vix", "us10y", "hy"} — FRED 계열(us10y, hy)은 발표 시차를 고려해 하루 늦춰 쓴다.
    """
    ext = ext or {}
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out = out.join(ind.ichimoku(df, p.tenkan, p.kijun, p.senkou_b, p.displacement))
    out["ma_short"] = ind.sma(close, p.ma_short)
    out["ma_mid"] = ind.sma(close, p.ma_mid)
    out["ma_long"] = ind.sma(close, p.ma_long)
    out["vr"] = ind.volume_ratio(df, p.vr_period)
    out["vr_pct"] = ind.rolling_pct_rank(out["vr"], p.pct_window)
    obv = ind.obv(df)
    out["obv"] = obv
    out["obv_ma"] = ind.sma(obv, p.obv_ma)
    out["dist_days"] = ind.distribution_days(df, p.dist_lookback)
    out["rsi"] = ind.rsi(close, p.rsi_period)
    out = out.join(ind.macd(close, p.macd_fast, p.macd_slow, p.macd_signal))
    out = out.join(ind.adx(df, p.adx_period))
    out["pct_b"] = ind.bollinger_pct_b(close, p.bb_period)
    out["rvol"] = ind.realized_vol(close, p.vol_period)
    out["rvol_pct"] = ind.rolling_pct_rank(out["rvol"], p.pct_window)
    out["dd"] = ind.drawdown(close, p.dd_window)
    out["mom12_1"] = ind.momentum_12_1(close)
    out["down_streak"] = ind.down_streak(close)

    out["volume_raw"] = df["Volume"]
    b = _align(bench, df.index)
    out["bench_raw"] = b
    ratio = close / b
    out["rs_ratio"] = ratio
    out["rs_ratio_ma"] = ind.sma(ratio, p.rs_ma)
    out["rs_ret"] = ratio / ratio.shift(p.rs_lookback) - 1

    vix = _align(ext.get("vix"), df.index)
    out["vix"] = vix
    out["vix_ma5"] = vix.rolling(5).mean()
    out["vix_ma50"] = vix.rolling(50).mean()
    us10y = _align(ext.get("us10y"), df.index, lag=1)
    out["us10y"] = us10y
    out["us10y_chg20"] = us10y - us10y.shift(20)
    hy = _align(ext.get("hy"), df.index, lag=1)
    out["hy"] = hy
    out["hy_ma50"] = hy.rolling(50).mean()
    return out


def label(total: float, card: Card | Combo) -> str | None:
    if pd.isna(total):
        return None
    if total >= card.strong_buy:
        return LABELS[4]
    if total >= card.buy:
        return LABELS[3]
    if total <= card.strong_sell:
        return LABELS[0]
    if total <= card.sell:
        return LABELS[1]
    return LABELS[2]


def _events(total: pd.Series, card: Card) -> pd.Series:
    """매수선 상향 돌파 = BUY, 매도선 하향 돌파 = SELL. 같은 신호가 연달아 나오지 않도록 번갈아 낸다."""
    events = [""] * len(total)
    last = ""
    prev = np.nan
    for k, t in enumerate(total.to_numpy()):
        if not np.isnan(prev) and not np.isnan(t):
            if last != "BUY" and t >= card.buy > prev:
                events[k] = last = "BUY"
            elif last != "SELL" and t <= card.sell < prev:
                events[k] = last = "SELL"
        prev = t
    return pd.Series(events, index=total.index)


def score_with_indicators(x: pd.DataFrame, p: Profile, card: Card) -> pd.DataFrame:
    out = x.copy()
    items = pd.DataFrame({it.key: it.fn(x, p) for it in card.items}, index=x.index)
    out = out.join(items)

    required = [it.key for it in card.items if it.category not in OPTIONAL]
    required_ok = items[required].notna().all(axis=1)
    # optional 항목: 원자료가 (지금까지) 있었는데 지표가 한 번도 안 나왔으면 워밍업 중 → 총점 보류
    for it in card.items:
        if it.category in OPTIONAL and it.input_col:
            seen_input = x[it.input_col].notna().cummax()
            warmed = items[it.key].notna().cummax()
            required_ok &= ~(seen_input & ~warmed)
    maxes = pd.Series({it.key: it.max for it in card.items})
    avail_max = items.notna().mul(maxes, axis=1).sum(axis=1)
    raw = items.sum(axis=1) / avail_max.replace(0, np.nan) * 100
    total = raw.where(required_ok)

    for cat in card.categories:
        keys = [it.key for it in card.items if it.category == cat]
        out[cat] = items[keys].sum(axis=1, min_count=1).where(items[keys].notna().any(axis=1))

    if card.gate is not None:
        gate = card.gate(x, p).fillna(False).astype(bool)
        out["gate"] = gate
        total = total.where(gate | total.isna(), np.minimum(total, card.gate_cap))

    out["max_used"] = avail_max.where(required_ok)
    vol_keys = [it.key for it in card.items if it.category in VOLUME_CATEGORIES]
    out["volume_used"] = items[vol_keys].notna().all(axis=1) if vol_keys else False
    out["total"] = total.round(1)
    out["total_s"] = out["total"].rolling(card.smooth).mean().round(1) if card.smooth > 1 else out["total"]
    out["signal"] = out["total_s"].map(lambda t: label(t, card))
    out["event"] = _events(out["total_s"], card)
    return out


def compute_score(df: pd.DataFrame, p: Profile, card: Card | str = V0,
                  bench: pd.Series | None = None, ext: Mapping[str, pd.Series] | None = None) -> pd.DataFrame:
    """지표, 항목별 점수, 카테고리 소계, 총점, 신호를 담은 DataFrame 반환.

    optional 카테고리(거래량/상대강도/매크로)는 데이터가 없으면 빼고, 남은 만점 기준으로 100점 환산한다.
    total_s 는 카드의 smooth 기간 이동평균 점수이며 signal/event 는 total_s 기준이다.
    """
    if isinstance(card, str) and card in COMBOS:
        return compute_all(df, p, [card], bench, ext)[card]
    if isinstance(card, str):
        card = get_card(card)
    x = compute_indicators(df, p, bench, ext)
    return score_with_indicators(x, p, card)


def combine(entry: pd.DataFrame, exit_: pd.DataFrame, combo: Combo) -> pd.DataFrame:
    """진입 카드(v3) BUY 돌파로 들어가고 청산 카드(v2) 점수가 청산선 이하면 나오는 상태 기계.

    각 카드 자체의 BUY/SELL 교대 규칙과 무관하게 점수 수준으로 판단한다
    (예: v2 가 이미 SELL 을 낸 뒤 다시 40 아래로 가도 청산되도록).
    """
    e = entry["total_s"].to_numpy()
    x = exit_["total_s"].to_numpy()
    state = np.full(len(e), np.nan)
    events = [""] * len(e)
    cur = 0.0
    started = False
    for k in range(len(e)):
        if np.isnan(x[k]):
            if started:
                state[k] = cur
            continue
        started = True
        pullback = k > 0 and not np.isnan(e[k]) and not np.isnan(e[k - 1]) and e[k] >= combo.entry_level > e[k - 1]
        trend = combo.trend_entry is not None and k > 0 and not np.isnan(x[k - 1]) \
            and x[k] >= combo.trend_entry > x[k - 1]
        if cur == 0 and (pullback or trend) and x[k] > combo.exit_level:
            cur = 1.0
            events[k] = "BUY"
        elif cur == 1 and x[k] <= combo.exit_level:
            cur = 0.0
            events[k] = "SELL"
        state[k] = cur

    out = exit_[["close"]].copy()
    out[combo.exit] = exit_["total_s"]
    out[combo.entry] = entry["total_s"]
    out["state"] = state
    trend = exit_["total_s"]
    if combo.rank == "v2v3":
        rank_part = (trend + entry["total_s"]) / 4
    else:
        rank_part = trend / 2
    out["total"] = (50 * out["state"] + rank_part).round(1)
    out["total_s"] = out["total"]
    out["signal"] = out["total_s"].map(lambda t: label(t, combo))
    out["event"] = pd.Series(events, index=out.index)
    out["max_used"] = exit_["max_used"]
    return out


def compute_all(df: pd.DataFrame, p: Profile, cards, bench=None, ext=None) -> dict[str, pd.DataFrame]:
    """여러 카드(조합 카드 포함)를 지표 1회 계산으로 평가."""
    x = compute_indicators(df, p, bench, ext)
    names = [c if isinstance(c, str) else c.name for c in cards]
    needed = []
    for n in names:
        parts = [COMBOS[n].entry, COMBOS[n].exit] if n in COMBOS else [n]
        needed += [q for q in parts if q not in needed]
    base = {n: score_with_indicators(x, p, get_card(n)) for n in needed}
    out = {}
    for n in names:
        if n in COMBOS:
            c = COMBOS[n]
            out[n] = combine(base[c.entry], base[c.exit], c)
        else:
            out[n] = base[n]
    return out


def breakdown(row: pd.Series, card: Card | str = V0) -> pd.DataFrame:
    """한 날짜의 항목별 점수표."""
    if isinstance(card, str):
        card = get_card(card)
    rows = [(it.category, it.key, row[it.key], it.max, it.desc) for it in card.items]
    return pd.DataFrame(rows, columns=["category", "item", "points", "max", "rule"])
