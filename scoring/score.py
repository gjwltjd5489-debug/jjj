"""100점 만점 매수/매도 점수 계산. 기준표는 docs/scoring_table.md 참고."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .profiles import Profile

# (카테고리, 항목, 만점)
ITEMS = [
    ("ichimoku", "A1_cloud", 10),
    ("ichimoku", "A2_tenkan_kijun", 6),
    ("ichimoku", "A3_chikou", 6),
    ("ichimoku", "A4_future_cloud", 4),
    ("ichimoku", "A5_above_kijun", 4),
    ("ma", "B1_above_short", 4),
    ("ma", "B2_above_mid", 5),
    ("ma", "B3_above_long", 7),
    ("ma", "B4_alignment", 8),
    ("ma", "B5_mid_slope", 3),
    ("ma", "B6_long_slope", 3),
    ("volume", "C1_vr", 14),
    ("volume", "C2_obv", 6),
    ("momentum", "D1_rsi", 10),
    ("momentum", "D2_macd", 10),
]
CATEGORIES = ("ichimoku", "ma", "volume", "momentum")
CATEGORY_MAX = {c: sum(m for cat, _, m in ITEMS if cat == c) for c in CATEGORIES}
assert sum(CATEGORY_MAX.values()) == 100

LABELS = ("강한 매도", "매도", "중립", "매수", "강한 매수")


def _cond(cond: pd.Series, points: float, *inputs: pd.Series) -> pd.Series:
    valid = pd.concat(inputs, axis=1).notna().all(axis=1)
    return pd.Series(np.where(cond, points, 0.0), index=cond.index).where(valid)


def _band(values: pd.Series, bands) -> pd.Series:
    out = pd.Series(np.nan, index=values.index)
    remaining = values.notna()
    for upper, points in bands:
        hit = remaining & (values < upper)
        out[hit] = points
        remaining &= ~hit
    return out


def compute_indicators(df: pd.DataFrame, p: Profile) -> pd.DataFrame:
    close = df["Close"]
    ich = ind.ichimoku(df, p.tenkan, p.kijun, p.senkou_b, p.displacement)
    mac = ind.macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
    obv = ind.obv(df)
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out = out.join(ich)
    out["ma_short"] = ind.sma(close, p.ma_short)
    out["ma_mid"] = ind.sma(close, p.ma_mid)
    out["ma_long"] = ind.sma(close, p.ma_long)
    out["vr"] = ind.volume_ratio(df, p.vr_period)
    out["obv"] = obv
    out["obv_ma"] = ind.sma(obv, p.obv_ma)
    out["rsi"] = ind.rsi(close, p.rsi_period)
    out = out.join(mac)
    return out


def score_items(x: pd.DataFrame, p: Profile) -> pd.DataFrame:
    c = x["close"]
    s = pd.DataFrame(index=x.index)

    # A. 일목균형표 (30)
    top = x[["span_a", "span_b"]].max(axis=1)
    bottom = x[["span_a", "span_b"]].min(axis=1)
    a1 = pd.Series(np.select([c > top, c < bottom], [10.0, 0.0], 5.0), index=x.index)
    s["A1_cloud"] = a1.where(x["span_a"].notna() & x["span_b"].notna())
    rel = (x["tenkan"] - x["kijun"]) / x["kijun"]
    a2 = pd.Series(np.select([rel > p.tk_tolerance, rel < -p.tk_tolerance], [6.0, 0.0], 3.0), index=x.index)
    s["A2_tenkan_kijun"] = a2.where(rel.notna())
    s["A3_chikou"] = _cond(c > x["chikou_ref"], 6, x["chikou_ref"])
    s["A4_future_cloud"] = _cond(x["lead_a"] > x["lead_b"], 4, x["lead_a"], x["lead_b"])
    s["A5_above_kijun"] = _cond(c > x["kijun"], 4, x["kijun"])

    # B. 이동평균 (30)
    ms, mm, ml = x["ma_short"], x["ma_mid"], x["ma_long"]
    s["B1_above_short"] = _cond(c > ms, 4, ms)
    s["B2_above_mid"] = _cond(c > mm, 5, mm)
    s["B3_above_long"] = _cond(c > ml, 7, ml)
    b4 = pd.Series(np.select([(ms > mm) & (mm > ml), (ms < mm) & (mm < ml)], [8.0, 0.0], 4.0), index=x.index)
    s["B4_alignment"] = b4.where(ms.notna() & mm.notna() & ml.notna())
    mm_prev = mm.shift(p.ma_mid_slope_lookback)
    ml_prev = ml.shift(p.ma_long_slope_lookback)
    s["B5_mid_slope"] = _cond(mm > mm_prev, 3, mm, mm_prev)
    s["B6_long_slope"] = _cond(ml > ml_prev, 3, ml, ml_prev)

    # C. 거래량 (20)
    s["C1_vr"] = _band(x["vr"], p.vr_bands)
    s["C2_obv"] = _cond(x["obv"] > x["obv_ma"], 6, x["obv"], x["obv_ma"])

    # D. 모멘텀 (20)
    s["D1_rsi"] = _band(x["rsi"], p.rsi_bands)
    hist_prev = x["hist"].shift(1)
    d2 = (np.where(x["macd"] > x["signal"], 5.0, 0.0)
          + np.where(x["macd"] > 0, 3.0, 0.0)
          + np.where(x["hist"] > hist_prev, 2.0, 0.0))
    s["D2_macd"] = pd.Series(d2, index=x.index).where(x["macd"].notna() & hist_prev.notna())
    return s


def label(total: float, p: Profile) -> str | None:
    if pd.isna(total):
        return None
    if total >= p.strong_buy:
        return LABELS[4]
    if total >= p.buy:
        return LABELS[3]
    if total <= p.strong_sell:
        return LABELS[0]
    if total <= p.sell:
        return LABELS[1]
    return LABELS[2]


def compute_score(df: pd.DataFrame, p: Profile) -> pd.DataFrame:
    """지표, 항목별 점수, 카테고리 소계, 총점, 신호를 담은 DataFrame 반환.

    거래량이 없는 데이터(지수 등)는 거래량 카테고리를 빼고 80점 만점을 100점으로 환산한다.
    """
    x = compute_indicators(df, p)
    s = score_items(x, p)
    out = x.join(s)

    for cat in CATEGORIES:
        cols = [name for c, name, _ in ITEMS if c == cat]
        out[cat] = s[cols].sum(axis=1, min_count=len(cols)).where(s[cols].notna().all(axis=1))

    price_cats = ["ichimoku", "ma", "momentum"]
    price_ok = out[price_cats].notna().all(axis=1)
    has_volume = out["volume"].notna()
    raw = out[price_cats].sum(axis=1)
    no_vol_max = 100 - CATEGORY_MAX["volume"]
    total = np.where(has_volume, raw + out["volume"], raw * 100 / no_vol_max)
    out["volume_used"] = has_volume
    out["total"] = pd.Series(total, index=out.index).where(price_ok).round(1)
    out["signal"] = out["total"].map(lambda t: label(t, p))

    out["event"] = _events(out["total"], p)
    return out


def _events(total: pd.Series, p: Profile) -> pd.Series:
    """매수선 상향 돌파 = BUY, 매도선 하향 돌파 = SELL. 같은 신호가 연달아 나오지 않도록 번갈아 낸다."""
    events = [""] * len(total)
    last = ""
    prev = np.nan
    for i, t in enumerate(total.to_numpy()):
        if not np.isnan(prev) and not np.isnan(t):
            if last != "BUY" and t >= p.buy > prev:
                events[i] = last = "BUY"
            elif last != "SELL" and t <= p.sell < prev:
                events[i] = last = "SELL"
        prev = t
    return pd.Series(events, index=total.index)


def breakdown(row: pd.Series) -> pd.DataFrame:
    """한 날짜의 항목별 점수표."""
    rows = [(cat, name, row[name], m) for cat, name, m in ITEMS]
    return pd.DataFrame(rows, columns=["category", "item", "points", "max"])
