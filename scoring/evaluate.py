"""카드 평가: 점수 구간별 이후 수익률, 순위상관, 단순 롱/현금 전략 성과.

미래 수익률 구간은 서로 겹치므로(20일 수익률을 매일 계산) 통계적 유의성은 과대평가된다.
참고용 비교 지표로만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .score import LABELS


def forward_returns(close: pd.Series, horizons=(20, 60)) -> pd.DataFrame:
    return pd.DataFrame({f"fwd{h}": close.shift(-h) / close - 1 for h in horizons}, index=close.index)


def band_stats(out: pd.DataFrame, horizons=(20, 60)) -> pd.DataFrame:
    fwd = forward_returns(out["close"], horizons)
    df = out[["total_s", "signal"]].join(fwd).dropna(subset=["total_s"])
    rows = []
    for name in LABELS:
        g = df[df["signal"] == name]
        r = {"signal": name, "days": len(g), "share": len(g) / len(df) if len(df) else np.nan}
        for h in horizons:
            f = g[f"fwd{h}"].dropna()
            r[f"avg_{h}d"] = f.mean() if len(f) else np.nan
            r[f"win_{h}d"] = (f > 0).mean() if len(f) else np.nan
        rows.append(r)
    base = {"signal": "(전체)", "days": len(df), "share": 1.0}
    for h in horizons:
        f = df[f"fwd{h}"].dropna()
        base[f"avg_{h}d"] = f.mean()
        base[f"win_{h}d"] = (f > 0).mean()
    rows.append(base)
    return pd.DataFrame(rows)


def rank_ic(out: pd.DataFrame, horizon: int = 20) -> float:
    """점수와 이후 수익률의 스피어만 순위상관."""
    fwd = out["close"].shift(-horizon) / out["close"] - 1
    df = pd.DataFrame({"s": out["total_s"], "f": fwd}).dropna()
    if len(df) < 30:
        return np.nan
    return df["s"].rank().corr(df["f"].rank())


def positions_from_events(out: pd.DataFrame) -> pd.Series:
    """BUY 다음 날부터 보유, SELL 다음 날부터 현금. 첫 이벤트 전에는 현금."""
    state = out["event"].replace({"": np.nan, "BUY": 1.0, "SELL": 0.0}).astype(float).ffill().fillna(0.0)
    return state.shift(1).fillna(0.0)


def strategy_stats(out: pd.DataFrame, cost: float = 0.0005) -> dict:
    """이벤트 기반 롱/현금 전략 vs 같은 기간 보유. cost 는 편도 거래비용(기본 0.05%)."""
    valid = out["total_s"].notna()
    if not valid.any():
        return {}
    o = out.loc[valid.idxmax():]
    ret = o["close"].pct_change().fillna(0.0)
    pos = positions_from_events(o)
    trades = pos.diff().abs().fillna(0.0)
    strat = pos * ret - trades * cost
    years = len(o) / 252

    def cagr(r):
        return (1 + r).prod() ** (1 / years) - 1 if years > 0 else np.nan

    def mdd(r):
        eq = (1 + r).cumprod()
        return (eq / eq.cummax() - 1).min()

    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    return {
        "start": o.index[0].date(),
        "end": o.index[-1].date(),
        "cagr": cagr(strat),
        "mdd": mdd(strat),
        "sharpe": sharpe(strat),
        "exposure": pos.mean(),
        "round_trips": int((pos.diff() > 0).sum()),
        "bh_cagr": cagr(ret),
        "bh_mdd": mdd(ret),
        "bh_sharpe": sharpe(ret),
    }


def summarize(results: dict[str, pd.DataFrame], cost: float = 0.0005) -> pd.DataFrame:
    """카드별 요약 한 줄씩."""
    rows = []
    for name, out in results.items():
        st = strategy_stats(out, cost)
        bs = band_stats(out).set_index("signal")
        v = out["total_s"].dropna()
        rows.append({
            "card": name,
            "IC20": rank_ic(out, 20),
            "IC60": rank_ic(out, 60),
            "strong_buy_20d": bs.loc["강한 매수", "avg_20d"],
            "buy_20d": bs.loc["매수", "avg_20d"],
            "sell_20d": bs.loc["매도", "avg_20d"],
            "strong_sell_20d": bs.loc["강한 매도", "avg_20d"],
            "buy_zone_share": bs.loc[["매수", "강한 매수"], "days"].sum() / bs.loc["(전체)", "days"],
            "daily_jump": v.diff().abs().mean(),
            **{k: st.get(k) for k in ("cagr", "mdd", "sharpe", "exposure", "round_trips", "bh_cagr", "bh_mdd", "bh_sharpe")},
        })
    return pd.DataFrame(rows)


def fmt_pct(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "-")
    return df
