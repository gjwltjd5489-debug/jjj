"""바스켓 로테이션: 보유 자격이 있는 종목 중 점수 상위 n개를 동일가중으로 보유한다.

규칙 (매일 종가 기준 판단, 다음 날 수익부터 반영)
1. 보유 자격을 잃은 종목(v5 기준 v2 청산)은 즉시 매도하고 그 자리는 현금으로 둔다.
2. 빈 자리는 매일 채운다. 보유 자격이 있고 아직 안 가진 종목 중 점수가 가장 높은 것부터 넣는다.
   진입 타이밍(v3 눌림목)을 살리기 위해서다.
3. 교체는 리밸런스 날(rebalance 일마다)에만 한다. 보유 종목 순위가 n + buffer 밖으로 밀리면 교체한다.
4. 자리마다 1/n 비중이다. 채우지 못한 자리는 현금(cash 수익률)이다.
   거래비용은 비중 변화량 × cost 로 계산한다. 일별 동일가중 유지에 드는 미세 리밸런싱 비용은 무시한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RotationResult:
    weights: pd.DataFrame
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def perf_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {}
    years = len(ret) / 252
    eq = (1 + ret).cumprod()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    mdd = (eq / eq.cummax() - 1).min()
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }


def rotate(score: pd.DataFrame, eligible: pd.DataFrame, asset_ret: pd.DataFrame, top_n: int,
           rebalance: int = 5, buffer: int = 1, cost: float = 0.001,
           cash_ret: pd.Series | None = None) -> RotationResult:
    """score/eligible/asset_ret: 날짜 × 종목. eligible 은 0/1 (NaN = 데이터 없음)."""
    idx = score.index
    cols = list(score.columns)
    S = score.to_numpy(dtype=float)
    E = eligible.reindex(index=idx, columns=cols).fillna(0).to_numpy() > 0.5
    W = np.zeros((len(idx), len(cols)))
    held: list[int] = []
    log = []
    for t in range(len(idx)):
        for j in list(held):
            if not E[t, j]:
                held.remove(j)
                log.append((idx[t], cols[j], "청산(자격 상실)"))
        elig = [j for j in range(len(cols)) if E[t, j] and not np.isnan(S[t, j])]
        ranked = sorted(elig, key=lambda j: -S[t, j])
        if t % rebalance == 0:
            for j in list(held):
                if j not in ranked or ranked.index(j) >= top_n + buffer:
                    held.remove(j)
                    log.append((idx[t], cols[j], "교체(순위 하락)"))
        for j in ranked:
            if len(held) >= top_n:
                break
            if j not in held:
                held.append(j)
                log.append((idx[t], cols[j], "편입"))
        for j in held:
            W[t, j] = 1.0 / top_n
    weights = pd.DataFrame(W, index=idx, columns=cols)

    r = asset_ret.reindex(index=idx, columns=cols).fillna(0.0)
    cash = (cash_ret.reindex(idx).fillna(0.0) if cash_ret is not None else pd.Series(0.0, index=idx))
    w_prev = weights.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    port = (w_prev * r).sum(axis=1) + (1 - w_prev.sum(axis=1)) * cash - turnover.shift(1).fillna(0.0) * cost
    equity = (1 + port).cumprod()
    trades = pd.DataFrame(log, columns=["date", "ticker", "action"])
    stats = perf_stats(port)
    stats.update({
        "avg_holdings": (weights > 0).sum(axis=1).mean(),
        "cash_share": (1 - weights.sum(axis=1)).mean(),
        "turnover_per_year": turnover.sum() / (len(idx) / 252),
        "trades": int((trades["action"] == "편입").sum()),
    })
    return RotationResult(weights, port, equity, trades, stats)


def equal_weight(asset_ret: pd.DataFrame, rebalance: int = 21, cost: float = 0.001) -> pd.Series:
    """동일가중 보유 (rebalance 일마다 비중 재조정). 비교 기준."""
    r = asset_ret.fillna(0.0)
    n = r.shape[1]
    w = np.full(n, 1.0 / n)
    out = []
    for t in range(len(r)):
        if t % rebalance == 0:
            drift = w
            w = np.full(n, 1.0 / n)
            c = np.abs(w - drift).sum() * cost if t else 0.0
        else:
            c = 0.0
        rt = r.iloc[t].to_numpy()
        port = float(w @ rt) - c
        out.append(port)
        grown = w * (1 + rt)
        w = grown / grown.sum() if grown.sum() > 0 else w
    return pd.Series(out, index=r.index)


def yearly_returns(ret: pd.Series) -> pd.Series:
    return (1 + ret).groupby(ret.index.year).prod() - 1
