"""검증용: ETF 상장 전 구간을 지수·선물·펀드로 이어 붙인다 (메일은 최근 실제 데이터만 쓰므로 쓰지 않는다).

- 이어 붙이는 방법은 scoring/sources.py splice 와 같다 (연결 시점에 가격 비율을 맞춰 수익률 보존).
- 대리 구간은 미국 거래일(SPY)에 맞추고 (휴장이 다른 시장은 전날 값), 거래량은 비운다.
- 점수: 대리 구간처럼 거래량이 없는 날은 거래량 항목 없이 계산하고, 실제 ETF 거래량이 쌓이면 원래 점수를 쓴다.
  보유 상태는 그렇게 이은 3일 평균 점수로 규칙(run_rule)을 다시 돌려 만든다.
- 비트코인(IBIT)처럼 기초 자산 자체가 없던 종목은 잇지 않는다.

겹치는 기간에 실제 ETF 와 비교 (주간 수익 상관 · 연수익 실제 대 대리): docs/topk.md 8장
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .checklist_score import RULE, checklist_score, run_rule
from .profiles import Profile
from .sources import _normalize_ohlcv, fetch_fdr, load_prices, splice

# 종목: (실제 데이터 spec, 대리 데이터, 수익 배수, 설명)
# 대리 데이터 "A/B" 는 A 가격을 B(환율, 달러당 현지 통화)로 나눠 달러로 바꾼 것
PROXY = {
    "SLV": ("fdr:SLV", "SI=F", 1.0, "은 선물"),
    "UUP": ("fdr:UUP", "DX-Y.NYB", 1.0, "달러 지수 DXY"),
    "HYG": ("fdr:HYG", "VWEHX", 1.0, "뱅가드 하이일드 펀드 VWEHX"),
    "EDV": ("fdr:EDV", "TLT", 1.45, "TLT 수익 × 1.45 (듀레이션 맞춤)"),
    "MCHI": ("fdr:MCHI", "FXI", 1.0, "중국 대형주 FXI"),
    "INDA": ("fdr:INDA", "^BSESN/INR=X", 1.0, "센섹스 달러 환산"),
    "PDBC": ("fdr:PDBC+DBC", "^SPGSCI", 1.0, "DBC(2006-02~) 앞을 S&P GSCI"),
    "DBMF": ("fdr:DBMF", "RYMFX", 1.0, "라이덱스 매니지드 퓨처스 펀드 RYMFX (2007-02~)"),
}


@lru_cache(maxsize=None)
def _us_days(start: str) -> pd.DatetimeIndex:
    return load_prices("fdr:SPY", start).index


def scaled(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """전일 종가 대비 수익(종가·시가·고가·저가)을 k배 한 합성 가격."""
    cp = df["Close"].shift(1)
    rc = (df["Close"] / cp - 1).fillna(0.0) * k
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"].iloc[0] * (1 + rc).cumprod()
    prev = out["Close"].shift(1).fillna(out["Close"].iloc[0])
    for c in ("Open", "High", "Low"):
        out[c] = prev * (1 + (df[c] / cp - 1).fillna(0.0) * k)
    px = out[["Open", "High", "Low", "Close"]]
    out["High"], out["Low"] = px.max(axis=1), px.min(axis=1)
    out["Volume"] = np.nan
    return out


def proxy_prices(spec: str, k: float = 1.0, start: str = "1990-01-01") -> pd.DataFrame:
    """대리 데이터 (미국 거래일, 거래량 없음)."""
    sym, _, fx = spec.partition("/")
    df = _normalize_ohlcv(fetch_fdr(sym, start))
    if fx:
        rate = pd.to_numeric(fetch_fdr(fx, start)["Close"], errors="coerce")
        rate.index = pd.to_datetime(rate.index)
        df = df[["Open", "High", "Low", "Close"]].div(rate.reindex(df.index).ffill(), axis=0).dropna()
    if k != 1.0:
        df = scaled(df, k)
    us = _us_days(start)
    df = df.reindex(us[us >= df.index[0]]).ffill()
    df["Volume"] = np.nan
    return df


def load_backfilled(ticker: str, start: str = "2004-01-01") -> pd.DataFrame:
    """ticker 가격 (상장 전은 PROXY 로 이어 붙임). PROXY 에 없으면 그대로."""
    if ticker not in PROXY:
        return load_prices(f"fdr:{ticker}", start)
    spec, proxy, k, _ = PROXY[ticker]
    real = load_prices(spec, start)
    before = proxy_prices(proxy, k, str(pd.Timestamp(start) - pd.DateOffset(years=1)))
    return splice(real, before).loc[start:]


def checklist_score_bf(df: pd.DataFrame, p: Profile, bench: pd.Series | None = None) -> pd.DataFrame:
    """checklist_score 와 같지만, 거래량이 일부 구간에만 있으면 없는 구간은 거래량 항목 없이 점수를 낸다."""
    out = checklist_score(df, p, bench)
    vol = df["Volume"]
    if not (vol.isna().any() and vol.notna().any()):
        return out
    nv = checklist_score(df.assign(Volume=np.nan), p, bench)
    s = out["score_s"].where(out["score_s"].notna(), nv["score_s"])
    state, events, blocked = run_rule(s.to_numpy(), out["above200"].to_numpy(), out["overheat"].to_numpy(),
                                      out["highvol"].to_numpy(), RULE)
    return out.assign(score_s=s, state=state, event=events, blocked=blocked)
