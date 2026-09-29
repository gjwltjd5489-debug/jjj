"""바스켓 구성: 후보 종목 간 상관관계(수익률 + 점수)를 보고 상관이 낮은 조합을 고른다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    role: str
    inception: str
    backfill: str | None = None  # 상장 전 구간을 이어 붙일 대용 종목


# 후보 유니버스. inception 은 대략적 상장 시점.
CANDIDATES = {a.ticker: a for a in (
    Asset("QQQ", "나스닥100", "미국 대형 성장주", "1999-03"),
    Asset("VTV", "미국 대형 가치주", "미국 가치주", "2004-01"),
    Asset("VWO", "신흥국 주식", "신흥국·달러 약세 수혜", "2005-03"),
    Asset("EFA", "선진국(미국 제외) 주식", "해외 선진국", "2001-08"),
    Asset("GLD", "금", "통화 가치 하락·위기 헤지", "2004-11"),
    Asset("PDBC", "원자재 (DBC 계열)", "인플레이션", "2014-11", backfill="DBC"),
    Asset("DBC", "원자재", "인플레이션", "2006-02"),
    Asset("EDV", "미 초장기 국채 STRIPS", "디플레·경기침체 헤지", "2007-12"),
    Asset("TLT", "미 20년+ 국채", "디플레·경기침체 헤지", "2002-07"),
    Asset("IEF", "미 7~10년 국채", "중기 금리", "2002-07"),
    Asset("TIP", "물가연동국채", "실질금리", "2003-12"),
    Asset("UUP", "미 달러 인덱스", "위기 시 달러 강세", "2007-02"),
    Asset("VNQ", "미국 리츠", "부동산", "2004-09"),
    Asset("DBMF", "매니지드 퓨처스", "추세추종 (역사 짧음)", "2019-05"),
)}

# 사용자 제시 종목에서 출발해 상관이 낮도록 고른 기본 바스켓 (docs/rotation.md 참고)
DEFAULT_BASKET = ("QQQ", "VWO", "GLD", "PDBC", "EDV", "UUP")


def weekly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """주간(금요일) 로그수익률. 거래 시간대가 다른 해외 자산의 일간 비동기 효과를 줄인다."""
    w = closes.resample("W-FRI").last()
    return np.log(w).diff()


def return_corr(closes: pd.DataFrame, min_periods: int = 52) -> pd.DataFrame:
    return weekly_returns(closes).corr(min_periods=min_periods)


def score_corr(scores: pd.DataFrame, min_periods: int = 250) -> pd.DataFrame:
    """점수(지표) 시계열 상관: 같은 날 함께 좋아지고 나빠지는지."""
    return scores.corr(min_periods=min_periods)


def combined_corr(ret_c: pd.DataFrame, sc_c: pd.DataFrame, weight: float = 0.5) -> pd.DataFrame:
    """수익률 상관과 점수 상관의 가중평균 (weight = 수익률 비중)."""
    idx = ret_c.index.intersection(sc_c.index)
    return weight * ret_c.loc[idx, idx] + (1 - weight) * sc_c.loc[idx, idx]


def select_low_corr(corr: pd.DataFrame, k: int, must: tuple[str, ...] = ()) -> list[str]:
    """탐욕법: 이미 고른 종목들과의 평균 상관이 가장 낮은 후보를 하나씩 추가.

    must 가 없으면 전체에서 상관이 가장 낮은 쌍으로 시작한다. 음(-)의 상관은 분산 효과가 크므로 그대로 반영한다.
    """
    names = [n for n in corr.index if corr.loc[n].notna().sum() > 1]
    chosen = [m for m in must if m in names]
    if not chosen:
        c = corr.loc[names, names].where(~np.eye(len(names), dtype=bool))
        a, b = c.stack().idxmin()
        chosen = [a, b]
    while len(chosen) < min(k, len(names)):
        rest = [n for n in names if n not in chosen]
        avg = corr.loc[rest, chosen].mean(axis=1)
        worst = corr.loc[rest, chosen].max(axis=1)
        pick = pd.DataFrame({"avg": avg, "max": worst}).sort_values(["avg", "max"]).index[0]
        chosen.append(pick)
    return chosen[:k]


def avg_offdiag(corr: pd.DataFrame, names) -> float:
    c = corr.loc[names, names].to_numpy()
    mask = ~np.eye(len(names), dtype=bool)
    return float(np.nanmean(c[mask]))
