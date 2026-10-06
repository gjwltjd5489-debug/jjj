"""빅테크 칸: 나스닥 상장 종목(외국 기업 포함) 중 지난 1년 평균 거래대금 상위 10개를 매년 바꾼다.

- 후보: config/nasdaq_pool.csv (ticker, name, from, to — 나스닥 상장 연도 범위, 비우면 제한 없음)
- 목록: config/bigtech.csv (year = 적용 연도, rank, ticker, 평균 거래대금 백만 달러)
- 적용 연도 Y 의 목록은 Y−1년 마지막 거래일 종가까지 252거래일 평균 거래대금(종가 × 거래량)으로 정한다.
  그날 종가의 바구니 교체(월말)부터 새 목록을 쓴다.
- 새해 목록이 파일에 없으면 scripts/update_bigtech.py 로 만든다.
  아침 메일은 없으면 그 자리에서 계산하고 데이터 점검에 커밋 안내를 붙인다.

고정 M7 은 오늘의 승자를 과거에 알고 있었다는 가정이 들어가 과거 성과가 부풀려진다.
검증: docs/topk.md 7장 (scripts/eval_bigtech.py)
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from .topk import month_end

ROOT = Path(__file__).resolve().parents[1]
POOL_FILE = ROOT / "config" / "nasdaq_pool.csv"
MEMBERS_FILE = ROOT / "config" / "bigtech.csv"
GROUP, BENCH = "빅테크", "SPY"
TOP_N = 10
DV_DAYS, DV_MIN = 252, 200   # 평균 거래대금 기간 (거래일), 최소 필요 일수


def load_pool(path: Path = POOL_FILE) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def listed(row, year: int) -> bool:
    """year 말에 나스닥 상장이었는지 (파일의 from·to 범위)."""
    lo, hi = int(row["from"] or 0), int(row["to"] or 9999)
    return lo <= year <= hi


def load_members(path: Path = MEMBERS_FILE) -> dict[int, list[str]]:
    """적용 연도 → 종목 (순위 순)."""
    try:
        df = pd.read_csv(path, dtype={"year": int, "rank": int, "ticker": str})
    except FileNotFoundError:
        return {}
    return {int(y): list(g.sort_values("rank")["ticker"]) for y, g in df.groupby("year")}


@lru_cache(maxsize=1)
def _default_members() -> dict[int, list[str]]:
    return load_members()


@lru_cache(maxsize=1)
def pool_tickers() -> frozenset[str]:
    try:
        return frozenset(load_pool()["ticker"])
    except FileNotFoundError:
        return frozenset()


def membership_year(d: date) -> int:
    """d 종가에 쓰는 목록의 적용 연도. 그해 마지막 거래일 종가부터는 다음 해 목록."""
    return d.year + 1 if d >= month_end(date(d.year, 12, 1)) else d.year


def members_on(d: date, members: dict[int, list[str]] | None = None) -> list[str]:
    """d 종가 기준 빅테크 칸. 그 해 목록이 없으면 가장 최근 목록."""
    members = _default_members() if members is None else members
    if not members:
        return []
    y = membership_year(d)
    if y in members:
        return members[y]
    older = [k for k in members if k <= y]
    return members[max(older)] if older else []


def eligible(ticker: str, d: date, members: dict[int, list[str]] | None = None) -> bool:
    """바구니 후보 자격: 빅테크 후보군 종목이면 d 기준 목록에 있을 때만, 그 밖의 종목(ETF 등)은 늘."""
    if ticker not in pool_tickers():
        return True
    return ticker in members_on(d, members)


def rank_dollar_volume(prices: dict[str, pd.DataFrame], year: int, pool: pd.DataFrame | None = None) -> pd.Series:
    """적용 연도 year 의 순위: year−1년 마지막 거래일까지 252거래일 평균 거래대금 (백만 달러), 큰 순."""
    pool = load_pool() if pool is None else pool
    end = pd.Timestamp(month_end(date(year - 1, 12, 1)))
    out = {}
    for row in pool.to_dict("records"):
        t = row["ticker"]
        df = prices.get(t)
        if df is None or not listed(row, year - 1):
            continue
        df = df.loc[:end]
        if not len(df) or (end - df.index[-1]).days > 7:   # 그 무렵 거래되지 않던 종목
            continue
        dv = (df["Close"] * df["Volume"]).iloc[-DV_DAYS:]
        if dv.notna().sum() >= DV_MIN:
            out[t] = float(dv.mean()) / 1e6
    return pd.Series(out, dtype=float).sort_values(ascending=False)


def compute_members(year: int, loader: Callable[[str, str], pd.DataFrame],
                    pool: pd.DataFrame | None = None, n: int = TOP_N) -> pd.Series:
    """적용 연도 year 의 상위 n개 (종목 → 평균 거래대금). loader(ticker, start) → OHLCV."""
    pool = load_pool() if pool is None else pool
    start = f"{year - 3}-01-01"
    prices = {}
    for t in pool["ticker"]:
        try:
            prices[t] = loader(t, start)
        except Exception:   # 상장폐지 등으로 못 받는 종목은 건너뛴다
            continue
    return rank_dollar_volume(prices, year, pool).head(n)


def save_members(rows: dict[int, pd.Series], path: Path = MEMBERS_FILE) -> None:
    """적용 연도별 순위를 파일에 쓴다 (같은 연도는 덮어씀)."""
    try:
        old = pd.read_csv(path)
        old = old[~old["year"].isin(list(rows))]
    except FileNotFoundError:
        old = pd.DataFrame(columns=["year", "rank", "ticker", "dollar_volume_musd"])
    new = [{"year": y, "rank": i + 1, "ticker": t, "dollar_volume_musd": round(v)}
           for y, s in rows.items() for i, (t, v) in enumerate(s.items())]
    df = pd.concat([old, pd.DataFrame(new)], ignore_index=True).sort_values(["year", "rank"])
    df.to_csv(path, index=False)
    _default_members.cache_clear()


def load_watchlist(path: Path | str, d: date | None = None,
                   members: dict[int, list[str]] | None = None) -> pd.DataFrame:
    """watchlist.csv (ETF) + d 기준 빅테크 칸 (group '빅테크', 상대강도 비교 SPY)."""
    wl = pd.read_csv(path, dtype=str).fillna("")
    if "bench" not in wl.columns:
        wl["bench"] = ""
    wl = wl[wl["group"] != GROUP]
    names = dict(zip(*load_pool()[["ticker", "name"]].T.values)) if POOL_FILE.exists() else {}
    tech = members_on(d or date.today(), members)
    rows = pd.DataFrame({"group": GROUP, "ticker": tech, "name": [names.get(t, t) for t in tech], "bench": BENCH})
    return pd.concat([wl, rows], ignore_index=True)
