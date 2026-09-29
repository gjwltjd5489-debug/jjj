"""데이터 소스 지정 문자열(spec)을 읽어 DataFrame/Series 로 돌려준다.

spec 형식
- "data/QQQ.csv"          : CSV 파일 (Investing.com / Yahoo / FinanceDataReader 형식)
- "fdr:QQQ"               : FinanceDataReader 로 바로 받기 (미국 종목 기본 소스 = Yahoo)
- "fdr:INVESTING:QQQ"     : FinanceDataReader 의 Investing.com 소스
- "fdr:005930"            : 한국 종목 (기본 소스 = 네이버)
- "fdr:FRED:DGS10"        : FRED 시계열 (금리, 스프레드 등)
- "sample:nasdaq"         : 패키지에 들어있는 실제 데이터 (아래 SAMPLES)
- "fdr:PDBC+DBC"          : PDBC 상장 전 구간을 DBC 수익률로 이어 붙임 (backfill)

fdr: 은 `pip install finance-datareader`,
sample: 은 `pip install arch --no-deps` (nasdaq/sp500/vix/wti) 와
`pip install zipline-reloaded --no-deps` (longbond/cash) 가 필요하다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import load_investing_csvs, load_series_csv

SAMPLES = {
    "nasdaq": "NASDAQ Composite 1999~2018 (arch)",
    "sp500": "S&P 500 1999~2018 (arch)",
    "vix": "VIX 2014~2018 (arch)",
    "wti": "WTI 원유 현물 1986~2019 (arch, FRED DCOILWTICO)",
    "longbond": "미 20년물 금리로 만든 합성 초장기채 지수, 듀레이션 24 (zipline 국채금리 1990~2017.3)",
    "cash": "미 3개월물 금리로 만든 현금 지수 (zipline 국채금리 1990~2017.3)",
}
ARCH_SAMPLES = {"nasdaq", "sp500", "vix", "wti"}
LONGBOND_DURATION = 24.0
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_ohlcv(df: pd.DataFrame, adjust: bool = True) -> pd.DataFrame:
    """OHLCV 로 정리. adjust=True 이고 'Adj Close' 가 있으면 시가·고가·저가·종가를 배당/분할 조정한다."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    if adjust and "Adj Close" in df.columns and "Close" in df.columns:
        ratio = pd.to_numeric(df["Adj Close"], errors="coerce") / pd.to_numeric(df["Close"], errors="coerce")
        for col in ("Open", "High", "Low", "Close"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * ratio
    for col in OHLCV:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df = df[OHLCV]
    # 시가/고가/저가가 없으면(일부 지수) 종가로 채운다
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def _fdr():
    try:
        import FinanceDataReader as fdr  # noqa: N813
    except ImportError as e:
        raise RuntimeError("fdr: 소스는 `pip install finance-datareader` 가 필요합니다") from e
    return fdr


def fetch_fdr(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = _fdr().DataReader(symbol, start, end)
    if df is None or len(df) == 0:
        raise ValueError(f"fdr:{symbol} 데이터가 비어 있습니다 (심볼/네트워크 확인)")
    return df


def _package_dir(pkg: str, hint: str) -> Path:
    spec = importlib.util.find_spec(pkg)  # 패키지를 import 하지 않고 데이터 파일만 찾는다
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"이 샘플은 `{hint}` 가 필요합니다")
    return Path(spec.submodule_search_locations[0])


def _treasury() -> pd.DataFrame:
    path = _package_dir("zipline", "pip install zipline-reloaded --no-deps") / "resources" / "market_data" / "treasury_curves.csv"
    t = pd.read_csv(path)
    t.index = pd.to_datetime(t["Time Period"].str[:10])
    t.index.name = "Date"
    return t.drop(columns=["Time Period"]).apply(pd.to_numeric, errors="coerce")


def rate_index(yields: pd.Series, duration: float = 0.0, start: float = 100.0) -> pd.Series:
    """금리(소수) 시계열 → 총수익 지수. 이자(캐리) + 듀레이션·볼록성 가격 효과."""
    y = yields.dropna()
    dt = y.index.to_series().diff().dt.days.fillna(0) / 365.0
    dy = y.diff().fillna(0)
    ret = y.shift(1).fillna(y.iloc[0]) * dt - duration * dy + 0.5 * duration ** 2 * dy ** 2
    return start * (1 + ret).cumprod()


def load_sample(name: str) -> pd.DataFrame:
    if name not in SAMPLES:
        raise ValueError(f"sample:{name} 없음 (가능: {', '.join(SAMPLES)})")
    if name in ARCH_SAMPLES:
        path = _package_dir("arch", "pip install arch --no-deps") / "data" / name / f"{name}.csv.gz"
        raw = pd.read_csv(path, parse_dates=["Date"], date_format="%m/%d/%Y").set_index("Date")
        if name == "wti":
            raw = raw.rename(columns={"DCOILWTICO": "Close"})
            raw["Close"] = pd.to_numeric(raw["Close"], errors="coerce")
        return raw
    t = _treasury()
    if name == "longbond":
        return pd.DataFrame({"Close": rate_index(t["20year"], LONGBOND_DURATION)})
    return pd.DataFrame({"Close": rate_index(t["3month"], 0.0)})


def splice(primary: pd.DataFrame, proxy: pd.DataFrame) -> pd.DataFrame:
    """primary 시작 전 구간을 proxy 로 채운다. 가격은 연결 시점에 맞춰 비율 조정(수익률 보존)."""
    first = primary.index[0]
    common = proxy.index[proxy.index <= first]
    if len(common) == 0:
        return primary
    anchor = common[-1]
    scale = primary["Close"].iloc[0] / proxy.loc[anchor, "Close"]
    before = proxy.loc[proxy.index < first].copy()
    for col in ("Open", "High", "Low", "Close"):
        before[col] = before[col] * scale
    return pd.concat([before, primary])


def load_prices(specs: str | Iterable[str], start: str | None = None) -> pd.DataFrame:
    """가격 데이터(OHLCV). CSV 는 여러 개를 합칠 수 있다."""
    specs = [specs] if isinstance(specs, str) else list(specs)
    fdr_specs = [s for s in specs if s.startswith(("fdr:", "sample:"))]
    if fdr_specs and len(specs) > 1:
        raise ValueError("fdr:/sample: 소스는 하나만 지정하세요")
    if fdr_specs:
        kind, _, sym = fdr_specs[0].partition(":")
        parts = sym.split("+")  # "PDBC+DBC": 뒤 심볼로 앞 심볼 상장 전 구간 채우기
        frames = [_normalize_ohlcv(fetch_fdr(q, start) if kind == "fdr" else load_sample(q)) for q in parts]
        df = frames[0]
        for proxy in frames[1:]:
            df = splice(df, proxy)
    else:
        df = load_investing_csvs(specs)
    return df.loc[start:] if start else df


def load_series(spec: str | None, start: str | None = None) -> pd.Series | None:
    """단일 시계열 (벤치마크 종가, VIX, 금리 등). spec 이 None 이면 None."""
    if not spec:
        return None
    kind, _, sym = spec.partition(":")
    if kind in ("fdr", "sample"):
        df = fetch_fdr(sym, start) if kind == "fdr" else load_sample(sym)
        col = next((c for c in ("Close", "Adj Close", "vix") if c in df.columns), df.columns[0])
        s = pd.to_numeric(df[col], errors="coerce")
        s.index = pd.to_datetime(s.index)
    elif Path(spec).exists():
        s = load_series_csv(spec)
    else:
        raise FileNotFoundError(spec)
    s = s.dropna().sort_index()
    return s.loc[start:] if start else s
