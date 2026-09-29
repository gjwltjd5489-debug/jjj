"""데이터 소스 지정 문자열(spec)을 읽어 DataFrame/Series 로 돌려준다.

spec 형식
- "data/QQQ.csv"          : CSV 파일 (Investing.com / Yahoo / FinanceDataReader 형식)
- "fdr:QQQ"               : FinanceDataReader 로 바로 받기 (미국 종목 기본 소스 = Yahoo)
- "fdr:INVESTING:QQQ"     : FinanceDataReader 의 Investing.com 소스
- "fdr:005930"            : 한국 종목 (기본 소스 = 네이버)
- "fdr:FRED:DGS10"        : FRED 시계열 (금리, 스프레드 등)
- "sample:nasdaq"         : arch 패키지에 들어있는 실제 데이터 (nasdaq / sp500 / vix, 1999~2018)

fdr: 은 `pip install finance-datareader`, sample: 은 `pip install arch --no-deps` 가 필요하다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import load_investing_csvs, load_series_csv

SAMPLES = {"nasdaq": "NASDAQ Composite 1999~2018", "sp500": "S&P 500 1999~2018", "vix": "VIX 2014~2018"}
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
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


def _sample_path(name: str) -> Path:
    if name not in SAMPLES:
        raise ValueError(f"sample:{name} 없음 (가능: {', '.join(SAMPLES)})")
    spec = importlib.util.find_spec("arch")  # arch 를 import 하지 않고 데이터 파일만 찾는다
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("sample: 소스는 `pip install arch --no-deps` 가 필요합니다")
    return Path(spec.submodule_search_locations[0]) / "data" / name / f"{name}.csv.gz"


def load_sample(name: str) -> pd.DataFrame:
    raw = pd.read_csv(_sample_path(name), parse_dates=["Date"], date_format="%m/%d/%Y").set_index("Date")
    return raw


def load_prices(specs: str | Iterable[str], start: str | None = None) -> pd.DataFrame:
    """가격 데이터(OHLCV). CSV 는 여러 개를 합칠 수 있다."""
    specs = [specs] if isinstance(specs, str) else list(specs)
    fdr_specs = [s for s in specs if s.startswith(("fdr:", "sample:"))]
    if fdr_specs and len(specs) > 1:
        raise ValueError("fdr:/sample: 소스는 하나만 지정하세요")
    if fdr_specs:
        spec = fdr_specs[0]
        kind, _, sym = spec.partition(":")
        df = fetch_fdr(sym, start) if kind == "fdr" else load_sample(sym)
        df = _normalize_ohlcv(df)
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
