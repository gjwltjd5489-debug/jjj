"""Investing.com 'Historical Data' CSV 로더.

Investing.com 사이트의 과거 데이터 페이지에서 내려받은 CSV를 읽어
표준 OHLCV DataFrame(Date 인덱스, 오름차순)으로 변환한다.

지원 형식
- 영문 사이트: "Date","Price","Open","High","Low","Vol.","Change %"
- 한국어 사이트: "날짜","종가","시가","고가","저가","거래량","변동 %"

숫자는 "1,234.56", 거래량은 "45.67M" / "1.2B" / "850K" / "-" 형태를 처리한다.
Investing.com 가격은 배당 미조정 가격이다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

COLUMN_MAP = {
    # English
    "date": "Date",
    "price": "Close",
    "close": "Close",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "vol.": "Volume",
    "volume": "Volume",
    "change %": "Change",
    # Korean
    "날짜": "Date",
    "종가": "Close",
    "현재가": "Close",
    "시가": "Open",
    "고가": "High",
    "저가": "Low",
    "거래량": "Volume",
    "변동 %": "Change",
    "변동%": "Change",
}

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%d.%m.%Y", "%b %d, %Y")


def _parse_number(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in ("", "-", "—"):
        return np.nan
    mult = 1.0
    if s[-1].upper() in _SUFFIX:
        mult = _SUFFIX[s[-1].upper()]
        s = s[:-1]
    s = s.rstrip("%")
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def _parse_dates(values: pd.Series) -> pd.Series:
    # 한국어 사이트는 "2025- 09- 26" 처럼 공백이 섞여 나온다.
    cleaned = values.astype(str).str.strip().str.replace(r"\s+", "", regex=True)
    cleaned = cleaned.str.replace(r"(\d{4})년(\d{1,2})월(\d{1,2})일", r"\1-\2-\3", regex=True)
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(cleaned, format=fmt, errors="coerce")
        if parsed.notna().all():
            return parsed
    parsed = pd.to_datetime(cleaned, errors="coerce")
    if parsed.isna().any():
        bad = cleaned[parsed.isna()].head(3).tolist()
        raise ValueError(f"날짜를 해석할 수 없습니다: {bad}")
    return parsed


def load_investing_csv(path: str | Path) -> pd.DataFrame:
    """Investing.com CSV 한 개를 읽어 OHLCV DataFrame으로 반환한다."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    rename = {}
    for col in raw.columns:
        key = re.sub(r"\s+", " ", col.strip().lower())
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = raw.rename(columns=rename)

    missing = {"Date", "Close", "High", "Low"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: 필요한 컬럼이 없습니다 {sorted(missing)} (컬럼: {list(raw.columns)})")

    out = pd.DataFrame(index=_parse_dates(df["Date"]))
    out.index.name = "Date"
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            out[col] = df[col].map(_parse_number).to_numpy(dtype=float)
        else:
            out[col] = np.nan
    return out.sort_index()


def load_investing_csvs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """여러 CSV(기간을 나눠 받은 파일)를 합치고 중복 날짜는 뒤 파일 기준으로 정리한다."""
    frames = [load_investing_csv(p) for p in paths]
    if not frames:
        raise ValueError("CSV 파일이 하나 이상 필요합니다")
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])
