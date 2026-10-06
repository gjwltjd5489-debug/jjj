"""메일 보조 정보: 원/달러 환율, 다가오는 일정 (FOMC·CPI·실적 발표).

- FOMC 결정일·CPI 발표일은 config/macro_events.csv 에 직접 적는다 (연준·노동통계국 공식 일정).
  다음 CPI 날짜가 파일에 없으면 메일에 갱신 안내가 붙는다.
- 실적 발표 예정일은 yfinance 로 받는다. 설치돼 있지 않거나 실패하면 조용히 건너뛴다.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MACRO_FILE = ROOT / "config" / "macro_events.csv"
WEEKDAY = "월화수목금토일"
CPI_WARN_DAYS = 35  # 마지막 CPI 날짜가 이 기간 안이면 파일 갱신 안내


def fx_summary(close: pd.Series) -> dict | None:
    """환율 종가 → 최근 값, 전일 대비, 1개월 대비."""
    s = close.dropna()
    if len(s) < 2:
        return None
    last = s.iloc[-1]
    month = s.loc[: s.index[-1] - pd.DateOffset(months=1)]
    return {"date": str(s.index[-1].date()), "rate": float(last), "d1": float(last / s.iloc[-2] - 1),
            "m1": float(last / month.iloc[-1] - 1) if len(month) else float("nan")}


def tbill_daily(start: str) -> pd.Series | None:
    """현금 일간 수익률: 13주 미국 단기국채 금리(^IRX, 연 %) ÷ 100 ÷ 252. 실패하면 None (현금 수익 0으로 계산)."""
    try:
        import yfinance as yf
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        x = yf.download("^IRX", start=start, progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    except Exception:
        return None
    if x is None or not len(x):
        return None
    x.index = pd.to_datetime(x.index).tz_localize(None)
    return x / 100 / 252


def macro_events(path: Path = MACRO_FILE) -> list[tuple[date, str, str]]:
    """(날짜, 종류, 설명) 목록."""
    try:
        df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return []
    return [(date.fromisoformat(r.date), r.kind, r.label) for r in df.itertuples(index=False)]


def earnings_dates(tickers: list[str], today: date) -> dict[str, date]:
    """종목별 다음 실적 발표 예정일 (yfinance). ETF 등 없는 종목은 빠진다."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    out = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar
            ds = cal.get("Earnings Date") if isinstance(cal, dict) else None
            ds = sorted(d for d in (ds or []) if d >= today)
            if ds:
                out[t] = ds[0]
        except Exception:
            continue
    return out


def upcoming(today: date, days: int, earnings: dict[str, date] | None = None,
             macro: list[tuple[date, str, str]] | None = None) -> list[tuple[date, str]]:
    """오늘부터 days 일 안의 일정 (날짜순)."""
    macro = macro_events() if macro is None else macro
    end = today + timedelta(days=days)
    items = [(d, label) for d, _, label in macro if today <= d <= end]
    items += [(d, f"{t} 실적 발표 (예정)") for t, d in (earnings or {}).items() if today <= d <= end]
    return sorted(items)


def calendar_note(today: date, macro: list[tuple[date, str, str]] | None = None) -> str | None:
    macro = macro_events() if macro is None else macro
    cpi = [d for d, kind, _ in macro if kind == "CPI"]
    if not cpi or max(cpi) < today + timedelta(days=CPI_WARN_DAYS):
        return "config/macro_events.csv 에 다음 CPI 발표일을 추가해야 합니다 (노동통계국 일정)."
    return None


def fmt_day(d: date) -> str:
    return f"{d.month}/{d.day}({WEEKDAY[d.weekday()]})"
