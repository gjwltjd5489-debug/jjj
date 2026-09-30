from datetime import date

import pandas as pd

from scoring.extras import calendar_note, fmt_day, fx_summary, macro_events, upcoming

MACRO = [(date(2026, 10, 14), "CPI", "CPI 발표"), (date(2026, 10, 28), "FOMC", "FOMC 금리 결정"),
         (date(2026, 12, 10), "CPI", "CPI 발표")]


def test_upcoming_window_merges_earnings():
    ev = upcoming(date(2026, 10, 1), 14, {"TSLA": date(2026, 10, 21), "AAPL": date(2026, 10, 13)}, MACRO)
    assert ev == [(date(2026, 10, 13), "AAPL 실적 발표 (예정)"), (date(2026, 10, 14), "CPI 발표")]
    assert upcoming(date(2026, 10, 14), 0, {}, MACRO) == [(date(2026, 10, 14), "CPI 발표")]


def test_calendar_note_when_cpi_dates_run_out():
    assert calendar_note(date(2026, 10, 1), MACRO) is None
    assert "CPI" in calendar_note(date(2026, 11, 20), MACRO)


def test_macro_file_parses_and_fmt_day():
    ev = macro_events()
    assert ev and all(kind in ("CPI", "FOMC") for _, kind, _ in ev)
    assert fmt_day(date(2026, 10, 14)) == "10/14(수)"


def test_fx_summary():
    idx = pd.bdate_range("2026-08-01", "2026-09-30")
    s = pd.Series(1300.0, index=idx)
    s.iloc[-1] = 1313.0
    fx = fx_summary(s)
    assert round(fx["rate"], 1) == 1313.0 and round(fx["d1"], 3) == 0.01 and round(fx["m1"], 3) == 0.01
