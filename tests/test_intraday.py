import numpy as np
import pandas as pd

from scoring import get_profile
from scoring.intraday import IntradayRow, check_ticker, classify, make_subject, provisional_frame, render_html
from tests.test_checklist import smooth

P = get_profile("QQQ")


def row(m, p, price=100.0, prev=100.0, open_=100.0, atr_pct=0.01):
    r = IntradayRow("X", "", "", True, prev, open_, price, open_ / prev - 1, price / prev - 1, atr_pct, m, p)
    r.notes = classify(r)
    return {k for k, _ in r.notes}


BASE = {"score": 80.0, "score_s": 75.0, "state": 1.0, "event": "", "blocked": "", "stop": 95.0,
        "above200": True, "overheat": False, "highvol": False, "close": 100.0}


def test_buy_keep_weak_cancel():
    m = dict(BASE, event="BUY")
    assert row(m, dict(BASE)) == {"buy_keep"}
    assert row(m, dict(BASE, score_s=65.0)) == {"buy_weak"}
    assert row(m, dict(BASE, overheat=True)) == {"buy_weak"}
    assert row(m, dict(BASE, event="SELL", score_s=38.0)) == {"buy_cancel"}


def test_sell_flags_only_real_bounce():
    m = dict(BASE, event="SELL", state=0.0)
    assert row(m, dict(BASE, state=0.0, score_s=42.0), price=100.3) == {"sell_keep"}
    assert row(m, dict(BASE, state=0.0), price=102.0, atr_pct=0.01) == {"sell_weak"}


def test_new_candidates_stop_and_gap():
    assert row(dict(BASE, state=0.0), dict(BASE, event="BUY")) == {"new_buy"}
    assert row(dict(BASE), dict(BASE, event="SELL", state=0.0)) == {"new_sell"}
    assert "stop_break" in row(dict(BASE), dict(BASE), price=94.0)
    assert "stop_near" in row(dict(BASE), dict(BASE), price=95.5)
    assert "gap" in row(dict(BASE), dict(BASE), open_=103.0, atr_pct=0.01)
    assert "gap" not in row(dict(BASE), dict(BASE), open_=101.0, atr_pct=0.01)


def test_provisional_volume_uses_20d_median():
    df = smooth(0.001, n=60)
    df.iloc[-1, df.columns.get_loc("Volume")] = 1.0  # 장 초반이라 거래량이 아주 적음
    pv = provisional_frame(df)
    assert pv["Volume"].iloc[-1] == df["Volume"].iloc[-21:-1].median()
    assert df["Volume"].iloc[-1] == 1.0  # 원본은 그대로


def test_check_ticker_detects_intraday_bar_and_morning_basis():
    df = smooth(0.002, n=700)
    today = df.index[-1].date()
    r = check_ticker(df, P, "X", "", "", today)
    assert r.has_intraday and r.prev_close == df["Close"].iloc[-2] and r.price == df["Close"].iloc[-1]
    stale = check_ticker(df, P, "X", "", "", today + pd.Timedelta(days=1))
    assert not stale.has_intraday


def test_subject_and_modes():
    r = IntradayRow("QQQ", "", "", True, 100, 100, 101, 0, 0.01, 0.01, dict(BASE, event="BUY"), dict(BASE))
    r.notes = classify(r)
    meta = {"mode": "normal", "date": "2026-09-30", "time": "10:03", "minutes": 33}
    assert "매수 유지 QQQ" in make_subject([r], meta)
    hol = {"mode": "holiday", "date": "2026-11-26", "time": "10:03", "minutes": 33, "holiday": "추수감사절"}
    assert make_subject([], hol).startswith("[장초반 확인] 휴장") and "휴장일" in render_html([], hol)
    assert not np.isnan(r.move)
