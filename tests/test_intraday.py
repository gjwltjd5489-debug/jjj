import numpy as np
import pandas as pd

from scoring import get_profile
from scoring.intraday import (IntradayRow, check_ticker, classify, make_subject, provisional_frame, render_html,
                              render_text, table_rows)
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


def test_near_entry_and_exit_lines():
    assert row(dict(BASE, state=0.0, score_s=58.0), dict(BASE, state=0.0, score_s=66.0)) == {"near_buy"}
    assert row(dict(BASE, state=0.0, score_s=58.0), dict(BASE, state=0.0, score_s=60.0)) == set()   # 한 칸 밖
    assert row(dict(BASE, state=0.0, blocked="과열"), dict(BASE, state=0.0, score_s=66.0)) == set()  # 아침 보류는 반복 안 함
    assert row(dict(BASE, score_s=55.0), dict(BASE, score_s=45.0)) == {"near_sell"}


def test_table_keeps_alerts_and_holdings_and_summarises_rest():
    def mk(t, m, p, price=100.0):
        r = IntradayRow(t, "", "", True, 100.0, 100.0, price, 0.0, price / 100 - 1, 0.01, m, p)
        r.notes = classify(r)
        return r
    quiet = dict(BASE, state=0.0, score_s=20.0)
    rows = [mk("Q", quiet, quiet), mk("H", dict(BASE), dict(BASE)),
            mk("S", dict(BASE), dict(BASE), price=95.5), mk("M", quiet, dict(quiet, score_s=30.0))]
    show, rest = table_rows(rows)
    assert [r.ticker for r in show] == ["S", "H", "M"] and [r.ticker for r in rest] == ["Q"]
    meta = {"mode": "normal", "date": "2026-09-30", "time": "10:03", "minutes": 33,
            "today_events": [("9/30(수)", "FOMC 금리 결정 14:00 ET")]}
    text = render_text(rows, meta)
    assert "해당 없음: 아침 신호 재확인" in text and "나머지 1종목" in text and "FOMC" in text
    calm = render_html([mk("Q", quiet, quiet)], meta)
    assert "특이사항 없음" in calm
