import pytest
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
        "above200": True, "overheat": False, "highvol": False, "close": 100.0, "ma200": 90.0}


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
    # 200일선 아래 보유 종목: 점수가 퇴출선 한 칸 안
    assert row(dict(BASE, score_s=55.0), dict(BASE, score_s=45.0, above200=False)) == {"near_sell"}
    # 200일선 위: 점수로는 팔지 않으므로 알리지 않는다
    assert row(dict(BASE, score_s=55.0), dict(BASE, score_s=45.0)) == set()
    # 눌림(점수 ≤ 40) 중에 200일선 2% 안으로 접근하면 알린다
    assert row(dict(BASE, score_s=38.0), dict(BASE, score_s=35.0, ma200=98.5), price=100.0) == {"near_sell"}
    assert row(dict(BASE, score_s=38.0), dict(BASE, score_s=35.0, ma200=90.0), price=100.0) == set()


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


def _live(ticker, move, name="", notes=()):
    r = IntradayRow(ticker, name, "", True, 100.0, 100.0, 100.0 * (1 + move), 0.0, move, 0.01, dict(BASE), dict(BASE))
    r.notes = list(notes)
    return r


def _morning(**ov):
    """아침 메일 바구니(checklist.basket_state 결과)의 필요한 부분만."""
    from datetime import date
    return {"base": date(2026, 9, 30), "today": date(2026, 10, 5), "rebalance": False, "next": date(2026, 10, 15),
            "rows": [{"ticker": "AAA", "ret_hold": 0.10, "since": date(2026, 9, 15), "since_cut": False},
                     {"ticker": "BBB", "ret_hold": 0.0, "since": date(2026, 9, 30), "since_cut": False}],
            "w_now": {"AAA": 0.5, "BBB": 0.3}, "cash_now": 0.1, "sold_today": [], "month_ret": 0.02,
            "ytd": {"ret": 0.20}, "ov": {"dist": -0.05, "risk_off": False, "qld": True, "q_now": 0.0, "q_ret": float("nan"),
                                         "cq": 0.0, "blocked": False, "t0": None, "today": [], **ov}}


def test_basket_live_today_period_and_flags():
    from datetime import date
    from scoring.intraday import basket_live
    rows = [_live("AAA", 0.02, notes=[("near_sell", "x")]), _live("BBB", -0.01), _live("SPY", 0.005), _live("QQQ", 0.01)]
    spy_prev = pd.Series(100.0, index=pd.bdate_range("2025-01-01", periods=250))
    bl = basket_live(_morning(), rows, spy_prev, None, date(2026, 10, 6))
    # 오늘 = 0.5 × 2% + 0.3 × (−1%) = +0.7%, 이번 기간 = (1.02)(1.007) − 1, 올해 = (1.2)(1.007) − 1
    assert bl["today"] == pytest.approx(0.007) and bl["period"] == pytest.approx(1.02 * 1.007 - 1)
    assert bl["ytd"] == pytest.approx(1.2 * 1.007 - 1)
    aaa = bl["hold"][0]
    assert aaa["ret_hold"] == pytest.approx(1.1 * 1.02 - 1) and aaa["flags"] == ["↘ 퇴출 근접"]
    assert bl["signals"] == [] and bl["dist"] < 0
    meta = {"mode": "normal", "date": "2026-10-06", "time": "10:03", "minutes": 33, "holiday": None, "basket": bl}
    assert make_subject(rows, meta).startswith("[장초반 확인] 2026-10-06 10:03 ET · 바구니 +0.7%")
    text = render_text(rows, meta)
    assert "바구니 오늘 +0.7% (같은 시각 SPY +0.5% · QQQ +1.0%) · 이번 기간 +2.7% (9/30 종가 → 지금) · 올해 +20.8%" in text
    assert "AAA 50.0% · 지금 102.00 (+2.0%) · 편입 후 +12.2% (9/15~) · ↘ 퇴출 근접" in text
    assert "🧺 바구니 현재 상황" in render_html(rows, meta)


def test_basket_live_signals_filter_dip_buy_exit_and_stop():
    from datetime import date
    from scoring.intraday import basket_live
    spy_prev = pd.Series(100.0, index=pd.bdate_range("2025-01-01", periods=250))
    # 어제는 200일선 위 → 지금 SPY −12%: 종가가 이대로면 필터 이탈 + QLD 10% 매수
    rows = [_live("AAA", 0.0), _live("BBB", 0.0), _live("SPY", -0.12)]
    bl = basket_live(_morning(), rows, spy_prev, None, date(2026, 10, 6))
    assert [k for k, _ in bl["signals"]] == ["filter", "dip_buy"] and "QLD 10% 매수 신호" in bl["signals"][1][1]
    meta = {"mode": "normal", "date": "2026-10-06", "time": "10:03", "minutes": 33, "holiday": None, "basket": bl}
    assert "SPY 200일선 이탈 후보 · QLD 매수 후보" in make_subject(rows, meta)
    # QLD 10% 보유(매수가 대비 −20%) 중 SPY 가 200일선 위로 + QLD −7% → 청산 신호, 매수가 대비 −25.6% 라 손절도
    held = _morning(dist=0.08, risk_off=True, q_now=0.10, q_ret=-0.20, cq=0.10)
    bl = basket_live(held, [_live("AAA", 0.0), _live("BBB", 0.0), _live("SPY", 0.01)], spy_prev, -0.07, date(2026, 10, 6))
    assert [k for k, _ in bl["signals"]] == ["dip_exit", "dip_stop"]
    assert bl["qld"]["ret"] == pytest.approx(0.8 * 0.93 - 1) and bl["today"] == pytest.approx(0.10 * -0.07)
    # 이미 단계만큼 산 상태(10%)면 −12% 에서 추가 매수 신호 없음, 손절 뒤 쉬는 중이면 매수 신호 없음
    assert "dip_buy" not in [k for k, _ in basket_live(_morning(cq=0.10, q_now=0.1, q_ret=0.0, risk_off=True, dist=0.11),
                                                      rows, spy_prev, 0.0, date(2026, 10, 6))["signals"]]
    assert basket_live(_morning(blocked=True, risk_off=True, dist=0.11), rows, spy_prev, None,
                       date(2026, 10, 6))["signals"] == []
