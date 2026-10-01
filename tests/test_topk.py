import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.topk import (basket_weights, mid_month, month_end, next_month_end, next_rebalance, prev_month_end,
                         prev_rebalance)


def test_basket_weights_top_k_ties_and_cash():
    sc = {"A": 90.0, "B": 80.0, "C": 80.0, "D": 70.0, "E": 55.0, "F": 95.0}
    held = {t: True for t in sc} | {"F": False}          # F 는 규칙상 보유가 아니라 제외
    assert basket_weights(sc, held, n=60, k=2) == {"A": 0.5, "B": 0.25, "C": 0.25}  # 동점 B·C 가 한 자리를 나눔
    w = basket_weights(sc, held, n=60, k=10)             # 후보 4개 → 10%씩, 현금 60%
    assert set(w) == {"A", "B", "C", "D"} and abs(sum(w.values()) - 0.4) < 1e-12


def test_month_end_calendar():
    assert month_end(date(2026, 9, 15)) == date(2026, 9, 30)
    assert month_end(date(2026, 5, 4)) == date(2026, 5, 29)    # 5/31 일요일
    assert month_end(date(2026, 12, 1)) == date(2026, 12, 31)
    assert prev_month_end(date(2026, 10, 1)) == date(2026, 9, 30)
    assert next_month_end(date(2026, 9, 30)) == date(2026, 10, 30)
    # 월 2회 교체일: 15일(휴장이면 직전 거래일)과 월말
    assert mid_month(date(2026, 11, 3)) == date(2026, 11, 13)   # 11/15 일요일 → 금요일
    assert prev_rebalance(date(2026, 9, 29)) == date(2026, 9, 15)
    assert prev_rebalance(date(2026, 9, 15)) == date(2026, 8, 31)
    assert next_rebalance(date(2026, 9, 29)) == date(2026, 9, 30)
    assert next_rebalance(date(2026, 9, 30)) == date(2026, 10, 15)


def test_mail_basket_matches_backtest_selection():
    spec = importlib.util.spec_from_file_location("eval_topk", Path(__file__).parents[1] / "scripts" / "eval_topk.py")
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2026-01-01", periods=8)
    cols = list("ABCDEFGHIJKL")
    S = pd.DataFrame(rng.choice([50, 62.5, 75, 83.3, 91.7, 100.0], size=(8, 12)), index=idx, columns=cols)
    H = pd.DataFrame((rng.random((8, 12)) > 0.3).astype(float), index=idx, columns=cols)
    book = et.Book(S, pd.DataFrame(100.0, index=idx, columns=cols), H)
    for d in range(8):
        for k in (3, 5, 10):
            mail = basket_weights(S.iloc[d].to_dict(), (H.iloc[d] == 1).to_dict(), n=60, k=k)
            assert np.allclose(book.target(d, 60, k, True), [mail.get(c, 0.0) for c in cols])


def _reports(last: str):
    from scoring.checklist import TickerReport
    idx = pd.bdate_range("2026-08-03", last)
    before = idx <= "2026-09-15"
    out = []
    # 7종목: 9/15 바구니는 T00~T04 (5개), 9/30 에는 T00 점수 하락·T03 규칙 매도 → T05·T06 편입
    for i in range(7):
        s1 = 95 - i * 3 if i < 5 else 62
        s2 = {0: 50, 5: 99, 6: 98}.get(i, 90 - i)
        held2 = i != 3
        close = np.where(idx <= "2026-09-15", 100.0, 110.0 if i == 4 else 100.0)  # T04 는 9/15 뒤 +10%
        hist = pd.DataFrame({"score_s": np.where(before, s1, s2).astype(float),
                             "state": np.where(before, 1.0, float(held2)), "close": close}, index=idx)
        sig = {"score": s2, "score_s": s2, "prev_score_s": s2, "proj": s2, "held": held2, "event": "",
               "blocked": "", "stop": 95.0, "close": 100.0, "above200": True, "ma200": 90.0, "overheat": False,
               "highvol": False, "recent": [], "hist": hist}
        out.append(TickerReport(ticker=f"T{i:02d}", name="", group="", date=idx[-1], close=100.0, change=0.0,
                                from_high=0.0, checks=[], sig=sig))
    return out


def test_basket_state_between_and_on_rebalance_day():
    from scoring.checklist import basket_state, make_subject, render_html, render_text
    mid = basket_state(_reports("2026-09-29"))              # 9/29: 9/15 바구니 유지
    assert not mid["rebalance"] and mid["base"] == date(2026, 9, 15) and mid["next"] == date(2026, 9, 30)
    assert [x["ticker"] for x in mid["rows"]] == [f"T{i:02d}" for i in range(5)]
    assert all(abs(x["weight"] - 0.2) < 1e-12 for x in mid["rows"])
    assert not next(x for x in mid["rows"] if x["ticker"] == "T03")["held_now"]  # 교체 사이 매도 신호 표시
    # 이번 기간 수익률: T04 +10%, 나머지 0% → 바구니 +2% (20%씩)
    assert abs(next(x for x in mid["rows"] if x["ticker"] == "T04")["ret"] - 0.10) < 1e-12
    assert abs(mid["month_ret"] - 0.02) < 1e-12
    # 편입 후 수익률: T04 는 8/14·8/31·9/15 바구니에 계속 있었다 → 데이터 첫 교체일(8/14)부터, 더 이전은 데이터 없음 표시
    t04 = next(x for x in mid["rows"] if x["ticker"] == "T04")
    assert t04["since"] == date(2026, 8, 14) and t04["since_cut"] and abs(t04["ret_hold"] - 0.10) < 1e-12
    reb = basket_state(_reports("2026-09-30"))               # 9/30: 월말 → 교체
    assert reb["rebalance"] and reb["base"] == date(2026, 9, 30) and reb["prev"] == date(2026, 9, 15)
    assert reb["next"] == date(2026, 10, 15)
    names = [x["ticker"] for x in reb["rows"]]
    assert {"T05", "T06"} <= set(names) and not {"T00", "T03"} & set(names)
    out = {t: why for t, why, _ in reb["out"]}
    assert out["T00"].startswith("점수 50") and out["T03"] == "규칙상 매도"
    meta = {"mode": "normal", "failed": []}
    reps = _reports("2026-09-30")
    assert "바구니 교체 +2 −2" in make_subject(reps, meta)
    assert "🧺 바구니" in render_html(reps, meta) and "제외 T03: 규칙상 매도" in render_text(reps, meta)
    assert abs(reb["month_ret"] - 0.02) < 1e-12 and "지난 바구니 수익률 (9/15 → 9/30 종가): +2.0%" in render_text(reps, meta)
    assert "이번 기간 수익률 (9/15 종가 → 9/29 종가): +2.0%" in render_text(_reports("2026-09-29"), meta)
    assert "T04 20.0% · 점수 83 → 오늘 86 · 편입 후 +10.0% (~8/14~)" in render_text(_reports("2026-09-29"), meta)
    # 교체일: 유지 종목은 편입 후, 제외 종목은 보유 기간 수익률, 신규는 표시만
    t04r = next(x for x in reb["rows"] if x["ticker"] == "T04")
    assert t04r["since"] == date(2026, 8, 14)
    text = render_text(reps, meta)
    assert "제외 T00: 점수 50 < 60 · 보유 기간 +0.0% (~8/14~9/30)" in text and "T05 20.0% · 점수 99 (신규)" in text


def test_basket_ytd_compounds_with_costs():
    from scoring.checklist import TickerReport, basket_ytd, render_text
    idx = pd.bdate_range("2025-12-01", "2026-09-29")
    reps = []
    for i, t in enumerate(["T00", "T01", "T02", "T03", "T04", "SPY"]):
        close = np.where(idx >= "2026-03-02", 110.0, 100.0) if t in ("T00", "SPY") else np.full(len(idx), 100.0)
        hist = pd.DataFrame({"score_s": 90.0 - i if t != "SPY" else 30.0, "state": 1.0, "close": close}, index=idx)
        sig = {"score": 90.0, "score_s": 90.0 - i, "prev_score_s": 90.0, "proj": 90.0, "held": True, "event": "",
               "blocked": "", "stop": 95.0, "close": close[-1], "above200": True, "ma200": 90.0, "overheat": False,
               "highvol": False, "recent": [], "hist": hist}
        reps.append(TickerReport(ticker=t, name="", group="", date=idx[-1], close=close[-1], change=0.0,
                                 from_high=0.0, checks=[], sig=sig))
    y = basket_ytd(reps, date(2026, 9, 29))
    # 5종목 20%씩 계속 보유, T00 만 +10% → 약 +2%. 처음 매수 비용 0.05%와 교체 때 비중 되돌리는 비용이 조금 빠진다
    assert y["start"] == date(2025, 12, 31) and y["trades"] == 18
    assert 0.0185 < y["ret"] < 0.0195 and abs(y["bench_ret"] - 0.10) < 1e-12
    assert "올해 누적 수익률 (12/31 → 9/29 종가, 교체 18회): +1.9% · 같은 기간 SPY +10.0%" in render_text(reps, {"mode": "normal", "failed": []})
