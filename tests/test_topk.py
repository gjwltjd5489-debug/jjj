import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.topk import basket_weights, month_end, next_month_end, prev_month_end


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
    idx = pd.bdate_range("2026-07-01", last)
    aug = idx <= "2026-08-31"
    out = []
    # 12종목: 8월 말 상위 10개는 T00~T09, 9월 말에는 T00·T01 이 빠지고 T10·T11 이 들어옴
    for i in range(12):
        s_aug = 95 - i * 2 if i < 10 else 61
        s_sep = 90 - i if i >= 2 else 50            # T00·T01 은 9월에 점수 50 → 제외
        held_sep = i != 3                            # T03 은 9월에 규칙 매도
        if i in (10, 11):
            s_sep = 99
        close = np.where(idx <= "2026-08-31", 100.0, 110.0 if i == 4 else 100.0)  # T04 는 9월에 +10%
        hist = pd.DataFrame({"score_s": np.where(aug, s_aug, s_sep).astype(float),
                             "state": np.where(aug, 1.0, float(held_sep)), "close": close}, index=idx)
        sig = {"score": s_sep, "score_s": s_sep, "prev_score_s": s_sep, "proj": s_sep, "held": held_sep, "event": "",
               "blocked": "", "stop": 95.0, "close": 100.0, "above200": True, "ma200": 90.0, "overheat": False,
               "highvol": False, "recent": [], "hist": hist}
        out.append(TickerReport(ticker=f"T{i:02d}", name="", group="", date=idx[-1], close=100.0, change=0.0,
                                from_high=0.0, checks=[], sig=sig))
    return out


def test_basket_state_mid_month_and_rebalance_day():
    from scoring.checklist import basket_state, make_subject, render_html, render_text
    mid = basket_state(_reports("2026-09-29"))              # 9/29: 8/31 바구니 유지
    assert not mid["rebalance"] and mid["base"] == date(2026, 8, 31) and mid["next"] == date(2026, 9, 30)
    assert [x["ticker"] for x in mid["rows"]] == [f"T{i:02d}" for i in range(10)]
    assert not next(x for x in mid["rows"] if x["ticker"] == "T03")["held_now"]  # 달 중간 매도 신호 표시
    # 이번 달 수익률: T04 +10%, 나머지 0% → 바구니 +1% (10%씩)
    assert abs(next(x for x in mid["rows"] if x["ticker"] == "T04")["ret"] - 0.10) < 1e-12
    assert abs(mid["month_ret"] - 0.01) < 1e-12
    reb = basket_state(_reports("2026-09-30"))               # 9/30: 월말 → 교체
    assert reb["rebalance"] and reb["base"] == date(2026, 9, 30) and reb["next"] == date(2026, 10, 30)
    names = [x["ticker"] for x in reb["rows"]]
    assert {"T10", "T11"} <= set(names) and not {"T00", "T01", "T03"} & set(names)
    out = {t: why for t, why, _ in reb["out"]}
    assert out["T00"].startswith("점수 50") and out["T03"] == "규칙상 매도"
    meta = {"mode": "normal", "failed": []}
    reps = _reports("2026-09-30")
    assert "바구니 교체 +" in make_subject(reps, meta)
    assert "이번 달 바구니" in render_html(reps, meta) and "제외 T03: 규칙상 매도" in render_text(reps, meta)
    assert abs(reb["month_ret"] - 0.01) < 1e-12 and "지난달 바구니 수익률" in render_text(reps, meta)
    assert "이번 달 수익률 (8/31 종가 → 9/29 종가): +1.0%" in render_text(_reports("2026-09-29"), meta)
