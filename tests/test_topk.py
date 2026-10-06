import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoring.topk import (basket_weights, dip_target, mid_month, month_end, next_month_end, next_rebalance,
                         prev_month_end, prev_rebalance, trim_bottom)


def test_basket_weights_top_k_ties_and_cash():
    sc = {"A": 90.0, "B": 80.0, "C": 80.0, "D": 70.0, "E": 55.0, "F": 95.0}
    held = {t: True for t in sc} | {"F": False}          # F 는 규칙상 보유가 아니라 제외
    # 동점 B·C: 최근 60거래일 수익률이 높은 C 가 들어간다 (자리를 나누지 않음). 수익률이 없으면 알파벳 순
    assert basket_weights(sc, held, n=60, k=2, mom={"B": 0.05, "C": 0.12}) == {"A": 0.5, "C": 0.5}
    assert basket_weights(sc, held, n=60, k=2) == {"A": 0.5, "B": 0.5}
    # 9/30 실제 사례: 5위 동점(NVDA·SOXX 92) → 5개만
    real = {"AMD": 100, "UUP": 100, "PLTR": 99.0, "DBMF": 93.3, "NVDA": 91.7, "SOXX": 91.7}
    w = basket_weights(real, {t: True for t in real}, mom={"NVDA": 0.21, "SOXX": 0.26})
    assert len(w) == 5 and "SOXX" in w and "NVDA" not in w and all(v == 0.2 for v in w.values())
    w = basket_weights(sc, held, n=60, k=10)             # 후보 4개 → 10%씩, 현금 60%
    assert set(w) == {"A", "B", "C", "D"} and abs(sum(w.values()) - 0.4) < 1e-12
    # 중복 제거: IWM·VTV·SCHD 는 점수가 높아도 후보에서 빠진다
    w = basket_weights({**sc, "IWM": 100.0, "VTV": 99.0}, {**held, "IWM": True, "VTV": True}, n=60, k=2)
    assert w == {"A": 0.5, "B": 0.5}


def test_dip_ladder_and_trim():
    # SPY 200일선 대비 −4.9% → 0, −5% → 5%, −12% → 10%, −80% → 최대 50%
    assert [dip_target(x) for x in (0.0, 0.049, 0.05, 0.12, 0.8, float("nan"))] == [0.0, 0.0, 0.05, 0.10, 0.5, 0.0]
    w = {"A": 0.2, "B": 0.2, "C": 0.1}                     # 순위 순 → 하위(C)부터 줄인다
    assert trim_bottom(w, 0.15) == pytest.approx(0.15) and w == {"A": 0.2, "B": pytest.approx(0.15)}


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
    t03 = next(x for x in mid["rows"] if x["ticker"] == "T03")
    # 교체 사이 규칙 매도: 9/16 종가에 팔아 현금 (다음 교체일까지 들고 있지 않음)
    assert not t03["held_now"] and t03["sold"] == date(2026, 9, 16) and abs(t03["ret"]) < 1e-12
    assert abs(mid["cash_now"] - 0.2 / 1.02) < 1e-9
    # 이번 기간 수익률: T04 +10%, 나머지 0% → 바구니 +2% (20%씩), T03 판 비용 0.05% (9/16 비중 0.2/1.02)
    assert abs(next(x for x in mid["rows"] if x["ticker"] == "T04")["ret"] - 0.10) < 1e-12
    sell_cost = 0.2 / 1.02 * 0.0005
    assert abs(mid["month_ret"] - (0.02 - sell_cost)) < 1e-12
    # 편입 후 수익률: T04 는 8/14·8/31·9/15 바구니에 계속 있었다 → 데이터 첫 교체일(8/14)부터, 더 이전은 데이터 없음 표시
    t04 = next(x for x in mid["rows"] if x["ticker"] == "T04")
    assert t04["since"] == date(2026, 8, 14) and t04["since_cut"] and abs(t04["ret_hold"] - 0.10) < 1e-12
    reb = basket_state(_reports("2026-09-30"))               # 9/30: 월말 → 교체
    assert reb["rebalance"] and reb["base"] == date(2026, 9, 30) and reb["prev"] == date(2026, 9, 15)
    assert reb["next"] == date(2026, 10, 15)
    names = [x["ticker"] for x in reb["rows"]]
    assert {"T05", "T06"} <= set(names) and not {"T00", "T03"} & set(names)
    out = {t: why for t, why, _ in reb["out"]}
    assert out["T00"].startswith("점수 50") and out["T03"] == "규칙상 매도 (9/16 현금화)"
    meta = {"mode": "normal", "failed": []}
    reps = _reports("2026-09-30")
    assert "바구니 교체 +2 −2" in make_subject(reps, meta)
    assert "🧺 바구니" in render_html(reps, meta) and "제외 T03: 규칙상 매도" in render_text(reps, meta)
    assert abs(reb["month_ret"] - (0.02 - sell_cost)) < 1e-12
    assert "지난 바구니 수익률 (9/15 → 9/30 종가): +2.0%" in render_text(reps, meta)
    assert "이번 기간 수익률 (9/15 종가 → 9/29 종가): +2.0%" in render_text(_reports("2026-09-29"), meta)
    text29 = render_text(_reports("2026-09-29"), meta)
    assert "T04 20.0% · 점수 83 → 오늘 86 · 편입 후 +10.0% (~8/14~)" in text29
    assert "T03 20.0% · 점수 86 → 오늘 87 · 🔻 9/16 규칙 매도 → 현금 · 매도까지 +0.0%" in text29
    assert "현금 0% → 교체 사이 매도 뒤 20%" in text29
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


def test_mail_simulation_matches_backtest_with_mid_exit_and_cash():
    """메일의 바구니 계산(_simulate)과 검증 엔진(Book.run mid_exit·rf)이 같은 값을 내는지."""
    from scoring.checklist import TickerReport, _simulate
    from scoring.market_calendar import is_trading_day
    spec = importlib.util.spec_from_file_location("eval_topk", Path(__file__).parents[1] / "scripts" / "eval_topk.py")
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    idx = pd.DatetimeIndex([d for d in pd.bdate_range("2026-01-15", "2026-06-30") if is_trading_day(d.date())])
    rng = np.random.default_rng(3)
    cols = [f"T{i:02d}" for i in range(9)]
    C = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, (len(idx), 9)), axis=0)), index=idx, columns=cols)
    S = pd.DataFrame(rng.choice([55, 62.5, 75, 83.3, 91.7], size=(len(idx), 9)), index=idx, columns=cols)
    H = pd.DataFrame((rng.random((len(idx), 9)) > 0.15).astype(float), index=idx, columns=cols)
    rf = pd.Series(0.04 / 252, index=idx)
    reps = [TickerReport(ticker=t, name="", group="", date=idx[-1], close=float(C[t].iloc[-1]), change=0.0, from_high=0.0,
                         checks=[], sig={"hist": pd.DataFrame({"score_s": S[t], "state": H[t], "close": C[t]})})
            for t in cols]
    sim = _simulate(reps, idx[0].date(), idx[-1].date(), rf)
    rfa = rf.to_numpy().copy()
    rfa[0] = 0.0   # 엔진은 첫날 현금으로 시작해 그날 이자가 붙는다 — 메일 계산은 첫날 종가에 바구니를 짠 뒤부터
    out, _, _ = et.Book(S, C, H).run(60, 5, -1, True, mid_exit=True, rf=rfa)
    assert sim["sells"] and abs(sim["value"] - float((1 + out).prod())) < 1e-12


def test_subject_and_cash_interest():
    from scoring.checklist import TickerReport, basket_ytd, make_subject
    # 9/16: T03 이 교체 다음 날 규칙 매도 → 제목에 '바구니 매도'
    assert "바구니 매도 T03" in make_subject(_reports("2026-09-16"), {"mode": "normal", "failed": []})
    # 후보가 없으면 전부 현금 → 올해 수익률 = 단기금리 복리
    idx = pd.bdate_range("2025-12-01", "2026-03-31")
    hist = pd.DataFrame({"score_s": 30.0, "state": 0.0, "close": 100.0}, index=idx)
    rep = TickerReport(ticker="T00", name="", group="", date=idx[-1], close=100.0, change=0.0, from_high=0.0, checks=[],
                       sig={"hist": hist})
    rf = pd.Series(0.0002, index=idx)
    y = basket_ytd([rep], date(2026, 3, 31), rf=rf)
    days = ((idx > "2025-12-31") & (idx <= "2026-03-31")).sum()
    assert abs(y["ret"] - (1.0002 ** days - 1)) < 1e-12


def test_sold_then_rebought_restarts_entry():
    """교체 사이에 팔았다가 다음 교체일에 다시 들어온 종목은 편입일이 그 교체일 (편입 후 0%)."""
    from scoring.checklist import TickerReport, basket_state
    idx = pd.bdate_range("2026-08-03", "2026-09-30")
    reps = []
    for i in range(5):
        state = np.ones(len(idx))
        close = np.full(len(idx), 100.0)
        if i == 0:   # 9/16~9/22 규칙 매도 구간, 그 사이 가격 −10%, 9/23 부터 다시 보유
            state[(idx >= "2026-09-16") & (idx <= "2026-09-22")] = 0.0
            close[idx >= "2026-09-17"] = 90.0
        hist = pd.DataFrame({"score_s": 90.0 - i, "state": state, "close": close}, index=idx)
        sig = {"score": 90.0, "score_s": 90.0 - i, "prev_score_s": 90.0, "proj": 90.0, "held": True, "event": "",
               "blocked": "", "stop": 95.0, "close": close[-1], "above200": True, "ma200": 90.0, "overheat": False,
               "highvol": False, "recent": [], "hist": hist}
        reps.append(TickerReport(ticker=f"T{i:02d}", name="", group="", date=idx[-1], close=close[-1], change=0.0,
                                 from_high=0.0, checks=[], sig=sig))
    b = basket_state(reps)
    t0 = next(x for x in b["rows"] if x["ticker"] == "T00")
    assert b["rebalance"] and t0["since"] == date(2026, 9, 30) and abs(t0["ret_hold"]) < 1e-12
    assert abs(t0["ret"]) < 1e-12          # 지난 기간: 9/16 종가(100)에 팔아 손실 없음


def _overlay_case(drop: float = 0.992):
    """SPY 가 2~3월에 매일 drop 배씩 빠졌다가 4월부터 회복하는 가짜 시장 (QLD = SPY 2배).
    0.992 → 200일선 −24% 까지 (QLD 손절), 0.997 → −10% 남짓 (회복 때 청산)."""
    from scoring.market_calendar import is_trading_day
    days = pd.DatetimeIndex([d for d in pd.bdate_range("2025-01-02", "2026-09-30") if is_trading_day(d.date())])
    t = np.arange(len(days))
    path = np.where(days < "2026-02-02", 1 + 0.0004 * t, np.nan)
    i0 = int(np.flatnonzero(days >= "2026-02-02")[0])
    lvl = path[i0 - 1]
    for i in range(i0, len(days)):           # 2~3월 하락 → 4월 이후 회복
        lvl *= drop if days[i] < pd.Timestamp("2026-04-01") else 1.006
        path[i] = lvl
    spy = pd.Series(100 * path, index=days)
    qld = pd.Series(100 * np.cumprod(1 + 2 * spy.pct_change().fillna(0.0)), index=days)
    return days, spy, qld


@pytest.mark.parametrize("drop, last", [(0.992, "dip_stop"), (0.997, "dip_exit")])
def test_mail_overlay_matches_backtest(drop, last):
    """SPY 200일선 필터·QLD 하락 매수까지 메일 계산(_simulate)과 검증 엔진(Book.run)이 같은 값을 내는지."""
    from scoring.checklist import TickerReport, _simulate
    spec = importlib.util.spec_from_file_location("eval_topk", Path(__file__).parents[1] / "scripts" / "eval_topk.py")
    et = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(et)
    days, spy, qld = _overlay_case(drop)
    idx = days[days >= "2026-01-15"]
    rng = np.random.default_rng(5)
    cols = [f"T{i:02d}" for i in range(9)] + ["IWM"]
    C = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0, 0.015, (len(idx), 10)), axis=0)), index=idx, columns=cols)
    S = pd.DataFrame(rng.choice([62.5, 75, 83.3, 91.7, 100.0], size=(len(idx), 10)), index=idx, columns=cols)
    S["IWM"] = 100.0                                     # 점수 1등이어도 바구니 후보에서 빠진다
    H = pd.DataFrame((rng.random((len(idx), 10)) > 0.05).astype(float), index=idx, columns=cols)
    group = {c: ("지수" if i % 2 == 0 else "원자재·통화") for i, c in enumerate(cols)}
    rf = pd.Series(0.04 / 252, index=idx)
    reps = [TickerReport(ticker=c, name="", group=group[c], date=idx[-1], close=float(C[c].iloc[-1]), change=0.0,
                         from_high=0.0, checks=[], sig={"hist": pd.DataFrame({"score_s": S[c], "state": H[c], "close": C[c]})})
            for c in cols]
    ov = {"spy": spy, "qld": qld, "group": group}
    sim = _simulate(reps, idx[0].date(), idx[-1].date(), rf, ov=ov)
    dips = [k for _, k, _ in sim["events"] if k.startswith("dip")]
    assert dips[0] == "dip_buy" and dips[-1] == last and "IWM" not in sim["w"]
    rfa = rf.to_numpy().copy()
    rfa[0] = 0.0
    ma = spy.rolling(200).mean()
    dist = (1 - spy / ma).reindex(idx).to_numpy()
    rq = qld.pct_change().reindex(idx).fillna(0.0).to_numpy()
    book = et.Book(S, C, H)
    out, _, _ = book.run(60, 5, -1, True, mid_exit=True, rf=rfa, exclude=np.array([c == "IWM" for c in cols]),
                         equity=np.array([group[c] == "지수" for c in cols]), dist=dist, rq=rq)
    assert [k for _, k, _ in book.events] == [k for _, k, _ in sim["events"] if k.startswith("dip")]
    assert abs(sim["value"] - float((1 + out).prod())) < 1e-12
    # 다음 날 시가 체결: 시가가 전일 종가와 같으면(밤사이 0) 종가 체결과 거의 같아야 한다
    book.run(60, 5, -1, True, mid_exit=True, rf=rfa, exclude=np.array([c == "IWM" for c in cols]),
             equity=np.array([group[c] == "지수" for c in cols]), dist=dist, rq=rq, record=True)
    g2 = 1 + C.pct_change().fillna(0.0).to_numpy()
    ro = et.next_open_returns(book.rec, np.ones_like(g2), g2, np.ones(len(idx)), 1 + rq, rfa)
    assert abs(float(np.prod(1 + ro)) / float((1 + out).prod()) - 1) < 1e-3


def test_mail_shows_filter_and_qld():
    """SPY 200일선 아래: 주식 종목 절반 표시, QLD 매수 신호가 제목·본문에 나오고 회복하면 청산."""
    from scoring.checklist import TickerReport, basket_state, make_subject, render_html, render_text
    days, spy, qld = _overlay_case(0.997)
    group = {"T00": "지수", "T01": "빅테크", "T02": "원자재·통화", "T03": "채권·부동산", "T04": "대체", "VTV": "섹터·스타일"}

    def reps(last: str):
        idx = days[days <= last]
        out = []
        for i, t in enumerate(group):
            hist = pd.DataFrame({"score_s": 95.0 - i, "state": 1.0, "close": 100.0}, index=idx)
            sig = {"score": 90.0, "score_s": 95.0 - i, "prev_score_s": 90.0, "proj": 90.0, "held": True, "event": "",
                   "blocked": "", "stop": 95.0, "close": 100.0, "above200": True, "ma200": 90.0, "overheat": False,
                   "highvol": False, "recent": [], "hist": hist}
            out.append(TickerReport(ticker=t, name="", group=group[t], date=idx[-1], close=100.0, change=0.0,
                                    from_high=0.0, checks=[], sig=sig))
        return out
    ov = {"spy": spy, "qld": qld, "group": group}
    meta = {"mode": "normal", "failed": [], "overlay": ov}
    dist = 1 - spy / spy.rolling(200).mean()
    first_buy = dist.index[dist >= 0.05][0].date()            # SPY 가 200일선보다 5% 아래로 처음 내려간 날
    b = basket_state(reps(str(first_buy)), ov=ov)
    assert ("dip_buy", (0.05, pytest.approx(float(dist.loc[str(first_buy)])))) in b["ov"]["today"]
    assert "QLD +5%" in make_subject(reps(str(first_buy)), meta)
    text = render_text(reps(str(first_buy)), meta)
    assert "🟢 QLD 5% 매수 신호" in text and "주식 종목은 절반(10%)" in text
    assert "VTV" not in [x["ticker"] for x in b["rows"]]       # 중복 제거
    # 그다음 교체일: 주식 종목(T00·T01)은 10%, 나머지는 20%, QLD 몫은 현금에서
    nxt = next_rebalance(first_buy)
    b2 = basket_state(reps(str(nxt)), ov=ov)
    w = {x["ticker"]: x["weight"] for x in b2["rows"]}
    assert w["T00"] == pytest.approx(0.1) and w["T01"] == pytest.approx(0.1) and w["T02"] == pytest.approx(0.2)
    assert b2["ov"]["q_base"] > 0 and "½ 필터" in render_html(reps(str(nxt)), meta)
    # 200일선 회복 날: QLD 청산 신호
    back = dist.index[(dist.index > pd.Timestamp(nxt)) & (dist <= 0)][0].date()
    assert "QLD 청산" in make_subject(reps(str(back)), meta)
    assert basket_state(reps(str(back)), ov=ov)["ov"]["q_now"] == 0.0
