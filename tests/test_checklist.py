import numpy as np
import pandas as pd

from scoring import get_profile
from scoring.checklist import DOWN, UP, build_report, render_html, render_markdown
from tests.test_scoring import synthetic

P = get_profile("QQQ")


def smooth(drift, n=600, seed=0):
    """잡음이 작아 마지막 날도 추세 방향이 분명한 시계열."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.002, n)))
    vol = np.where(np.diff(close, prepend=close[0]) * np.sign(drift) > 0, 2e6, 1e6)
    return pd.DataFrame({"Open": close, "High": close * 1.002, "Low": close * 0.998, "Close": close, "Volume": vol},
                        index=pd.bdate_range("2020-01-01", periods=n))


def test_uptrend_mostly_bullish_and_renders():
    up = build_report(smooth(0.002), P, "UP", "상승", "테스트")
    down = build_report(smooth(-0.002), P, "DN", "하락", "테스트")
    assert up.ups >= 3 and down.downs >= 4
    assert up.check("200일선").status == UP and down.check("200일선").status == DOWN
    assert up.verdicts()["장기 추세"] == "↑" and down.verdicts()["장기 추세"] == "↓"
    md = render_markdown([up, down])
    page = render_html([up, down])
    assert "| UP |" in md and "오늘의 이벤트" in md and "성향별 판정" in md
    assert "<table" in page and "UP" in page


def test_missing_volume_marks_neutral():
    r = build_report(synthetic(n=600, seed=2).assign(Volume=np.nan), P, "IDX")
    assert r.check("VR").na and r.verdicts()["거래량"] == "–"


def test_detects_20ma_breakout_event():
    df = synthetic(n=400, drift=-0.001, seed=4)
    df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-30:].max() * 1.1
    r = build_report(df, P, "X")
    assert "20일선 상향 돌파" in r.events


# ---------------------------------------------------------------- 휴장일, 변화, 데이터 점검

from datetime import date, datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from scoring.checklist import build_with_history, data_issues, make_subject, render_text  # noqa: E402
from scoring.market_calendar import nyse_holidays, session_status  # noqa: E402

KST = ZoneInfo("Asia/Seoul")


def test_nyse_holidays_known_dates():
    h26 = nyse_holidays(2026)
    for d in ["2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
              "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]:
        assert date.fromisoformat(d) in h26, d
    assert date(2027, 12, 24) in nyse_holidays(2027)          # 성탄절 토요일 → 금요일
    assert date(2021, 12, 31) not in nyse_holidays(2022)      # 토요일 신정은 당겨 쉬지 않음
    assert date(2027, 3, 26) in nyse_holidays(2027)           # 성금요일


def test_session_status_normal_holiday_and_before_close():
    s = session_status(datetime(2026, 9, 30, 7, 37, tzinfo=KST))
    assert s["target"] == date(2026, 9, 29) and s["holiday"] is None
    h = session_status(datetime(2026, 11, 27, 7, 37, tzinfo=KST))
    assert h["holiday"] == "추수감사절" and h["target"] == date(2026, 11, 25) and h["next_open"] == date(2026, 11, 27)
    early = session_status(datetime(2026, 9, 29, 12, 0, tzinfo=ZoneInfo("America/New_York")))
    assert early["target"] == date(2026, 9, 28)               # 장중에는 전날이 마지막 정규장


def test_changes_vs_previous_session():
    df = smooth(0.002)
    df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-2] * 0.9   # 마지막 날 급락
    r = build_with_history(df, P, "X")
    names = {now.name for _, now in r.changes}
    assert "구름" in names or "전환/기준" in names or "RSI" in names
    assert r.prev_ups > r.ups


def test_data_issues_jump_volume_and_stale():
    df = smooth(0.002)
    df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-2] * 1.5
    df.iloc[-1, df.columns.get_loc("High")] = df["Close"].iloc[-1]
    df.iloc[-1, df.columns.get_loc("Volume")] = 0
    r = build_report(df, P, "X")
    issues = data_issues(df, r, expected=date(2099, 1, 1))
    text = " ".join(issues)
    assert "급변" in text and "거래량 없음" in text and "데이터 지연" in text


def test_subjects_and_plain_text():
    r = build_with_history(smooth(0.002), P, "UP")
    normal = {"mode": "normal", "failed": []}
    assert make_subject([r], normal).startswith("[미장] ") and len(make_subject([r], normal)) <= 90
    hol = {"mode": "holiday", "failed": [], "checked": "2026-11-26", "holiday": "추수감사절",
           "next_open": "2026-11-27", "target": "2026-11-25"}
    assert make_subject([r], hol).startswith("[휴장]")
    assert "휴장" in render_html([r], hol) and "전일 대비 변화" not in render_html([r], hol)
    delayed = {"mode": "delayed", "failed": [], "target": "2099-01-01"}
    assert make_subject([r], delayed).startswith("[데이터 지연]")
    r.issues = ["테스트 경고"]
    assert make_subject([r], normal).startswith("[점검 필요]")
    text = render_text([r], normal)
    assert "**" not in text and "|" not in text and "#" not in text


def test_v2_groups_are_one_or_two_items_and_no_duplicates():
    from scoring.checklist import GROUPS
    names = [n for _, items, _ in GROUPS for n in items]
    assert len(names) == len(set(names)) and all(1 <= len(items) <= 2 for _, items, _ in GROUPS)
    r = build_report(smooth(0.002), P, "UP")
    assert {c.name for c in r.checks} == set(names)
    assert "%B" not in names and "50일선" not in names  # 중복 항목 제거


def test_relative_strength_needs_benchmark():
    df = smooth(0.002)
    no = build_report(df, P, "A")
    assert no.verdicts()["상대강도"] == "–"
    weak_bench = df["Close"] * 0.5
    strong_bench = pd.Series(100 * np.exp(np.linspace(0, 3, len(df))), index=df.index)
    assert build_report(df, P, "A", bench=weak_bench).check("상대강도").status in (UP, DOWN)
    assert build_report(df, P, "A", bench=strong_bench).verdicts()["상대강도"] == "↓"


def test_rsi_overbought_is_not_a_warning():
    df = smooth(0.004)
    r = build_report(df, P, "X")
    rsi = r.check("RSI")
    assert rsi.status == UP  # 강한 추세의 RSI≥70 은 ⚠️ 가 아니라 ✅


def test_vr_capped_when_no_down_days():
    from scoring.indicators import VR_CAP, volume_ratio
    df = pd.DataFrame({"Close": np.arange(1, 30, dtype=float), "Volume": 1.0})
    assert volume_ratio(df, 20).iloc[-1] == VR_CAP


def test_trade_signal_groups_buy_and_sell():
    from scoring.checklist import TickerReport, signal_groups
    base = dict(name="", group="", date=pd.Timestamp("2026-09-29"), close=100.0, change=0.0, from_high=0.0, checks=[])

    def sig(**kw):
        d = {"score": 50.0, "score_s": 50.0, "held": False, "event": "", "blocked": "", "stop": 95.0, "close": 100.0}
        d.update(kw)
        return d
    reps = [
        TickerReport(ticker="B", sig=sig(score=83, score_s=75, held=True, event="BUY"), **base),
        TickerReport(ticker="S", sig=sig(score=33, score_s=38, event="SELL"), **base),
        TickerReport(ticker="K", sig=sig(score=83, score_s=80, blocked="과열"), **base),
        TickerReport(ticker="N", sig=sig(score=50, score_s=45, held=True), **base),
        TickerReport(ticker="H", sig=sig(score=83, score_s=83, held=True), **base),
    ]
    g = signal_groups(reps)
    assert [r.ticker for r, _ in g["buy"]] == ["B"] and "-5.0%" in g["buy"][0][1]
    assert [r.ticker for r, _ in g["sell"]] == ["S"]
    assert [r.ticker for r, _ in g["blocked"]] == ["K"] and "과열" in g["blocked"][0][1]
    assert [r.ticker for r, _ in g["near_exit"]] == ["N"] and [r.ticker for r, _ in g["hold"]] == ["H"]


def test_trade_signal_on_real_like_series():
    r = build_with_history(smooth(0.002, n=700), P, "UP")
    assert r.sig is not None and r.sig["stop"] < r.sig["close"]
    assert abs(r.sig["score"] - r.score) < 1e-9  # 신호 점수 = 메일 표 색으로 센 점수
    assert "매수·매도 신호" in render_html([r], {"mode": "normal", "failed": []})


# ---------------------------------------------------------------- 메일 개편 (색 변화만, 임박 신호, 보유 표, 기록)

def _rep(ticker="X", **sig):
    from scoring.checklist import TickerReport
    d = {"score": 50.0, "score_s": 50.0, "prev_score_s": 50.0, "proj": 50.0, "held": False, "event": "", "blocked": "",
         "stop": 95.0, "close": 100.0, "above200": True, "overheat": False, "highvol": False, "recent": []}
    d.update(sig)
    return TickerReport(ticker=ticker, name="", group="", date=pd.Timestamp("2026-09-29"), close=100.0, change=0.0,
                        from_high=0.0, checks=[], sig=d)


def test_near_signals_use_tomorrow_projection():
    from scoring.checklist import make_subject, signal_groups
    reps = [_rep("NB", score=83, score_s=64, proj=72),                       # 내일 진입 가능
            _rep("NR", score=83, score_s=64, proj=72, above200=False),       # 가능하지만 200일선 아래
            _rep("NE", score=33, score_s=55, proj=39, held=True),            # 내일 퇴출 가능
            _rep("H", score=83, score_s=80, proj=81, held=True)]
    g = signal_groups(reps)
    assert [r.ticker for r, _ in g["near_buy"]] == ["NB", "NR"] and "진입 보류" in g["near_buy"][1][1]
    assert [r.ticker for r, _ in g["near_exit"]] == ["NE"] and "매도 신호" in g["near_exit"][0][1]
    assert [r.ticker for r, _ in g["hold"]] == ["H"]
    subj = make_subject(reps, {"mode": "normal", "failed": []})
    assert subj.startswith("[미장] 9/29 · 매수 없음 · 매도 없음") and "NB↑" in subj and "NE↓" in subj


def test_holdings_table_shows_stop_and_room():
    from scoring.checklist import holdings, render_html, render_text
    reps = [_rep("A", score_s=75, held=True, stop=96.0), _rep("B", score_s=90, held=True, event="BUY"), _rep("C")]
    assert [r.ticker for r in holdings(reps)] == ["B", "A"]
    text = render_text(reps, {"mode": "normal", "failed": []})
    assert "A 75 · 종가 100.00 · 손절 참고 96.00 (-4.0%) · 퇴출선까지 35점" in text


def test_color_only_changes_and_no_hover_text():
    from scoring.checklist import Check, _verdict_changes, change_lines, render_html
    r = build_with_history(smooth(0.002), P, "UP")
    adx = r.check("ADX")
    prev = [Check(c.group, c.name, c.status, c.detail, c.tag, c.short, c.na) for c in r.checks]
    # 같은 🟠 안의 변화(횡보→약함)는 변화로 세지 않는다
    r.checks = [c if c.name != "ADX" else Check(c.group, c.name, "➖", "22 추세 약함", "약함", "ADX 22") for c in r.checks]
    r.prev_checks = [c if c.name != "ADX" else Check(c.group, c.name, "➖", "15 횡보", "횡보", "ADX 15") for c in prev]
    assert _verdict_changes(r) == []
    # 색이 바뀌면(🟠→🟢) 바뀐 항목의 현재 수치와 함께 보인다
    r.checks = [c if c.name != "ADX" else Check(c.group, c.name, UP, "30 강한 상승추세", "", "ADX 30") for c in r.checks]
    assert [g for g, _, _ in _verdict_changes(r)] == ["추세 강도"]
    summary, lines = change_lines([r])
    assert summary.startswith("색이 바뀐 성향 1건") and "(ADX 30)" in lines[0]
    page = render_html([r], {"mode": "normal", "failed": []})
    assert "마우스" not in page and "docs/checklist_score.md" in page and "상세" not in page
    assert adx is not None


def test_recent_signal_log_and_summary():
    from scoring.checklist import recent_signals, recent_summary
    reps = [_rep("A", recent=[{"date": pd.Timestamp("2026-09-10"), "event": "BUY", "close": 90.0, "ret": 0.1}]),
            _rep("B", recent=[{"date": pd.Timestamp("2026-09-20"), "event": "SELL", "close": 110.0, "ret": -0.05}])]
    items = recent_signals(reps)
    assert [r.ticker for r, _ in items] == ["B", "A"]  # 최신순
    s = recent_summary(items)
    assert "매수 1건: 신호 뒤 평균 +10.0% (오른 것 1건)" in s and "매도 1건: 신호 뒤 평균 -5.0% (매도 뒤 더 내린 것 1건)" in s


def test_market_lines_fx_and_calendar():
    from scoring.checklist import market_lines
    r = build_with_history(smooth(0.002), P, "UP")
    meta = {"fx": {"rate": 1355.5, "d1": -0.003, "m1": 0.012},
            "upcoming": [("10/14(수)", "CPI 발표 (9월분) 08:30 ET")]}
    lines = market_lines([r], meta)
    assert lines[0].startswith("1종목 중 200일선 위 1")
    assert "원/달러 1,355.5원 (전일 대비 -0.3% · 1개월 +1.2%)" in lines
    assert lines[-1] == "다가오는 일정 (2주): 10/14(수) CPI 발표 (9월분) 08:30 ET"
