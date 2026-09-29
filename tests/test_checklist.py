import numpy as np
import pandas as pd

from scoring import get_profile
from scoring.checklist import DOWN, NEUTRAL, UP, build_report, render_html, render_markdown
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
    assert up.ups >= 8 and down.downs >= 8
    assert up.check("200일선").status == UP and down.check("200일선").status == DOWN
    md = render_markdown([up, down])
    page = render_html([up, down])
    assert "| UP |" in md and "오늘의 이벤트" in md
    assert "<table" in page and "UP" in page


def test_missing_volume_marks_neutral():
    r = build_report(synthetic(n=600, seed=2).assign(Volume=np.nan), P, "IDX")
    assert r.check("VR(20)").status == NEUTRAL


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
    assert "20일선" in names and r.prev_ups > r.ups


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
    assert make_subject([r], normal).startswith("[미장 체크리스트]")
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
