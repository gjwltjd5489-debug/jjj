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
