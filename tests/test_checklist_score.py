import numpy as np
import pandas as pd
import pytest

from scoring import get_profile
from scoring.checklist import GROUPS, build_report
from scoring.checklist_score import (FAMILY_ITEMS, RULE, ScoreRule, checklist_score, points_from_verdicts,
                                     run_rule, strategy_returns)
from tests.test_checklist import smooth

P = get_profile("QQQ")
MAP = {"↑": 1, "↓": -1, "–": None}


def test_families_match_email_groups():
    email = {g: items for g, items, kind in GROUPS if kind == "dir"}
    assert email == FAMILY_ITEMS


@pytest.mark.parametrize("drift,seed", [(0.002, 0), (-0.002, 1), (0.0005, 2), (0.0, 3), (0.001, 4)])
def test_history_score_equals_email_score(drift, seed):
    """점수 이력의 마지막 값 = 메일 표의 색으로 센 점수 (여러 시점)."""
    df = smooth(drift, n=700, seed=seed)
    bench = smooth(0.001, n=700, seed=9)["Close"]
    for cut in (420, 560, 700):
        rep = build_report(df.iloc[:cut], P, "X", bench=bench.iloc[:cut])
        email = points_from_verdicts({g: MAP.get(v, 0) for g, v in rep.verdicts().items() if g not in ("변동성", "과열")},
                                     dict(RULE.weights))
        hist = checklist_score(df.iloc[:cut], P, bench=bench.iloc[:cut])
        assert hist["score"].iloc[-1] == pytest.approx(email)
        assert rep.score == pytest.approx(email)


def test_points_from_verdicts():
    assert points_from_verdicts({"a": 1, "b": 1, "c": 0, "d": -1, "e": None}) == pytest.approx(50 + 50 * 1 / 4)
    assert np.isnan(points_from_verdicts({"a": None}))
    # 배점: 장기 ×2, 모멘텀·거래량 ×0.5 — 장기 🟢 하나가 모멘텀·거래량 🔴 둘보다 무겁다
    w = dict(RULE.weights)
    assert w["장기 추세"] == 2 and w["모멘텀"] == w["거래량"] == 0.5
    v = {"장기 추세": 1, "중기 추세": 0, "모멘텀": -1, "추세 강도": 0, "거래량": -1, "상대강도": None}
    assert points_from_verdicts(v, w) == pytest.approx(50 + 50 * (2 - 0.5 - 0.5) / 5)
    # 장기 🟢 + 나머지 모두 🔴 = 33점 (동일 배점이면 17점) — 단기 성향만 꺾인 눌림에서 점수가 덜 흔들린다
    worst = {f: -1 for f in w} | {"장기 추세": 1}
    assert points_from_verdicts(worst, w) == pytest.approx(50 + 50 * (2 - 4) / 6)


def test_run_rule_entry_exit_and_blocks():
    s = np.array([50, 72, 75, 60, 45, 39, 72, 72, 80], dtype=float)
    above = np.array([True] * 9)
    over = np.array([False] * 6 + [True, False, False])
    hv = np.array([False] * 9)
    below = np.array([True] * 5 + [False] + [True] * 3)  # 5번째 날만 200일선 아래
    state, ev, blocked = run_rule(s, below, over, hv, RULE)
    assert ev == ["", "BUY", "", "", "", "SELL", "", "BUY", ""]
    assert blocked[6] == "과열" and state.tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1]
    # 200일선 위에서는 점수가 40 아래여도 팔지 않는다 (눌림으로 보고 보유)
    st_up, ev_up, _ = run_rule(s, above, over, hv, RULE)
    assert "SELL" not in ev_up and st_up[5] == 1
    # exit_above200 을 주면 200일선 위에서도 그 선에서 판다 (이전 규칙 = 40)
    _, ev_old, _ = run_rule(s, above, over, hv, ScoreRule(exit_above200=40))
    assert ev_old[5] == "SELL"
    # 200일선 아래면 진입하지 않는다
    _, ev2, bl2 = run_rule(np.array([50, 80.0]), np.array([False, False]), np.array([False, False]),
                           np.array([False, False]), RULE)
    assert ev2 == ["", ""] and bl2[1] == "200일선 아래"
    # 차단을 끈 규칙
    _, ev3, _ = run_rule(np.array([50, 80.0]), np.array([True, True]), np.array([False, True]),
                         np.array([False, True]), ScoreRule(block_overheat=False, block_highvol=False))
    assert ev3 == ["", "BUY"]


def test_no_look_ahead_in_score_and_state():
    df = smooth(0.0008, n=800, seed=5)
    full = checklist_score(df, P)
    for cut in (500, 650, 800):
        part = checklist_score(df.iloc[:cut], P)
        for col in ("score", "score_s", "state"):
            a, b = part[col].iloc[-1], full[col].iloc[cut - 1]
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), (col, cut)


def test_strategy_returns_next_day():
    idx = pd.bdate_range("2024-01-01", periods=4)
    out = pd.DataFrame({"close": [100, 110, 121, 121.0], "state": [1, 1, 0, 0.0]}, index=idx)
    r = strategy_returns(out, cost=0.0)
    assert r.tolist() == pytest.approx([0, 0.10, 0.10, 0])


def test_projection_is_tomorrow_average_if_score_repeats():
    from scoring.checklist_score import checklist_score
    from tests.test_checklist import smooth
    o = checklist_score(smooth(0.002, n=700), get_profile("QQQ")).dropna(subset=["score_s"])
    s = o["score"]
    expect = (s.shift(1) + 2 * s) / 3
    assert np.allclose(o["proj"].iloc[5:], expect.iloc[5:])
