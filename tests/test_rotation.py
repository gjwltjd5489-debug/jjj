import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoring import get_profile
from scoring.basket import avg_offdiag, return_corr, select_low_corr
from scoring.cards import V5, V6
from scoring.portfolio import equal_weight, perf_stats, rotate
from scoring.score import combine, compute_score
from scoring.sources import _normalize_ohlcv, rate_index, splice
from tests.test_scoring import synthetic

ROOT = Path(__file__).resolve().parents[1]


def frames(entry, exit_):
    idx = pd.bdate_range("2021-01-01", periods=len(entry))
    e = pd.DataFrame({"total_s": entry, "close": 1.0}, index=idx)
    x = pd.DataFrame({"total_s": exit_, "close": np.arange(len(entry), dtype=float), "max_used": 100.0}, index=idx)
    return e, x


def test_combo_enters_on_v3_cross_and_exits_on_v2_level():
    entry = [50, 60, 70, 72, 50, 66, 70, 40, 70]
    exit_ = [60, 60, 60, 55, 45, 39, 50, 50, 60]
    out = combine(*frames(entry, exit_), V5)
    # k=2: v3 60→70 상향 돌파 & v2 60>40 → 진입 / k=5: v2 39 ≤ 40 → 청산
    # k=8: v3 40→70 상향 돌파 → 재진입
    assert out["event"].tolist() == ["", "", "BUY", "", "", "SELL", "", "", "BUY"]
    assert out["state"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 1]
    # 점수 = 50 × 보유 + v2 / 2
    assert out["total"].tolist()[2] == 50 + 30


def test_combo_does_not_enter_when_exit_condition_true():
    out = combine(*frames([50, 70, 50, 70], [30, 35, 45, 45]), V5)
    assert out["event"].tolist() == ["", "", "", "BUY"]


def test_combo_ignores_card_alternation_rules():
    # v3 가 이미 한 번 BUY 를 낸 뒤라도, 청산 후 다시 돌파하면 재진입해야 한다
    entry = [50, 70, 80, 60, 70]
    exit_ = [50, 50, 35, 50, 50]
    out = combine(*frames(entry, exit_), V5)
    assert out["event"].tolist() == ["", "BUY", "SELL", "", "BUY"]


def test_v6_also_enters_on_trend_confirmation():
    entry = [50, 50, 50, 50]
    exit_ = [60, 65, 72, 75]
    assert combine(*frames(entry, exit_), V5)["event"].tolist() == ["", "", "", ""]
    assert combine(*frames(entry, exit_), V6)["event"].tolist() == ["", "", "BUY", ""]


def test_combo_via_compute_score():
    out = compute_score(synthetic(n=900, drift=0.001, seed=2), get_profile("QQQ"), "v5")
    assert set(out["state"].dropna().unique()) <= {0.0, 1.0}
    assert out["total"].dropna().between(0, 100).all()
    ev = out.loc[out["event"] != "", "event"].tolist()
    assert all(a != b for a, b in zip(ev, ev[1:]))


def mk(values, cols):
    idx = pd.bdate_range("2022-01-03", periods=len(values))
    return pd.DataFrame(values, index=idx, columns=cols, dtype=float)


def test_rotate_top_n_exit_and_fill():
    cols = ["A", "B", "C"]
    score = mk([[90, 80, 70], [90, 80, 70], [90, 80, 70], [90, 80, 95]], cols)
    elig = mk([[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]], cols)
    ret = mk([[0, 0, 0], [0.1, 0.2, 0.3], [0.01, 0.02, 0.03], [0, 0, 0]], cols)
    r = rotate(score, elig, ret, top_n=2, rebalance=100, buffer=0, cost=0.0)
    w = r.weights
    assert w.iloc[0].tolist() == [0.5, 0.5, 0.0]       # A, B 상위 2
    assert w.iloc[1].tolist() == [0.5, 0.0, 0.5]       # B 자격 상실 → 즉시 청산, 빈자리 C 로 채움
    assert w.iloc[3].tolist() == [0.5, 0.0, 0.5]       # 리밸런스 날이 아니면 순위 바뀌어도 교체 없음
    # 수익률은 전날 비중으로: day1 = 0.5*0.1 + 0.5*0.2
    assert r.returns.iloc[1] == pytest.approx(0.15)
    assert r.returns.iloc[2] == pytest.approx(0.5 * 0.01 + 0.5 * 0.03)


def test_rotate_rebalance_replaces_with_buffer():
    cols = ["A", "B", "C"]
    score = mk([[90, 80, 70], [60, 80, 95], [60, 80, 95]], cols)
    elig = mk([[1, 1, 1]] * 3, cols)
    ret = mk([[0, 0, 0]] * 3, cols)
    r0 = rotate(score, elig, ret, top_n=1, rebalance=1, buffer=0, cost=0.0)
    assert r0.weights.iloc[1].tolist() == [0, 0, 1]    # 매일 리밸런스, 버퍼 0 → C 로 교체
    r1 = rotate(score, elig, ret, top_n=1, rebalance=1, buffer=5, cost=0.0)
    assert r1.weights.iloc[1].tolist() == [1, 0, 0]    # 버퍼가 크면 유지


def test_rotate_cash_and_costs():
    cols = ["A"]
    score = mk([[90], [90], [90]], cols)
    elig = mk([[0], [1], [1]], cols)
    ret = mk([[0.0], [0.0], [0.1]], cols)
    cash = pd.Series(0.001, index=score.index)
    r = rotate(score, elig, ret, top_n=1, rebalance=5, cost=0.01, cash_ret=cash)
    assert r.returns.iloc[1] == pytest.approx(0.001)          # 전날 현금 100%
    assert r.returns.iloc[2] == pytest.approx(0.1 - 0.01)     # 보유 수익 − 전날 매수 비용


def test_equal_weight_constant_returns():
    ret = mk([[0.01, 0.01]] * 30, ["A", "B"])
    assert equal_weight(ret, 5, cost=0.0).iloc[-1] == pytest.approx(0.01)


def test_select_low_corr_prefers_negative_and_must():
    names = ["Q", "V", "G", "B"]
    c = pd.DataFrame([[1, .9, .1, -.3], [.9, 1, .2, -.2], [.1, .2, 1, .2], [-.3, -.2, .2, 1]], index=names, columns=names)
    assert select_low_corr(c, 3, ("Q",)) == ["Q", "B", "G"]
    assert select_low_corr(c, 2) == ["Q", "B"]
    assert avg_offdiag(c, ["Q", "B"]) == pytest.approx(-0.3)


def test_return_corr_weekly():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 0.01, 400))
    closes = pd.DataFrame({"A": np.exp(base), "B": np.exp(base * 1.0), "C": np.exp(-base)}, index=idx)
    rc = return_corr(closes)
    assert rc.loc["A", "B"] == pytest.approx(1.0) and rc.loc["A", "C"] == pytest.approx(-1.0)


def test_splice_keeps_returns_and_level():
    idx = pd.bdate_range("2020-01-01", periods=6)
    proxy = pd.DataFrame({c: [10, 11, 12, 13, 14, 15.0] for c in ("Open", "High", "Low", "Close")} | {"Volume": 1.0}, index=idx)
    primary = pd.DataFrame({c: [26, 28, 30.0] for c in ("Open", "High", "Low", "Close")} | {"Volume": 2.0}, index=idx[3:])
    out = splice(primary, proxy)
    assert len(out) == 6 and out["Close"].iloc[3] == 26
    # 연결 시점 비율 26/13 = 2 로 조정 → 이전 수익률 보존
    assert out["Close"].iloc[:3].tolist() == [20, 22, 24]


def test_adjusted_ohlc():
    idx = pd.bdate_range("2020-01-01", periods=2)
    raw = pd.DataFrame({"Open": [10, 10.0], "High": [11, 11.0], "Low": [9, 9.0], "Close": [10, 10.0],
                        "Adj Close": [9, 10.0], "Volume": [1, 1]}, index=idx)
    df = _normalize_ohlcv(raw)
    assert df["Close"].tolist() == [9, 10] and df["High"].iloc[0] == pytest.approx(9.9)
    assert _normalize_ohlcv(raw, adjust=False)["Close"].tolist() == [10, 10]


def test_rate_index_carry_and_duration():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    flat = rate_index(pd.Series([0.05, 0.05, 0.05], index=idx))
    assert flat.iloc[-1] / flat.iloc[0] - 1 == pytest.approx((1 + 0.05 * 366 / 365) * (1 + 0.05) - 1, rel=1e-6)
    up = rate_index(pd.Series([0.05, 0.06, 0.06], index=idx), duration=20)
    assert up.iloc[1] < up.iloc[0]


def test_perf_stats_basic():
    st = perf_stats(pd.Series([0.01, -0.02, 0.01, 0.0]))
    assert st["mdd"] == pytest.approx(-0.02)


def test_rotate_basket_script_smoke(tmp_path):
    for k, (t, drift) in enumerate({"AAA": 0.0008, "BBB": -0.0002, "CCC": 0.0004}.items()):
        df = synthetic(n=800, drift=drift, seed=10 + k).iloc[::-1]
        lines = ['"Date","Price","Open","High","Low","Vol.","Change %"']
        for d, r in df.iterrows():
            lines.append(f'"{d:%m/%d/%Y}","{r.Close:.2f}","{r.Open:.2f}","{r.High:.2f}","{r.Low:.2f}","{r.Volume / 1e6:.2f}M","0%"')
        (tmp_path / f"{t}.csv").write_text("\n".join(lines))
    cmd = [sys.executable, str(ROOT / "scripts" / "rotate_basket.py"), "--select-from", "AAA", "BBB", "CCC",
           "--k", "3", "--top", "2", "--source", str(tmp_path / "{t}.csv")]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "v5 상위 2" in res.stdout and "현재 보유안" in res.stdout
