import sys
import types

import numpy as np
import pandas as pd
import pytest

from scoring import get_profile
from scoring import indicators as ind
from scoring.cards import CARDS, V1, V3, V4, band, linear
from scoring.data import load_series_csv
from scoring.evaluate import positions_from_events, strategy_stats
from scoring.score import compute_all, compute_score
from scoring.sources import load_prices, load_series
from tests.test_scoring import synthetic

P = get_profile("QQQ")


def bench_for(df, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(0.0003 + rng.normal(0, 0.008, len(df)))), index=df.index)


def ext_for(df, seed=2):
    rng = np.random.default_rng(seed)
    n = len(df)
    return {
        "vix": pd.Series(18 + np.cumsum(rng.normal(0, 0.5, n)).clip(-8, 30), index=df.index),
        "us10y": pd.Series(3 + np.cumsum(rng.normal(0, 0.03, n)), index=df.index),
        "hy": pd.Series(4 + np.cumsum(rng.normal(0, 0.02, n)).clip(-2, 6), index=df.index),
    }


@pytest.mark.parametrize("name", list(CARDS))
def test_card_items_within_bounds_and_total_0_100(name):
    df = synthetic(n=900, drift=0.0004, seed=5)
    out = compute_score(df, P, name, bench=bench_for(df), ext=ext_for(df))
    card = CARDS[name]
    for it in card.items:
        col = out[it.key].dropna()
        assert len(col) > 0, it.key
        assert (col >= -1e-9).all() and (col <= it.max + 1e-9).all(), it.key
    total = out["total"].dropna()
    assert len(total) > 0 and total.between(0, 100).all()
    assert (out["max_used"].dropna() == 100).all()


@pytest.mark.parametrize("name", list(CARDS))
def test_no_look_ahead(name):
    """앞부분만 넣고 계산한 점수가 전체 데이터로 계산한 같은 날 점수와 같아야 한다."""
    df = synthetic(n=800, drift=0.0003, seed=9)
    bench, ext = bench_for(df), ext_for(df)
    full = compute_score(df, P, name, bench=bench, ext=ext)
    for cut in (500, 650, 799):
        part = compute_score(df.iloc[:cut], P, name, bench=bench.iloc[:cut], ext={k: v.iloc[:cut] for k, v in ext.items()})
        a, b = part["total"].iloc[-1], full["total"].iloc[cut - 1]
        assert a == pytest.approx(b, abs=1e-6), (name, cut)


def test_optional_categories_rescale():
    df = synthetic(n=700, drift=0.002, seed=4)
    v1 = compute_score(df, P, V1)  # 벤치마크 없음 → 상대강도 10점 제외
    assert (v1["max_used"].dropna() == 90).all()
    v4 = compute_score(df, P, V4)  # 매크로 없음 → 25점 제외
    assert (v4["max_used"].dropna() == 75).all()
    no_vol = df.assign(Volume=np.nan)
    v4n = compute_score(no_vol, P, V4)  # 거래량도 없음 → 60점 만점
    assert (v4n["max_used"].dropna() == 60).all()
    assert v4n["total"].dropna().between(0, 100).all()


def test_v3_gate_caps_score_in_downtrend():
    df = synthetic(n=800, drift=-0.002, seed=3)
    out = compute_score(df, P, V3)
    down = out[out["gate"] == False].dropna(subset=["total"])  # noqa: E712
    assert len(down) > 100
    assert (down["total"] <= V3.gate_cap).all()


def test_strong_uptrend_scores_high_downtrend_low():
    up = synthetic(n=900, drift=0.003, seed=1)
    down = synthetic(n=900, drift=-0.003, seed=1)
    for name in ("v1", "v2", "v4"):
        hi = compute_score(up, P, name)["total"].dropna().tail(200).mean()
        lo = compute_score(down, P, name)["total"].dropna().tail(200).mean()
        assert hi > 60 and lo < 30, (name, hi, lo)


def test_compute_all_matches_single():
    df = synthetic(n=700, seed=8)
    allres = compute_all(df, P, ["v1", "v2"])
    single = compute_score(df, P, "v2")
    pd.testing.assert_series_equal(allres["v2"]["total"], single["total"])


def test_band_and_linear_helpers():
    s = pd.Series([-1.0, 0.1, 0.5, np.nan])
    assert band(s, ((0, 10), (0.5, 5), (float("inf"), 0))).tolist()[:3] == [10, 5, 0]
    assert np.isnan(band(s, ((0, 10),)).iloc[3])
    lin = linear(pd.Series([-0.1, 0.0, 0.05, 0.2]), -0.05, 0.05, 0, 10)
    assert lin.tolist() == pytest.approx([0, 5, 10, 10])


def test_adx_uptrend_direction():
    n = 200
    close = pd.Series(np.linspace(100, 200, n))
    df = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close})
    a = ind.adx(df).iloc[-1]
    assert a["plus_di"] > a["minus_di"] and a["adx"] > 40


def test_distribution_days_and_streak():
    df = pd.DataFrame({
        "Close": [100, 99, 98.9, 97, 98, 96.0],
        "Volume": [10, 20, 30, 25, 40, 50.0],
    })
    # 하락 & 거래량 증가: 1번(−1%, 20>10), 5번(−2%, 50>40). 2번은 −0.1%라 제외, 3번은 거래량 감소
    assert ind.distribution_days(df, lookback=5).iloc[-1] == 2
    assert ind.down_streak(df["Close"]).tolist()[1:] == [1, 2, 3, 0, 1]


def test_positions_are_next_day():
    idx = pd.bdate_range("2020-01-01", periods=6)
    out = pd.DataFrame({"event": ["", "BUY", "", "", "SELL", ""]}, index=idx)
    assert positions_from_events(out).tolist() == [0, 0, 1, 1, 1, 0]


def test_strategy_stats_always_in_equals_buy_hold():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 150, 300), index=idx)
    out = pd.DataFrame({"close": close, "total_s": 50.0, "event": [""] * 300}, index=idx)
    out.iloc[0, out.columns.get_loc("event")] = "BUY"
    st = strategy_stats(out, cost=0.0)
    # 첫날 BUY → 다음 날부터 보유: 첫날 수익률 0 이므로 보유 전략과 같다
    assert st["cagr"] == pytest.approx(st["bh_cagr"])


def test_load_series_csv_formats(tmp_path):
    fred = tmp_path / "DGS10.csv"
    fred.write_text("observation_date,DGS10\n2024-01-02,3.95\n2024-01-03,.\n2024-01-04,3.99\n")
    s = load_series_csv(fred)
    assert s.tolist() == [3.95, 3.99]
    cboe = tmp_path / "VIX_History.csv"
    cboe.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2024,13.2,14.2,13.1,13.2\n01/03/2024,13.9,14.5,13.6,14.0\n")
    v = load_series_csv(cboe)
    assert v.iloc[-1] == 14.0 and v.index[-1] == pd.Timestamp("2024-01-03")


def test_fdr_source_with_fake_module(monkeypatch):
    idx = pd.bdate_range("2023-01-02", periods=5)
    frame = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": [1, 2, 3, 4, 5.0],
                          "Volume": 100, "Adj Close": 1.0}, index=idx)
    calls = []
    fake = types.SimpleNamespace(DataReader=lambda sym, start=None, end=None: calls.append(sym) or frame)
    monkeypatch.setitem(sys.modules, "FinanceDataReader", fake)
    df = load_prices("fdr:INVESTING:QQQ")
    assert calls == ["INVESTING:QQQ"]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert load_series("fdr:SPY").tolist() == [1, 2, 3, 4, 5.0]


def test_sample_source_if_available():
    import importlib.util
    if importlib.util.find_spec("arch") is None:
        pytest.skip("arch 미설치")
    df = load_prices("sample:nasdaq")
    assert len(df) > 5000 and df["Volume"].notna().all()
