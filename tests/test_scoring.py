import numpy as np
import pandas as pd
import pytest

from scoring import compute_score, get_profile, load_investing_csv
from scoring.indicators import volume_ratio
from scoring.score import ITEMS

EN_CSV = '''"Date","Price","Open","High","Low","Vol.","Change %"
"09/26/2025","1,595.97","592.10","597.00","590.50","45.67M","0.44%"
"09/25/2025","594.00","590.00","596.00","588.00","850.5K","-0.31%"
"09/24/2025","595.85","595.00","598.00","593.00","-","0.10%"
'''

KR_CSV = '''"날짜","종가","시가","고가","저가","거래량","변동 %"
"2025- 09- 26","595.97","592.10","597.00","590.50","1.2B","0.44%"
"2025- 09- 25","594.00","590.00","596.00","588.00","45.67M","-0.31%"
'''


def synthetic(n=600, drift=0.001, seed=0, volume=True):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": rng.uniform(1e6, 2e6, n) if volume else np.nan,
    }, index=idx)
    return df


def test_load_english_csv(tmp_path):
    p = tmp_path / "qqq.csv"
    p.write_text(EN_CSV, encoding="utf-8")
    df = load_investing_csv(p)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2025-09-24", "2025-09-25", "2025-09-26"]
    assert df["Close"].iloc[-1] == pytest.approx(1595.97)
    assert df["Volume"].iloc[-1] == pytest.approx(45.67e6)
    assert df["Volume"].iloc[1] == pytest.approx(850.5e3)
    assert np.isnan(df["Volume"].iloc[0])


def test_load_korean_csv(tmp_path):
    p = tmp_path / "qqq_kr.csv"
    p.write_text("﻿" + KR_CSV, encoding="utf-8")
    df = load_investing_csv(p)
    assert df.index[-1] == pd.Timestamp("2025-09-26")
    assert df["Volume"].iloc[-1] == pytest.approx(1.2e9)


def test_volume_ratio_hand_calc():
    df = pd.DataFrame({
        "Close": [10, 11, 10, 10, 12],
        "Volume": [0, 100, 50, 20, 30],
    }, dtype=float)
    vr = volume_ratio(df, period=4)
    # up=100+30, down=50, flat=20 → (130+10)/(50+10)*100
    assert vr.iloc[-1] == pytest.approx(140 / 60 * 100)
    assert vr.iloc[:4].isna().all()


def test_item_max_sum_is_100():
    assert sum(m for _, _, m in ITEMS) == 100


def test_score_range_and_trend_direction():
    p = get_profile("QQQ")
    up = compute_score(synthetic(drift=0.003), p)["total"].dropna()
    down = compute_score(synthetic(drift=-0.003), p)["total"].dropna()
    assert len(up) > 0 and len(down) > 0
    assert up.between(0, 100).all() and down.between(0, 100).all()
    assert up.tail(100).mean() > 70
    assert down.tail(100).mean() < 30


def test_items_never_exceed_max():
    out = compute_score(synthetic(seed=3, drift=0.0), get_profile("QQQ"))
    for _, name, m in ITEMS:
        col = out[name].dropna()
        assert (col >= 0).all() and (col <= m).all(), name


def test_missing_volume_is_rescaled():
    out = compute_score(synthetic(volume=False, drift=0.003), get_profile("QQQ"))
    valid = out.dropna(subset=["total"])
    assert not valid["volume_used"].any()
    row = valid.iloc[-1]
    expected = (row["ichimoku"] + row["ma"] + row["momentum"]) * 100 / 80
    assert row["total"] == pytest.approx(round(expected, 1))


def test_warmup_rows_have_no_score():
    out = compute_score(synthetic(n=300), get_profile("QQQ"))
    # 200일 이평 + 20일 기울기 이전에는 점수가 없어야 한다
    assert out["total"].iloc[:219].isna().all()
    assert out["total"].iloc[-1] == out["total"].iloc[-1]  # not NaN


def test_events_are_threshold_crossings():
    p = get_profile("QQQ")
    out = compute_score(synthetic(seed=7, drift=0.0005, n=1500), p)
    prev = out["total"].shift(1)
    buys = out[out["event"] == "BUY"]
    sells = out[out["event"] == "SELL"]
    assert (buys["total"] >= p.buy).all() and (prev[buys.index] < p.buy).all()
    assert (sells["total"] <= p.sell).all() and (prev[sells.index] > p.sell).all()
    seq = out.loc[out["event"] != "", "event"].tolist()
    assert len(seq) > 2
    assert all(a != b for a, b in zip(seq, seq[1:]))
