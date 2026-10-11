import numpy as np
import pandas as pd
import pytest

from scoring import get_profile
from scoring.backfill import checklist_score_bf, scaled
from scoring.checklist_score import checklist_score
from scoring.sources import splice
from tests.test_checklist import smooth

P = get_profile("SPY")


def test_scaled_multiplies_daily_returns():
    df = smooth(0.001, n=50, seed=1)
    df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0]) * 1.001
    out = scaled(df, 1.45)
    r, r2 = df["Close"].pct_change().dropna(), out["Close"].pct_change().dropna()
    assert np.allclose(r2, 1.45 * r)
    assert np.allclose((out["Open"] / out["Close"].shift(1) - 1).dropna(), 1.45 * 0.001)
    assert (out["High"] >= out[["Open", "Close"]].max(axis=1)).all() and out["Volume"].isna().all()


def test_score_without_volume_fills_proxy_years_only():
    # 앞 400일은 거래량 없는 대리 데이터, 뒤 400일은 실제 ETF
    df = smooth(0.001, n=800, seed=2)
    proxy, real = df.iloc[:400].assign(Volume=np.nan), df.iloc[400:]
    joined = splice(real, proxy)
    plain = checklist_score(joined, P)
    bf = checklist_score_bf(joined, P)
    assert plain["score_s"].iloc[:400].isna().all()                       # 그냥 두면 대리 구간에 점수가 없다
    assert bf["score_s"].iloc[250:400].notna().all()
    late = bf.index[-100:]
    assert bf.loc[late, "score_s"].equals(plain.loc[late, "score_s"])      # 실제 거래량이 쌓이면 원래 점수
    assert set(np.unique(bf["state"].dropna())) <= {0.0, 1.0}


def test_full_volume_is_unchanged():
    df = smooth(0.001, n=500, seed=3)
    a, b = checklist_score(df, P), checklist_score_bf(df, P)
    pd.testing.assert_series_equal(a["state"], b["state"])
    assert b["score_s"].iloc[-1] == pytest.approx(a["score_s"].iloc[-1])
