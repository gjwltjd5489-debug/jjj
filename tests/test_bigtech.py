from datetime import date

import pandas as pd

from scoring.bigtech import eligible, listed, load_watchlist, members_on, membership_year, rank_dollar_volume

M = {2025: ["AAPL", "MSFT"], 2026: ["NVDA", "AAPL"]}


def test_membership_switches_on_last_trading_day():
    assert membership_year(date(2025, 12, 30)) == 2025
    assert membership_year(date(2025, 12, 31)) == 2026          # 마지막 거래일 종가 교체부터 새 목록
    assert members_on(date(2025, 12, 31), M) == ["NVDA", "AAPL"]
    assert members_on(date(2027, 3, 2), M) == ["NVDA", "AAPL"]  # 목록이 없으면 가장 최근 목록
    assert not eligible("MSFT", date(2026, 3, 2), M) and eligible("MSFT", date(2025, 6, 2), M)
    assert eligible("SPY", date(2026, 3, 2), M)                 # ETF 는 늘 후보


def test_rank_dollar_volume_respects_listing_years():
    idx = pd.bdate_range("2024-01-01", "2025-12-31")
    px = {t: pd.DataFrame({"Close": 10.0, "Volume": v}, index=idx) for t, v in (("A", 3e6), ("B", 2e6), ("C", 1e6))}
    pool = pd.DataFrame({"ticker": ["A", "B", "C"], "name": "", "from": ["", "", ""], "to": ["2024", "", ""]})
    r = rank_dollar_volume(px, 2026, pool)                      # A 는 2024년까지만 나스닥 → 2025년 말 순위에서 빠짐
    assert list(r.index) == ["B", "C"] and abs(r["B"] - 20.0) < 1e-9
    assert listed({"from": "2017", "to": ""}, 2017) and not listed({"from": "2017", "to": ""}, 2016)


def test_load_watchlist_appends_members(tmp_path):
    f = tmp_path / "wl.csv"
    f.write_text("group,ticker,name,bench\n지수,SPY,S&P 500,\n빅테크,OLD,옛 종목,SPY\n", encoding="utf-8")
    wl = load_watchlist(f, date(2026, 5, 1), M)
    assert list(wl["ticker"]) == ["SPY", "NVDA", "AAPL"] and set(wl[wl.group == "빅테크"]["bench"]) == {"SPY"}
