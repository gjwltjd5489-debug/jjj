"""빅테크 칸 · 바구니 운용 방식 검증 (실제 데이터): 고정 M7 vs 나스닥 거래대금 상위 10, 교체 사이 매도·현금 이자.

- 고정 M7: 2026년 이전 방식. 오늘의 승자 7개를 2006년부터 알고 있었다는 가정이라 과거 성과가 부풀려진다.
- 나스닥 거래대금 상위 10: config/bigtech.csv (scripts/update_bigtech.py 가 만든 목록, 매년 마지막 거래일에 교체)
- D: 교체 사이에 규칙 매도 신호가 나면 그날 종가에 팔아 현금 · E: 현금에 단기국채 금리(^IRX)
- 나머지 22개 ETF 는 config/watchlist.csv 그대로. 바구니: 규칙상 보유 · n=60 · 상위 5개 · 월 2회 교체.
한계: 상장폐지된 옛 나스닥 대형주(야후·옛 델 등)는 가격이 없어 후보에서 빠졌다 → 과거 성과가 조금 낙관적.

예)
  python scripts/eval_bigtech.py --asof 2026-10-02 --md docs/results/bigtech_eval.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_topk as et  # noqa: E402
from scoring import get_profile  # noqa: E402
from scoring.bigtech import GROUP, load_members  # noqa: E402
from scoring.checklist_score import checklist_score  # noqa: E402
from scoring.portfolio import perf_stats  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

M7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
CRISES = {"2008": ("2007-10-01", "2009-03-31"), "2020": ("2020-02-01", "2020-04-30"),
          "2022": ("2022-01-01", "2022-10-31"), "2025 관세": ("2025-02-01", "2025-05-31")}
SEEDS = 20


def tbill(start: str, idx: pd.DatetimeIndex) -> np.ndarray:
    import yfinance as yf
    x = yf.download("^IRX", start=start, progress=False, auto_adjust=False)["Close"].squeeze()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    return (x.reindex(idx).ffill().bfill() / 100 / 252).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--asof", default="2026-10-02")
    ap.add_argument("--md", help="결과 마크다운 저장")
    args = ap.parse_args()
    end = pd.Timestamp(args.asof)
    A = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    periods = [("2006~2017", "2006-01-01", "2017-12-31"), (f"2018~{(A - pd.Timedelta(days=1)).date()}", "2018-01-01",
                                                           A - pd.Timedelta(days=1)), ("최근 1년", A, end)]
    wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    etf = wl[wl["group"] != GROUP]
    members = load_members()
    stocks = sorted(set(M7) | {t for v in members.values() for t in v})
    cache: dict[str, pd.DataFrame] = {}

    def prices(t: str) -> pd.DataFrame:
        if t not in cache:
            cache[t] = load_prices(f"fdr:{et.BACKFILL.get(t, t)}", "2004-01-01").loc[:end]
        return cache[t]
    S, C, H = {}, {}, {}
    for t, b in [(r.ticker, r.bench) for r in etf.itertuples(index=False)] + [(t, "SPY") for t in stocks]:
        o = checklist_score(prices(t), get_profile(t), prices(b)["Close"] if b else None)
        S[t], C[t], H[t] = o["score_s"], o["close"], o["state"]
        print(f"{t} 완료", file=sys.stderr)
    S = pd.DataFrame(S).loc["2005-06-01":]
    idx = S.index
    C, H = pd.DataFrame(C).reindex(idx), pd.DataFrame(H).reindex(idx)
    rf = tbill("2005-01-01", idx)
    last = {y: idx[idx.year == y][-1] for y in set(idx.year)}

    def mask(by_year: dict[int, list[str]]) -> pd.DataFrame:
        """적용 연도 Y 목록은 Y−1년 마지막 거래일 종가(교체)부터 Y년 마지막 거래일 전날까지."""
        m = pd.DataFrame(True, index=idx, columns=S.columns)
        m[stocks] = False
        for y, v in by_year.items():
            a = last.get(y - 1, idx[0])
            b = last.get(y, idx[-1] + pd.Timedelta(days=1))
            m.loc[(idx >= a) & (idx < b), [t for t in v if t in stocks]] = True
        return m
    fixed = mask({y: M7 for y in range(2005, idx[-1].year + 2)})
    dyn = mask(members)

    def run(m: pd.DataFrame, de: bool, rng=None) -> pd.Series:
        book = et.Book(S.where(m), C, H.where(m, 0.0))
        return book.run(60, 5, -1, True, rng, mid_exit=de, rf=rf if de else None)[0]

    def mdd(r: pd.Series) -> float:
        eq = (1 + r).cumprod()
        return float((eq / eq.cummax() - 1).min())

    def triple(r: pd.Series, a, b) -> str:
        ps = perf_stats(r.loc[a:b])
        return f"{ps['cagr']:.1%} · {ps['mdd']:.1%} · {ps['sharpe']:.2f}"

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)
    out(f"# 빅테크 칸 · 교체 사이 매도·현금 이자 검증 (~{end.date()})\n")
    out("표기: 연수익 · 최대낙폭 · 샤프. 바구니 = 규칙상 보유 · 3일 평균 60점 이상 상위 5개 · 15일·월말 교체\n")
    rows, series = [], {}
    for name, m, de in (("고정 M7 · 지금까지 운용", fixed, False), ("고정 M7 · D+E", fixed, True),
                        ("나스닥 거래대금 상위 10 · 지금까지 운용", dyn, False),
                        ("**나스닥 거래대금 상위 10 · D+E (지금 메일)**", dyn, True)):
        r = run(m, de)
        series[name] = r
        rec = {"빅테크 칸 · 운용": name, **{pn: triple(r, a, b) for pn, a, b in periods},
               "2006~ 최대낙폭": f"{mdd(r.loc['2006':]):.1%}",
               "위기 낙폭 " + " · ".join(CRISES): " · ".join(f"{mdd(r.loc[a:b]):.0%}" for a, b in CRISES.values())}
        if de:
            rnd = [run(m, True, np.random.default_rng(sd)) for sd in range(SEEDS)]
            rec["무작위보다 샤프 높은 비율"] = " / ".join(
                f"{(perf_stats(r.loc[a:b])['sharpe'] > np.array([perf_stats(x.loc[a:b])['sharpe'] for x in rnd])).mean():.0%}"
                for _, a, b in periods[:2])
        rows.append(rec)
    out(pd.DataFrame(rows).fillna("").to_markdown(index=False) + "\n")
    out("무작위 = 같은 후보 중 무작위 5개를 20번 뽑았을 때 점수순이 샤프가 높았던 비율 (2006~2017 / 2018~)\n")
    yr = pd.DataFrame({k.strip("*"): ((1 + v).groupby(v.index.year).prod() - 1) * 100 for k, v in series.items()})
    yr["SPY"] = ((1 + C["SPY"].pct_change().fillna(0)).groupby(idx.year).prod() - 1) * 100
    out("## 연도별 수익률 (%)\n")
    out(yr.loc[2006:].round(1).to_markdown() + "\n")
    out("## 연도별 빅테크 칸 (나스닥 거래대금 상위 10)\n")
    out("\n".join(f"- {y}: {', '.join(v)}" for y, v in sorted(members.items())) + "\n")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
