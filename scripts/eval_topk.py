"""점수 상위 k개 동일가중 보유 전략 검증 (watchlist 전 종목, 실제 데이터).

규칙
- 리밸런싱일 종가에 3일 평균 점수가 n 이상인 종목 중 점수 상위 k개를 1/k씩 담는다 (다음 날부터 수익 반영).
- 경계에서 점수가 같은 종목들은 남은 자리를 나눠 갖는다. n 이상이 k개보다 적으면 빈자리는 현금.
- '규칙 보유 중' 변형: 후보를 지금 매수·매도 규칙상 보유 구간인 종목으로 한정한다.
- 리밸런싱 사이에는 비중이 가격 따라 움직이고, 리밸런싱 때 바뀐 비중만큼 편도 0.05% 비용.

비교: 29종목 동일가중 보유, 지금 규칙을 종목별로 적용한 동일가중, SPY·QQQ,
      같은 후보 중 무작위 k개(30번) — 점수 순위가 정말 도움이 되는지.

예)
  python scripts/eval_topk.py --asof 2026-09-29 --md docs/results/topk_eval.md
  python scripts/eval_topk.py --ns 70 75 80 85 90 95 100 --ks 5 8 10 --md docs/results/topk_eval_high_n.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.bigtech import load_watchlist  # noqa: E402
from scoring.checklist_score import checklist_score, strategy_returns  # noqa: E402
from scoring.portfolio import perf_stats  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

BACKFILL = {"PDBC": "PDBC+DBC"}
COST = 0.0005
TECH = ("SPY", "QQQ", "SOXX", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA")
CRISES = {"2008": ("2007-10-01", "2009-03-31"), "2020": ("2020-02-01", "2020-04-30"),
          "2022": ("2022-01-01", "2022-10-31"), "2025 관세": ("2025-02-01", "2025-05-31")}
FREQ = {"매주": 5, "2주": 10, "매월": 21, "월말": 0, "월2회": -1}
# 0 = 매월 마지막 거래일 종가, -1 = 매월 15일(휴장이면 직전 거래일)과 마지막 거래일 종가 (메일의 바구니)
NS, KS, SEEDS = (50, 60, 70, 80), (3, 5, 8, 10), 30


class Book:
    """점수·가격·규칙 보유 상태를 날짜 × 종목 배열로 들고 백테스트한다."""

    def __init__(self, S: pd.DataFrame, C: pd.DataFrame, H: pd.DataFrame):
        self.dates, self.cols = S.index, S.columns
        self.s = S.to_numpy()
        self.r = C.pct_change().fillna(0.0).to_numpy()
        self.h = H.fillna(0.0).to_numpy() == 1
        self.month_end = np.r_[self.dates[1:].month != self.dates[:-1].month, True]
        nxt = np.r_[self.dates[1:].day, 99]
        self.semi = self.month_end | ((self.dates.day <= 15) & ((nxt > 15) | self.month_end))

    def target(self, d: int, n: float, k: int, rule: bool, rng=None) -> np.ndarray:
        s = self.s[d]
        elig = ~np.isnan(s) & (s >= n)
        if rule:
            elig &= self.h[d]
        w = np.zeros(len(s))
        m = elig.sum()
        if m == 0:
            return w
        if m <= k:
            w[elig] = 1.0 / k
            return w
        if rng is not None:
            w[rng.choice(np.flatnonzero(elig), k, replace=False)] = 1.0 / k
            return w
        sc = np.where(elig, s, -np.inf)
        v = np.sort(sc)[-k]
        above, tie = sc > v, sc == v
        w[above] = 1.0 / k
        w[tie] = (k - above.sum()) / (tie.sum() * k)
        return w

    def run(self, n: float, k: int, every: int, rule: bool = False, rng=None, mid_exit: bool = False,
            rf: np.ndarray | None = None) -> tuple[pd.Series, float, dict]:
        """mid_exit: 교체 사이에 규칙 매도(보유 상태 0)가 나면 그날 종가에 팔아 현금 · rf: 현금 일간 수익률 (날짜 순)."""
        w = np.zeros(len(self.cols))
        out = np.zeros(len(self.dates))
        turn, cnt, cash, tech = 0.0, [], [], []
        ti = [i for i, c in enumerate(self.cols) if c in TECH]
        for d in range(len(self.dates)):
            out[d] = (w * self.r[d]).sum() + ((1 - w.sum()) * rf[d] if rf is not None else 0.0)
            grown = w * (1 + self.r[d])
            w = grown / (1 + out[d]) if 1 + out[d] > 0 else grown
            reb = self.month_end[d] if every == 0 else self.semi[d] if every == -1 else d % every == 0
            if mid_exit and not reb:
                sell = (w > 0) & ~self.h[d]
                if sell.any():
                    turn += w[sell].sum()
                    out[d] -= w[sell].sum() * COST
                    w[sell] = 0.0
            if reb:
                nw = self.target(d, n, k, rule, rng)
                tc = np.abs(nw - w).sum()
                turn += tc
                out[d] -= tc * COST
                w = nw
                if not np.isnan(self.s[d]).all():
                    cnt.append((nw > 0).sum())
                    cash.append(1 - nw.sum())
                    tech.append(nw[ti].sum())
        info = {"종목 수": np.mean(cnt), "현금": np.mean(cash), "기술주": np.mean(tech)}
        return pd.Series(out, index=self.dates), turn / (len(self.dates) / 252), info


def mdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--asof", default="2026-09-29")
    ap.add_argument("--md", help="결과 마크다운 저장")
    ap.add_argument("--ns", type=float, nargs="+", default=list(NS), help="점수 기준 n 목록")
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS), help="보유 종목 수 k 목록")
    ap.add_argument("--freqs", nargs="+", default=["매주", "매월"], choices=list(FREQ), help="리밸런싱 주기")
    args = ap.parse_args()

    end = pd.Timestamp(args.asof)
    A = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    periods = [("2006~2017", pd.Timestamp("2006-01-01"), pd.Timestamp("2017-12-31")),
               (f"2018~{(A - pd.Timedelta(days=1)).date()}", pd.Timestamp("2018-01-01"), A - pd.Timedelta(days=1)),
               ("최근 1년", A, end)]
    wl = load_watchlist(args.watchlist)
    cache: dict[str, pd.DataFrame] = {}

    def prices(t: str) -> pd.DataFrame:
        if t not in cache:
            cache[t] = load_prices(f"fdr:{BACKFILL.get(t, t)}", args.start).loc[:end]
        return cache[t]

    S, C, H, RR = {}, {}, {}, {}
    for row in wl.itertuples(index=False):
        df = prices(row.ticker)
        bench = prices(row.bench)["Close"] if row.bench else None
        o = checklist_score(df, get_profile(row.ticker), bench)
        S[row.ticker], C[row.ticker], H[row.ticker] = o["score_s"], o["close"], o["state"]
        RR[row.ticker] = strategy_returns(o, COST)
        print(f"{row.ticker} 완료", file=sys.stderr)
    S = pd.DataFrame(S).loc["2005-06-01":]
    C, H, RR = pd.DataFrame(C).reindex(S.index), pd.DataFrame(H).reindex(S.index), pd.DataFrame(RR).reindex(S.index)
    book = Book(S, C, H)

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    def pct(v) -> str:
        return "-" if pd.isna(v) else f"{v * 100:.1f}%"

    def triple(r: pd.Series, a, b) -> str:
        ps = perf_stats(r.loc[a:b])
        return f"{pct(ps['cagr'])} · {pct(ps['mdd'])} · {ps['sharpe']:.2f}"

    def row_of(name: str, r: pd.Series, extra: dict | None = None) -> dict:
        rec = {"전략": name, **(extra or {})}
        for pn, a, b in periods:
            rec[pn] = triple(r, a, b)
        rec["2008 · 2020 · 2022 · 2025 낙폭"] = " · ".join(pct(mdd(r.loc[a:b])) for a, b in CRISES.values())
        return rec

    out(f"# 점수 상위 k개 동일가중 보유 전략 검증 (~{end.date()})\n")
    out("표기: 연수익 · 최대낙폭 · 샤프\n")
    avail = ~np.isnan(S.to_numpy())
    ew = pd.Series(np.where(avail, book.r, 0.0).sum(axis=1) / np.maximum(avail.sum(axis=1), 1), index=S.index)
    base = [row_of("29종목 동일가중 보유 (매일 리밸런싱)", ew),
            row_of("지금 규칙을 종목별 적용 (종목당 1/29, 나머지 현금)", RR.mean(axis=1, skipna=True)),
            row_of("SPY", C["SPY"].pct_change().fillna(0.0)), row_of("QQQ", C["QQQ"].pct_change().fillna(0.0))]
    out("## 1. 기준선\n")
    out(pd.DataFrame(base).to_markdown(index=False) + "\n")

    for rule, title in ((False, "점수 n 이상 중 상위 k"), (True, "규칙상 보유 종목 중 점수 n 이상 상위 k")):
        out(f"## {'3' if rule else '2'}. {title}\n")
        out("무작위 = 같은 후보 중 무작위 k개를 30번 뽑았을 때, 점수순이 그보다 샤프가 높았던 비율 (괄호: 무작위 샤프 중앙값)\n")
        rows = []
        for fn in args.freqs:
            every = FREQ[fn]
            for n in args.ns:
                for k in args.ks:
                    r, turn, info = book.run(n, k, every, rule)
                    rnd = [book.run(n, k, every, rule, np.random.default_rng(sd))[0] for sd in range(SEEDS)]
                    rec = row_of(f"{fn} n={n:g} k={k}", r, {"종목 수": f"{info['종목 수']:.1f}",
                                                         "현금": f"{info['현금']:.0%}", "기술주": f"{info['기술주']:.0%}",
                                                         "회전율/년": f"{turn:.0f}"})
                    for pn, a, b in periods:
                        rs = np.array([perf_stats(x.loc[a:b])["sharpe"] for x in rnd])
                        mine = perf_stats(r.loc[a:b])["sharpe"]
                        rec[f"무작위 {pn}"] = f"{(mine > rs).mean():.0%} ({np.median(rs):.2f})"
                    rows.append(rec)
        out(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
