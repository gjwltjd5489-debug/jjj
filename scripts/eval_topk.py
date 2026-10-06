"""점수 상위 k개 동일가중 보유 전략 검증 (watchlist 전 종목, 실제 데이터).

규칙
- 리밸런싱일 종가에 3일 평균 점수가 n 이상인 종목 중 점수 상위 k개를 1/k씩 담는다 (다음 날부터 수익 반영).
- 경계에서 점수가 같으면 최근 60거래일 수익률이 높은 종목이 들어간다. n 이상이 k개보다 적으면 빈자리는 현금.
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
from scoring.topk import DIP_STOP, MOM_DAYS, RISK_OFF_CUT, dip_target  # noqa: E402
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
        self.mom = (C / C.shift(MOM_DAYS) - 1).to_numpy()
        self.h = H.fillna(0.0).to_numpy() == 1
        self.month_end = np.r_[self.dates[1:].month != self.dates[:-1].month, True]
        nxt = np.r_[self.dates[1:].day, 99]
        self.semi = self.month_end | ((self.dates.day <= 15) & ((nxt > 15) | self.month_end))

    def target(self, d: int, n: float, k: int, rule: bool, rng=None, exclude=None, ranked: bool = False):
        """교체일 목표 비중. exclude: 후보에서 뺄 열(bool 배열). ranked=True 면 (비중, 순위 순 열 번호)."""
        s = self.s[d]
        elig = ~np.isnan(s) & (s >= n)
        if rule:
            elig &= self.h[d]
        if exclude is not None:
            elig &= ~exclude
        w = np.zeros(len(s))
        if rng is not None and elig.sum() > k:
            pick = list(rng.choice(np.flatnonzero(elig), k, replace=False))
        else:
            # 점수 → 최근 MOM_DAYS 거래일 수익률 → 열 이름 순 (메일의 basket_weights 와 같음)
            mom = np.where(np.isnan(self.mom[d]), -np.inf, self.mom[d])
            pick = sorted(np.flatnonzero(elig), key=lambda i: (-s[i], -mom[i], self.cols[i]))[:k]
        w[pick] = 1.0 / k
        return (w, pick) if ranked else w

    def run(self, n: float, k: int, every: int, rule: bool = False, rng=None, mid_exit: bool = False,
            rf: np.ndarray | None = None, exclude: np.ndarray | None = None, equity: np.ndarray | None = None,
            dist: np.ndarray | None = None, rq: np.ndarray | None = None,
            offset: int = 0) -> tuple[pd.Series, float, dict]:
        """mid_exit: 교체 사이에 규칙 매도(보유 상태 0)가 나면 그날 종가에 팔아 현금 · rf: 현금 일간 수익률 (날짜 순).

        메일의 바구니 운용 (scoring/topk.py, scoring/checklist.py _simulate 와 같은 계산):
        exclude: 후보에서 뺄 열 · dist: SPY 가 200일선보다 아래인 비율 (+면 아래, NaN = 200일선 없음) →
        equity 열은 아래일 때 비중 절반 (교체일, 그리고 교체 사이 처음 내려간 날 한 번) ·
        rq: QLD 일간 수익률 → 하락 매수 (dist 5%마다 5%, 최대 50%, −25% 손절, 200일선 회복 시 청산).
        offset: 교체일을 이만큼 거래일 뒤로 민다 (교체일 운 확인용, 월2회만)."""
        N = len(self.cols)
        w = np.zeros(N)
        out = np.zeros(len(self.dates))
        turn, cnt, cash, tech = 0.0, [], [], []
        ti = [i for i, c in enumerate(self.cols) if c in TECH]
        semi = self.semi
        if offset:
            semi = np.zeros(len(self.dates), bool)
            semi[np.minimum(np.flatnonzero(self.semi) + offset, len(self.dates) - 1)] = True
        eq = equity if equity is not None else np.zeros(N, bool)
        q = cq = qb = 0.0
        blocked = cut_done = False
        order: list[int] = []
        self.events = []

        def trim(wv: np.ndarray, need: float) -> float:
            got = 0.0
            for i in reversed(order):
                if got >= need - 1e-12:
                    break
                x = min(wv[i], need - got)
                wv[i] -= x
                got += x
            return got
        for d in range(len(self.dates)):
            x = dist[d] if dist is not None else np.nan
            ro = x > 0
            ro_prev = d > 0 and dist is not None and dist[d - 1] > 0
            r_q = rq[d] if rq is not None else 0.0
            out[d] = (w * self.r[d]).sum() + q * r_q + ((1 - w.sum() - q) * rf[d] if rf is not None else 0.0)
            grown = w * (1 + self.r[d])
            w = grown / (1 + out[d]) if 1 + out[d] > 0 else grown
            q, qb = q * (1 + r_q) / (1 + out[d]), qb / (1 + out[d])
            reb = self.month_end[d] if every == 0 else semi[d] if every == -1 else d % every == 0
            if not reb and dist is not None and ro and not ro_prev and not cut_done:
                c = w * eq * (1 - RISK_OFF_CUT)
                turn += c.sum()
                out[d] -= c.sum() * COST
                w = w - c
                cut_done = True
            if mid_exit and not reb:
                sell = (w > 0) & ~self.h[d]
                if sell.any():
                    turn += w[sell].sum()
                    out[d] -= w[sell].sum() * COST
                    w[sell] = 0.0
            if reb:
                cut_done = False
                nw, order = self.target(d, n, k, rule, rng, exclude, ranked=True)
                if ro:
                    nw = np.where(eq, nw * RISK_OFF_CUT, nw)
                over = nw.sum() + q - 1
                if over > 1e-12:
                    trim(nw, over)
                tc = np.abs(nw - w).sum()
                turn += tc
                out[d] -= tc * COST
                w = nw
                if not np.isnan(self.s[d]).all():
                    cnt.append((nw > 0).sum())
                    cash.append(1 - nw.sum())
                    tech.append(nw[ti].sum())
            if rq is None or np.isnan(x):
                continue
            if blocked and x <= 0:
                blocked = False
            if cq > 0:
                stop = q < (1 + DIP_STOP) * qb
                if stop or x <= 0:
                    turn += q
                    out[d] -= q * COST
                    self.events.append((d, "dip_stop" if stop else "dip_exit", q))
                    q = cq = qb = 0.0
                    blocked = bool(stop and x > 0)
                    if not reb:   # 판 돈으로 지금 바구니 종목을 목표 비중까지
                        for i in order:
                            if w[i] <= 0:
                                continue
                            add = min(max(0.0, (RISK_OFF_CUT if ro and eq[i] else 1.0) / k - w[i]), 1 - w.sum())
                            if add > 0:
                                w[i] += add
                                turn += add
                                out[d] -= add * COST
                    continue
            if x > 0 and not blocked:
                tgt = dip_target(x)
                if tgt > cq + 1e-12:
                    amt = tgt - cq
                    room = 1 - w.sum() - q
                    if amt > room + 1e-12:
                        got = trim(w, amt - room)
                        turn += got
                        out[d] -= got * COST
                    turn += amt
                    out[d] -= amt * COST
                    q, qb, cq = q + amt, qb + amt, tgt
                    self.events.append((d, "dip_buy", amt))
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
