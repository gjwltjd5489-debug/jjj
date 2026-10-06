"""바구니 운용 검증 (실제 데이터): 중복 종목 제거 · SPY 200일선 필터 · QLD 하락 매수 (docs/topk.md 8장).

- ① 이전 메일: 나스닥 거래대금 상위 10 · 교체 사이 매도 · 현금 이자 (7장)
- ② ① + IWM·VTV·SCHD 제외 + SPY 200일선 아래면 주식 종목 절반 (교체일, 그리고 교체 사이 처음 내려간 날)
- ③ ② + QLD 하락 매수 (지금 메일, scoring/topk.py)
교체일 운을 덜기 위해 교체일을 0~4거래일 밀어 5번 돌린 평균을 쓴다 (괄호: 5번 중 최저~최고).
샤프 = (일간 수익 − 단기국채 금리) 평균 ÷ 일간 수익 표준편차 × √252.
QLD: 실제 가격 (2006-06 상장 전은 2 × QQQ − 연 0.95% − 단기금리로 합성).

예)
  python scripts/eval_overlay.py --asof 2026-10-02 --md docs/results/overlay_eval.md
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
from eval_bigtech import tbill  # noqa: E402
from scoring import get_profile  # noqa: E402
from scoring.bigtech import BENCH, GROUP, load_members  # noqa: E402
from scoring.checklist_score import checklist_score  # noqa: E402
from scoring.sources import load_prices  # noqa: E402
from scoring.topk import BASKET_EXCLUDE, BASKET_K, BASKET_N, DIP_TICKER, EQUITY_GROUPS  # noqa: E402

CRISES = {"2008": ("2007-10-01", "2009-03-31"), "2020": ("2020-02-01", "2020-04-30"),
          "2022": ("2022-01-01", "2022-10-31"), "2025": ("2025-02-01", "2025-05-31")}
OFFSETS = range(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--asof", default="2026-10-02")
    ap.add_argument("--md", help="결과 마크다운 저장")
    args = ap.parse_args()
    end = pd.Timestamp(args.asof)
    A = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    periods = [("2006~2017", "2006-01-01", "2017-12-31"),
               (f"2018~{(A - pd.Timedelta(days=1)).date()}", "2018-01-01", A - pd.Timedelta(days=1)),
               ("최근 1년", A, end)]
    wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    etf = wl[wl["group"] != GROUP]
    members = load_members()
    stocks = sorted({t for v in members.values() for t in v})
    cache: dict[str, pd.DataFrame] = {}

    def prices(t: str) -> pd.DataFrame:
        if t not in cache:
            cache[t] = load_prices(f"fdr:{et.BACKFILL.get(t, t)}", "2004-01-01").loc[:end]
        return cache[t]
    S, C, H = {}, {}, {}
    for t, b in [(r.ticker, r.bench) for r in etf.itertuples(index=False)] + [(t, BENCH) for t in stocks]:
        o = checklist_score(prices(t), get_profile(t), prices(b)["Close"] if b else None)
        S[t], C[t], H[t] = o["score_s"], o["close"], o["state"]
        print(f"{t} 완료", file=sys.stderr)
    S = pd.DataFrame(S).loc["2005-06-01":]
    idx = S.index
    C, H = pd.DataFrame(C).reindex(idx), pd.DataFrame(H).reindex(idx)
    rf = tbill("2005-01-01", idx)
    rfs = pd.Series(rf, index=idx)
    last = {y: idx[idx.year == y][-1] for y in set(idx.year)}
    m = pd.DataFrame(True, index=idx, columns=S.columns)
    m[stocks] = False
    for y, v in members.items():
        a = last.get(y - 1, idx[0])
        b = last.get(y, idx[-1] + pd.Timedelta(days=1))
        m.loc[(idx >= a) & (idx < b), [t for t in v if t in stocks]] = True
    group = {**dict(zip(etf["ticker"], etf["group"])), **{t: GROUP for t in stocks}}
    cols = list(S.columns)
    exclude = np.array([c in BASKET_EXCLUDE for c in cols])
    equity = np.array([group[c] in EQUITY_GROUPS for c in cols])
    spy = C["SPY"]
    dist = (1 - spy / spy.rolling(200).mean()).to_numpy()
    q = load_prices(f"fdr:{DIP_TICKER}", "2004-01-01")["Close"].reindex(idx)
    syn = 2 * C["QQQ"].pct_change().fillna(0.0) - (0.0095 / 252 + rfs)
    rq = q.pct_change().where(q.pct_change().notna(), syn).fillna(0.0).to_numpy()
    book = et.Book(S.where(m), C, H.where(m, 0.0))

    def sharpe(r: pd.Series, a, b) -> float:
        x = r.loc[a:b]
        return float((x - rfs.loc[a:b]).mean() / x.std() * np.sqrt(252))

    def cagr(r: pd.Series, a, b) -> float:
        x = r.loc[a:b]
        return float((1 + x).prod() ** (252 / len(x)) - 1)

    def mdd(r: pd.Series) -> float:
        eq = (1 + r).cumprod()
        return float((eq / eq.cummax() - 1).min())

    variants = [("① 이전 메일 (7장)", {}),
                ("② ① + 중복 제거 + SPY 200일선 필터", dict(exclude=exclude, equity=equity, dist=dist)),
                ("**③ ② + QLD 하락 매수 (지금 메일)**", dict(exclude=exclude, equity=equity, dist=dist, rq=rq))]
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)
    out(f"# 바구니 운용 검증: 중복 제거 · SPY 200일선 필터 · QLD 하락 매수 (~{end.date()})\n")
    out(f"바구니 = 규칙상 보유 · {BASKET_N:g}점 이상 상위 {BASKET_K}개 · 15일·월말 교체 · 교체 사이 규칙 매도 · 현금 이자. "
        "교체일을 0~4거래일 민 5번의 평균 (괄호: 최저~최고).\n")
    rows, eps = [], None
    for name, kw in variants:
        rs = []
        for o in OFFSETS:
            r = book.run(BASKET_N, BASKET_K, -1, True, mid_exit=True, rf=rf, offset=o, **kw)[0]
            rs.append(r)
            if o == 0 and "rq" in kw:
                eps = list(book.events)
        rec = {"운용": name}
        for pn, a, b in periods:
            s = [sharpe(r, a, b) for r in rs]
            rec[f"{pn} 샤프"] = f"{np.mean(s):.2f} ({min(s):.2f}~{max(s):.2f})"
            rec[f"{pn} 연수익"] = f"{np.mean([cagr(r, a, b) for r in rs]):.1%}"
        rec["2006~ 연수익 · 샤프 · 최대낙폭"] = (f"{np.mean([cagr(r, '2006-01-01', end) for r in rs]):.1%} · "
                                         f"{np.mean([sharpe(r, '2006-01-01', end) for r in rs]):.2f} · "
                                         f"{np.mean([mdd(r.loc['2006':]) for r in rs]):.0%}")
        rec["위기 낙폭 " + " · ".join(CRISES)] = " · ".join(f"{np.mean([mdd(r.loc[a:b]) for r in rs]):.0%}"
                                                       for a, b in CRISES.values())
        rows.append(rec)
    out(pd.DataFrame(rows).to_markdown(index=False) + "\n")
    out(f"## {DIP_TICKER} 하락 매수 기록 (교체일 0, 첫 매수 → 매도)\n")
    ep, cur = [], None
    for d, kind, amt in eps or []:
        if kind == "dip_buy":
            cur = cur or {"from": idx[d].date(), "buys": 0.0}
            cur["buys"] += amt
        elif cur:
            ret = float(np.prod(1 + rq[idx.get_loc(pd.Timestamp(cur["from"])) + 1:d + 1]) - 1)
            ep.append({"첫 매수": cur["from"], "매도": idx[d].date(), "산 비율 합": f"{cur['buys']:.0%}",
                       "첫 매수 뒤 QLD": f"{ret:+.0%}", "이유": "손절 −25%" if kind == "dip_stop" else "200일선 회복"})
            cur = None
    if cur:
        ep.append({"첫 매수": cur["from"], "매도": "보유 중", "산 비율 합": f"{cur['buys']:.0%}", "첫 매수 뒤 QLD": "",
                   "이유": ""})
    out(pd.DataFrame(ep).to_markdown(index=False) + "\n")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
