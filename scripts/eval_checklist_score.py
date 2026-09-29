"""체크리스트 점수 규칙 검증: watchlist 전 종목, 실제 데이터.

비교: 보유(buy & hold) / 이전 규칙(v6 조합 카드) / 새 체크리스트 점수 규칙
- 종목별 CAGR·MDD·샤프·노출·연간 매매 횟수, 기간 분할(앞/뒤)
- 점수 구간별 이후 20일 수익률·변동성, 날짜별 종목 간 순위상관(점수 높은 종목이 다음 20일에 더 올랐나)
- 임계값·평활·위험 차단 민감도

예)
  pip install finance-datareader
  python scripts/eval_checklist_score.py --start 2005-01-01 --split 2018-01-01 --md docs/results/checklist_score_eval.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.checklist_score import RULE, checklist_score, strategy_returns  # noqa: E402
from scoring.portfolio import perf_stats  # noqa: E402
from scoring.score import compute_all  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

BACKFILL = {"PDBC": "PDBC+DBC"}
COST = 0.0005


def metrics(ret: pd.Series, state: pd.Series | None) -> dict:
    st = perf_stats(ret)
    years = len(ret) / 252
    if state is not None:
        pos = state.fillna(0)
        st["exposure"] = pos.mean()
        st["trades_per_year"] = (pos.diff() > 0).sum() / years if years else np.nan
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--md", help="결과 마크다운 저장")
    args = ap.parse_args()

    wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    cache: dict[str, pd.DataFrame] = {}

    def prices(t):
        if t not in cache:
            cache[t] = load_prices(f"fdr:{BACKFILL.get(t, t)}", args.start)
        return cache[t]

    variants = {
        "기본 (70/40, 3일 평균, 과열·고변동 차단)": RULE,
        "위험 차단 없음": replace(RULE, block_overheat=False, block_highvol=False),
        "평활 없음": replace(RULE, smooth=1),
        "진입 60 / 퇴출 40": replace(RULE, entry=60),
        "진입 80 / 퇴출 50": replace(RULE, entry=80, exit=50),
    }

    rows, sens_rows, pooled = [], [], []
    periods = [("전체", None, None), ("앞", None, pd.Timestamp(args.split) - pd.Timedelta(days=1)),
               ("뒤", pd.Timestamp(args.split), None)]
    for row in wl.itertuples(index=False):
        t = row.ticker
        df = prices(t)
        bench = prices(row.bench)["Close"] if row.bench else None
        p = get_profile(t)
        new = checklist_score(df, p, bench)
        v6 = compute_all(df, p, ["v6"], bench)["v6"]
        start = max(new["score_s"].first_valid_index(), v6["total"].first_valid_index())
        new, v6 = new.loc[start:], v6.loc[start:]
        bh = new["close"].pct_change().fillna(0.0)
        r_new = strategy_returns(new, COST)
        r_v6 = strategy_returns(v6, COST)
        for pname, a, b in periods:
            if len(bh.loc[a:b]) < 252:
                continue
            rows.append({"ticker": t, "period": pname, "start": bh.loc[a:b].index[0].date(),
                         **{f"bh_{k}": v for k, v in metrics(bh.loc[a:b], None).items()},
                         **{f"v6_{k}": v for k, v in metrics(r_v6.loc[a:b], v6["state"].loc[a:b]).items()},
                         **{f"new_{k}": v for k, v in metrics(r_new.loc[a:b], new["state"].loc[a:b]).items()}})
        bh_all = perf_stats(bh)
        for vname, rule in variants.items():
            o = checklist_score(df, p, bench, rule).loc[start:]
            m = metrics(strategy_returns(o, COST), o["state"])
            sens_rows.append({"variant": vname, "ticker": t, **m,
                              "bh_sharpe": bh_all["sharpe"], "bh_mdd": bh_all["mdd"]})
        f20 = new["close"].shift(-20) / new["close"] - 1
        v20 = np.log(new["close"]).diff().rolling(20).std().shift(-20) * np.sqrt(252)
        pooled.append(pd.DataFrame({"ticker": t, "score": new["score"], "score_s": new["score_s"],
                                    "state": new["state"], "f20": f20, "v20": v20}))
        print(f"{t}: {start.date()}~ 완료", file=sys.stderr)

    R = pd.DataFrame(rows)
    P = pd.concat(pooled)
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    def pct(v):
        return "-" if pd.isna(v) else f"{v * 100:.1f}%"

    for pname in ("전체", "앞", "뒤"):
        sub = R[R.period == pname]
        if sub.empty:
            continue
        out(f"\n## {pname} 기간 ({'~' + str(pd.Timestamp(args.split).date()) if pname == '앞' else str(pd.Timestamp(args.split).date()) + '~' if pname == '뒤' else '가능한 전체'}) — {len(sub)}종목")
        tbl = pd.DataFrame({
            "종목": sub.ticker, "시작": sub.start,
            "보유 CAGR": sub.bh_cagr.map(pct), "보유 MDD": sub.bh_mdd.map(pct), "보유 샤프": sub.bh_sharpe.round(2),
            "v6 CAGR": sub.v6_cagr.map(pct), "v6 MDD": sub.v6_mdd.map(pct), "v6 샤프": sub.v6_sharpe.round(2),
            "새 CAGR": sub.new_cagr.map(pct), "새 MDD": sub.new_mdd.map(pct), "새 샤프": sub.new_sharpe.round(2),
            "새 노출": sub.new_exposure.map(pct), "새 매매/년": sub.new_trades_per_year.round(1),
        })
        out(tbl.to_markdown(index=False) if _tab() else tbl.to_string(index=False))
        out("")
        out("| 요약 (종목 중앙값) | 보유 | 이전 규칙 v6 | 새 점수 규칙 |")
        out("|---|---:|---:|---:|")
        out(f"| CAGR | {pct(sub.bh_cagr.median())} | {pct(sub.v6_cagr.median())} | {pct(sub.new_cagr.median())} |")
        out(f"| MDD | {pct(sub.bh_mdd.median())} | {pct(sub.v6_mdd.median())} | {pct(sub.new_mdd.median())} |")
        out(f"| 샤프 | {sub.bh_sharpe.median():.2f} | {sub.v6_sharpe.median():.2f} | {sub.new_sharpe.median():.2f} |")
        out(f"| 노출 | 100% | {pct(sub.v6_exposure.median())} | {pct(sub.new_exposure.median())} |")
        out(f"| 매매/년 | – | {sub.v6_trades_per_year.median():.1f} | {sub.new_trades_per_year.median():.1f} |")
        n = len(sub)
        out(f"| MDD가 보유보다 작은 종목 | – | {(sub.v6_mdd > sub.bh_mdd).sum()}/{n} | {(sub.new_mdd > sub.bh_mdd).sum()}/{n} |")
        out(f"| 샤프가 보유보다 높은 종목 | – | {(sub.v6_sharpe > sub.bh_sharpe).sum()}/{n} | {(sub.new_sharpe > sub.bh_sharpe).sum()}/{n} |")
        out(f"| 샤프가 v6보다 높은 종목 | – | – | {(sub.new_sharpe > sub.v6_sharpe).sum()}/{n} |")

    out("\n## 점수(3일 평균) 구간별 이후 20일 (전 종목 합산, 구간 겹침 주의)")
    bins = [-1, 20, 40, 60, 80, 101]
    labels = ["0~20", "20~40", "40~60", "60~80", "80~100"]
    P["bucket"] = pd.cut(P.score_s, bins=bins, labels=labels, right=False)
    g = P.dropna(subset=["f20", "bucket"]).groupby("bucket", observed=True)
    tb = pd.DataFrame({"비중": g.size() / g.size().sum(), "평균 수익률": g.f20.mean(), "상승 확률": g.f20.apply(lambda s: (s > 0).mean()),
                       "이후 변동성(연)": g.v20.mean(), "하위 5% 수익률": g.f20.quantile(0.05)})
    tb = tb.map(pct).reset_index().rename(columns={"bucket": "점수"})
    out(tb.to_markdown(index=False) if _tab() else tb.to_string(index=False))

    L = P.rename_axis("Date").reset_index()
    piv_s = L.pivot_table(index="Date", columns="ticker", values="score")
    piv_f = L.pivot_table(index="Date", columns="ticker", values="f20").reindex(index=piv_s.index, columns=piv_s.columns)
    ics = []
    for d in piv_s.index:
        a, b = piv_s.loc[d], piv_f.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= 10 and a[m].nunique() > 1:
            ics.append(a[m].rank().corr(b[m].rank()))
    ics = pd.Series(ics)
    out(f"\n종목 간 순위상관(점수 vs 이후 20일 수익률, 날짜별 평균): {ics.mean():.3f} (양수인 날 {(ics > 0).mean():.0%}, {len(ics)}일)")

    out("\n## 민감도 (전체 기간, 종목 중앙값)")
    S = pd.DataFrame(sens_rows)
    sg = S.groupby("variant", sort=False)
    st = pd.DataFrame({"CAGR": sg.cagr.median().map(pct), "MDD": sg.mdd.median().map(pct), "샤프": sg.sharpe.median().round(2),
                       "노출": sg.exposure.median().map(pct), "매매/년": sg.trades_per_year.median().round(1),
                       "샤프 > 보유": sg.apply(lambda d: f"{(d.sharpe > d.bh_sharpe).sum()}/{len(d)}"),
                       "MDD < 보유": sg.apply(lambda d: f"{(d.mdd > d.bh_mdd).sum()}/{len(d)}")}).reset_index().rename(columns={"variant": "규칙"})
    out(st.to_markdown(index=False) if _tab() else st.to_string(index=False))

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tab() -> bool:
    import importlib.util
    return importlib.util.find_spec("tabulate") is not None


if __name__ == "__main__":
    main()
