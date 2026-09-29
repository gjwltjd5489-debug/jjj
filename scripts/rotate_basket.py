"""바스켓 로테이션: v5(v3 진입 + v2 청산) 보유 자격 종목 중 점수 상위 n개 보유.

예)
  # 기본 바스켓 (QQQ VWO GLD PDBC EDV UUP), 인터넷에서 바로 받기
  pip install finance-datareader
  python scripts/rotate_basket.py --top 2 --cash fdr:BIL

  # 후보에서 상관 낮은 6개 자동 선정 후 로테이션
  python scripts/rotate_basket.py --select-from QQQ VTV VWO EFA GLD PDBC EDV TIP UUP VNQ --k 6 --must QQQ

  # Investing.com CSV (data/QQQ.csv, data/GLD.csv ...) 로
  python scripts/rotate_basket.py --source "data/{t}.csv"

  # 인터넷 없이: 패키지 내장 실제 데이터로 만든 대용 바스켓 데모 (1999~2017)
  pip install arch zipline-reloaded --no-deps
  python scripts/rotate_basket.py --demo
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring import get_profile  # noqa: E402
from scoring.basket import (CANDIDATES, DEFAULT_BASKET, avg_offdiag, combined_corr,  # noqa: E402
                            return_corr, score_corr, select_low_corr)
from scoring.cards import COMBOS  # noqa: E402
from scoring.evaluate import positions_from_events  # noqa: E402
from scoring.portfolio import equal_weight, perf_stats, rotate, yearly_returns  # noqa: E402
from scoring.score import compute_all  # noqa: E402
from scoring.sources import load_prices, load_series  # noqa: E402

DEMO = {
    "NASDAQ": ("sample:nasdaq", "나스닥 종합 (QQQ 대용)"),
    "SP500": ("sample:sp500", "S&P 500 (미국 대형주)"),
    "WTI": ("sample:wti", "WTI 원유 현물 (원자재 대용, 롤 수익 없음)"),
    "LONGBOND": ("sample:longbond", "20년물 금리 합성 초장기채 (EDV 대용)"),
}


def spec_for(ticker: str, template: str) -> str:
    asset = CANDIDATES.get(ticker)
    if template.startswith("fdr:") and asset and asset.backfill:
        return template.format(t=f"{ticker}+{asset.backfill}")
    return template.format(t=ticker)


def load_universe(specs: dict[str, str], start: str | None):
    data, failed = {}, {}
    for t, spec in specs.items():
        try:
            data[t] = load_prices(spec, start)
        except Exception as e:  # 네트워크/심볼 오류는 종목별로 보고하고 계속
            failed[t] = f"{type(e).__name__}: {e}"
    return data, failed


def score_universe(data: dict[str, pd.DataFrame], rank: str):
    for name in ("v5", "v6"):
        if COMBOS[name].rank != rank:
            COMBOS[name] = replace(COMBOS[name], rank=rank)
    return {t: compute_all(df, get_profile(t), ["v2", "v3", "v5", "v6"]) for t, df in data.items()}


def fmt(v, pct=True, digits=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v * 100:.{digits}f}%" if pct else f"{v:.2f}"


def stats_row(name, st):
    return {
        "전략": name, "CAGR": fmt(st.get("cagr")), "변동성": fmt(st.get("vol")), "샤프": fmt(st.get("sharpe"), False),
        "MDD": fmt(st.get("mdd")), "칼마": fmt(st.get("calmar"), False),
        "평균 보유": fmt(st.get("avg_holdings"), False) if "avg_holdings" in st else "-",
        "현금 비중": fmt(st.get("cash_share")) if "cash_share" in st else "-",
        "연 회전율": fmt(st.get("turnover_per_year"), False) if "turnover_per_year" in st else "-",
        "편입 횟수": st.get("trades", "-"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="바스켓 로테이션 (v5 상위 n개)")
    ap.add_argument("--tickers", nargs="+", default=list(DEFAULT_BASKET), help="바스켓 종목")
    ap.add_argument("--source", default="fdr:{t}", help='데이터 템플릿 (기본 "fdr:{t}", 예: "data/{t}.csv")')
    ap.add_argument("--select-from", nargs="+", help="후보 종목: 상관 낮은 k 개를 자동 선정")
    ap.add_argument("--k", type=int, default=6, help="자동 선정 종목 수")
    ap.add_argument("--must", nargs="*", default=[], help="자동 선정 시 반드시 포함할 종목")
    ap.add_argument("--corr-weight", type=float, default=0.5, help="선정 기준: 수익률 상관 비중 (나머지는 점수 상관)")
    ap.add_argument("--top", type=int, default=2, help="보유 종목 수 n")
    ap.add_argument("--rebalance", type=int, default=5, help="교체 판단 주기(거래일, 기본 5 = 주 1회)")
    ap.add_argument("--buffer", type=int, default=1, help="순위 여유 (n+buffer 밖으로 밀려야 교체)")
    ap.add_argument("--cost", type=float, default=0.001, help="비중 변화 1당 거래비용 (기본 0.1%%)")
    ap.add_argument("--rank", choices=["v2", "v2v3"], default="v2", help="순위 점수: v2 추세 / v2+v3 평균")
    ap.add_argument("--cash", help="현금 수익률용 가격 (예: fdr:BIL, sample:cash). 없으면 0%%")
    ap.add_argument("--start", help="데이터 시작일")
    ap.add_argument("--demo", action="store_true", help="패키지 내장 실제 데이터로 만든 대용 바스켓")
    ap.add_argument("--md", help="결과 마크다운 저장 경로")
    args = ap.parse_args()

    lines: list[str] = []

    def out(text=""):
        print(text)
        lines.append(text)

    if args.demo:
        specs = {k: v[0] for k, v in DEMO.items()}
        cash_spec = args.cash or "sample:cash"
        select_from, must = list(specs), args.must or ["NASDAQ"]
        k = min(args.k, 3)
        out("대용 바스켓 데모 (패키지 내장 실제 데이터, 실제 ETF 아님)")
        for t, (_, desc) in DEMO.items():
            out(f"  {t:<9} {desc}")
    else:
        pool = args.select_from or args.tickers
        specs = {t: spec_for(t, args.source) for t in pool}
        cash_spec, select_from, must, k = args.cash, args.select_from, args.must, args.k

    data, failed = load_universe(specs, args.start)
    for t, err in failed.items():
        out(f"! {t} 불러오기 실패: {err}")
    if len(data) < 2:
        out("종목이 2개 이상 필요합니다. 데이터 소스를 확인하세요 (docs/data_sources.md).")
        sys.exit(1)

    res = score_universe(data, args.rank)
    closes = pd.DataFrame({t: d["Close"] for t, d in data.items()})
    v2_scores = pd.DataFrame({t: r["v2"]["total_s"] for t, r in res.items()})

    out("\n## 데이터")
    for t, d in data.items():
        out(f"  {t:<9} {d.index[0].date()} ~ {d.index[-1].date()} ({len(d)}행)")

    # 1) 상관관계와 종목 선정
    rc = return_corr(closes)
    sc = score_corr(v2_scores)
    cc = combined_corr(rc, sc, args.corr_weight)
    out("\n## 상관관계 (주간 수익률)")
    out(rc.round(2).to_string())
    out("\n## 상관관계 (v2 추세 점수)")
    out(sc.round(2).to_string())
    if select_from:
        chosen = select_low_corr(cc, k, tuple(must))
        out(f"\n## 선정 (수익률 {args.corr_weight:.0%} + 점수 {1 - args.corr_weight:.0%} 상관 기준, {k}개)")
        out(f"  {' '.join(chosen)}  (평균 상관 {avg_offdiag(cc, chosen):.2f}, 전체 후보 평균 {avg_offdiag(cc, list(cc.index)):.2f})")
    else:
        chosen = [t for t in args.tickers if t in data]
        out(f"\n## 바스켓: {' '.join(chosen)} (평균 상관 {avg_offdiag(cc, chosen):.2f})")

    # 2) 현재 상태
    v5 = {t: res[t]["v5"] for t in chosen}
    score = pd.DataFrame({t: v5[t]["total"] for t in chosen})
    elig = pd.DataFrame({t: v5[t]["state"] for t in chosen})
    rets = pd.DataFrame({t: data[t]["Close"].pct_change() for t in chosen})
    common = score.dropna().index
    if len(common) < 252:
        out("\n공통 기간이 1년 미만이라 백테스트를 생략합니다.")
        sys.exit(0)
    score, elig, rets = score.loc[common], elig.loc[common], rets.reindex(common)
    cash = load_series(cash_spec).reindex(common).ffill().pct_change() if cash_spec else None

    last = common[-1]
    status = []
    for t in chosen:
        row = v5[t].loc[last]
        status.append({"종목": t, "종가": round(row["close"], 2), "v2 추세": row["v2"], "v3 눌림": row["v3"],
                       "보유 자격": "O" if row["state"] == 1 else "-", "v5 점수": row["total"]})
    st_df = pd.DataFrame(status).sort_values("v5 점수", ascending=False)
    out(f"\n## 현재 점수 ({last.date()})")
    out(st_df.to_string(index=False))

    # 3) 백테스트
    out(f"\n## 백테스트 {common[0].date()} ~ {last.date()} (주 {args.rebalance}거래일 교체 판단, 비용 {args.cost:.1%}, "
        f"현금 {'= ' + cash_spec if cash_spec else '0%'})")
    rows = []
    main_res = None
    for n in sorted({1, 2, 3, args.top} & set(range(1, len(chosen) + 1))):
        r = rotate(score, elig, rets, n, args.rebalance, args.buffer, args.cost, cash)
        rows.append(stats_row(f"v5 상위 {n}", r.stats))
        if n == args.top:
            main_res = r
    # 비교 1: v6 (v3 눌림 또는 v2 추세 확인 진입) — v5 결과를 보고 만든 사후 아이디어
    v6_score = pd.DataFrame({t: res[t]["v6"]["total"] for t in chosen}).reindex(common)
    v6_elig = pd.DataFrame({t: res[t]["v6"]["state"] for t in chosen}).reindex(common)
    r6 = rotate(v6_score, v6_elig, rets, args.top, args.rebalance, args.buffer, args.cost, cash)
    rows.append(stats_row(f"v6 상위 {args.top} (사후 아이디어)", r6.stats))
    # 비교 2: v2 단독 로테이션 (v2 BUY~SELL 보유, v2 점수 순위)
    v2_elig = pd.DataFrame({t: positions_from_events(res[t]["v2"]).shift(-1) for t in chosen}).reindex(common)
    v2_score = v2_scores[chosen].reindex(common)
    r2 = rotate(v2_score, v2_elig, rets, args.top, args.rebalance, args.buffer, args.cost, cash)
    rows.append(stats_row(f"v2 단독 상위 {args.top}", r2.stats))
    ew = equal_weight(rets, 21, args.cost)
    rows.append(stats_row("동일가중 보유(월 리밸런스)", perf_stats(ew)))
    for t in chosen:
        rows.append(stats_row(f"{t} 보유", perf_stats(rets[t].fillna(0.0))))
    out(pd.DataFrame(rows).to_string(index=False))

    yr = pd.DataFrame({f"v5 상위 {args.top}": yearly_returns(main_res.returns),
                       f"v6 상위 {args.top}": yearly_returns(r6.returns),
                       f"v2 상위 {args.top}": yearly_returns(r2.returns),
                       "동일가중": yearly_returns(ew)})
    out("\n## 연도별 수익률")
    out(yr.map(lambda v: f"{v * 100:.1f}%").to_string())

    share = (main_res.weights > 0).mean().sort_values(ascending=False)
    out(f"\n## 종목별 보유 기간 비중 (v5 상위 {args.top}) · 현금 {main_res.stats['cash_share']:.0%}")
    out("  " + "  ".join(f"{t} {v:.0%}" for t, v in share.items()))

    held_now = [t for t in chosen if main_res.weights.loc[last, t] > 0]
    out(f"\n## 현재 보유안 (v5 상위 {args.top}): {', '.join(held_now) or '전부 현금'}")
    out("최근 매매")
    out(main_res.trades.tail(10).to_string(index=False))

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n저장: {args.md}")


if __name__ == "__main__":
    main()
