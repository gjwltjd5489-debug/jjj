"""모든 카드를 같은 데이터로 비교 (전체 기간 + 기간 분할).

예)
  pip install arch --no-deps
  python scripts/compare_cards.py sample:nasdaq --bench sample:sp500 --split 2009-01-01
  python scripts/compare_cards.py fdr:QQQ --bench fdr:SPY --vix fdr:VIX --split 2016-01-01 --md out/compare.md
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_score import add_data_args, load_inputs  # noqa: E402

from scoring import get_profile  # noqa: E402
from scoring.cards import CARDS, COMBOS  # noqa: E402
from scoring.evaluate import fmt_pct, summarize  # noqa: E402
from scoring.score import compute_all  # noqa: E402

PCT = ["strong_buy_20d", "buy_20d", "sell_20d", "strong_sell_20d", "buy_zone_share",
       "cagr", "mdd", "exposure", "bh_cagr", "bh_mdd"]
COLUMNS = {
    "card": "카드", "IC20": "IC20", "IC60": "IC60",
    "strong_buy_20d": "강매수 20일", "buy_20d": "매수 20일", "sell_20d": "매도 20일", "strong_sell_20d": "강매도 20일",
    "buy_zone_share": "매수구간 비중", "daily_jump": "일평균 변동",
    "cagr": "CAGR", "mdd": "MDD", "sharpe": "샤프", "exposure": "노출", "round_trips": "매매",
    "bh_cagr": "보유 CAGR", "bh_mdd": "보유 MDD", "bh_sharpe": "보유 샤프",
}


def table(results, start=None, end=None, cost=0.0005) -> pd.DataFrame:
    sub = {k: v.loc[start:end] for k, v in results.items()}
    s = summarize(sub, cost)
    s = fmt_pct(s, PCT)
    for c in ("IC20", "IC60", "sharpe", "bh_sharpe", "daily_jump"):
        s[c] = s[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    return s.rename(columns=COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description="카드 비교")
    add_data_args(ap)
    ap.add_argument("--split", help="기간 분할 날짜 (앞/뒤 따로 평가)")
    ap.add_argument("--cost", type=float, default=0.0005, help="편도 거래비용 (기본 0.0005)")
    ap.add_argument("--md", help="결과를 마크다운 파일로 저장")
    args = ap.parse_args()

    df, bench, ext = load_inputs(args)
    results = compute_all(df, get_profile(args.ticker), list(CARDS) + list(COMBOS), bench, ext)

    periods = [("전체", None, None)]
    if args.split:
        split = pd.Timestamp(args.split)
        periods += [(f"~{(split - pd.Timedelta(days=1)).date()}", None, split - pd.Timedelta(days=1)),
                    (f"{split.date()}~", split, None)]

    lines = [f"데이터 {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행), "
             f"벤치마크 {'있음' if bench is not None else '없음'}, 외부 {', '.join(ext) or '없음'}"]
    for name, a, b in periods:
        t = table(results, a, b, args.cost)
        print(f"\n[{name}]")
        print(t.to_string(index=False))
        lines += ["", f"### {name}", "", t.to_markdown(index=False) if _has_tabulate() else t.to_string(index=False)]

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n저장: {args.md}")


def _has_tabulate() -> bool:
    return importlib.util.find_spec("tabulate") is not None


if __name__ == "__main__":
    main()
