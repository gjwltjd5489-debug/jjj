"""미국장 마감 후 주요 종목 지표 체크리스트 (RSI, 볼린저, MACD, 일목, VR, OBV, 이동평균).

예)
  pip install finance-datareader
  python scripts/daily_checklist.py                               # config/watchlist.csv 전체
  python scripts/daily_checklist.py --tickers QQQ NVDA GLD
  python scripts/daily_checklist.py --out-md out/checklist.md --out-html out/checklist.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.checklist import build_report, render_html, render_markdown  # noqa: E402
from scoring.sources import load_prices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="일일 지표 체크리스트")
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"), help="group,ticker,name CSV")
    ap.add_argument("--tickers", nargs="+", help="watchlist 대신 이 종목들만")
    ap.add_argument("--source", default="fdr:{t}", help='데이터 템플릿 (기본 "fdr:{t}")')
    ap.add_argument("--years", type=int, default=3, help="불러올 기간(년). 200일선·백분위 계산에 2년 이상 필요")
    ap.add_argument("--out-md", help="마크다운 저장 경로")
    ap.add_argument("--out-html", help="HTML(메일 본문용) 저장 경로")
    args = ap.parse_args()

    if args.tickers:
        wl = pd.DataFrame({"group": "", "ticker": args.tickers, "name": ""})
    else:
        wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")

    reports, failed = [], []
    for row in wl.itertuples(index=False):
        try:
            df = load_prices(args.source.format(t=row.ticker), start)
            reports.append(build_report(df, get_profile(row.ticker), row.ticker, row.name, row.group))
        except Exception as e:  # 한 종목 실패가 전체를 막지 않게
            failed.append(f"{row.ticker}: {type(e).__name__}: {e}")
    if not reports:
        print("데이터를 하나도 불러오지 못했습니다:\n" + "\n".join(failed))
        sys.exit(1)

    md = render_markdown(reports)
    if failed:
        md += "\n\n불러오기 실패: " + "; ".join(failed)
    print(md)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    if args.out_html:
        page = render_html(reports)
        if failed:
            page += "<p style='color:#c62828'>불러오기 실패: " + "; ".join(failed) + "</p>"
        Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_html).write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
