"""장 초반 확인 메일: 아침 신호를 장중 가격으로 다시 점검 (개장 30분 뒤 실행 용도).

예)
  python scripts/intraday_check.py --out-dir out/intraday      # intraday.html, subject.txt, body.txt, meta.json
  python scripts/intraday_check.py --tickers QQQ NVDA
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.intraday import check_ticker, make_subject, render_html, render_text  # noqa: E402
from scoring.market_calendar import NY, holiday_name, is_trading_day  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

OPEN = time(9, 30)


def main() -> None:
    ap = argparse.ArgumentParser(description="장 초반 확인")
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--now", help="기준 시각 (ISO, 시간대 포함) — 테스트용")
    ap.add_argument("--out-dir", default=str(ROOT / "out" / "intraday"))
    args = ap.parse_args()

    now = (datetime.fromisoformat(args.now) if args.now else datetime.now(tz=NY)).astimezone(NY)
    today = now.date()
    minutes = max(0, int((datetime.combine(today, now.time()) - datetime.combine(today, OPEN)).total_seconds() // 60))
    meta = {"date": str(today), "time": now.strftime("%H:%M"), "minutes": minutes, "holiday": None}

    rows = []
    if not is_trading_day(today):
        meta["mode"] = "holiday"
        meta["holiday"] = holiday_name(today) or "주말"
    else:
        wl = (pd.DataFrame({"group": "", "ticker": args.tickers, "name": "", "bench": "SPY"}) if args.tickers
              else pd.read_csv(args.watchlist, dtype=str).fillna(""))
        start = (pd.Timestamp(today) - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
        cache: dict[str, pd.DataFrame] = {}

        def prices(t):
            if t not in cache:
                cache[t] = load_prices(f"fdr:{t}", start)
            return cache[t]
        failed = []
        for r in wl.itertuples(index=False):
            try:
                bench = prices(r.bench)["Close"] if r.bench and r.bench != r.ticker else None
                df = prices(r.ticker)
                if args.now:  # 과거 날짜 재현: 그날까지의 데이터만 (마지막 막대 = 그날, 장중 대신 종가)
                    df = df.loc[: pd.Timestamp(today)]
                    bench = bench.loc[: pd.Timestamp(today)] if bench is not None else None
                rows.append(check_ticker(df, get_profile(r.ticker), r.ticker, r.name, r.group, today, bench))
            except Exception as ex:  # 한 종목 실패가 전체를 막지 않게
                failed.append(f"{r.ticker} ({type(ex).__name__})")
        meta["failed"] = failed
        meta["mode"] = "normal" if any(r.has_intraday for r in rows) else "no_data"

    subject = make_subject(rows, meta)
    body = render_text(rows, meta)
    page = render_html(rows, meta)
    if meta.get("failed"):
        body += "\n\n불러오기 실패: " + ", ".join(meta["failed"])
        page += "<p style='color:#c62828'>불러오기 실패: " + ", ".join(meta["failed"]) + "</p>"
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "subject.txt").write_text(subject + "\n", encoding="utf-8")
    (d / "body.txt").write_text(body + "\n", encoding="utf-8")
    (d / "intraday.html").write_text(page, encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({**meta, "subject": subject}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(subject + "\n")
    print(body)


if __name__ == "__main__":
    main()
