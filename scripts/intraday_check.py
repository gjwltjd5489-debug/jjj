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
from scoring.bigtech import BENCH, GROUP as BIGTECH, load_members, load_watchlist, membership_year  # noqa: E402
from scoring.checklist import TickerReport, basket_state, trade_signal  # noqa: E402
from scoring.extras import earnings_dates, fmt_day, tbill_daily, upcoming  # noqa: E402
from scoring.intraday import basket_live, check_ticker, make_subject, render_html, render_text  # noqa: E402
from scoring.market_calendar import NY, holiday_name, is_trading_day  # noqa: E402
from scoring.sources import load_prices  # noqa: E402
from scoring.topk import DIP_TICKER  # noqa: E402

OPEN = time(9, 30)


def prev_report(df: pd.DataFrame, ticker: str, name: str, group: str, bench: pd.Series | None) -> TickerReport:
    """전일 종가까지의 데이터로 만든 최소 보고서 (바구니 계산용: 점수·규칙 상태·종가 기록)."""
    sig = trade_signal(df, get_profile(ticker), bench)
    return TickerReport(ticker=ticker, name=name, group=group, date=df.index[-1], close=float(df["Close"].iloc[-1]),
                        change=0.0, from_high=0.0, checks=[], sig=sig)


def live_basket(prices, upto, before, prev_reports, rows, wl, today, start, no_extras: bool) -> dict | None:
    """아침 메일의 바구니를 다시 계산(전일 종가까지)하고 지금 가격을 입힌다."""
    reports = [r for r in prev_reports if r.sig]
    # 지난해 빅테크 칸 종목: 메일 표에는 없지만 지난 기간 계산에 필요 (아침 메일과 같음)
    need = membership_year(today)
    for t in load_members().get(need - 1, []):
        if t not in set(wl["ticker"]):
            try:
                df = before(upto(prices(t)))
                reports.append(prev_report(df, t, t, BIGTECH, before(upto(prices(BENCH)))["Close"]))
            except Exception:
                pass
    spy = before(upto(prices("SPY")))["Close"]
    try:
        qdf = upto(prices(DIP_TICKER))
        qld_prev = before(qdf)["Close"]
        qld_move = float(qdf["Close"].iloc[-1] / qdf["Close"].iloc[-2] - 1) if qdf.index[-1].date() == today else None
    except Exception:
        qld_prev, qld_move = None, None
    ov = {"spy": spy, "qld": qld_prev, "group": {r.ticker: r.group for r in reports}}
    rf = None if no_extras else tbill_daily(start)
    b = basket_state([r for r in reports if r.sig], rf=rf, ov=ov)
    return basket_live(b, rows, spy, qld_move, today)


def main() -> None:
    ap = argparse.ArgumentParser(description="장 초반 확인")
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--now", help="기준 시각 (ISO, 시간대 포함) — 테스트용")
    ap.add_argument("--out-dir", default=str(ROOT / "out" / "intraday"))
    ap.add_argument("--no-extras", action="store_true", help="실적 일정 조회 생략 (오프라인 테스트용)")
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
              else load_watchlist(args.watchlist, today))
        start = (pd.Timestamp(today) - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
        cache: dict[str, pd.DataFrame] = {}

        def prices(t):
            if t not in cache:
                cache[t] = load_prices(f"fdr:{t}", start)
            return cache[t]
        def upto(df: pd.DataFrame) -> pd.DataFrame:
            # 과거 날짜 재현(--now): 그날까지의 데이터만 (마지막 막대 = 그날, 장중 대신 종가)
            return df.loc[: pd.Timestamp(today)] if args.now else df

        def before(df: pd.DataFrame) -> pd.DataFrame:
            """오늘 막대를 뺀 데이터 (아침 메일과 같은 기준)."""
            return df.iloc[:-1] if len(df) and df.index[-1].date() == today else df

        failed, prev_reports = [], []
        for r in wl.itertuples(index=False):
            try:
                bench = upto(prices(r.bench))["Close"] if r.bench and r.bench != r.ticker else None
                df = upto(prices(r.ticker))
                rows.append(check_ticker(df, get_profile(r.ticker), r.ticker, r.name, r.group, today, bench))
            except Exception as ex:  # 한 종목 실패가 전체를 막지 않게
                failed.append(f"{r.ticker} ({type(ex).__name__})")
                continue
            try:   # 바구니 계산용 (실패해도 이 종목의 장중 점검은 그대로)
                prev_reports.append(prev_report(before(df), r.ticker, r.name, r.group,
                                                before(bench) if bench is not None else None))
            except Exception:
                pass
        meta["failed"] = failed
        meta["mode"] = "normal" if any(r.has_intraday for r in rows) else "no_data"
        if meta["mode"] == "normal" and not args.tickers:
            # 바구니 현재 상황: 아침 메일과 같은 바구니(전일 종가 기준)를 지금 가격으로
            try:
                meta["basket"] = live_basket(prices, upto, before, prev_reports, rows, wl, today, start, args.no_extras)
            except Exception as ex:
                failed.append(f"바구니 계산 ({type(ex).__name__}: {ex})")
        # 오늘·내일 일정 (FOMC·CPI·실적 발표): 장중 매매 전 확인용
        earn = {} if args.no_extras else earnings_dates(list(wl["ticker"]), today)
        meta["today_events"] = [(fmt_day(d), label) for d, label in upcoming(today, 1, earn)]

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
    plain = {k: v for k, v in meta.items() if k != "basket"}
    if meta.get("basket"):
        plain["basket_today"] = round(meta["basket"]["today"], 6)
        plain["basket_signals"] = [k for k, _ in meta["basket"]["signals"]]
    (d / "meta.json").write_text(json.dumps({**plain, "subject": subject}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(subject + "\n")
    print(body)


if __name__ == "__main__":
    main()
