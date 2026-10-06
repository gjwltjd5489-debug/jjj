"""미국장 마감 후 주요 종목 지표 체크리스트 v2 (성향별 판정: 장기·중기 추세, 모멘텀, 추세 강도, 거래량, 상대강도, 변동성, 과열).

- 직전 거래일과 비교해 바뀐 항목을 강조한다.
- 데이터를 점검한다: 기준일 지연, 급변, 거래량 누락, 계산 안 된 지표.
- NYSE 휴장일에는 휴장 안내 형식으로 바꾼다.
- 마지막으로 끝난 정규장까지만 쓴다 (장중에 실행해도 진행 중인 오늘 막대를 종가로 쓰지 않음).
- 원/달러 환율과 다가오는 일정(FOMC·CPI·실적 발표)을 붙인다.
- 메일 제목·본문 파일까지 만들어서, 루틴은 파일만 읽어 보내면 된다.

예)
  pip install finance-datareader
  python scripts/daily_checklist.py --out-dir out            # out/checklist.html, checklist.md, subject.txt, body.txt, meta.json
  python scripts/daily_checklist.py --tickers QQQ NVDA GLD
  python scripts/daily_checklist.py --now "2026-11-27T07:37+09:00" --out-dir out   # 휴장일 형식 미리보기
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.checklist import (build_with_history, make_subject, render_html, render_markdown,  # noqa: E402
                               render_text)
from scoring.bigtech import BENCH, compute_members, load_members, load_watchlist, membership_year, save_members  # noqa: E402
from scoring.bigtech import GROUP as BIGTECH  # noqa: E402
from scoring.extras import calendar_note, earnings_dates, fmt_day, fx_summary, tbill_daily, upcoming  # noqa: E402
from scoring.market_calendar import session_status  # noqa: E402
from scoring.topk import DIP_TICKER  # noqa: E402
from scoring.sources import fetch_fdr, load_prices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="일일 지표 체크리스트")
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"), help="group,ticker,name CSV")
    ap.add_argument("--tickers", nargs="+", help="watchlist 대신 이 종목들만")
    ap.add_argument("--source", default="fdr:{t}", help='데이터 템플릿 (기본 "fdr:{t}")')
    ap.add_argument("--years", type=int, default=3, help="불러올 기간(년). 200일선·백분위 계산에 2년 이상 필요")
    ap.add_argument("--now", help="기준 시각 (ISO, 시간대 포함). 휴장·지연 판단 테스트용")
    ap.add_argument("--out-dir", help="checklist.md/html, subject.txt, body.txt, meta.json 저장 폴더")
    ap.add_argument("--out-md", help="마크다운 저장 경로")
    ap.add_argument("--out-html", help="HTML 저장 경로")
    ap.add_argument("--no-extras", action="store_true", help="환율·실적 일정 조회 생략 (오프라인 테스트용)")
    args = ap.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else None
    status = session_status(now)

    notes: list[str] = []
    extra: list[str] = []
    if args.tickers:
        wl = pd.DataFrame({"group": "", "ticker": args.tickers, "name": "", "bench": "SPY"})
    else:
        # 빅테크 칸: 그날 기준 나스닥 거래대금 상위 10 (scoring/bigtech.py). 새해 목록이 없으면 여기서 계산
        members = load_members()
        need = membership_year(status["target"])
        if need not in members:
            if args.no_extras:
                notes.append(f"빅테크 칸 {need}년 목록이 config/bigtech.csv 에 없어 지난 목록을 그대로 씀 "
                             "— scripts/update_bigtech.py 실행 후 커밋 필요")
            else:
                ranked = compute_members(need, lambda t, s: load_prices(f"fdr:{t}", s))
                save_members({need: ranked})
                members = load_members()
                notes.append(f"빅테크 칸 {need}년 목록을 새로 계산함 ({', '.join(ranked.index)}) "
                             "— config/bigtech.csv 커밋 필요")
        wl = load_watchlist(args.watchlist, status["target"], members)
        # 지난해 빅테크 칸 종목은 메일 표에는 빼고 바구니 지난 기간 계산에만 쓴다
        extra = [t for t in members.get(need - 1, []) if t not in set(wl["ticker"])]
    if "bench" not in wl.columns:
        wl["bench"] = ""
    start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")

    # 마지막으로 끝난 정규장까지만: 장중에 실행하면 진행 중인 오늘 막대가 종가처럼 쓰이는 것을 막는다.
    # (--now 로 과거 시점을 흉내 낼 때도 그 시점까지만)
    cut = pd.Timestamp(status["target"])
    benches: dict[str, pd.Series | None] = {}

    def bench_for(sym: str, ticker: str) -> pd.Series | None:
        if not sym or sym == ticker:
            return None
        if sym not in benches:
            try:
                benches[sym] = load_prices(args.source.format(t=sym), start)["Close"].loc[:cut]
            except Exception:
                benches[sym] = None
        return benches[sym]

    reports, failed = [], []
    for row in wl.itertuples(index=False):
        try:
            df = load_prices(args.source.format(t=row.ticker), start).loc[:cut]
            reports.append(build_with_history(df, get_profile(row.ticker), row.ticker, row.name, row.group,
                                              expected=status["target"], bench=bench_for(row.bench, row.ticker)))
        except Exception as e:  # 한 종목 실패가 전체를 막지 않게
            failed.append(f"{row.ticker} ({type(e).__name__}: {e})")
    if not reports:
        print("데이터를 하나도 불러오지 못했습니다:\n" + "\n".join(failed))
        sys.exit(1)
    basket_extra = []
    for t in extra:
        try:
            df = load_prices(args.source.format(t=t), start).loc[:cut]
            basket_extra.append(build_with_history(df, get_profile(t), t, t, BIGTECH, expected=status["target"],
                                                   bench=bench_for(BENCH, t)))
        except Exception:
            pass

    latest = max(r.date for r in reports).date()
    if status["holiday"]:
        mode = "holiday"
        # 휴장일에는 기준일 지연 경고가 의미 없으므로 뺀다
        for r in reports:
            r.issues = [i for i in r.issues if "데이터 지연" not in i]
    elif latest < status["target"]:
        mode = "delayed"
    else:
        mode = "normal"
    meta = {"mode": mode, "failed": failed, "notes": notes, **{k: str(v) if v else v for k, v in status.items()}}
    meta["basket_extra"] = basket_extra
    meta["rf"] = None if args.no_extras else tbill_daily(start)
    # SPY 200일선 필터·QLD 하락 매수 (scoring/topk.py). QLD 가격을 못 받으면 메일에 그렇게 적고 필터만 쓴다
    try:
        qld = load_prices(args.source.format(t=DIP_TICKER), start)["Close"].loc[:cut]
    except Exception:
        qld = None
    spy = bench_for("SPY", "")
    # 올해 누적 수익률 비교: QQQ 는 표에 있고, 60/40 은 SPY 60 · AGG 40
    try:
        meta["bench_px"] = {"AGG": load_prices(args.source.format(t="AGG"), start)["Close"].loc[:cut]}
    except Exception:
        meta["bench_px"] = None
    meta["overlay"] = None if spy is None else {"spy": spy, "qld": qld,
                                                "group": {r.ticker: r.group for r in reports + basket_extra}}
    ev_from = status["next_open"]  # 일정은 다음 거래일부터
    if not args.no_extras:
        try:
            fx = fetch_fdr("USD/KRW", start)["Close"]  # 환율은 배당 조정 없이 종가 그대로
            fx.index = pd.to_datetime(fx.index)
            meta["fx"] = fx_summary(fx.loc[: pd.Timestamp(status["checked"]) + pd.Timedelta(days=1)] if now else fx)
        except Exception as e:
            meta["fx"] = None
            print(f"환율 불러오기 실패: {type(e).__name__}", file=sys.stderr)
        earn = earnings_dates(list(wl["ticker"]), ev_from)
    else:
        earn = {}
    meta["upcoming"] = [(fmt_day(d), label) for d, label in upcoming(ev_from, 14, earn)]
    meta["calendar_note"] = calendar_note(ev_from)

    md = render_markdown(reports, meta)
    subject = make_subject(reports, meta)
    print(subject + "\n")
    print(md)

    outputs = {}
    if args.out_dir:
        d = Path(args.out_dir)
        outputs = {"md": d / "checklist.md", "html": d / "checklist.html"}
        d.mkdir(parents=True, exist_ok=True)
        (d / "subject.txt").write_text(subject + "\n", encoding="utf-8")
        (d / "body.txt").write_text(render_text(reports, meta) + "\n", encoding="utf-8")
        plain = {k: v for k, v in meta.items() if k not in ("basket_extra", "rf", "overlay", "bench_px")}
        (d / "meta.json").write_text(json.dumps({**plain, "latest": str(latest), "subject": subject},
                                                ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        outputs["md"] = Path(args.out_md)
    if args.out_html:
        outputs["html"] = Path(args.out_html)
    if "md" in outputs:
        outputs["md"].parent.mkdir(parents=True, exist_ok=True)
        outputs["md"].write_text(md + "\n", encoding="utf-8")
    if "html" in outputs:
        outputs["html"].parent.mkdir(parents=True, exist_ok=True)
        outputs["html"].write_text(render_html(reports, meta), encoding="utf-8")


if __name__ == "__main__":
    main()
