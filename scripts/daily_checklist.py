"""미국장 마감 후 주요 종목 지표 체크리스트 (RSI, 볼린저, MACD, 일목, VR, OBV, 이동평균).

- 직전 거래일과 비교해 바뀐 항목을 강조한다.
- 데이터를 점검한다: 기준일 지연, 급변, 거래량 누락, 계산 안 된 지표.
- NYSE 휴장일에는 휴장 안내 형식으로 바꾼다.
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
from scoring.market_calendar import session_status  # noqa: E402
from scoring.sources import load_prices  # noqa: E402


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
    args = ap.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else None
    status = session_status(now)

    if args.tickers:
        wl = pd.DataFrame({"group": "", "ticker": args.tickers, "name": ""})
    else:
        wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")

    reports, failed = [], []
    for row in wl.itertuples(index=False):
        try:
            df = load_prices(args.source.format(t=row.ticker), start)
            # --now 로 과거 시점을 흉내 낼 때는 그 시점까지의 데이터만 쓴다
            df = df.loc[: pd.Timestamp(status["target"])] if now else df
            reports.append(build_with_history(df, get_profile(row.ticker), row.ticker, row.name, row.group,
                                              expected=status["target"]))
        except Exception as e:  # 한 종목 실패가 전체를 막지 않게
            failed.append(f"{row.ticker} ({type(e).__name__}: {e})")
    if not reports:
        print("데이터를 하나도 불러오지 못했습니다:\n" + "\n".join(failed))
        sys.exit(1)

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
    meta = {"mode": mode, "failed": failed, **{k: str(v) if v else v for k, v in status.items()}}

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
        (d / "meta.json").write_text(json.dumps({**meta, "latest": str(latest), "subject": subject},
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
