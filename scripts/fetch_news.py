"""언론사 RSS(config/news_feeds.csv)를 받아 구간 안 기사만 모아 루틴이 요약할 재료를 만든다.

피드 하나가 실패해도 나머지로 진행한다. 모두 실패해도 종료 코드는 0이고 meta.json 의 status 가 "fail" 이 된다
(메일은 뉴스 없이 나가야 하므로).

예)
  python scripts/fetch_news.py --kind morning --out-dir out/news     # items.txt, items.json, instructions.txt, meta.json
  python scripts/fetch_news.py --kind intraday --now "2026-10-13T10:03-04:00" --out-dir out/news
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.news import KST, fmt_kst, in_window, instructions, parse_feed, window  # noqa: E402

FEEDS = ROOT / "config" / "news_feeds.csv"
UA = "Mozilla/5.0 (compatible; jjj-news/1.0; +https://github.com/gjwltjd5489-debug/jjj)"
BLOCKED = re.compile(r"403|Tunnel connection failed|Forbidden")


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml"})
    return urllib.request.urlopen(req, timeout=20).read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["morning", "intraday"], required=True)
    ap.add_argument("--out-dir", default="out/news")
    ap.add_argument("--feeds", default=str(FEEDS))
    ap.add_argument("--now", help="ISO 시각 (미리보기용)")
    args = ap.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(KST)
    since, until = window(args.kind, now)
    feeds = pd.read_csv(args.feeds, dtype=str)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)

    def one(row):
        try:
            return row.name, parse_feed(fetch(row.url), row.name), None
        except Exception as ex:  # noqa: BLE001 - 피드 하나의 실패는 기록만 한다
            return row.name, [], f"{type(ex).__name__}: {str(ex).splitlines()[0][:150] if str(ex) else ''}"

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(one, list(feeds.itertuples(index=False))))
    status, everything = {}, []
    for name, items, err in results:
        if err:
            status[name] = f"실패 {err}"
            continue
        dated = sum(i["time"] is not None for i in items)
        status[name] = f"ok {len(items)}건 (시각 있음 {dated})"
        everything += items
    items = in_window(everything, since, until)

    ok = any(v.startswith("ok") for v in status.values())
    meta = {"kind": args.kind, "since": since.isoformat(), "until": until.isoformat(),
            "status": "ok" if ok else "fail", "count": len(items), "feeds": status}
    if not ok:
        hosts = sorted({urlparse(u).hostname for u in feeds["url"]})
        meta["error"] = ("뉴스 사이트 접속 차단 (루틴 환경의 네트워크 허용 도메인에 추가 필요: " + ", ".join(hosts) + ")"
                         if all(BLOCKED.search(v) for v in status.values()) else "모든 RSS 피드를 받지 못함")

    lines = [f"[{fmt_kst(i['time'])}] ({i['source']}) {i['title']}" + (f"\n  {i['desc']}" if i["desc"] else "")
             + (f"\n  {i['url']}" if i["url"] else "") for i in items]
    (d / "items.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "items.json").write_text(json.dumps([{**i, "time": i["time"].isoformat()} for i in items],
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "news.json").unlink(missing_ok=True)
    try:
        tickers = pd.read_csv(ROOT / "config" / "watchlist.csv", dtype=str)["ticker"].tolist()
        bt = pd.read_csv(ROOT / "config" / "bigtech.csv")
        tickers += bt.loc[bt["year"] == bt["year"].max(), "ticker"].tolist()
    except Exception:  # noqa: BLE001
        tickers = []
    (d / "instructions.txt").write_text(instructions(args.kind, since, until, list(dict.fromkeys(tickers)),
                                                     str(d.resolve()), len(items)), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NEWS={d.resolve()} status={meta['status']} items={len(items)}"
          + (f" error={meta['error']}" if not ok else ""))
    for name, v in status.items():
        print(f"  {name}: {v}")


if __name__ == "__main__":
    main()
