"""루틴이 쓴 뉴스 요약(news.json)을 메일 HTML·텍스트 맨 아래에 붙인다.

fetch_news.py 가 실패했거나 news.json 이 없으면 "뉴스를 가져오지 못함" 한 줄을 붙인다.
같은 파일에 두 번 돌려도 뉴스 구역은 하나만 남는다.

예)
  python scripts/append_news.py --news-dir out/news --html out/checklist.html --body out/body.txt
  python scripts/append_news.py --news-dir out/news --html out/intraday/intraday.html --body out/intraday/body.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.news import MARK, TEXT_MARK, append, load_summary, render_html, render_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--news-dir", default="out/news")
    ap.add_argument("--html", required=True)
    ap.add_argument("--body", required=True)
    args = ap.parse_args()

    nd = Path(args.news_dir)
    try:
        meta = json.loads((nd / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as ex:
        sys.exit(f"meta.json 을 읽지 못함 (fetch_news.py 를 먼저 실행): {ex}")
    summary = load_summary(nd / "news.json", meta)
    for path, section, sep, mark in ((args.html, render_html(summary, meta), "", MARK),
                                     (args.body, render_text(summary, meta), "\n\n", TEXT_MARK)):
        p = Path(path)
        p.write_text(append(p.read_text(encoding="utf-8"), section, sep, mark), encoding="utf-8")
    print(f"뉴스 {summary.get('status')} {len(summary.get('items') or [])}건 붙임"
          + (f" ({summary['note']})" if summary.get("note") else ""))


if __name__ == "__main__":
    main()
