"""saveticker.com/news 화면을 받아 루틴이 요약할 재료를 만든다.

헤드리스 크롬(Playwright)으로 페이지를 열고 아래로 내려 뉴스를 더 불러온 뒤
화면 글자(page.txt)와 페이지가 받은 JSON(api.json)을 저장한다. Playwright 가 없으면 HTML 만 받는다.
실패해도 종료 코드는 0이고, meta.json 의 status 가 "fail" 이 된다 (메일은 뉴스 없이 나가야 하므로).

예)
  python -m pip install playwright==1.56.0          # 크롬은 /opt/pw-browsers 에 이미 있음
  python scripts/fetch_news.py --kind morning --out-dir out/news     # page.txt, api.json, instructions.txt, meta.json
  python scripts/fetch_news.py --kind intraday --now "2026-10-13T10:03-04:00" --out-dir out/news
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.news import KST, URL, instructions, window  # noqa: E402

API_MAX = 2_000_000   # api.json 최대 글자 수
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/141.0 Safari/537.36")


def fetch_browser(url: str, scrolls: int) -> tuple[str, list]:
    from playwright.sync_api import sync_playwright

    found: list = []

    def keep(resp):
        if "json" in (resp.headers.get("content-type") or ""):
            try:
                found.append({"url": resp.url, "data": resp.json()})
            except Exception:  # noqa: BLE001 - 본문이 JSON 이 아니면 버린다
                pass

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=UA, locale="ko-KR", timezone_id="Asia/Seoul",
                                viewport={"width": 1280, "height": 2000})
        page.on("response", keep)
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=20_000)
        except Exception:  # noqa: BLE001 - 계속 연결을 여는 사이트도 있다
            pass
        height = 0
        for _ in range(scrolls):
            more = page.get_by_text(re.compile(r"^\s*(더\s?보기|more|load more)\s*$", re.I))
            if more.count():
                try:
                    more.first.click(timeout=3_000)
                except Exception:  # noqa: BLE001
                    pass
            page.mouse.wheel(0, 6000)
            page.wait_for_timeout(1_500)
            h = page.evaluate("document.body.scrollHeight")
            if h == height and not more.count():
                break
            height = h
        text = page.inner_text("body")
        browser.close()
    return text, found


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style") and self.skip > 0

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.out.append(data.strip())


def fetch_plain(url: str) -> tuple[str, list]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko-KR"})
    raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    found = []
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', raw, re.S)
    if m:
        found.append({"url": url + " (__NEXT_DATA__)", "data": json.loads(m.group(1))})
    p = _Text()
    p.feed(raw)
    return "\n".join(p.out), found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["morning", "intraday"], required=True)
    ap.add_argument("--out-dir", default="out/news")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--now", help="ISO 시각 (미리보기용)")
    ap.add_argument("--scrolls", type=int, default=10)
    args = ap.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(KST)
    since, until = window(args.kind, now)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"kind": args.kind, "url": args.url, "since": since.isoformat(), "until": until.isoformat()}

    text, found, errors = "", [], []
    for name, fn in (("browser", lambda: fetch_browser(args.url, args.scrolls)), ("plain", lambda: fetch_plain(args.url))):
        try:
            text, found = fn()
            meta["method"] = name
            if len(text.strip()) >= 200:
                break
            errors.append(f"{name}: 글자 {len(text.strip())}자뿐")
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{name}: {type(ex).__name__}: {str(ex).splitlines()[0][:200] if str(ex) else ''}")
    ok = len(text.strip()) >= 200
    meta.update(status="ok" if ok else "fail", chars=len(text), api=len(found))
    if not ok:
        detail = " / ".join(errors) or "내용 없음"
        blocked = re.search(r"403|TUNNEL_CONNECTION_FAILED|Tunnel connection failed", detail)
        meta["error"] = ("사이트 접속 차단 (루틴 환경의 네트워크 허용 도메인에 www.saveticker.com 추가 필요)"
                         if blocked else detail)
        meta["detail"] = detail

    api = json.dumps(found, ensure_ascii=False)
    (d / "page.txt").write_text(text, encoding="utf-8")
    (d / "api.json").write_text(api[:API_MAX] if len(api) <= API_MAX else json.dumps(
        [{"url": f["url"]} for f in found], ensure_ascii=False), encoding="utf-8")
    (d / "news.json").unlink(missing_ok=True)
    try:
        import pandas as pd
        tickers = pd.read_csv(ROOT / "config" / "watchlist.csv", dtype=str)["ticker"].tolist()
        bt = pd.read_csv(ROOT / "config" / "bigtech.csv")
        tickers += bt.loc[bt["year"] == bt["year"].max(), "ticker"].tolist()
    except Exception:  # noqa: BLE001
        tickers = []
    (d / "instructions.txt").write_text(instructions(args.kind, since, until, list(dict.fromkeys(tickers)),
                                                     str(d.resolve())), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NEWS={d.resolve()} status={meta['status']} method={meta.get('method', '-')} chars={len(text)} "
          f"api={len(found)}" + (f" error={meta['error']}" if not ok else ""))


if __name__ == "__main__":
    main()
