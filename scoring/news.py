"""메일 아래에 붙이는 주요 뉴스 (언론사 RSS, config/news_feeds.csv).

- 아침 메일: 직전 장초반 메일(뉴욕 10:03) 뒤부터 지금까지
- 장초반 메일: 직전 아침 메일(서울 06:00) 뒤부터 지금까지
- RSS 를 받아 구간 안 기사만 남기는 것은 이 모듈이, 중요한 뉴스 고르기·요약은
  루틴(Claude)이 instructions.txt 를 따라 news.json 으로 쓰고, 이 모듈이 그 파일을 메일 맨 아래에 붙인다.
"""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET_XML
from datetime import datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
MARK = "<!--news-->"
TEXT_MARK = "─" * 20   # 텍스트 본문의 뉴스 구역 시작 줄

# 각 루틴의 예정 시각 (루틴 cron 과 같게 유지). 요일: 월=0
SCHEDULE = {
    "morning": (KST, time(6, 0), {1, 2, 3, 4, 5}),   # 서울 화~토 06:00
    "intraday": (ET, time(10, 3), {0, 1, 2, 3, 4}),  # 뉴욕 월~금 10:03
}
OTHER = {"morning": "intraday", "intraday": "morning"}
LABEL = {"morning": "아침 메일", "intraday": "장초반 메일"}
MAX_ITEMS = {"morning": 8, "intraday": 6}


def last_run(kind: str, now: datetime) -> datetime:
    """now 이전(같은 시각 포함하지 않음) 가장 최근의 kind 루틴 예정 시각."""
    tz, at, days = SCHEDULE[kind]
    local = now.astimezone(tz)
    for back in range(8):
        d = local.date() - timedelta(days=back)
        t = datetime.combine(d, at, tzinfo=tz)
        if d.weekday() in days and t < local:
            return t
    raise ValueError(kind)


def window(kind: str, now: datetime) -> tuple[datetime, datetime]:
    """kind 메일에 넣을 뉴스 구간: 다른 메일의 직전 예정 시각 ~ now."""
    return last_run(OTHER[kind], now), now


def _date(text: str | None) -> datetime | None:
    if not text:
        return None
    text = text.strip()
    try:
        t = parsedate_to_datetime(text)          # RSS: "Tue, 13 Oct 2026 05:12:00 GMT"
    except (TypeError, ValueError):
        try:
            t = datetime.fromisoformat(text.replace("Z", "+00:00"))   # Atom: ISO 8601
        except ValueError:
            return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _plain(text: str | None, n: int = 300) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


def parse_feed(xml: str | bytes, source: str) -> list[dict]:
    """RSS 2.0 / Atom 을 [{time, source, title, desc, url}] 로. 시각이 없는 기사는 time None."""
    root = ET_XML.fromstring(xml)
    for el in root.iter():                       # 네임스페이스 떼기
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    out = []
    for it in list(root.iter("item")) + list(root.iter("entry")):
        link = it.findtext("link") or ""
        if not link.strip():
            link = next((l.get("href", "") for l in it.findall("link") if l.get("rel", "alternate") == "alternate"), "")
        when = _date(it.findtext("pubDate") or it.findtext("published") or it.findtext("updated")
                     or it.findtext("date"))
        out.append({"time": when, "source": source, "title": _plain(it.findtext("title"), 200),
                    "desc": _plain(it.findtext("description") or it.findtext("summary") or it.findtext("content")),
                    "url": link.strip()})
    return out


def in_window(items: list[dict], since: datetime, until: datetime) -> list[dict]:
    """구간 안 기사만, 최신순. 같은 링크·같은 제목은 먼저 나온 것 하나만."""
    seen, out = set(), []
    for it in sorted((i for i in items if i["time"] and since < i["time"] <= until), key=lambda i: i["time"]):
        keys = {it["title"].lower()} | ({it["url"]} if it["url"] else set())
        if keys & seen:
            continue
        seen |= keys
        out.append(it)
    return out[::-1]


def fmt_kst(t: datetime) -> str:
    t = t.astimezone(KST)
    return f"{t.month}/{t.day} {t:%H:%M}"


def window_label(since: datetime, until: datetime) -> str:
    return f"{fmt_kst(since)} ~ {fmt_kst(until)} (한국시간)"


def instructions(kind: str, since: datetime, until: datetime, tickers: list[str], out_dir: str, count: int) -> str:
    """루틴이 읽고 따를 뉴스 고르기·요약 지침."""
    n = MAX_ITEMS[kind]
    return f"""# 주요 뉴스 요약 지침 ({LABEL[kind]})

입력: {out_dir}/items.txt (언론사 RSS 기사 {count}건: [한국시간] (매체) 제목 / 요약 / 링크, 최신순)
구간: {since.astimezone(KST):%Y-%m-%d %H:%M} ~ {until.astimezone(KST):%Y-%m-%d %H:%M} 한국시간
      ({since.astimezone(ET):%m/%d %H:%M} ~ {until.astimezone(ET):%m/%d %H:%M} 뉴욕시간)
items.txt 에는 이미 구간 안 기사만 들어 있다.

1. 미국 증시에 영향이 큰 것만 고른다 (중요한 순서, 최대 {n}개. 적으면 적은 대로):
   - 연준·금리(FOMC, 연준 인사 발언, 국채금리 급변), 경제지표(CPI·고용·PCE·ISM·소매판매 등) 결과
   - 관세·무역·정부 정책·지정학(전쟁·제재·선거), 셧다운·부채한도
   - 대형주 실적·가이던스·대형 인수합병·규제, 반도체·AI 업황
   - 유가·달러·금·비트코인의 큰 움직임, 아시아·유럽 증시 급변
   - 관심 종목: {", ".join(tickers)}
   빼는 것: 단순 시세 요약, 소형주·목표가 조정, 광고·칼럼·의견·생활 기사.
   같은 사건을 여러 매체가 쓴 경우 하나로 합치고, 가장 먼저 나온 기사의 시각·링크를 쓴다.
2. 각 뉴스: 제목은 한국어 한 줄(40자 안쪽), 요약은 한국어 1~2문장으로 숫자(%, 금액, 예상치 대비)를 살려 쓴다.
   items.txt 의 제목·요약에 있는 내용만 쓴다. 요약이 너무 짧아 핵심(숫자·결과)을 알 수 없을 때만
   WebSearch 로 같은 사건 보도를 찾아 보탠다 — 주체와 내용이 같고 보도 날짜가 구간 안(하루 안쪽)일 때만 쓴다.
   언론사 사이트에 curl·WebFetch 로 직접 접속하지 않는다.
3. {out_dir}/news.json 에 아래 형식으로 저장한다 (UTF-8, JSON 만):
{{"status": "ok", "items": [
  {{"time": "10/13 05:12", "title": "...", "summary": "...", "tickers": ["NVDA"], "source": "CNBC", "url": "https://..."}}
]}}
   - time 은 items.txt 의 한국시간 그대로, source 는 매체 이름, url 은 기사 링크.
   - 고를 뉴스가 없으면 {{"status": "none", "items": []}}.
"""


def load_summary(path: Path | str | None, fetch_meta: dict) -> dict:
    """news.json 을 읽는다. 가져오기에 실패했거나 파일이 없으면 실패 안내로 바꾼다."""
    if fetch_meta.get("status") != "ok":
        return {"status": "fail", "note": f"뉴스를 가져오지 못함: {fetch_meta.get('error', '알 수 없는 오류')}",
                "items": []}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    except (OSError, ValueError) as ex:
        return {"status": "fail", "note": f"요약 파일을 읽지 못함: {ex}", "items": []}
    if not isinstance(data, dict):
        return {"status": "fail", "note": "요약 파일 없음", "items": []}
    data.setdefault("items", [])
    data.setdefault("status", "ok" if data["items"] else "none")
    return data


def _head(fetch_meta: dict) -> str:
    since, until = (datetime.fromisoformat(fetch_meta[k]) for k in ("since", "until"))
    return f"📰 주요 뉴스 · {window_label(since, until)}"


def _sources(fetch_meta: dict) -> str:
    names = list(dict.fromkeys(n.split()[0] for n, r in (fetch_meta.get("feeds") or {}).items() if r.startswith("ok")))
    return "출처: " + (", ".join(names) + " RSS" if names else "언론사 RSS")


def render_html(summary: dict, fetch_meta: dict) -> str:
    e = html.escape
    parts = [MARK, '<div class="ck">', f"<h3>{e(_head(fetch_meta))}</h3>"]
    items = summary.get("items") or []
    if summary.get("status") == "fail":
        parts.append(f"<div class='box warn'>{e(summary.get('note') or '뉴스를 가져오지 못함')}</div>")
    elif not items:
        parts.append("<div class='m'>구간 안에 고를 만한 주요 뉴스 없음</div>")
    else:
        parts.append("<ul>")
        for it in items:
            title = e(str(it.get("title", "")))
            url = str(it.get("url") or "")
            if url.startswith(("http://", "https://")):
                title = f"<a href='{e(url)}' style='color:#222'>{title}</a>"
            tick = " ".join(it.get("tickers") or [])
            parts.append(f"<li style='margin-bottom:6px'><span class='m'>{e(str(it.get('time', '')))}</span> "
                         f"<b>{title}</b>" + (f" <span class='m'>{e(tick)}</span>" if tick else "")
                         + f"<br><span style='font-size:13px'>{e(str(it.get('summary', '')))}</span>"
                         + (f" <span class='m' style='font-size:12px'>({e(str(it['source']))})</span>"
                            if it.get("source") else "") + "</li>")
        parts.append("</ul>")
    parts.append(f"<div class='m' style='font-size:12px;margin-top:4px'>{e(_sources(fetch_meta))}"
                 " · 고르기·요약은 자동 (원문 확인 권장)</div></div>")
    return "".join(parts)


def render_text(summary: dict, fetch_meta: dict) -> str:
    lines = [TEXT_MARK, _head(fetch_meta)]
    items = summary.get("items") or []
    if summary.get("status") == "fail":
        lines.append(f"- {summary.get('note') or '뉴스를 가져오지 못함'}")
    elif not items:
        lines.append("- 구간 안에 고를 만한 주요 뉴스 없음")
    for it in items:
        tick = " ".join(it.get("tickers") or [])
        lines.append(f"- {it.get('time', '')} {it.get('title', '')}" + (f" ({tick})" if tick else ""))
        lines.append(f"  {it.get('summary', '')}" + (f" ({it['source']})" if it.get("source") else ""))
    lines.append(_sources(fetch_meta))
    return "\n".join(lines)


def append(doc: str, section: str, sep: str = "", mark: str = MARK) -> str:
    """doc 맨 아래(</body> 가 있으면 그 앞)에 section 을 붙인다. 이미 붙어 있으면 바꾼다."""
    if mark in doc:
        i = doc.index(mark)
        j = doc.rfind("</body>")
        doc = doc[:i].rstrip("\n") + (doc[j:] if j > i else "")
    if "</body>" in doc:
        i = doc.rindex("</body>")
        return doc[:i] + section + doc[i:]
    return doc.rstrip("\n") + sep + section + "\n"
