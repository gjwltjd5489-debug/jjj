"""메일 아래에 붙이는 주요 뉴스 (saveticker.com/news).

- 아침 메일: 직전 장초반 메일(뉴욕 10:03) 뒤부터 지금까지
- 장초반 메일: 직전 아침 메일(서울 06:00) 뒤부터 지금까지
- 뉴스 고르기·요약은 루틴(Claude)이 instructions.txt 를 따라 news.json 으로 쓰고,
  이 모듈이 그 파일을 메일 HTML·텍스트 맨 아래에 붙인다.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")
URL = "https://www.saveticker.com/news"
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


def fmt_kst(t: datetime) -> str:
    t = t.astimezone(KST)
    return f"{t.month}/{t.day} {t:%H:%M}"


def window_label(since: datetime, until: datetime) -> str:
    return f"{fmt_kst(since)} ~ {fmt_kst(until)} (한국시간)"


def instructions(kind: str, since: datetime, until: datetime, tickers: list[str], out_dir: str) -> str:
    """루틴이 읽고 따를 뉴스 고르기·원문 찾기·요약 지침."""
    n = MAX_ITEMS[kind]
    return f"""# 주요 뉴스 요약 지침 ({LABEL[kind]})

입력: {out_dir}/page.txt (saveticker.com/news 화면 글자), {out_dir}/links.json (화면의 링크 글자·주소),
      {out_dir}/api.json (페이지가 받은 JSON, 있으면)
가져온 시각: {fmt_kst(until)} 한국시간 = {until.astimezone(ET):%m/%d %H:%M} 뉴욕시간
구간: {since.astimezone(KST):%Y-%m-%d %H:%M} ~ {until.astimezone(KST):%Y-%m-%d %H:%M} 한국시간
      ({since.astimezone(ET):%m/%d %H:%M} ~ {until.astimezone(ET):%m/%d %H:%M} 뉴욕시간)

1. 구간 안에 올라온 뉴스만 쓴다. "3시간 전" 같은 상대 시각은 가져온 시각 기준으로 계산한다.
   시각을 알 수 없는 뉴스는 뺀다. api.json 에 발행 시각 필드가 있으면 그것을 우선한다.
2. 제목(과 화면에 보이는 짧은 본문)만 보고 미국 증시에 영향이 큰 것만 고른다 (중요한 순서, 최대 {n}개):
   - 연준·금리(FOMC, 연준 인사 발언, 국채금리 급변), 경제지표(CPI·고용·PCE·ISM·소매판매 등) 결과
   - 관세·무역·정부 정책·지정학(전쟁·제재·선거), 셧다운·부채한도
   - 대형주 실적·가이던스·대형 인수합병·규제, 반도체·AI 업황
   - 유가·달러·금·비트코인의 큰 움직임, 아시아·유럽 증시 급변
   - 관심 종목: {", ".join(tickers)}
   빼는 것: 단순 시세 요약, 소형주 목표가 조정, 광고·칼럼·의견, 같은 사건의 중복(하나로 합침).
3. 고른 뉴스마다 원문 보도를 찾는다 (이 환경은 언론사 사이트에 직접 접속할 수 없으므로 WebSearch 를 쓴다):
   - links.json 에 그 뉴스의 외부 원문 주소가 있으면 그 주소와 제목으로 검색한다.
   - 없으면 제목의 핵심어(회사·인물·숫자)를 영어로 바꿔 검색한다. mode 는 "standard", 맞는 결과가 없을 때만
     "extended" 로 한 번 더. 전체 검색은 {n + 4}번 안쪽으로 한다.
   - **같은 사건으로 인정하는 조건**: 주체(회사·기관·인물)와 내용이 제목과 같고, 보도 날짜가 구간 안(또는 하루 안쪽)이다.
     하나라도 어긋나면(예전 기사, 비슷한 다른 사건) 쓰지 않는다.
   - 인정되면 basis "원문", url 은 그 보도 주소. 인정되는 것이 없으면 basis "제목", url 은 links.json 의 saveticker 주소(없으면 빈 문자열)이고
     요약은 제목·화면 글자에 있는 내용만으로 쓴다.
4. 각 뉴스: 제목은 한국어 한 줄(40자 안쪽), 요약은 한국어 1~2문장으로 숫자(%, 금액, 예상치 대비)를 살려 쓴다.
   과장·추측을 더하지 말고 원문(또는 제목)에 있는 내용만 쓴다. 검색 결과끼리 숫자가 다르면 숫자를 빼고 쓴다.
5. {out_dir}/news.json 에 아래 형식으로 저장한다 (UTF-8, JSON 만):
{{"status": "ok", "items": [
  {{"time": "10/13 05:12", "title": "...", "summary": "...", "tickers": ["NVDA"], "basis": "원문", "source": "Reuters", "url": "https://..."}}
]}}
   - time 은 한국시간 "월/일 시:분" (saveticker 에 올라온 시각). source 는 원문 매체 이름(basis "제목"이면 빈 문자열).
   - 고를 뉴스가 없으면 {{"status": "none", "items": []}}.
   - page.txt 가 비었거나 로그인·차단 화면이라 뉴스 목록을 읽을 수 없으면 {{"status": "fail", "note": "이유 한 줄", "items": []}}.
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


def _basis(it: dict) -> str:
    """요약 근거: 원문 매체 이름, 원문을 못 찾았으면 '제목만'."""
    if it.get("basis") == "제목":
        return "원문 못 찾음 · 제목 기준"
    return str(it.get("source") or "원문")


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
                         + f" <span class='m' style='font-size:12px'>({e(_basis(it))})</span></li>")
        parts.append("</ul>")
    parts.append(f"<div class='m' style='font-size:12px;margin-top:4px'>출처: <a href='{URL}'>saveticker.com/news</a>"
                 " · 원문은 웹 검색으로 찾음 · 고르기·요약은 자동 (원문 확인 권장)</div></div>")
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
        lines.append(f"  {it.get('summary', '')} ({_basis(it)})")
    lines.append(f"출처: {URL}")
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
