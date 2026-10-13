from datetime import datetime

from scoring.news import (ET, KST, MARK, TEXT_MARK, append, in_window, load_summary, parse_feed, render_html,
                          render_text, window)

META = {"status": "ok", "since": "2026-10-13T10:03:00-04:00", "until": "2026-10-14T06:05:00+09:00",
        "feeds": {"CNBC": "ok 30건", "CNBC 경제": "ok 20건", "한국경제 증권": "실패 403"}}


def test_morning_window_starts_at_last_intraday_mail():
    since, until = window("morning", datetime(2026, 10, 14, 6, 5, tzinfo=KST))   # 수요일 아침 (뉴욕 화 17:05)
    assert since == datetime(2026, 10, 13, 10, 3, tzinfo=ET)
    sat = window("morning", datetime(2026, 10, 17, 6, 5, tzinfo=KST))[0]        # 토요일 아침 = 뉴욕 금 장 뒤
    assert sat == datetime(2026, 10, 16, 10, 3, tzinfo=ET)


def test_intraday_window_starts_at_last_morning_mail():
    since, _ = window("intraday", datetime(2026, 10, 13, 10, 3, tzinfo=ET))     # 화요일 장초반
    assert since == datetime(2026, 10, 13, 6, 0, tzinfo=KST)
    mon, _ = window("intraday", datetime(2026, 10, 12, 10, 3, tzinfo=ET))       # 월요일: 토 아침 메일 뒤 주말 전부
    assert mon == datetime(2026, 10, 10, 6, 0, tzinfo=KST)


def test_render_items_and_escape():
    s = {"status": "ok", "items": [{"time": "10/14 05:12", "title": "CPI <예상 상회>", "summary": "전년 대비 3.1%",
                                    "tickers": ["SPY"], "url": "https://example.com/a"}]}
    h = render_html(s, META)
    assert "CPI &lt;예상 상회&gt;" in h and "href='https://example.com/a'" in h and "10/13 23:03 ~ 10/14 06:05" in h
    t = render_text(s, META)
    assert "- 10/14 05:12 CPI <예상 상회> (SPY)" in t and "  전년 대비 3.1%" in t


def test_bad_url_is_not_linked():
    s = {"items": [{"title": "x", "url": "javascript:alert(1)"}]}
    assert "javascript" not in render_html(s, META)


def test_fetch_failure_or_missing_summary_becomes_note(tmp_path):
    s = load_summary(tmp_path / "news.json", {"status": "fail", "error": "403"})
    assert s["status"] == "fail" and "403" in s["note"]
    assert "403" in render_html(s, META)
    assert load_summary(tmp_path / "news.json", META)["status"] == "fail"
    (tmp_path / "news.json").write_text('{"items": []}', encoding="utf-8")
    assert load_summary(tmp_path / "news.json", META)["status"] == "none"


def test_append_is_idempotent():
    doc = "<style></style><div class='ck'>본문</div>"
    once = append(doc, render_html({"items": []}, META))
    twice = append(once, render_html({"items": []}, META))
    assert once == twice and once.startswith(doc) and once.count(MARK) == 1
    t = render_text({"items": []}, META)
    body = append(append("본문\n", t, "\n\n", TEXT_MARK), t, "\n\n", TEXT_MARK)
    assert body == "본문\n\n" + t + "\n" and "<!--" not in body
    wrapped = append("<html><body>x</body></html>", MARK + "N")
    assert append(wrapped, MARK + "M") == "<html><body>x" + MARK + "M</body></html>"



def test_source_and_feed_footer():
    s = {"items": [{"title": "a", "summary": "x", "source": "Reuters"}]}
    h = render_html(s, META)
    assert "(Reuters)" in h and "출처: CNBC RSS" in h and "한국경제" not in h
    assert "  x (Reuters)" in render_text(s, META)


RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Fed holds rates</title><link>https://ex.com/a</link><description>&lt;p&gt;Powell said &lt;b&gt;no rush&lt;/b&gt;.&lt;/p&gt;</description>
<pubDate>Tue, 13 Oct 2026 19:30:00 GMT</pubDate></item>
<item><title>Old story</title><link>https://ex.com/b</link><pubDate>Mon, 12 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>No date</title><link>https://ex.com/c</link></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>엔비디아 신고가</title><link rel="alternate" href="https://ex.kr/1"/><updated>2026-10-14T05:00:00+09:00</updated>
<summary>시간외 3% 상승</summary></entry>
<entry><title>Fed holds rates</title><link href="https://ex.com/a"/><published>2026-10-13T19:40:00Z</published></entry>
</feed>"""


def test_parse_rss_and_atom():
    r = parse_feed(RSS, "CNBC")
    assert [i["title"] for i in r] == ["Fed holds rates", "Old story", "No date"]
    assert r[0]["desc"] == "Powell said no rush ." and r[0]["url"] == "https://ex.com/a"
    assert r[0]["time"] == datetime(2026, 10, 13, 15, 30, tzinfo=ET) and r[2]["time"] is None
    a = parse_feed(ATOM.encode(), "한경")
    assert a[0]["url"] == "https://ex.kr/1" and a[0]["desc"] == "시간외 3% 상승"
    assert a[0]["time"] == datetime(2026, 10, 14, 5, 0, tzinfo=KST)


def test_in_window_filters_sorts_and_dedupes():
    since, until = datetime(2026, 10, 13, 10, 3, tzinfo=ET), datetime(2026, 10, 14, 6, 5, tzinfo=KST)
    got = in_window(parse_feed(RSS, "CNBC") + parse_feed(ATOM, "한경"), since, until)
    assert [(i["source"], i["title"]) for i in got] == [("한경", "엔비디아 신고가"), ("CNBC", "Fed holds rates")]
