from datetime import datetime

from scoring.news import ET, KST, MARK, TEXT_MARK, append, load_summary, render_html, render_text, window

META = {"status": "ok", "since": "2026-10-13T10:03:00-04:00", "until": "2026-10-14T06:05:00+09:00"}


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


def test_basis_label_shows_source_or_headline_only():
    s = {"items": [{"title": "a", "summary": "x", "basis": "원문", "source": "Reuters"},
                   {"title": "b", "summary": "y", "basis": "제목"}]}
    h = render_html(s, META)
    assert "(Reuters)" in h and "(원문 못 찾음 · 제목 기준)" in h
    assert "  y (원문 못 찾음 · 제목 기준)" in render_text(s, META)
