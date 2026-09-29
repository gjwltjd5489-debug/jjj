"""일일 지표 체크리스트: 점수 대신 지표별 상태(✅ 강세 / ❌ 약세 / ⚠️ 주의)와 오늘 발생한 이벤트를 요약한다."""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiles import Profile
from .score import compute_indicators

UP, DOWN, WARN, NEUTRAL = "✅", "❌", "⚠️", "➖"


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str


@dataclass
class TickerReport:
    ticker: str
    name: str
    group: str
    date: pd.Timestamp
    close: float
    change: float
    from_high: float
    checks: list[Check]
    events: list[str] = field(default_factory=list)
    prev_checks: list[Check] | None = None   # 직전 거래일 체크리스트
    issues: list[str] = field(default_factory=list)  # 데이터 점검 경고

    @property
    def changes(self) -> list[tuple[Check, Check]]:
        """(직전, 오늘) 상태가 바뀐 항목."""
        if not self.prev_checks:
            return []
        prev = {c.name: c for c in self.prev_checks}
        return [(prev[c.name], c) for c in self.checks if c.name in prev and prev[c.name].status != c.status]

    @property
    def prev_ups(self) -> int | None:
        return None if self.prev_checks is None else sum(c.status == UP for c in self.prev_checks)

    @property
    def ups(self) -> int:
        return sum(c.status == UP for c in self.checks)

    @property
    def downs(self) -> int:
        return sum(c.status == DOWN for c in self.checks)

    def check(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)


def _crossed_up(a: pd.Series, b, k: int = -1) -> bool:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return bool(a.iloc[k] > b.iloc[k] and a.iloc[k - 1] <= b.iloc[k - 1])


def _crossed_down(a: pd.Series, b, k: int = -1) -> bool:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return bool(a.iloc[k] < b.iloc[k] and a.iloc[k - 1] >= b.iloc[k - 1])


def build_report(df: pd.DataFrame, p: Profile, ticker: str, name: str = "", group: str = "") -> TickerReport:
    x = compute_indicators(df, p)
    close = x["close"]
    mid = close.rolling(p.bb_period).mean()
    sd = close.rolling(p.bb_period).std(ddof=0)
    bandwidth = 4 * sd / mid
    r = x.iloc[-1]
    c = r["close"]
    s, m, lng = f"{p.ma_short}일", f"{p.ma_mid}일", f"{p.ma_long}일"
    checks: list[Check] = []

    def add(group_, name_, cond, up_text, down_text):
        if pd.isna(cond):
            checks.append(Check(group_, name_, NEUTRAL, "데이터 부족"))
        else:
            checks.append(Check(group_, name_, UP if cond else DOWN, up_text if cond else down_text))

    # 이동평균
    for label, col in ((s, "ma_short"), (m, "ma_mid"), (lng, "ma_long")):
        gap = c / r[col] - 1 if r[col] else np.nan
        add("이동평균", f"{label}선", (c > r[col]) if pd.notna(r[col]) else np.nan,
            f"위 ({gap:+.1%})", f"아래 ({gap:+.1%})")
    ms, mm, ml = r["ma_short"], r["ma_mid"], r["ma_long"]
    if ms > mm > ml:
        checks.append(Check("이동평균", "배열", UP, "정배열"))
    elif ms < mm < ml:
        checks.append(Check("이동평균", "배열", DOWN, "역배열"))
    else:
        checks.append(Check("이동평균", "배열", WARN, "혼조"))

    # 일목균형표
    top, bottom = max(r["span_a"], r["span_b"]), min(r["span_a"], r["span_b"])
    if c > top:
        checks.append(Check("일목", "구름", UP, "구름 위"))
    elif c < bottom:
        checks.append(Check("일목", "구름", DOWN, "구름 아래"))
    else:
        checks.append(Check("일목", "구름", WARN, "구름 안"))
    add("일목", "전환/기준", r["tenkan"] > r["kijun"], "전환선 > 기준선", "전환선 < 기준선")
    add("일목", "후행스팬", c > r["chikou_ref"], "26일 전 종가 위", "26일 전 종가 아래")

    # MACD
    add("MACD", "시그널", r["macd"] > r["signal"], "MACD > 시그널", "MACD < 시그널")
    add("MACD", "0선", r["macd"] > 0, "0선 위", "0선 아래")

    # RSI
    rsi = r["rsi"]
    if rsi >= 70:
        checks.append(Check("RSI", "RSI(14)", WARN, f"{rsi:.0f} 과매수"))
    elif rsi <= 30:
        checks.append(Check("RSI", "RSI(14)", WARN, f"{rsi:.0f} 과매도"))
    else:
        checks.append(Check("RSI", "RSI(14)", UP if rsi >= 50 else DOWN, f"{rsi:.0f}"))

    # 볼린저밴드
    pb = r["pct_b"]
    if pb > 1:
        checks.append(Check("볼린저", "%B", WARN, f"{pb:.2f} 상단 돌파"))
    elif pb < 0:
        checks.append(Check("볼린저", "%B", WARN, f"{pb:.2f} 하단 이탈"))
    else:
        checks.append(Check("볼린저", "%B", UP if pb >= 0.5 else DOWN,
                            f"{pb:.2f} 중심선 {'위' if pb >= 0.5 else '아래'}"))

    # 거래량
    vr = r["vr"]
    if pd.isna(vr):
        checks.append(Check("거래량", "VR(20)", NEUTRAL, "거래량 없음"))
    elif vr >= 450:
        checks.append(Check("거래량", "VR(20)", WARN, f"{vr:.0f}% 과열"))
    elif vr <= 70:
        checks.append(Check("거래량", "VR(20)", WARN, f"{vr:.0f}% 침체"))
    else:
        checks.append(Check("거래량", "VR(20)", UP if vr >= 100 else DOWN,
                            f"{vr:.0f}% {'매수세 우위' if vr >= 100 else '매도세 우위'}"))
    add("거래량", "OBV", r["obv"] > r["obv_ma"] if pd.notna(r["obv_ma"]) else np.nan,
        "OBV > 20일 평균", "OBV < 20일 평균")

    # 오늘 이벤트
    ev: list[str] = []
    if _crossed_up(x["ma_mid"], x["ma_long"]) or any(_crossed_up(x["ma_mid"], x["ma_long"], k) for k in range(-5, -1)):
        ev.append(f"골든크로스 ({m}선 > {lng}선, 최근 5일)")
    if _crossed_down(x["ma_mid"], x["ma_long"]) or any(_crossed_down(x["ma_mid"], x["ma_long"], k) for k in range(-5, -1)):
        ev.append(f"데드크로스 ({m}선 < {lng}선, 최근 5일)")
    for label, col in ((s, "ma_short"), (lng, "ma_long")):
        if _crossed_up(close, x[col]):
            ev.append(f"{label}선 상향 돌파")
        if _crossed_down(close, x[col]):
            ev.append(f"{label}선 하향 이탈")
    if _crossed_up(x["macd"], x["signal"]):
        ev.append("MACD 시그널 상향 교차")
    if _crossed_down(x["macd"], x["signal"]):
        ev.append("MACD 시그널 하향 교차")
    cloud_top = x[["span_a", "span_b"]].max(axis=1)
    cloud_bottom = x[["span_a", "span_b"]].min(axis=1)
    if _crossed_up(close, cloud_top):
        ev.append("일목 구름 상향 돌파")
    if _crossed_down(close, cloud_bottom):
        ev.append("일목 구름 하향 이탈")
    if _crossed_up(x["tenkan"], x["kijun"]):
        ev.append("전환선 > 기준선 교차 (호전)")
    if _crossed_down(x["tenkan"], x["kijun"]):
        ev.append("전환선 < 기준선 교차 (역전)")
    if _crossed_up(x["rsi"], 70):
        ev.append("RSI 70 진입 (과매수)")
    if _crossed_down(x["rsi"], 30):
        ev.append("RSI 30 진입 (과매도)")
    if _crossed_up(x["pct_b"], 1):
        ev.append("볼린저 상단 돌파")
    if _crossed_down(x["pct_b"], 0):
        ev.append("볼린저 하단 이탈")
    bw = bandwidth.dropna()
    if len(bw) > 126 and bw.iloc[-1] <= bw.iloc[-126:].min():
        ev.append("볼린저 밴드폭 6개월 최저 (스퀴즈, 큰 움직임 전조)")
    hi252 = close.rolling(252, min_periods=60).max()
    if c >= hi252.iloc[-1]:
        ev.append("52주 신고가")

    prev = close.iloc[-2] if len(close) > 1 else np.nan
    return TickerReport(
        ticker=ticker, name=name, group=group, date=x.index[-1], close=float(c),
        change=float(c / prev - 1) if prev else np.nan,
        from_high=float(c / hi252.iloc[-1] - 1) if pd.notna(hi252.iloc[-1]) else np.nan,
        checks=checks, events=ev,
    )


# ---------------------------------------------------------------- 직전 거래일 비교 + 데이터 점검

MAX_DAILY_MOVE = 0.25  # 이보다 큰 하루 등락은 분할·데이터 오류 가능성으로 표시


def data_issues(df: pd.DataFrame, rep: TickerReport, expected=None) -> list[str]:
    issues = []
    last = df.index[-1].date()
    if expected is not None and last < expected:
        issues.append(f"기준일 {last} (기대 {expected}) — 데이터 지연")
    if pd.notna(rep.change) and abs(rep.change) > MAX_DAILY_MOVE:
        issues.append(f"하루 {rep.change:+.1%} 급변 — 분할·데이터 오류 가능, 원자료 확인 필요")
    vol = df["Volume"].iloc[-1]
    if df["Volume"].notna().any() and (pd.isna(vol) or vol <= 0):
        issues.append("당일 거래량 없음 — VR·OBV 신뢰도 낮음")
    row = df.iloc[-1]
    if pd.notna(row["High"]) and pd.notna(row["Low"]) and (
            row["High"] < row["Low"] or not (row["Low"] * 0.999 <= row["Close"] <= row["High"] * 1.001)):
        issues.append("고가·저가·종가 불일치")
    missing = [c.name for c in rep.checks if c.status == NEUTRAL and c.name != "VR(20)"]
    if missing:
        issues.append(f"계산 안 된 지표: {', '.join(missing)}")
    return issues


def build_with_history(df: pd.DataFrame, p: Profile, ticker: str, name: str = "", group: str = "",
                       expected=None) -> TickerReport:
    """오늘 체크리스트 + 직전 거래일 체크리스트(비교용) + 데이터 점검."""
    rep = build_report(df, p, ticker, name, group)
    if len(df) > 2:
        rep.prev_checks = build_report(df.iloc[:-1], p, ticker, name, group).checks
    rep.issues = data_issues(df, rep, expected)
    return rep


# ---------------------------------------------------------------- 출력

COLUMNS = [("추세", ["20일선", "50일선", "200일선", "배열"]), ("일목", ["구름", "전환/기준", "후행스팬"]),
           ("MACD", ["시그널", "0선"]), ("RSI", ["RSI(14)"]), ("볼린저", ["%B"]), ("VR", ["VR(20)"]), ("OBV", ["OBV"])]

LEGEND = ("범례: ✅ 강세 / ❌ 약세 / ⚠️ 주의(과열·과매도·혼조·구름 안) / ➖ 데이터 없음. 노란 칸(마크다운은 [ ])은 직전 거래일과 달라진 항목. "
          "추세 = 종가>20·50·200일선, 정배열 · 일목 = 구름 위, 전환>기준, 후행스팬 · MACD = 시그널 위, 0선 위 · "
          "RSI 50~70 ✅, 30~50 ❌ · 볼린저 %B 0.5~1 ✅ · VR 100~450% ✅ · OBV > 20일 평균 ✅. 연구용 요약이며 매매 권유가 아님.")


def _changed_names(rep: TickerReport) -> set[str]:
    return {now.name for _, now in rep.changes}


def _headline(reports: list[TickerReport]) -> list[str]:
    lines = []
    strong = [r.ticker for r in reports if r.downs == 0 and r.ups >= 10]
    weak = [r.ticker for r in reports if r.ups <= 3]
    overheat = [r.ticker for r in reports if r.check("RSI(14)").detail.endswith("과매수") or r.check("%B").detail.endswith("상단 돌파")]
    oversold = [r.ticker for r in reports if r.check("RSI(14)").detail.endswith("과매도") or r.check("%B").detail.endswith("하단 이탈")]
    below200 = [r.ticker for r in reports if r.check("200일선").status == DOWN]
    if strong:
        lines.append(f"강세 정렬 (❌ 없음): {', '.join(strong)}")
    if weak:
        lines.append(f"약세 (✅ 3개 이하): {', '.join(weak)}")
    if overheat:
        lines.append(f"과열 주의 (RSI≥70 또는 볼린저 상단 돌파): {', '.join(overheat)}")
    if oversold:
        lines.append(f"과매도 (RSI≤30 또는 볼린저 하단 이탈): {', '.join(oversold)}")
    if below200:
        lines.append(f"200일선 아래: {', '.join(below200)}")
    return lines


def change_lines(reports: list[TickerReport]) -> tuple[str, list[str]]:
    """(요약 한 줄, 종목별 변화 줄). ✅ 개수 변화가 큰 순."""
    rows = [r for r in reports if r.changes]
    if not any(r.prev_checks for r in reports):
        return "직전 거래일 비교 불가", []
    better = [r for r in rows if r.prev_ups is not None and r.ups > r.prev_ups]
    worse = [r for r in rows if r.prev_ups is not None and r.ups < r.prev_ups]
    n = sum(len(r.changes) for r in rows)
    summary = f"변화 {n}건 · 개선 {len(better)}종목 · 악화 {len(worse)}종목"
    lines = []
    for r in sorted(rows, key=lambda r: -abs(r.ups - (r.prev_ups or 0))):
        diff = f"✅ {r.prev_ups}→{r.ups}"
        items = ", ".join(f"{now.name} {prev.status}→{now.status}" for prev, now in r.changes)
        lines.append(f"{r.ticker} ({diff}): {items}")
    return summary, lines


def quality_lines(reports: list[TickerReport], failed: list[str]) -> list[str]:
    lines = [f"{r.ticker}: {i}" for r in reports for i in r.issues]
    lines += [f"{f} — 불러오기 실패" for f in failed]
    return lines


def _md_cell(rep: TickerReport, names: list[str]) -> str:
    changed = _changed_names(rep)
    return "".join(f"[{rep.check(n).status}]" if n in changed else rep.check(n).status for n in names)


def make_subject(reports: list[TickerReport], meta: dict) -> str:
    date = max(r.date for r in reports).date()
    q = quality_lines(reports, meta.get("failed", []))
    if meta.get("mode") == "holiday":
        return f"[휴장] 미장 체크리스트 · {meta['checked']} {meta['holiday']} · 다음 개장 {meta['next_open']}"
    if meta.get("mode") == "delayed":
        return f"[데이터 지연] 미장 체크리스트 · 기대 {meta['target']}, 수신 {date}"
    summary, _ = change_lines(reports)
    heads = _headline(reports)
    lead = heads[0] if heads else ""
    subject = f"[미장 체크리스트] {date} · {summary.split(' · ')[0]} · {lead}"
    if q:
        subject = "[점검 필요] " + subject
    return subject if len(subject) <= 100 else subject[:99] + "…"


def render_markdown(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    date = max(r.date for r in reports).date()
    total = len(reports[0].checks)
    q = quality_lines(reports, meta.get("failed", []))
    heads = _headline(reports)
    if meta.get("mode") == "holiday":
        out = [f"# 미국장 휴장 안내 · {meta['checked']} {meta['holiday']}", "",
               f"다음 거래일: {meta['next_open']}. 새 종가가 없어 체크리스트는 직전 거래일({date}) 기준이며 변화는 없습니다.", ""]
        out += ["## 데이터 점검"] + ([f"- ⚠️ {x}" for x in q] or ["- 이상 없음"]) + [""]
        out += [f"## 직전 거래일({date}) 요약"] + [f"- {h}" for h in heads] + [""]
        ev = [f"- **{r.ticker}**: {', '.join(r.events)}" for r in reports if r.events]
        out += ["## 직전 거래일 이벤트"] + (ev or ["- 없음"])
        return "\n".join(out)

    out = [f"# 미국장 지표 체크리스트 ({date} 종가 기준)", ""]
    if meta.get("mode") == "delayed":
        out += [f"> ⚠️ 데이터 지연: 기대 기준일 {meta['target']}, 받은 데이터 {date}. 아래는 받은 데이터 기준입니다.", ""]
    out += ["## 데이터 점검"] + ([f"- ⚠️ {x}" for x in q] or [f"- 이상 없음 ({len(reports)}종목, 기준일 {date})"]) + [""]
    out += ["## 한눈에 보기"] + [f"- {h}" for h in heads] + [""]
    summary, lines = change_lines(reports)
    out += [f"## 전일 대비 변화 — {summary}"] + ([f"- {x}" for x in lines] or ["- 없음"]) + [""]
    out.append("## 체크리스트")
    header = ["종목", "종가", "등락", "고점대비"] + [g for g, _ in COLUMNS] + ["✅/❌"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in reports:
        row = [f"{r.ticker}" + (" ⚠️" if r.issues else ""), f"{r.close:,.2f}", f"{r.change:+.1%}", f"{r.from_high:+.1%}"]
        row += [_md_cell(r, names) for _, names in COLUMNS]
        prev = f" (전일 {r.prev_ups})" if r.prev_ups is not None and r.prev_ups != r.ups else ""
        row.append(f"{r.ups}/{r.downs}{prev} (총 {total})")
        out.append("| " + " | ".join(row) + " |")
    out += ["", "## 오늘의 이벤트"]
    ev = [f"- **{r.ticker}**: {', '.join(r.events)}" for r in reports if r.events]
    out += ev or ["- 없음"]
    out += ["", "## 상세"]
    for r in reports:
        out.append(f"- **{r.ticker}** {r.name}: " + " · ".join(f"{c.name} {c.status} {c.detail}" for c in r.checks))
    out += ["", LEGEND]
    return "\n".join(out)


def render_text(reports: list[TickerReport], meta: dict | None = None) -> str:
    """메일 일반 텍스트 본문 (마크다운 기호 없음)."""
    meta = meta or {}
    date = max(r.date for r in reports).date()
    q = quality_lines(reports, meta.get("failed", []))
    lines = []
    if meta.get("mode") == "holiday":
        lines += [f"미국장 휴장 안내: {meta['checked']} {meta['holiday']}", f"다음 거래일: {meta['next_open']}",
                  f"아래는 직전 거래일({date}) 요약입니다.", ""]
    else:
        lines += [f"미국장 지표 체크리스트 ({date} 종가 기준)", ""]
        if meta.get("mode") == "delayed":
            lines += [f"주의: 데이터 지연 (기대 {meta['target']}, 받은 데이터 {date})", ""]
    lines += ["데이터 점검"] + ([f"- {x}" for x in q] or ["- 이상 없음"]) + [""]
    lines += ["한눈에 보기"] + [f"- {h}" for h in _headline(reports)] + [""]
    if meta.get("mode") != "holiday":
        summary, cl = change_lines(reports)
        lines += [f"전일 대비 변화: {summary}"] + [f"- {x}" for x in cl] + [""]
    ev = [f"- {r.ticker}: {', '.join(r.events)}" for r in reports if r.events]
    lines += ["이벤트"] + (ev or ["- 없음"]) + ["", "표와 상세는 HTML 메일에서 볼 수 있습니다. 연구용 요약이며 매매 권유가 아닙니다."]
    return "\n".join(lines)


_CSS = ("<style>.ck{font-family:-apple-system,Segoe UI,Malgun Gothic,sans-serif;font-size:14px;color:#222}"
        ".ck table{border-collapse:collapse}.ck th{padding:6px 8px;background:#f2f2f2;border-bottom:2px solid #999;"
        "white-space:nowrap}.ck td{padding:4px 8px;border-bottom:1px solid #ddd;white-space:nowrap}"
        ".ck td.c{text-align:center}.ck td.l{text-align:left}.ck td.d{white-space:normal;font-size:13px}"
        ".ck .g{font-weight:bold;color:#555;padding-top:8px}.ck .m{color:#777}"
        ".ck .x{background:#ffe58a;border-radius:3px;padding:0 1px}"
        ".ck .box{border-left:4px solid #999;background:#fafafa;padding:6px 10px;margin:8px 0}"
        ".ck .warn{border-left-color:#e65100;background:#fff3e0}.ck .ok{border-left-color:#2e7d32;background:#f1f8e9}"
        ".ck h3{margin:14px 0 4px}.ck ul{margin:0;padding-left:20px}</style>")


def _html_cell(rep: TickerReport, names: list[str]) -> str:
    changed = {now.name: prev.status for prev, now in rep.changes}
    out = []
    for n in names:
        st = rep.check(n).status
        if n in changed:
            out.append(f"<span class='x' title='전일 {changed[n]}'>{st}</span>")
        else:
            out.append(st)
    return "".join(out)


def _ul(items: list[str], e) -> str:
    return "<ul>" + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul>"


def render_html(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    e = html.escape
    date = max(r.date for r in reports).date()
    total = len(reports[0].checks)
    q = quality_lines(reports, meta.get("failed", []))
    heads = _headline(reports)
    qbox = (f"<div class='box warn'><b>데이터 점검 ⚠️</b>{_ul(q, e)}</div>" if q
            else f"<div class='box ok'><b>데이터 점검</b>: 이상 없음 ({len(reports)}종목, 기준일 {date})</div>")
    parts = [_CSS, '<div class="ck">']

    if meta.get("mode") == "holiday":
        parts.append(f"<h2 style='margin:0 0 8px'>🇺🇸 미국장 휴장 · {e(str(meta['checked']))} {e(meta['holiday'])}</h2>")
        parts.append(f"<p>다음 거래일: <b>{e(str(meta['next_open']))}</b>. 새 종가가 없어 체크리스트는 "
                     f"직전 거래일({date}) 기준이며 변화는 없습니다.</p>")
        parts.append(qbox)
        parts.append(f"<h3>직전 거래일({date}) 요약</h3>" + _ul(heads, e))
        evs = [r for r in reports if r.events]
        parts.append("<h3>직전 거래일 이벤트</h3>" + (
            "<ul>" + "".join(f"<li><b>{e(r.ticker)}</b>: {e(', '.join(r.events))}</li>" for r in evs) + "</ul>"
            if evs else "<p>없음</p>"))
        parts.append(f"<p class='m' style='font-size:12px'>{e(LEGEND)}</p></div>")
        return "".join(parts)

    parts.append(f"<h2 style='margin:0 0 8px'>미국장 지표 체크리스트 <span class='m' style='font-weight:normal'>({date} 종가)</span></h2>")
    if meta.get("mode") == "delayed":
        parts.append(f"<div class='box warn'><b>데이터 지연</b>: 기대 기준일 {e(str(meta['target']))}, "
                     f"받은 데이터 {date}. 아래는 받은 데이터 기준입니다.</div>")
    parts.append(qbox)
    if heads:
        parts.append("<h3>한눈에 보기</h3>" + _ul(heads, e))
    summary, cl = change_lines(reports)
    parts.append(f"<h3>전일 대비 변화 <span class='m' style='font-weight:normal'>— {e(summary)}</span></h3>")
    parts.append(_ul(cl, e) if cl else "<p>없음</p>")

    parts.append("<h3>체크리스트 <span class='m' style='font-weight:normal;font-size:12px'>노란 칸 = 전일과 달라진 항목</span></h3><table><tr>")
    for h in ["종목", "종가", "등락", "고점대비"] + [g for g, _ in COLUMNS] + ["✅/❌"]:
        parts.append(f"<th>{e(h)}</th>")
    parts.append("</tr>")
    group = None
    ncol = 5 + len(COLUMNS)
    for r in reports:
        if r.group and r.group != group:
            group = r.group
            parts.append(f"<tr><td colspan='{ncol}' class='g'>{e(group)}</td></tr>")
        color = "#c62828" if r.change > 0 else "#1565c0" if r.change < 0 else "#222"
        flag = " ⚠️" if r.issues else ""
        parts.append(f"<tr><td class='l'><b>{e(r.ticker)}</b>{flag} <span class='m'>{e(r.name)}</span></td>")
        parts.append(f"<td class='c'>{r.close:,.2f}</td><td class='c'><span style='color:{color}'>{r.change:+.1%}</span></td>")
        parts.append(f"<td class='c'>{r.from_high:+.1%}</td>")
        for _, names in COLUMNS:
            parts.append(f"<td class='c'>{_html_cell(r, names)}</td>")
        delta = ""
        if r.prev_ups is not None and r.prev_ups != r.ups:
            arrow = "▲" if r.ups > r.prev_ups else "▼"
            delta = f" <span class='x'>{arrow}{abs(r.ups - r.prev_ups)}</span>"
        parts.append(f"<td class='c'>{r.ups}/{r.downs}{delta} <span class='m'>({total})</span></td></tr>")
    parts.append("</table>")

    evs = [r for r in reports if r.events]
    parts.append("<h3>오늘의 이벤트</h3>" + (
        "<ul>" + "".join(f"<li><b>{e(r.ticker)}</b>: {e(', '.join(r.events))}</li>" for r in evs) + "</ul>"
        if evs else "<p>없음</p>"))
    parts.append("<h3>상세</h3><table>")
    for r in reports:
        detail = " · ".join(f"{c.name} {c.status} {c.detail}" for c in r.checks)
        parts.append(f"<tr><td class='l'><b>{e(r.ticker)}</b></td><td class='d'>{e(detail)}</td></tr>")
    parts.append(f"</table><p class='m' style='font-size:12px;margin-top:16px'>{e(LEGEND)}</p></div>")
    return "".join(parts)
