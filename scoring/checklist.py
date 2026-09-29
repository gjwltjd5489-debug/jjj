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


# ---------------------------------------------------------------- 출력

COLUMNS = [("추세", ["20일선", "50일선", "200일선", "배열"]), ("일목", ["구름", "전환/기준", "후행스팬"]),
           ("MACD", ["시그널", "0선"]), ("RSI", ["RSI(14)"]), ("볼린저", ["%B"]), ("VR", ["VR(20)"]), ("OBV", ["OBV"])]


def _cell(rep: TickerReport, names: list[str]) -> str:
    return "".join(rep.check(n).status for n in names)


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


def render_markdown(reports: list[TickerReport]) -> str:
    date = max(r.date for r in reports).date()
    total = len(reports[0].checks)
    out = [f"# 미국장 지표 체크리스트 ({date} 종가 기준)", ""]
    out += ["## 한눈에 보기"] + [f"- {h}" for h in _headline(reports)] + [""]
    out.append("## 체크리스트")
    header = ["종목", "종가", "등락", "고점대비"] + [g for g, _ in COLUMNS] + ["✅/❌"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in reports:
        row = [f"{r.ticker}", f"{r.close:,.2f}", f"{r.change:+.1%}", f"{r.from_high:+.1%}"]
        row += [_cell(r, names) for _, names in COLUMNS]
        row.append(f"{r.ups}/{r.downs} (총 {total})")
        out.append("| " + " | ".join(row) + " |")
    out += ["", "## 오늘의 이벤트"]
    any_ev = False
    for r in reports:
        if r.events:
            any_ev = True
            out.append(f"- **{r.ticker}**: {', '.join(r.events)}")
    if not any_ev:
        out.append("- 없음")
    out += ["", "## 상세"]
    for r in reports:
        out.append(f"- **{r.ticker}** {r.name}: " + " · ".join(f"{c.name} {c.status} {c.detail}" for c in r.checks))
    out += ["", LEGEND]
    return "\n".join(out)


LEGEND = ("범례: ✅ 강세 / ❌ 약세 / ⚠️ 주의(과열·과매도·혼조·구름 안) / ➖ 데이터 없음. "
          "추세 = 종가>20·50·200일선, 정배열 · 일목 = 구름 위, 전환>기준, 후행스팬 · MACD = 시그널 위, 0선 위 · "
          "RSI 50~70 ✅, 30~50 ❌ · 볼린저 %B 0.5~1 ✅ · VR 100~450% ✅ · OBV > 20일 평균 ✅. 연구용 요약이며 매매 권유가 아님.")


def render_html(reports: list[TickerReport]) -> str:
    date = max(r.date for r in reports).date()
    total = len(reports[0].checks)
    e = html.escape
    td, tdl = 'class="c"', 'class="l"'
    parts = ["<style>.ck{font-family:-apple-system,Segoe UI,Malgun Gothic,sans-serif;font-size:14px;color:#222}"
             ".ck table{border-collapse:collapse}.ck th{padding:6px 8px;background:#f2f2f2;border-bottom:2px solid #999;"
             "white-space:nowrap}.ck td{padding:4px 8px;border-bottom:1px solid #ddd;white-space:nowrap}"
             ".ck td.c{text-align:center}.ck td.l{text-align:left}.ck td.d{white-space:normal;font-size:13px}"
             ".ck .g{font-weight:bold;color:#555;padding-top:8px}.ck .m{color:#777}</style>",
             '<div class="ck">',
             f"<h2 style='margin:0 0 8px'>미국장 지표 체크리스트 <span class='m' style='font-weight:normal'>({date} 종가)</span></h2>"]
    heads = _headline(reports)
    if heads:
        parts.append("<ul style='margin:4px 0 12px;padding-left:20px'>" + "".join(f"<li>{e(h)}</li>" for h in heads) + "</ul>")
    parts.append("<table><tr>")
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
        parts.append("<tr>")
        parts.append(f"<td {tdl}><b>{e(r.ticker)}</b> <span class='m'>{e(r.name)}</span></td>")
        parts.append(f"<td {td}>{r.close:,.2f}</td><td {td}><span style='color:{color}'>{r.change:+.1%}</span></td>")
        parts.append(f"<td {td}>{r.from_high:+.1%}</td>")
        for _, names in COLUMNS:
            parts.append(f"<td {td}>{_cell(r, names)}</td>")
        parts.append(f"<td {td}>{r.ups}/{r.downs} <span class='m'>({total})</span></td></tr>")
    parts.append("</table>")
    parts.append("<h3 style='margin:16px 0 4px'>오늘의 이벤트</h3>")
    evs = [r for r in reports if r.events]
    if evs:
        parts.append("<ul style='margin:0;padding-left:20px'>" +
                     "".join(f"<li><b>{e(r.ticker)}</b>: {e(', '.join(r.events))}</li>" for r in evs) + "</ul>")
    else:
        parts.append("<p style='margin:0'>없음</p>")
    parts.append("<h3 style='margin:16px 0 4px'>상세</h3><table>")
    for r in reports:
        detail = " · ".join(f"{c.name} {c.status} {c.detail}" for c in r.checks)
        parts.append(f"<tr><td {tdl}><b>{e(r.ticker)}</b></td><td class='d'>{e(detail)}</td></tr>")
    parts.append("</table>")
    parts.append(f"<p style='color:#777;font-size:12px;margin-top:16px'>{e(LEGEND)}</p></div>")
    return "".join(parts)
