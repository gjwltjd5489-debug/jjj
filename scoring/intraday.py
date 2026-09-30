"""장 초반 확인: 아침 신호(전일 종가 기준)를 장중 가격으로 다시 점검한다.

- 아침 신호: 오늘 막대를 뺀 데이터로 계산 (아침 메일과 같은 기준)
- 잠정 신호: 지금 가격을 오늘 종가로 가정해 다시 계산. 장 초반 거래량은 하루치와 비교가 안 되므로
  오늘 거래량은 직전 20일 중앙값으로 가정한다.
- 규칙 검증은 종가 기준이므로 잠정 신호는 참고용이며 종가까지 바뀔 수 있다.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .checklist import _BADGE_CSS, _CSS
from .checklist_score import RULE, checklist_score
from .profiles import Profile

GAP_MIN = 0.02      # 시가 갭 경고 최소폭
STOP_NEAR = 0.01    # 손절 참고가 위 1% 이내면 '근접'


@dataclass
class IntradayRow:
    ticker: str
    name: str
    group: str
    has_intraday: bool
    prev_close: float = np.nan
    open: float = np.nan
    price: float = np.nan
    gap: float = np.nan
    move: float = np.nan
    atr_pct: float = np.nan
    m: dict = field(default_factory=dict)   # 아침(전일 종가 기준) 상태
    p: dict = field(default_factory=dict)   # 잠정(현재가 기준) 상태
    notes: list[tuple[str, str]] = field(default_factory=list)  # (종류, 설명)


def _last(o: pd.DataFrame) -> dict:
    r = o.iloc[-1]
    return {k: (r[k].item() if hasattr(r[k], "item") else r[k])
            for k in ("score", "score_s", "state", "event", "blocked", "stop", "above200", "overheat", "highvol", "close")}


def provisional_frame(df: pd.DataFrame) -> pd.DataFrame:
    """오늘 거래량을 직전 20일 중앙값으로 바꾼 사본 (장 초반 거래량 왜곡 방지)."""
    out = df.copy()
    med = df["Volume"].iloc[:-1].tail(20).median()
    if pd.notna(med):
        out.iloc[-1, out.columns.get_loc("Volume")] = med
    return out


def check_ticker(df: pd.DataFrame, p: Profile, ticker: str, name: str, group: str, today,
                 bench: pd.Series | None = None) -> IntradayRow:
    has = df.index[-1].date() == today
    if not has:
        return IntradayRow(ticker, name, group, False, m=_last(checklist_score(df, p, bench)))
    prev = df.iloc[:-1]
    bench_prev = bench.loc[: prev.index[-1]] if bench is not None else None
    m = _last(checklist_score(prev, p, bench_prev))
    pv = _last(checklist_score(provisional_frame(df), p, bench))
    bar = df.iloc[-1]
    pc = float(prev["Close"].iloc[-1])
    atr_pct = (pc - m["stop"]) / RULE.stop_atr / pc if pd.notna(m["stop"]) else np.nan
    row = IntradayRow(ticker, name, group, True, pc, float(bar["Open"]), float(bar["Close"]),
                      float(bar["Open"]) / pc - 1, float(bar["Close"]) / pc - 1, atr_pct, m, pv)
    row.notes = classify(row)
    return row


def _reasons(p: dict) -> list[str]:
    out = []
    if pd.notna(p.get("score_s")) and p["score_s"] < RULE.entry:
        out.append(f"잠정 3일 평균 {p['score_s']:.0f} < {RULE.entry:g}")
    if not p.get("above200", True):
        out.append("200일선 아래")
    if p.get("overheat"):
        out.append("과열")
    if p.get("highvol"):
        out.append("고변동")
    return out


def classify(r: IntradayRow) -> list[tuple[str, str]]:
    m, p = r.m, r.p
    notes = []
    now = f"현재가 {r.price:,.2f} ({r.move:+.1%})"
    ps = f"잠정 3일 평균 {p['score_s']:.0f}" if pd.notna(p.get("score_s")) else "잠정 점수 없음"
    if m.get("event") == "BUY":
        if p.get("event") == "SELL":
            notes.append(("buy_cancel", f"매수 신호 취소 수준 — {ps} ≤ {RULE.exit:g} · {now}"))
        elif _reasons(p):
            notes.append(("buy_weak", f"매수 신호 약화 — {', '.join(_reasons(p))} · {now}"))
        else:
            notes.append(("buy_keep", f"매수 신호 유지 — {ps} (아침 {m['score_s']:.0f}) · {now}"))
    elif m.get("event") == "SELL":
        # 전일 종가에 난 매도 신호는 재진입(70 이상) 전까지 유효. 평소 하루 변동폭보다 큰 장중 반등만 따로 알린다.
        bounce = r.move if pd.notna(r.move) else 0.0
        if pd.notna(r.atr_pct) and bounce >= r.atr_pct:
            notes.append(("sell_weak", f"매도 신호 — 장중 반등 {bounce:+.1%}가 평소 하루 변동폭 {r.atr_pct:.1%}보다 큼, "
                                       f"매도 전 확인 · {ps} · {now}"))
        else:
            notes.append(("sell_keep", f"매도 신호 유지 — {ps} (재진입은 {RULE.entry:g} 이상) · {now}"))
    elif p.get("event") == "BUY":
        notes.append(("new_buy", f"현재가가 종가까지 유지되면 매수 신호 — {ps} ≥ {RULE.entry:g} · {now}"))
    elif p.get("event") == "SELL":
        notes.append(("new_sell", f"현재가가 종가까지 유지되면 매도 신호 — {ps} ≤ {RULE.exit:g} · {now}"))
    if m.get("state") == 1 and m.get("event") != "SELL" and pd.notna(m.get("stop")):
        if r.price < m["stop"]:
            notes.append(("stop_break", f"손절 참고가 {m['stop']:,.2f} 이탈 · {now}"))
        elif r.price < m["stop"] * (1 + STOP_NEAR):
            notes.append(("stop_near", f"손절 참고가 {m['stop']:,.2f} 근접 ({r.price / m['stop'] - 1:+.1%}) · {now}"))
    if pd.notna(r.gap) and abs(r.gap) >= max(GAP_MIN, r.atr_pct if pd.notna(r.atr_pct) else 0):
        notes.append(("gap", f"시가 갭 {r.gap:+.1%} (평소 하루 변동폭 {r.atr_pct:.1%}) · {now}"))
    return notes


# ---------------------------------------------------------------- 출력

SECTIONS = [
    ("아침 신호 재확인", ["buy_keep", "buy_weak", "buy_cancel", "sell_keep", "sell_weak"]),
    ("장중 신규 후보 (종가 확정 전)", ["new_buy", "new_sell"]),
    ("손절 참고가 점검 (규칙상 보유 구간)", ["stop_break", "stop_near"]),
    ("큰 시가 갭", ["gap"]),
]
BADGE = {
    "buy_keep": ("bg", "🟢 매수 유지"), "buy_weak": ("bo", "🟠 매수 약화"), "buy_cancel": ("br", "🔴 매수 취소"),
    "sell_keep": ("br", "🔴 매도 유지"), "sell_weak": ("bo", "🟠 매도 약화"),
    "new_buy": ("bg", "🟢 신규 매수 후보"), "new_sell": ("br", "🔴 신규 매도 후보"),
    "stop_break": ("br", "🔴 손절선 이탈"), "stop_near": ("bo", "🟠 손절선 근접"), "gap": ("bn", "⚪ 갭"),
}
NOTE = ("검증된 규칙은 종가 기준이다. 잠정 점수는 현재가를 오늘 종가로 가정한 값이라 종가까지 바뀔 수 있다. "
        "장 초반 거래량은 하루치와 비교할 수 없어 20일 중앙값으로 가정했다. 가격은 수 분 지연될 수 있다. 매매 권유가 아님.")


def _items(rows: list[IntradayRow], kinds: list[str]) -> list[tuple[IntradayRow, str, str]]:
    return [(r, k, t) for r in rows for k, t in r.notes if k in kinds]


def make_subject(rows: list[IntradayRow], meta: dict) -> str:
    if meta["mode"] == "holiday":
        return f"[장초반 확인] 휴장 · {meta['date']} {meta['holiday']}"
    if meta["mode"] == "no_data":
        return f"[장초반 확인] {meta['date']} · 장중 데이터 없음"
    counts = {}
    for r in rows:
        for k, _ in r.notes:
            counts[k] = counts.get(k, 0) + 1
    order = [("buy_keep", "매수 유지"), ("buy_weak", "매수 약화"), ("buy_cancel", "매수 취소"), ("sell_keep", "매도 유지"),
             ("sell_weak", "매도 약화"), ("new_buy", "신규 매수 후보"), ("new_sell", "신규 매도 후보"),
             ("stop_break", "손절선 이탈"), ("stop_near", "손절선 근접")]
    parts = []
    for k, label in order:
        names = [r.ticker for r in rows for kk, _ in r.notes if kk == k]
        if names:
            parts.append(f"{label} {','.join(names)}")
    return f"[장초반 확인] {meta['date']} {meta['time']} ET · " + (" · ".join(parts) or "아침 신호 없음·특이 사항 없음")


def render_text(rows: list[IntradayRow], meta: dict) -> str:
    lines = [f"장 초반 확인 ({meta['date']} {meta['time']} 뉴욕 시각, 개장 후 약 {meta['minutes']}분)", ""]
    if meta["mode"] != "normal":
        lines.append("휴장일입니다." if meta["mode"] == "holiday" else "장중 데이터가 아직 없습니다 (Yahoo 지연 또는 개장 전).")
        return "\n".join(lines)
    for title, kinds in SECTIONS:
        items = _items(rows, kinds)
        lines.append(title)
        lines += [f"- {BADGE[k][1]} {r.ticker}: {t}" for r, k, t in items] or ["- 없음"]
        lines.append("")
    lines.append(NOTE)
    return "\n".join(lines)


def render_html(rows: list[IntradayRow], meta: dict) -> str:
    e = html.escape
    parts = [_CSS.replace("</style>", _BADGE_CSS + "</style>"), '<div class="ck">',
             f"<h2 style='margin:0 0 8px'>장 초반 확인 <span class='m' style='font-weight:normal'>"
             f"({e(meta['date'])} {e(meta['time'])} ET · 개장 후 약 {meta['minutes']}분)</span></h2>"]
    if meta["mode"] != "normal":
        msg = f"휴장일입니다 ({e(meta.get('holiday') or '')})." if meta["mode"] == "holiday" else \
            "장중 데이터가 아직 없습니다 (Yahoo 지연 또는 개장 전)."
        parts.append(f"<div class='box warn'>{msg}</div></div>")
        return "".join(parts)
    missing = [r.ticker for r in rows if not r.has_intraday]
    if missing:
        parts.append(f"<div class='box warn'><b>장중 데이터 없음</b>: {e(', '.join(missing))}</div>")
    for title, kinds in SECTIONS:
        items = _items(rows, kinds)
        parts.append(f"<h3>{e(title)}</h3>")
        if not items:
            parts.append("<p class='m'>없음</p>")
            continue
        parts.append("<ul>" + "".join(
            f"<li><span class='b {BADGE[k][0]}'>{BADGE[k][1]}</span> <b>{e(r.ticker)}</b> "
            f"<span class='m'>{e(r.name)}</span> — {e(t)}</li>" for r, k, t in items) + "</ul>")

    parts.append("<h3>전 종목 <span class='m' style='font-weight:normal;font-size:12px'>"
                 "아침 = 전일 종가 기준 3일 평균 점수, 잠정 = 현재가 기준</span></h3><table><tr>")
    for h in ["종목", "전일 종가", "시가 갭", "현재가", "등락", "아침 점수", "잠정 점수", "규칙 상태"]:
        parts.append(f"<th>{e(h)}</th>")
    parts.append("</tr>")
    group = None
    for r in rows:
        if r.group and r.group != group:
            group = r.group
            parts.append(f"<tr><td colspan='8' class='g'>{e(group)}</td></tr>")
        if not r.has_intraday:
            parts.append(f"<tr><td class='l'><b>{e(r.ticker)}</b></td><td colspan='7' class='m'>장중 데이터 없음</td></tr>")
            continue

        def sc(v):
            if pd.isna(v):
                return "<span class='b bn'>–</span>"
            c = "bg" if v >= RULE.entry else "br" if v <= RULE.exit else "bo"
            return f"<span class='b {c}'>{v:.0f}</span>"
        state = "보유 구간" if r.m.get("state") == 1 else "대기"
        if r.m.get("event"):
            state = {"BUY": "🟢 아침 매수", "SELL": "🔴 아침 매도"}[r.m["event"]]
        color = "#c62828" if r.move > 0 else "#1565c0" if r.move < 0 else "#222"
        gcolor = "#c62828" if r.gap > 0 else "#1565c0" if r.gap < 0 else "#222"
        parts.append(
            f"<tr><td class='l'><b>{e(r.ticker)}</b><br><span class='m' style='font-size:11px'>{e(r.name)}</span></td>"
            f"<td class='c'>{r.prev_close:,.2f}</td><td class='c'><span style='color:{gcolor}'>{r.gap:+.1%}</span></td>"
            f"<td class='c'>{r.price:,.2f}</td><td class='c'><span style='color:{color}'>{r.move:+.1%}</span></td>"
            f"<td class='c'>{sc(r.m.get('score_s'))}</td><td class='c'>{sc(r.p.get('score_s'))}</td>"
            f"<td class='c'>{e(state)}</td></tr>")
    parts.append(f"</table><p class='m' style='font-size:12px;margin-top:12px'>{e(NOTE)}</p></div>")
    return "".join(parts)
