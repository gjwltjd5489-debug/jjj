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
from .topk import BASKET_K, DIP_STOP, DIP_TICKER, RISK_OFF_CUT, dip_target, rebalance_days

GAP_MIN = 0.02      # 시가 갭 경고 최소폭
STOP_NEAR = 0.01    # 손절 참고가 위 1% 이내면 '근접'
STEP = 50 / 6       # 점수 한 칸 (배점 1인 성향 하나가 🟠에서 🟢·🔴로 바뀌는 폭, 성향 6개 기준)
MA200_NEAR = 0.02   # 눌림 중인 보유 종목이 200일선 위 2% 이내면 '200일선 근접'


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
            for k in ("score", "score_s", "state", "event", "blocked", "stop", "above200", "overheat", "highvol", "close",
                      "ma200")}


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
    elif pd.notna(p.get("score_s")):
        held = m.get("state") == 1
        # 신호선 한 칸(성향 6개 기준 약 8점) 안쪽. 아침에 이미 '진입 보류'였던 종목(과열 등)은 매일 반복되므로 뺀다.
        if not held and not m.get("blocked") and not p.get("blocked") and RULE.entry - STEP <= p["score_s"] < RULE.entry:
            notes.append(("near_buy", f"{ps} (진입 {RULE.entry:g}, 아침 {m['score_s']:.0f}) · {now}"))
        elif held and not p.get("above200", True) and p["score_s"] <= RULE.exit + STEP:
            notes.append(("near_sell", f"200일선 아래 · {ps} (퇴출 {RULE.exit:g}, 아침 {m['score_s']:.0f}) · {now}"))
        elif (held and p.get("above200", True) and p["score_s"] <= RULE.exit and pd.notna(p.get("ma200"))
              and r.price <= p["ma200"] * (1 + MA200_NEAR)):
            # 200일선 위 눌림: 점수는 이미 퇴출선 아래라 200일선을 깨면 매도 신호
            notes.append(("near_sell", f"눌림 중 200일선 {p['ma200']:,.2f} 근접 ({r.price / p['ma200'] - 1:+.1%}) — "
                                       f"종가가 200일선 아래면 매도 신호 · {ps} · {now}"))
    if m.get("state") == 1 and m.get("event") != "SELL" and pd.notna(m.get("stop")):
        if r.price < m["stop"]:
            notes.append(("stop_break", f"손절 참고가 {m['stop']:,.2f} 이탈 · {now}"))
        elif r.price < m["stop"] * (1 + STOP_NEAR):
            notes.append(("stop_near", f"손절 참고가 {m['stop']:,.2f} 근접 ({r.price / m['stop'] - 1:+.1%}) · {now}"))
    if pd.notna(r.gap) and abs(r.gap) >= max(GAP_MIN, r.atr_pct if pd.notna(r.atr_pct) else 0):
        notes.append(("gap", f"시가 갭 {r.gap:+.1%} (평소 하루 변동폭 {r.atr_pct:.1%}) · {now}"))
    return notes


# ---------------------------------------------------------------- 바구니 현재 상황

BASKET_FLAGS = {"new_sell": "🔴 종가 매도 후보", "near_sell": "↘ 퇴출 근접", "stop_break": "🔴 손절선 이탈",
                "stop_near": "🟠 손절선 근접"}
QLD_STOP_NEAR = 0.03   # QLD 묶음이 손절선(−25%)까지 3%p 안이면 '근접'
BASKET_SUBJECT = {"filter": "SPY 200일선 이탈 후보", "dip_buy": "QLD 매수 후보", "dip_exit": "QLD 청산 후보",
                  "dip_stop": "QLD 손절 후보"}


def _md(d) -> str:
    return f"{d.month}/{d.day}"


def basket_live(b: dict | None, rows: list[IntradayRow], spy_prev: pd.Series | None, qld_move: float | None,
                today) -> dict | None:
    """아침 메일의 바구니(전일 종가 기준, checklist.basket_state)를 지금 가격으로 다시 본 상태.

    비중은 전일 종가 거래 뒤 비중(아침 메일대로 오늘 시가에 주문했다고 보고). 지금 수익은 전일 종가 대비.
    signals: 지금 가격이 종가까지 유지되면 나올 바구니 신호 (SPY 200일선 필터·QLD 매수·청산·손절)."""
    if not b:
        return None
    nan = float("nan")
    live = {r.ticker: r for r in rows if r.has_intraday}
    info = {x["ticker"]: x for x in b["rows"]}
    order = [x["ticker"] for x in b["rows"]] + [t for t in b["w_now"] if t not in info]
    hold, total, missing = [], 0.0, []
    for t in order:
        w = b["w_now"].get(t, 0.0)
        if w <= 1e-9:
            continue
        r = live.get(t)
        m = r.move if r is not None and pd.notna(r.move) else nan
        if pd.isna(m):
            missing.append(t)
        else:
            total += w * m
        x = info.get(t, {})
        since0 = x.get("ret_hold", nan)
        hold.append({"ticker": t, "name": r.name if r is not None else "", "weight": w, "price": r.price if r is not None else nan,
                     "move": m, "since": x.get("since"), "since_cut": x.get("since_cut", False),
                     "ret_hold": (1 + since0) * (1 + m) - 1 if pd.notna(since0) and pd.notna(m) else nan,
                     "flags": [BASKET_FLAGS[k] for k, _ in (r.notes if r is not None else []) if k in BASKET_FLAGS]})
    o = b.get("ov") or {}
    qld = None
    if o.get("q_now", 0.0) > 0:
        qm = qld_move if qld_move is not None and pd.notna(qld_move) else nan
        if pd.isna(qm):
            missing.append(DIP_TICKER)
        else:
            total += o["q_now"] * qm
        qret = (1 + o["q_ret"]) * (1 + qm) - 1 if pd.notna(o.get("q_ret", nan)) and pd.notna(qm) else nan
        qld = {"weight": o["q_now"], "move": qm, "ret": qret, "t0": o.get("t0")}
    # SPY 200일선: 직전 199일 종가 + 지금 가격
    dist = nan
    spy = live.get("SPY")
    if spy_prev is not None and spy is not None and pd.notna(spy.price):
        c = spy_prev.dropna()
        if len(c) >= 199:
            ma = (float(c.iloc[-199:].sum()) + spy.price) / 200
            dist = 1 - spy.price / ma
    signals = []
    if o and pd.notna(dist):
        if not o.get("risk_off") and dist > 0:
            signals.append(("filter", f"종가가 이대로면 SPY 200일선 이탈 (지금 {-dist:+.1%}) → 내일 아침 메일에서 "
                                      f"주식 종목 절반({100 / BASKET_K * RISK_OFF_CUT:g}%) 매도 신호"))
        if o.get("qld"):
            if o.get("cq", 0) > 0 and dist <= 0:
                signals.append(("dip_exit", f"종가가 이대로면 {DIP_TICKER} 청산 신호 (SPY 200일선 회복, 지금 {-dist:+.1%})"))
            elif dist > 0 and not o.get("blocked"):
                add = dip_target(dist) - o.get("cq", 0.0)
                if add > 1e-9:
                    signals.append(("dip_buy", f"종가가 이대로면 {DIP_TICKER} {add:.0%} 매수 신호 (SPY 200일선 {-dist:+.1%})"))
    if qld and pd.notna(qld["ret"]):
        if qld["ret"] <= DIP_STOP:
            signals.append(("dip_stop", f"종가가 이대로면 {DIP_TICKER} 손절 (매수가 대비 {qld['ret']:+.1%})"))
        elif qld["ret"] <= DIP_STOP + QLD_STOP_NEAR:
            signals.append(("dip_stop_near", f"{DIP_TICKER} 손절선 근접 (매수가 대비 {qld['ret']:+.1%}, 손절 {DIP_STOP:.0%})"))
    # 아침 메일이 오늘 하라고 한 일 (전일 종가 신호)
    todo = []
    if b.get("rebalance"):
        todo.append(f"{_md(b['base'])} 종가 기준 바구니 교체 → 오늘 새 비중으로 맞추기")
    if b.get("sold_today"):
        todo.append(f"규칙 매도 {', '.join(b['sold_today'])} → 팔고 현금")
    for k, v in o.get("today", []):
        todo.append({"cut": f"SPY 200일선 이탈 → 주식 종목({', '.join(v) if isinstance(v, list) else ''}) 절반 매도",
                     "dip_buy": f"{DIP_TICKER} {v[0]:.0%} 매수" if k == "dip_buy" else "",
                     "dip_exit": f"{DIP_TICKER} 전량 매도 → 바구니 종목 채우기",
                     "dip_stop": f"{DIP_TICKER} 손절 매도"}.get(k, ""))
    prev_ret, ytd = b.get("month_ret", nan), (b.get("ytd") or {}).get("ret", nan)
    period = total if b.get("rebalance") else (1 + prev_ret) * (1 + total) - 1
    return {"base": b["base"], "prev": b["today"], "rebalance_prev": b.get("rebalance", False), "hold": hold, "qld": qld,
            "cash": b.get("cash_now", 0.0), "today": total, "period": period,
            "period_from": b["today"] if b.get("rebalance") else b["base"],
            "ytd": (1 + ytd) * (1 + total) - 1 if pd.notna(ytd) else nan,
            "cmp": [(t, live[t].move) for t in ("SPY", "QQQ") if t in live and pd.notna(live[t].move)],
            "dist": dist, "dist_morning": o.get("dist", nan), "risk_off": bool(o.get("risk_off")),
            "signals": signals, "todo": [t for t in todo if t], "missing": missing,
            "reb_today": today in rebalance_days(today), "next": b.get("next")}


def _basket_summary(bl: dict) -> str:
    cmp = " · ".join(f"{t} {m:+.1%}" for t, m in bl["cmp"])
    out = f"바구니 오늘 {_pct(bl['today'])}" + (f" (같은 시각 {cmp})" if cmp else "")
    out += f" · 이번 기간 {_pct(bl['period'])} ({_md(bl['period_from'])} 종가 → 지금)"
    if pd.notna(bl["ytd"]):
        out += f" · 올해 {_pct(bl['ytd'])}"
    return out


def _pct(x: float) -> str:
    """+0.0% 형식 (반올림해서 0 이면 -0.0% 대신 +0.0%)."""
    return f"{0.0 if abs(x) < 0.0005 else x:+.1%}"


def _basket_filter_line(bl: dict) -> str | None:
    if pd.isna(bl["dist"]):
        return None
    now = f"SPY 200일선 대비 지금 {-bl['dist']:+.1%}" + (f" (어제 종가 {-bl['dist_morning']:+.1%})"
                                                     if pd.notna(bl["dist_morning"]) else "")
    if not bl["risk_off"]:
        return now
    half = f"주식 종목은 절반({100 / BASKET_K * RISK_OFF_CUT:g}%)"
    return now + (f" → {half} 유지 중, 회복이 이어지면 다음 교체일에 원래 비중으로" if bl["dist"] <= 0 else f" → {half}")


def basket_lines(bl: dict) -> list[str]:
    lines = [f"바구니 현재 상황 ({_md(bl['prev'])} 종가 거래 뒤 바구니 · 지금 가격, 전일 종가 대비)"]
    if bl["todo"]:
        lines.append("- 오늘 할 일 (어제 종가 신호, 아침 메일): " + " · ".join(bl["todo"]))
    lines += [f"- ⚠️ {t}" for _, t in bl["signals"]]
    if bl["reb_today"]:
        lines.append("- 오늘 종가가 교체일 — 내일 아침 메일에서 새 바구니 확인")
    lines.append("- " + _basket_summary(bl))
    if _basket_filter_line(bl):
        lines.append("- " + _basket_filter_line(bl))
    for h in bl["hold"]:
        since = "" if pd.isna(h["ret_hold"]) else f" · 편입 후 {h['ret_hold']:+.1%} ({'~' if h['since_cut'] else ''}{_md(h['since'])}~)"
        now = "–" if pd.isna(h["move"]) else f"{h['price']:,.2f} ({h['move']:+.1%})"
        lines.append(f"- {h['ticker']} {h['weight'] * 100:.1f}% · 지금 {now}{since}"
                     + (" · " + ", ".join(h["flags"]) if h["flags"] else ""))
    if bl["qld"]:
        q = bl["qld"]
        lines.append(f"- {DIP_TICKER} {q['weight'] * 100:.1f}% (하락 매수) · 지금 "
                     + ("–" if pd.isna(q["move"]) else f"{q['move']:+.1%}")
                     + ("" if pd.isna(q["ret"]) else f" · 매수가 대비 {q['ret']:+.1%}"))
    lines.append(f"- 현금 {bl['cash'] * 100:.0f}%")
    if bl["missing"]:
        lines.append(f"- 장중 가격 없음 (오늘 0%로 계산): {', '.join(bl['missing'])}")
    return lines


def basket_html(bl: dict) -> str:
    e = html.escape
    parts = [f"<h3>🧺 바구니 현재 상황 <span class='m' style='font-weight:normal;font-size:12px'>"
             f"{_md(bl['prev'])} 종가 거래 뒤 바구니 · 지금 가격 (전일 종가 대비)</span></h3>"]
    box = ([f"<b>오늘 할 일</b> (어제 종가 신호): {e(' · '.join(bl['todo']))}"] if bl["todo"] else []) + \
        [f"⚠️ {e(t)}" for _, t in bl["signals"]] + \
        (["오늘 종가가 교체일 — 내일 아침 메일에서 새 바구니 확인"] if bl["reb_today"] else [])
    if box:
        parts.append("<div class='box sig'>" + "<br>".join(box) + "</div>")
    cmp = " · ".join(f"{e(t)} <span style='color:{_pct_color(m)}'>{m:+.1%}</span>" for t, m in bl["cmp"])
    parts.append(f"<p style='margin:2px 0 4px;font-size:13px'>바구니 오늘 <b><span style='color:{_pct_color(bl['today'])}'>"
                 f"{_pct(bl['today'])}</span></b>" + (f" <span class='m'>(같은 시각 {cmp})</span>" if cmp else "")
                 + f"<br>이번 기간 <b>{_pct(bl['period'])}</b> <span class='s10'>({_md(bl['period_from'])} 종가 → 지금)</span>"
                 + ("" if pd.isna(bl["ytd"]) else f" · 올해 <b>{_pct(bl['ytd'])}</b>")
                 + (f"<br><span class='m'>{e(_basket_filter_line(bl))}</span>" if _basket_filter_line(bl) else "") + "</p>")
    parts.append("<table class='tb'><tr><th class='s'>종목</th><th class='s'>비중</th><th class='s'>지금<br>등락</th>"
                 "<th class='s'>편입 후<br>(지금)</th><th class='s'>상태</th></tr>")
    for h in bl["hold"]:
        now = "–" if pd.isna(h["move"]) else (f"{h['price']:,.2f}<br><span style='color:{_pct_color(h['move'])};font-size:12px'>"
                                               f"{h['move']:+.1%}</span>")
        since = "–" if pd.isna(h["ret_hold"]) else (f"<b><span style='color:{_pct_color(h['ret_hold'])}'>{h['ret_hold']:+.1%}</span>"
                                                     f"</b><br><span class='s10'>{'~' if h['since_cut'] else ''}"
                                                     f"{_md(h['since'])}~</span>")
        parts.append(f"<tr><td class='l' style='font-size:13px'><b>{e(h['ticker'])}</b><br><span class='m nm'>{e(h['name'])}"
                     f"</span></td><td class='c'><b>{h['weight'] * 100:.1f}%</b></td><td class='c' style='font-size:12px'>{now}</td>"
                     f"<td class='c' style='font-size:12px'>{since}</td>"
                     f"<td class='c' style='font-size:11px'>{'<br>'.join(e(f) for f in h['flags']) or '–'}</td></tr>")
    if bl["qld"]:
        q = bl["qld"]
        mv = "–" if pd.isna(q["move"]) else f"<span style='color:{_pct_color(q['move'])}'>{q['move']:+.1%}</span>"
        rt = "–" if pd.isna(q["ret"]) else (f"<b><span style='color:{_pct_color(q['ret'])}'>{q['ret']:+.1%}</span></b>"
                                            f"<br><span class='s10'>매수가 대비</span>")
        parts.append(f"<tr><td class='l' style='font-size:13px'><b>{DIP_TICKER}</b><br><span class='m nm'>하락 매수</span></td>"
                     f"<td class='c'><b>{q['weight'] * 100:.1f}%</b></td><td class='c' style='font-size:12px'>{mv}</td>"
                     f"<td class='c' style='font-size:12px'>{rt}</td><td class='c'>–</td></tr>")
    parts.append(f"<tr><td class='l m' style='font-size:13px'>현금</td><td class='c'><b>{bl['cash'] * 100:.0f}%</b></td>"
                 "<td colspan='3'></td></tr></table>")
    if bl["missing"]:
        parts.append(f"<p class='m' style='font-size:11px;margin:2px 0'>장중 가격 없음 (오늘 0%로 계산): "
                     f"{e(', '.join(bl['missing']))}</p>")
    return "".join(parts)


# ---------------------------------------------------------------- 출력

SECTIONS = [
    ("아침 신호 재확인", ["buy_keep", "buy_weak", "buy_cancel", "sell_keep", "sell_weak"]),
    ("장중 신규 후보·신호선 근접 (종가 확정 전)", ["new_buy", "new_sell", "near_buy", "near_sell"]),
    ("손절 참고가 점검 (규칙상 보유 구간)", ["stop_break", "stop_near"]),
    ("큰 시가 갭", ["gap"]),
]
BADGE = {
    "buy_keep": ("bg", "🟢 매수 유지"), "buy_weak": ("bo", "🟠 매수 약화"), "buy_cancel": ("br", "🔴 매수 취소"),
    "sell_keep": ("br", "🔴 매도 유지"), "sell_weak": ("bo", "🟠 매도 약화"),
    "new_buy": ("bg", "🟢 신규 매수 후보"), "new_sell": ("br", "🔴 신규 매도 후보"),
    "near_buy": ("bn", "↗ 진입선 근접"), "near_sell": ("bo", "↘ 퇴출선 근접"),
    "stop_break": ("br", "🔴 손절선 이탈"), "stop_near": ("bo", "🟠 손절선 근접"), "gap": ("bn", "⚪ 갭"),
}
SUBJECT_ORDER = [("buy_keep", "매수 유지"), ("buy_weak", "매수 약화"), ("buy_cancel", "매수 취소"),
                 ("sell_keep", "매도 유지"), ("sell_weak", "매도 약화"), ("new_buy", "신규 매수 후보"),
                 ("new_sell", "신규 매도 후보"), ("stop_break", "손절선 이탈"), ("stop_near", "손절선 근접"),
                 ("near_buy", "진입 근접"), ("near_sell", "퇴출 근접")]
NOTE = ("검증된 규칙은 종가 기준이다. 잠정 점수는 현재가를 오늘 종가로 가정한 3일 평균이라 종가까지 바뀔 수 있다. "
        "장 초반 거래량은 하루치와 비교할 수 없어 20일 중앙값으로 가정했다. 가격은 수 분 지연될 수 있다. 매매 권유가 아님.")


def _items(rows: list[IntradayRow], kinds: list[str]) -> list[tuple[IntradayRow, str, str]]:
    return [(r, k, t) for k in kinds for r in rows for kk, t in r.notes if kk == k]


def make_subject(rows: list[IntradayRow], meta: dict) -> str:
    if meta["mode"] == "holiday":
        return f"[장초반 확인] 휴장 · {meta['date']} {meta['holiday']}"
    if meta["mode"] == "no_data":
        return f"[장초반 확인] {meta['date']} · 장중 데이터 없음"
    parts = []
    bl = meta.get("basket")
    if bl:
        parts.append(f"바구니 {_pct(bl['today'])}")
        parts += [BASKET_SUBJECT[k] for k, _ in bl["signals"] if k in BASKET_SUBJECT]
    for k, label in SUBJECT_ORDER:
        names = [r.ticker for r in rows for kk, _ in r.notes if kk == k]
        if names:
            parts.append(f"{label} {','.join(names)}")
    if not parts or (bl and len(parts) == 1):
        parts.append("특이사항 없음")
    return f"[장초반 확인] {meta['date']} {meta['time']} ET · " + " · ".join(parts)


def table_rows(rows: list[IntradayRow]) -> tuple[list[IntradayRow], list[IntradayRow]]:
    """(표에 보여줄 종목, 한 줄로 줄일 종목). 표 = 알림 있는 종목 → 보유 구간 → 점수가 한 칸 이상 움직인 종목."""
    kinds = [k for _, ks in SECTIONS for k in ks]

    def moved(r: IntradayRow) -> bool:
        a, b = r.m.get("score_s"), r.p.get("score_s")
        return pd.notna(a) and pd.notna(b) and abs(b - a) >= STEP

    def rank(r: IntradayRow) -> tuple:
        first = min((kinds.index(k) for k, _ in r.notes if k in kinds), default=len(kinds))
        return (first, 0 if r.m.get("state") == 1 else 1, -(r.p.get("score_s") or 0))
    live = [r for r in rows if r.has_intraday]
    show = [r for r in live if r.notes or r.m.get("state") == 1 or r.m.get("event") or moved(r)]
    return sorted(show, key=rank), [r for r in live if r not in show]


def _today_line(meta: dict) -> str | None:
    ev = meta.get("today_events")
    return ("오늘·내일 일정: " + " · ".join(f"{d} {label}" for d, label in ev)) if ev else None


def render_text(rows: list[IntradayRow], meta: dict) -> str:
    lines = [f"장 초반 확인 ({meta['date']} {meta['time']} 뉴욕 시각, 개장 후 약 {meta['minutes']}분)", ""]
    if meta["mode"] != "normal":
        lines.append("휴장일입니다." if meta["mode"] == "holiday" else "장중 데이터가 아직 없습니다 (Yahoo 지연 또는 개장 전).")
        return "\n".join(lines)
    if _today_line(meta):
        lines += [_today_line(meta), ""]
    if meta.get("basket"):
        lines += basket_lines(meta["basket"]) + [""]
    empty = []
    for title, kinds in SECTIONS:
        items = _items(rows, kinds)
        if not items:
            empty.append(title.split(" (")[0])
            continue
        lines += [title] + [f"- {BADGE[k][1]} {r.ticker}: {t}" for r, k, t in items] + [""]
    if len(empty) == len(SECTIONS):
        lines += ["특이사항 없음 — 아침 신호·신규 후보·손절선·시가 갭 모두 해당 없음", ""]
    elif empty:
        lines += ["해당 없음: " + " · ".join(empty), ""]
    show, rest = table_rows(rows)
    if show:
        lines.append("주요 종목 (3일 평균 아침 → 잠정)")
        for r in show:
            stop = (f" · 손절 {r.m['stop']:,.2f} (현재가 {r.price / r.m['stop'] - 1:+.1%})"
                    if r.m.get("state") == 1 and pd.notna(r.m.get("stop")) else "")
            lines.append(f"- {r.ticker} {r.price:,.2f} ({r.move:+.1%}) · {_fmt(r.m.get('score_s'))} → "
                         f"{_fmt(r.p.get('score_s'))}{stop}")
        lines.append("")
    if rest:
        lines += [f"나머지 {len(rest)}종목 (대기 · 점수 변화 작음): " + ", ".join(r.ticker for r in rest), ""]
    lines.append(NOTE)
    return "\n".join(lines)


def _fmt(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.0f}"


def _sc(v) -> str:
    if v is None or pd.isna(v):
        return "<span class='b bn'>–</span>"
    c = "bg" if v >= RULE.entry else "br" if v <= RULE.exit else "bo"
    return f"<span class='b {c}'>{v:.0f}</span>"


def _pct_color(x: float) -> str:
    return "#c62828" if x > 0 else "#1565c0" if x < 0 else "#222"


def render_html(rows: list[IntradayRow], meta: dict) -> str:
    e = html.escape
    parts = [_CSS.replace("</style>", _BADGE_CSS + "</style>"), '<div class="ck">',
             f"<h2 style='margin:0 0 6px'>장 초반 확인 <span class='m' style='font-weight:normal;font-size:14px'>"
             f"({e(meta['date'])} {e(meta['time'])} ET · 개장 후 약 {meta['minutes']}분)</span></h2>"]
    if meta["mode"] != "normal":
        msg = f"휴장일입니다 ({e(meta.get('holiday') or '')})." if meta["mode"] == "holiday" else \
            "장중 데이터가 아직 없습니다 (Yahoo 지연 또는 개장 전)."
        parts.append(f"<div class='box warn'>{msg}</div></div>")
        return "".join(parts)
    missing = [r.ticker for r in rows if not r.has_intraday]
    if missing:
        parts.append(f"<div class='box warn'><b>장중 데이터 없음</b>: {e(', '.join(missing))}</div>")
    if _today_line(meta):
        parts.append(f"<div class='box'>📅 {e(_today_line(meta))}</div>")
    if meta.get("basket"):
        parts.append(basket_html(meta["basket"]))
    empty = []
    for title, kinds in SECTIONS:
        items = _items(rows, kinds)
        if not items:
            empty.append(title.split(" (")[0])
            continue
        parts.append(f"<h3>{e(title)}</h3><ul>" + "".join(
            f"<li><span class='b {BADGE[k][0]}'>{BADGE[k][1]}</span> <b>{e(r.ticker)}</b> "
            f"<span class='m'>{e(r.name)}</span> — {e(t)}</li>" for r, k, t in items) + "</ul>")
    if len(empty) == len(SECTIONS):
        parts.append("<div class='box ok'><b>특이사항 없음</b> — 아침 신호·신규 후보·손절선·시가 갭 모두 해당 없음</div>")
    elif empty:
        parts.append(f"<p class='m' style='font-size:12px;margin:8px 0'>해당 없음: {e(' · '.join(empty))}</p>")

    show, rest = table_rows(rows)
    if show:
        parts.append("<h3>주요 종목 <span class='m' style='font-weight:normal;font-size:12px'>"
                     "알림 → 보유 구간 → 점수가 크게 움직인 종목 순</span></h3>"
                     "<table class='tb'><tr><th class='s'>종목</th><th class='s'>현재가</th><th class='s'>3일 평균<br>아침→잠정</th>"
                     "<th class='s'>손절<br>참고가</th><th class='s'>상태</th></tr>")
        for r in show:
            state = "보유" if r.m.get("state") == 1 else "대기"
            if r.m.get("event"):
                state = {"BUY": "🟢 아침<br>매수", "SELL": "🔴 아침<br>매도"}[r.m["event"]]
            stop = "–"
            if r.m.get("state") == 1 and pd.notna(r.m.get("stop")):
                stop = (f"{r.m['stop']:,.2f}<br><span class='m' style='font-size:11px'>"
                        f"현재가 {r.price / r.m['stop'] - 1:+.1%}</span>")
            parts.append(
                f"<tr><td class='l' style='font-size:13px'><b>{e(r.ticker)}</b><br><span class='m nm'>{e(r.name)}</span></td>"
                f"<td class='c' style='font-size:12px'>{r.price:,.2f}<br><span style='color:{_pct_color(r.move)}'>{r.move:+.1%}</span>"
                f"<br><span class='m' style='font-size:10px'>갭 {r.gap:+.1%}</span></td>"
                f"<td class='c'>{_sc(r.m.get('score_s'))} → {_sc(r.p.get('score_s'))}</td>"
                f"<td class='c' style='font-size:12px'>{stop}</td><td class='c' style='font-size:11px'>{state}</td></tr>")
        parts.append("</table>")
    if rest:
        parts.append(f"<p class='m' style='font-size:12px;margin:6px 0'>나머지 {len(rest)}종목 (대기 · 점수 변화 작음): "
                     f"{e(', '.join(r.ticker for r in rest))}</p>")
    parts.append(f"<p class='m' style='font-size:11px;margin-top:12px'>{e(NOTE)}</p></div>")
    return "".join(parts)
