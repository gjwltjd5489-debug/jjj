"""일일 지표 체크리스트 v2: 점수 대신 성향별 판정과 지표별 상태, 오늘 발생한 이벤트를 요약한다.

성향(각 1~2개, 서로 겹치지 않게): 장기 추세 · 중기 추세 · 모멘텀 · 추세 강도 · 거래량 · 상대강도 · 변동성 · 과열.
방향 항목은 ✅ 강세 / ❌ 약세 / ➖ 중립, 위험 항목(변동성·과열)은 ⚠️ 주의 / ➖ 정상.
기준은 가능한 한 종목별 1년 백분위를 쓴다. 검토 근거는 docs/checklist_review.md.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profiles import Profile
from .checklist_score import RULE, checklist_score, points_from_verdicts
from .indicators import rolling_pct_rank
from .bigtech import eligible
from .topk import (BASKET_K, BASKET_N, MOM_DAYS, basket_weights, month_end, next_rebalance, prev_rebalance,
                   rebalance_days)
from .score import compute_indicators

UP, DOWN, WARN, NEUTRAL = "✅", "❌", "⚠️", "➖"

# 단기 반등 후보: 200일선 위에서 나온 단기 급락 신호 → 과거(29종목, 2005~2026) 같은 종목·같은 국면의
# 평소보다 높았던 이후 수익. 이평 돌파·MACD·구름·신고가·스퀴즈 등 나머지 이벤트는 이후 수익률과
# 무관해서 메일에서 뺐다 (docs/events.md).
REBOUND = {"RSI 30 진입": "5일 +0.9%p", "ADX 25 상향 (하락 방향)": "5일 +0.8%p", "볼린저 하단 이탈": "20일 +0.8%p"}
REBOUND_NOTE = "200일선 위 단기 급락 · 과거 평소 대비 이후 수익 · 매매 신호 아님"


@dataclass
class Check:
    group: str
    name: str
    status: str
    detail: str
    tag: str = ""      # 성향 판정에 쓰는 짧은 표기 (변동성 '고변동', 과열 '정상' 등)
    short: str = ""    # 상세 표에 쓰는 핵심 수치 (예: '+7.0%', 'RSI 52')
    na: bool = False   # 데이터 부족으로 계산 안 됨


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
    rebound: list[str] = field(default_factory=list)  # 오늘 나온 단기 반등 후보 신호 (REBOUND)
    prev_checks: list[Check] | None = None   # 직전 거래일 체크리스트
    issues: list[str] = field(default_factory=list)  # 데이터 점검 경고
    sig: dict | None = None  # 체크리스트 점수 기반 매수·매도 신호 (scoring/checklist_score.py)

    @property
    def changes(self) -> list[tuple[Check, Check]]:
        """(직전, 오늘) 상태가 바뀐 항목."""
        if not self.prev_checks:
            return []
        prev = {c.name: c for c in self.prev_checks}
        return [(prev[c.name], c) for c in self.checks if c.name in prev and prev[c.name].status != c.status]

    @property
    def prev_ups(self) -> int | None:
        if self.prev_checks is None:
            return None
        return sum(verdict(self.prev_checks, g) == "↑" for g, _, kind in GROUPS if kind == "dir")

    @property
    def score(self) -> float:
        """체크리스트 점수 (0~100): 방향 성향 (🟢 수 − 🔴 수) ÷ 판정 가능 수 → 50 ± 50."""
        return checklist_points(self.checks)

    @property
    def prev_score(self) -> float | None:
        return None if self.prev_checks is None else checklist_points(self.prev_checks)

    def verdicts(self) -> dict[str, str]:
        return {g: verdict(self.checks, g) for g, _, _ in GROUPS}

    def prev_verdicts(self) -> dict[str, str] | None:
        return None if self.prev_checks is None else {g: verdict(self.prev_checks, g) for g, _, _ in GROUPS}

    @property
    def ups(self) -> int:
        """방향 성향 중 ↑ 개수."""
        return sum(verdict(self.checks, g) == "↑" for g, _, kind in GROUPS if kind == "dir")

    @property
    def downs(self) -> int:
        return sum(verdict(self.checks, g) == "↓" for g, _, kind in GROUPS if kind == "dir")

    def check(self, name: str) -> Check:
        return next(c for c in self.checks if c.name == name)


def _crossed_up(a: pd.Series, b, k: int = -1) -> bool:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return bool(a.iloc[k] > b.iloc[k] and a.iloc[k - 1] <= b.iloc[k - 1])


def _crossed_down(a: pd.Series, b, k: int = -1) -> bool:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return bool(a.iloc[k] < b.iloc[k] and a.iloc[k - 1] >= b.iloc[k - 1])


def build_report(df: pd.DataFrame, p: Profile, ticker: str, name: str = "", group: str = "",
                 bench: pd.Series | None = None) -> TickerReport:
    """bench: 상대강도 비교 대상 종가 (없으면 상대강도는 '해당 없음')."""
    x = compute_indicators(df, p, bench)
    close = x["close"]
    r = x.iloc[-1]
    c = r["close"]
    m, lng = f"{p.ma_mid}일", f"{p.ma_long}일"
    checks: list[Check] = []

    def na(group_, name_):
        checks.append(Check(group_, name_, NEUTRAL, "데이터 부족", "–", na=True))

    def add(group_, name_, cond, up_text, down_text):
        if pd.isna(cond):
            na(group_, name_)
        else:
            checks.append(Check(group_, name_, UP if cond else DOWN, up_text if cond else down_text))

    # 장기 추세
    if pd.isna(r["ma_long"]):
        na("장기 추세", "200일선")
    else:
        add("장기 추세", "200일선", c > r["ma_long"], f"{lng}선 위 ({c / r['ma_long'] - 1:+.1%})",
            f"{lng}선 아래 ({c / r['ma_long'] - 1:+.1%})")
    ms, mm, ml = r["ma_short"], r["ma_mid"], r["ma_long"]
    if pd.isna(ml):
        na("장기 추세", "배열")
    elif ms > mm > ml:
        checks.append(Check("장기 추세", "배열", UP, "정배열"))
    elif ms < mm < ml:
        checks.append(Check("장기 추세", "배열", DOWN, "역배열"))
    else:
        checks.append(Check("장기 추세", "배열", NEUTRAL, "혼조"))

    # 중기 추세 (일목)
    if pd.isna(r["span_a"]) or pd.isna(r["span_b"]):
        na("중기 추세", "구름")
    else:
        top, bottom = max(r["span_a"], r["span_b"]), min(r["span_a"], r["span_b"])
        if c > top:
            checks.append(Check("중기 추세", "구름", UP, "구름 위"))
        elif c < bottom:
            checks.append(Check("중기 추세", "구름", DOWN, "구름 아래"))
        else:
            checks.append(Check("중기 추세", "구름", NEUTRAL, "구름 안"))
    add("중기 추세", "전환/기준", r["tenkan"] > r["kijun"] if pd.notna(r["kijun"]) else np.nan,
        "전환선 > 기준선", "전환선 < 기준선")

    # 모멘텀
    add("모멘텀", "MACD", r["macd"] > r["signal"] if pd.notna(r["macd"]) else np.nan,
        "MACD > 시그널", "MACD < 시그널")
    rsi = r["rsi"]
    if pd.isna(rsi):
        na("모멘텀", "RSI")
    elif rsi >= 50:
        checks.append(Check("모멘텀", "RSI", UP, f"{rsi:.0f}" + (" 강한 모멘텀" if rsi >= 70 else "")))
    else:
        checks.append(Check("모멘텀", "RSI", DOWN, f"{rsi:.0f}" + (" 과매도·반등 후보" if rsi <= 30 else "")))

    # 추세 강도 (ADX + DMI 방향)
    adx = r["adx"]
    if pd.isna(adx):
        na("추세 강도", "ADX")
    elif adx >= 25:
        up = r["plus_di"] > r["minus_di"]
        checks.append(Check("추세 강도", "ADX", UP if up else DOWN,
                            f"{adx:.0f} 강한 {'상승' if up else '하락'}추세"))
    else:
        checks.append(Check("추세 강도", "ADX", NEUTRAL,
                            f"{adx:.0f} {'횡보 — 추세 신호 신뢰 낮음' if adx < 20 else '추세 약함'}",
                            "횡보" if adx < 20 else "약함"))

    # 거래량 (종목별 1년 백분위 / OBV 20일 기울기)
    vr, vr_pct = r["vr"], r["vr_pct"]
    if pd.isna(vr_pct):
        na("거래량", "VR")
    else:
        extreme = " (1년 중 상위권)" if vr_pct >= 0.9 else " (1년 중 하위권)" if vr_pct <= 0.1 else ""
        checks.append(Check("거래량", "VR", UP if vr_pct >= 0.5 else DOWN,
                            f"{vr:.0f}%, 1년 백분위 {vr_pct:.0%}{extreme}"))
    obv_chg = x["obv"] - x["obv"].shift(20)
    add("거래량", "OBV", obv_chg.iloc[-1] > 0 if pd.notna(obv_chg.iloc[-1]) else np.nan,
        "20일간 OBV 증가", "20일간 OBV 감소")

    # 상대강도
    if bench is None:
        checks.append(Check("상대강도", "상대강도", NEUTRAL, "해당 없음", "–"))
    elif pd.isna(r["rs_ratio_ma"]):
        na("상대강도", "상대강도")
    else:
        add("상대강도", "상대강도", r["rs_ratio"] > r["rs_ratio_ma"],
            f"벤치마크 대비 비율이 50일 평균 위 (3개월 상대수익 {r['rs_ret']:+.1%})",
            f"벤치마크 대비 비율이 50일 평균 아래 (3개월 상대수익 {r['rs_ret']:+.1%})")

    # 변동성 (위험 항목)
    rv, rv_pct = r["rvol"], r["rvol_pct"]
    if pd.isna(rv_pct):
        na("변동성", "변동성")
    elif rv_pct >= 0.8:
        checks.append(Check("변동성", "변동성", WARN, f"연 {rv:.0%}, 1년 백분위 {rv_pct:.0%}", "고변동"))
    else:
        tag = "저변동" if rv_pct <= 0.2 else "보통"
        checks.append(Check("변동성", "변동성", NEUTRAL, f"연 {rv:.0%}, 1년 백분위 {rv_pct:.0%}", tag))

    # 과열 (50일선 이격도 1년 백분위, 위험 항목)
    disp = close / x["ma_mid"] - 1
    disp_pct = rolling_pct_rank(disp, p.pct_window)
    dv, dp = disp.iloc[-1], disp_pct.iloc[-1]
    if pd.isna(dp):
        na("과열", "이격도")
    elif dp >= 0.95:
        checks.append(Check("과열", "이격도", WARN, f"{m}선 대비 {dv:+.1%}, 1년 백분위 {dp:.0%}", "과열"))
    elif dp <= 0.05:
        checks.append(Check("과열", "이격도", WARN, f"{m}선 대비 {dv:+.1%}, 1년 백분위 {dp:.0%}", "과매도"))
    else:
        checks.append(Check("과열", "이격도", NEUTRAL, f"{m}선 대비 {dv:+.1%}, 1년 백분위 {dp:.0%}", "정상"))

    # 상세 표용 핵심 수치
    def rank_text(pct: float) -> str:
        return f"상위 {1 - pct:.0%}" if pct >= 0.5 else f"하위 {pct:.0%}"

    shorts = {
        "200일선": f"200일선 {c / r['ma_long'] - 1:+.1%}" if pd.notna(r["ma_long"]) else "",
        "MACD": ("MACD▲" if r["macd"] > r["signal"] else "MACD▼") if pd.notna(r["macd"]) else "",
        "RSI": f"RSI {rsi:.0f}" if pd.notna(rsi) else "",
        "ADX": f"ADX {adx:.0f}" if pd.notna(adx) else "",
        "VR": f"VR {vr:.0f}% ({rank_text(vr_pct)})" if pd.notna(vr_pct) else "",
        "OBV": ("OBV▲" if obv_chg.iloc[-1] > 0 else "OBV▼") if pd.notna(obv_chg.iloc[-1]) else "",
        "상대강도": f"3개월 {r['rs_ret']:+.1%}" if bench is not None and pd.notna(r["rs_ret"]) else "",
        "변동성": f"연 {rv:.0%} ({rank_text(rv_pct)})" if pd.notna(rv_pct) else "",
        "이격도": f"50일선 {dv:+.1%} ({rank_text(dp)})" if pd.notna(dp) else "",
    }
    for chk in checks:
        chk.short = shorts.get(chk.name) or ("" if chk.na else chk.detail.replace("전환선 ", "전환").replace(" 기준선", "기준"))

    # 단기 반등 후보: 200일선 위에서의 단기 급락 (REBOUND)
    ev: list[str] = []
    if pd.notna(r["ma_long"]) and c > r["ma_long"]:
        if _crossed_down(x["rsi"], 30):
            ev.append("RSI 30 진입")
        if _crossed_up(x["adx"], 25) and r["plus_di"] <= r["minus_di"]:
            ev.append("ADX 25 상향 (하락 방향)")
        if _crossed_down(x["pct_b"], 0):
            ev.append("볼린저 하단 이탈")
    hi252 = close.rolling(252, min_periods=60).max()

    prev = close.iloc[-2] if len(close) > 1 else np.nan
    return TickerReport(
        ticker=ticker, name=name, group=group, date=x.index[-1], close=float(c),
        change=float(c / prev - 1) if prev else np.nan,
        from_high=float(c / hi252.iloc[-1] - 1) if pd.notna(hi252.iloc[-1]) else np.nan,
        checks=checks, rebound=ev,
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
    missing = [c.name for c in rep.checks if c.na]
    if missing:
        issues.append(f"계산 안 된 지표: {', '.join(missing)}")
    return issues


def build_with_history(df: pd.DataFrame, p: Profile, ticker: str, name: str = "", group: str = "",
                       expected=None, bench: pd.Series | None = None) -> TickerReport:
    """오늘 체크리스트 + 직전 거래일 체크리스트(비교용) + 데이터 점검."""
    rep = build_report(df, p, ticker, name, group, bench)
    if len(df) > 2:
        prev_bench = bench.loc[: df.index[-2]] if bench is not None else None
        rep.prev_checks = build_report(df.iloc[:-1], p, ticker, name, group, prev_bench).checks
    rep.issues = data_issues(df, rep, expected)
    rep.sig = trade_signal(df, p, bench)
    return rep




# ---------------------------------------------------------------- 체크리스트 점수 + 매수·매도 신호

_POINT = {"↑": 1, "↓": -1, "–": None}
RECENT_DAYS = 60   # '최근 신호 기록' 기간 (거래일)
RECENT_ROWS = 12   # 신호 기록 표에 보여줄 최대 건수
TOP_CHANGES = 5    # '전일 대비 변화'에서 자세히 보여줄 종목 수
DOC_URL = "https://github.com/gjwltjd5489-debug/jjj/blob/claude/festive-johnson-oi145x/docs/checklist_score.md"


def checklist_points(checks: list[Check]) -> float:
    """메일 표의 방향 성향 색으로 계산하는 점수 (🟢 +1, 🟠 0, 🔴 −1, ⚪ 제외)."""
    dirs = [g for g, _, kind in GROUPS if kind == "dir"]
    return points_from_verdicts({g: _POINT.get(verdict(checks, g), 0) for g in dirs}, dict(RULE.weights))


def color_counts(checks: list[Check]) -> tuple[int, int, int]:
    """방향 성향의 (🟢, 🟠, 🔴) 개수."""
    vals = [_POINT.get(verdict(checks, g), 0) for g, _, kind in GROUPS if kind == "dir" and verdict(checks, g) != "–"]
    return sum(v == 1 for v in vals), sum(v == 0 for v in vals), sum(v == -1 for v in vals)


def trade_signal(df: pd.DataFrame, p: Profile, bench: pd.Series | None = None) -> dict | None:
    """체크리스트 점수 규칙의 오늘 상태와 최근 신호 기록. 지표가 준비되지 않았으면 None."""
    try:
        o = checklist_score(df, p, bench)
    except Exception:
        return None
    last = o.iloc[-1]
    if pd.isna(last["score_s"]) or pd.isna(last["state"]):
        return None
    tail = o.iloc[-RECENT_DAYS:]
    recent = [{"date": ts, "event": ev, "close": float(c), "ret": float(last["close"] / c - 1)}
              for ts, ev, c in zip(tail.index, tail["event"], tail["close"]) if ev]
    return {"score": float(last["score"]), "score_s": float(last["score_s"]),
            "prev_score_s": float(o["score_s"].iloc[-2]) if len(o) > 1 else np.nan,
            "proj": float(last["proj"]), "held": bool(last["state"] == 1),
            "event": last["event"], "blocked": last["blocked"], "stop": float(last["stop"]),
            "close": float(last["close"]), "above200": bool(last["above200"]), "ma200": float(last["ma200"]),
            "overheat": bool(last["overheat"]), "highvol": bool(last["highvol"]), "recent": recent,
            "hist": o[["score_s", "state", "close"]].iloc[-300:].copy()}  # 바구니 계산용 (지난 교체일들, 약 1년)


def _entry_risks(g: dict) -> list[str]:
    """지금 진입하면 막히는 이유 (200일선 아래·과열·고변동)."""
    out = []
    if RULE.need_above_200 and not g.get("above200", True):
        out.append("200일선 아래")
    if RULE.block_overheat and g.get("overheat"):
        out.append("과열")
    if RULE.block_highvol and g.get("highvol"):
        out.append("고변동")
    return out


def _stop_text(g: dict) -> str:
    if pd.isna(g.get("stop", np.nan)):
        return "손절 참고가 없음"
    return f"손절 참고 {g['stop']:,.2f} ({g['stop'] / g['close'] - 1:+.1%})"


def _ma200_text(g: dict) -> str:
    ma = g.get("ma200", np.nan)
    return "200일선 –" if pd.isna(ma) else f"200일선 {ma:,.2f} (종가가 {g['close'] / ma - 1:+.1%})"


def signal_groups(reports: list[TickerReport]) -> dict[str, list[tuple[TickerReport, str]]]:
    """buy / sell / blocked(진입 보류) / near_buy(진입 임박) / near_exit(청산 임박) / pullback(눌림, 보유 유지)
    / hold(그 밖의 보유 구간).

    임박 = 당일 점수가 내일도 같으면 내일 3일 평균이 진입선·퇴출선을 넘는 경우 (청산 임박은 3일 평균 50 이하도 포함).
    퇴출은 200일선 아래에서만 나므로 청산 임박도 200일선 아래 종목만. 200일선 위에서 점수가 퇴출선 아래면 '눌림'.
    """
    out = {k: [] for k in ("buy", "sell", "blocked", "near_buy", "near_exit", "pullback", "hold")}
    for r in reports:
        g = r.sig
        if not g:
            continue
        up, mid, dn = color_counts(r.checks)
        cnt = f"🟢{up} 🟠{mid} 🔴{dn}"
        s_txt = f"3일 평균 {g['score_s']:.0f} (당일 {g['score']:.0f})"
        proj = g.get("proj", np.nan)
        if g["event"] == "BUY":
            out["buy"].append((r, f"{s_txt} ≥ {RULE.entry:g} · {cnt} · {_stop_text(g)}"))
        elif g["event"] == "SELL":
            out["sell"].append((r, f"3일 평균 {g['score_s']:.0f} ≤ {RULE.exit:g} · {cnt}"))
        elif not g["held"]:
            if g["blocked"]:
                out["blocked"].append((r, f"{s_txt} ≥ {RULE.entry:g}이지만 {g['blocked']}"))
            elif pd.notna(proj) and proj >= RULE.entry:
                risk = _entry_risks(g)
                out["near_buy"].append((r, f"{s_txt} · 당일 점수가 내일도 같으면 내일 평균 {proj:.0f} → 매수 신호"
                                           + (f" (단, 지금은 {'·'.join(risk)}라 진입 보류)" if risk else "")))
        elif g.get("above200", True) and RULE.exit_above200 is None:
            if g["score_s"] <= RULE.exit:
                out["pullback"].append((r, f"3일 평균 {g['score_s']:.0f} ≤ {RULE.exit:g}이지만 200일선 위라 보유 유지 · "
                                           f"{_ma200_text(g)} · 200일선을 깨면 매도"))
            else:
                out["hold"].append((r, f"{g['score_s']:.0f}"))
        elif g["score_s"] <= RULE.near_exit or (pd.notna(proj) and proj <= RULE.exit):
            t = f"200일선 아래 · 3일 평균 {g['score_s']:.0f} (퇴출선 {RULE.exit:g}까지 {g['score_s'] - RULE.exit:.0f})"
            if pd.notna(proj) and proj <= RULE.exit:
                t += f" · 당일 점수가 내일도 같으면 내일 평균 {proj:.0f} → 매도 신호"
            out["near_exit"].append((r, t))
        else:
            out["hold"].append((r, f"{g['score_s']:.0f}"))
    for k in ("buy", "near_buy", "hold"):
        out[k].sort(key=lambda t: -t[0].sig["score_s"])
    return out


def holdings(reports: list[TickerReport]) -> list[TickerReport]:
    """규칙상 보유 구간 종목 (오늘 매수 신호 포함), 3일 평균 점수 순."""
    return sorted((r for r in reports if r.sig and r.sig["held"]), key=lambda r: -r.sig["score_s"])


# ---------------------------------------------------------------- 바구니 (docs/topk.md)

BASKET_RULE = (f"규칙상 보유 종목 중 3일 평균 {BASKET_N:g}점 이상 상위 {BASKET_K}개를 {100 / BASKET_K:g}%씩, "
               "매월 15일(휴장이면 직전 거래일)·마지막 거래일 종가 기준으로 교체 (점수가 같으면 최근 60거래일 수익률 순). "
               "교체 사이에 규칙 매도 신호가 나면 바로 팔아 현금으로 두고, 빈자리와 현금은 단기국채 금리를 받는다고 계산")


def _state_at(reports: list[TickerReport], d) -> tuple[dict, dict]:
    """d 종가 기준 (3일 평균 점수, 규칙상 보유) — 그날 막대가 없으면 5일 안의 직전 값.
    빅테크 칸 종목은 d 기준 목록(scoring/bigtech.py)에 있을 때만 후보."""
    scores, held = {}, {}
    day = pd.Timestamp(d).date()
    for r in reports:
        h = r.sig.get("hist") if r.sig else None
        if h is None or h.empty or not eligible(r.ticker, day):
            continue
        h = h.loc[: pd.Timestamp(d)]
        if len(h) and (pd.Timestamp(d) - h.index[-1]).days <= 5 and pd.notna(h["score_s"].iloc[-1]):
            scores[r.ticker] = float(h["score_s"].iloc[-1])
            held[r.ticker] = h["state"].iloc[-1] == 1
    return scores, held


def _mom_at(reports: list[TickerReport], d) -> dict[str, float]:
    """d 종가 기준 최근 MOM_DAYS 거래일 수익률 (바구니 동점 가르기용)."""
    out = {}
    for r in reports:
        h = r.sig.get("hist") if r.sig else None
        if h is None or h.empty:
            continue
        c = h["close"].loc[: pd.Timestamp(d)]
        if len(c) > MOM_DAYS and c.iloc[-1 - MOM_DAYS]:
            out[r.ticker] = float(c.iloc[-1] / c.iloc[-1 - MOM_DAYS] - 1)
    return out


def _basket_at(reports: list[TickerReport], d) -> tuple[dict, dict, dict]:
    """d 종가 기준 (바구니 비중, 3일 평균 점수, 규칙상 보유)."""
    sc, hd = _state_at(reports, d)
    return basket_weights(sc, hd, mom=_mom_at(reports, d)), sc, hd


def _close_at(reports: list[TickerReport], d=None) -> dict[str, float]:
    """d 종가 (배당 조정). d=None 이면 가장 최근 종가. 그날 막대가 없으면 5일 안의 직전 값."""
    out = {}
    for r in reports:
        h = r.sig.get("hist") if r.sig else None
        if h is None or h.empty or "close" not in h:
            continue
        if d is not None:
            h = h.loc[: pd.Timestamp(d)]
            if not len(h) or (pd.Timestamp(d) - h.index[-1]).days > 5:
                continue
        out[r.ticker] = float(h["close"].iloc[-1])
    return out


def _ret(c1: dict, c0: dict, t: str) -> float:
    return c1[t] / c0[t] - 1 if t in c1 and t in c0 and c0[t] else float("nan")


def _frames(reports: list[TickerReport]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 기록 → (종가, 규칙상 보유) 표. 날짜는 모든 종목 날짜의 합집합, 빈 날은 직전 값."""
    cl, st = {}, {}
    for r in reports:
        h = r.sig.get("hist") if r.sig else None
        if h is not None and not h.empty:
            cl[r.ticker], st[r.ticker] = h["close"], h["state"]
    if not cl:
        return pd.DataFrame(), pd.DataFrame()
    c = pd.DataFrame(cl).sort_index()
    return c.ffill(), pd.DataFrame(st).reindex(c.index).ffill()


def _rf_on(rf: pd.Series | None, d: pd.Timestamp) -> float:
    """그날 현금 일간 수익률 (단기국채 금리 ÷ 252). 없으면 0."""
    if rf is None or rf.empty:
        return 0.0
    s = rf.loc[:d]
    return float(s.iloc[-1]) if len(s) and (d - s.index[-1]).days <= 7 else 0.0


BASKET_COST = 0.0005  # 바뀐 비중에 매기는 편도 비용 (검증과 같음)


def _simulate(reports: list[TickerReport], start, end, rf: pd.Series | None = None, buy_cost: bool = True,
              trade_at_end: bool = True) -> dict:
    """start 교체일 종가에 바구니를 짜고 end 종가까지 규칙대로 운용 (scripts/eval_topk.py 의 Book 과 같은 계산).

    교체일마다 basket_weights 로 바꾸고, 교체 사이에 규칙 매도 신호(보유 상태 0)가 나면 그날 종가에 팔아 현금.
    현금은 rf(일간)를 받는다. 비용: 바뀐 비중 × BASKET_COST (buy_cost=False 면 처음 사는 비용은 뺌,
    trade_at_end=False 면 end 가 교체일이어도 그날 교체는 하지 않음 — 방금 끝난 기간 성과용).
    반환: value(끝 가치, 시작 1), trades(교체 횟수, 시작 포함), sells[(종목, 날짜)], w(끝 비중)"""
    C, H = _frames(reports)
    w, _, _ = _basket_at(reports, start)
    value = 1.0 - (sum(w.values()) * BASKET_COST if buy_cost else 0.0)
    trades, sells = 1, []
    if C.empty:
        return {"value": value, "trades": trades, "sells": sells, "w": w}
    s0, s1 = pd.Timestamp(start), pd.Timestamp(end)
    prev = C.loc[:s0].iloc[-1] if len(C.loc[:s0]) else C.iloc[0]
    for d in C.index[(C.index > s0) & (C.index <= s1)]:
        c = C.loc[d]
        cash = 1.0 - sum(w.values())
        grown = {t: x * (c[t] / prev[t]) if pd.notna(c.get(t)) and pd.notna(prev.get(t)) and prev[t] else x
                 for t, x in w.items()}
        tot = sum(grown.values()) + cash * (1 + _rf_on(rf, d))
        w = {t: x / tot for t, x in grown.items()}
        day = d.date()
        if day == pd.Timestamp(end).date() and not trade_at_end:
            pass
        elif day in rebalance_days(day):
            nw, _, _ = _basket_at(reports, day)
            tot -= sum(abs(nw.get(t, 0.0) - w.get(t, 0.0)) for t in set(nw) | set(w)) * BASKET_COST
            w, trades = nw, trades + 1
        else:
            out = [t for t in w if H.loc[d].get(t) == 0]
            for t in out:
                tot -= w.pop(t) * BASKET_COST
                sells.append((t, day))
        value *= tot
        prev = c
    return {"value": value, "trades": trades, "sells": sells, "w": w}


def _entry_dates(reports: list[TickerReport], members, start) -> dict:
    """start 교체일 바구니의 종목들이 끊기지 않고 들어 있던 가장 이른 교체일 (편입일).
    교체 사이에 규칙 매도로 판 적이 있으면 거기서 끊는다.

    반환: 종목 → (편입일, 데이터가 모자라 더 거슬러 가지 못했는지)."""
    first = min((r.sig["hist"].index[0].date() for r in reports if r.sig and r.sig.get("hist") is not None
                 and not r.sig["hist"].empty), default=start)
    _, H = _frames(reports)
    entry = {t: (start, False) for t in members}
    track, later = set(members), start
    while track:
        d = prev_rebalance(later)
        if d < first:
            entry.update({t: (entry[t][0], True) for t in track})
            break
        w, _, _ = _basket_at(reports, d)
        between = H.loc[(H.index > pd.Timestamp(d)) & (H.index < pd.Timestamp(later))]
        for t in list(track):
            if t in w and (t not in between or bool((between[t].dropna() == 1).all())):
                entry[t] = (d, False)
            else:
                track.discard(t)
        later = d
    return entry


def basket_ytd(reports: list[TickerReport], today, bench: str = "SPY", rf: pd.Series | None = None) -> dict | None:
    """올해 누적 수익률: 작년 마지막 거래일 종가부터 규칙대로 운용했다고 가정 (_simulate)."""
    start = month_end(today.replace(year=today.year - 1, month=12, day=1))
    hists = [r.sig["hist"] for r in reports if r.sig and r.sig.get("hist") is not None and not r.sig["hist"].empty]
    if not hists or min(h.index[0].date() for h in hists) > start:
        return None
    sim = _simulate(reports, start, today, rf)
    c_s, c_t = _close_at(reports, start), _close_at(reports)
    bench_ret = c_t[bench] / c_s[bench] - 1 if bench in c_s and bench in c_t else float("nan")
    return {"start": start, "ret": sim["value"] - 1.0, "bench": bench, "bench_ret": bench_ret,
            "trades": sim["trades"], "sells": len(sim["sells"])}


def basket_state(reports: list[TickerReport], rf: pd.Series | None = None) -> dict | None:
    """현재 바구니와 직전 바구니 대비 변화. 오늘 종가가 교체일(15일·월말)이면 '오늘 교체'.
    reports 에는 지난해 빅테크 칸 종목(메일 표에는 안 나오는 것)도 넣어야 지난 기간 계산이 맞다."""
    if not any(r.sig and r.sig.get("hist") is not None for r in reports):
        return None
    today = max(r.date for r in reports).date()
    rebalance = today in rebalance_days(today)
    base = today if rebalance else prev_rebalance(today)
    prev = prev_rebalance(base)
    w, sc, hd = _basket_at(reports, base)
    w0, _, _ = _basket_at(reports, prev)
    if not sc:
        return None
    now = {r.ticker: r.sig for r in reports if r.sig}
    # 수익률: 평소에는 이번 기간(지난 교체일 종가 → 오늘), 교체일에는 방금 끝난 기간(지난 바구니 기준)
    sim = _simulate(reports, prev, base, rf, buy_cost=False, trade_at_end=False) if rebalance else \
        _simulate(reports, base, today, rf, buy_cost=False)
    sold = dict(sim["sells"])
    c_now, c_base, c_prev = _close_at(reports), _close_at(reports, base), _close_at(reports, prev)
    out = []
    for t in sorted(set(w0) - set(w)):
        if t in sold:
            why = f"규칙상 매도 ({_md(sold[t])} 현금화)"
        elif not eligible(t, base):
            why = "빅테크 칸에서 빠짐"
        elif not hd.get(t):
            why = "규칙상 매도"
        else:
            why = f"점수 {sc[t]:.0f} < {BASKET_N:g}" if sc.get(t, 0) < BASKET_N else f"순위 밖 (점수 {sc[t]:.0f})"
        out.append((t, why))
    # 편입 후 수익률: 바구니에 처음 들어온 교체일 종가 → 오늘 종가 (판 종목은 판 날 종가까지)
    ent = _entry_dates(reports, set(w), base)
    ent0 = _entry_dates(reports, set(w0) - set(w), prev)
    c_ent = {t: _close_at(reports, d).get(t) for t, (d, _) in {**ent, **ent0}.items()}

    def end_close(t: str) -> float | None:
        return _close_at(reports, sold[t]).get(t) if t in sold else c_now.get(t)

    def since(t: str, e: dict, in_basket: bool) -> dict:
        d, cut = e[t]
        # 교체일 새 바구니 종목은 오늘 종가까지 (지난 기간에 팔았다가 다시 들어온 종목은 편입일이 오늘이라 0%)
        c0, c1 = c_ent.get(t), c_now.get(t) if in_basket and rebalance else end_close(t)
        return {"since": d, "since_cut": cut, "ret_hold": c1 / c0 - 1 if c0 and c1 else float("nan")}

    def period_ret(t: str, c0: dict) -> float:
        c1 = end_close(t)
        return c1 / c0[t] - 1 if c1 and t in c0 and c0[t] else float("nan")
    rows = [{"ticker": t, "weight": w[t], "score": sc[t], "now": now.get(t, {}).get("score_s", float("nan")),
             "new": t not in w0, "held_now": bool(now.get(t, {}).get("held", True)),
             "sold": None if rebalance else sold.get(t),
             "ret": (period_ret(t, c_prev) if t in w0 else float("nan")) if rebalance else period_ret(t, c_base),
             **since(t, ent, True)}
            for t in sorted(w, key=lambda x: (-sc[x], x))]
    out = [(t, why, since(t, ent0, False)) for t, why in out]
    cash = max(0.0, 1.0 - sum(w.values()))
    return {"today": today, "rebalance": rebalance, "base": base, "prev": prev,
            "next": next_rebalance(today), "rows": rows, "out": out, "cash": cash,
            "cash_now": cash if rebalance else max(0.0, 1.0 - sum(sim["w"].values())),
            "sold_today": [] if rebalance else [t for t, d in sold.items() if d == today],
            "month_ret": sim["value"] - 1.0, "rf": rf is not None and not rf.empty,
            "name": {r.ticker: r.name for r in reports}, "ytd": basket_ytd(reports, today, rf=rf)}


def _md(d) -> str:
    return f"{d.month}/{d.day}"


def _since_text(x: dict) -> str:
    return f"{'~' if x['since_cut'] else ''}{_md(x['since'])}~"


def _ret_label(b: dict) -> str:
    return (f"지난 바구니 수익률 ({_md(b['prev'])} → {_md(b['base'])} 종가)" if b["rebalance"]
            else f"이번 기간 수익률 ({_md(b['base'])} 종가 → {_md(b['today'])} 종가)")


def _ytd_text(b: dict) -> str | None:
    y = b.get("ytd")
    if not y:
        return None
    vs = "" if pd.isna(y["bench_ret"]) else f" · 같은 기간 {y['bench']} {y['bench_ret']:+.1%}"
    sells = f"·교체 사이 매도 {y['sells']}회" if y.get("sells") else ""
    return (f"올해 누적 수익률 ({_md(y['start'])} → {_md(b['today'])} 종가, 교체 {y['trades']}회{sells}): "
            f"{y['ret']:+.1%}{vs}")


def _pct_html(x: float) -> str:
    if pd.isna(x):
        return "–"
    return f"<span style='color:{_price_color(x)}'>{x:+.1%}</span>"


def _sold_text(b: dict, x: dict) -> str:
    """교체 사이 규칙 매도 표시."""
    if x["sold"] == b["today"]:
        return "🔻 오늘 규칙 매도 신호 → 다음 거래일에 팔고 현금으로"
    return f"🔻 {_md(x['sold'])} 규칙 매도 → 현금"


def _cash_text(b: dict) -> str:
    sold = [x for x in b["rows"] if x.get("sold")]
    return f"현금 {b['cash'] * 100:.0f}%" + (f" → 교체 사이 매도 뒤 {b['cash_now'] * 100:.0f}%" if sold else "")


def _cash_note(b: dict) -> str:
    return "배당 포함, 현금은 단기국채 금리" if b.get("rf") else "배당 포함, 현금 수익 0으로 계산"


def _basket_lines(b: dict) -> list[str]:
    new = [x["ticker"] for x in b["rows"] if x["new"]]
    if b["rebalance"]:
        head = (f"오늘({_md(b['base'])}) 종가 기준 교체 — 다음 거래일에 아래 비중으로 맞추기 · "
                f"신규 {', '.join(new) or '없음'} · 제외 {', '.join(t for t, *_ in b['out']) or '없음'}")
    else:
        head = f"{_md(b['base'])} 종가 기준 바구니 · 다음 교체 {_md(b['next'])} 종가 (다음 날 아침 메일)"
    lines = [head]
    for x in b["rows"]:
        tag = " (신규)" if x["new"] and b["rebalance"] else ""
        new_today = x["new"] and b["rebalance"]
        if x.get("sold"):
            ret = f" · 매도까지 {x['ret']:+.1%}" if pd.notna(x["ret"]) else ""
            lines.append(f"{x['ticker']} {x['weight'] * 100:.1f}% · 점수 {x['score']:.0f} → 오늘 {x['now']:.0f} · "
                         f"{_sold_text(b, x)}{ret}")
            continue
        ret = "" if new_today or pd.isna(x["ret_hold"]) else f" · 편입 후 {x['ret_hold']:+.1%} ({_since_text(x)})"
        lines.append(f"{x['ticker']} {x['weight'] * 100:.1f}% · 점수 {x['score']:.0f}"
                     + ("" if b["rebalance"] else f" → 오늘 {x['now']:.0f}") + ret + tag)
    if b["rebalance"]:
        lines += [f"제외 {t}: {why}" + ("" if pd.isna(h["ret_hold"]) else
                                       f" · 보유 기간 {h['ret_hold']:+.1%} ({_since_text(h)}"
                                       f"{_md(b['base']) if '현금화' not in why else ''})")
                  for t, why, h in b["out"]]
    lines.append(_cash_text(b))
    lines.append(f"{_ret_label(b)}: {b['month_ret']:+.1%} ({_cash_note(b)})")
    if _ytd_text(b):
        lines.append(_ytd_text(b) + " (규칙대로 운용했다고 가정, 비용 0.05% 반영)")
    return lines


def _basket_html(b: dict) -> str:
    e = html.escape
    parts = [f"<h3>🧺 바구니 <span class='m' style='font-weight:normal;font-size:12px'>{e(BASKET_RULE)}</span></h3>"]
    new = [x["ticker"] for x in b["rows"] if x["new"]]
    if b["rebalance"]:
        parts.append(f"<div class='box sig'><b>오늘({_md(b['base'])}) 종가 기준 교체</b> — 다음 거래일에 아래 비중으로 맞추세요 · "
                     f"신규 <b>{e(', '.join(new) or '없음')}</b> · 제외 <b>{e(', '.join(t for t, *_ in b['out']) or '없음')}</b></div>")
    else:
        parts.append(f"<p class='m' style='margin:2px 0 4px;font-size:13px'>{_md(b['base'])} 종가 기준 · "
                     f"다음 교체 {_md(b['next'])} 종가 (다음 날 아침 메일)</p>")
    parts.append(f"<p style='margin:2px 0 4px;font-size:13px'>{e(_ret_label(b))}: <b>{_pct_html(b['month_ret'])}</b> "
                 f"<span class='s10'>{e(_cash_note(b))}</span></p>")
    y = b.get("ytd")
    if y:
        vs = "" if pd.isna(y["bench_ret"]) else f" <span class='m'>· 같은 기간 {e(y['bench'])} {_pct_html(y['bench_ret'])}</span>"
        parts.append(f"<p style='margin:2px 0 4px;font-size:13px'>올해 누적 수익률 ({_md(y['start'])} → {_md(b['today'])} 종가): "
                     f"<b>{_pct_html(y['ret'])}</b>{vs}<br><span class='s10'>교체 {y['trades']}회·교체 사이 매도 "
                     f"{y['sells']}회를 규칙대로 했다고 가정 · 비용 0.05% 반영</span></p>")
    parts.append("<table class='tb'><tr><th class='s'>종목</th><th class='s'>비중</th>"
                 f"<th class='s'>점수<br>{_md(b['base'])}</th>" + ("" if b["rebalance"] else "<th class='s'>점수<br>오늘</th>")
                 + "<th class='s'>비고<br>편입 후 수익률</th></tr>")
    for x in b["rows"]:
        if b["rebalance"] and x["new"]:
            note = "🆕 신규"
        elif x.get("sold"):
            short = "🔻 오늘 매도 신호<br>내일 팔고 현금" if x["sold"] == b["today"] else f"🔻 {_md(x['sold'])} 매도<br>→ 현금"
            note = (f"<span class='wr'>{short}</span>"
                    + ("" if pd.isna(x["ret"]) else f"<br><span class='s10'>매도까지 {_pct_html(x['ret'])}</span>"))
        else:
            note = f"<b>{_pct_html(x['ret_hold'])}</b><br><span class='s10'>{_since_text(x)}</span>"
        parts.append(f"<tr><td class='l' style='font-size:13px'><b>{e(x['ticker'])}</b><br><span class='m nm'>"
                     f"{e(b['name'].get(x['ticker'], ''))}</span></td><td class='c'><b>{x['weight'] * 100:.1f}%</b></td>"
                     f"<td class='c'>{_sc_badge(x['score'])}</td>"
                     + ("" if b["rebalance"] else f"<td class='c'>{_sc_badge(x['now'])}</td>")
                     + f"<td class='c' style='font-size:12px'>{note}</td></tr>")
    if b["rebalance"]:
        for t, why, h in b["out"]:
            parts.append(f"<tr><td class='l' style='font-size:13px;color:#999'><s>{e(t)}</s></td><td class='c m'>0%</td>"
                         f"<td class='c m' colspan='2' style='font-size:12px'>제외 · {e(why)}"
                         + ("" if pd.isna(h["ret_hold"]) else
                            f"<br>보유 기간 {_pct_html(h['ret_hold'])} ({_since_text(h)}"
                            f"{_md(b['base']) if '현금화' not in why else ''})") + "</td></tr>")
    cash_now = "" if abs(b["cash_now"] - b["cash"]) < 0.005 else f"<br><span class='s10'>매도 뒤 {b['cash_now'] * 100:.0f}%</span>"
    parts.append(f"<tr><td class='l m' style='font-size:13px'>현금</td><td class='c'><b>{b['cash'] * 100:.0f}%</b>{cash_now}</td>"
                 f"<td colspan='{2 if b['rebalance'] else 3}'></td></tr></table>")
    return "".join(parts)


RULE_LINE = (f"점수 = 방향 성향 6개의 가중 합 (🟢 +1 · 🟠 0 · 🔴 −1, 장기 ×2 · 모멘텀·거래량 ×0.5 · 나머지 ×1)을 "
             f"0~100으로 환산 (50 = 중립). 진입: 3일 평균 {RULE.entry:g} 이상 + 200일선 위 + 과열·고변동 아님 · "
             f"퇴출: 200일선 아래 + 3일 평균 {RULE.exit:g} 이하 (200일선 위에서는 눌림으로 보고 보유) · "
             f"손절 참고가 = 종가 − {RULE.stop_atr:g}×ATR(14).")
VALID_LINE = ("검증(29종목, 2006~2026, 종목 중앙값): 계속 보유 대비 최대낙폭 −59%→−34%, 연수익 8.5%→6.9%, 매매 연 1.3회. "
              "점수는 수익 예측이 아니라 하락 위험을 줄이는 용도. 매매 권유가 아님.")
_SIG_LABEL = {"buy": ("bg", "🟢 매수"), "sell": ("br", "🔴 매도"), "blocked": ("bn", "⚠️ 진입 보류"),
              "near_buy": ("bg", "🟢 진입 임박"), "near_exit": ("bo", "🟠 청산 임박"),
              "pullback": ("bn", "🟠 눌림 (보유 유지)")}


def _signal_items(reports: list[TickerReport]) -> list[tuple[str, TickerReport | None, str]]:
    """(종류, 종목, 설명). 신규 매수·매도가 없으면 한 줄로 줄인다."""
    g = signal_groups(reports)
    items = [(k, r, t) for k in ("buy", "sell") for r, t in g[k]] or [("none", None, "오늘 신규 매수·매도 신호 없음")]
    return items + [(k, r, t) for k in ("blocked", "near_buy", "near_exit", "pullback") for r, t in g[k]]


def _signal_lines(reports: list[TickerReport]) -> list[str]:
    return [t if r is None else f"{_SIG_LABEL[k][1]} {r.ticker} — {t}" for k, r, t in _signal_items(reports)]


def _exit_room(g: dict) -> tuple[str, str]:
    """(200일선 대비 종가, 점수 여유). 퇴출 = 200일선 아래 + 3일 평균 40 이하."""
    ma = g.get("ma200", np.nan)
    trend = "200일선 –" if pd.isna(ma) else (f"200일선 {g['close'] / ma - 1:+.1%}" if g["close"] >= ma else "200일선 아래")
    room = g["score_s"] - RULE.exit
    return trend, (f"점수 여유 {room:.0f}" if room > 0 else "점수는 퇴출선 아래")


def _holding_text(r: TickerReport) -> str:
    g = r.sig
    trend, room = _exit_room(g)
    return f"{r.ticker} {g['score_s']:.0f} · 종가 {g['close']:,.2f} · {_stop_text(g)} · {trend} · {room}"


def _sc_badge(v: float) -> str:
    if v is None or pd.isna(v):
        return "<span class='b bn'>–</span>"
    return f"<span class='b b{score_color(v)}'>{v:.0f}</span>"


def _signal_html(reports: list[TickerReport]) -> str:
    e = html.escape
    rows = []
    for k, r, t in _signal_items(reports):
        if r is None:
            rows.append(f"<li><b>{e(t)}</b></li>")
        else:
            cls, label = _SIG_LABEL[k]
            rows.append(f"<li><span class='b {cls}'>{label}</span> <b>{e(r.ticker)}</b> "
                        f"<span class='m'>{e(r.name)}</span> — {e(t)}</li>")
    parts = ["<div class='box sig'><b>오늘의 매수·매도 신호</b><ul>" + "".join(rows) + "</ul>"]
    hs = holdings(reports)
    if hs:
        parts.append(f"<p style='margin:10px 0 2px'><b>규칙상 보유 구간 {len(hs)}종목</b> "
                     f"<span class='m' style='font-size:11px'>손절 참고가 = 종가 − {RULE.stop_atr:g}×ATR(14)</span></p>"
                     "<table class='tb'><tr><th class='s'>종목</th><th class='s'>점수<br>3일평균</th><th class='s'>종가</th>"
                     "<th class='s'>손절<br>참고가</th><th class='s'>매도 조건까지<br>200일선 · 점수</th></tr>")
        for r in hs:
            g = r.sig
            stop = (f"{g['stop']:,.2f}<br><span class='s10'>{g['stop'] / g['close'] - 1:+.1%}</span>"
                    if pd.notna(g["stop"]) else "–")
            ma, room = g.get("ma200", np.nan), g["score_s"] - RULE.exit
            trend = ("–" if pd.isna(ma) else f"{g['close'] / ma - 1:+.1%}" if g["close"] >= ma
                     else "<span class='wr'>200일선 아래</span>")
            room = f"여유 {room:.0f}" if room > 0 else "<span class='wr'>퇴출선 아래</span>"
            parts.append(f"<tr><td class='l' style='font-size:13px'><b>{e(r.ticker)}</b><br><span class='m nm'>{e(r.name)}</span></td>"
                         f"<td class='c'>{_sc_badge(g['score_s'])}</td><td class='c' style='font-size:12px'>{g['close']:,.2f}</td>"
                         f"<td class='c' style='font-size:12px'>{stop}</td>"
                         f"<td class='c' style='font-size:12px'>{trend}<br><span class='s10'>{room}</span></td></tr>")
        parts.append("</table>")
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------- 성향 판정

# (성향, 항목들, 종류) — dir: 방향(↑ ↓ ↔), risk: 위험(⚠️ 또는 정상)
GROUPS = [
    ("장기 추세", ["200일선", "배열"], "dir"),
    ("중기 추세", ["구름", "전환/기준"], "dir"),
    ("모멘텀", ["MACD", "RSI"], "dir"),
    ("추세 강도", ["ADX"], "dir"),
    ("거래량", ["VR", "OBV"], "dir"),
    ("상대강도", ["상대강도"], "dir"),
    ("변동성", ["변동성"], "risk"),
    ("과열", ["이격도"], "risk"),
]
SHORT = {"장기 추세": "장기", "중기 추세": "중기", "모멘텀": "모멘텀", "추세 강도": "추세",
         "거래량": "거래량", "상대강도": "상대", "변동성": "변동성", "과열": "과열"}


def verdict(checks: list[Check], group: str) -> str:
    """성향 판정: 방향 성향은 ✅ +1, ❌ −1 합계로 ↑/↓/↔, 위험 성향은 ⚠️태그 또는 태그."""
    items = [c for c in checks if c.group == group]
    kind = next(k for g, _, k in GROUPS if g == group)
    if not items or all(c.na for c in items):
        return "–"
    if kind == "risk":
        c = items[0]
        return f"⚠️{c.tag}" if c.status == WARN else c.tag
    score = sum(1 if c.status == UP else -1 if c.status == DOWN else 0 for c in items)
    if score > 0:
        return "↑"
    if score < 0:
        return "↓"
    tags = [c.tag for c in items if c.tag]
    return tags[0] if len(items) == 1 and tags else "↔"


# ---------------------------------------------------------------- 신호등 표시 (초록 / 주황 / 빨강)

DOT = {"g": "🟢", "o": "🟠", "r": "🔴", "n": "⚪"}
BG = {"g": "#c8e6c9", "o": "#ffe0b2", "r": "#ffcdd2", "n": "#eeeeee"}
FG = {"g": "#1b5e20", "o": "#e65100", "r": "#b71c1c", "n": "#757575"}
CHANGED_BG = "#fff59d"  # 전일과 색이 바뀐 칸
_UP_LABEL = {"추세 강도": "상승", "상대강도": "강함"}
_DN_LABEL = {"추세 강도": "하락", "상대강도": "약함"}


def badge(group: str, v: str) -> tuple[str, str]:
    """판정 → (색 키, 한글 라벨). 초록 = 강세·정상, 주황 = 혼조·주의, 빨강 = 약세."""
    if v == "↑":
        return "g", _UP_LABEL.get(group, "강세")
    if v == "↓":
        return "r", _DN_LABEL.get(group, "약세")
    if v == "↔":
        return "o", "혼조"
    if v == "–":
        return "n", "–"
    if v.startswith("⚠️"):
        return "o", v.replace("⚠️", "")
    if v in ("횡보", "약함"):
        return "o", v
    return "g", v  # 보통·저변동·정상


def score_color(score: float) -> str:
    if pd.isna(score):
        return "n"
    return "g" if score >= RULE.entry else "r" if score <= RULE.exit else "o"


def badge_text(group: str, v: str) -> str:
    color, label = badge(group, v)
    return f"{DOT[color]}{label}"


_BADGE_CSS = (".ck .b{display:inline-block;padding:2px 4px;border-radius:4px;font-weight:bold;white-space:nowrap}"
              + "".join(f".ck .b.b{k}{{background:{BG[k]};color:{FG[k]}}}" for k in BG)
              + ".ck .sig{border-left-color:#1565c0;background:#f3f8fe}.ck .sig li{margin:3px 0}"
              + ".ck .nm{font-size:10px;white-space:normal;overflow-wrap:anywhere}.ck th.s{font-size:10px;padding:4px 1px}"
              + ".ck td.f{padding:3px 1px;text-align:center;font-size:10px;line-height:1.15;font-weight:bold}"
              + ".ck td.f i{font-style:normal;font-size:15px}.ck table.tb td{padding:3px 2px}"
              + "".join(f".ck td.f{k}{{color:{FG[k]}}}" for k in FG)
              + f".ck td.x{{background:{CHANGED_BG}}}.ck .s10{{font-size:10px;color:#777}}")


# ---------------------------------------------------------------- 출력

LEGEND = ("🟢 강세·정상 · 🟠 혼조·주의 · 🔴 약세 · ⚪ 해당 없음 · 노란 칸 = 전일과 색이 바뀐 성향 · 과열 칸의 🟠은 과열 또는 과매도 · "
          "머리글 ×2·×½ = 점수 배점. "
          "장기 = 200일선·이평 배열 · 중기 = 일목 구름·전환/기준 · 모멘텀 = MACD 시그널·RSI 50 · 추세 = ADX≥25와 DMI 방향 · "
          "거래량 = VR 1년 백분위 50%·OBV 20일 증감 · 상대 = 벤치마크(SPY, 비트코인·반도체는 QQQ) 대비 비율의 50일 평균 · "
          "변동성 = 20일 변동성 1년 상위 20%면 🟠 · 과열 = 50일선 이격도 1년 상위 5%(과열)·하위 5%(과매도)면 🟠.")


def _v(r: TickerReport, g: str) -> str:
    return verdict(r.checks, g)


def _status(r: TickerReport, name: str) -> str | None:
    return next((c.status for c in r.checks if c.name == name), None)


def _headline(reports: list[TickerReport]) -> list[str]:
    """위험 표시: 과열·고변동·200일선 아래."""
    rules = [
        ("🟠 과열 주의 (50일선 이격 1년 상위 5%)", lambda r: _v(r, "과열") == "⚠️과열"),
        ("🟠 고변동 (변동성 1년 상위 20%)", lambda r: _v(r, "변동성") == "⚠️고변동"),
        ("🔴 200일선 아래", lambda r: _status(r, "200일선") == DOWN),
    ]
    lines = []
    for label, cond in rules:
        names = [r.ticker for r in reports if cond(r)]
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return lines


def market_lines(reports: list[TickerReport], meta: dict) -> list[str]:
    """시장 폭 · 원/달러 · 다가오는 일정."""
    sigs = [r.sig for r in reports if r.sig]
    n200 = sum(_status(r, "200일선") == UP for r in reports)
    line = f"{len(reports)}종목 중 200일선 위 {n200} · 규칙상 보유 구간 {sum(g['held'] for g in sigs)}"
    if sigs:
        line += f" · 평균 점수(3일) {np.mean([g['score_s'] for g in sigs]):.0f}"
    lines = [line]
    fx = meta.get("fx")
    if fx:
        m1 = f" · 1개월 {fx['m1']:+.1%}" if pd.notna(fx.get("m1")) else ""
        lines.append(f"원/달러 {fx['rate']:,.1f}원 (전일 대비 {fx['d1']:+.1%}{m1})")
    if "upcoming" in meta:
        ev = meta["upcoming"]
        lines.append("다가오는 일정 (2주): " + (" · ".join(f"{d} {label}" for d, label in ev) if ev else "없음"))
    return lines


def _verdict_changes(r: TickerReport) -> list[tuple[str, str, str]]:
    """전일과 색이 달라진 성향 (성향, 전일 판정, 오늘 판정). 같은 색 안의 변화(횡보→약함 등)는 뺀다."""
    prev = r.prev_verdicts()
    if prev is None:
        return []
    now = r.verdicts()
    return [(g, prev[g], now[g]) for g, _, _ in GROUPS if badge(g, prev[g])[0] != badge(g, now[g])[0]]


def _score_delta(r: TickerReport) -> float:
    if r.prev_score is None or pd.isna(r.prev_score) or pd.isna(r.score):
        return 0.0
    return r.score - r.prev_score


def _changed_values(r: TickerReport, g: str) -> str:
    """성향 안에서 전일과 판정이 바뀐 항목의 현재 수치."""
    changed = {now.name for _, now in r.changes}
    return ", ".join(c.short or "–" for c in r.checks if c.group == g and c.name in changed)


def change_lines(reports: list[TickerReport]) -> tuple[str, list[str]]:
    """(요약, 줄 목록). 당일 점수 변화가 큰 종목 TOP_CHANGES개는 자세히, 나머지는 한 줄로."""
    if not any(r.prev_checks for r in reports):
        return "직전 거래일 비교 불가", []
    rows = [(r, vc) for r in reports if (vc := _verdict_changes(r))]
    rows.sort(key=lambda t: (-abs(_score_delta(t[0])), -len(t[1])))
    n = sum(len(vc) for _, vc in rows)
    better = sum(_score_delta(r) > 0 for r, _ in rows)
    worse = sum(_score_delta(r) < 0 for r, _ in rows)
    lines = []
    for r, vc in rows[:TOP_CHANGES]:
        parts = []
        for g, a, b in vc:
            vals = _changed_values(r, g)
            parts.append(f"{SHORT[g]} {badge_text(g, a)}→{badge_text(g, b)}" + (f" ({vals})" if vals else ""))
        sc = f" (당일 점수 {r.prev_score:.0f}→{r.score:.0f})" if round(_score_delta(r)) else ""
        lines.append(f"{r.ticker}{sc}: " + ", ".join(parts))
    if rows[TOP_CHANGES:]:
        lines.append("그 밖에: " + " · ".join(
            f"{r.ticker} " + ", ".join(f"{SHORT[g]} {DOT[badge(g, a)[0]]}→{DOT[badge(g, b)[0]]}" for g, a, b in vc)
            for r, vc in rows[TOP_CHANGES:]))
    return f"색이 바뀐 성향 {n}건 · 개선 {better}종목 · 악화 {worse}종목", lines


def recent_signals(reports: list[TickerReport]) -> list[tuple[TickerReport, dict]]:
    """최근 RECENT_DAYS 거래일의 매수·매도 신호, 최신순."""
    items = [(r, s) for r in reports if r.sig for s in r.sig.get("recent", [])]
    return sorted(items, key=lambda t: t[1]["date"], reverse=True)


def recent_summary(items: list[tuple[TickerReport, dict]]) -> str:
    def part(kind: str, label: str, good: str, is_good) -> str:
        xs = [s["ret"] for _, s in items if s["event"] == kind]
        if not xs:
            return f"{label} 0건"
        return f"{label} {len(xs)}건: 신호 뒤 평균 {np.mean(xs):+.1%} ({good} {sum(is_good(x) for x in xs)}건)"
    return (part("BUY", "매수", "오른 것", lambda x: x > 0) + " · "
            + part("SELL", "매도", "매도 뒤 더 내린 것", lambda x: x < 0))


def _recent_text(r: TickerReport, s: dict) -> str:
    d = s["date"]
    label = "매수" if s["event"] == "BUY" else "매도"
    return f"{d.month}/{d.day} {r.ticker} {label} {s['close']:,.2f} → 지금 {s['ret']:+.1%}"


def quality_lines(reports: list[TickerReport], failed: list[str], notes: list[str] | None = None) -> list[str]:
    lines = [f"{r.ticker}: {i}" for r in reports for i in r.issues]
    lines += [f"{f} — 불러오기 실패" for f in failed]
    return lines + list(notes or [])


def _basket(reports: list[TickerReport], meta: dict) -> dict | None:
    """메일용 바구니. meta['basket_extra']: 메일 표에는 안 나오지만 지난 기간 계산에 필요한 종목(지난해 빅테크 칸),
    meta['rf']: 현금 일간 수익률 (단기국채 금리 ÷ 252)."""
    return basket_state(reports + list(meta.get("basket_extra") or []), rf=meta.get("rf"))


def make_subject(reports: list[TickerReport], meta: dict) -> str:
    date = max(r.date for r in reports).date()
    q = quality_lines(reports, meta.get("failed", []), meta.get("notes"))
    if meta.get("mode") == "holiday":
        return f"[휴장] 미장 체크리스트 · {meta['checked']} {meta['holiday']} · 다음 개장 {meta['next_open']}"
    if meta.get("mode") == "delayed":
        return f"[데이터 지연] 미장 체크리스트 · 기대 {meta['target']}, 수신 {date}"
    g = signal_groups(reports)

    def names(k: str) -> str:
        return ",".join(r.ticker for r, _ in g[k])
    parts = [f"매수 {names('buy') or '없음'}", f"매도 {names('sell') or '없음'}"]
    b = _basket(reports, meta)
    if b and b["rebalance"]:
        parts.append(f"바구니 교체 +{sum(x['new'] for x in b['rows'])} −{len(b['out'])}")
    if b and b["sold_today"]:
        parts.append(f"바구니 매도 {','.join(b['sold_today'])}")
    if g["blocked"]:
        parts.append(f"보류 {names('blocked')}")
    near = [r.ticker + "↑" for r, _ in g["near_buy"]] + [r.ticker + "↓" for r, _ in g["near_exit"]]
    if near:
        parts.append("임박 " + " ".join(near))
    if g["pullback"]:
        parts.append(f"눌림 {names('pullback')}")
    subject = f"[미장] {date.month}/{date.day} · " + " · ".join(parts)
    if q:
        subject = "[점검 필요] " + subject
    return subject if len(subject) <= 90 else subject[:89] + "…"


def _compact_row(r: TickerReport) -> str:
    """텍스트 본문용 한 줄: 3일 평균 점수 + 성향 색."""
    sc = r.sig["score_s"] if r.sig else r.score
    dots = [DOT[badge(g, _v(r, g))[0]] for g, _, kind in GROUPS if kind == "dir"]
    risk = [DOT[badge(g, _v(r, g))[0]] for g, _, kind in GROUPS if kind == "risk"]
    return f"{r.ticker} {'–' if pd.isna(sc) else f'{sc:.0f}'}: {''.join(dots)} / {''.join(risk)}"


def _sections(reports: list[TickerReport], meta: dict) -> list[tuple[str, list[str]]]:
    """본문 구역 (제목, 줄 목록) — 텍스트·마크다운 공용. 휴장일에는 신호·변화·기록을 뺀다."""
    date = max(r.date for r in reports).date()
    holiday = meta.get("mode") == "holiday"
    q = quality_lines(reports, meta.get("failed", []), meta.get("notes"))
    secs = [("데이터 점검", q or [f"이상 없음 ({len(reports)}종목, 기준일 {date})"])]
    if not holiday:
        secs.append(("오늘의 매수·매도 신호", _signal_lines(reports)))
        hs = holdings(reports)
        if hs:
            secs.append((f"규칙상 보유 구간 {len(hs)}종목 (3일 평균 · 종가 · 손절 참고가 · 종가의 200일선 대비 · 점수 여유)",
                         [_holding_text(r) for r in hs]))
        b = _basket(reports, meta)
        if b:
            secs.append((f"바구니 (규칙 보유 중 {BASKET_N:g}점 이상 상위 {BASKET_K}개 · {100 / BASKET_K:g}%씩 · 15일·월말 교체 · "
                         "교체 사이 규칙 매도는 바로 현금)",
                         _basket_lines(b)))
    look = market_lines(reports, meta) + _headline(reports)
    if meta.get("calendar_note"):
        look.append("참고: " + meta["calendar_note"])
    secs.append((f"직전 거래일({date}) 요약" if holiday else "한눈에 보기", look))
    if not holiday:
        summary, cl = change_lines(reports)
        secs.append((f"전일 대비 변화 — {summary}", cl or ["없음"]))
        secs.append(("성향별 판정 (3일 평균 점수: 장기 중기 모멘텀 추세 거래량 상대 / 변동성 과열)",
                     [_compact_row(r) for r in reports]))
    secs.append((f"{'직전 거래일 ' if holiday else ''}단기 반등 후보 ({REBOUND_NOTE})", rebound_lines(reports) or ["없음"]))
    if not holiday:
        items = recent_signals(reports)
        secs.append((f"최근 {RECENT_DAYS}거래일 신호 기록",
                     [recent_summary(items)] + [_recent_text(r, s) for r, s in items[:RECENT_ROWS]] if items else ["없음"]))
    return secs


def _title_lines(reports: list[TickerReport], meta: dict) -> list[str]:
    date = max(r.date for r in reports).date()
    if meta.get("mode") == "holiday":
        return [f"미국장 휴장 안내: {meta['checked']} {meta['holiday']}", f"다음 거래일: {meta['next_open']}",
                f"새 종가가 없어 아래는 직전 거래일({date}) 기준이며 변화는 없습니다."]
    out = [f"미국장 지표 체크리스트 ({date} 종가 기준)"]
    if meta.get("mode") == "delayed":
        out.append(f"주의: 데이터 지연 (기대 기준일 {meta['target']}, 받은 데이터 {date}). 아래는 받은 데이터 기준입니다.")
    return out


def render_markdown(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    title = _title_lines(reports, meta)
    out = [f"# {title[0]}", ""] + [f"> {x}" for x in title[1:]] + ([""] if title[1:] else [])
    for sec, items in _sections(reports, meta):
        if sec.startswith("성향별 판정"):
            header = ["종목", "점수(3일)", "종가", "등락", "고점대비"] + [SHORT[g] for g, _, _ in GROUPS]
            out += ["## 성향별 판정", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            for r in reports:
                changed = {g for g, _, _ in _verdict_changes(r)}
                sc = r.sig["score_s"] if r.sig else r.score
                row = [r.ticker + (" ⚠️" if r.issues else ""),
                       "–" if pd.isna(sc) else f"{DOT[score_color(sc)]}{sc:.0f}",
                       f"{r.close:,.2f}", f"{r.change:+.1%}", f"{r.from_high:+.1%}"]
                row += [f"[{badge_text(g, _v(r, g))}]" if g in changed else badge_text(g, _v(r, g)) for g, _, _ in GROUPS]
                out.append("| " + " | ".join(row) + " |")
            out.append("")
            continue
        out += [f"## {sec}"] + [f"- {x}" for x in items] + [""]
    out += [LEGEND, "", RULE_LINE, "", f"{VALID_LINE} [근거 문서]({DOC_URL})"]
    return "\n".join(out)


def render_text(reports: list[TickerReport], meta: dict | None = None) -> str:
    """메일 일반 텍스트 본문 (마크다운 기호 없음)."""
    meta = meta or {}
    lines = _title_lines(reports, meta) + [""]
    for sec, items in _sections(reports, meta):
        lines += [sec] + [f"- {x}" for x in items] + [""]
    lines += [RULE_LINE, f"{VALID_LINE} 근거: {DOC_URL}"]
    return "\n".join(lines)


_CSS = ("<style>.ck{font-family:-apple-system,Segoe UI,Malgun Gothic,sans-serif;font-size:14px;color:#222}"
        ".ck table{border-collapse:collapse}.ck th{padding:5px 4px;background:#f2f2f2;border-bottom:2px solid #999;"
        "white-space:nowrap}.ck td{padding:3px 4px;border-bottom:1px solid #ddd;white-space:nowrap}"
        ".ck td.c{text-align:center}.ck td.l{text-align:left}.ck td.d{white-space:normal;font-size:13px}"
        ".ck .g{font-weight:bold;color:#555;padding-top:8px}.ck .m{color:#777}"
        ".ck .up{color:#2e7d32;font-weight:bold}.ck .dn{color:#c62828;font-weight:bold}"
        ".ck .wr{color:#e65100;font-weight:bold}"
        ".ck .box{border-left:4px solid #999;background:#fafafa;padding:6px 10px;margin:8px 0}"
        ".ck .warn{border-left-color:#e65100;background:#fff3e0}.ck .ok{border-left-color:#2e7d32;background:#f1f8e9}"
        ".ck h3{margin:14px 0 4px}.ck ul{margin:0;padding-left:20px}</style>")


def _ul(items: list[str], e) -> str:
    return "<ul>" + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul>"


def _price_color(x: float) -> str:
    return "#c62828" if x > 0 else "#1565c0" if x < 0 else "#222"


def _table_html(reports: list[TickerReport]) -> str:
    """성향별 판정 표: 점수(3일 평균, 당일) · 종가/등락/고점대비 · 성향 8개 (색 동그라미 + 짧은 라벨)."""
    e = html.escape
    parts = ["<table class='tb'><tr><th class='s'>종목</th><th class='s'>점수<br>3일평균</th><th class='s'>종가</th>"]
    wts = dict(RULE.weights)
    mark = {2.0: "×2", 0.5: "×½"}
    parts += [f"<th class='s'>{e(SHORT[g])}" + (f"<br><span style='font-weight:normal'>{mark[wts[g]]}</span>"
                                                 if wts.get(g) in mark else "") + "</th>" for g, _, _ in GROUPS]
    parts.append("</tr>")
    group = None
    for r in reports:
        if r.group and r.group != group:
            group = r.group
            parts.append(f"<tr><td colspan='{3 + len(GROUPS)}' class='g'>{e(group)}</td></tr>")
        flag = " ⚠️" if r.issues else ""
        parts.append(f"<tr><td class='l' style='font-size:13px'><b>{e(r.ticker)}</b>{flag}<br><span class='m nm'>{e(r.name)}</span></td>")
        g = r.sig
        if g:
            parts.append(f"<td class='c'>{_sc_badge(g['score_s'])}<br><span class='s10'>당일 {g['score']:.0f}</span></td>")
        else:
            parts.append(f"<td class='c'>{_sc_badge(r.score)}</td>")
        hi = f"<br><span class='s10'>고점{r.from_high:+.0%}</span>" if pd.notna(r.from_high) else ""
        parts.append(f"<td class='c' style='font-size:12px'>{r.close:,.2f}<br>"
                     f"<span style='color:{_price_color(r.change)}'>{r.change:+.1%}</span>{hi}</td>")
        changed = {gg for gg, _, _ in _verdict_changes(r)}
        for gg, _, _ in GROUPS:
            color, label = badge(gg, _v(r, gg))
            chg = " x" if gg in changed else ""
            # 색만으로 뜻이 갈리지 않는 칸은 동그라미만. 과열 칸의 🟠(과열/과매도)만 글자를 붙인다.
            tag = f"<br>{e(label)}" if gg == "과열" and color == "o" else ""
            parts.append(f"<td class='f f{color}{chg}'><i>{DOT[color]}</i>{tag}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _recent_html(reports: list[TickerReport]) -> str:
    e = html.escape
    items = recent_signals(reports)
    head = (f"<h3>최근 {RECENT_DAYS}거래일 신호 기록 <span class='m' style='font-weight:normal;font-size:12px'>"
            "신호일 종가 → 지금</span></h3>")
    if not items:
        return head + "<p class='m'>없음</p>"
    rows = []
    for r, s in items[:RECENT_ROWS]:
        cls, label = ("bg", "🟢 매수") if s["event"] == "BUY" else ("br", "🔴 매도")
        d = s["date"]
        rows.append(f"<tr><td class='c'>{d.month}/{d.day}</td><td class='l'><b>{e(r.ticker)}</b></td>"
                    f"<td class='c'><span class='b {cls}'>{label}</span></td><td class='c'>{s['close']:,.2f}</td>"
                    f"<td class='c'><span style='color:{_price_color(s['ret'])}'>{s['ret']:+.1%}</span></td></tr>")
    more = (f"<p class='m' style='font-size:12px;margin:2px 0'>외 {len(items) - RECENT_ROWS}건</p>"
            if len(items) > RECENT_ROWS else "")
    return (head + f"<p style='margin:0 0 4px;font-size:13px'>{e(recent_summary(items))}</p>"
            "<table><tr><th>날짜</th><th>종목</th><th>신호</th><th>신호일 종가</th><th>지금까지</th></tr>"
            + "".join(rows) + "</table>" + more)


def rebound_lines(reports: list[TickerReport]) -> list[str]:
    """예: 'SOXX: RSI 30 진입 (과거 5일 +0.9%p)'."""
    return [f"{r.ticker}: " + ", ".join(f"{k} (과거 {REBOUND[k]})" for k in r.rebound) for r in reports if r.rebound]


def _rebound_html(reports: list[TickerReport], title: str) -> str:
    e = html.escape
    lines = rebound_lines(reports)
    return (f"<h3>{e(title)} <span class='m' style='font-weight:normal;font-size:12px'>— {e(REBOUND_NOTE)}</span></h3>"
            + (_ul(lines, e) if lines else "<p class='m'>없음</p>"))


def render_html(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    e = html.escape
    date = max(r.date for r in reports).date()
    q = quality_lines(reports, meta.get("failed", []), meta.get("notes"))
    qbox = (f"<div class='box warn'><b>데이터 점검 ⚠️</b>{_ul(q, e)}</div>" if q
            else f"<p class='m' style='margin:4px 0;font-size:13px'>✅ 데이터 점검: 이상 없음 ({len(reports)}종목, 기준일 {date})</p>")
    look = market_lines(reports, meta) + _headline(reports)
    note = meta.get("calendar_note")
    look_html = _ul(look, e) + (f"<p class='m' style='font-size:11px;margin:2px 0'>참고: {e(note)}</p>" if note else "")
    footer = (f"<p class='m' style='font-size:11px;margin-top:16px'>{e(LEGEND)}<br>{e(RULE_LINE)}<br>{e(VALID_LINE)} "
              f"<a href='{DOC_URL}'>근거 문서</a></p></div>")
    parts = [_CSS.replace("</style>", _BADGE_CSS + "</style>"), '<div class="ck">']

    if meta.get("mode") == "holiday":
        parts.append(f"<h2 style='margin:0 0 8px'>🇺🇸 미국장 휴장 · {e(str(meta['checked']))} {e(meta['holiday'])}</h2>")
        parts.append(f"<p>다음 거래일: <b>{e(str(meta['next_open']))}</b>. 새 종가가 없어 체크리스트는 "
                     f"직전 거래일({date}) 기준이며 변화는 없습니다.</p>")
        parts += [qbox, f"<h3>직전 거래일({date}) 요약</h3>" + look_html, _rebound_html(reports, "직전 거래일 단기 반등 후보"), footer]
        return "".join(parts)

    parts.append(f"<h2 style='margin:0 0 6px'>미국장 지표 체크리스트 <span class='m' style='font-weight:normal'>({date} 종가)</span></h2>")
    if meta.get("mode") == "delayed":
        parts.append(f"<div class='box warn'><b>데이터 지연</b>: 기대 기준일 {e(str(meta['target']))}, "
                     f"받은 데이터 {date}. 아래는 받은 데이터 기준입니다.</div>")
    b = _basket(reports, meta)
    parts += [qbox, _signal_html(reports), _basket_html(b) if b else "", "<h3>한눈에 보기</h3>" + look_html]
    summary, cl = change_lines(reports)
    parts.append(f"<h3>전일 대비 변화 <span class='m' style='font-weight:normal;font-size:12px'>— {e(summary)}</span></h3>")
    parts.append(_ul(cl, e) if cl else "<p class='m'>없음</p>")
    parts.append("<h3>성향별 판정 <span class='m' style='font-weight:normal;font-size:12px'>"
                 f"점수 {RULE.entry:g} 이상 🟢 · {RULE.exit:g} 이하 🔴 · 노란 칸 = 전일과 색이 바뀐 성향</span></h3>")
    parts.append(_table_html(reports))
    parts += [_rebound_html(reports, "단기 반등 후보"), _recent_html(reports), footer]
    return "".join(parts)
