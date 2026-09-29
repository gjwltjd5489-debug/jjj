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
from .indicators import rolling_pct_rank
from .score import compute_indicators

UP, DOWN, WARN, NEUTRAL = "✅", "❌", "⚠️", "➖"


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
        if self.prev_checks is None:
            return None
        return sum(verdict(self.prev_checks, g) == "↑" for g, _, kind in GROUPS if kind == "dir")

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
    mid = close.rolling(p.bb_period).mean()
    sd = close.rolling(p.bb_period).std(ddof=0)
    bandwidth = 4 * sd / mid
    r = x.iloc[-1]
    c = r["close"]
    s, m, lng = f"{p.ma_short}일", f"{p.ma_mid}일", f"{p.ma_long}일"
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
    if _crossed_up(x["adx"], 25):
        ev.append(f"ADX 25 상향 ({'상승' if r['plus_di'] > r['minus_di'] else '하락'}추세 시작)")
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
    return rep


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
SHORT = {"장기 추세": "장기", "중기 추세": "중기", "모멘텀": "모멘텀", "추세 강도": "추세강도",
         "거래량": "거래량", "상대강도": "상대강도", "변동성": "변동성", "과열": "과열"}


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


def badge_text(group: str, v: str) -> str:
    color, label = badge(group, v)
    return f"{DOT[color]}{label}"


def _badge_html(group: str, v: str) -> str:
    color, label = badge(group, v)
    return f"<span class='b b{color}'>{DOT[color]} {html.escape(label)}</span>"


_BADGE_CSS = (".ck .b{display:inline-block;padding:2px 4px;border-radius:4px;font-weight:bold;white-space:nowrap}"
              + "".join(f".ck .b.b{k}{{background:{BG[k]};color:{FG[k]}}}" for k in BG)
              + ".ck .chg{display:inline-block;outline:2px solid #fbc02d;border-radius:5px}.ck .dot{color:#f9a825}"
              + ".ck td.dt{font-size:12px;line-height:1.45;padding:4px 6px;vertical-align:top;border-bottom:2px solid #fff}")


# ---------------------------------------------------------------- 출력

LEGEND = ("범례: 🟢 강세·정상 / 🟠 혼조·주의(고변동·과열·과매도·횡보) / 🔴 약세 / ⚪ 해당 없음. "
          "상세의 항목은 ✅ 강세 / ❌ 약세 / ➖ 중립 / ⚠️ 주의. 테두리 칸(마크다운은 [ ])은 직전 거래일과 판정이 달라진 성향. "
          "장기 = 200일선·이평 배열 · 중기 = 일목 구름·전환/기준 · 모멘텀 = MACD 시그널·RSI 50 · 추세 강도 = ADX≥25와 DMI 방향 · "
          "거래량 = VR 1년 백분위 50%·OBV 20일 증감 · 상대강도 = SPY 대비 비율의 50일 평균 · 변동성 = 20일 변동성 1년 백분위 80% 이상 ⚠️ · "
          "과열 = 50일선 이격도 1년 백분위 95% 이상(과열)·5% 이하(과매도) ⚠️. 현재 상태 요약이며 예측이나 매매 권유가 아님.")


def _v(r: TickerReport, g: str) -> str:
    return verdict(r.checks, g)


def _headline(reports: list[TickerReport]) -> list[str]:
    def pick(cond):
        return [r.ticker for r in reports if cond(r)]
    rules = [
        ("🟢 강세 정렬 (장기·중기·모멘텀 모두 강세)", lambda r: _v(r, "장기 추세") == _v(r, "중기 추세") == _v(r, "모멘텀") == "↑"),
        ("🔴 약세 정렬 (장기·중기·모멘텀 모두 약세)", lambda r: _v(r, "장기 추세") == _v(r, "중기 추세") == _v(r, "모멘텀") == "↓"),
        ("🟠 조정 중 (장기 강세, 중기·모멘텀 약세)", lambda r: _v(r, "장기 추세") == "↑" and _v(r, "중기 추세") == _v(r, "모멘텀") == "↓"),
        ("🟠 반등 시도 (장기 약세, 중기·모멘텀 강세)", lambda r: _v(r, "장기 추세") == "↓" and _v(r, "중기 추세") == _v(r, "모멘텀") == "↑"),
        ("🟠 반등 후보 (200일선 위 + RSI≤30)", lambda r: r.check("200일선").status == UP and "반등 후보" in r.check("RSI").detail),
        ("🟠 과열 주의 (50일선 이격 1년 상위 5%)", lambda r: _v(r, "과열") == "⚠️과열"),
        ("🟠 고변동 (변동성 1년 상위 20%)", lambda r: _v(r, "변동성") == "⚠️고변동"),
        ("🔴 200일선 아래", lambda r: r.check("200일선").status == DOWN),
    ]
    lines = []
    for label, cond in rules:
        names = pick(cond)
        if names:
            lines.append(f"{label}: {', '.join(names)}")
    return lines


def _verdict_changes(r: TickerReport) -> list[tuple[str, str, str]]:
    prev = r.prev_verdicts()
    if prev is None:
        return []
    now = r.verdicts()
    return [(g, prev[g], now[g]) for g, _, _ in GROUPS if prev[g] != now[g]]


def change_lines(reports: list[TickerReport]) -> tuple[str, list[str]]:
    """(요약 한 줄, 종목별 변화 줄). 성향 판정이 바뀐 것 위주, 바뀐 항목을 괄호로."""
    if not any(r.prev_checks for r in reports):
        return "직전 거래일 비교 불가", []
    rows = [(r, _verdict_changes(r)) for r in reports]
    rows = [(r, vc) for r, vc in rows if vc]
    n = sum(len(vc) for _, vc in rows)
    better = sum(r.ups > (r.prev_ups or 0) for r, _ in rows)
    worse = sum(r.ups < (r.prev_ups or 0) for r, _ in rows)
    summary = f"변화 {n}건 · 개선 {better}종목 · 악화 {worse}종목"
    lines = []
    for r, vc in sorted(rows, key=lambda t: -len(t[1])):
        parts = []
        for g, a, b in vc:
            items = [f"{now.name} {prev.status}→{now.status}" for prev, now in r.changes if now.group == g]
            parts.append(f"{SHORT[g]} {badge_text(g, a)}→{badge_text(g, b)}" + (f" ({', '.join(items)})" if items else ""))
        lines.append(f"{r.ticker}: " + ", ".join(parts))
    return summary, lines


def quality_lines(reports: list[TickerReport], failed: list[str]) -> list[str]:
    lines = [f"{r.ticker}: {i}" for r in reports for i in r.issues]
    lines += [f"{f} — 불러오기 실패" for f in failed]
    return lines


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


TINT = {"g": "#eef7ee", "o": "#fff5e6", "r": "#fdeeee", "n": "#f6f6f6"}


def _detail_cell_items(r: TickerReport, g: str) -> list[tuple[str, bool]]:
    """(핵심 수치, 전일 대비 바뀜) 목록."""
    changed = {now.name for _, now in r.changes}
    return [(c.short or "–", c.name in changed) for c in r.checks if c.group == g]


def _detail_md_cell(r: TickerReport, g: str) -> str:
    color, _ = badge(g, verdict(r.checks, g))
    items = [f"**{t}**" if ch else t for t, ch in _detail_cell_items(r, g)]
    return f"{DOT[color]} " + " · ".join(items)


def _detail_html_cell(r: TickerReport, g: str) -> str:
    color, _ = badge(g, verdict(r.checks, g))
    items = [f"<b>{html.escape(t)}</b> <span class='dot'>●</span>" if ch else html.escape(t)
             for t, ch in _detail_cell_items(r, g)]
    return f"<td class='dt' style='background:{TINT[color]};border-left:3px solid {FG[color]}'>{'<br>'.join(items)}</td>"


def _detail_line(r: TickerReport) -> str:
    parts = []
    for g, names, _ in GROUPS:
        items = " / ".join(f"{c.name} {c.status} {c.detail}" for c in r.checks if c.group == g)
        parts.append(f"{SHORT[g]} {badge_text(g, verdict(r.checks, g))} [{items}]")
    return " · ".join(parts)


def render_markdown(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    date = max(r.date for r in reports).date()
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
    out += ["## 한눈에 보기"] + ([f"- {h}" for h in heads] or ["- 특이 사항 없음"]) + [""]
    summary, lines = change_lines(reports)
    out += [f"## 전일 대비 변화 — {summary}"] + ([f"- {x}" for x in lines] or ["- 없음"]) + [""]
    out.append("## 성향별 판정")
    header = ["종목", "종가", "등락", "고점대비"] + [SHORT[g] for g, _, _ in GROUPS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in reports:
        changed = {g for g, _, _ in _verdict_changes(r)}
        row = [r.ticker + (" ⚠️" if r.issues else ""), f"{r.close:,.2f}", f"{r.change:+.1%}", f"{r.from_high:+.1%}"]
        row += [f"[{badge_text(g, _v(r, g))}]" if g in changed else badge_text(g, _v(r, g)) for g, _, _ in GROUPS]
        out.append("| " + " | ".join(row) + " |")
    out += ["", "## 오늘의 이벤트"]
    ev = [f"- **{r.ticker}**: {', '.join(r.events)}" for r in reports if r.events]
    out += ev or ["- 없음"]
    out += ["", "## 상세 (성향별 핵심 수치, 굵은 글씨 = 전일 대비 바뀐 항목)"]
    header = ["종목"] + [SHORT[g] for g, _, _ in GROUPS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in reports:
        out.append("| " + " | ".join([f"**{r.ticker}**"] + [_detail_md_cell(r, g) for g, _, _ in GROUPS]) + " |")
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
    lines += ["한눈에 보기"] + ([f"- {h}" for h in _headline(reports)] or ["- 특이 사항 없음"]) + [""]
    if meta.get("mode") != "holiday":
        summary, cl = change_lines(reports)
        lines += [f"전일 대비 변화: {summary}"] + [f"- {x}" for x in cl] + [""]
        lines += ["성향별 판정 (장기 / 중기 / 모멘텀 / 추세강도 / 거래량 / 상대강도 / 변동성 / 과열)"]
        lines += [f"- {r.ticker}: " + " / ".join(badge_text(g, _v(r, g)) for g, _, _ in GROUPS) for r in reports] + [""]
    ev = [f"- {r.ticker}: {', '.join(r.events)}" for r in reports if r.events]
    lines += ["이벤트"] + (ev or ["- 없음"]) + ["", "표와 상세는 HTML 메일에서 볼 수 있습니다. 현재 상태 요약이며 매매 권유가 아닙니다."]
    return "\n".join(lines)


_CSS = ("<style>.ck{font-family:-apple-system,Segoe UI,Malgun Gothic,sans-serif;font-size:14px;color:#222}"
        ".ck table{border-collapse:collapse}.ck th{padding:5px 5px;background:#f2f2f2;border-bottom:2px solid #999;"
        "white-space:nowrap}.ck td{padding:3px 5px;border-bottom:1px solid #ddd;white-space:nowrap}"
        ".ck td.c{text-align:center}.ck td.l{text-align:left}.ck td.d{white-space:normal;font-size:13px}"
        ".ck .g{font-weight:bold;color:#555;padding-top:8px}.ck .m{color:#777}"
        ".ck .x{background:#ffe58a;border-radius:3px;padding:0 3px}"
        ".ck .up{color:#2e7d32;font-weight:bold}.ck .dn{color:#c62828;font-weight:bold}.ck .nt{color:#888}"
        ".ck .wr{color:#e65100;font-weight:bold}"
        ".ck .box{border-left:4px solid #999;background:#fafafa;padding:6px 10px;margin:8px 0}"
        ".ck .warn{border-left-color:#e65100;background:#fff3e0}.ck .ok{border-left-color:#2e7d32;background:#f1f8e9}"
        ".ck h3{margin:14px 0 4px}.ck ul{margin:0;padding-left:20px}</style>")


def _ul(items: list[str], e) -> str:
    return "<ul>" + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul>"


def render_html(reports: list[TickerReport], meta: dict | None = None) -> str:
    meta = meta or {}
    e = html.escape
    date = max(r.date for r in reports).date()
    q = quality_lines(reports, meta.get("failed", []))
    heads = _headline(reports)
    qbox = (f"<div class='box warn'><b>데이터 점검 ⚠️</b>{_ul(q, e)}</div>" if q
            else f"<div class='box ok'><b>데이터 점검</b>: 이상 없음 ({len(reports)}종목, 기준일 {date})</div>")
    parts = [_CSS.replace("</style>", _BADGE_CSS + "</style>"), '<div class="ck">']

    if meta.get("mode") == "holiday":
        parts.append(f"<h2 style='margin:0 0 8px'>🇺🇸 미국장 휴장 · {e(str(meta['checked']))} {e(meta['holiday'])}</h2>")
        parts.append(f"<p>다음 거래일: <b>{e(str(meta['next_open']))}</b>. 새 종가가 없어 체크리스트는 "
                     f"직전 거래일({date}) 기준이며 변화는 없습니다.</p>")
        parts.append(qbox)
        parts.append(f"<h3>직전 거래일({date}) 요약</h3>" + (_ul(heads, e) if heads else "<p>특이 사항 없음</p>"))
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
    parts.append("<h3>한눈에 보기</h3>" + (_ul(heads, e) if heads else "<p>특이 사항 없음</p>"))
    summary, cl = change_lines(reports)
    parts.append(f"<h3>전일 대비 변화 <span class='m' style='font-weight:normal'>— {e(summary)}</span></h3>")
    parts.append(_ul(cl, e) if cl else "<p>없음</p>")

    parts.append("<h3>성향별 판정 <span class='m' style='font-weight:normal;font-size:12px'>"
                 "🟢 강세·정렬 / 🟠 혼조·주의 / 🔴 약세 · 노란 테두리와 ● = 전일과 달라진 성향 (마우스를 올리면 전일 판정)</span></h3><table><tr>")
    for h in ["종목", "종가", "등락", "고점대비"] + [SHORT[g] for g, _, _ in GROUPS]:
        parts.append(f"<th>{e(h)}</th>")
    parts.append("</tr>")
    group = None
    ncol = 4 + len(GROUPS)
    for r in reports:
        if r.group and r.group != group:
            group = r.group
            parts.append(f"<tr><td colspan='{ncol}' class='g'>{e(group)}</td></tr>")
        color = "#c62828" if r.change > 0 else "#1565c0" if r.change < 0 else "#222"
        flag = " ⚠️" if r.issues else ""
        parts.append(f"<tr><td class='l'><b>{e(r.ticker)}</b>{flag} <span class='m'>{e(r.name)}</span></td>")
        parts.append(f"<td class='c'>{r.close:,.2f}</td><td class='c'><span style='color:{color}'>{r.change:+.1%}</span></td>")
        parts.append(f"<td class='c'>{r.from_high:+.1%}</td>")
        prev = {g: a for g, a, _ in _verdict_changes(r)}
        for g, _, _ in GROUPS:
            cell = _badge_html(g, _v(r, g))
            if g in prev:
                cell = f"<span class='chg' title='전일 {e(badge(g, prev[g])[1])}'>{cell}</span> <span class='dot'>●</span>"
            parts.append(f"<td class='c'>{cell}</td>")
        parts.append("</tr>")
    parts.append("</table>")

    evs = [r for r in reports if r.events]
    parts.append("<h3>오늘의 이벤트</h3>" + (
        "<ul>" + "".join(f"<li><b>{e(r.ticker)}</b>: {e(', '.join(r.events))}</li>" for r in evs) + "</ul>"
        if evs else "<p>없음</p>"))
    parts.append("<h3>상세 <span class='m' style='font-weight:normal;font-size:12px'>"
                 "성향별 핵심 수치 · 칸 색 = 성향 판정 · 굵은 글씨와 ● = 전일 대비 바뀐 항목</span></h3><table><tr><th>종목</th>")
    for g, _, _ in GROUPS:
        parts.append(f"<th>{e(SHORT[g])}</th>")
    parts.append("</tr>")
    group = None
    for r in reports:
        if r.group and r.group != group:
            group = r.group
            parts.append(f"<tr><td colspan='{len(GROUPS) + 1}' class='g'>{e(group)}</td></tr>")
        parts.append(f"<tr><td class='l'><b>{e(r.ticker)}</b></td>" + "".join(_detail_html_cell(r, g) for g, _, _ in GROUPS) + "</tr>")
    parts.append(f"</table><p class='m' style='font-size:12px;margin-top:16px'>{e(LEGEND)}</p></div>")
    return "".join(parts)
