"""체크리스트 점수와 매수·매도 규칙 — 메일의 '성향별 판정'과 같은 기준.

점수 (0~100, 50 = 중립)
    = 50 + 50 × (🟢 성향 수 − 🔴 성향 수) ÷ 판정 가능한 방향 성향 수
    방향 성향: 장기 추세 · 중기 추세 · 모멘텀 · 추세 강도 · 거래량 · 상대강도(벤치마크가 있을 때만)
    🟠(혼조·횡보·약함)은 0, ⚪(해당 없음)은 분모에서 뺀다. 메일 표의 색만 세면 누구나 다시 계산할 수 있다.

매수·매도 규칙 (결과를 보고 맞춘 값이 아니라 먼저 정한 값)
    진입: 3일 평균 점수 ≥ 70, 종가 > 200일선, 과열(50일선 이격 1년 상위 5%)·고변동(변동성 1년 상위 20%) 아님
    퇴출: 3일 평균 점수 ≤ 40
    손절 참고가: 종가 − 2 × ATR(14)

항목 판정은 scoring/checklist.py 의 build_report 와 같아야 한다 (tests/test_checklist_score.py 에서 검사).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import atr, rolling_pct_rank
from .profiles import Profile
from .score import compute_indicators

FAMILY_ITEMS = {
    "장기 추세": ["200일선", "배열"],
    "중기 추세": ["구름", "전환/기준"],
    "모멘텀": ["MACD", "RSI"],
    "추세 강도": ["ADX"],
    "거래량": ["VR", "OBV"],
    "상대강도": ["상대강도"],
}
CORE_FAMILIES = ["장기 추세", "중기 추세", "모멘텀", "추세 강도"]


@dataclass(frozen=True)
class ScoreRule:
    entry: float = 70          # 3일 평균 점수가 이 이상이면 진입 후보
    exit: float = 40           # 3일 평균 점수가 이 이하이면 퇴출
    smooth: int = 3            # 점수 평활 기간(일)
    near_exit: float = 50      # 보유 구간인데 이 이하이면 '청산 임박'
    need_above_200: bool = True
    block_overheat: bool = True
    block_highvol: bool = True
    stop_atr: float = 2.0      # 손절 참고가 = 종가 − stop_atr × ATR(14)


RULE = ScoreRule()


def item_states(x: pd.DataFrame, has_bench: bool) -> pd.DataFrame:
    """항목별 판정 이력: ✅ +1, ➖ 0, ❌ −1, 계산 불가 NaN (checklist.build_report 와 같은 기준)."""
    idx = x.index
    c = x["close"]

    def sign(cond: pd.Series, valid: pd.Series) -> pd.Series:
        return pd.Series(np.where(cond, 1.0, -1.0), index=idx).where(valid)

    def tri(up: pd.Series, down: pd.Series, valid: pd.Series) -> pd.Series:
        return pd.Series(np.select([up, down], [1.0, -1.0], 0.0), index=idx).where(valid)

    ms, mm, ml = x["ma_short"], x["ma_mid"], x["ma_long"]
    top = x[["span_a", "span_b"]].max(axis=1)
    bottom = x[["span_a", "span_b"]].min(axis=1)
    adx = x["adx"]
    obv_chg = x["obv"] - x["obv"].shift(20)
    st = pd.DataFrame(index=idx)
    st["200일선"] = sign(c > ml, ml.notna())
    st["배열"] = tri((ms > mm) & (mm > ml), (ms < mm) & (mm < ml), ml.notna())
    st["구름"] = tri(c > top, c < bottom, x["span_a"].notna() & x["span_b"].notna())
    st["전환/기준"] = sign(x["tenkan"] > x["kijun"], x["kijun"].notna())
    st["MACD"] = sign(x["macd"] > x["signal"], x["macd"].notna())
    st["RSI"] = sign(x["rsi"] >= 50, x["rsi"].notna())
    st["ADX"] = pd.Series(np.where(adx >= 25, np.where(x["plus_di"] > x["minus_di"], 1.0, -1.0), 0.0),
                          index=idx).where(adx.notna())
    st["VR"] = sign(x["vr_pct"] >= 0.5, x["vr_pct"].notna())
    st["OBV"] = sign(obv_chg > 0, obv_chg.notna())
    if has_bench:
        st["상대강도"] = sign(x["rs_ratio"] > x["rs_ratio_ma"], x["rs_ratio_ma"].notna())
    return st


def family_verdicts(states: pd.DataFrame) -> pd.DataFrame:
    """성향 판정 이력: 🟢 +1, 🟠 0, 🔴 −1, ⚪ NaN (checklist.verdict 와 같은 기준)."""
    out = {}
    for fam, items in FAMILY_ITEMS.items():
        cols = [i for i in items if i in states.columns]
        if not cols:
            continue
        sub = states[cols]
        out[fam] = np.sign(sub.fillna(0).sum(axis=1)).where(sub.notna().any(axis=1))
    return pd.DataFrame(out, index=states.index)


def score_from_verdicts(fv: pd.DataFrame, required: list[str] | None = None) -> pd.Series:
    n = fv.notna().sum(axis=1)
    raw = 50 + 50 * fv.sum(axis=1) / n.replace(0, np.nan)
    if required:
        raw = raw.where(fv[[f for f in required if f in fv.columns]].notna().all(axis=1))
    return raw


def points_from_verdicts(verdicts: dict[str, int | None]) -> float:
    """오늘 하루치 판정(+1/0/−1, None = 해당 없음) → 점수. 메일 표에서 바로 계산하는 방식."""
    vals = [v for v in verdicts.values() if v is not None]
    return float("nan") if not vals else 50 + 50 * sum(vals) / len(vals)


def run_rule(score_s: np.ndarray, above200: np.ndarray, overheat: np.ndarray, highvol: np.ndarray,
             rule: ScoreRule = RULE) -> tuple[np.ndarray, list[str], list[str]]:
    """상태 기계. 반환: 보유 상태(1/0/NaN), 이벤트(BUY/SELL/''), 진입 보류 사유."""
    n = len(score_s)
    state = np.full(n, np.nan)
    events = [""] * n
    blocked = [""] * n
    cur, started = 0.0, False
    for k in range(n):
        s = score_s[k]
        if np.isnan(s):
            if started:
                state[k] = cur
            continue
        started = True
        if cur == 0 and s >= rule.entry:
            reasons = []
            if rule.need_above_200 and not above200[k]:
                reasons.append("200일선 아래")
            if rule.block_overheat and overheat[k]:
                reasons.append("과열")
            if rule.block_highvol and highvol[k]:
                reasons.append("고변동")
            if reasons:
                blocked[k] = "·".join(reasons)
            else:
                cur = 1.0
                events[k] = "BUY"
        elif cur == 1 and s <= rule.exit:
            cur = 0.0
            events[k] = "SELL"
        state[k] = cur
    return state, events, blocked


def checklist_score(df: pd.DataFrame, p: Profile, bench: pd.Series | None = None,
                    rule: ScoreRule = RULE) -> pd.DataFrame:
    """날짜별 점수·3일 평균·위험 표시·보유 상태·매수/매도 이벤트."""
    x = compute_indicators(df, p, bench)
    st = item_states(x, bench is not None)
    fv = family_verdicts(st)
    required = list(CORE_FAMILIES)
    if df["Volume"].notna().any():
        required.append("거래량")
    if bench is not None:
        required.append("상대강도")
    score = score_from_verdicts(fv, required)
    score_s = score.rolling(rule.smooth).mean() if rule.smooth > 1 else score
    # 오늘 점수가 내일도 그대로일 때의 내일 평균 (신호 임박 판단용)
    proj = score_s + (score - score.shift(rule.smooth - 1)) / rule.smooth if rule.smooth > 1 else score
    above200 = (x["close"] > x["ma_long"]).to_numpy()
    disp_pct = rolling_pct_rank(x["close"] / x["ma_mid"] - 1, p.pct_window)
    overheat = (disp_pct >= 0.95).to_numpy()
    highvol = (x["rvol_pct"] >= 0.8).to_numpy()
    state, events, blocked = run_rule(score_s.to_numpy(), above200, overheat, highvol, rule)
    out = pd.DataFrame({
        "close": x["close"], "score": score, "score_s": score_s, "proj": proj,
        "above200": above200, "overheat": overheat, "highvol": highvol,
        "state": state, "event": events, "blocked": blocked,
        "stop": x["close"] - rule.stop_atr * atr(df, 14),
    }, index=x.index)
    return out.join(fv.add_prefix("f_"))


def strategy_returns(out: pd.DataFrame, cost: float = 0.0005) -> pd.Series:
    """보유 상태를 다음 날부터 적용한 일간 수익률 (편도 cost 차감)."""
    ret = out["close"].pct_change().fillna(0.0)
    pos = out["state"].fillna(0.0).shift(1).fillna(0.0)
    trades = pos.diff().abs().fillna(0.0)
    return pos * ret - trades * cost
