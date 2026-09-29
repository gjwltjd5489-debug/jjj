"""점수 카드(기준표) 정의.

카드 = 항목(Item) 목록 + 신호 임계값. 각 항목은 지표 DataFrame(x)과 Profile(p)을 받아
0 ~ 만점 사이 점수 Series를 돌려준다(계산 불가 구간은 NaN).
optional 카테고리(거래량, 상대강도, 매크로)는 데이터가 없으면 빼고 남은 만점 기준으로 100점 환산한다.
기준표 설명은 docs/scorecards.md 참고.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .profiles import INF, Profile

ItemFn = Callable[[pd.DataFrame, Profile], pd.Series]

CATEGORY_NAMES = {
    "ichimoku": "일목균형표",
    "ma": "이동평균",
    "volume": "거래량",
    "momentum": "모멘텀",
    "trend": "추세",
    "long_trend": "장기 추세",
    "mid_trend": "중기 추세(일목)",
    "strength": "추세 강도",
    "risk": "변동성/리스크",
    "position": "가격 위치",
    "relative": "상대강도",
    "regime": "국면",
    "pullback": "눌림 깊이",
    "capitulation": "투매(거래량)",
    "support": "지지",
    "internals": "시장 내부(거래량)",
    "macro": "매크로/심리",
}

OPTIONAL = frozenset({"volume", "relative", "macro", "capitulation", "internals"})

# optional 카테고리가 기대는 원자료 컬럼 (지표 DataFrame 기준). 원자료가 아예 없으면 항목을 빼고 환산하고,
# 원자료는 있는데 지표가 아직 계산 전(워밍업)이면 총점을 비운다.
INPUT_OF_CATEGORY = {
    "volume": "volume_raw",
    "capitulation": "volume_raw",
    "internals": "volume_raw",
    "relative": "bench_raw",
}


@dataclass(frozen=True)
class Item:
    key: str
    category: str
    max: float
    fn: ItemFn
    desc: str
    needs: str | None = None  # optional 항목의 원자료 컬럼 (없으면 INPUT_OF_CATEGORY)

    @property
    def input_col(self) -> str | None:
        return self.needs or INPUT_OF_CATEGORY.get(self.category)


@dataclass(frozen=True)
class Card:
    name: str
    title: str
    items: tuple[Item, ...]
    strong_buy: float = 80
    buy: float = 65
    sell: float = 35
    strong_sell: float = 20
    smooth: int = 1                  # 신호 판단에 쓰는 점수 이동평균 기간
    gate: ItemFn | None = None       # False 인 날은 총점을 gate_cap 으로 제한
    gate_cap: float = 100
    gate_desc: str = ""

    @property
    def categories(self) -> list[str]:
        seen: list[str] = []
        for it in self.items:
            if it.category not in seen:
                seen.append(it.category)
        return seen

    def category_max(self, cat: str) -> float:
        return sum(it.max for it in self.items if it.category == cat)


# ---------------------------------------------------------------- helpers

def _valid(*inputs: pd.Series) -> pd.Series:
    return pd.concat(inputs, axis=1).notna().all(axis=1)


def cond(mask: pd.Series, points: float, *inputs: pd.Series) -> pd.Series:
    return pd.Series(np.where(mask, points, 0.0), index=mask.index).where(_valid(*inputs))


def band(values: pd.Series, bands) -> pd.Series:
    """[(상한, 점수), ...] — 값이 상한 미만인 첫 구간의 점수."""
    out = pd.Series(np.nan, index=values.index)
    remaining = values.notna()
    for upper, points in bands:
        hit = remaining & (values < upper)
        out[hit] = points
        remaining &= ~hit
    return out


def linear(values: pd.Series, x0: float, x1: float, p0: float, p1: float) -> pd.Series:
    """x0 에서 p0, x1 에서 p1 로 선형 보간하고 구간 밖은 끝값으로 고정 (계단 대신 연속 점수)."""
    t = ((values - x0) / (x1 - x0)).clip(0, 1)
    return p0 + (p1 - p0) * t


def cloud_position(x: pd.DataFrame, above: float, inside: float) -> pd.Series:
    c = x["close"]
    top = x[["span_a", "span_b"]].max(axis=1)
    bottom = x[["span_a", "span_b"]].min(axis=1)
    pts = pd.Series(np.select([c > top, c < bottom], [above, 0.0], inside), index=x.index)
    return pts.where(_valid(x["span_a"], x["span_b"]))


def adx_score(x: pd.DataFrame, max_pts: float) -> pd.Series:
    """추세 없음(ADX<=15) = 절반, 강한 상승추세 = 만점, 강한 하락추세 = 0."""
    direction = np.sign(x["plus_di"] - x["minus_di"])
    strength = ((x["adx"] - 15) / 20).clip(0, 1)
    half = max_pts / 2
    return (half + half * direction * strength).where(_valid(x["adx"], x["plus_di"], x["minus_di"]))


def ma_slope_up(x: pd.DataFrame, col: str, lookback: int, pts: float) -> pd.Series:
    prev = x[col].shift(lookback)
    return cond(x[col] > prev, pts, x[col], prev)


def macd_points(x: pd.DataFrame, cross: float, zero: float, rising: float) -> pd.Series:
    hist_prev = x["hist"].shift(1)
    pts = (np.where(x["macd"] > x["signal"], cross, 0.0)
           + np.where(x["macd"] > 0, zero, 0.0)
           + np.where(x["hist"] > hist_prev, rising, 0.0))
    return pd.Series(pts, index=x.index).where(_valid(x["macd"], hist_prev))


def vol_regime(x: pd.DataFrame, pts: float) -> pd.Series:
    """실현변동성 1년 백분위: 30% 이하 만점 → 90% 이상 0점."""
    return linear(x["rvol_pct"], 0.3, 0.9, pts, 0.0)


def dist_score(x: pd.DataFrame, pts: float) -> pd.Series:
    """분산일(25일 중 거래량 증가 하락일) 수: 3 이하 만점, 4~5 2/3, 6~7 1/3, 8 이상 0."""
    return band(x["dist_days"], ((4, pts), (6, pts * 2 / 3), (8, pts / 3), (INF, 0)))


def i(key, cat, mx, fn, desc, needs=None) -> Item:
    return Item(key, cat, mx, fn, desc, needs)


# ---------------------------------------------------------------- v0 (기존)

V0 = Card(
    name="v0",
    title="v0 기존 (일목+이평+VR+RSI/MACD)",
    items=(
        i("A1_cloud", "ichimoku", 10, lambda x, p: cloud_position(x, 10, 5), "구름 위 10 / 안 5 / 아래 0"),
        i("A2_tenkan_kijun", "ichimoku", 6,
          lambda x, p: _tk(x, p), "전환>기준 6 / ±0.1% 3 / 아래 0"),
        i("A3_chikou", "ichimoku", 6,
          lambda x, p: cond(x["close"] > x["chikou_ref"], 6, x["chikou_ref"]), "종가 > 26일 전 종가"),
        i("A4_future_cloud", "ichimoku", 4,
          lambda x, p: cond(x["lead_a"] > x["lead_b"], 4, x["lead_a"], x["lead_b"]), "미래 구름 양운"),
        i("A5_above_kijun", "ichimoku", 4,
          lambda x, p: cond(x["close"] > x["kijun"], 4, x["kijun"]), "종가 > 기준선"),
        i("B1_above_short", "ma", 4, lambda x, p: cond(x["close"] > x["ma_short"], 4, x["ma_short"]), "종가 > 단기선"),
        i("B2_above_mid", "ma", 5, lambda x, p: cond(x["close"] > x["ma_mid"], 5, x["ma_mid"]), "종가 > 중기선"),
        i("B3_above_long", "ma", 7, lambda x, p: cond(x["close"] > x["ma_long"], 7, x["ma_long"]), "종가 > 장기선"),
        i("B4_alignment", "ma", 8, lambda x, p: _alignment(x, 8), "정배열 8 / 부분 4 / 역배열 0"),
        i("B5_mid_slope", "ma", 3, lambda x, p: ma_slope_up(x, "ma_mid", p.ma_mid_slope_lookback, 3), "중기선 상승"),
        i("B6_long_slope", "ma", 3, lambda x, p: ma_slope_up(x, "ma_long", p.ma_long_slope_lookback, 3), "장기선 상승"),
        i("C1_vr", "volume", 14, lambda x, p: band(x["vr"], p.vr_bands), "VR 절대 구간 (역발상)"),
        i("C2_obv", "volume", 6, lambda x, p: cond(x["obv"] > x["obv_ma"], 6, x["obv"], x["obv_ma"]), "OBV > OBV 20일선"),
        i("D1_rsi", "momentum", 10, lambda x, p: band(x["rsi"], p.rsi_bands), "RSI 혼합 구간"),
        i("D2_macd", "momentum", 10, lambda x, p: macd_points(x, 5, 3, 2), "MACD>시그널 5, >0 3, 히스토 증가 2"),
    ),
)


def _tk(x: pd.DataFrame, p: Profile) -> pd.Series:
    rel = (x["tenkan"] - x["kijun"]) / x["kijun"]
    pts = pd.Series(np.select([rel > p.tk_tolerance, rel < -p.tk_tolerance], [6.0, 0.0], 3.0), index=x.index)
    return pts.where(rel.notna())


def _alignment(x: pd.DataFrame, pts: float) -> pd.Series:
    ms, mm, ml = x["ma_short"], x["ma_mid"], x["ma_long"]
    out = pd.Series(np.select([(ms > mm) & (mm > ml), (ms < mm) & (mm < ml)], [pts, 0.0], pts / 2), index=x.index)
    return out.where(_valid(ms, mm, ml))


# ---------------------------------------------------------------- v1 균형형

V1 = Card(
    name="v1",
    title="v1 균형형 (중복 제거 + 변동성·추세강도·상대강도)",
    items=(
        i("T1_dist_long", "trend", 10,
          lambda x, p: linear(x["close"] / x["ma_long"] - 1, -0.05, 0.05, 0, 10), "200일선 대비 -5%→0, +5%→10 (연속)"),
        i("T2_cloud", "trend", 8, lambda x, p: cloud_position(x, 8, 4), "구름 위 8 / 안 4 / 아래 0"),
        i("T3_golden", "trend", 6,
          lambda x, p: cond(x["ma_mid"] > x["ma_long"], 6, x["ma_mid"], x["ma_long"]), "50일선 > 200일선"),
        i("T4_long_slope", "trend", 6,
          lambda x, p: ma_slope_up(x, "ma_long", p.ma_long_slope_lookback, 6), "200일선 20일 전보다 상승"),
        i("S1_adx", "strength", 10, lambda x, p: adx_score(x, 10), "ADX·DMI: 강한 상승 10 / 무추세 5 / 강한 하락 0"),
        i("M1_mom12_1", "momentum", 8,
          lambda x, p: linear(x["mom12_1"], -0.10, 0.20, 0, 8), "12-1개월 수익률 -10%→0, +20%→8"),
        i("M2_macd", "momentum", 7, lambda x, p: macd_points(x, 4, 3, 0), "MACD>시그널 4, MACD>0 3"),
        i("R1_vol", "risk", 10, lambda x, p: vol_regime(x, 10), "20일 변동성 1년 백분위 30%↓ 10 → 90%↑ 0"),
        i("R2_drawdown", "risk", 5,
          lambda x, p: band(x["dd"], ((-0.20, 0), (-0.10, 2), (INF, 5))), "52주 고점 대비 -10% 이내 5 / -20% 이내 2"),
        i("V1_vr_pct", "volume", 5,
          lambda x, p: band(x["vr_pct"], ((0.2, 5), (0.9, 3), (INF, 1))), "VR 1년 백분위 20%↓ 5 / 중간 3 / 90%↑ 1"),
        i("V2_dist", "volume", 5, lambda x, p: dist_score(x, 5), "분산일 ≤3 5 / 4~5 3.3 / 6~7 1.7 / ≥8 0"),
        i("P1_rsi", "position", 10,
          lambda x, p: band(x["rsi"], ((30, 10), (40, 8), (50, 6), (60, 5), (70, 3), (INF, 1))),
          "RSI 낮을수록 가점 (과매도 10 → 과매수 1)"),
        i("RS1_ratio", "relative", 5,
          lambda x, p: cond(x["rs_ratio"] > x["rs_ratio_ma"], 5, x["rs_ratio"], x["rs_ratio_ma"]), "벤치마크 대비 비율 > 50일 평균"),
        i("RS2_ret", "relative", 5,
          lambda x, p: cond(x["rs_ret"] > 0, 5, x["rs_ret"]), "3개월 상대수익 > 0"),
    ),
    strong_buy=80, buy=65, sell=35, strong_sell=20,
)


# ---------------------------------------------------------------- v2 추세추종형

V2 = Card(
    name="v2",
    title="v2 추세추종형 (큰 추세 탑승, 추세 이탈 시 청산)",
    items=(
        i("L1_above_long", "long_trend", 10,
          lambda x, p: cond(x["close"] > x["ma_long"], 10, x["ma_long"]), "종가 > 200일선"),
        i("L2_golden", "long_trend", 10,
          lambda x, p: cond(x["ma_mid"] > x["ma_long"], 10, x["ma_mid"], x["ma_long"]), "50일선 > 200일선"),
        i("L3_long_slope", "long_trend", 10,
          lambda x, p: ma_slope_up(x, "ma_long", p.ma_long_slope_lookback, 10), "200일선 상승"),
        i("L4_mom12_1", "long_trend", 10,
          lambda x, p: cond(x["mom12_1"] > 0, 10, x["mom12_1"]), "12-1개월 수익률 > 0"),
        i("I1_cloud", "mid_trend", 10, lambda x, p: cloud_position(x, 10, 5), "구름 위 10 / 안 5"),
        i("I2_tk", "mid_trend", 5,
          lambda x, p: cond(x["tenkan"] > x["kijun"], 5, x["tenkan"], x["kijun"]), "전환선 > 기준선"),
        i("I3_chikou", "mid_trend", 5,
          lambda x, p: cond(x["close"] > x["chikou_ref"], 5, x["chikou_ref"]), "후행스팬 > 26일 전 종가"),
        i("I4_future", "mid_trend", 5,
          lambda x, p: cond(x["lead_a"] > x["lead_b"], 5, x["lead_a"], x["lead_b"]), "미래 구름 양운"),
        i("S1_adx", "strength", 15, lambda x, p: adx_score(x, 15), "ADX·DMI: 강한 상승 15 / 무추세 7.5 / 강한 하락 0"),
        i("R1_vol", "risk", 10, lambda x, p: vol_regime(x, 10), "변동성 백분위 30%↓ 10 → 90%↑ 0"),
        i("V1_dist", "volume", 10, lambda x, p: dist_score(x, 10), "분산일 ≤3 10 / 4~5 6.7 / 6~7 3.3 / ≥8 0"),
    ),
    strong_buy=85, buy=70, sell=40, strong_sell=25, smooth=3,
)


# ---------------------------------------------------------------- v3 눌림목형

def _uptrend(x: pd.DataFrame, p: Profile) -> pd.Series:
    prev = x["ma_long"].shift(p.ma_long_slope_lookback)
    return (x["close"] > x["ma_long"]) & (x["ma_long"] > prev)


V3 = Card(
    name="v3",
    title="v3 눌림목형 (상승 국면에서만 과매도 매수)",
    items=(
        i("G1_above_long", "regime", 10,
          lambda x, p: cond(x["close"] > x["ma_long"], 10, x["ma_long"]), "종가 > 200일선"),
        i("G2_long_slope", "regime", 10,
          lambda x, p: ma_slope_up(x, "ma_long", p.ma_long_slope_lookback, 10), "200일선 상승"),
        i("G3_golden", "regime", 10,
          lambda x, p: cond(x["ma_mid"] > x["ma_long"], 10, x["ma_mid"], x["ma_long"]), "50일선 > 200일선"),
        i("PB1_rsi", "pullback", 15,
          lambda x, p: band(x["rsi"], ((30, 15), (40, 12), (50, 8), (60, 4), (INF, 0))), "RSI <30 15 / <40 12 / <50 8 / <60 4"),
        i("PB2_pct_b", "pullback", 10,
          lambda x, p: band(x["pct_b"], ((0, 10), (0.2, 8), (0.5, 4), (INF, 0))), "볼린저 %B <0 10 / <0.2 8 / <0.5 4"),
        i("PB3_streak", "pullback", 8,
          lambda x, p: band(x["down_streak"], ((2, 0), (3, 3), (4, 6), (INF, 8))), "연속 하락 2일 3 / 3일 6 / 4일↑ 8"),
        i("PB4_below_short", "pullback", 7,
          lambda x, p: cond(x["close"] < x["ma_short"], 7, x["ma_short"]), "종가 < 20일선"),
        i("CP1_vr_pct", "capitulation", 15,
          lambda x, p: band(x["vr_pct"], ((0.10, 15), (0.25, 10), (0.5, 5), (INF, 0))), "VR 1년 백분위 10%↓ 15 / 25%↓ 10 / 50%↓ 5"),
        i("SP1_cloud", "support", 8,
          lambda x, p: cloud_position(x, 8, 8), "구름 위 또는 안 8 (구름 하단 이탈 0)"),
        i("SP2_near_mid", "support", 7,
          lambda x, p: cond(x["close"] >= x["ma_mid"] * 0.97, 7, x["ma_mid"]), "종가 ≥ 50일선 -3%"),
    ),
    strong_buy=80, buy=65, sell=30, strong_sell=20,
    gate=_uptrend, gate_cap=45, gate_desc="종가 > 200일선 이고 200일선 상승일 때만 45점 초과 허용",
)


# ---------------------------------------------------------------- v4 리스크관리형

V4 = Card(
    name="v4",
    title="v4 리스크관리형 (변동성·낙폭·매크로로 하락장 회피)",
    items=(
        i("T1_dist_long", "trend", 15,
          lambda x, p: linear(x["close"] / x["ma_long"] - 1, -0.05, 0.05, 0, 15), "200일선 대비 -5%→0, +5%→15"),
        i("T2_golden", "trend", 10,
          lambda x, p: cond(x["ma_mid"] > x["ma_long"], 10, x["ma_mid"], x["ma_long"]), "50일선 > 200일선"),
        i("T3_cloud", "trend", 10, lambda x, p: cloud_position(x, 10, 5), "구름 위 10 / 안 5"),
        i("R1_vol", "risk", 15, lambda x, p: vol_regime(x, 15), "변동성 백분위 30%↓ 15 → 90%↑ 0"),
        i("R2_drawdown", "risk", 10,
          lambda x, p: band(x["dd"], ((-0.20, 0), (-0.10, 2), (-0.05, 6), (INF, 10))), "고점 대비 -5% 이내 10 / -10% 6 / -20% 2"),
        i("N1_dist", "internals", 15, lambda x, p: dist_score(x, 15), "분산일 ≤3 15 / 4~5 10 / 6~7 5 / ≥8 0"),
        i("MA1_vix", "macro", 10,
          lambda x, p: band(x["vix"], ((15, 10), (20, 8), (25, 5), (30, 2), (INF, 0))), "VIX <15 10 / <20 8 / <25 5 / <30 2", "vix"),
        i("MA2_vix_trend", "macro", 5,
          lambda x, p: cond(x["vix_ma5"] < x["vix_ma50"], 5, x["vix_ma5"], x["vix_ma50"]), "VIX 5일 평균 < 50일 평균", "vix"),
        i("MA3_rates", "macro", 5,
          lambda x, p: cond(x["us10y_chg20"] < 0.30, 5, x["us10y_chg20"]), "미 10년물 20일 상승폭 < 0.30%p", "us10y"),
        i("MA4_credit", "macro", 5,
          lambda x, p: cond(x["hy"] < x["hy_ma50"], 5, x["hy"], x["hy_ma50"]), "하이일드 스프레드 < 50일 평균", "hy"),
    ),
    strong_buy=80, buy=65, sell=40, strong_sell=25, smooth=3,
)


CARDS = {c.name: c for c in (V0, V1, V2, V3, V4)}


def get_card(name: str) -> Card:
    try:
        return CARDS[name.lower()]
    except KeyError:
        raise ValueError(f"알 수 없는 카드: {name} (가능: {', '.join(CARDS)}, {', '.join(COMBOS)})") from None


for _card in CARDS.values():
    assert abs(sum(it.max for it in _card.items) - 100) < 1e-9, _card.name


# ---------------------------------------------------------------- v5 조합형 (v3 진입 + v2 청산)

@dataclass(frozen=True)
class Combo:
    """진입 카드와 청산 카드를 묶은 상태 기계.

    - 진입: 대기 상태에서 entry 카드 점수가 entry_level 을 상향 돌파하고, 청산 조건이 아닐 때
    - 청산: 보유 상태에서 exit 카드 점수(평활)가 exit_level 이하
    - 점수(순위용 100점) = 보유 자격 50 + 추세 점수(v2)의 절반 50
      rank="v2v3" 이면 50 + (v2 + v3) / 4
    - trend_entry 가 있으면 청산 카드 점수가 그 값을 상향 돌파할 때도 진입 (눌림 없이 오르는 추세 대응)
    """

    name: str
    title: str
    entry: str = "v3"
    exit: str = "v2"
    entry_level: float = 65
    exit_level: float = 40
    rank: str = "v2"
    trend_entry: float | None = None
    strong_buy: float = 90     # 보유 중 & v2 ≥ 80
    buy: float = 50            # 보유 중
    sell: float = 20           # 대기 & v2 ≤ 40
    strong_sell: float = 12.5  # 대기 & v2 ≤ 25
    smooth: int = 1


V5 = Combo(name="v5", title="v5 조합형 (v3 눌림목 진입 + v2 추세 이탈 청산)")
V6 = Combo(name="v6", title="v6 조합형 (v3 눌림목 또는 v2 추세 확인 진입 + v2 청산)", trend_entry=70)

COMBOS = {c.name: c for c in (V5, V6)}
