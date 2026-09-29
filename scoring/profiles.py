"""종목별 점수 파라미터(프로파일).

다른 종목으로 확장할 때는 DEFAULT를 복사해 기간/구간/임계값만 바꿔 PROFILES에 등록한다.
구간(bands)은 [(상한, 점수), ...] 형태이며, 값이 상한 '미만'인 첫 구간의 점수를 준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

INF = float("inf")


@dataclass(frozen=True)
class Profile:
    name: str

    # 일목균형표
    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    displacement: int = 26
    tk_tolerance: float = 0.001  # 전환선/기준선 차이가 0.1% 이내면 '같음'

    # 이동평균 (단기, 중기, 장기)
    ma_short: int = 20
    ma_mid: int = 50
    ma_long: int = 200
    ma_mid_slope_lookback: int = 10
    ma_long_slope_lookback: int = 20

    # 거래량
    vr_period: int = 20
    vr_bands: tuple = ((70, 14), (120, 11), (200, 8), (300, 5), (450, 2), (INF, 0))
    obv_ma: int = 20

    # 모멘텀
    rsi_period: int = 14
    rsi_bands: tuple = ((30, 7), (50, 5), (70, 10), (INF, 3))
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 신호 구간 (총점 기준)
    strong_buy: float = 80
    buy: float = 65
    sell: float = 35
    strong_sell: float = 20

    notes: str = field(default="", compare=False)


DEFAULT = Profile(name="DEFAULT", notes="미국 ETF/대형주 기본값 (20/50/200 이평)")

QQQ = replace(
    DEFAULT,
    name="QQQ",
    notes="Invesco QQQ Trust. 기본값 그대로 시작, 데이터로 VR 구간 보정 예정",
)

# 확장 예시: 한국 지수/종목은 5/20/60/120 관례를 따르므로 20/60/120으로 시작
KOSPI = replace(DEFAULT, name="KOSPI", ma_short=20, ma_mid=60, ma_long=120,
                notes="한국 관례 이평(20/60/120). 지수는 거래량이 없을 수 있음")

PROFILES = {p.name: p for p in (DEFAULT, QQQ, KOSPI)}


def get_profile(ticker: str) -> Profile:
    return PROFILES.get(ticker.upper(), replace(DEFAULT, name=ticker.upper()))
