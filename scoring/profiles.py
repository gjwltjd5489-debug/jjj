"""종목별 점수 파라미터(프로파일).

다른 종목으로 확장할 때는 DEFAULT를 복사해 기간/구간만 바꿔 PROFILES에 등록한다.
구간(bands)은 [(상한, 점수), ...] 형태이며, 값이 상한 '미만'인 첫 구간의 점수를 준다.
여기는 지표 계산 파라미터만 둔다. 점수 배분과 신호 임계값은 scoring/cards.py 의 카드별로 정한다.
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

    # v1~v4 추가 지표
    adx_period: int = 14
    bb_period: int = 20
    vol_period: int = 20          # 실현변동성 기간
    pct_window: int = 252         # 백분위 계산 창 (1년)
    dd_window: int = 252          # 고점 대비 낙폭 창 (52주)
    dist_lookback: int = 25       # 분산일 집계 기간
    rs_ma: int = 50               # 상대강도 비율 이평
    rs_lookback: int = 63         # 상대수익 비교 기간 (3개월)
    benchmark: str = "SPY"        # 상대강도 비교 대상 (참고용 이름)

    notes: str = field(default="", compare=False)


DEFAULT = Profile(name="DEFAULT", notes="미국 ETF/대형주 기본값 (20/50/200 이평)")

QQQ = replace(
    DEFAULT,
    name="QQQ",
    notes="Invesco QQQ Trust. 기본값 그대로 시작, 데이터로 VR 구간 보정 예정",
)

# 확장 예시: 한국 지수/종목은 5/20/60/120 관례를 따르므로 20/60/120으로 시작
KOSPI = replace(DEFAULT, name="KOSPI", ma_short=20, ma_mid=60, ma_long=120, benchmark="KS11",
                notes="한국 관례 이평(20/60/120). 지수는 거래량이 없을 수 있음")

PROFILES = {p.name: p for p in (DEFAULT, QQQ, KOSPI)}


def get_profile(ticker: str) -> Profile:
    return PROFILES.get(ticker.upper(), replace(DEFAULT, name=ticker.upper()))
