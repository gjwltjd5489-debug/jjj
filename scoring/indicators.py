"""기술적 지표 계산 (pandas 기반, 외부 TA 라이브러리 없음)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
             displacement: int = 26) -> pd.DataFrame:
    """일목균형표.

    - tenkan / kijun: 전환선, 기준선
    - span_a / span_b: 오늘 날짜 위에 그려진(= displacement 일 전에 계산된) 선행스팬 → 현재 구름
    - lead_a / lead_b: 오늘 계산되어 displacement 일 뒤에 그려질 선행스팬 → 미래 구름
    - chikou_ref: 후행스팬 비교 대상(displacement 일 전 종가)
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    lead_a = (tenkan_line + kijun_line) / 2
    lead_b = (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    return pd.DataFrame({
        "tenkan": tenkan_line,
        "kijun": kijun_line,
        "lead_a": lead_a,
        "lead_b": lead_b,
        "span_a": lead_a.shift(displacement),
        "span_b": lead_b.shift(displacement),
        "chikou_ref": close.shift(displacement),
    }, index=df.index)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """VR(%) = (상승일 거래량 + 보합일 거래량/2) / (하락일 거래량 + 보합일 거래량/2) * 100."""
    diff = df["Close"].diff()
    vol = df["Volume"]
    up = vol.where(diff > 0, 0.0).rolling(period).sum()
    down = vol.where(diff < 0, 0.0).rolling(period).sum()
    flat = vol.where(diff == 0, 0.0).rolling(period).sum()
    denom = down + flat / 2
    vr = (up + flat / 2) / denom.replace(0, np.nan) * 100
    # 첫 행(diff 없음)이 창에 포함되는 구간과 거래량 결측 구간은 NaN
    valid = diff.notna().astype(float).rolling(period).sum().eq(period) & vol.notna().rolling(period).sum().eq(period)
    return vr.where(valid)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    result = (direction * df["Volume"]).cumsum()
    return result.where(df["Volume"].notna())


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI."""
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    out = pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=close.index)
    # 초기 slow+signal 구간은 EMA가 안정되지 않았으므로 제외
    out.iloc[: slow + signal] = np.nan
    return out


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return _wilder(tr.where(prev.notna()), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder DMI/ADX. 반환: plus_di, minus_di, adx."""
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0).where(up.notna())
    minus_dm = down.where((down > up) & (down > 0), 0.0).where(down.notna())
    a = atr(df, period)
    plus_di = 100 * _wilder(plus_dm, period) / a
    minus_di = 100 * _wilder(minus_dm, period) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, period)}, index=df.index)


def bollinger_pct_b(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return (close - (mid - k * sd)) / (2 * k * sd)


def realized_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """연율화 실현변동성."""
    return np.log(close).diff().rolling(period).std() * np.sqrt(252)


def rolling_pct_rank(series: pd.Series, window: int = 252) -> pd.Series:
    """직전 window 일 중 오늘 값의 백분위(0~1). 과거 값만 사용."""
    return series.rolling(window).rank(pct=True)


def drawdown(close: pd.Series, window: int = 252) -> pd.Series:
    """window 일 최고 종가 대비 하락률 (0 이하)."""
    return close / close.rolling(window).max() - 1


def distribution_days(df: pd.DataFrame, lookback: int = 25, min_drop: float = 0.002) -> pd.Series:
    """최근 lookback 일 중 '거래량 증가 + 0.2% 이상 하락'한 날 수 (IBD 방식 분산일)."""
    ret = df["Close"].pct_change()
    vol = df["Volume"]
    dist = ((ret <= -min_drop) & (vol > vol.shift(1))).astype(float)
    valid = vol.notna() & vol.shift(1).notna() & ret.notna()
    dist = dist.where(valid)
    return dist.rolling(lookback).sum()


def momentum_12_1(close: pd.Series, long: int = 252, skip: int = 21) -> pd.Series:
    """12-1개월 모멘텀: 최근 1개월을 제외한 12개월 수익률."""
    return close.shift(skip) / close.shift(long) - 1


def down_streak(close: pd.Series) -> pd.Series:
    """연속 하락일 수."""
    down = close.diff() < 0
    groups = (~down).cumsum()
    return down.astype(int).groupby(groups).cumsum().astype(float).where(close.diff().notna())
