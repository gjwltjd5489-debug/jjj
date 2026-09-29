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
