"""기술적 이벤트(이평 돌파, MACD 교차, RSI 과매도 등) 뒤의 수익률 검증 (watchlist 전 종목, 실제 데이터).

예전 메일의 '오늘의 이벤트' 20종을 전 기간에 다시 찾아, 이후 5·20·60·120거래일 수익률을
같은 종목의 평소 수익률과 비교한다. 결과: docs/events.md

- 초과 수익 = 이벤트 뒤 수익 − 같은 종목 모든 날의 평균 수익 (2장은 같은 200일선 국면의 평균과 비교)
- t = 같은 달(60일 이상은 같은 분기) 이벤트를 한 묶음으로 본 군집 t. 폭락장에 여러 종목이 한꺼번에
  이벤트를 내므로 건별로 세면 유의성이 부풀려진다.

예)
  python scripts/eval_events.py --asof 2026-10-02 --md docs/results/events_eval.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.score import compute_indicators  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

BACKFILL = {"PDBC": "PDBC+DBC"}
H = (5, 20, 60, 120)
EQUITY = {"지수", "섹터·스타일", "해외 증시", "빅테크"}
ALL = "(모든 날)"


def _up(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a > b) & (a.shift() <= b.shift())


def _dn(a: pd.Series, b) -> pd.Series:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return (a < b) & (a.shift() >= b.shift())


def detect(x: pd.DataFrame, bb_period: int) -> pd.DataFrame:
    """예전 메일 이벤트 판정과 같은 조건 (골든·데드크로스는 교차 당일만)."""
    c = x["close"]
    mid, sd = c.rolling(bb_period).mean(), c.rolling(bb_period).std(ddof=0)
    bw = 4 * sd / mid
    top, bot = x[["span_a", "span_b"]].max(axis=1), x[["span_a", "span_b"]].min(axis=1)
    adx_up = _up(x["adx"], 25)
    e = {"골든크로스 (50>200)": _up(x["ma_mid"], x["ma_long"]), "데드크로스 (50<200)": _dn(x["ma_mid"], x["ma_long"]),
         "20일선 상향 돌파": _up(c, x["ma_short"]), "20일선 하향 이탈": _dn(c, x["ma_short"]),
         "200일선 상향 돌파": _up(c, x["ma_long"]), "200일선 하향 이탈": _dn(c, x["ma_long"]),
         "MACD 상향 교차": _up(x["macd"], x["signal"]), "MACD 하향 교차": _dn(x["macd"], x["signal"]),
         "구름 상향 돌파": _up(c, top), "구름 하향 이탈": _dn(c, bot),
         "전환>기준 (호전)": _up(x["tenkan"], x["kijun"]), "전환<기준 (역전)": _dn(x["tenkan"], x["kijun"]),
         "ADX 25↑ 상승 방향": adx_up & (x["plus_di"] > x["minus_di"]),
         "ADX 25↑ 하락 방향": adx_up & (x["plus_di"] <= x["minus_di"]),
         "RSI 70 진입": _up(x["rsi"], 70), "RSI 30 진입": _dn(x["rsi"], 30),
         "볼린저 상단 돌파": _up(x["pct_b"], 1), "볼린저 하단 이탈": _dn(x["pct_b"], 0),
         "밴드폭 6개월 최저": (bw <= bw.rolling(126).min()) & (bw.notna().cumsum() > 126),
         "52주 신고가": c >= c.rolling(252, min_periods=60).max()}
    return pd.DataFrame(e).fillna(False).astype(bool)


def cluster_t(s: pd.Series, dates: pd.Series, freq: str) -> float:
    m = s.mean()
    res = (s - m).groupby(dates.dt.to_period(freq)).sum()
    se = np.sqrt((res ** 2).sum()) / len(s)
    return float(m / se) if se > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--asof", default="2026-10-02")
    ap.add_argument("--md", help="결과 마크다운 저장")
    args = ap.parse_args()
    end = pd.Timestamp(args.asof)
    wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    grp = dict(zip(wl["ticker"], wl["group"]))

    rows = []
    for t in wl["ticker"]:
        p = get_profile(t)
        x = compute_indicators(load_prices(f"fdr:{BACKFILL.get(t, t)}", "2004-01-01").loc[:end], p, None)
        c = x["close"]
        ok = x["ma_long"].notna()
        ev = detect(x, p.bb_period)[ok]
        base = pd.DataFrame({"t": t, "date": ev.index, "above200": (c > x["ma_long"])[ok].to_numpy(),
                             **{f"r{h}": (c.shift(-h) / c - 1)[ok].to_numpy() for h in H}})
        rows.append(base.assign(ev=ALL))
        for name in ev.columns:
            rows.append(base[ev[name].to_numpy()].assign(ev=name))
        print(f"{t} 완료", file=sys.stderr)
    d = pd.concat(rows, ignore_index=True)
    d["kind"] = d["t"].map(lambda t: "주식" if grp.get(t) in EQUITY else "비주식")
    allday = d[d["ev"] == ALL]
    for h in H:
        d[f"x{h}"] = d[f"r{h}"] - d["t"].map(allday.groupby("t")[f"r{h}"].mean())
        reg = allday.groupby(["t", "above200"])[f"r{h}"].mean()
        d[f"y{h}"] = d[f"r{h}"] - pd.Series(list(zip(d["t"], d["above200"]))).map(reg).to_numpy()

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out(f"# 기술적 이벤트 뒤 수익률 (~{end.date()}, watchlist {len(wl)}종목)\n")
    out("평소 수익 (모든 날 평균): " + " · ".join(f"{h}일 {allday[f'r{h}'].mean():+.2%}" for h in H)
        + " / 상승 확률: " + " · ".join(f"{h}일 {(allday[f'r{h}'].dropna() > 0).mean():.0%}" for h in H) + "\n")

    out("## 1. 이벤트별 초과 수익 (같은 종목 평소 대비, 괄호: 군집 t)\n")
    rec = []
    for name in d["ev"].value_counts().index.drop(ALL):
        g = d[d["ev"] == name]
        r = {"이벤트": name, "건수": len(g)}
        for h in H:
            s = g[f"x{h}"].dropna()
            r[f"{h}일"] = f"{s.mean():+.2%} ({cluster_t(s, g.loc[s.index, 'date'], 'M' if h <= 20 else 'Q'):+.1f})"
        a, b = (g[g["date"] < "2018-01-01"]["x20"].mean(), g[g["date"] >= "2018-01-01"]["x20"].mean())
        r["20일 ~2017 / 2018~"] = f"{a:+.2%} / {b:+.2%}"
        r["20일 움직임 크기 (평소=1)"] = f"{g['r20'].abs().mean() / allday['r20'].abs().mean():.2f}"
        rec.append(r)
    out(pd.DataFrame(rec).to_markdown(index=False) + "\n")

    out("## 2. 200일선 국면별 (같은 종목·같은 국면 평소 대비, 괄호: 군집 t)\n")
    rec = []
    for name in ("RSI 30 진입", "ADX 25↑ 하락 방향", "볼린저 하단 이탈", "20일선 하향 이탈", "MACD 하향 교차",
                 "밴드폭 6개월 최저"):
        for above in (True, False):
            g = d[(d["ev"] == name) & (d["above200"] == above)].dropna(subset=["y5", "y20"])
            yr5, yr20 = g.groupby(g["date"].dt.year)["y5"].mean(), g.groupby(g["date"].dt.year)["y20"].mean()
            rec.append({"이벤트": name, "200일선": "위" if above else "아래", "건수": len(g),
                        "5일": f"{g['y5'].mean():+.2%} ({cluster_t(g['y5'], g['date'], 'M'):+.1f})",
                        "20일": f"{g['y20'].mean():+.2%} ({cluster_t(g['y20'], g['date'], 'M'):+.1f})",
                        "60일": f"{g['y60'].mean():+.2%}",
                        "5일 플러스 연도": f"{(yr5 > 0).sum()}/{len(yr5)}", "20일 플러스 연도": f"{(yr20 > 0).sum()}/{len(yr20)}"})
    out(pd.DataFrame(rec).to_markdown(index=False) + "\n")

    out("## 3. RSI 30 진입: 종류·기간별 (이벤트 뒤 평균 / 같은 묶음 평소)\n")
    g = d[d["ev"] == "RSI 30 진입"]
    rec = []
    per = {"~2012": (0, 2012), "2013~19": (2013, 2019), "2020~": (2020, 2100)}
    for label, m, mb in ([("주식", g["kind"] == "주식", allday["kind"] == "주식"),
                          ("비주식", g["kind"] == "비주식", allday["kind"] == "비주식")]
                         + [(k, g["date"].dt.year.between(*v), allday["date"].dt.year.between(*v)) for k, v in per.items()]):
        r = {"구분": label, "건수": int(m.sum())}
        for h in (5, 20, 60):
            r[f"{h}일"] = f"{g[m][f'r{h}'].mean():+.2%} / {allday[mb][f'r{h}'].mean():+.2%}"
        rec.append(r)
    out(pd.DataFrame(rec).to_markdown(index=False) + "\n")
    yr = g.groupby(g["date"].dt.year).agg(n=("x5", "size"), x5=("x5", "mean"))
    out("연도별 5일 초과: " + " · ".join(f"{y} {v:+.1%}({n})" for y, n, v in yr.itertuples()) + "\n")

    out("## 4. 밴드폭 6개월 최저 뒤 움직임 크기 (20일 수익의 절댓값)\n")
    vol = {}
    for name in (ALL, "밴드폭 6개월 최저"):
        g = d[d["ev"] == name]
        vol[name] = {"건수": len(g), "20일 움직임 크기 평균": f"{g['r20'].abs().mean():.2%}",
                     "10% 넘게 움직인 비율": f"{(g['r20'].abs() > 0.10).mean():.0%}"}
    out(pd.DataFrame(vol).T.to_markdown() + "\n")

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
