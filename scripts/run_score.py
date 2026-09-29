"""Investing.com CSV로 점수를 계산한다.

예)
  python scripts/run_score.py --ticker QQQ data/QQQ.csv
  python scripts/run_score.py --ticker QQQ data/QQQ_2010_2017.csv data/QQQ_2018_now.csv --eval --out out/qqq_score.csv
  python scripts/run_score.py --ticker QQQ data/QQQ.csv --date 2025-04-08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring import breakdown, compute_score, get_profile, load_investing_csvs  # noqa: E402
from scoring.score import CATEGORIES, CATEGORY_MAX  # noqa: E402


def print_breakdown(out: pd.DataFrame, date: str | None) -> None:
    valid = out.dropna(subset=["total"])
    if valid.empty:
        print("점수를 계산할 만큼 데이터가 없습니다 (최소 약 250거래일 필요).")
        return
    if date:
        row = valid.loc[:date].iloc[-1]
    else:
        row = valid.iloc[-1]
    print(f"\n[{row.name.date()}] 종가 {row['close']:.2f}  총점 {row['total']:.1f} / 100  → {row['signal']}")
    if not row["volume_used"]:
        print("  (거래량 없음: 거래량 20점 제외 후 80점 → 100점 환산)")
    for cat in CATEGORIES:
        v = row[cat]
        print(f"  {cat:<9} {'-' if pd.isna(v) else f'{v:.0f}':>3} / {CATEGORY_MAX[cat]}")
    print()
    print(breakdown(row).to_string(index=False))
    print(f"\n  VR {row['vr']:.1f}%  RSI {row['rsi']:.1f}  MACD {row['macd']:.3f}")


def evaluate(out: pd.DataFrame, horizons=(20, 60)) -> None:
    """점수 구간별 이후 수익률 통계 (참고용, 과최적화 주의)."""
    close = out["close"]
    df = pd.DataFrame({"total": out["total"], "signal": out["signal"]})
    for h in horizons:
        df[f"fwd{h}"] = close.shift(-h) / close - 1
    df = df.dropna(subset=["total"])
    order = ["강한 매도", "매도", "중립", "매수", "강한 매수"]
    rows = []
    for name in order:
        g = df[df["signal"] == name]
        r = {"signal": name, "days": len(g), "share": len(g) / len(df)}
        for h in horizons:
            f = g[f"fwd{h}"].dropna()
            r[f"avg_{h}d"] = f.mean()
            r[f"win_{h}d"] = (f > 0).mean() if len(f) else float("nan")
        rows.append(r)
    base = {"signal": "(전체)", "days": len(df), "share": 1.0}
    for h in horizons:
        f = df[f"fwd{h}"].dropna()
        base[f"avg_{h}d"] = f.mean()
        base[f"win_{h}d"] = (f > 0).mean()
    rows.append(base)
    table = pd.DataFrame(rows)
    pct = [c for c in table.columns if c.startswith(("avg", "win", "share"))]
    table[pct] = table[pct].map(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "-")
    print("\n점수 구간별 이후 수익률 (배당 미포함, 거래비용 미반영)")
    print(table.to_string(index=False))

    events = out.loc[out["event"] != "", ["close", "total", "event"]].tail(10)
    if not events.empty:
        print("\n최근 신호 발생일")
        print(events.to_string())

    vr = out["vr"].dropna()
    if not vr.empty:
        q = vr.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print("\nVR 분포 (구간 보정용): " + "  ".join(f"p{int(k * 100)}={v:.0f}" for k, v in q.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="100점 매수/매도 점수")
    ap.add_argument("csv", nargs="+", help="Investing.com 과거 데이터 CSV (여러 개 가능)")
    ap.add_argument("--ticker", default="QQQ", help="프로파일 이름 (기본 QQQ)")
    ap.add_argument("--date", help="해당 날짜(또는 직전 거래일) 점수표 출력")
    ap.add_argument("--eval", action="store_true", help="점수 구간별 이후 수익률 통계 출력")
    ap.add_argument("--out", help="전체 결과 CSV 저장 경로")
    args = ap.parse_args()

    profile = get_profile(args.ticker)
    df = load_investing_csvs(args.csv)
    print(f"{profile.name}: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행)")
    out = compute_score(df, profile)

    print_breakdown(out, args.date)
    if args.eval:
        evaluate(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, encoding="utf-8-sig")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
