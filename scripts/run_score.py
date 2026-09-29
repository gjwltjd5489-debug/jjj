"""점수 계산 (최신 날짜 또는 지정 날짜의 점수표).

데이터 지정(spec): CSV 경로 | fdr:심볼 | sample:nasdaq  (자세한 형식은 scoring/sources.py)

예)
  python scripts/run_score.py data/QQQ.csv                          # v0, Investing.com CSV
  python scripts/run_score.py data/QQQ.csv --card v3 --bench data/SPY.csv
  python scripts/run_score.py fdr:QQQ --card all --bench fdr:SPY --vix fdr:VIX --us10y fdr:FRED:DGS10 --hy fdr:FRED:BAMLH0A0HYM2
  python scripts/run_score.py sample:nasdaq --bench sample:sp500 --card v2 --date 2008-10-10 --eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring import get_profile  # noqa: E402
from scoring.cards import CARDS, CATEGORY_NAMES, get_card  # noqa: E402
from scoring.evaluate import band_stats, fmt_pct, strategy_stats  # noqa: E402
from scoring.score import breakdown, compute_all  # noqa: E402
from scoring.sources import load_prices, load_series  # noqa: E402


def add_data_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("data", nargs="+", help="가격 데이터: CSV 경로(여러 개 가능) | fdr:QQQ | sample:nasdaq")
    ap.add_argument("--ticker", default="QQQ", help="프로파일 이름 (기본 QQQ)")
    ap.add_argument("--start", help="이 날짜 이후 데이터만 사용")
    ap.add_argument("--bench", help="상대강도 벤치마크 (예: data/SPY.csv | fdr:SPY | sample:sp500)")
    ap.add_argument("--vix", help="VIX (예: fdr:VIX | CBOE VIX_History.csv)")
    ap.add_argument("--us10y", help="미 10년물 금리 (예: fdr:FRED:DGS10)")
    ap.add_argument("--hy", help="하이일드 스프레드 (예: fdr:FRED:BAMLH0A0HYM2)")


def load_inputs(args):
    df = load_prices(args.data, args.start)
    bench = load_series(args.bench)
    ext = {k: load_series(getattr(args, k)) for k in ("vix", "us10y", "hy")}
    ext = {k: v for k, v in ext.items() if v is not None}
    return df, bench, ext


def pick_row(out: pd.DataFrame, date: str | None) -> pd.Series | None:
    valid = out.dropna(subset=["total"])
    if valid.empty:
        return None
    return valid.loc[:date].iloc[-1] if date else valid.iloc[-1]


def print_card(out: pd.DataFrame, card_name: str, date: str | None) -> None:
    card = get_card(card_name)
    row = pick_row(out, date)
    if row is None:
        print("점수를 계산할 만큼 데이터가 없습니다 (최소 약 280거래일 필요).")
        return
    print(f"\n== {card.title}")
    print(f"[{row.name.date()}] 종가 {row['close']:.2f}  총점 {row['total']:.1f}"
          + (f" (최근 {card.smooth}일 평균 {row['total_s']:.1f})" if card.smooth > 1 else "")
          + f"  → {row['signal']}")
    if row["max_used"] < 100:
        print(f"  (데이터 없는 선택 항목 제외: {row['max_used']:.0f}점 만점 → 100점 환산)")
    if card.gate is not None and not row["gate"]:
        print(f"  (게이트 미충족: {card.gate_desc})")
    for cat in card.categories:
        v = row[cat]
        print(f"  {CATEGORY_NAMES.get(cat, cat):<12} {'-' if pd.isna(v) else f'{v:.1f}':>5} / {card.category_max(cat):.0f}")
    print()
    bd = breakdown(row, card)
    bd["points"] = bd["points"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    print(bd.to_string(index=False))


def print_eval(out: pd.DataFrame) -> None:
    table = fmt_pct(band_stats(out), ["share", "avg_20d", "win_20d", "avg_60d", "win_60d"])
    print("\n점수 구간별 이후 수익률 (구간 겹침·배당 미포함, 참고용)")
    print(table.to_string(index=False))
    st = strategy_stats(out)
    if st:
        print(f"\n이벤트 전략(BUY~SELL 보유, 편도 0.05%) {st['start']} ~ {st['end']}: "
              f"CAGR {st['cagr'] * 100:.1f}%  MDD {st['mdd'] * 100:.1f}%  노출 {st['exposure'] * 100:.0f}%  "
              f"매매 {st['round_trips']}회 | 보유: CAGR {st['bh_cagr'] * 100:.1f}%  MDD {st['bh_mdd'] * 100:.1f}%")
    events = out.loc[out["event"] != "", ["close", "total_s", "event"]].tail(8)
    if not events.empty:
        print("\n최근 신호 발생일")
        print(events.to_string())
    vr = out["vr"].dropna()
    if not vr.empty:
        q = vr.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print("\nVR 분포: " + "  ".join(f"p{int(k * 100)}={v:.0f}" for k, v in q.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="100점 매수/매도 점수")
    add_data_args(ap)
    ap.add_argument("--card", default="v0", help=f"카드 ({', '.join(CARDS)}, all)")
    ap.add_argument("--date", help="해당 날짜(또는 직전 거래일) 점수표 출력")
    ap.add_argument("--eval", action="store_true", help="점수 구간별 이후 수익률·전략 성과 출력")
    ap.add_argument("--out", help="결과 CSV 저장 경로 (카드 하나일 때)")
    args = ap.parse_args()

    profile = get_profile(args.ticker)
    df, bench, ext = load_inputs(args)
    print(f"{profile.name}: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}행)"
          + (f"  벤치마크 {len(bench)}행" if bench is not None else "")
          + (f"  외부: {', '.join(ext)}" if ext else ""))

    names = list(CARDS) if args.card == "all" else [args.card]
    results = compute_all(df, profile, names, bench, ext)

    if len(names) > 1:
        rows = []
        for name, out in results.items():
            row = pick_row(out, args.date)
            if row is not None:
                rows.append({"card": name, "date": row.name.date(), "total": row["total"],
                             "signal": row["signal"], "title": get_card(name).title})
        print()
        print(pd.DataFrame(rows).to_string(index=False))
    for name, out in results.items():
        print_card(out, name, args.date)
        if args.eval:
            print_eval(out)
    if args.out and len(names) == 1:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results[names[0]].to_csv(args.out, encoding="utf-8-sig")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
