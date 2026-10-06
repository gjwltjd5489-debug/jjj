"""빅테크 칸 목록 갱신: 나스닥 상장 종목 중 지난 1년 평균 거래대금 상위 10개 → config/bigtech.csv

매년 마지막 거래일 장 마감 뒤(또는 1월 첫 주) 한 번 돌리고 커밋한다. 규칙: scoring/bigtech.py, 검증: docs/topk.md 7장.

예)
  python scripts/update_bigtech.py                 # 올해 마지막 거래일이 지났으면 내년 목록, 아니면 올해 목록
  python scripts/update_bigtech.py --years 2006-2026   # 과거 목록 전체 다시 만들기
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring.bigtech import load_pool, membership_year, rank_dollar_volume, save_members, TOP_N  # noqa: E402
from scoring.sources import load_prices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", help="적용 연도 범위 (예: 2006-2026). 비우면 지금 필요한 한 해")
    args = ap.parse_args()
    if args.years:
        a, _, b = args.years.partition("-")
        years = list(range(int(a), int(b or a) + 1))
    else:
        years = [membership_year(date.today())]
    pool = load_pool()
    start = f"{min(years) - 3}-01-01"
    prices = {}
    for t in pool["ticker"]:
        try:
            prices[t] = load_prices(f"fdr:{t}", start)
        except Exception:
            print(f"{t}: 가격 없음 (건너뜀)", file=sys.stderr)
    rows = {}
    for y in years:
        rows[y] = rank_dollar_volume(prices, y, pool).head(TOP_N)
        print(f"{y}: " + ", ".join(f"{t} {v:,.0f}" for t, v in rows[y].items()) + " (백만 달러/일)")
    save_members(rows)


if __name__ == "__main__":
    main()
