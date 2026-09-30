"""최근 1년 매수·매도 신호 사후 점검 + 보완 규칙 검증 (watchlist 전 종목, 실제 데이터).

1. 최근 1년 종목별 성과: 계속 보유 vs 체크리스트 점수 규칙
2. 신호 뒤 수익률: 과거(시작~점검 시작 전날) vs 최근 1년, 무조건부 평균과 비교
3. 매년 같은 날짜로 자른 1년 구간의 초과수익(규칙 − 보유) 분포 — 최근 1년이 이례적이었나
4. 매도 휩쏘(매도가보다 비싸게 재매수)와 매매 단위 성과, 매도 시 200일선 위/아래
5. 최근 1년 재진입이 막힌 날(과열·고변동·200일선 아래)과 그동안의 가격 변화
6. 보완 규칙 후보를 2006~2017 / 2018~점검 전 / 최근 1년으로 나눠 검증 (최근 1년에만 맞춘 규칙은 과최적화)
7. 배점: 성향별 정보량(2006~2017), 배점 후보 × 퇴출 방식, 점수 자체의 위험 구분력과 안정성
8. 보유 종목 매도 기준: 현재 배점에서 더 일찍 파는 규칙들(퇴출선 50·60, 보유 중 최고 점수 대비, 가격 손절)

1~6장은 점검 기간에 실제로 쓰던 이전 규칙(동일 배점, 어디서나 40 퇴출)으로 계산한다.

예)
  python scripts/review_signals.py --asof 2026-09-29 --md docs/results/signal_review_eval.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring import get_profile  # noqa: E402
from scoring.checklist_score import EQUAL, RULE, WEIGHTS, checklist_score, strategy_returns  # noqa: E402
from scoring.indicators import atr  # noqa: E402
from scoring.portfolio import perf_stats  # noqa: E402
from scoring.score import compute_indicators  # noqa: E402
from scoring.sources import load_prices  # noqa: E402

BACKFILL = {"PDBC": "PDBC+DBC"}
COST = 0.0005
HORIZONS = (5, 20, 60)

# 점검 기간에 쓰던 규칙: 동일 배점, 200일선 위에서도 40이면 퇴출
OLD = replace(RULE, weights=tuple(EQUAL.items()), exit_above200=RULE.exit)
FAM = list(WEIGHTS)
WEIGHT_SETS = {
    "W0 동일 (이전)": EQUAL,
    "W1 장기 ×2": {**EQUAL, "장기 추세": 2.0},
    "W2 시간축 계층 (장기 3·중기 2)": {**EQUAL, "장기 추세": 3.0, "중기 추세": 2.0},
    "W3 모멘텀·거래량 ×0.5": {**EQUAL, "모멘텀": 0.5, "거래량": 0.5},
    "W4 장기 ×2 + 모멘텀·거래량 ×0.5 (채택)": WEIGHTS,
    "W5 데이터형 (2006~2017 회귀)": {"장기 추세": 2.5, "중기 추세": 1.0, "모멘텀": 0.5, "추세 강도": 1.0, "거래량": 0.0, "상대강도": 0.0},
}

# 보완 규칙 후보 — exit_above: 200일선 위일 때 퇴출선 (None = 점수로 퇴출하지 않음)
VARIANTS = {
    "기본 (현재 규칙)": {},
    "A 200일선 위에선 점수 퇴출 안 함": dict(exit_above=None),
    "A2 200일선 위 퇴출선 25": dict(exit_above=25.0),
    "B 고변동 차단 해제": dict(block_hv=False),
    "A+B": dict(exit_above=None, block_hv=False),
    "A2+B": dict(exit_above=25.0, block_hv=False),
    "D 퇴출에 50일선 이탈 확인": dict(need_below_ma50=True),
    "E 추적 손절(고점 − 3ATR) 추가": dict(trail=3.0),
}
# 8장: 현재 배점(W4)에서 보유 종목 매도 기준 후보
EXITS = {
    "현재 (200일선 아래 + 40)": dict(exit_above=None),
    "200일선 아래 + 50": dict(exit_above=None, exit_below=50.0),
    "200일선 아래 + 60": dict(exit_above=None, exit_below=60.0),
    "200일선 아래면 점수 무관": dict(exit_above=None, exit_below=101.0),
    "어디서나 50": dict(exit_above=50.0, exit_below=50.0),
    "어디서나 60": dict(exit_above=60.0, exit_below=60.0),
    "현재 + 보유 중 최고 점수 −30": dict(exit_above=None, trail_score=30.0),
    "현재 + 보유 중 최고 점수 −40": dict(exit_above=None, trail_score=40.0),
    "현재 + 매수 때 점수 −30": dict(exit_above=None, from_entry=30.0),
    "현재 + 매수가 −2ATR 손절": dict(exit_above=None, stop_entry_atr=2.0),
    "현재 + 보유 중 고점 대비 −10%": dict(exit_above=None, trail_pct=0.10),
    "현재 + 보유 중 고점 대비 −15%": dict(exit_above=None, trail_pct=0.15),
}
CRISES = {"2008 금융위기": ("2007-10-01", "2009-03-31"), "2011 유럽위기": ("2011-04-01", "2011-12-31"),
          "2015-16 조정": ("2015-05-01", "2016-03-01"), "2018 4분기": ("2018-09-01", "2018-12-31"),
          "2020 코로나": ("2020-02-01", "2020-04-30"), "2022 긴축": ("2022-01-01", "2022-10-31"),
          "2025 관세 충격": ("2025-02-01", "2025-05-31")}


def run_variant(o: pd.DataFrame, ma50: pd.Series, a14: pd.Series, *, exit_above=OLD.exit, exit_below=OLD.exit,
                block_hv=RULE.block_highvol, block_oh=RULE.block_overheat, trail=None,
                need_below_ma50=False, trail_score=None, from_entry=None, stop_entry_atr=None,
                trail_pct=None) -> pd.Series:
    """checklist_score 의 run_rule 을 넓힌 상태 기계 (보유 1 / 대기 0).

    trail: 보유 중 최고 종가 − trail×ATR 아래면 매도 · trail_pct: 최고 종가 대비 이 비율 넘게 빠지면 매도
    trail_score: 보유 중 최고 3일 평균 점수보다 이만큼 낮아지면 매도 · from_entry: 매수 때 점수보다 이만큼 낮아지면 매도
    stop_entry_atr: 매수가 − 이 배수×ATR(매수일) 아래면 매도
    """
    s, ab = o["score_s"].to_numpy(), o["above200"].to_numpy()
    oh, hv, c = o["overheat"].to_numpy(), o["highvol"].to_numpy(), o["close"].to_numpy()
    m50, at = ma50.to_numpy(), a14.to_numpy()
    st = np.full(len(s), np.nan)
    cur, started, peak = 0.0, False, np.nan
    peak_s = entry_s = stop = np.nan
    for k in range(len(s)):
        if np.isnan(s[k]):
            if started:
                st[k] = cur
            continue
        started = True
        if cur == 0:
            if s[k] >= RULE.entry and ab[k] and not (block_oh and oh[k]) and not (block_hv and hv[k]):
                cur, peak, peak_s, entry_s = 1.0, c[k], s[k], s[k]
                stop = c[k] - stop_entry_atr * at[k] if stop_entry_atr is not None else np.nan
        else:
            peak, peak_s = max(peak, c[k]), max(peak_s, s[k])
            thr = exit_above if ab[k] else exit_below
            out = thr is not None and s[k] <= thr and (not need_below_ma50 or c[k] < m50[k])
            if trail is not None and not np.isnan(at[k]) and c[k] < peak - trail * at[k]:
                out = True
            if trail_pct is not None and c[k] < peak * (1 - trail_pct):
                out = True
            if trail_score is not None and s[k] <= peak_s - trail_score:
                out = True
            if from_entry is not None and s[k] <= entry_s - from_entry:
                out = True
            if stop_entry_atr is not None and c[k] < stop:
                out = True
            if out:
                cur = 0.0
        st[k] = cur
    return pd.Series(st, index=o.index)


def state_returns(close: pd.Series, state: pd.Series) -> pd.Series:
    r = close.pct_change().fillna(0.0)
    pos = state.fillna(0.0).shift(1).fillna(0.0)
    return pos * r - pos.diff().abs().fillna(0.0) * COST


def mdd(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=str(ROOT / "config" / "watchlist.csv"))
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--asof", default="2026-09-29", help="점검 마지막 날 (이날까지 1년을 점검)")
    ap.add_argument("--md", help="결과 마크다운 저장")
    args = ap.parse_args()

    end = pd.Timestamp(args.asof)
    A = end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    split = pd.Timestamp("2018-01-01")
    periods = [("2006~2017", pd.Timestamp("2006-01-01"), split - pd.Timedelta(days=1)),
               (f"2018~{(A - pd.Timedelta(days=1)).date()}", split, A - pd.Timedelta(days=1)),
               ("최근 1년", A, end)]
    wl = pd.read_csv(args.watchlist, dtype=str).fillna("")
    cache: dict[str, pd.DataFrame] = {}

    def prices(t: str) -> pd.DataFrame:
        if t not in cache:
            cache[t] = load_prices(f"fdr:{BACKFILL.get(t, t)}", args.start).loc[:end]
        return cache[t]

    tk, ev, years, blocks, vrows = [], [], [], [], []
    port: dict[str, list[pd.Series]] = {v: [] for v in VARIANTS} | {"보유": []}
    fam, wrows, quality, halves = [], [], {w: [] for w in WEIGHT_SETS}, []
    xport: dict[str, list[pd.Series]] = {k: [] for k in EXITS}
    xrows, xtrades = [], []
    wport: dict[str, list[pd.Series]] = {}
    for row in wl.itertuples(index=False):
        t = row.ticker
        df = prices(t)
        bench = prices(row.bench)["Close"] if row.bench else None
        p = get_profile(t)
        o = checklist_score(df, p, bench, OLD)
        x = compute_indicators(df, p, bench)
        a14 = atr(df, 14)
        first = o["score_s"].first_valid_index()
        o, x, a14 = o.loc[first:], x.loc[first:], a14.loc[first:]
        c, idx = o["close"], o.index
        bh = c.pct_change().fillna(0.0)
        sr = strategy_returns(o, COST)

        # 1. 최근 1년
        W = o.loc[A:]
        tk.append({"종목": t, "보유": (1 + bh.loc[A:]).prod() - 1, "규칙": (1 + sr.loc[A:]).prod() - 1,
                   "보유 MDD": mdd(bh.loc[A:]), "규칙 MDD": mdd(sr.loc[A:]), "노출": W["state"].mean(),
                   "매수": int((W["event"] == "BUY").sum()), "매도": int((W["event"] == "SELL").sum())})

        # 2·4. 신호별
        ev_idx = [k for k in range(len(o)) if o["event"].iloc[k]]
        for j, k in enumerate(ev_idx):
            nxt = ev_idx[j + 1] if j + 1 < len(ev_idx) else None
            seg = c.iloc[k:(nxt if nxt is not None else len(c)) + 1]
            rec = {"t": t, "date": idx[k], "ev": o["event"].iloc[k], "last": idx[k] >= A,
                   "above200": bool(o["above200"].iloc[k]), "closed": nxt is not None,
                   "next_px": (c.iloc[nxt] if nxt is not None else c.iloc[-1]) / c.iloc[k] - 1,
                   "next_days": (nxt - k) if nxt is not None else np.nan,
                   "seg_min": seg.min() / c.iloc[k] - 1,
                   "dd20": c.iloc[k] / c.iloc[max(0, k - 20):k + 1].max() - 1}
            for h in HORIZONS:
                rec[f"f{h}"] = c.iloc[k + h] / c.iloc[k] - 1 if k + h < len(c) else np.nan
            ev.append(rec)
        for last in (False, True):
            m = (idx >= A) if last else (idx < A)
            ev.append({"t": t, "ev": "ALL", "last": last,
                       **{f"f{h}": (c.shift(-h) / c - 1)[m].mean() for h in HORIZONS}})

        # 3. 매년 같은 날짜로 자른 1년 구간
        y = A - pd.DateOffset(years=1)
        while y >= first:
            seg_s, seg_b = sr.loc[y:y + pd.DateOffset(years=1) - pd.Timedelta(days=1)], \
                bh.loc[y:y + pd.DateOffset(years=1) - pd.Timedelta(days=1)]
            if len(seg_b) >= 240:
                years.append({"t": t, "y0": y, "ex": (1 + seg_s).prod() - (1 + seg_b).prod(),
                              "bh": (1 + seg_b).prod() - 1, "strat": (1 + seg_s).prod() - 1})
            y -= pd.DateOffset(years=1)
        years.append({"t": t, "y0": A, "ex": (1 + sr.loc[A:]).prod() - (1 + bh.loc[A:]).prod(),
                      "bh": (1 + bh.loc[A:]).prod() - 1, "strat": (1 + sr.loc[A:]).prod() - 1})

        # 5. 재진입 차단
        blk = (W["state"] == 0) & (W["blocked"] != "")
        before = o["state"].loc[:A - pd.Timedelta(days=1)]
        held_prev = W["state"].shift(1).fillna(before.iloc[-1] if len(before) else 0.0)  # 그날 수익에 적용되는 포지션
        wr = c.loc[A:].pct_change().fillna(0.0)
        reasons = W["blocked"][blk].str.split("·").explode().value_counts()
        blocks.append({"종목": t, "현금 보유 일수": int((held_prev == 0).sum()),
                       "현금일 가격 변화": (1 + wr[held_prev == 0]).prod() - 1,
                       "막힌 일수": int(blk.sum()),
                       "막힌 다음 날 가격 변화": (1 + wr[blk.shift(1, fill_value=False)]).prod() - 1,
                       "고변동": int(reasons.get("고변동", 0)), "과열": int(reasons.get("과열", 0)),
                       "200일선 아래": int(reasons.get("200일선 아래", 0))})

        # 6. 보완 규칙
        port["보유"].append(bh.rename(t))
        for vn, kw in VARIANTS.items():
            st = run_variant(o, x["ma_mid"], a14, **kw)
            rr = state_returns(c, st)
            port[vn].append(rr.rename(t))
            for pn, a, b in periods:
                seg = rr.loc[a:b]
                if len(seg) < 200:
                    continue
                ps, pb, pos = perf_stats(seg), perf_stats(bh.loc[a:b]), st.loc[a:b].fillna(0.0)
                vrows.append({"v": vn, "t": t, "p": pn, "cagr": ps["cagr"], "mdd": ps["mdd"], "sharpe": ps["sharpe"],
                              "bh_cagr": pb["cagr"], "bh_mdd": pb["mdd"], "bh_sharpe": pb["sharpe"],
                              "expo": pos.mean(), "tpy": (pos.diff() > 0).sum() / (len(seg) / 252)})
        # 7. 배점: 성향별 정보량, 배점 후보 × (이전 퇴출 / A), 점수 품질
        fmin = c[::-1].rolling(60, min_periods=20).min()[::-1].shift(-1) / c - 1
        fvol = np.log(c).diff().rolling(20).std().shift(-20) * np.sqrt(252)
        f60 = c.shift(-60) / c - 1
        fam.append(pd.DataFrame({"t": t, "fmin": fmin, "fvol": fvol, "f60": f60,
                                 **{f: o.get(f"f_{f}", pd.Series(np.nan, index=idx)) for f in FAM}}))
        for wn, w in WEIGHT_SETS.items():
            ow = checklist_score(df, p, bench, replace(OLD, weights=tuple(w.items()))).loc[first:]
            sc = ow["score_s"]
            cross = ((sc >= RULE.entry) != (sc.shift(1) >= RULE.entry)) & sc.notna() & sc.shift(1).notna()
            quality[wn].append(pd.DataFrame({"t": t, "s": sc, "fmin": fmin, "fvol": fvol, "f60": f60,
                                             "cross": cross.astype(float)}))
            sts = {"이전 퇴출": run_variant(ow, x["ma_mid"], a14), "A": run_variant(ow, x["ma_mid"], a14, exit_above=None)}
            sts["반반"] = (sts["이전 퇴출"] + sts["A"]) / 2  # 40 이하에 절반, 200일선 이탈에 나머지
            for rn, st in sts.items():
                rr = state_returns(c, sts["이전 퇴출"]) / 2 + state_returns(c, sts["A"]) / 2 if rn == "반반" \
                    else state_returns(c, st)
                wport.setdefault(f"{wn} + {rn}", []).append(rr.rename(t))
                for pn, a, b in periods:
                    seg = rr.loc[a:b]
                    if len(seg) < 200:
                        continue
                    ps, pos = perf_stats(seg), st.loc[a:b].fillna(0.0)
                    wrows.append({"w": wn, "r": rn, "t": t, "p": pn, "sharpe": ps["sharpe"],
                                  "tpy": (pos.diff() > 0).sum() / (len(seg) / 252)})
            if w is WEIGHTS:  # 반반의 '절반 매도'(200일선 위에서 40 이하) 이후 결과
                gap = sts["A"].fillna(0.0) - sts["이전 퇴출"].fillna(0.0)
                for d in gap.index[(gap > 0) & (gap.shift(1).fillna(0.0) <= 0)]:
                    k = c.index.get_loc(d)
                    rest = gap.iloc[k:]
                    done = bool((rest <= 0).any())
                    until = rest.index[(rest <= 0).argmax()] if done else rest.index[-1]
                    how = ("진행 중" if not done else "200일선 이탈 → 나머지도 매도" if sts["A"].loc[until] == 0
                           else "점수 회복 → 절반 되사기")
                    halves.append({"last": d >= A, "how": how, "ret": c.loc[until] / c.iloc[k] - 1,
                                   "days": c.index.get_loc(until) - k})
        # 8. 보유 종목 매도 기준 (현재 배점)
        on = checklist_score(df, p, bench).loc[first:]
        for xn, kw in EXITS.items():
            st = run_variant(on, x["ma_mid"], a14, **kw)
            rr = state_returns(c, st)
            xport[xn].append(rr.rename(t))
            for pn, a, b in periods:
                seg = rr.loc[a:b]
                if len(seg) < 200:
                    continue
                ps, pos = perf_stats(seg), st.loc[a:b].fillna(0.0)
                xrows.append({"v": xn, "t": t, "p": pn, "sharpe": ps["sharpe"],
                              "tpy": (pos.diff() > 0).sum() / (len(seg) / 252)})
            chg = st.diff()
            buys, sells = list(st.index[chg == 1]), list(st.index[chg == -1])
            for bd in buys:
                sd = next((z for z in sells if z > bd), None)
                if sd is None or bd >= A:
                    continue
                seg = c.loc[bd:sd]
                nb = next((z for z in buys if z > sd), None)
                xtrades.append({"v": xn, "entry_s": on["score_s"].loc[bd], "exit_s": on["score_s"].loc[sd],
                                "ret": c[sd] / c[bd] - 1, "from_peak": c[sd] / seg.max() - 1, "days": len(seg) - 1,
                                "rebuy": (c[nb] / c[sd] - 1) if nb is not None else np.nan})
        print(f"{t} 완료", file=sys.stderr)

    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    def pct(v) -> str:
        return "-" if pd.isna(v) else f"{v * 100:.1f}%"

    def table(df: pd.DataFrame) -> None:
        out(df.to_markdown(index=False))
        out("")

    out(f"# 신호 사후 점검 ({A.date()} ~ {end.date()})\n")
    T = pd.DataFrame(tk)
    out("## 1. 최근 1년 종목별: 계속 보유 vs 점수 규칙\n")
    table(T.assign(**{k: T[k].map(pct) for k in ("보유", "규칙", "보유 MDD", "규칙 MDD", "노출")}))
    out(f"중앙값: 보유 {pct(T['보유'].median())} / 규칙 {pct(T['규칙'].median())} · MDD 보유 {pct(T['보유 MDD'].median())} / "
        f"규칙 {pct(T['규칙 MDD'].median())} · 노출 {pct(T['노출'].median())} · 매수 {T['매수'].sum()}건 · 매도 {T['매도'].sum()}건\n")

    E = pd.DataFrame(ev)
    out("## 2. 신호 뒤 평균 수익률 (ALL = 그 기간 아무 날이나 샀을 때)\n")
    g = E.groupby(["last", "ev"])[[f"f{h}" for h in HORIZONS]].mean()
    n = E[E.ev != "ALL"].groupby(["last", "ev"]).size()
    rows = []
    for (last, e), r in g.iterrows():
        rows.append({"기간": "최근 1년" if last else "과거", "신호": e, "건수": n.get((last, e), "-"),
                     **{f"{h}일 뒤": pct(r[f'f{h}']) for h in HORIZONS}})
    table(pd.DataFrame(rows))

    Y = pd.DataFrame(years)
    hist, last = Y[Y.y0 < A], Y[Y.y0 == A]
    yr = hist.groupby(hist.y0.dt.year).agg(n=("t", "size"), 보유=("bh", "median"), 규칙=("strat", "median"),
                                          초과=("ex", "median")).reset_index().rename(columns={"y0": "시작 연도"})
    out(f"## 3. 매년 {A.month}/{A.day} 시작 1년 구간 — 초과수익(규칙 − 보유) 종목 중앙값\n")
    table(yr.assign(**{k: yr[k].map(pct) for k in ("보유", "규칙", "초과")}))
    rank = (yr["초과"] < last.ex.median()).sum()
    out(f"최근 1년: 보유 {pct(last.bh.median())}, 규칙 {pct(last.strat.median())}, 초과 {pct(last.ex.median())} "
        f"→ 과거 {len(yr)}개 구간 중 초과수익이 이보다 나빴던 구간 {rank}개. "
        f"과거 종목·구간 전체 초과수익 중앙값 {pct(hist.ex.median())}\n")

    S, Bu = E[E.ev == "SELL"], E[E.ev == "BUY"]

    def sell_stats(d: pd.DataFrame) -> dict:
        c_ = d[d.closed]
        return {"건수": len(d), "재매수가 > 매도가 (휩쏘)": pct((c_.next_px > 0).mean()),
                "재매수가 (매도가 대비, 중앙값)": pct(c_.next_px.median()), "재매수까지 거래일": f"{c_.next_days.median():.0f}",
                "매도 뒤 최저 (피한 하락, 중앙값)": pct(d.seg_min.median()), "매도 시 20일 고점 대비": pct(d.dd20.median()),
                "20일 뒤 평균": pct(d.f20.mean()), "60일 뒤 평균": pct(d.f60.mean())}

    def buy_stats(d: pd.DataFrame) -> dict:
        c_ = d[d.closed]
        return {"건수": len(d), "매매 수익 평균": pct(d.next_px.mean()), "매매 수익 중앙값": pct(d.next_px.median()),
                "승률": pct((d.next_px > 0).mean()), "보유 거래일 (중앙값)": f"{c_.next_days.median():.0f}",
                "20일 뒤 평균": pct(d.f20.mean())}

    out("## 4. 매도 휩쏘와 매매 성과\n")
    out("매도 신호 (재매수 = 다음 매수 신호)\n")
    table(pd.DataFrame({"과거": sell_stats(S[~S["last"]]), "최근 1년": sell_stats(S[S["last"]])}).reset_index()
          .rename(columns={"index": "항목"}))
    out("매수 신호 (매수 → 다음 매도까지 한 번의 매매, 진행 중이면 점검 마지막 날 종가)\n")
    table(pd.DataFrame({"과거": buy_stats(Bu[~Bu["last"]]), "최근 1년": buy_stats(Bu[Bu["last"]])}).reset_index()
          .rename(columns={"index": "항목"}))
    rows = []
    for lst in (False, True):
        for a200 in (True, False):
            d = S[(S["last"] == lst) & (S.above200 == a200)]
            c_ = d[d.closed]
            rows.append({"기간": "최근 1년" if lst else "과거", "매도 시점": "200일선 위" if a200 else "200일선 아래",
                         "건수": len(d), "휩쏘 비율": pct((c_.next_px > 0).mean()), "20일 뒤": pct(d.f20.mean()),
                         "60일 뒤": pct(d.f60.mean()), "피한 하락": pct(d.seg_min.median())})
    out("매도 시점이 200일선 위였나\n")
    table(pd.DataFrame(rows))

    Bl = pd.DataFrame(blocks).sort_values("현금일 가격 변화", ascending=False)
    out("## 5. 최근 1년 현금 보유 구간과 재진입 차단\n")
    table(Bl.assign(**{k: Bl[k].map(pct) for k in ("현금일 가격 변화", "막힌 다음 날 가격 변화")}))
    out(f"막힌 사유 합계(일): 고변동 {Bl['고변동'].sum()}, 과열 {Bl['과열'].sum()}, 200일선 아래 {Bl['200일선 아래'].sum()}\n")

    V = pd.DataFrame(vrows)
    out("## 6. 보완 규칙 검증\n")
    for pn, _, _ in periods:
        sub = V[V.p == pn]
        base = sub[sub.v == "기본 (현재 규칙)"].set_index("t")
        gg = sub.groupby("v", sort=False)
        tb = pd.DataFrame({"종목": gg.size(), "CAGR": gg.cagr.median().map(pct), "MDD": gg.mdd.median().map(pct),
                           "샤프": gg.sharpe.median().round(2), "노출": gg.expo.median().map(pct),
                           "매매/년": gg.tpy.median().round(1),
                           "샤프 > 기본": gg.apply(lambda d: f"{(d.set_index('t').sharpe > base.sharpe.reindex(d.t)).sum()}/{len(d)}")})
        b0 = sub[sub.v == "기본 (현재 규칙)"]
        out(f"### {pn} (종목 중앙값) — 보유: CAGR {pct(b0.bh_cagr.median())}, MDD {pct(b0.bh_mdd.median())}, "
            f"샤프 {b0.bh_sharpe.median():.2f}\n")
        table(tb.reset_index().rename(columns={"v": "규칙"}))
    ew = {k: pd.concat(v, axis=1).mean(axis=1, skipna=True) for k, v in port.items()}
    rows = []
    for vn, r in ew.items():
        rec = {"규칙": vn}
        for pn, a, b in periods:
            ps = perf_stats(r.loc[a:b])
            rec[f"{pn} CAGR"], rec[f"{pn} MDD"], rec[f"{pn} 샤프"] = pct(ps["cagr"]), pct(ps["mdd"]), round(ps["sharpe"], 2)
        rows.append(rec)
    out("### 동일가중 포트폴리오 (종목별 결과를 날짜마다 평균)\n")
    table(pd.DataFrame(rows))
    rows = [{"규칙": vn, **{name: pct(mdd(r.loc[a:b])) for name, (a, b) in CRISES.items()}} for vn, r in ew.items()]
    out("### 위기 구간 최대 낙폭 (동일가중 포트폴리오)\n")
    table(pd.DataFrame(rows))

    out("## 7. 배점\n")
    F = pd.concat(fam)
    tr = F[F.index <= periods[0][2]]
    rows = []
    for f in FAM:
        g = tr.groupby(f)[["fvol", "fmin", "f60"]].mean()
        if 1.0 not in g.index or -1.0 not in g.index:
            continue
        flips = np.median([(d[f].dropna().diff().abs() > 0).sum() / (d[f].notna().sum() / 252)
                           for _, d in tr.groupby("t") if d[f].notna().sum() > 252])
        rows.append({"성향": f, "이후 변동성 🟢": pct(g.loc[1, "fvol"]), "🔴": pct(g.loc[-1, "fvol"]),
                     "이후 60일 최저 🟢": pct(g.loc[1, "fmin"]), "🔴 ": pct(g.loc[-1, "fmin"]),
                     "60일 수익 🟢−🔴": pct(g.loc[1, "f60"] - g.loc[-1, "f60"]), "색 바뀜/년": f"{flips:.0f}"})
    out(f"### 성향별 정보량 ({periods[0][0]}, 전 종목 합산)\n")
    table(pd.DataFrame(rows))
    d = tr.dropna(subset=["fmin"]).copy()
    for f in FAM:
        d[f] = d[f].astype(float).fillna(0.0)
    y = (d["fmin"] - d.groupby("t")["fmin"].transform("mean")).astype(float).to_numpy()
    beta = np.linalg.lstsq(np.c_[np.ones(len(d)), d[FAM].to_numpy(float)], y, rcond=None)[0][1:]
    out("이후 60일 최저(종목 평균 제거)를 성향 색으로 회귀한 계수 — 클수록 🟢일 때 하락이 얕음: "
        + ", ".join(f"{f} {b:+.4f}" for f, b in zip(FAM, beta)) + "\n")
    out("### 배점 × 퇴출 방식 (동일가중 포트폴리오: 수익 · 최대낙폭 · 샤프 / 종목 중앙값 매매/년)\n")
    W7 = pd.DataFrame(wrows)
    rows = []
    for key, lst in wport.items():
        r = pd.concat(lst, axis=1).mean(axis=1, skipna=True)
        wn, rn = key.rsplit(" + ", 1)
        rec = {"배점": wn, "퇴출": rn}
        for pn, a, b in periods:
            ps = perf_stats(r.loc[a:b])
            sub = W7[(W7.w == wn) & (W7.r == rn) & (W7.p == pn)]
            rec[pn] = f"{pct(ps['cagr'])} · {pct(ps['mdd'])} · {ps['sharpe']:.2f} / {sub.tpy.median():.1f}"
        for name in ("2008 금융위기", "2020 코로나", "2025 관세 충격"):
            a, b = CRISES[name]
            rec[name] = pct(mdd(r.loc[a:b]))
        rows.append(rec)
    table(pd.DataFrame(rows))
    H = pd.DataFrame(halves)
    if not H.empty:
        out("### 반반 (W4 배점): 200일선 위에서 40 이하로 절반 매도한 뒤\n")
        rows = []
        for lst in (False, True):
            for how, d in H[H["last"] == lst].groupby("how"):
                rows.append({"기간": "최근 1년" if lst else "과거", "결말": how, "건수": len(d),
                             "절반 매도 → 결말까지 가격 변화 평균": pct(d.ret.mean()), "중앙값": pct(d.ret.median()),
                             "걸린 거래일": f"{d.days.median():.0f}"})
        table(pd.DataFrame(rows))
        out("가격 변화가 +면 판 절반을 더 비싸게 되산 것(손해), −면 판 절반이 그만큼 하락을 피한 것(이익).\n")
    out("### 점수 자체의 품질 (3일 평균, 매매 규칙과 무관)\n")
    rows = []
    for wn, lst in quality.items():
        Q = pd.concat(lst)
        for pn, a, b in periods:
            q = Q[(Q.index >= a) & (Q.index <= b)].dropna(subset=["s"])
            hi, lo = q[q.s >= RULE.entry], q[q.s <= RULE.exit]
            ic = q.dropna(subset=["fmin"]).groupby("t").apply(lambda z: z.s.rank().corr(z.fmin.rank())).median()
            yrs = (q.index.max() - q.index.min()).days / 365.25
            rows.append({"배점": wn, "기간": pn, "이후 변동성 ≥70 / ≤40": f"{pct(hi.fvol.mean())} / {pct(lo.fvol.mean())}",
                         "이후 60일 최저 ≥70 / ≤40": f"{pct(hi.fmin.mean())} / {pct(lo.fmin.mean())}",
                         "점수-하락 순위상관": f"{ic:.3f}", "70선 교차/년": f"{q.groupby('t').cross.sum().median() / yrs:.1f}"})
    table(pd.DataFrame(rows))

    out("## 8. 보유 종목 매도 기준 (현재 배점)\n")
    out("### 동일가중 포트폴리오 (수익 · 최대낙폭 · 샤프) / 종목 중앙값 샤프, 샤프가 현재보다 높은 종목\n")
    X = pd.DataFrame(xrows)
    xb = X[X.v == "현재 (200일선 아래 + 40)"].set_index(["t", "p"])
    rows = []
    for xn, lst in xport.items():
        r = pd.concat(lst, axis=1).mean(axis=1, skipna=True)
        j = X[X.v == xn].set_index(["t", "p"]).join(xb, rsuffix="_b")
        rec = {"매도 기준": xn}
        for pn, a, b in periods:
            ps = perf_stats(r.loc[a:b])
            jj = j.xs(pn, level="p")
            rec[pn] = (f"{pct(ps['cagr'])} · {pct(ps['mdd'])} · {ps['sharpe']:.2f} / "
                       f"{jj.sharpe.median():.2f}, {(jj.sharpe > jj.sharpe_b).sum()}/{len(jj)}")
        for name in ("2008 금융위기", "2020 코로나", "2025 관세 충격"):
            a, b = CRISES[name]
            rec[name] = pct(mdd(r.loc[a:b]))
        rows.append(rec)
    table(pd.DataFrame(rows))
    XT = pd.DataFrame(xtrades)
    out(f"### 매매 단위 (점검 시작 전까지 끝난 매매)\n")
    rows = []
    for xn, d in XT.groupby("v", sort=False):
        pos_sum = d.ret[d.ret > 0].sum()
        rows.append({"매도 기준": xn, "매매 수": len(d), "매도 때 점수 (중앙값)": f"{d.exit_s.median():.0f}",
                     "매도가: 보유 중 고점 대비": pct(d.from_peak.median()), "매수가 대비 (중앙값)": pct(d.ret.median()),
                     "매수가 대비 (평균)": pct(d.ret.mean()), "30% 넘게 번 매매": int((d.ret > 0.3).sum()),
                     "상위 10% 매매의 이익 비중": pct(d.ret.sort_values(ascending=False).head(len(d) // 10).sum() / pos_sum),
                     "재매수가 > 매도가": pct((d.rebuy.dropna() > 0).mean()), "보유 거래일": f"{d.days.median():.0f}"})
    table(pd.DataFrame(rows))
    cur = XT[XT.v == "현재 (200일선 아래 + 40)"].copy()
    cur["구간"] = pd.cut(cur.entry_s, [69.9, 80, 90, 101], labels=["70~80", "80~90", "90~100"])
    rows = [{"매수 때 3일 평균": str(k), "매매 수": len(d), "승률": pct((d.ret > 0).mean()), "매수가 대비 평균": pct(d.ret.mean()),
             "중앙값": pct(d.ret.median()), "매도가: 고점 대비": pct(d.from_peak.median()), "보유 거래일": f"{d.days.median():.0f}"}
            for k, d in cur.groupby("구간", observed=True)]
    out("### 현재 규칙: 매수 때 점수 구간별\n")
    table(pd.DataFrame(rows))

    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
