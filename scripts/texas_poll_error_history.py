"""
texas_poll_error_history.py

How wrong have Texas statewide polls been, and in which direction?
Input to the Texas-poll calibration (NEXT-STEPS, 2026-10-07): the weight a
Texas poll average gets against the model's structural statewide number is
sigma_model^2 / (sigma_model^2 + sigma_poll^2), after removing the polls'
historical bias. This script measures sigma_poll and the bias.

Inputs (built from Wikipedia race articles, 2014-2024):
  data/raw/historical/tx_statewide_polls_2014_2024.csv
  data/raw/historical/tx_statewide_results_2014_2024.csv
Build them from the collection CSVs with --merge DIR.

For each race, two poll averages, one poll per pollster (its latest in window):
  final   polls ending in the last 30 days before Election Day
  oct1    polls ending Aug 15 - Oct 1 (when the model would use them)
Margins are two-party D margin, D/(D+R) - R/(D+R), in points.
error = poll margin - actual margin   (positive = polls too Democratic)
Partisan-sponsored polls are excluded by default (--include-partisan to keep).

Usage:
  python scripts/texas_poll_error_history.py --merge <dir-with-agent-csvs>
  python scripts/texas_poll_error_history.py
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "raw" / "historical"
POLLS = HIST / "tx_statewide_polls_2014_2024.csv"
RESULTS = HIST / "tx_statewide_results_2014_2024.csv"
ELECTION_DAY = {2014: date(2014, 11, 4), 2016: date(2016, 11, 8), 2018: date(2018, 11, 6),
                2020: date(2020, 11, 3), 2022: date(2022, 11, 8), 2024: date(2024, 11, 5)}
MIDTERM = {2014, 2018, 2022}
SALIENCE = {"president": "top", "us_senate": "top", "governor": "top",
            "lt_governor": "down", "attorney_general": "down", "comptroller": "down",
            "railroad_commissioner": "down", "generic_us_house": "generic",
            "generic_tx_legislature": "generic"}


def merge(src: Path):
    polls = pd.concat([pd.read_csv(p) for p in sorted(src.glob("polls_*.csv"))], ignore_index=True)
    res = pd.concat([pd.read_csv(p) for p in sorted(src.glob("results_*.csv"))], ignore_index=True)
    polls.to_csv(POLLS, index=False)
    res.to_csv(RESULTS, index=False)
    print(f"merged {len(polls)} polls, {len(res)} results -> {POLLS.name}, {RESULTS.name}")


def d_margin(r, d):
    return (d - r) / (d + r) * 100


def average(p: pd.DataFrame, lo: date, hi: date) -> tuple[float, int]:
    w = p[(p.field_end >= lo) & (p.field_end <= hi)]
    if w.empty:
        return np.nan, 0
    # one per pollster: its latest poll in window; LV over RV on ties
    w = w.assign(pop_rank=w.population.map({"LV": 0, "RV": 1}).fillna(2))
    w = w.sort_values(["pollster", "field_end", "pop_rank"], ascending=[True, False, True])
    w = w.drop_duplicates("pollster")
    return float(w.margin.mean()), len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", type=Path)
    ap.add_argument("--include-partisan", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge)

    polls = pd.read_csv(POLLS)
    res = pd.read_csv(RESULTS)
    polls["field_end"] = pd.to_datetime(polls.field_end, errors="coerce").dt.date
    polls = polls.dropna(subset=["field_end", "r_pct", "d_pct"])
    if not args.include_partisan:
        polls = polls[polls.partisan_flag.isna() | (polls.partisan_flag.astype(str).str.strip() == "")]
    polls["margin"] = d_margin(polls.r_pct, polls.d_pct)
    res["actual"] = d_margin(res.r_pct, res.d_pct)

    rows = []
    for (cyc, race), p in polls.groupby(["cycle", "race"]):
        a = res[(res.cycle == cyc) & (res.race == race)]
        if a.empty:
            continue
        ed = ELECTION_DAY[int(cyc)]
        fin, nf = average(p, ed - timedelta(days=30), ed)
        o1, no = average(p, date(int(cyc), 8, 15), date(int(cyc), 10, 1))
        act = float(a.actual.iloc[0])
        rows.append({"cycle": int(cyc), "race": race, "type": "midterm" if cyc in MIDTERM else "presidential",
                     "salience": SALIENCE.get(race, "?"), "actual_D_margin": act,
                     "final_avg": fin, "n_final": nf, "err_final": fin - act,
                     "oct1_avg": o1, "n_oct1": no, "err_oct1": o1 - act})
    t = pd.DataFrame(rows).sort_values(["cycle", "race"])
    pd.set_option("display.width", 200)
    print("\n=== Texas statewide polls vs results (D margin, points; err + = polls too D) ===")
    print(t.round(1).to_string(index=False))

    print("\n=== Summary (races with >= 2 pollsters in window) ===")
    for col, ncol in (("err_final", "n_final"), ("err_oct1", "n_oct1")):
        s = t[t[ncol] >= 2]
        print(f"\n{col}:")
        for label, g in (("all", s), *s.groupby("type"), *s.groupby("salience")):
            if len(g):
                e = g[col]
                print(f"  {label:<13} races {len(g):>2}  mean {e.mean():+5.1f}  sd {e.std(ddof=1) if len(g) > 1 else float('nan'):4.1f}"
                      f"  rms {np.sqrt((e ** 2).mean()):4.1f}   by cycle: "
                      + "  ".join(f"{c} {v:+.1f}" for c, v in g.groupby("cycle")[col].mean().items()))
    # cycle-level error: races in a cycle share most of their miss
    cyc = t[t.n_oct1 >= 2].groupby(["cycle", "type"]).err_oct1.mean().reset_index()
    print("\nCycle-level as-of-Oct-1 miss (mean over that cycle's races):")
    print(cyc.round(1).to_string(index=False))
    mid = cyc[cyc.type == "midterm"].err_oct1
    print(f"  midterms: mean {mid.mean():+.1f}, sd {mid.std(ddof=1):.1f} (n={len(mid)})")
    compare_to_model(dict(zip(cyc.cycle, cyc.err_oct1)))


# National generic ballot, D margin. Oct-1 = mean of 538 (archived
# generic_ballot_averages.csv; 2014 is a 2020 back-cast) and RCP (Wayback
# captures near Oct 1); actual = two-party House popular vote. Collected
# 2026-10-07; URLs in NEXT-STEPS / the session record.
NAT_OCT1 = {2014: np.mean([-0.64, -2.9]), 2018: np.mean([8.42, 7.4]), 2022: np.mean([1.31, -0.9])}
NAT_ACTUAL = {2014: -5.9, 2018: 8.7, 2022: -2.8}
# Model structural level miss, leave-one-cycle-out, shipped spec with the
# education term (scripts/midterm_education_test.py), share pp x 2 = margin.
STRUCT_MISS = {2014: 2 * 0.4353, 2018: 2 * -0.6686, 2022: 2 * -0.8007}
ENV_MARGIN_PASS = 0.92   # 2 x env coefficient 0.0046


def compare_to_model(tx_oct1: dict):
    """Model's real-time statewide miss (structural + national poll miss passed
    through) vs the Texas polls' Oct-1 miss, midterms only."""
    print("\n=== Model real-time miss vs Texas polls, midterms (margin, + = too D) ===")
    em, ep = [], []
    for y in sorted(NAT_OCT1):
        m = STRUCT_MISS[y] + ENV_MARGIN_PASS * (NAT_OCT1[y] - NAT_ACTUAL[y])
        em.append(m)
        ep.append(tx_oct1[y])
        print(f"  {y}: national Oct-1 miss {NAT_OCT1[y] - NAT_ACTUAL[y]:+.1f}   model {m:+.1f}   TX polls {tx_oct1[y]:+.1f}")
    em, ep = np.array(em), np.array(ep)
    d = ep - em
    w = -(em * d).sum() / (d ** 2).sum()
    print(f"  rms model {np.sqrt((em ** 2).mean()):.1f}, TX polls {np.sqrt((ep ** 2).mean()):.1f}, "
          f"corr {np.corrcoef(em, ep)[0, 1]:+.2f}; error-minimizing weight on TX polls {w:+.2f}")
    for wt in (0.25, 0.5):
        print(f"  weight {wt}: blended rms {np.sqrt((((1 - wt) * em + wt * ep) ** 2).mean()):.2f}")


if __name__ == "__main__":
    main()
