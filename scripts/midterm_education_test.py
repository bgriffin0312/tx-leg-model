"""
midterm_education_test.py

Does a fixed midterm EDUCATION term earn its place out of sample?

scripts/group_error_estimate.py --education found that, against the open-race
baseline, every clean midterm runs more Democratic in districts with more
college-educated white adults: +13.8 / +17.1 / +13.6 pp per unit of centered
white-BA+ share (2014/2018/2022, se ~3), and that once education is in, the
Black/Hispanic midterm tilt that the race-only term chased mostly disappears.

Term tested (centered on the Texas share, so it moves districts relative to one
another and leaves the statewide level to the environment dial):
    + b_col * (white_col_share - w_TX_white_col)
    [+ race terms, as a variant]

Leave-one-cycle-out, same rows and scoring as scripts/refit_clean_cycles.py.
Also run on the presidential baseline as a robustness check: if the effect is a
quirk of the open-race composite it should vanish there.

Usage:
  python scripts/midterm_education_test.py
  python scripts/midterm_education_test.py --race-edu   # also test race x education cells

Race x education cells (2026-10-07, Brennan asked): splitting Hispanic and
Black by BA+ as well (Catalist-style cells, ACS C15002I/B) does WORSE out of
sample at every step -- RMSE 3.13 (white edu only) -> 3.52 (+Hispanic split)
-> 3.85 (full cells) on 259 races. ~100 district results per cycle cannot
identify six cells; Black college coefficients swing -37 to +41 by cycle.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.argv.append("--education")          # group_error_estimate reads this at import
import group_error_estimate as ge  # noqa: E402
import refit_clean_cycles as rc  # noqa: E402


def prep(baseline: str) -> pd.DataFrame:
    w = ge.texas_weights()
    d = ge.attach_shares(rc.load(rc.bb.PROC / "phase1_dataset.csv", baseline))
    for g in ("white_col", "black", "hispanic", "other"):
        d[f"c_{g}"] = d[g] - w[g]
    return d.dropna(subset=["c_white_col", "c_black", "c_hispanic"])


def run(label, rows, specs):
    print(f"\n=== {label}, n={len(rows)} ===")
    out = []
    for name, terms in specs.items():
        res, resid = rc.loo(rows, terms)
        c, se, _ = rc.ols(rows, terms)
        comp = [t for t in terms if t.startswith("c_")]
        if comp:
            print(f"  {name}: " + ", ".join(f"{t} {c[t] * 100:+.2f} (se {se[t] * 100:.2f})" for t in comp)
                  + f"   pass-through {c['x']:.3f}")
        col = rows.loc[resid.index, "white_col"]
        hi, lo = col >= col.quantile(0.8), col <= col.quantile(0.2)
        out.append({"spec": name, "rmse": np.sqrt((resid ** 2).mean()),
                    "brier": res.brier.mean(), "abs_seat_err": res.seat_err.abs().mean(),
                    "resid_top20%col": resid[hi].mean() * 100, "resid_bot20%col": resid[lo].mean() * 100,
                    **{f"lvl_{r.cycle}": r.mean_resid * 100 for r in res.itertuples()}})
    print(pd.DataFrame(out).round(4).to_string(index=False))


def main():
    pd.set_option("display.width", 230)
    for baseline in ("open", "pres"):
        d = prep(baseline)
        full = d.dropna(subset=rc.FIN)
        inc = rc.BASE + rc.INC
        ship = inc + ["challenger_viability_flag", "ie_c"]
        run(f"{baseline.upper()} baseline, no-finance sample (all three cycles real)", d, {
            "+inc": inc,
            "+inc+edu": inc + ["c_white_col"],
            "+inc+edu+race": inc + ["c_white_col", "c_black", "c_hispanic"],
        })
        if baseline == "open":
            run("OPEN baseline, finance sample (shipped spec)", full, {
                "SHIP": ship, "SHIP+edu": ship + ["c_white_col"]})


def race_edu_cells():
    w = ge.texas_weights()
    d = ge.attach_shares(rc.load(rc.bb.PROC / "phase1_dataset.csv", "open"))
    cells = ("white_col", "black_noncol", "black_col", "hispanic_noncol", "hispanic_col", "other")
    for g in cells:
        d[f"c_{g}"] = d[g] - w[g]
    d = d.dropna(subset=[f"c_{g}" for g in cells])
    inc = rc.BASE + rc.INC
    run("OPEN baseline, race x education cells (no-finance sample)", d, {
        "+inc": inc,
        "+inc+white edu": inc + ["c_white_col"],
        "+inc+white+Hispanic edu": inc + ["c_white_col", "c_hispanic_col", "c_hispanic_noncol"],
        "+inc+full cells": inc + [f"c_{g}" for g in cells],
    })


if __name__ == "__main__":
    if "--race-edu" in sys.argv:
        race_edu_cells()
        sys.exit()
    main()
