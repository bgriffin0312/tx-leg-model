"""
midterm_composition_test.py

Does a FIXED midterm composition term earn its place out of sample?

scripts/group_error_estimate.py found that, against the open-race baseline
(drawn from presidential-year electorates), midterm residuals run Republican
in proportion to a district's Black and Hispanic CVAP share in all three clean
midterms (Hispanic -8.5/-6.0/-1.8, Black -4.5/-12.1/-8.9 pp per unit centered
share). The likely mechanism is midterm turnout drop-off, which a presidential-
year baseline cannot see. A mean-zero error layer cannot fix a systematic miss,
so test a term in the linear prediction:

    + b_black * (black_share - w_TX_black) + b_hisp * (hisp_share - w_TX_hisp)
    (+ b_other * (other_share - w_TX_other) as a variant)

Same rows and leave-one-cycle-out scoring as scripts/refit_clean_cycles.py.
This is a property of the electorate measured from election results, not a
poll-driven level term like the one deleted on 2026-10-07.

Usage:
  python scripts/midterm_composition_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402
import group_error_estimate as ge  # noqa: E402


def main():
    pd.set_option("display.width", 220)
    w = ge.texas_weights()
    d = ge.attach_shares(rc.load(rc.bb.PROC / "phase1_dataset.csv", "open"))
    for g in ("black", "hispanic", "other"):
        d[f"c_{g}"] = d[g] - w[g]
    d = d.dropna(subset=["c_black", "c_hispanic", "c_other"])
    full = d.dropna(subset=rc.FIN)
    specs = {
        "+inc": rc.BASE + rc.INC,
        "+inc+comp(B,H)": rc.BASE + rc.INC + ["c_black", "c_hispanic"],
        "+inc+comp(B,H,O)": rc.BASE + rc.INC + ["c_black", "c_hispanic", "c_other"],
    }
    ship = rc.BASE + rc.INC + ["challenger_viability_flag", "ie_c"]
    specs_fin = {"SHIP": ship, "SHIP+comp(B,H)": ship + ["c_black", "c_hispanic"]}

    for label, rows, sp in (("no-finance sample (all three cycles real)", d, specs),
                            ("finance sample (shipped spec)", full, specs_fin)):
        print(f"\n=== {label}, n={len(rows)} ===")
        out = []
        for name, terms in sp.items():
            res, resid = rc.loo(rows, terms)
            out.append({"spec": name, "rmse": np.sqrt((resid ** 2).mean()),
                        "brier": res.brier.mean(), "abs_seat_err": res.seat_err.abs().mean(),
                        **{f"lvl_{r.cycle}": r.mean_resid * 100 for r in res.itertuples()},
                        **{f"seat_{r.cycle}": r.seat_err for r in res.itertuples()}})
            c, se, _ = rc.ols(rows, terms)
            comp = [t for t in terms if t.startswith("c_")]
            if comp:
                print(f"  {name}: " + ", ".join(f"{t} {c[t] * 100:+.2f}pp (se {se[t] * 100:.2f})" for t in comp))
        print(pd.DataFrame(out).round(4).to_string(index=False))

        # Held-out residual by Hispanic share, with and without the term.
        names = list(sp)
        for name in (names[0], names[1]):
            _, resid = rc.loo(rows, sp[name])
            h = rows.loc[resid.index, "hispanic"]
            b = rows.loc[resid.index, "black"]
            print(f"  {name:<18} held-out mean resid (pred-actual, pp): "
                  f"Hisp>=60% {resid[h >= .6].mean() * 100:+.2f} (n={int((h >= .6).sum())})   "
                  f"Black>=35% {resid[b >= .35].mean() * 100:+.2f} (n={int((b >= .35).sum())})   "
                  f"rest {resid[(h < .6) & (b < .35)].mean() * 100:+.2f}")


if __name__ == "__main__":
    main()
