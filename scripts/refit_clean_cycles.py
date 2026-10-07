"""
refit_clean_cycles.py

Fit the model's regression on the clean-vintage midterm cycles (2014, 2018,
2022) and test each block of terms out of sample. This is the script behind
the 2026-10-07 adoption of the clean-cycle refit on the open-race baseline;
branch `refit-clean-cycles` (2026-09-09) produced its coefficients by hand.

Specifications (every one fit on the same rows):
  base       baseline + env + senate
  +inc       base + dem_incumbent + rep_incumbent
  +fin       base + viability flag + dem fundraising share
  +inc+fin   both            ("full" -- what the model ships)
  +inc+fin+ie                adds centred IE share, to re-estimate IE on these terms
  master_imposed  +inc+fin with incumbency and finance FIXED at master's values
                  (dem_inc .0676, rep_inc -.0804, viab .0446, share .0731),
                  rest refit -- tests whether master's large values hold up

Out of sample: leave-one-cycle-out. Each held-out cycle is scored on residual
sd, mean residual (the cycle's level miss), Brier and House seat error, using
a common sigma so the specs differ only in their predictions.

Sigma decomposition (for the forecast sigma) from the LOO residuals:
  national = RMS of held-out cycle mean residuals
  idio     = pooled within-cycle sd
  total    = sqrt(national^2 + idio^2)

Usage:
  python scripts/refit_clean_cycles.py                     # open baseline (adopted)
  python scripts/refit_clean_cycles.py --baseline pres     # reproduce the branch
  python scripts/refit_clean_cycles.py --phase1 path.csv   # alternative dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_backtest as bb  # noqa: E402  (downballot + CVAP loaders)

CYCLES = [2014, 2018, 2022]
SIGMA_SCORE = 0.0440
MASTER = {"dem_incumbent": 0.0676, "rep_incumbent": -0.0804,
          "challenger_viability_flag": 0.0446, "dem_fundraising_share": 0.0731}
BASE = ["x", "national_env", "chamber_senate"]
INC = ["dem_incumbent", "rep_incumbent"]
FIN = ["challenger_viability_flag", "dem_fundraising_share"]
SPECS = {
    "base": BASE,
    "+inc": BASE + INC,
    "+fin": BASE + FIN,
    "+inc+viab": BASE + INC + ["challenger_viability_flag"],
    "+inc+share": BASE + INC + ["dem_fundraising_share"],
    "+inc+ie": BASE + INC + ["ie_c"],
    "+inc+fin": BASE + INC + FIN,
    "+inc+fin+ie": BASE + INC + FIN + ["ie_c"],
    "SHIP": BASE + INC + ["challenger_viability_flag", "ie_c"],  # adopted 2026-10-07
}


def load(phase1: Path, baseline: str) -> pd.DataFrame:
    if phase1 != bb.PROC / "phase1_dataset.csv":
        bb.PROC = phase1.parent  # baseline_backtest reads PROC/phase1_dataset.csv
        if phase1.name != "phase1_dataset.csv":
            raise SystemExit("--phase1 must point at a file named phase1_dataset.csv")
    df = bb.load()
    df["x"] = df["dem_pres_2p_baseline"] if baseline == "pres" else df["open_mean_d2p"]
    df = df[(df.contested == True) & (df.on_ballot == True) & df.dem_2p_share.notna()
            & df.x.notna()].copy()
    if "pres_baseline_correct_vintage" in df:
        df = df[df.pres_baseline_correct_vintage == 1]
    for c in INC + FIN + ["national_env", "chamber_senate", "ie_dem_share"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ie_c"] = df["ie_dem_share"].fillna(0.5) - 0.5
    # The 2026 model applies the flag SIGNED (viable_opposition_signed: +1 a
    # viable D opposing an R, -1 a viable R opposing a D). Training stored it
    # unsigned, so a viable R challenger to a D incumbent counted as pro-D.
    # Sign it the same way here; open seats (challenger = the lower raiser,
    # i.e. both sides viable) carry no direction.
    df["challenger_viability_flag"] = df["challenger_viability_flag"] * np.select(
        [df["rep_incumbent"] == 1, df["dem_incumbent"] == 1], [1.0, -1.0], 0.0)
    return df


def ols(d, terms, offset=None):
    X = np.column_stack([np.ones(len(d))] + [d[t].values for t in terms])
    y = d["dem_2p_share"].values - (0 if offset is None else offset)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return dict(zip(["intercept"] + terms, beta)), dict(zip(["intercept"] + terms, se)), np.sqrt(s2)


def predict(d, coefs, offset=None):
    p = np.full(len(d), coefs["intercept"])
    for t, b in coefs.items():
        if t != "intercept":
            p = p + b * d[t].values
    return p + (0 if offset is None else offset)


def master_offset(d):
    return sum(MASTER[t] * d[t].values for t in MASTER)


def score(test, pred):
    actual = test.dem_2p_share.values
    r = pred - actual
    win = norm.cdf(pred, 0.5, SIGMA_SCORE)
    aw = (actual > 0.5).astype(float)
    house = (test.chamber_lower == "house").values
    return {"n": len(test), "resid_sd": r.std(), "mean_resid": r.mean(),
            "rmse": np.sqrt((r ** 2).mean()), "brier": np.mean((win - aw) ** 2),
            "seat_err": win[house].sum() - aw[house].sum()}


def loo(d, terms, imposed=False):
    rows, resids = [], []
    for yr in CYCLES:
        tr, te = d[d.year != yr], d[d.year == yr]
        if imposed:
            c, _, _ = ols(tr, BASE, master_offset(tr))
            p = predict(te, c, master_offset(te))
        else:
            c, _, _ = ols(tr, terms)
            p = predict(te, c)
        rows.append({"cycle": yr} | score(te, p))
        resids.append(pd.Series(p - te.dem_2p_share.values, index=te.index))
    return pd.DataFrame(rows), pd.concat(resids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["open", "pres"], default="open")
    ap.add_argument("--phase1", type=Path, default=bb.PROC / "phase1_dataset.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 220)

    allrows = load(args.phase1, args.baseline)
    full = allrows.dropna(subset=FIN)
    print(f"Baseline: {args.baseline}.  Rows: {len(allrows)} contested clean-vintage "
          f"(no-finance sample), {len(full)} with finance.  By cycle: "
          f"{full.groupby('year').size().to_dict()}")

    print("\n=== In-sample coefficients (finance sample) ===")
    out = {}
    for name, terms in SPECS.items():
        c, se, s = ols(full, terms)
        out[name] = c
        print(f"\n{name}  (in-sample resid sd {s:.4f})")
        for t in c:
            print(f"  {t:<28} {c[t]:+.4f}  se {se[t]:.4f}  t {c[t] / se[t]:+6.2f}")
    c, se, s = ols(full, BASE, master_offset(full))
    print(f"\nmaster_imposed  (inc+fin fixed at master; in-sample resid sd {s:.4f})")
    for t in c:
        print(f"  {t:<28} {c[t]:+.4f}  se {se[t]:.4f}")

    print("\n=== No-finance fit for WAR (no-finance sample, base + incumbency) ===")
    c, se, s = ols(allrows, BASE + INC)
    for t in c:
        print(f"  {t:<28} {c[t]:+.4f}  se {se[t]:.4f}")
    print(f"  sigma (in-sample)            {s:.4f}   n={len(allrows)}")

    print("\n=== Leave-one-cycle-out (finance sample; sigma for scoring fixed at "
          f"{SIGMA_SCORE}) ===")
    summary = []
    for name, terms in list(SPECS.items()) + [("master_imposed", None)]:
        res, resid = loo(full, terms, imposed=(name == "master_imposed"))
        nat = np.sqrt((res.mean_resid ** 2).mean())
        idio = np.sqrt(((resid - resid.groupby(full.loc[resid.index, "year"]).transform("mean")) ** 2).mean())
        summary.append({"spec": name, "rmse": np.sqrt((resid ** 2).mean()),
                        "brier": res.brier.mean(), "abs_seat_err": res.seat_err.abs().mean(),
                        "sigma_nat": nat, "sigma_idio": idio,
                        "sigma_total": np.sqrt(nat ** 2 + idio ** 2),
                        **{f"lvl_{r.cycle}": r.mean_resid * 100 for r in res.itertuples()},
                        **{f"seat_{r.cycle}": r.seat_err for r in res.itertuples()}})
    print(pd.DataFrame(summary).round(4).to_string(index=False))

    # Incumbency on the larger no-finance sample (2014 has 55 races here vs 8
    # with finance), so all three cycles carry weight in the held-out test.
    print("\n=== Incumbency test on the no-finance sample (all three cycles real) ===")
    summary = []
    for name in ("base", "+inc"):
        res, resid = loo(allrows, SPECS[name])
        nat = np.sqrt((res.mean_resid ** 2).mean())
        idio = np.sqrt(((resid - resid.groupby(allrows.loc[resid.index, "year"]).transform("mean")) ** 2).mean())
        summary.append({"spec": name, "n": len(allrows), "rmse": np.sqrt((resid ** 2).mean()),
                        "brier": res.brier.mean(), "abs_seat_err": res.seat_err.abs().mean(),
                        "sigma_nat": nat, "sigma_idio": idio, "sigma_total": np.sqrt(nat ** 2 + idio ** 2),
                        **{f"lvl_{r.cycle}": r.mean_resid * 100 for r in res.itertuples()},
                        **{f"seat_{r.cycle}": r.seat_err for r in res.itertuples()}})
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    c, se, _ = ols(allrows, BASE)
    print("\nbase fit on the no-finance sample, for reference:")
    for t in c:
        print(f"  {t:<28} {c[t]:+.4f}  se {se[t]:.4f}")

    # Incumbency where it should matter most: does an incumbent run ahead of
    # the baseline in the competitive band, by party?
    d = allrows.copy()
    cb, _, _ = ols(d, BASE)
    d["resid"] = d.dem_2p_share - predict(d, cb)
    d["band"] = np.where(np.abs(predict(d, cb) - 0.5) < 0.10, "within 10pp", "safer")
    print("\nMean residual vs base fit (actual - predicted, D share, pp), by incumbency:")
    for band in ("within 10pp", "safer"):
        for lab, m in (("D incumbent", d.dem_incumbent == 1), ("R incumbent", d.rep_incumbent == 1),
                       ("open seat", (d.dem_incumbent == 0) & (d.rep_incumbent == 0))):
            g = d[(d.band == band) & m]
            if len(g):
                print(f"  {band:<12} {lab:<12} n={len(g):>3}  {g.resid.mean() * 100:+.2f}pp "
                      f"(se {g.resid.std() / np.sqrt(len(g)) * 100:.2f})")


if __name__ == "__main__":
    main()
