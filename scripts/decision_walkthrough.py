"""Walk one district through the pending decision, term by term.

Choice 1 — coefficients: master (pass-through 0.596, fit on mismatched
           geography) vs refit-clean-cycles (0.976, clean cycles, sigma 0.044).
Choice 2 — baseline: 2024 presidential vs 2024 open-race downballot mean
           (three CCA seats) vs the half-and-half blend.
Also shown: the demographic level term and Hispanic constant, which the
poll-integration backtests already said to remove, so each cell is shown with
and without them.

Usage: python scripts/decision_walkthrough.py --district 41 [--chamber house]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model  # noqa: E402
import model_config as mc  # noqa: E402

MASTER = dict(model.COEFS)
REFIT = dict(MASTER)
REFIT.update({"intercept": -0.0322, "dem_pres_2p_baseline": 0.9759, "dem_incumbent": 0.0394,
              "rep_incumbent": 0.0078, "chamber_senate": 0.0069, "national_env": 0.0036,
              "challenger_viability_flag": 0.0015, "dem_fundraising_share": 0.0057,
              "sigma": 0.0440})
COEF_SETS = {"master": MASTER, "refit": REFIT}


def load(baseline):
    os.environ["TXLEG_BASELINE"] = baseline
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return model.load_districts()


def full_prediction(df, coefs, env):
    import contextlib, io
    saved = dict(model.COEFS)
    model.COEFS.clear(); model.COEFS.update(coefs)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            pred = model.build_linear_predictions(df, env, mc.RACE_GENERIC_BALLOT_D_SHARE)
    finally:
        model.COEFS.clear(); model.COEFS.update(saved)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--district", type=int, default=41)
    ap.add_argument("--chamber", default="house")
    ap.add_argument("--env", type=float, default=model.CURRENT_ENV)
    args = ap.parse_args()
    env = args.env
    rg = mc.RACE_GENERIC_BALLOT_D_SHARE
    nat_avg = sum(mc.NATIONAL_DEMO_WEIGHTS[r] * rg[r] for r in mc.NATIONAL_DEMO_WEIGHTS)

    frames = {b: load(b) for b in ("pres", "open", "blend_open")}
    row0 = frames["pres"][(frames["pres"]["chamber_lower"] == args.chamber)
                          & (frames["pres"]["district"] == args.district)].iloc[0]
    hisp = float(row0["pct_hispanic"]) / 100 if float(row0["pct_hispanic"]) > 1 else float(row0["pct_hispanic"])
    shares = {r: float(row0[c]) / 100 for r, c in (("white_nh", "pct_white_nh"), ("black_nh", "pct_black_nh"),
                                                    ("hispanic", "pct_hispanic"), ("other", "pct_other"))}
    demo_dev = sum(shares[r] * rg[r] for r in shares) - nat_avg
    hisp_const = mc.TX_HISPANIC_ADJUSTMENT * hisp
    dem_inc = 1.0 if (row0["incumbent_party"] == "D" and str(row0.get("open_seat", "")).lower() not in ("true", "1")) else 0.0
    rep_inc = 1.0 if (row0["incumbent_party"] == "R" and str(row0.get("open_seat", "")).lower() not in ("true", "1")) else 0.0
    sen = 1.0 if args.chamber == "senate" else 0.0

    print(f"=== {args.chamber.upper()} {args.district}: {row0['incumbent']} ({row0['incumbent_party']}), "
          f"open seat: {row0.get('open_seat')}, Hispanic CVAP {hisp:.1%}, env D{env:+.1f} ===")
    print("Inputs that never change across the choices:")
    print(f"  CVAP shares: white {shares['white_nh']:.1%}  Black {shares['black_nh']:.1%}  "
          f"Hispanic {shares['hispanic']:.1%}  other {shares['other']:.1%}")
    print(f"  demo level term  = sum(share x national D by race) - national avg = {demo_dev * 100:+.2f}pp")
    print(f"  Hispanic constant = {mc.TX_HISPANIC_ADJUSTMENT} x {hisp:.3f} = {hisp_const * 100:+.2f}pp")
    print(f"  incumbency flags: dem_inc {dem_inc:.0f}, rep_inc {rep_inc:.0f}")
    bases = {}
    for b, df in frames.items():
        r = df[(df["chamber_lower"] == args.chamber) & (df["district"] == args.district)].iloc[0]
        bases[b] = float(r["dem_pres_2p_baseline"])
    print(f"  baselines: presidential {bases['pres']:.4f}   open (3 CCA) {bases['open']:.4f}   blend {bases['blend_open']:.4f}")

    print("\nTerm by term (all in D two-party share). 'WAR/finance/IE' is everything the full model adds beyond the structural terms.")
    hdr = f"{'coefs':7s} {'baseline':10s} {'intercept':>9s} {'pass x base':>12s} {'env':>7s} {'incumb':>7s} {'demo+const':>10s} {'WAR/fin/IE':>10s} {'pred':>7s} {'P(D)':>6s} | {'no demo term':>12s} {'P(D)':>6s}"
    print(hdr)
    for cname, coefs in COEF_SETS.items():
        for b, df in frames.items():
            r = df[(df["chamber_lower"] == args.chamber) & (df["district"] == args.district)]
            base = bases[b]
            intercept = coefs["intercept"]
            pb = coefs["dem_pres_2p_baseline"] * base
            envt = coefs["national_env"] * env
            inc = coefs["dem_incumbent"] * dem_inc + coefs["rep_incumbent"] * rep_inc + coefs["chamber_senate"] * sen
            demo = demo_dev + hisp_const
            full = float(full_prediction(df, coefs, env).loc[r.index[0]])
            structural = intercept + pb + envt + inc + demo
            rem = full - structural
            sig = coefs["sigma"]
            p = norm.cdf(full, 0.5, sig)
            nod = full - demo
            pnod = norm.cdf(nod, 0.5, sig)
            print(f"{cname:7s} {b:10s} {intercept:+9.4f} {pb:+12.4f} {envt:+7.4f} {inc:+7.4f} {demo:+10.4f} {rem:+10.4f} {full:7.4f} {p:6.1%} | {nod:12.4f} {pnod:6.1%}")
    print(f"\nsigma for P(D): master {MASTER['sigma']}, refit {REFIT['sigma']}. Today's published number is the "
          f"'master / pres' row with the demo term included.")


if __name__ == "__main__":
    main()
