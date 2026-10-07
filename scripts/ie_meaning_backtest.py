"""
ie_meaning_backtest.py

What does independent-expenditure money actually signal? The model reads it
as a directional push: the more of a race's IE money favours the Democrat,
the better the Democrat does (ie_dem_share, coef ~ +0.07). Two alternative
readings are tested here on the Phase-1 training rows (contested midterm
races, 2002-2022):

  1. WEAKNESS. Money spent DEFENDING the favourite means the favourite is in
     trouble, so the favourite underperforms the baseline despite it.
  2. CONTEXT. The effect differs in tight races vs races the baseline calls
     safe-ish (where spending may be targeting reflecting private polling).

Method: fit the full model WITHOUT the IE terms, take its prediction p_hat,
orient everything toward the predicted favourite, and look at how the
favourite's residual (actual minus predicted, favourite's side) varies with
whether IE money defended the favourite or attacked it.

Usage:
  python scripts/ie_meaning_backtest.py
  python scripts/ie_meaning_backtest.py --min-ie 25000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent.parent

BASE_TERMS = ["dem_pres_2p_baseline", "dem_incumbent", "rep_incumbent", "chamber_senate",
              "national_env", "dem_fundraising_share", "log_challenger_fundraising",
              "challenger_viability_flag"]


def ols(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return beta, se, X1 @ beta


def summarize(g, label):
    if len(g) == 0:
        print(f"  {label:<44} n=  0")
        return
    m, s = g.fav_resid.mean(), g.fav_resid.std(ddof=1) if len(g) > 1 else float("nan")
    se = s / np.sqrt(len(g)) if len(g) > 1 else float("nan")
    upset = (g.fav_actual < 0.5).mean()
    exp_upset = g.fav_upset_prob.mean()
    print(f"  {label:<44} n={len(g):>3}  fav resid {m * 100:+5.2f}pp (se {se * 100:4.2f})"
          f"  upsets {upset:5.1%} vs {exp_upset:5.1%} expected")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-ie", type=float, default=50_000)
    args = ap.parse_args()

    d = pd.read_csv(ROOT / "data" / "processed" / "phase1_dataset.csv")
    d = d[(d.contested == 1) | (d.contested == True)]
    d = d[d.year.isin([2002, 2006, 2010, 2014, 2018, 2022])]
    d = d.dropna(subset=BASE_TERMS + ["dem_2p_share"]).copy()
    d["ie_total"] = d.ie_total.fillna(0)

    # Base model without IE; residual SD for upset expectations.
    beta, se, p_hat = ols(d.dem_2p_share.values, d[BASE_TERMS].values)
    d["p_hat"] = p_hat
    sigma = np.std(d.dem_2p_share - d.p_hat, ddof=len(BASE_TERMS) + 1)
    from math import erf, sqrt
    d["fav_is_d"] = d.p_hat >= 0.5
    d["fav_pred"] = np.where(d.fav_is_d, d.p_hat, 1 - d.p_hat)
    d["fav_actual"] = np.where(d.fav_is_d, d.dem_2p_share, 1 - d.dem_2p_share)
    d["fav_resid"] = d.fav_actual - d.fav_pred
    d["fav_upset_prob"] = [0.5 * (1 + erf(-(fp - 0.5) / (sigma * sqrt(2)))) for fp in d.fav_pred]
    d["margin_band"] = pd.cut(d.fav_pred - 0.5, [-0.01, 0.05, 0.10, 0.20, 1],
                              labels=["tossup <5pp", "lean 5-10pp", "likely 10-20pp", "safe 20pp+"])

    ie = d[d.ie_total >= args.min_ie].copy()
    ie["fav_ie_share"] = np.where(ie.fav_is_d, ie.ie_d_favor.fillna(0), ie.ie_r_favor.fillna(0)) / ie.ie_total
    ie["posture"] = np.select([ie.fav_ie_share >= 0.75, ie.fav_ie_share <= 0.25],
                              ["DEFENSE (>=75% backs favourite)", "OFFENSE (>=75% backs underdog)"],
                              "MIXED")

    print(f"Contested midterm rows: {len(d)}; with IE >= ${args.min_ie:,.0f}: {len(ie)}; "
          f"base-model residual sd {sigma * 100:.2f}pp")
    print("\nFavourite's residual vs. base model (no IE terms). Positive = favourite beat its prediction.")
    print("\nAll races, by whether they drew IE money:")
    summarize(d[d.ie_total < 1], "no IE at all")
    summarize(d[(d.ie_total >= 1) & (d.ie_total < args.min_ie)], f"IE under ${args.min_ie / 1000:.0f}K")
    summarize(ie, f"IE >= ${args.min_ie / 1000:.0f}K")

    print("\nIE races by posture (the model predicts DEFENSE > 0 > OFFENSE):")
    for p in ["DEFENSE (>=75% backs favourite)", "MIXED", "OFFENSE (>=75% backs underdog)"]:
        summarize(ie[ie.posture == p], p)

    print("\nIE races by posture x how close the base model called it:")
    for band in ie.margin_band.cat.categories:
        for p in ["DEFENSE (>=75% backs favourite)", "OFFENSE (>=75% backs underdog)", "MIXED"]:
            summarize(ie[(ie.margin_band == band) & (ie.posture == p)], f"{band:<15} {p[:7]}")

    print("\nNon-IE races by band (the comparison group):")
    for band in d.margin_band.cat.categories:
        summarize(d[(d.margin_band == band) & (d.ie_total < 1)], f"{band:<15} no IE")

    # Does the slope on ie_dem_share differ by closeness? Regression with interaction.
    d2 = d.copy()
    d2["ie_dem_share"] = d2.ie_dem_share.fillna(0.5)
    d2["ie_log_total"] = d2.ie_log_total.fillna(0)
    d2["close"] = ((d2.p_hat - 0.5).abs() < 0.10).astype(float)
    d2["ie_c"] = d2.ie_dem_share - 0.5
    d2["ie_x_close"] = d2.ie_c * d2.close
    terms = BASE_TERMS + ["ie_c", "ie_log_total", "close", "ie_x_close"]
    b, s, _ = ols(d2.dem_2p_share.values, d2[terms].values)
    print("\nRegression with interaction (ie_c = ie_dem_share - 0.5; close = base model within 10pp):")
    for name, bi, si in zip(["Intercept"] + terms, b, s):
        if name.startswith("ie") or name == "close":
            print(f"  {name:<16} {bi:+.4f}  se {si:.4f}  t {bi / si:+.2f}")

    print("\nBy cycle, IE >= threshold, favourite residual by posture:")
    for yr in sorted(ie.year.unique()):
        g = ie[ie.year == yr]
        parts = []
        for p in ["DEFENSE", "OFFENSE", "MIXED"]:
            h = g[g.posture.str.startswith(p)]
            if len(h):
                parts.append(f"{p[:3]} n={len(h)} {h.fav_resid.mean() * 100:+.1f}pp")
        print(f"  {yr}: " + "   ".join(parts))


if __name__ == "__main__":
    main()
