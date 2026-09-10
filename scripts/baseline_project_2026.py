"""What changes in 2026 if the baseline is Railroad Commissioner or a blend?

Structural model only (baseline + incumbency + chamber + environment), each
baseline with its own pooled clean-cycle coefficients from
scripts/baseline_backtest.py, so the comparison is like for like. No WAR,
finance or IE terms, no Monte Carlo: this shows where the *baseline* moves
seats, not the full model's answer.

Usage: python scripts/baseline_project_2026.py [--env 9.1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import baseline_backtest as bb  # noqa: E402

HIST = ROOT / "data" / "raw" / "historical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=float, default=9.1, help="D-R generic ballot margin, pp")
    args = ap.parse_args()

    hist = bb.load()
    sample = hist[(hist["contested"] == True) & (hist["on_ballot"] == True)
                  & hist["dem_2p_share"].notna()].copy()

    d = pd.read_csv(ROOT / "data" / "processed" / "districts_2026.csv")
    d = d[d["up_in_2026"] == True].copy()
    d["chamber_lower"] = d["chamber"].str.lower()
    is_open = d["open_seat"].astype(str).str.lower().isin(["true", "1", "yes"])
    d["dem_incumbent"] = ((d["incumbent_party"] == "D") & ~is_open).astype(float)
    d["rep_incumbent"] = ((d["incumbent_party"] == "R") & ~is_open).astype(float)
    d["chamber_senate"] = (d["chamber_lower"] == "senate").astype(float)
    d["national_env"] = args.env
    parts = []
    for chamber, fname in (("house", "tx_downballot_house_2024_planh2316.csv"),
                           ("senate", "tx_downballot_senate_2024_plans2168.csv")):
        x = pd.read_csv(HIST / fname)[["district", "pres_d2p", "rrc_d2p", "judicial_mean_d2p", "open_mean_d2p"]]
        x["chamber_lower"] = chamber
        parts.append(x)
    d = d.merge(pd.concat(parts), on=["chamber_lower", "district"], how="left")
    d["blend_pres_rrc"] = 0.5 * (d["pres_d2p"] + d["rrc_d2p"])
    d["blend_pres_open"] = 0.5 * (d["pres_d2p"] + d["open_mean_d2p"])

    cols = {"pres": "pres_d2p", "rrc": "rrc_d2p", "blend": "blend_pres_rrc",
            "open": "open_mean_d2p", "blend_open": "blend_pres_open"}
    for label, xcol in cols.items():
        beta, sig = bb.fit(sample[sample[xcol].notna()], xcol)
        d[f"pred_{label}"] = bb.predict(d, xcol, beta)
        print(f"{label:6s} pooled coefs: intercept {beta[0]:+.4f} pass {beta[1]:.4f} "
              f"dem_inc {beta[2]:+.4f} rep_inc {beta[3]:+.4f} senate {beta[4]:+.4f} env {beta[5]:.4f} sigma {sig:.4f}")

    house = d[d["chamber_lower"] == "house"]
    print(f"\nEnvironment D{args.env:+.1f}. House seats predicted D (>50%), structural model only:")
    for label in cols:
        print(f"  {label:6s} {int((house[f'pred_{label}'] > 0.5).sum())} / 150")

    preds = [f"pred_{k}" for k in cols]
    d["shift_open_pp"] = (d["pred_open"] - d["pred_pres"]) * 100
    mv = d[(d[preds].max(axis=1) > 0.42) & (d[preds].min(axis=1) < 0.58)].copy()
    mv = mv.sort_values("shift_open_pp")
    pd.set_option("display.width", 220)
    show = ["chamber", "district", "incumbent", "incumbent_party", "pct_hispanic",
            "pres_d2p", "rrc_d2p", "open_mean_d2p"] + preds + ["shift_open_pp"]
    print("\nSeats within 42-58% under any baseline, sorted by how far the open-race baseline moves them (pp):")
    print(mv[show].round(3).to_string(index=False))
    for k in ("rrc", "open", "blend_open"):
        crossers = mv[((mv["pred_pres"] > 0.5) != (mv[f"pred_{k}"] > 0.5))]
        print(f"Seats that cross 50% between presidential and {k}: "
              f"{[(r.chamber[0] + str(r.district)) for r in crossers.itertuples()]}")


if __name__ == "__main__":
    main()
