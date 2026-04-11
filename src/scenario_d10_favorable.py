"""
scenario_d10_favorable.py

One-off "most favorable environment possible" thought experiment, run April 2026.

Two scenarios on top of the standard model:
  1. D+10 wave (e.g., national collapse from Iran war fallout + price increases)
  2. D+10 wave AND every D challenger in a competitive R-held seat hits the
     fundraising viability threshold

"Competitive R-held" is defined as: an R-held district (open or incumbent)
whose 2024 presidential D 2-party share was at least 35%. Below that the
seat is too safe-R for fundraising viability to plausibly matter.

Note: the model uses dual-track logic — incumbents with career WAR data
skip the finance term entirely (their fundraising is already embedded in
their WAR estimate). The viability flip only moves predictions for open
seats and non-WAR incumbents.

Output:
  output/scenario_d10_favorable.csv  — per-district win probs for both runs
  Console: scenario seat totals and the districts that flipped category
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    load_districts,
    run_monte_carlo,
    _SENATE_D_HOLDOVER,
    HOUSE_MAJORITY,
    SENATE_MAJORITY,
    N_SIMULATIONS,
)
from model_config import RACE_GENERIC_BALLOT_D_SHARE


def scenario_summary(label: str, result: dict) -> str:
    house = result["expected_house_seats"]
    house_p = result["house_control_prob"] * 100
    senate = result["expected_senate_seats"]
    senate_total = senate + _SENATE_D_HOLDOVER
    senate_p = result["senate_control_prob"] * 100
    return (
        f"  {label:38s}  House: {house:5.1f} ({house_p:4.1f}%)   "
        f"Senate: {senate_total:4.1f}/31 ({senate_p:4.1f}%)"
    )


def main():
    print("=" * 78)
    print("  TX Legislature 2026 — D+10 Favorable Scenario (thought experiment)")
    print("=" * 78)
    print("  Premise: national collapse (Iran war fallout, continued inflation)")
    print("           pushes generic ballot to D+10, AND every D challenger in")
    print("           a competitive R-held seat hits fundraising viability.\n")

    df = load_districts()
    race_generic = RACE_GENERIC_BALLOT_D_SHARE

    # ---- Baseline runs for context ----
    res_d5 = run_monte_carlo(df, env_dial=5, race_generic=race_generic, n_sims=N_SIMULATIONS)
    res_d8 = run_monte_carlo(df, env_dial=8, race_generic=race_generic, n_sims=N_SIMULATIONS)
    res_d10 = run_monte_carlo(df, env_dial=10, race_generic=race_generic, n_sims=N_SIMULATIONS)

    # ---- Identify "competitive R-held" seats ----
    is_r_held = df["incumbent_party"].astype(str).str.upper() == "R"
    pres_baseline = pd.to_numeric(df["dem_pres_2p_baseline"], errors="coerce").fillna(0)
    competitive_mask = is_r_held & (pres_baseline >= 0.35)

    n_targets = int(competitive_mask.sum())
    n_open = int(
        (competitive_mask
         & df["open_seat"].astype(str).str.lower().isin(["true", "1", "yes"]))
        .sum()
    )
    n_already_viable = int(
        (competitive_mask & (df["challenger_viability_flag_early"] == 1)).sum()
    )
    print(f"  Competitive R-held seats (dem_pres_2p ≥ 0.35): {n_targets}")
    print(f"    of which open seats:                          {n_open}")
    print(f"    of which D challenger already viable:         {n_already_viable}")
    print(f"    new viability flips this scenario applies:    {n_targets - n_already_viable}\n")

    # ---- Modified df: flip viability for all competitive R-held seats ----
    df_fav = df.copy()
    df_fav.loc[competitive_mask, "challenger_viability_flag_early"] = 1

    res_d10_fav = run_monte_carlo(
        df_fav, env_dial=10, race_generic=race_generic, n_sims=N_SIMULATIONS
    )

    # ---- Print summary ----
    print("-" * 78)
    print("  Scenario seat totals (Senate shown as total seats out of 31)")
    print("-" * 78)
    print(scenario_summary("D+5 (current baseline)", res_d5))
    print(scenario_summary("D+8 (2018 wave reference)", res_d8))
    print(scenario_summary("D+10 (collapse, current finance)", res_d10))
    print(scenario_summary("D+10 + universal D viability", res_d10_fav))

    # Marginal effects
    delta_d8_to_d10 = res_d10["expected_house_seats"] - res_d8["expected_house_seats"]
    delta_d10_to_fav = res_d10_fav["expected_house_seats"] - res_d10["expected_house_seats"]
    delta_d5_to_fav = res_d10_fav["expected_house_seats"] - res_d5["expected_house_seats"]
    print()
    print(f"  Marginal House seat impact (expected):")
    print(f"    D+8  → D+10  (env wave alone):                    +{delta_d8_to_d10:.1f} seats")
    print(f"    D+10 → D+10 + viability (finance flip alone):     +{delta_d10_to_fav:.1f} seats")
    print(f"    D+5  → D+10 + viability (everything combined):    +{delta_d5_to_fav:.1f} seats")
    print()
    print(f"  House control probability under most favorable scenario: "
          f"{res_d10_fav['house_control_prob']*100:.1f}%")
    print(f"  Senate control probability under most favorable scenario: "
          f"{res_d10_fav['senate_control_prob']*100:.1f}%")

    # ---- Per-district movement ----
    print()
    print("-" * 78)
    print("  Districts that moved into D-favored (>50%) under D+10 + viability")
    print("-" * 78)
    df["wp_d10"] = res_d10["district_win_probs"].values
    df["wp_d10_fav"] = res_d10_fav["district_win_probs"].values
    df["wp_d5"] = res_d5["district_win_probs"].values

    flips = df[(df["wp_d10_fav"] > 0.5) & (df["wp_d5"] <= 0.5)].copy()
    flips = flips.sort_values("wp_d10_fav", ascending=False)
    if flips.empty:
        print("  (none — even at D+10 with universal viability, no new seats cross 50%)")
    else:
        print(f"  {'Chamber':8s} {'Dist':>5s}  {'Incumbent':28s} {'Open':>5s}  "
              f"{'D+5':>6s}  {'D+10':>6s}  {'D+10F':>6s}")
        for _, r in flips.iterrows():
            ch = r["chamber"][:8]
            dist = int(r["district"])
            inc = str(r["incumbent"])[:28]
            is_open = str(r.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
            print(f"  {ch:8s} {dist:>5d}  {inc:28s} {'open' if is_open else 'inc':>5s}  "
                  f"{r['wp_d5']*100:5.1f}%  {r['wp_d10']*100:5.1f}%  {r['wp_d10_fav']*100:5.1f}%")

    # ---- Districts where the viability flip moved the prob ≥3pp ----
    print()
    print("-" * 78)
    print("  Districts where the viability flip alone moved P(D win) by ≥3pp")
    print("-" * 78)
    df["delta_fav"] = df["wp_d10_fav"] - df["wp_d10"]
    big_movers = df[(df["delta_fav"] >= 0.03) & competitive_mask].copy()
    big_movers = big_movers.sort_values("delta_fav", ascending=False)
    if big_movers.empty:
        print("  (none — every competitive R-held seat is either already viable")
        print("   or on the WAR track where the finance flag is bypassed)")
    else:
        print(f"  {'Chamber':8s} {'Dist':>5s}  {'Incumbent':28s} {'Open':>5s}  "
              f"{'D+10':>6s}  {'D+10F':>6s}  {'Δpp':>6s}")
        for _, r in big_movers.iterrows():
            ch = r["chamber"][:8]
            dist = int(r["district"])
            inc = str(r["incumbent"])[:28]
            is_open = str(r.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
            print(f"  {ch:8s} {dist:>5d}  {inc:28s} {'open' if is_open else 'inc':>5s}  "
                  f"{r['wp_d10']*100:5.1f}%  {r['wp_d10_fav']*100:5.1f}%  "
                  f"{r['delta_fav']*100:+5.1f}")

    # ---- Save full output ----
    out_path = OUTPUT / "scenario_d10_favorable.csv"
    out_df = df[[
        "chamber", "district", "incumbent", "incumbent_party", "open_seat",
        "dem_pres_2p_baseline", "wp_d5", "wp_d10", "wp_d10_fav", "delta_fav",
    ]].copy()
    out_df.columns = [
        "chamber", "district", "incumbent", "incumbent_party", "open_seat",
        "dem_pres_2p_baseline",
        "win_prob_d_d5", "win_prob_d_d10", "win_prob_d_d10_favorable", "viability_flip_delta",
    ]
    out_df.to_csv(out_path, index=False)
    print(f"\n  Per-district output: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
