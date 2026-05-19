"""
scenario_ie_equalized.py

Scenario analysis: what happens if IE spending equalizes in competitive races
AND all D challengers in those races meet minimum financial viability?

"Competitive" is defined as: predicted D two-party share within 10pp of 50%
(i.e., between 40% and 60%) under the BASELINE model WITHOUT IE adjustments.
This isolates the races where IE spending could plausibly tip the outcome.

Two layered scenarios for each environment:
  1. IE EQUALIZED — in competitive races, IE spending is neutralized
     (ie_d_favor = ie_r_favor = 0, so ie_dem_share = 0.5 → zero IE adjustment)
  2. IE EQUALIZED + UNIVERSAL VIABILITY — additionally, every D challenger
     in those competitive races hits the fundraising viability threshold
     (challenger_viability_flag_early = 1)

Note: the viability flip only affects districts on the finance track.
Incumbents with WAR data skip the finance term entirely (WAR already
captures fundraising ability), so the viability flip is a no-op for them.

Output:
  output/scenario_ie_equalized.csv       — per-district win probs across scenarios
  output/scenario_ie_equalized_summary.csv — seat totals and control probabilities
  Console: scenario comparison tables
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    load_districts,
    build_linear_predictions,
    run_monte_carlo,
    _SENATE_D_HOLDOVER,
    HOUSE_MAJORITY,
    SENATE_MAJORITY,
    N_SIMULATIONS,
    IE_COEFFICIENT,
    IE_MIN_THRESHOLD,
    IE_WEIGHT,
    _ie_weight_override,
)
from model import CURRENT_ENV
from model_config import RACE_GENERIC_BALLOT_D_SHARE, ENV_SCENARIOS


def _effective_ie_weight() -> float:
    return _ie_weight_override if _ie_weight_override is not None else IE_WEIGHT


def identify_competitive_no_ie(df: pd.DataFrame, env_dial: float,
                                race_generic: dict, margin: float = 0.10
                                ) -> pd.Series:
    """
    Identify competitive districts under a model run WITHOUT IE effects.

    Returns a boolean mask: True for districts whose linear prediction
    (excluding IE adjustment) is within `margin` of 0.50.

    We compute this by getting the full linear prediction and then subtracting
    back out the IE adjustment that build_linear_predictions applied.
    """
    predicted_with_ie = build_linear_predictions(df, env_dial, race_generic)

    # Reconstruct the IE adjustment that was applied
    ie_d = pd.to_numeric(df.get("ie_d_favor", pd.Series(0, index=df.index)),
                         errors="coerce").fillna(0)
    ie_r = pd.to_numeric(df.get("ie_r_favor", pd.Series(0, index=df.index)),
                         errors="coerce").fillna(0)
    ie_total = ie_d + ie_r
    ie_active = ie_total >= IE_MIN_THRESHOLD
    ie_dem_share = pd.Series(0.5, index=df.index)
    ie_dem_share[ie_active & (ie_total > 0)] = (
        ie_d[ie_active & (ie_total > 0)] / ie_total[ie_active & (ie_total > 0)]
    )
    eff_weight = _effective_ie_weight()
    ie_adjustment = IE_COEFFICIENT * eff_weight * (ie_dem_share - 0.5) * ie_active

    # Strip IE to get the baseline-without-IE prediction
    predicted_no_ie = predicted_with_ie - ie_adjustment

    # Competitive = within margin of 50%
    competitive = (predicted_no_ie >= (0.5 - margin)) & (predicted_no_ie <= (0.5 + margin))
    return competitive, predicted_no_ie


def scenario_label(env_dial: float) -> str:
    if env_dial >= 0:
        return f"D+{env_dial:.0f}"
    return f"R+{-env_dial:.0f}"


def fmt_seats(result: dict) -> str:
    house = result["expected_house_seats"]
    house_p = result["house_control_prob"] * 100
    senate = result["expected_senate_seats"] + _SENATE_D_HOLDOVER
    senate_p = result["senate_control_prob"] * 100
    return (f"House: {house:5.1f} ({house_p:5.1f}% maj)   "
            f"Senate: {senate:4.1f}/31 ({senate_p:5.1f}% maj)")


def main():
    print("=" * 80)
    print("  TX Legislature 2026 — IE Equalization + Universal Viability Scenario")
    print("=" * 80)
    print()
    print("  Premise: in races that are competitive under the baseline model")
    print("  (predicted D 2p share within 10pp of 50%, EXCLUDING IE effects):")
    print("    Scenario A: IE spending equalizes (neutralized to zero net effect)")
    print("    Scenario B: IE equalizes AND all D challengers hit viability threshold")
    print()

    df = load_districts()
    race_generic = RACE_GENERIC_BALLOT_D_SHARE

    # Use standard env scenarios + current (dedup if current ≈ an existing scenario)
    envs = list(ENV_SCENARIOS)
    if not any(abs(e - CURRENT_ENV) < 0.5 for e in envs):
        envs.append(round(CURRENT_ENV, 1))
    envs = sorted(envs)

    # Storage for output
    all_district_rows = []
    summary_rows = []

    for env_dial in envs:
        env_lbl = scenario_label(env_dial)
        print(f"\n{'━' * 80}")
        print(f"  Environment: {env_lbl}")
        print(f"{'━' * 80}")

        # --- Identify competitive races (without IE) ---
        competitive_mask, pred_no_ie = identify_competitive_no_ie(
            df, env_dial, race_generic, margin=0.10
        )
        n_competitive = int(competitive_mask.sum())

        # Among competitive races, how many have active IE spending?
        ie_d = pd.to_numeric(df.get("ie_d_favor", 0), errors="coerce").fillna(0)
        ie_r = pd.to_numeric(df.get("ie_r_favor", 0), errors="coerce").fillna(0)
        ie_total = ie_d + ie_r
        ie_active = ie_total >= IE_MIN_THRESHOLD
        n_ie_active_competitive = int((competitive_mask & ie_active).sum())

        # Among competitive races, how many lack viability?
        chal_flag = pd.to_numeric(
            df.get("challenger_viability_flag_early",
                   df.get("challenger_viability_flag", pd.Series(0, index=df.index))),
            errors="coerce"
        ).fillna(0)
        n_not_viable = int((competitive_mask & (chal_flag == 0)).sum())

        # Party breakdown of competitive seats
        inc_party = df["incumbent_party"].astype(str).str.upper()
        is_open = df["open_seat"].astype(str).str.lower().isin(["true", "1", "yes"])
        n_comp_r = int((competitive_mask & (inc_party == "R")).sum())
        n_comp_d = int((competitive_mask & (inc_party == "D")).sum())
        n_comp_open = int((competitive_mask & is_open).sum())

        print(f"  Competitive races (within 10pp of 50%, no IE): {n_competitive}")
        print(f"    R-held: {n_comp_r}   D-held: {n_comp_d}   Open: {n_comp_open}")
        print(f"    With active IE (≥${IE_MIN_THRESHOLD/1e3:.0f}K): {n_ie_active_competitive}")
        print(f"    D challenger NOT yet viable: {n_not_viable}")

        # --- Run 1: Baseline (current data, as-is) ---
        res_baseline = run_monte_carlo(df, env_dial, race_generic, N_SIMULATIONS)

        # --- Run 2: IE equalized in competitive races ---
        df_ie_eq = df.copy()
        # Zero out both sides of IE in competitive races → ie_dem_share = 0.5 → zero adjustment
        df_ie_eq.loc[competitive_mask, "ie_d_favor"] = 0
        df_ie_eq.loc[competitive_mask, "ie_r_favor"] = 0
        res_ie_eq = run_monte_carlo(df_ie_eq, env_dial, race_generic, N_SIMULATIONS)

        # --- Run 3: IE equalized + universal viability in competitive races ---
        df_ie_eq_viable = df_ie_eq.copy()
        df_ie_eq_viable.loc[competitive_mask, "challenger_viability_flag_early"] = 1
        res_ie_eq_viable = run_monte_carlo(
            df_ie_eq_viable, env_dial, race_generic, N_SIMULATIONS
        )

        # --- Print comparison ---
        print()
        print(f"  {'Scenario':<42s}  {fmt_seats(res_baseline)}")
        lbl_base = f"{env_lbl} baseline"
        lbl_ie = f"{env_lbl} + IE equalized"
        lbl_both = f"{env_lbl} + IE eq + viability"
        print(f"  {lbl_base:<42s}  {fmt_seats(res_baseline)}")
        print(f"  {lbl_ie:<42s}  {fmt_seats(res_ie_eq)}")
        print(f"  {lbl_both:<42s}  {fmt_seats(res_ie_eq_viable)}")

        # Marginal effects
        delta_ie = res_ie_eq["expected_house_seats"] - res_baseline["expected_house_seats"]
        delta_viab = res_ie_eq_viable["expected_house_seats"] - res_ie_eq["expected_house_seats"]
        delta_total = res_ie_eq_viable["expected_house_seats"] - res_baseline["expected_house_seats"]
        print(f"\n  Marginal House seat impact (expected):")
        print(f"    IE equalization alone:       {delta_ie:+.1f} seats")
        print(f"    Viability flip (on top):     {delta_viab:+.1f} seats")
        print(f"    Combined effect:             {delta_total:+.1f} seats")

        # Senate marginals
        s_delta_ie = (res_ie_eq["expected_senate_seats"] -
                      res_baseline["expected_senate_seats"])
        s_delta_total = (res_ie_eq_viable["expected_senate_seats"] -
                         res_baseline["expected_senate_seats"])
        print(f"    Senate IE equalization:      {s_delta_ie:+.1f} seats")
        print(f"    Senate combined:             {s_delta_total:+.1f} seats")

        # --- Per-district movers (win prob shifted ≥3pp) ---
        wp_base = res_baseline["district_win_probs"]
        wp_both = res_ie_eq_viable["district_win_probs"]
        wp_ie_eq = res_ie_eq["district_win_probs"]
        delta_pp = (wp_both - wp_base)

        movers = df[competitive_mask & (delta_pp.abs() >= 0.03)].copy()
        movers["wp_base"] = wp_base.reindex(movers.index)
        movers["wp_ie_eq"] = wp_ie_eq.reindex(movers.index)
        movers["wp_both"] = wp_both.reindex(movers.index)
        movers["delta_pp"] = delta_pp.reindex(movers.index)
        movers["pred_no_ie"] = pred_no_ie.reindex(movers.index)
        movers = movers.sort_values("delta_pp", ascending=False)

        if not movers.empty:
            print(f"\n  Districts moved ≥3pp by IE equalization + viability ({env_lbl}):")
            print(f"  {'':1s}{'Ch':3s} {'Dist':>4s}  {'Incumbent':26s}  {'Pty':3s}  "
                  f"{'NoIE':>6s}  {'Base':>6s}  {'IE eq':>6s}  {'Both':>6s}  {'Δpp':>6s}")
            for _, r in movers.iterrows():
                ch = r["chamber"][0]
                dist = int(r["district"])
                inc = str(r["incumbent"])[:26]
                pty = str(r["incumbent_party"])[:3]
                is_op = str(r.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
                flag = "○" if is_op else " "
                print(f"  {flag}{ch:3s} {dist:>4d}  {inc:26s}  {pty:3s}  "
                      f"{r['pred_no_ie']*100:5.1f}%  {r['wp_base']*100:5.1f}%  "
                      f"{r['wp_ie_eq']*100:5.1f}%  {r['wp_both']*100:5.1f}%  "
                      f"{r['delta_pp']*100:+5.1f}")

        # --- Flips: races that cross 50% threshold ---
        flips_to_d = df[competitive_mask
                        & (wp_base < 0.5) & (wp_both >= 0.5)].copy()
        flips_to_d["wp_base"] = wp_base.reindex(flips_to_d.index)
        flips_to_d["wp_both"] = wp_both.reindex(flips_to_d.index)
        if not flips_to_d.empty:
            print(f"\n  Races flipping to D-favored (>50%) under {env_lbl} + IE eq + viability:")
            for _, r in flips_to_d.iterrows():
                ch = r["chamber"][0]
                dist = int(r["district"])
                inc = str(r["incumbent"])[:26]
                print(f"    {ch}D{dist:3d}  {inc:26s}  "
                      f"{r['wp_base']*100:.1f}% → {r['wp_both']*100:.1f}%")

        # --- Collect per-district output ---
        for idx, row in df.iterrows():
            all_district_rows.append({
                "env_dial": env_dial,
                "scenario_env": env_lbl,
                "chamber": row["chamber"],
                "district": int(row["district"]),
                "incumbent": row["incumbent"],
                "incumbent_party": row["incumbent_party"],
                "open_seat": row.get("open_seat", ""),
                "dem_pres_2p_baseline": row.get("dem_pres_2p_baseline", ""),
                "competitive": bool(competitive_mask.loc[idx]),
                "pred_no_ie": round(pred_no_ie.loc[idx], 4),
                "wp_baseline": round(wp_base.loc[idx], 4),
                "wp_ie_equalized": round(wp_ie_eq.loc[idx], 4),
                "wp_ie_eq_viable": round(wp_both.loc[idx], 4),
            })

        # Collect summary
        hd_base = res_baseline["house_seat_dist"]
        hd_both = res_ie_eq_viable["house_seat_dist"]
        summary_rows.append({
            "env_dial": env_dial,
            "scenario_env": env_lbl,
            "n_competitive": n_competitive,
            "n_ie_active_competitive": n_ie_active_competitive,
            "n_not_viable": n_not_viable,
            # Baseline
            "house_seats_baseline": round(res_baseline["expected_house_seats"], 1),
            "house_control_baseline": round(res_baseline["house_control_prob"], 4),
            "senate_seats_baseline": round(
                res_baseline["expected_senate_seats"] + _SENATE_D_HOLDOVER, 1),
            "senate_control_baseline": round(res_baseline["senate_control_prob"], 4),
            # IE equalized
            "house_seats_ie_eq": round(res_ie_eq["expected_house_seats"], 1),
            "house_control_ie_eq": round(res_ie_eq["house_control_prob"], 4),
            "senate_seats_ie_eq": round(
                res_ie_eq["expected_senate_seats"] + _SENATE_D_HOLDOVER, 1),
            "senate_control_ie_eq": round(res_ie_eq["senate_control_prob"], 4),
            # IE equalized + viability
            "house_seats_ie_eq_viable": round(res_ie_eq_viable["expected_house_seats"], 1),
            "house_control_ie_eq_viable": round(res_ie_eq_viable["house_control_prob"], 4),
            "senate_seats_ie_eq_viable": round(
                res_ie_eq_viable["expected_senate_seats"] + _SENATE_D_HOLDOVER, 1),
            "senate_control_ie_eq_viable": round(
                res_ie_eq_viable["senate_control_prob"], 4),
            # Marginal effects
            "house_delta_ie_eq": round(delta_ie, 1),
            "house_delta_viability": round(delta_viab, 1),
            "house_delta_combined": round(delta_total, 1),
            # Percentiles (combined scenario)
            "house_p10_baseline": int(np.percentile(hd_base, 10)),
            "house_p90_baseline": int(np.percentile(hd_base, 90)),
            "house_p10_combined": int(np.percentile(hd_both, 10)),
            "house_p90_combined": int(np.percentile(hd_both, 90)),
        })

    # --- Final summary table ---
    print(f"\n\n{'━' * 80}")
    print("  SUMMARY: IE Equalization + Universal Viability Impact Across Environments")
    print(f"{'━' * 80}")
    print(f"  {'Env':>5s}  {'Comp':>4s}  "
          f"{'House Base':>10s}  {'House IE eq':>11s}  {'House Both':>10s}  {'ΔSeats':>6s}  "
          f"{'Maj% Base':>9s}  {'Maj% Both':>9s}")
    print(f"  {'─'*5}  {'─'*4}  {'─'*10}  {'─'*11}  {'─'*10}  {'─'*6}  {'─'*9}  {'─'*9}")
    for s in summary_rows:
        print(f"  {s['scenario_env']:>5s}  {s['n_competitive']:>4d}  "
              f"{s['house_seats_baseline']:>10.1f}  {s['house_seats_ie_eq']:>11.1f}  "
              f"{s['house_seats_ie_eq_viable']:>10.1f}  "
              f"{s['house_delta_combined']:>+5.1f}  "
              f"{s['house_control_baseline']*100:>8.1f}%  "
              f"{s['house_control_ie_eq_viable']*100:>8.1f}%")

    print(f"\n  {'Env':>5s}  "
          f"{'Senate Base':>11s}  {'Senate Both':>11s}  {'ΔSeats':>6s}  "
          f"{'Maj% Base':>9s}  {'Maj% Both':>9s}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*11}  {'─'*6}  {'─'*9}  {'─'*9}")
    for s in summary_rows:
        s_delta = (s["senate_seats_ie_eq_viable"] - s["senate_seats_baseline"])
        print(f"  {s['scenario_env']:>5s}  "
              f"{s['senate_seats_baseline']:>11.1f}  "
              f"{s['senate_seats_ie_eq_viable']:>11.1f}  "
              f"{s_delta:>+5.1f}  "
              f"{s['senate_control_baseline']*100:>8.1f}%  "
              f"{s['senate_control_ie_eq_viable']*100:>8.1f}%")

    # --- Save CSVs ---
    dist_path = OUTPUT / "scenario_ie_equalized.csv"
    pd.DataFrame(all_district_rows).to_csv(dist_path, index=False)
    print(f"\n  Per-district output: {dist_path.relative_to(ROOT)}")

    summ_path = OUTPUT / "scenario_ie_equalized_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)
    print(f"  Summary output:     {summ_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
