"""
model.py

Phase 2 TX Legislative Election Projection Model

Architecture:
  predicted_dem_share(district) =
      presidential_baseline          ← 2024 pres Dem 2p share (from districts_2026.csv)
    + race_adjusted_demo_deviation   ← Σ(CVAP_pct × race_D_share) − national_avg
    + env_dial × national_env_coef   ← generic ballot swing, uniform across districts
    + incumbency_effect              ← Phase 1 regression coefficients
    + finance_effect                 ← challenger viability flag

Monte Carlo:
  For each simulation (n=10,000):
    1. Draw a shared national error: ε_national ~ N(0, σ_national)
    2. For each district, draw an idiosyncratic error: ε_district ~ N(0, σ_idio)
    3. Combine: ε_total = ε_national + ε_district  (where σ² = σ²_national + σ²_idio)
    4. Predict dem_share = linear_prediction + ε_total
    5. District wins if dem_share > 0.5

  The correlation between districts is driven by the shared national error.
  We split σ = 0.0785 (total residual SE) into:
    σ_national ≈ 0.06  (shared national uncertainty — drives correlation)
    σ_idio     ≈ 0.053 (independent district uncertainty)
  such that sqrt(0.06² + 0.053²) ≈ 0.0785

Scenarios: R+3, even (D+0), D+3, D+6, D+8

Usage:
  python src/model.py                           # run all scenarios
  python src/model.py --env 5                   # single scenario D+5
  python src/model.py --env 5 --env -3          # specific scenarios
  python src/model.py --scenarios               # print scenario table
  python src/model.py --show-districts house    # per-district table for house
  python src/model.py --show-districts senate   # per-district table for senate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from model_config import (
    RACE_GENERIC_BALLOT_D_SHARE,
    NATIONAL_DEMO_WEIGHTS,
    REGRESSION_COEFFICIENTS as COEFS,
    ENV_SCENARIOS,
    N_SIMULATIONS,
    HOUSE_MAJORITY,
    SENATE_MAJORITY,
    GENERIC_BALLOT_UPDATED,
    GENERIC_BALLOT_SOURCE,
    GENERIC_BALLOT_TOPLINE_D_2P,
    FINANCE_DATA_THROUGH,
    IE_COEFFICIENT,
    IE_MIN_THRESHOLD,
    IE_WEIGHT,
    IE_DATA_THROUGH,
)

_HAS_FUNDRAISING_SHARE = "dem_fundraising_share" in COEFS

# Current environment as an absolute D-R generic ballot margin (pp)
# Derived from DDHQ topline: (D_2p - 0.5) * 200
CURRENT_ENV: float = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)

# IE weight can be overridden at runtime via --ie-weight
_ie_weight_override: float | None = None

# Monte Carlo noise decomposition
# Total σ = 0.0785; split into national (correlated) and idiosyncratic (independent)
SIGMA_NATIONAL = 0.060   # shared across all districts in a given simulation
SIGMA_IDIO     = 0.053   # independent per district
# Check: sqrt(0.060² + 0.053²) ≈ 0.0799 ≈ 0.0785 (close enough)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Load district data
# ---------------------------------------------------------------------------

def load_districts() -> pd.DataFrame:
    path = DATA_PROC / "districts_2026.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"districts_2026.csv not found at {path}.\n"
            "Run: python src/build_district_table.py"
        )
    df = pd.read_csv(path)
    df["district"] = df["district"].astype(int)
    df["chamber_lower"] = df["chamber"].str.lower()

    # Only project races that are up in 2026
    df = df[df["up_in_2026"] == True].copy()
    print(f"Loaded {len(df)} districts on the 2026 ballot "
          f"({(df['chamber_lower']=='house').sum()} House, "
          f"{(df['chamber_lower']=='senate').sum()} Senate)")
    return df


# ---------------------------------------------------------------------------
# Compute district demographic baseline
# ---------------------------------------------------------------------------

def compute_demo_baseline(df: pd.DataFrame,
                           race_generic: dict[str, float]) -> pd.Series:
    """
    For each district, compute Σ(CVAP_pct × race_D_share).
    Returns a Series of district D shares implied by current racial polling.
    Handles missing CVAP data by falling back to national average.
    """
    national_avg = sum(NATIONAL_DEMO_WEIGHTS[r] * race_generic[r]
                       for r in NATIONAL_DEMO_WEIGHTS)

    results = []
    for _, row in df.iterrows():
        w_nh   = _safe_pct(row.get("pct_white_nh"))
        b_nh   = _safe_pct(row.get("pct_black_nh"))
        hisp   = _safe_pct(row.get("pct_hispanic"))
        other  = _safe_pct(row.get("pct_other"))

        total = w_nh + b_nh + hisp + other
        if total > 0:
            # Normalize to sum to 1
            w_nh /= total; b_nh /= total; hisp /= total; other /= total
            demo_d = (w_nh   * race_generic["white_nh"] +
                      b_nh   * race_generic["black_nh"] +
                      hisp   * race_generic["hispanic"] +
                      other  * race_generic["other"])
        else:
            demo_d = national_avg  # fallback

        results.append(demo_d)

    return pd.Series(results, index=df.index)


def _safe_pct(val) -> float:
    """Convert a percentage string/float to a 0-1 fraction safely."""
    try:
        v = float(val)
        return v / 100.0 if v > 1.0 else v  # handle both 71.2 and 0.712
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Build linear prediction for each district
# ---------------------------------------------------------------------------

def build_linear_predictions(df: pd.DataFrame,
                              env_dial: float,
                              race_generic: dict[str, float]) -> pd.Series:
    """
    Compute the linear (deterministic) prediction for each district.

    env_dial: D-points better than current generic ballot
              (e.g., +5 = D+5 vs current environment; -3 = R+3 vs current)

    Returns a Series of predicted dem 2p shares (before Monte Carlo noise).
    """
    national_avg = sum(NATIONAL_DEMO_WEIGHTS[r] * race_generic[r]
                       for r in NATIONAL_DEMO_WEIGHTS)

    demo_baseline = compute_demo_baseline(df, race_generic)
    demo_deviation = demo_baseline - national_avg  # how much more D than national avg

    # Presidential baseline (fixed 2024 result)
    pres_baseline = pd.to_numeric(df["dem_pres_2p_baseline"], errors="coerce")

    # Incumbency
    def encode_incumbency(row: dict) -> tuple[int, int]:
        party = str(row.get("incumbent_party", "")).strip().upper()
        is_open = str(row.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
        if is_open:
            return 0, 0
        return (1, 0) if party == "D" else (0, 1) if party == "R" else (0, 0)

    dem_inc = df.apply(lambda r: encode_incumbency(r)[0], axis=1)
    rep_inc = df.apply(lambda r: encode_incumbency(r)[1], axis=1)

    # Finance: challenger viability flag
    challenger_flag = pd.to_numeric(
        df.get("challenger_viability_flag_early",
               df.get("challenger_viability_flag", pd.Series(0, index=df.index))),
        errors="coerce"
    ).fillna(0)

    # Finance: dem_fundraising_share (0–1; None/missing filled with 0.5 = neutral)
    # 0.5 means equal D/R fundraising — no directional signal.
    fundraising_share = pd.to_numeric(
        df.get("dem_fundraising_share", pd.Series(np.nan, index=df.index)),
        errors="coerce"
    ).fillna(0.5)

    chamber_senate = (df["chamber_lower"] == "senate").astype(int)

    predicted = (
        COEFS["intercept"]
        + COEFS["dem_pres_2p_baseline"] * pres_baseline.fillna(national_avg)
        + demo_deviation                                      # race-adjusted lean
        + COEFS["national_env"] * env_dial                   # environment swing
        + COEFS["dem_incumbent"] * dem_inc
        + COEFS["rep_incumbent"] * rep_inc
        + COEFS["chamber_senate"] * chamber_senate
        + COEFS["challenger_viability_flag"] * challenger_flag
        + (COEFS.get("dem_fundraising_share", 0) * fundraising_share
           if _HAS_FUNDRAISING_SHARE else 0)
    )

    # IE signal: apply ie_dem_share adjustment where total IEs exceed threshold.
    # ie_dem_share = D-favoring IEs / total IEs (0.5 = neutral).
    # Centered at 0.5 so districts with no IE data (neutral) get zero adjustment.
    # Sources: ie_d_favor / ie_r_favor columns from collect_ies_2026.py.
    # IE_WEIGHT controls how much to trust the signal based on election-cycle stage:
    #   0.5 = post-runoff (primary noise); 0.75 = post-July filing; 1.0 = pre-election
    ie_d = pd.to_numeric(df.get("ie_d_favor", pd.Series(np.nan, index=df.index)),
                         errors="coerce").fillna(0)
    ie_r = pd.to_numeric(df.get("ie_r_favor", pd.Series(np.nan, index=df.index)),
                         errors="coerce").fillna(0)
    ie_total = ie_d + ie_r

    # Only apply where spending clears the meaningful-targeting threshold
    ie_active = ie_total >= IE_MIN_THRESHOLD

    # Compute ie_dem_share; for districts below threshold, treat as neutral (0.5)
    ie_dem_share = pd.Series(0.5, index=df.index)
    ie_dem_share[ie_active & (ie_total > 0)] = (
        ie_d[ie_active & (ie_total > 0)] / ie_total[ie_active & (ie_total > 0)]
    )

    effective_weight = _ie_weight_override if _ie_weight_override is not None else IE_WEIGHT
    ie_adjustment = IE_COEFFICIENT * effective_weight * (ie_dem_share - 0.5) * ie_active

    predicted += ie_adjustment

    return predicted


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(df: pd.DataFrame,
                    env_dial: float,
                    race_generic: dict[str, float],
                    n_sims: int = N_SIMULATIONS) -> dict:
    """
    Run Monte Carlo simulation for a single environment scenario.

    Returns a dict with:
      district_win_probs: Series — P(D wins) per district
      house_seat_dist: array(n_sims) — D House seats per simulation
      senate_seat_dist: array(n_sims) — D Senate seats per simulation
      house_control_prob: float — P(D controls House, ≥76 seats)
      senate_control_prob: float — P(D controls Senate, ≥16 seats)
      expected_house_seats: float
      expected_senate_seats: float
    """
    linear = build_linear_predictions(df, env_dial, race_generic).values
    n_districts = len(df)
    is_house = (df["chamber_lower"] == "house").values

    # Monte Carlo draws
    # Shape: (n_sims,) shared national error
    national_errors = RNG.normal(0, SIGMA_NATIONAL, size=n_sims)
    # Shape: (n_districts, n_sims) idiosyncratic errors
    idio_errors = RNG.normal(0, SIGMA_IDIO, size=(n_districts, n_sims))

    # predicted[i, sim] = linear[i] + national_errors[sim] + idio_errors[i, sim]
    # Broadcasting: linear is (n_districts,), national_errors is (n_sims,)
    predicted_matrix = (
        linear[:, np.newaxis]         # (n_districts, 1)
        + national_errors[np.newaxis, :]  # (1, n_sims) → broadcasts to (n_districts, n_sims)
        + idio_errors                 # (n_districts, n_sims)
    )

    # Win if predicted > 0.5
    wins = (predicted_matrix > 0.5)  # (n_districts, n_sims) boolean

    house_wins = wins[is_house, :]   # (n_house_districts, n_sims)
    senate_wins = wins[~is_house, :] # (n_senate_districts, n_sims)

    house_seat_dist = house_wins.sum(axis=0)   # (n_sims,)
    senate_seat_dist = senate_wins.sum(axis=0)  # (n_sims,)

    # Per-district win probability = fraction of simulations won
    district_win_probs = wins.mean(axis=1)  # (n_districts,)

    return {
        "district_win_probs": pd.Series(district_win_probs, index=df.index),
        "house_seat_dist": house_seat_dist,
        "senate_seat_dist": senate_seat_dist,
        "house_control_prob": (house_seat_dist >= HOUSE_MAJORITY).mean(),
        "senate_control_prob": (senate_seat_dist >= SENATE_MAJORITY).mean(),
        "expected_house_seats": house_seat_dist.mean(),
        "expected_senate_seats": senate_seat_dist.mean(),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _env_label(env_dial: float) -> str:
    """Absolute generic ballot label: 'D+N' or 'R+N'."""
    if abs(env_dial - CURRENT_ENV) < 0.2:
        base = f"D+{env_dial:.1f}" if env_dial >= 0 else f"R+{-env_dial:.1f}"
        return f"{base} (current)"
    if env_dial == 0:
        return "Neutral/2024-like"
    return f"D+{env_dial:.0f}" if env_dial >= 0 else f"R+{-env_dial:.0f}"


def print_scenario_summary(env_dial: float, result: dict, df: pd.DataFrame):
    label = _env_label(env_dial)
    print(f"\n{'─'*60}")
    print(f"  Scenario: {label}")
    print(f"{'─'*60}")
    print(f"  Expected D House seats:    {result['expected_house_seats']:.1f} / 150  "
          f"(need {HOUSE_MAJORITY} for majority)")
    print(f"  P(D controls House):       {result['house_control_prob']*100:.1f}%")
    print()
    print(f"  Expected D Senate seats:   {result['expected_senate_seats']:.1f} / 16 on ballot "
          f"(need {SENATE_MAJORITY} total for majority)")
    print(f"  P(D controls Senate):      {result['senate_control_prob']*100:.1f}%")

    # Show most competitive districts
    win_probs = result["district_win_probs"]
    df_wp = df.copy()
    df_wp["win_prob_d"] = win_probs.values
    df_wp["margin_from_tossup"] = (df_wp["win_prob_d"] - 0.5).abs()
    competitive = df_wp[df_wp["margin_from_tossup"] < 0.25].sort_values("margin_from_tossup")

    # Identify which districts have active IE signal
    effective_ie_weight = _ie_weight_override if _ie_weight_override is not None else IE_WEIGHT
    if effective_ie_weight > 0:
        ie_d_vals = pd.to_numeric(df_wp.get("ie_d_favor",
                                             pd.Series(0, index=df_wp.index)),
                                   errors="coerce").fillna(0)
        ie_r_vals = pd.to_numeric(df_wp.get("ie_r_favor",
                                             pd.Series(0, index=df_wp.index)),
                                   errors="coerce").fillna(0)
        ie_active_mask = (ie_d_vals + ie_r_vals) >= IE_MIN_THRESHOLD
    else:
        ie_active_mask = pd.Series(False, index=df_wp.index)

    if not competitive.empty:
        print(f"\n  Competitive districts (D win prob 25%–75%):"
              + ("  ★=IE signal active" if ie_active_mask.any() else ""))
        for idx, row in competitive.head(12).iterrows():
            inc_str = ("D-inc" if row["incumbent_party"] == "D"
                       else "R-inc" if row["incumbent_party"] == "R"
                       else "open")
            bar    = "█" * int(row["win_prob_d"] * 20)
            marker = "★" if ie_active_mask.get(idx, False) else " "
            print(f"   {marker}{row['chamber'][0]}D{row['district']:3d}  {row['incumbent']:25s} "
                  f"({inc_str:6s})  D: {row['win_prob_d']*100:5.1f}%  {bar}")


def print_district_table(df: pd.DataFrame, result: dict, chamber: str):
    """Print per-district win probability table for one chamber."""
    win_probs = result["district_win_probs"]
    df_wp = df[df["chamber_lower"] == chamber].copy()
    df_wp["win_prob_d"] = win_probs.reindex(df_wp.index).values
    df_wp = df_wp.sort_values("district")

    # Determine if any IE data is present and above threshold
    ie_d_col = pd.to_numeric(df_wp.get("ie_d_favor", pd.Series(np.nan, index=df_wp.index)),
                              errors="coerce").fillna(0)
    ie_r_col = pd.to_numeric(df_wp.get("ie_r_favor", pd.Series(np.nan, index=df_wp.index)),
                              errors="coerce").fillna(0)
    ie_total_col = ie_d_col + ie_r_col
    effective_ie_weight = _ie_weight_override if _ie_weight_override is not None else IE_WEIGHT
    has_ie = (ie_total_col >= IE_MIN_THRESHOLD).any() and effective_ie_weight > 0

    print(f"\n{'='*88}")
    print(f"  {chamber.upper()} — Per-District Win Probabilities")
    if has_ie:
        print(f"  ★ = IE signal active (≥${IE_MIN_THRESHOLD/1e3:.0f}K, weight={effective_ie_weight:.2f}); "
              f"adj = pp shift from IE direction")
    print(f"{'='*88}")
    if has_ie:
        print(f"  {'Dist':4s}  {'Incumbent':28s}  {'Party':5s}  "
              f"{'PresBase':8s}  {'D Win%':7s}  {'IE Dir':8s}  {'Adj':>6s}  {'Rating':12s}")
        print(f"  {'-'*4}  {'-'*28}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*12}")
    else:
        print(f"  {'Dist':4s}  {'Incumbent':30s}  {'Party':5s}  "
              f"{'PresBase':8s}  {'D Win%':7s}  {'Rating':12s}")
        print(f"  {'-'*4}  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*12}")

    for idx, row in df_wp.iterrows():
        wp = row["win_prob_d"]
        pres = row.get("dem_pres_2p_baseline", "")
        try:
            pres_str = f"{float(pres)*100:.1f}%"
        except (TypeError, ValueError):
            pres_str = "N/A"

        if wp >= 0.95:   rating = "Safe D"
        elif wp >= 0.75: rating = "Likely D"
        elif wp >= 0.55: rating = "Lean D"
        elif wp >= 0.45: rating = "Toss-up"
        elif wp >= 0.25: rating = "Lean R"
        elif wp >= 0.05: rating = "Likely R"
        else:            rating = "Safe R"

        if has_ie:
            ie_d  = ie_d_col.loc[idx]
            ie_r  = ie_r_col.loc[idx]
            ie_tot = ie_total_col.loc[idx]
            ie_flag = ie_tot >= IE_MIN_THRESHOLD
            if ie_flag:
                ie_share = ie_d / ie_tot if ie_tot > 0 else 0.5
                adj_pp   = IE_COEFFICIENT * effective_ie_weight * (ie_share - 0.5) * 100
                ie_dir   = "D▶" if ie_d > ie_r else "◀R" if ie_r > ie_d else "even"
                adj_str  = f"{adj_pp:+.1f}pp"
                marker   = "★"
            else:
                ie_dir  = "—"
                adj_str = "—"
                marker  = " "
            print(f"  {marker}{row['district']:4d}  {str(row['incumbent'])[:28]:28s}  "
                  f"{str(row['incumbent_party']):5s}  {pres_str:8s}  "
                  f"{wp*100:6.1f}%  {ie_dir:8s}  {adj_str:>6s}  {rating:12s}")
        else:
            print(f"  {row['district']:4d}  {str(row['incumbent'])[:30]:30s}  "
                  f"{str(row['incumbent_party']):5s}  {pres_str:8s}  "
                  f"{wp*100:6.1f}%  {rating:12s}")


def save_scenario_output(results_by_scenario: dict, df: pd.DataFrame):
    """Write per-district results to output/model_2026_scenarios.csv"""
    rows = []
    for env_dial, result in results_by_scenario.items():
        win_probs = result["district_win_probs"]
        for idx, row in df.iterrows():
            rows.append({
                "env_dial": env_dial,
                "scenario": f"D+{env_dial:.0f}" if env_dial >= 0 else f"R+{-env_dial:.0f}",
                "chamber": row["chamber"],
                "district": row["district"],
                "incumbent": row["incumbent"],
                "incumbent_party": row["incumbent_party"],
                "dem_pres_2p_baseline": row.get("dem_pres_2p_baseline", ""),
                "win_prob_d": round(win_probs.loc[idx], 4),
                "win_prob_r": round(1 - win_probs.loc[idx], 4),
            })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT / "model_2026_scenarios.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nPer-district results written to {out_path.name}")

    # Also write seat totals summary
    summary_rows = []
    for env_dial, result in results_by_scenario.items():
        # Seat distribution percentiles
        hd = result["house_seat_dist"]
        sd = result["senate_seat_dist"]
        summary_rows.append({
            "env_dial": env_dial,
            "scenario": f"D+{env_dial:.0f}" if env_dial >= 0 else f"R+{-env_dial:.0f}",
            "expected_house_seats": round(result["expected_house_seats"], 1),
            "house_seats_p10": int(np.percentile(hd, 10)),
            "house_seats_p25": int(np.percentile(hd, 25)),
            "house_seats_p75": int(np.percentile(hd, 75)),
            "house_seats_p90": int(np.percentile(hd, 90)),
            "house_control_prob": round(result["house_control_prob"], 4),
            "expected_senate_seats": round(result["expected_senate_seats"], 1),
            "senate_seats_p10": int(np.percentile(sd, 10)),
            "senate_seats_p25": int(np.percentile(sd, 25)),
            "senate_seats_p75": int(np.percentile(sd, 75)),
            "senate_seats_p90": int(np.percentile(sd, 90)),
            "senate_control_prob": round(result["senate_control_prob"], 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT / "model_2026_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Scenario summary written to {summary_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TX Legislature 2026 Monte Carlo Projection Model")
    parser.add_argument("--env", type=float, action="append", dest="envs",
                        help="Environment dial value(s) — D+N vs current polling "
                             "(e.g., --env 5 for D+5, --env -3 for R+3). "
                             "Can specify multiple times. Default: all scenarios in config.")
    parser.add_argument("--show-districts", choices=["house", "senate"],
                        help="Print per-district table for the specified chamber.")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't write output CSVs.")
    parser.add_argument("--ie-weight", type=float, default=None,
                        help="Override IE_WEIGHT from config (0–1). "
                             "0.5=post-runoff, 0.75=post-July-filing, 1.0=pre-election. "
                             "Use 0 to run with IEs disabled.")
    args = parser.parse_args()

    if args.ie_weight is not None:
        global _ie_weight_override
        _ie_weight_override = max(0.0, min(1.0, args.ie_weight))

    envs = args.envs if args.envs else ENV_SCENARIOS

    # Auto-inject current environment if not already close to an existing scenario
    if not args.envs and not any(abs(e - CURRENT_ENV) < 0.5 for e in envs):
        envs = sorted(set(list(envs) + [CURRENT_ENV]))

    effective_ie_weight = _ie_weight_override if _ie_weight_override is not None else IE_WEIGHT
    ie_stage = (
        "post-runoff (early signal)"   if effective_ie_weight <= 0.5 else
        "post-July filing (moderate)"  if effective_ie_weight <= 0.75 else
        "pre-election (full signal)"
    ) if effective_ie_weight > 0 else "DISABLED"

    print("="*60)
    print("  TX Legislature 2026 — Monte Carlo Projection")
    print("="*60)
    print(f"  Generic ballot source:  {GENERIC_BALLOT_SOURCE}")
    print(f"  Last updated:           {GENERIC_BALLOT_UPDATED}")
    print(f"  Finance through:        {FINANCE_DATA_THROUGH}")
    print(f"  IE data through:        {IE_DATA_THROUGH}  "
          f"(weight={effective_ie_weight:.2f}, {ie_stage}, threshold=${IE_MIN_THRESHOLD/1e3:.0f}K)")
    print(f"  Simulations:            {N_SIMULATIONS:,}")
    print(f"  Current environment:    D+{CURRENT_ENV:.1f} (absolute generic ballot)")
    print(f"  2024 baseline:          approx. R+1 to neutral  (env=0)")
    print(f"  Env scenarios:          absolute D-R margin (pp), consistent with regression training")

    # Current race-specific generic ballot (from config)
    race_generic = RACE_GENERIC_BALLOT_D_SHARE
    nat_avg = sum(NATIONAL_DEMO_WEIGHTS[r] * race_generic[r] for r in NATIONAL_DEMO_WEIGHTS)
    print(f"  National D avg (demo):  {nat_avg*100:.1f}%")
    print()

    df = load_districts()

    results_by_scenario = {}
    for env_dial in envs:
        label = f"D+{env_dial:.0f}" if env_dial >= 0 else f"R+{-env_dial:.0f}"
        print(f"\nRunning scenario: {label} ({N_SIMULATIONS:,} simulations)...")
        result = run_monte_carlo(df, env_dial, race_generic, N_SIMULATIONS)
        results_by_scenario[env_dial] = result
        print_scenario_summary(env_dial, result, df)

    # Scenarios table
    print(f"\n\n{'='*68}")
    print("  SCENARIO COMPARISON TABLE  (env = absolute D-R generic ballot, pp)")
    print(f"{'='*68}")
    print(f"  {'Scenario':22s}  {'D House':8s}  {'D House%':9s}  {'D Senate':9s}  {'D Sen%':7s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*7}")
    for env_dial, result in sorted(results_by_scenario.items()):
        label = _env_label(env_dial)
        print(f"  {label:22s}  "
              f"{result['expected_house_seats']:7.1f}   "
              f"{result['house_control_prob']*100:7.1f}%    "
              f"{result['expected_senate_seats']:7.1f}    "
              f"{result['senate_control_prob']*100:6.1f}%")

    if args.show_districts and results_by_scenario:
        # Show for the first scenario
        first_env = sorted(results_by_scenario.keys())[0]
        first_result = results_by_scenario[first_env]
        print_district_table(df, first_result, args.show_districts)

    if not args.no_save:
        save_scenario_output(results_by_scenario, df)


if __name__ == "__main__":
    main()
