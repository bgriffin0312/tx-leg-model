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
import re
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
    WAR_PERSISTENCE_COEF,
)

_HAS_FUNDRAISING_SHARE = "dem_fundraising_share" in COEFS

# Current environment as an absolute D-R generic ballot margin (pp)
# Derived from DDHQ topline: (D_2p - 0.5) * 200
CURRENT_ENV: float = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)

# IE weight can be overridden at runtime via --ie-weight
_ie_weight_override: float | None = None

# Monte Carlo noise decomposition
# Total σ = 0.0785; split into national (correlated) and idiosyncratic (independent)
SIGMA_TOTAL = 0.0785

SIGMA_SPLITS = {
    "high-corr": {  # current default: ~58% shared variance
        "national": 0.060,
        "idio":     (SIGMA_TOTAL**2 - 0.060**2)**0.5,  # = 0.05062
        "desc":     "high correlation (58% shared variance)",
    },
    "low-corr": {   # proposed: ~33% shared variance
        "national": 0.045,
        "idio":     (SIGMA_TOTAL**2 - 0.045**2)**0.5,  # = 0.06437
        "desc":     "low correlation (33% shared variance)",
    },
}
# Both satisfy: sqrt(national² + idio²) = SIGMA_TOTAL = 0.0785 exactly

_sigma_split = "high-corr"  # active split; changed via --sigma-split
SIGMA_NATIONAL = SIGMA_SPLITS[_sigma_split]["national"]
SIGMA_IDIO     = SIGMA_SPLITS[_sigma_split]["idio"]

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# WAR lookup helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Normalize a candidate name for matching against the WAR table."""
    if not name or (isinstance(name, float) and np.isnan(name)):
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_war_cache: dict | None = None   # lazy-loaded; None means not yet attempted
_war_names_by_party: dict | None = None  # {party: [(norm_name, tokens, avg_war), ...]}


def _fuzzy_war_match(query_norm: str, party: str) -> float | None:
    """
    Try fuzzy matching against WAR candidates when exact match fails.
    Strategy 1: subset match — all tokens in shorter name appear in longer name
                (handles middle names: "john bryant" ⊂ "john wiley bryant")
    Strategy 2: last-name + first-initial — "jm lozano" matches "jose manuel lozano"
                if last tokens match and first chars of first tokens match
    Returns avg_war or None.
    """
    if not _war_names_by_party or party not in _war_names_by_party:
        return None

    query_tokens = set(query_norm.split())
    if not query_tokens:
        return None
    query_first_char = query_norm.split()[0][0] if query_norm.split() else ""
    query_last = query_norm.split()[-1] if query_norm.split() else ""

    for cand_norm, cand_tokens, avg_war in _war_names_by_party[party]:
        # Strategy 1: subset match (all tokens of shorter name in longer)
        if query_tokens and cand_tokens:
            shorter, longer = (query_tokens, cand_tokens) if len(query_tokens) <= len(cand_tokens) else (cand_tokens, query_tokens)
            if shorter <= longer and len(shorter) >= 2:
                return avg_war

        # Strategy 2: last name + first initial
        cand_first_char = cand_norm.split()[0][0] if cand_norm.split() else ""
        cand_last = cand_norm.split()[-1] if cand_norm.split() else ""
        if (query_last == cand_last and query_last
                and query_first_char == cand_first_char and query_first_char):
            return avg_war

    return None


def _get_war_lookup() -> dict[tuple[str, str], float]:
    """
    Return {(candidate_norm, party): avg_war} from candidate_war.csv.
    avg_war is the simple per-race average WAR in percentage-point units;
    we return it as-is and convert at call site (divide by 100).
    Uses avg_war (not career_war) because the persistence coefficient β=0.46
    was estimated from per-race WAR regressions, not cumulative career sums.
    Loaded once; returns {} if the file doesn't exist.
    Also builds _war_names_by_party for fuzzy matching.
    """
    global _war_cache, _war_names_by_party
    if _war_cache is not None:
        return _war_cache

    war_path = DATA_PROC / "candidate_war.csv"
    if not war_path.exists():
        print("  [WAR] candidate_war.csv not found — WAR term disabled.")
        _war_cache = {}
        _war_names_by_party = {}
        return _war_cache

    df = pd.read_csv(war_path)
    lookup: dict[tuple[str, str], float] = {}
    names_by_party: dict[str, list] = {"D": [], "R": []}
    for _, row in df.iterrows():
        norm_name = str(row.get("candidate_norm", "")).strip()
        party = str(row.get("party", "")).strip().upper()
        avg_war = row.get("avg_war")
        if norm_name and party in ("D", "R") and pd.notna(avg_war):
            lookup[(norm_name, party)] = float(avg_war)
            names_by_party.setdefault(party, []).append(
                (norm_name, set(norm_name.split()), float(avg_war))
            )

    print(f"  [WAR] Loaded {len(lookup)} candidates with avg WAR data.")
    _war_cache = lookup
    _war_names_by_party = names_by_party
    return _war_cache


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
    # IMPORTANT: This 0.5 fill is required by collect_finance_2026.py which
    # outputs None for districts with no D+R totals (open seats without
    # nominee-matched data, districts with no TEC filings, etc.)
    fundraising_share = pd.to_numeric(
        df.get("dem_fundraising_share", pd.Series(np.nan, index=df.index)),
        errors="coerce"
    )
    # Nullify dem_fundraising_share when total raised is below $10K —
    # one-sided filings produce 0.0 or 1.0 that are noise, not signal
    inc_raised = pd.to_numeric(df.get("incumbent_raised", 0), errors="coerce").fillna(0)
    chal_raised = pd.to_numeric(df.get("challenger_raised", 0), errors="coerce").fillna(0)
    total_raised = inc_raised + chal_raised
    low_total_mask = total_raised < 10_000
    n_nullified = (low_total_mask & fundraising_share.notna()).sum()
    fundraising_share[low_total_mask] = np.nan
    fundraising_share = fundraising_share.fillna(0.5)
    if n_nullified > 0:
        print(f"  dem_fundraising_share nullified for {n_nullified} districts "
              f"below $10K total raised threshold")

    chamber_senate = (df["chamber_lower"] == "senate").astype(int)

    # Finance term (used for districts without WAR data)
    finance_term = (
        COEFS["challenger_viability_flag"] * challenger_flag
        + (COEFS.get("dem_fundraising_share", 0) * fundraising_share
           if _HAS_FUNDRAISING_SHARE else 0)
    )

    # Structural baseline (no finance, no WAR yet)
    predicted = (
        COEFS["intercept"]
        + COEFS["dem_pres_2p_baseline"] * pres_baseline.fillna(national_avg)
        + demo_deviation                                      # race-adjusted lean
        # env_dial is absolute D-R margin in pp, matching regression
        # training data (2018=8.6, 2022=-2.8, 2024=-3.2)
        + COEFS["national_env"] * env_dial                   # environment swing
        + COEFS["dem_incumbent"] * dem_inc
        + COEFS["rep_incumbent"] * rep_inc
        + COEFS["chamber_senate"] * chamber_senate
    )

    # ---------------------------------------------------------------------------
    # Dual-track: WAR persistence (incumbents with career history) vs. finance
    # ---------------------------------------------------------------------------
    # For incumbents found in candidate_war.csv, we:
    #   • skip challenger_viability_flag and dem_fundraising_share (WAR already
    #     captures fundraising ability as a quality signal, so including both
    #     would double-count it)
    #   • add WAR_PERSISTENCE_COEF × avg_war as a quality adjustment
    #
    # avg_war sign convention (from compute_war.py):
    #   positive WAR → candidate overperforms fundamentals for THEIR party
    #   For D incumbents: positive WAR → higher dem_2p → add term
    #   For R incumbents: positive WAR → lower dem_2p → subtract term
    #
    # avg_war is per-race average WAR in percentage-point units; divide by 100.
    # We use avg_war (not career_war) because β=0.46 was estimated from
    # per-race regressions — applying it to cumulative career sums would
    # overstate the effect by ~2x for multi-cycle candidates.
    # ---------------------------------------------------------------------------
    war_lookup = _get_war_lookup()
    war_adjustments = pd.Series(0.0, index=df.index)
    has_war = pd.Series(False, index=df.index)

    n_exact = 0
    n_fuzzy = 0
    if war_lookup:
        for idx, row in df.iterrows():
            inc_name  = str(row.get("incumbent", "")).strip()
            inc_party = str(row.get("incumbent_party", "")).strip().upper()
            is_open = str(row.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
            if not inc_name or inc_party not in ("D", "R") or is_open:
                continue
            inc_norm = _normalize_name(inc_name)
            # Try exact match first
            avg_war = war_lookup.get((inc_norm, inc_party))
            match_type = "exact" if avg_war is not None else None
            # Try fuzzy match if exact fails
            if avg_war is None:
                avg_war = _fuzzy_war_match(inc_norm, inc_party)
                if avg_war is not None:
                    match_type = "fuzzy"
            if avg_war is not None:
                party_sign = 1 if inc_party == "D" else -1
                war_adjustments[idx] = party_sign * WAR_PERSISTENCE_COEF * avg_war / 100.0
                has_war[idx] = True
                if match_type == "exact":
                    n_exact += 1
                else:
                    n_fuzzy += 1

    # Apply: WAR districts get war_adjustment; others get finance_term
    predicted += np.where(has_war, war_adjustments, finance_term)

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
            is_open = str(row.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
            if is_open:
                inc_str = "open"
            elif row["incumbent_party"] == "D":
                inc_str = "D-inc"
            elif row["incumbent_party"] == "R":
                inc_str = "R-inc"
            else:
                inc_str = "open"
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
    parser.add_argument("--sigma-split", choices=list(SIGMA_SPLITS.keys()) + ["compare"],
                        default=None,
                        help="Correlation structure: 'high-corr' (default, 58%% shared), "
                             "'low-corr' (33%% shared), or 'compare' (run both and show diff).")
    args = parser.parse_args()

    if args.ie_weight is not None:
        global _ie_weight_override
        _ie_weight_override = max(0.0, min(1.0, args.ie_weight))

    if args.sigma_split and args.sigma_split != "compare":
        global SIGMA_NATIONAL, SIGMA_IDIO, _sigma_split
        _sigma_split = args.sigma_split
        SIGMA_NATIONAL = SIGMA_SPLITS[_sigma_split]["national"]
        SIGMA_IDIO = SIGMA_SPLITS[_sigma_split]["idio"]

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

    # Pre-load WAR table so the count prints before scenario runs
    war_lookup = _get_war_lookup()
    if war_lookup:
        # Count how many on-ballot incumbents have WAR data (exact + fuzzy)
        n_exact_pre = 0
        n_fuzzy_pre = 0
        for _, row in df.iterrows():
            inc_party = str(row.get("incumbent_party", "")).strip().upper()
            if inc_party not in ("D", "R"):
                continue
            is_open = str(row.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
            if is_open:
                continue
            inc_norm = _normalize_name(str(row.get("incumbent", "")).strip())
            if (inc_norm, inc_party) in war_lookup:
                n_exact_pre += 1
            elif _fuzzy_war_match(inc_norm, inc_party) is not None:
                n_fuzzy_pre += 1
        n_total = n_exact_pre + n_fuzzy_pre
        print(f"  WAR persistence:        β={WAR_PERSISTENCE_COEF}  "
              f"({n_total}/{len(df)} incumbents matched: "
              f"{n_exact_pre} exact, {n_fuzzy_pre} fuzzy)")
    print()

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

    if not args.no_save and (not args.sigma_split or args.sigma_split != "compare"):
        save_scenario_output(results_by_scenario, df)

    # --sigma-split compare: run both parameterizations and show side-by-side
    if args.sigma_split == "compare":
        print(f"\n\n{'='*72}")
        print("  SIGMA SPLIT COMPARISON  (D+5 scenario)")
        print(f"{'='*72}")

        compare_env = CURRENT_ENV
        compare_results = {}

        for split_name, split_cfg in SIGMA_SPLITS.items():
            # Temporarily override sigmas
            saved_nat, saved_idio = SIGMA_NATIONAL, SIGMA_IDIO
            # Need to modify module-level vars for run_monte_carlo
            import model as _self
            _self.SIGMA_NATIONAL = split_cfg["national"]
            _self.SIGMA_IDIO = split_cfg["idio"]

            result = run_monte_carlo(df, compare_env, race_generic, N_SIMULATIONS)
            compare_results[split_name] = result

            _self.SIGMA_NATIONAL = saved_nat
            _self.SIGMA_IDIO = saved_idio

        print(f"\n  {'':20s}  {'high-corr':>12s}  {'low-corr':>12s}  {'Difference':>12s}")
        print(f"  {'-'*60}")

        hi = compare_results["high-corr"]
        lo = compare_results["low-corr"]

        metrics = [
            ("Expected D House",    hi["expected_house_seats"],   lo["expected_house_seats"]),
            ("P(D House majority)", hi["house_control_prob"]*100, lo["house_control_prob"]*100),
            ("House P10",           np.percentile(hi["house_seat_dist"], 10),
                                    np.percentile(lo["house_seat_dist"], 10)),
            ("House P25",           np.percentile(hi["house_seat_dist"], 25),
                                    np.percentile(lo["house_seat_dist"], 25)),
            ("House P75",           np.percentile(hi["house_seat_dist"], 75),
                                    np.percentile(lo["house_seat_dist"], 75)),
            ("House P90",           np.percentile(hi["house_seat_dist"], 90),
                                    np.percentile(lo["house_seat_dist"], 90)),
            ("P10-P90 spread",      np.percentile(hi["house_seat_dist"], 90) - np.percentile(hi["house_seat_dist"], 10),
                                    np.percentile(lo["house_seat_dist"], 90) - np.percentile(lo["house_seat_dist"], 10)),
            ("Expected D Senate",   hi["expected_senate_seats"],  lo["expected_senate_seats"]),
        ]

        for label, hi_val, lo_val in metrics:
            fmt = ".1f" if "P(" not in label else ".1f"
            suffix = "%" if "P(" in label else ""
            print(f"  {label:20s}  {hi_val:>11{fmt}}{suffix}  {lo_val:>11{fmt}}{suffix}  "
                  f"{lo_val - hi_val:>+11{fmt}}{suffix}")

        print(f"\n  high-corr: σ_nat={SIGMA_SPLITS['high-corr']['national']}, "
              f"σ_idio={SIGMA_SPLITS['high-corr']['idio']:.3f} "
              f"({SIGMA_SPLITS['high-corr']['desc']})")
        print(f"  low-corr:  σ_nat={SIGMA_SPLITS['low-corr']['national']}, "
              f"σ_idio={SIGMA_SPLITS['low-corr']['idio']:.3f} "
              f"({SIGMA_SPLITS['low-corr']['desc']})")


if __name__ == "__main__":
    main()
