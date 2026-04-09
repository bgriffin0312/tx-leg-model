"""
backtest.py

Validate the Phase 2 TX legislative model against historical actual outcomes.

Compares model predictions (as of April of the cycle year) against actual
general election results for 2022 and 2018.

What this tests:
  - How well the model's linear predictions correlate with actual dem share
  - Whether predicted win probabilities are calibrated (e.g., districts given
    60% D win prob should win ~60% of the time)
  - How accurately the model predicts seat totals
  - Brier score: average squared error of win probability predictions

Data inputs per cycle:
  - Historical actual results:       data/raw/historical/tx_{chamber}_results_{year}.csv
  - Historical presidential baseline: data/raw/historical/tx_presidential_{chamber}_{pres_year}.csv
  - CVAP demographics:               data/raw/tx_cvap_{chamber}.csv
    (current CVAP used for 2022; approximate for 2018 due to boundary changes)
  - Historical config:               src/backtest_config.py (race-specific generic ballot)

Usage:
  python src/backtest.py --year 2022           # 2022 backtest
  python src/backtest.py --year 2018           # 2018 backtest
  python src/backtest.py                       # both years
  python src/backtest.py --year 2022 --no-mc  # skip Monte Carlo (fast, linear only)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_PROC = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from backtest_config import BACKTEST_CONFIGS

# Monte Carlo parameters (same as model.py)
SIGMA_TOTAL    = 0.0785
SIGMA_NATIONAL = 0.060
SIGMA_IDIO     = (SIGMA_TOTAL**2 - SIGMA_NATIONAL**2)**0.5  # = 0.05062
N_SIMULATIONS  = 10_000

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_historical_results(year: int, chamber: str) -> pd.DataFrame | None:
    """
    Load actual election results for a historical year.
    Returns DataFrame with columns including dem_2p_share, winner_party, contested, etc.
    """
    path = DATA_HIST / f"tx_{chamber}_results_{year}.csv"
    if not path.exists():
        print(f"  WARNING: {path.name} not found. Run collect_historical_results.py first.")
        return None
    df = pd.read_csv(path)
    df["district"] = df["district"].astype(int)
    print(f"  Loaded {len(df)} {chamber} districts for {year}")
    return df


def load_presidential_baseline(pres_year: int, chamber: str) -> pd.DataFrame | None:
    """
    Load presidential results by district for a historical year.
    Tries historical directory first, then falls back to current (2024) data.
    """
    path = DATA_HIST / f"tx_presidential_{chamber}_{pres_year}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["district"] = df["district"].astype(int)
        print(f"  Loaded presidential baseline from {path.name}")
        return df

    print(f"  WARNING: {path.name} not found.")
    print(f"  Run: python src/collect_historical_presidential.py --pres-year {pres_year}")
    return None


def load_cvap(chamber: str) -> pd.DataFrame | None:
    """
    Load CVAP demographics by district.
    Note: This is current (2022-cycle) CVAP. For 2018 backtest, boundaries differ.
    """
    path = DATA_RAW / f"tx_cvap_{chamber}.csv"
    if not path.exists():
        print(f"  WARNING: tx_cvap_{chamber}.csv not found. Run collect_cvap_by_district.py first.")
        return None
    df = pd.read_csv(path)
    df["district"] = df["district"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Build the backtest district DataFrame
# ---------------------------------------------------------------------------

def safe_pct(val) -> float:
    """Convert a percentage string/float to a 0-1 fraction safely."""
    try:
        v = float(val)
        return v / 100.0 if v > 1.0 else v
    except (TypeError, ValueError):
        return 0.0


def build_backtest_df(config: dict, chamber: str,
                      verbose: bool = False) -> pd.DataFrame | None:
    """
    Assemble district-level DataFrame for one chamber's backtest.

    Columns needed for model prediction:
      district, chamber, chamber_lower
      dem_pres_2p_baseline       ← from presidential baseline file
      pct_white_nh, pct_black_nh, pct_hispanic, pct_other  ← from CVAP
      dem_incumbent, rep_incumbent                          ← from results (r_/d_incumbent)
      challenger_viability_flag  ← 0 (no historical finance data)
      open_seat                  ← inferred (no incumbent on either side)
      actual_dem_2p_share        ← for validation
      actual_winner_party        ← for validation
      on_ballot, contested        ← filter flags
    """
    year        = config["results_year"]
    pres_year   = config["pres_year"]
    map_year    = config["map_year"]

    # Load finance data if configured
    finance_df = None
    if config.get("use_finance") and config.get("finance_file"):
        fin_path = DATA_HIST / config["finance_file"]
        if fin_path.exists():
            finance_df = pd.read_csv(fin_path)
            finance_df["district"] = finance_df["district"].astype(int)
            print(f"  Loaded finance data: {fin_path.name} ({len(finance_df)} rows)")
        else:
            print(f"  WARNING: Finance file {fin_path.name} not found")

    # Load actual results
    results = load_historical_results(year, chamber)
    if results is None:
        return None

    # Filter to districts on ballot (all House; only specific Senate seats per cycle)
    if chamber == "senate":
        senate_on_ballot = config.get("senate_districts_on_ballot", set(range(1, 32)))
        results = results[results["district"].isin(senate_on_ballot)].copy()

    # Filter to districts actually on ballot
    if "on_ballot" in results.columns:
        results = results[results["on_ballot"] == True].copy()

    # Load presidential baseline
    pres_df = load_presidential_baseline(pres_year, chamber)

    # Load CVAP (current boundaries — approximate for 2018)
    cvap_df = load_cvap(chamber)
    if map_year == 2018 and cvap_df is not None:
        print(f"  NOTE: Using current CVAP for 2018 backtest (boundary approximation)")

    # Build one row per district
    rows = []
    for _, res_row in results.iterrows():
        district = int(res_row["district"])

        # Presidential baseline
        dem_pres_2p = None
        if pres_df is not None:
            pres_match = pres_df[pres_df["district"] == district]
            if not pres_match.empty:
                dem_pres_2p = float(pres_match.iloc[0]["dem_pres_2p_baseline"])

        # CVAP demographics
        pct_white_nh = pct_black_nh = pct_hispanic = pct_other = None
        if cvap_df is not None:
            cvap_match = cvap_df[cvap_df["district"] == district]
            if not cvap_match.empty:
                c = cvap_match.iloc[0]
                pct_white_nh = safe_pct(c.get("pct_white_nh", 0))
                pct_black_nh = safe_pct(c.get("pct_black_nh", 0))
                pct_hispanic = safe_pct(c.get("pct_hispanic", 0))
                pct_other    = safe_pct(c.get("pct_other", 0))

        # Incumbency from results file
        r_inc = bool(res_row.get("r_incumbent", False))
        d_inc = bool(res_row.get("d_incumbent", False))
        dem_incumbent = int(d_inc)
        rep_incumbent = int(r_inc)
        open_seat = int(not r_inc and not d_inc)

        # Finance: challenger viability flag
        chal_viab = 0
        dem_fundraising_share = 0.5
        total_raised = 0.0
        if config.get("use_finance") and finance_df is not None:
            fin_match = finance_df[
                (finance_df["chamber"].str.lower() == chamber) &
                (finance_df["district"] == district)
            ]
            if not fin_match.empty:
                fr = fin_match.iloc[0]
                cv = fr.get("challenger_viability_flag", 0)
                chal_viab = int(cv) if pd.notna(cv) else 0
                dfs = fr.get("dem_fundraising_share")
                if pd.notna(dfs):
                    dem_fundraising_share = float(dfs)
                # Total raised for $10K minimum threshold
                dem_r = fr.get("dem_raised", 0)
                rep_r = fr.get("rep_raised", 0)
                total_raised = (float(dem_r) if pd.notna(dem_r) else 0) + \
                               (float(rep_r) if pd.notna(rep_r) else 0)

        # WAR: use pre-cycle WAR only (avoid data leakage)
        inc_name = ""
        if d_inc:
            inc_name = str(res_row.get("d_candidate", "")).strip()
        elif r_inc:
            inc_name = str(res_row.get("r_candidate", "")).strip()

        rows.append({
            "district": district,
            "chamber": chamber.title(),
            "chamber_lower": chamber.lower(),
            "dem_pres_2p_baseline": dem_pres_2p,
            "pct_white_nh": pct_white_nh,
            "pct_black_nh": pct_black_nh,
            "pct_hispanic": pct_hispanic,
            "pct_other": pct_other,
            "dem_incumbent": dem_incumbent,
            "rep_incumbent": rep_incumbent,
            "open_seat": open_seat,
            "incumbent": inc_name,
            "incumbent_party": ("D" if d_inc else "R" if r_inc else ""),
            "challenger_viability_flag": chal_viab,
            "dem_fundraising_share": dem_fundraising_share,
            "total_raised": total_raised,
            "actual_dem_2p_share": res_row.get("dem_2p_share"),
            "actual_winner_party": res_row.get("winner_party", ""),
            "contested": bool(res_row.get("contested", False)),
        })

    df = pd.DataFrame(rows)
    print(f"  Built {len(df)} {chamber} district rows")
    if verbose:
        pres_ok = df["dem_pres_2p_baseline"].notna().sum()
        cvap_ok = df["pct_white_nh"].notna().sum()
        print(f"    Presidential baseline: {pres_ok}/{len(df)} districts")
        print(f"    CVAP data: {cvap_ok}/{len(df)} districts")

    return df


# ---------------------------------------------------------------------------
# Compute model predictions (mirrors model.py logic)
# ---------------------------------------------------------------------------

def compute_demo_baseline(df: pd.DataFrame, race_generic: dict[str, float],
                           nat_weights: dict[str, float]) -> pd.Series:
    """Σ(CVAP_pct × race_D_share) for each district."""
    national_avg = sum(nat_weights[r] * race_generic[r] for r in nat_weights)

    results = []
    for _, row in df.iterrows():
        w_nh  = row["pct_white_nh"]  or 0.0
        b_nh  = row["pct_black_nh"]  or 0.0
        hisp  = row["pct_hispanic"]  or 0.0
        other = row["pct_other"]     or 0.0

        # Already in 0-1 range from build_backtest_df
        total = w_nh + b_nh + hisp + other
        if total > 0:
            w_nh /= total; b_nh /= total; hisp /= total; other /= total
            demo_d = (w_nh   * race_generic["white_nh"] +
                      b_nh   * race_generic["black_nh"] +
                      hisp   * race_generic["hispanic"] +
                      other  * race_generic["other"])
        else:
            demo_d = national_avg  # fallback: no CVAP data
        results.append(demo_d)

    return pd.Series(results, index=df.index)


def build_linear_predictions(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Compute linear (deterministic) model prediction for each district.
    Mirrors model.py but uses historical config instead of model_config.py.
    """
    race_generic = config["race_generic_ballot_d_share"]
    nat_weights  = config["national_demo_weights"]
    coefs        = config["regression_coefficients"]
    env_dial     = config["env_dial"]

    national_avg = sum(nat_weights[r] * race_generic[r] for r in nat_weights)

    demo_baseline  = compute_demo_baseline(df, race_generic, nat_weights)
    demo_deviation = demo_baseline - national_avg

    pres_baseline = pd.to_numeric(df["dem_pres_2p_baseline"], errors="coerce")
    chamber_senate = (df["chamber_lower"] == "senate").astype(int)

    # Finance term
    chal_flag = pd.to_numeric(df.get("challenger_viability_flag", 0), errors="coerce").fillna(0)
    dem_fs = pd.to_numeric(df.get("dem_fundraising_share", 0.5), errors="coerce")
    # Nullify dem_fundraising_share below $10K total raised (noise, not signal)
    total_raised = pd.to_numeric(df.get("total_raised", 0), errors="coerce").fillna(0)
    low_total = total_raised < 10_000
    n_nullified = (low_total & dem_fs.notna()).sum()
    dem_fs[low_total] = np.nan
    dem_fs = dem_fs.fillna(0.5)

    finance_term = (
        coefs["challenger_viability_flag"] * chal_flag
        + coefs.get("dem_fundraising_share", 0) * dem_fs
    )

    predicted = (
        coefs["intercept"]
        + coefs["dem_pres_2p_baseline"] * pres_baseline.fillna(national_avg)
        + demo_deviation
        + coefs["national_env"] * env_dial
        + coefs["dem_incumbent"] * df["dem_incumbent"]
        + coefs["rep_incumbent"] * df["rep_incumbent"]
        + coefs["chamber_senate"] * chamber_senate
    )

    # WAR persistence (mirrors model.py dual-track logic)
    war_persistence = config.get("war_persistence_coef", 0.46)
    war_max_year = config.get("war_max_year")
    has_war = pd.Series(False, index=df.index)
    war_adj = pd.Series(0.0, index=df.index)

    if config.get("use_war") and war_max_year:
        import re as _re
        race_war_path = DATA_PROC / "race_war.csv"
        if race_war_path.exists():
            rw = pd.read_csv(race_war_path)
            rw = rw[rw["year"] <= war_max_year]
            # Compute per-candidate avg WAR from pre-cycle data
            avg_war = (
                rw.groupby(["candidate_norm", "party"])["war_race"]
                .mean()
                .to_dict()
            )
            def _norm(name):
                if not name or (isinstance(name, float) and np.isnan(name)):
                    return ""
                s = str(name).lower().strip()
                s = _re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s)
                s = _re.sub(r"[^\w\s]", "", s)
                s = _re.sub(r"\s+", " ", s).strip()
                return s

            # Build fuzzy lookup index for subset/initial matching
            names_by_party: dict[str, list] = {"D": [], "R": []}
            for (norm_name, party), w in avg_war.items():
                names_by_party.setdefault(party, []).append(
                    (norm_name, set(norm_name.split()), w)
                )

            def _fuzzy_match(query_norm: str, party: str) -> float | None:
                query_tokens = set(query_norm.split())
                if not query_tokens:
                    return None
                query_first_char = query_norm.split()[0][0] if query_norm.split() else ""
                query_last = query_norm.split()[-1] if query_norm.split() else ""
                for cand_norm, cand_tokens, w in names_by_party.get(party, []):
                    # Subset: all tokens of shorter name in longer
                    shorter, longer = (query_tokens, cand_tokens) if len(query_tokens) <= len(cand_tokens) else (cand_tokens, query_tokens)
                    if shorter <= longer and len(shorter) >= 2:
                        return w
                    # Last name + first initial
                    cand_first_char = cand_norm.split()[0][0] if cand_norm.split() else ""
                    cand_last = cand_norm.split()[-1] if cand_norm.split() else ""
                    if (query_last == cand_last and query_last
                            and query_first_char == cand_first_char and query_first_char):
                        return w
                return None

            n_exact = 0
            n_fuzzy = 0
            for idx, row in df.iterrows():
                inc_name = str(row.get("incumbent", "")).strip()
                inc_party = str(row.get("incumbent_party", "")).strip().upper()
                is_open = bool(row.get("open_seat", 0))
                if not inc_name or inc_party not in ("D", "R") or is_open:
                    continue
                inc_norm = _norm(inc_name)
                w = avg_war.get((inc_norm, inc_party))
                match_type = "exact" if w is not None else None
                if w is None:
                    w = _fuzzy_match(inc_norm, inc_party)
                    if w is not None:
                        match_type = "fuzzy"
                if w is not None:
                    party_sign = 1 if inc_party == "D" else -1
                    war_adj[idx] = party_sign * war_persistence * w / 100.0
                    has_war[idx] = True
                    if match_type == "exact":
                        n_exact += 1
                    else:
                        n_fuzzy += 1

            print(f"  WAR: {n_exact + n_fuzzy} incumbents matched "
                  f"({n_exact} exact, {n_fuzzy} fuzzy, using races through {war_max_year})")

    # Dual-track: WAR districts use WAR, others use finance
    predicted += np.where(has_war, war_adj, finance_term)

    return predicted


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_calibration_metrics(df_all: pd.DataFrame,
                                  linear_preds: pd.Series) -> dict:
    """
    Compute calibration and accuracy metrics for contested races.

    The linear prediction is converted to a win probability using the CDF
    of the normal distribution (the same approach used in the Monte Carlo
    for the deterministic win probability before simulation noise).
    """
    from scipy.stats import norm

    # Work only on contested races with actual results
    mask = (
        df_all["contested"] == True
    ) & df_all["actual_dem_2p_share"].notna()

    if mask.sum() == 0:
        print("  WARNING: No contested races with results found")
        return {}

    contested = df_all[mask].copy()
    preds = linear_preds[mask].copy()

    # Predicted D 2p share (linear model output)
    contested["predicted_dem_share"] = preds.values

    # Win probability using normal CDF (P(dem > 0.5))
    sigma = 0.0785  # total model sigma (correlated + idiosyncratic)
    contested["predicted_win_prob"] = norm.cdf(
        contested["predicted_dem_share"], loc=0.5, scale=sigma
    )

    # Actual outcome
    contested["actual_dem_win"] = (contested["actual_winner_party"] == "D").astype(int)
    contested["actual_dem_share"] = pd.to_numeric(
        contested["actual_dem_2p_share"], errors="coerce"
    )

    # Brier score: mean squared error of win probability
    brier = np.mean((contested["predicted_win_prob"] - contested["actual_dem_win"]) ** 2)

    # Accuracy: did we predict the winner correctly?
    predicted_winner = (contested["predicted_win_prob"] >= 0.5).astype(int)
    accuracy = np.mean(predicted_winner == contested["actual_dem_win"])

    # Mean absolute error on dem share (for contested races)
    mae_share = np.mean(
        np.abs(contested["predicted_dem_share"] - contested["actual_dem_share"].fillna(
            contested["actual_dem_win"] * 0.55 + (1 - contested["actual_dem_win"]) * 0.35
        ))
    )

    # Calibration buckets: group by predicted win prob, compare to actual win rate
    calib = []
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        bucket = contested[
            (contested["predicted_win_prob"] >= lo) &
            (contested["predicted_win_prob"] < hi)
        ]
        if len(bucket) > 0:
            calib.append({
                "predicted_range": f"{lo*100:.0f}-{hi*100:.0f}%",
                "n_districts": len(bucket),
                "predicted_win_rate": bucket["predicted_win_prob"].mean(),
                "actual_win_rate": bucket["actual_dem_win"].mean(),
                "calibration_error": bucket["predicted_win_prob"].mean() - bucket["actual_dem_win"].mean(),
            })

    return {
        "n_contested": len(contested),
        "brier_score": brier,
        "accuracy": accuracy,
        "mae_dem_share": mae_share,
        "calibration_table": pd.DataFrame(calib),
        "detail_df": contested,
    }


def compute_seat_accuracy(df_all: pd.DataFrame, linear_preds: pd.Series,
                           config: dict, run_mc: bool = True) -> dict:
    """
    Compare predicted seat totals to actual outcomes.
    Runs Monte Carlo to get expected seats + control probability.
    """
    from scipy.stats import norm

    house = df_all[df_all["chamber_lower"] == "house"].copy()
    senate = df_all[df_all["chamber_lower"] == "senate"].copy()

    # Actual seat counts (only from on-ballot races)
    actual_house_d = (house["actual_winner_party"] == "D").sum()
    actual_house_r = (house["actual_winner_party"] == "R").sum()
    actual_senate_d = (senate["actual_winner_party"] == "D").sum()
    actual_senate_r = (senate["actual_winner_party"] == "R").sum()

    if run_mc:
        # Quick Monte Carlo for seat totals
        linear = linear_preds.values
        n_districts = len(df_all)
        is_house = (df_all["chamber_lower"] == "house").values

        national_errors = RNG.normal(0, SIGMA_NATIONAL, size=N_SIMULATIONS)
        idio_errors     = RNG.normal(0, SIGMA_IDIO, size=(n_districts, N_SIMULATIONS))
        predicted_matrix = (
            linear[:, np.newaxis] + national_errors[np.newaxis, :] + idio_errors
        )
        wins = (predicted_matrix > 0.5)
        house_seats = wins[is_house, :].sum(axis=0)
        senate_seats = wins[~is_house, :].sum(axis=0)

        expected_house_d = house_seats.mean()
        expected_senate_d = senate_seats.mean()
    else:
        # Deterministic: just use linear prediction
        sigma = 0.0785
        win_probs = norm.cdf(linear_preds.values, loc=0.5, scale=sigma)
        is_house_mask = df_all["chamber_lower"].values == "house"
        expected_house_d = win_probs[is_house_mask].sum()
        expected_senate_d = win_probs[~is_house_mask].sum()

    return {
        "actual_house_d": int(actual_house_d),
        "actual_house_r": int(actual_house_r),
        "actual_senate_d": int(actual_senate_d),
        "actual_senate_r": int(actual_senate_r),
        "expected_house_d": expected_house_d,
        "expected_senate_d": expected_senate_d,
        "house_error": expected_house_d - actual_house_d,
        "senate_error": expected_senate_d - actual_senate_d,
    }


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_calibration_results(metrics: dict, label: str):
    print(f"\n{'─'*60}")
    print(f"Calibration Metrics — {label}")
    print(f"{'─'*60}")
    n = metrics.get("n_contested", 0)
    if n == 0:
        print("  No contested races found.")
        return

    print(f"  Contested races evaluated: {n}")
    print(f"  Brier score:               {metrics['brier_score']:.4f}  "
          f"(lower=better; 0.25=random, 0=perfect)")
    print(f"  Winner accuracy:           {metrics['accuracy']*100:.1f}%")
    print(f"  Mean abs error (D share):  {metrics['mae_dem_share']:.3f} "
          f"({metrics['mae_dem_share']*100:.1f}pp)")

    calib_df = metrics.get("calibration_table")
    if calib_df is not None and not calib_df.empty:
        print(f"\n  Calibration table (predicted vs actual D win rate):")
        print(f"  {'Predicted':>12}  {'N':>5}  {'Pred%':>7}  {'Actual%':>8}  {'Error':>7}")
        for _, row in calib_df.iterrows():
            err = row["calibration_error"]
            flag = "  !" if abs(err) > 0.10 else ""
            print(f"  {row['predicted_range']:>12}  {int(row['n_districts']):>5}  "
                  f"{row['predicted_win_rate']*100:>6.1f}%  "
                  f"{row['actual_win_rate']*100:>7.1f}%  "
                  f"{err*100:>+6.1f}pp{flag}")


def print_seat_results(seat_metrics: dict, label: str):
    print(f"\n{'─'*60}")
    print(f"Seat Total Accuracy — {label}")
    print(f"{'─'*60}")
    print(f"  House:  Predicted D = {seat_metrics['expected_house_d']:.1f}  |  "
          f"Actual D = {seat_metrics['actual_house_d']}  |  "
          f"Error = {seat_metrics['house_error']:+.1f}")
    print(f"  Senate: Predicted D = {seat_metrics['expected_senate_d']:.1f}  |  "
          f"Actual D = {seat_metrics['actual_senate_d']}  |  "
          f"Error = {seat_metrics['senate_error']:+.1f}")


def print_worst_misses(metrics: dict, n: int = 10):
    """Print districts where the model was most wrong."""
    detail = metrics.get("detail_df")
    if detail is None or detail.empty:
        return

    detail = detail.copy()
    detail["error"] = (
        detail["predicted_win_prob"] - detail["actual_dem_win"]
    ).abs()
    worst = detail.nlargest(n, "error")

    print(f"\n  Largest prediction errors (predicted vs actual):")
    print(f"  {'District':>10}  {'Pred%':>7}  {'Actual Win':>11}  {'Error':>7}")
    for _, row in worst.iterrows():
        chamber_abbr = row["chamber"][0] + "D"
        actual_str = "D won" if row["actual_dem_win"] else "R won"
        print(f"  {chamber_abbr}{int(row['district']):<7d}  "
              f"{row['predicted_win_prob']*100:>6.1f}%  "
              f"{actual_str:>11}  "
              f"{row['error']*100:>+6.1f}pp")


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(cycle_year: int, run_mc: bool = True, verbose: bool = False,
                 save: bool = True):
    """
    Run full backtest for one cycle year (2022 or 2018).
    """
    config = BACKTEST_CONFIGS.get(cycle_year)
    if config is None:
        print(f"ERROR: No backtest config for {cycle_year}. Available: {list(BACKTEST_CONFIGS.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"BACKTEST: {config['label']}")
    print(f"{'='*60}")
    print(f"  Presidential baseline: {config['pres_year']}")
    print(f"  District map year:     {config['map_year']}")
    print(f"  Actual results year:   {config['results_year']}")
    print(f"\n  {config['notes']}")
    print(f"\n  Source: {config['source']}")
    if "ESTIMATED" in config.get("source", ""):
        print(f"\n  *** WARNING: Generic ballot values are ESTIMATED approximations.")
        print(f"  *** Update backtest_config.py with actual Civiqs historical data.")
        print(f"  *** Results below reflect the model structure; calibration may")
        print(f"  *** improve substantially with accurate historical polling values.")

    # Build combined DataFrame for both chambers
    print(f"\n{'─'*40}")
    print(f"Loading data...")
    dfs = []
    linear_preds_all = []

    for chamber in ("house", "senate"):
        print(f"\n  {chamber.title()}:")
        df_chamber = build_backtest_df(config, chamber, verbose=verbose)
        if df_chamber is None or df_chamber.empty:
            print(f"  Skipping {chamber} (no data)")
            continue

        # Check if we have enough presidential baseline data to proceed
        pres_ok = df_chamber["dem_pres_2p_baseline"].notna().sum()
        if pres_ok < len(df_chamber) * 0.5:
            print(f"  WARNING: Only {pres_ok}/{len(df_chamber)} districts have "
                  f"presidential baseline data.")
            if pres_ok == 0:
                print(f"  Skipping {chamber}: no presidential baseline available.")
                print(f"  Run: python src/collect_historical_presidential.py "
                      f"--pres-year {config['pres_year']}")
                continue

        linear_chamber = build_linear_predictions(df_chamber, config)
        dfs.append(df_chamber)
        linear_preds_all.append(linear_chamber)

    if not dfs:
        print(f"\nNo data available for {cycle_year} backtest.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    linear_all = pd.concat(linear_preds_all, ignore_index=True)

    print(f"\n{'─'*40}")
    print(f"Running model on {len(df_all)} districts...")

    # Seat accuracy
    seat_metrics = compute_seat_accuracy(df_all, linear_all, config, run_mc=run_mc)
    print_seat_results(seat_metrics, config["label"])

    # Calibration metrics (contested races only)
    calib_metrics = compute_calibration_metrics(df_all, linear_all)
    print_calibration_results(calib_metrics, config["label"])
    print_worst_misses(calib_metrics)

    # Summary of prediction coverage
    n_total = len(df_all)
    n_contested = calib_metrics.get("n_contested", 0)
    n_no_pres = df_all["dem_pres_2p_baseline"].isna().sum()
    n_no_cvap = df_all["pct_white_nh"].isna().sum()
    print(f"\n  Coverage:")
    print(f"    Total districts on ballot: {n_total}")
    print(f"    Contested (evaluated):     {n_contested}")
    print(f"    Missing presidential data: {n_no_pres}")
    print(f"    Missing CVAP data:         {n_no_cvap}")

    # Save output
    if save:
        out_detail = OUTPUT / f"backtest_{cycle_year}_detail.csv"
        if calib_metrics.get("detail_df") is not None:
            calib_metrics["detail_df"].to_csv(out_detail, index=False)
            print(f"\n  Saved detail: {out_detail.name}")

        # Save all districts (not just contested)
        df_all["linear_prediction"] = linear_all.values
        from scipy.stats import norm
        df_all["predicted_win_prob"] = norm.cdf(
            df_all["linear_prediction"], loc=0.5, scale=0.0785
        )
        out_all = OUTPUT / f"backtest_{cycle_year}_all_districts.csv"
        df_all.to_csv(out_all, index=False)
        print(f"  Saved all districts: {out_all.name}")

    return {
        "config": config,
        "seat_metrics": seat_metrics,
        "calib_metrics": calib_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 2 TX legislative model against historical outcomes"
    )
    parser.add_argument("--year", type=int, choices=list(BACKTEST_CONFIGS.keys()),
                        help="Cycle year to backtest (default: all)")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo simulation (fast mode)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print additional data coverage details")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output files")
    args = parser.parse_args()

    years = [args.year] if args.year else list(BACKTEST_CONFIGS.keys())

    print("=" * 60)
    print("TX LEGISLATIVE MODEL — HISTORICAL BACKTEST VALIDATION")
    print("=" * 60)
    print(f"Cycles to validate: {years}")
    print("\nNOTE: This validates the model structure as of April of each cycle year.")
    print("Calibration quality depends on accuracy of historical generic ballot values")
    print("in backtest_config.py. Update with actual Civiqs historical data for")
    print("accurate calibration assessment.")

    all_results = {}
    for year in years:
        result = run_backtest(
            year,
            run_mc=not args.no_mc,
            verbose=args.verbose,
            save=not args.no_save,
        )
        if result:
            all_results[year] = result

    # Cross-cycle summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-CYCLE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Year':>6}  {'Brier':>7}  {'Accuracy':>9}  {'H Error':>8}  {'S Error':>8}")
        for year, r in all_results.items():
            cm = r["calib_metrics"]
            sm = r["seat_metrics"]
            brier = cm.get("brier_score", float("nan"))
            acc = cm.get("accuracy", float("nan"))
            h_err = sm.get("house_error", float("nan"))
            s_err = sm.get("senate_error", float("nan"))
            print(f"{year:>6}  {brier:>7.4f}  {acc*100:>8.1f}%  "
                  f"{h_err:>+7.1f}  {s_err:>+7.1f}")


if __name__ == "__main__":
    main()
