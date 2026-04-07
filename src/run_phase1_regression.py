"""
run_phase1_regression.py

Phase 1 regression analysis for the Texas legislative election model.
Loads data/processed/phase1_dataset.csv and runs:

  Step 1 — Base OLS (up to 3 completeness levels):
    restricted:  national_env + incumbency + chamber_senate
    with_gov:    adds baseline_partisanship
    full:        adds dem_fundraising_share + log_challenger_fundraising +
                 challenger_viability_flag

  Step 2 — Rolling window comparison (2002-2010 vs 2014-2022)
    Fit each model specification on the early and late windows separately.
    Flag coefficients where the difference exceeds 2 combined standard errors.

  Step 3 — Year interaction test
    Add dem_fundraising_share × year_numeric interaction.
    Report F-test for joint significance of interactions.

  Step 4 — Recursive estimation
    Add cycles one at a time; track how coefficients evolve.
    Flags if adding the earliest cycles substantially shifts key coefficients.

  Step 5 — Leave-one-cycle-out cross-validation
    For each cycle year, fit on remaining 5 and predict on held-out year.
    Report MAE, winner prediction accuracy, and by-year breakdown.

  Step 6 — Summary output
    Write 4 files to output/:
      phase1_regression_summary.txt   — human-readable narrative
      phase1_coefficients.csv         — clean coefficient table
      phase1_temporal_stability.csv   — rolling window comparison
      phase1_cross_validation.csv     — LOO results by year

Usage:
  python src/run_phase1_regression.py
  python src/run_phase1_regression.py --min-completeness with_gov
"""

import argparse
import csv
import math
import warnings
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("ERROR: statsmodels not installed. Run: pip install statsmodels")
    raise SystemExit(1)

try:
    from tabulate import tabulate
    TABULATE_OK = True
except ImportError:
    TABULATE_OK = False

ROOT = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]

# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    path = DATA_PROC / "phase1_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Run: python src/build_phase1_dataset.py"
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def filter_regression_sample(df: pd.DataFrame, completeness_level: str) -> pd.DataFrame:
    """
    Filter to rows usable for regression at the given completeness level.
    Always excludes: not on ballot, uncontested, missing dem_2p_share.
    """
    mask = (
        (df["on_ballot"] == 1) &
        (df["uncontested"] == 0) &
        (df["dem_2p_share"].notna())
    )
    df = df[mask].copy()

    if completeness_level == "with_gov":
        df = df[df["baseline_partisanship"].notna()]
    elif completeness_level == "with_pres":
        df = df[df["dem_pres_2p_baseline"].notna()]
    elif completeness_level == "full":
        df = df[
            df["dem_pres_2p_baseline"].notna() &
            df["dem_fundraising_share"].notna() &
            df["log_challenger_fundraising"].notna()
        ]
    elif completeness_level == "with_ie":
        df = df[
            df["dem_pres_2p_baseline"].notna() &
            df["ie_dem_share"].notna() &
            df["ie_log_total"].notna()
        ]
    elif completeness_level == "full_ie":
        df = df[
            df["dem_pres_2p_baseline"].notna() &
            df["dem_fundraising_share"].notna() &
            df["log_challenger_fundraising"].notna() &
            df["ie_dem_share"].notna() &
            df["ie_log_total"].notna()
        ]

    df["year_numeric"] = df["year"] - 2002
    return df.reset_index(drop=True)


def choose_completeness_level(df: pd.DataFrame, min_level: str) -> list[str]:
    """
    Determine which completeness levels to run, based on data availability.
    Always runs 'restricted'. Runs 'with_gov' if >= 60% of contested rows
    have baseline_partisanship. Runs 'full' if >= 50% also have finance
    (threshold lowered from 60% because TEC data covers 2018+2022 well but
    earlier cycles have sparse electronic filing records).
    """
    levels = ["restricted"]

    contested = df[(df["on_ballot"] == 1) & (df["uncontested"] == 0)]
    if len(contested) == 0:
        return levels

    frac_gov = contested["baseline_partisanship"].notna().mean()
    frac_pres = contested["dem_pres_2p_baseline"].notna().mean() \
        if "dem_pres_2p_baseline" in contested.columns else 0.0
    frac_fin = (
        contested["dem_fundraising_share"].notna() &
        contested["log_challenger_fundraising"].notna()
    ).mean()
    frac_ie = (
        contested["ie_dem_share"].notna() &
        contested["ie_log_total"].notna()
    ).mean() if "ie_dem_share" in contested.columns else 0.0

    if frac_gov >= 0.6 or min_level in ("with_gov", "full"):
        levels.append("with_gov")
    if frac_pres >= 0.6 or min_level in ("with_pres", "full", "with_ie", "full_ie"):
        levels.append("with_pres")
    if (frac_pres >= 0.6 and frac_fin >= 0.5) or min_level in ("full", "full_ie"):
        levels.append("full")
    if frac_ie > 0.05 or min_level in ("with_ie", "full_ie"):
        levels.append("with_ie")
    if (frac_ie > 0.05 and frac_fin >= 0.5) or min_level == "full_ie":
        levels.append("full_ie")

    print(f"  IE data coverage: {frac_ie*100:.1f}% of contested on-ballot races")
    return levels


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------

BASE_VARS = "dem_incumbent + rep_incumbent + chamber_senate + national_env"
IE_VARS   = "ie_dem_share + ie_log_total"

FORMULAS = {
    "restricted": f"dem_2p_share ~ {BASE_VARS}",
    "with_gov":   f"dem_2p_share ~ baseline_partisanship + {BASE_VARS}",
    "with_pres":  f"dem_2p_share ~ dem_pres_2p_baseline + {BASE_VARS}",
    "full":       (f"dem_2p_share ~ dem_pres_2p_baseline + {BASE_VARS}"
                   f" + dem_fundraising_share + log_challenger_fundraising"
                   f" + challenger_viability_flag"),
    # IE models: add independent expenditure variables
    "with_ie":    (f"dem_2p_share ~ dem_pres_2p_baseline + {BASE_VARS}"
                   f" + {IE_VARS}"),
    "full_ie":    (f"dem_2p_share ~ dem_pres_2p_baseline + {BASE_VARS}"
                   f" + dem_fundraising_share + log_challenger_fundraising"
                   f" + challenger_viability_flag + {IE_VARS}"),
}

COEF_LABELS = {
    "Intercept": "Intercept",
    "baseline_partisanship": "Baseline partisanship (gov. Dem 2p share)",
    "dem_pres_2p_baseline": "Presidential baseline (2024 Dem 2p share)",
    "national_env": "National environment (D-R generic ballot)",
    "dem_incumbent": "Dem incumbent (vs. open seat)",
    "rep_incumbent": "Rep incumbent (vs. open seat)",
    "chamber_senate": "Senate chamber (vs. House)",
    "dem_fundraising_share": "Dem fundraising share (0-1)",
    "log_challenger_fundraising": "Log(challenger fundraising + 1)",
    "challenger_viability_flag": "Challenger viability flag (>threshold)",
    "ie_dem_share": "IE Dem share (D-favoring IEs / total IEs, 0-1)",
    "ie_log_total": "Log(total IE dollars + 1)",
    "ie_flag": "IE significance flag (total IEs >= $25k)",
}


def fit_model(df: pd.DataFrame, level: str, label: str = "") -> dict:
    """
    Fit OLS at a given completeness level. Returns a result dict.
    """
    sample = filter_regression_sample(df, level)
    if len(sample) < 30:
        return {"level": level, "label": label, "n": len(sample),
                "error": f"Too few observations ({len(sample)}) for level={level}"}

    formula = FORMULAS[level]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = smf.ols(formula, data=sample).fit(
            cov_type="HC3"  # heteroskedasticity-robust SEs
        )

    coefs = []
    for var in result.params.index:
        coefs.append({
            "variable": var,
            "label": COEF_LABELS.get(var, var),
            "coef": result.params[var],
            "se": result.bse[var],
            "t": result.tvalues[var],
            "p": result.pvalues[var],
            "ci_lo": result.conf_int().loc[var, 0],
            "ci_hi": result.conf_int().loc[var, 1],
        })

    return {
        "level": level,
        "label": label,
        "formula": formula,
        "n": len(sample),
        "r2": result.rsquared,
        "r2_adj": result.rsquared_adj,
        "residual_se": math.sqrt(result.mse_resid),
        "aic": result.aic,
        "coefs": coefs,
        "result_obj": result,
        "sample": sample,
    }


# ---------------------------------------------------------------------------
# Step 1: Base models
# ---------------------------------------------------------------------------

def step1_base_models(df: pd.DataFrame, levels: list[str]) -> list[dict]:
    print(f"\n{'='*60}")
    print("STEP 1: Base OLS models")
    print(f"{'='*60}")
    fits = []
    for level in levels:
        fit = fit_model(df, level)
        fits.append(fit)
        if "error" in fit:
            print(f"\n  [{level}] SKIPPED: {fit['error']}")
            continue
        print(f"\n  [{level}] n={fit['n']}, R²={fit['r2']:.4f}, "
              f"Adj-R²={fit['r2_adj']:.4f}, Residual SE={fit['residual_se']:.4f}")
        for c in fit["coefs"]:
            stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
            print(f"    {c['variable']:35s} {c['coef']:+.4f}  (SE={c['se']:.4f}, p={c['p']:.3f}){stars}")
    return fits


# ---------------------------------------------------------------------------
# Step 2: Rolling window temporal stability
# ---------------------------------------------------------------------------

def step2_rolling_window(df: pd.DataFrame, levels: list[str]) -> list[dict]:
    print(f"\n{'='*60}")
    print("STEP 2: Rolling window comparison (2002-2010 vs 2014-2022)")
    print(f"{'='*60}")

    early = df[df["year"] <= 2010]
    late = df[df["year"] >= 2014]

    comparisons = []
    level = levels[-1]  # use the most complete available level

    fit_early = fit_model(early, level, "2002-2010")
    fit_late = fit_model(late, level, "2014-2022")

    if "error" in fit_early or "error" in fit_late:
        print(f"  Cannot run rolling window: insufficient data in one window.")
        return []

    print(f"\n  Model level: {level}")
    print(f"  Early (2002-2010):  n={fit_early['n']}, R²={fit_early['r2']:.4f}")
    print(f"  Late  (2014-2022):  n={fit_late['n']},  R²={fit_late['r2']:.4f}")

    coef_early = {c["variable"]: c for c in fit_early["coefs"]}
    coef_late = {c["variable"]: c for c in fit_late["coefs"]}
    all_vars = list(coef_early.keys())

    print(f"\n  {'Variable':35s} {'Early':>10} {'Late':>10} {'Diff':>10}  {'Stable?':>8}")
    print(f"  {'-'*80}")
    for var in all_vars:
        if var not in coef_early or var not in coef_late:
            continue
        ce = coef_early[var]
        cl = coef_late[var]
        diff = cl["coef"] - ce["coef"]
        combined_se = math.sqrt(ce["se"] ** 2 + cl["se"] ** 2)
        stable = abs(diff) <= 2 * combined_se
        flag = "STABLE" if stable else "UNSTABLE"
        print(f"  {var:35s} {ce['coef']:+.4f}     {cl['coef']:+.4f}     {diff:+.4f}   {flag}")
        comparisons.append({
            "variable": var,
            "coef_early": ce["coef"],
            "se_early": ce["se"],
            "coef_late": cl["coef"],
            "se_late": cl["se"],
            "diff": diff,
            "combined_se": combined_se,
            "temporal_stability": flag,
        })

    unstable = [c for c in comparisons if c["temporal_stability"] == "UNSTABLE"]
    if unstable:
        print(f"\n  Temporally UNSTABLE coefficients ({len(unstable)}): "
              f"{[c['variable'] for c in unstable]}")
    else:
        print(f"\n  All coefficients are temporally stable.")

    return comparisons


# ---------------------------------------------------------------------------
# Step 3: Year interaction test
# ---------------------------------------------------------------------------

def step3_interaction_test(df: pd.DataFrame, levels: list[str]) -> dict:
    print(f"\n{'='*60}")
    print("STEP 3: Year interaction test (finance × time trend)")
    print(f"{'='*60}")

    level = "full" if "full" in levels else "with_gov" if "with_gov" in levels else "restricted"
    sample = filter_regression_sample(df, level)

    if "dem_fundraising_share" not in sample.columns or sample["dem_fundraising_share"].isna().all():
        print("  SKIPPED: No finance data available for interaction test.")
        return {}

    # Augmented model with fundraising × year interaction
    formula_base = FORMULAS[level]
    formula_inter = (
        formula_base
        + " + dem_fundraising_share:year_numeric"
        + " + challenger_viability_flag:year_numeric"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_base = smf.ols(formula_base, data=sample).fit()
        result_inter = smf.ols(formula_inter, data=sample).fit()

    # F-test for joint significance of interactions
    # (compare restricted vs unrestricted model)
    from statsmodels.stats.anova import anova_lm
    try:
        ftest = anova_lm(result_base, result_inter)
        f_stat = ftest.iloc[1]["F"]
        p_val = ftest.iloc[1]["Pr(>F)"]
    except Exception:
        f_stat = p_val = None

    print(f"\n  Base model R²: {result_base.rsquared:.4f}")
    print(f"  Interaction model R²: {result_inter.rsquared:.4f}")

    if f_stat is not None:
        print(f"  F-test for interactions: F={f_stat:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  FINDING: Fundraising × time interactions are statistically significant.")
            print("           Fundraising coefficients are changing over time.")
        else:
            print("  FINDING: No significant time trend in fundraising coefficients (p > 0.05).")
    else:
        print("  F-test unavailable — check model convergence.")

    # Print the interaction coefficients specifically
    inter_coefs = {k: v for k, v in result_inter.params.items() if "year_numeric" in k}
    if inter_coefs:
        print("\n  Interaction coefficients:")
        for var, coef in inter_coefs.items():
            se = result_inter.bse[var]
            p = result_inter.pvalues[var]
            print(f"    {var:50s} {coef:+.6f}  (SE={se:.6f}, p={p:.4f})")

    return {
        "f_stat": f_stat,
        "p_value": p_val,
        "r2_base": result_base.rsquared,
        "r2_interaction": result_inter.rsquared,
        "interaction_coefs": {
            var: {"coef": result_inter.params[var], "se": result_inter.bse[var],
                  "p": result_inter.pvalues[var]}
            for var in inter_coefs
        },
    }


# ---------------------------------------------------------------------------
# Step 4: Recursive estimation
# ---------------------------------------------------------------------------

def step4_recursive(df: pd.DataFrame, levels: list[str]) -> list[dict]:
    print(f"\n{'='*60}")
    print("STEP 4: Recursive estimation (adding one cycle at a time)")
    print(f"{'='*60}")

    level = levels[-1]
    recursive_results = []
    key_vars = ["national_env", "dem_incumbent", "rep_incumbent"]
    if level in ("with_gov", "full"):
        key_vars.insert(0, "baseline_partisanship")
    if level == "full":
        key_vars.append("dem_fundraising_share")

    print(f"\n  Model level: {level}")
    header = f"  {'Cycles':20s} {'n':>6}" + "".join(f" {v[:12]:>14}" for v in key_vars)
    print(header)
    print(f"  {'-'*80}")

    for i, cutoff_year in enumerate(YEARS):
        subset = df[df["year"] <= cutoff_year]
        fit = fit_model(subset, level, f"thru_{cutoff_year}")
        if "error" in fit:
            print(f"  Thru {cutoff_year}: SKIP ({fit['error']})")
            continue

        coef_map = {c["variable"]: c["coef"] for c in fit["coefs"]}
        cycles_label = f"thru {cutoff_year}"
        line = f"  {cycles_label:20s} {fit['n']:>6}"
        for var in key_vars:
            val = coef_map.get(var)
            line += f" {val:+14.4f}" if val is not None else f" {'N/A':>14}"
        print(line)

        recursive_results.append({
            "cutoff_year": cutoff_year,
            "n": fit["n"],
            "r2": fit["r2"],
            **{f"coef_{var}": coef_map.get(var) for var in key_vars},
        })

    # Stability assessment: compare coefficient from earliest vs latest
    if len(recursive_results) >= 2:
        first = recursive_results[0]
        last = recursive_results[-1]
        print(f"\n  Coefficient drift (thru {YEARS[0]} → thru {YEARS[-1]}):")
        for var in key_vars:
            c0 = first.get(f"coef_{var}")
            cf = last.get(f"coef_{var}")
            if c0 is not None and cf is not None:
                drift = cf - c0
                print(f"    {var:40s} {c0:+.4f} → {cf:+.4f}  (drift: {drift:+.4f})")

    return recursive_results


# ---------------------------------------------------------------------------
# Step 5: Leave-one-cycle-out cross-validation
# ---------------------------------------------------------------------------

def step5_cross_validation(df: pd.DataFrame, levels: list[str]) -> list[dict]:
    print(f"\n{'='*60}")
    print("STEP 5: Leave-one-cycle-out cross-validation")
    print(f"{'='*60}")

    level = levels[-1]
    cv_results = []

    print(f"\n  Model level: {level}")
    print(f"  {'Hold-out year':15s} {'n_train':>8} {'n_test':>8} {'MAE':>8} {'Winner%':>8} {'R²_oos':>8}")
    print(f"  {'-'*60}")

    all_errors = []
    all_correct = []

    for hold_year in YEARS:
        train = df[df["year"] != hold_year]
        test_raw = df[df["year"] == hold_year]

        train_sample = filter_regression_sample(train, level)
        test_sample = filter_regression_sample(test_raw, level)

        if len(train_sample) < 30 or len(test_sample) < 5:
            print(f"  {hold_year}:          SKIP (insufficient data)")
            continue

        formula = FORMULAS[level]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.ols(formula, data=train_sample).fit()

        preds = fit.predict(test_sample)
        actuals = test_sample["dem_2p_share"].values

        errors = np.abs(preds.values - actuals)
        mae = errors.mean()
        all_errors.extend(errors.tolist())

        # Winner prediction: dem wins if dem_2p_share > 0.5
        pred_winner_d = preds.values > 0.5
        actual_winner_d = actuals > 0.5
        correct = (pred_winner_d == actual_winner_d)
        winner_acc = correct.mean() * 100
        all_correct.extend(correct.tolist())

        # Out-of-sample R²
        ss_res = np.sum((actuals - preds.values) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else None

        print(f"  {hold_year:<15d} {len(train_sample):>8d} {len(test_sample):>8d} "
              f"{mae:>8.4f} {winner_acc:>7.1f}% {(r2_oos or 0):>8.4f}")

        cv_results.append({
            "hold_year": hold_year,
            "n_train": len(train_sample),
            "n_test": len(test_sample),
            "mae": mae,
            "winner_accuracy_pct": winner_acc,
            "r2_oos": r2_oos,
            "mean_actual": actuals.mean(),
            "mean_predicted": preds.mean(),
        })

    if all_errors:
        overall_mae = np.mean(all_errors)
        overall_acc = np.mean(all_correct) * 100
        print(f"\n  Overall LOO-CV:  MAE={overall_mae:.4f}, Winner accuracy={overall_acc:.1f}%")

    return cv_results


# ---------------------------------------------------------------------------
# Step 6: IE temporal analysis — does the IE effect vary by era?
# ---------------------------------------------------------------------------

def step6_ie_temporal(df: pd.DataFrame) -> dict:
    """
    Analyze how IE coefficients change across election cycles.

    Tests the specific user question: do IEs have a bigger impact in
    earlier (pre-Citizens United) or later (post-2010) election cycles?

    Runs the `with_ie` model on:
      1. Full sample (2002-2022)
      2. Early period (2002-2010): pre-Citizens United era
      3. Late period  (2014-2022): post-Citizens United era
      4. Year-by-year: leave-one-cycle-in models (single year fits, each a partial check)

    Also computes incremental R² from adding IE vars to the with_pres base.
    """
    print(f"\n{'='*60}")
    print("STEP 6: IE Temporal Analysis (early vs. late cycle IE effects)")
    print(f"{'='*60}")

    # Check if IE data is available
    if "ie_dem_share" not in df.columns or df["ie_dem_share"].isna().all():
        print("  SKIPPED: No IE data in dataset.")
        print("  Run: python src/collect_ies_historical.py")
        print("       python src/build_phase1_dataset.py")
        print("  Then re-run this regression.")
        return {}

    ie_sample_full = filter_regression_sample(df, "with_ie")
    if len(ie_sample_full) < 20:
        print(f"  SKIPPED: Only {len(ie_sample_full)} rows with IE data (need ≥20).")
        return {}

    print(f"\n  IE data coverage: {len(ie_sample_full)} district-years with IE observations")

    # Coverage by year
    print(f"\n  IE coverage by election year:")
    print(f"  {'Year':6s}  {'Districts w/ IE':18s}  {'Total $IE':>14s}  {'Frac all contested':>18s}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*14}  {'-'*18}")
    contested_base = filter_regression_sample(df, "with_pres")
    for year in YEARS:
        yr_total = contested_base[contested_base["year"] == year]
        yr_ie = ie_sample_full[ie_sample_full["year"] == year]
        total_ie = yr_ie["ie_total"].sum() if "ie_total" in yr_ie.columns else 0
        frac = len(yr_ie) / len(yr_total) if len(yr_total) > 0 else 0
        print(f"  {year:6d}  {len(yr_ie):18d}  ${total_ie:>13,.0f}  {frac*100:17.1f}%")

    # --- Full sample IE model vs base model ---
    print(f"\n  --- Model comparison: with_pres (no IE) vs. with_ie ---")
    fit_base = fit_model(ie_sample_full, "with_pres", "base_no_ie")
    fit_ie   = fit_model(ie_sample_full, "with_ie",   "full_ie_vars")
    if "error" in fit_base or "error" in fit_ie:
        print(f"  Could not fit models on IE sample.")
        return {}

    r2_base = fit_base["r2"]
    r2_ie   = fit_ie["r2"]
    delta_r2 = r2_ie - r2_base
    delta_se = fit_base["residual_se"] - fit_ie["residual_se"]

    print(f"  Base model R²:    {r2_base:.4f}  (Residual SE: {fit_base['residual_se']:.4f})")
    print(f"  + IE vars  R²:    {r2_ie:.4f}  (Residual SE: {fit_ie['residual_se']:.4f})")
    print(f"  Incremental R²:   {delta_r2:+.4f}")
    print(f"  SE reduction:     {delta_se:+.4f} pp  (positive = IE improves precision)")

    print(f"\n  IE coefficients (full sample, n={fit_ie['n']}):")
    ie_coefs_full = {c["variable"]: c for c in fit_ie["coefs"]}
    for var in ["ie_dem_share", "ie_log_total"]:
        c = ie_coefs_full.get(var)
        if c:
            stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
            print(f"    {var:35s} {c['coef']:+.4f}  (SE={c['se']:.4f}, p={c['p']:.3f}) {stars}")

    # --- Early vs. Late comparison ---
    print(f"\n  --- Early (2002-2010) vs. Late (2014-2022) IE coefficients ---")
    early = ie_sample_full[ie_sample_full["year"] <= 2010]
    late  = ie_sample_full[ie_sample_full["year"] >= 2014]

    period_results = {}
    for label, subset in [("Early 2002-2010", early), ("Late  2014-2022", late)]:
        if len(subset) < 15:
            print(f"  {label}: n={len(subset)} — too few for reliable fit (≥15 needed)")
            continue
        fit = fit_model(subset, "with_ie", label)
        if "error" in fit:
            print(f"  {label}: {fit['error']}")
            continue
        period_results[label] = fit
        coef_map = {c["variable"]: c for c in fit["coefs"]}
        print(f"\n  {label}:  n={fit['n']}, R²={fit['r2']:.4f}")
        for var in ["national_env", "ie_dem_share", "ie_log_total"]:
            c = coef_map.get(var)
            if c:
                stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
                print(f"    {var:35s} {c['coef']:+.4f}  (SE={c['se']:.4f}) {stars}")

    # Compare IE coefficients between periods
    ie_temporal_stability = {}
    if "Early 2002-2010" in period_results and "Late  2014-2022" in period_results:
        print(f"\n  --- IE coefficient temporal drift ---")
        coef_early = {c["variable"]: c for c in period_results["Early 2002-2010"]["coefs"]}
        coef_late  = {c["variable"]: c for c in period_results["Late  2014-2022"]["coefs"]}
        for var in ["ie_dem_share", "ie_log_total", "national_env"]:
            ce = coef_early.get(var)
            cl = coef_late.get(var)
            if ce and cl:
                diff = cl["coef"] - ce["coef"]
                combined_se = math.sqrt(ce["se"]**2 + cl["se"]**2)
                stable = abs(diff) <= 2 * combined_se
                flag = "STABLE" if stable else "UNSTABLE"
                print(f"  {var:35s}  Early: {ce['coef']:+.4f}  Late: {cl['coef']:+.4f}  "
                      f"Diff: {diff:+.4f}  [{flag}]")
                ie_temporal_stability[var] = {
                    "coef_early": ce["coef"], "se_early": ce["se"],
                    "coef_late":  cl["coef"], "se_late":  cl["se"],
                    "diff": diff, "temporal_stability": flag,
                }

    # --- Year-by-year incremental R² from IEs ---
    print(f"\n  --- Incremental R² from IEs by election year ---")
    print(f"  {'Year':6s}  {'n':>5}  {'Base R²':>8}  {'+ IE R²':>8}  {'Delta R²':>10}  "
          f"{'SE reduction':>13}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*13}")
    yr_ie_contributions = []
    for year in YEARS:
        yr_data = ie_sample_full[ie_sample_full["year"] == year]
        if len(yr_data) < 10:
            print(f"  {year:6d}  {len(yr_data):>5d}  (insufficient data for single-year fit)")
            continue
        fb = fit_model(yr_data, "with_pres", f"{year}_base")
        fi = fit_model(yr_data, "with_ie",   f"{year}_ie")
        if "error" in fb or "error" in fi:
            print(f"  {year:6d}  {len(yr_data):>5d}  (model fit error)")
            continue
        dr2 = fi["r2"] - fb["r2"]
        dse = fb["residual_se"] - fi["residual_se"]
        print(f"  {year:6d}  {len(yr_data):>5d}  {fb['r2']:>8.4f}  {fi['r2']:>8.4f}  "
              f"{dr2:>+10.4f}  {dse:>+12.4f}")
        yr_ie_contributions.append({
            "year": year, "n": len(yr_data),
            "r2_base": fb["r2"], "r2_ie": fi["r2"],
            "delta_r2": dr2, "se_reduction": dse,
        })

    # Summarize the early vs. late IE impact
    if yr_ie_contributions:
        early_yr = [r for r in yr_ie_contributions if r["year"] <= 2010]
        late_yr  = [r for r in yr_ie_contributions if r["year"] >= 2014]
        if early_yr:
            avg_early = np.mean([r["delta_r2"] for r in early_yr])
            print(f"\n  Average IE incremental R² — Early (2002-2010): {avg_early:+.4f}")
        if late_yr:
            avg_late = np.mean([r["delta_r2"] for r in late_yr])
            print(f"  Average IE incremental R² — Late  (2014-2022): {avg_late:+.4f}")
        if early_yr and late_yr:
            trend = avg_late - avg_early
            print(f"  Trend (late minus early):                       {trend:+.4f}")
            if trend > 0.02:
                print(f"  FINDING: IEs have a LARGER impact in later cycles (post-Citizens United)")
                print(f"           Consistent with IE spending growing substantially after 2010.")
            elif trend < -0.02:
                print(f"  FINDING: IEs had a larger impact in earlier cycles (unusual — check data)")
            else:
                print(f"  FINDING: IE impact is relatively STABLE across cycles.")

    return {
        "r2_base_on_ie_sample": r2_base,
        "r2_with_ie": r2_ie,
        "incremental_r2": delta_r2,
        "se_reduction": delta_se,
        "ie_coefs_full": ie_coefs_full,
        "ie_temporal_stability": ie_temporal_stability,
        "yr_ie_contributions": yr_ie_contributions,
    }


# ---------------------------------------------------------------------------
# Step 7: Write output files  (was Step 6)
# ---------------------------------------------------------------------------

def write_outputs(
    base_fits: list[dict],
    rolling_comparisons: list[dict],
    interaction_result: dict,
    recursive_results: list[dict],
    cv_results: list[dict],
    levels: list[str],
    ie_temporal: dict | None = None,
):
    print(f"\n{'='*60}")
    print("STEP 7: Writing output files")
    print(f"{'='*60}")

    _write_coefficients(base_fits)
    _write_temporal_stability(rolling_comparisons)
    _write_cross_validation(cv_results)
    _write_ie_temporal(ie_temporal)
    _write_summary(base_fits, rolling_comparisons, interaction_result,
                   recursive_results, cv_results, levels, ie_temporal)

    print(f"\n  Output files written to: {OUTPUT}")


def _write_coefficients(base_fits: list[dict]):
    path = OUTPUT / "phase1_coefficients.csv"
    fields = ["model_level", "variable", "label", "coef", "se", "t", "p",
              "ci_lo", "ci_hi", "n", "r2", "residual_se"]
    rows = []
    for fit in base_fits:
        if "error" in fit:
            continue
        for c in fit["coefs"]:
            rows.append({
                "model_level": fit["level"],
                "variable": c["variable"],
                "label": c["label"],
                "coef": round(c["coef"], 6),
                "se": round(c["se"], 6),
                "t": round(c["t"], 4),
                "p": round(c["p"], 6),
                "ci_lo": round(c["ci_lo"], 6),
                "ci_hi": round(c["ci_hi"], 6),
                "n": fit["n"],
                "r2": round(fit["r2"], 6),
                "residual_se": round(fit["residual_se"], 6),
            })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} coefficient rows → {path.name}")


def _write_temporal_stability(comparisons: list[dict]):
    path = OUTPUT / "phase1_temporal_stability.csv"
    if not comparisons:
        print(f"  No temporal stability data to write.")
        return
    fields = list(comparisons[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in r.items()} for r in comparisons])
    print(f"  Wrote temporal stability → {path.name}")


def _write_ie_temporal(ie_temporal: dict | None):
    if not ie_temporal:
        return
    path = OUTPUT / "phase1_ie_temporal.csv"
    rows = ie_temporal.get("yr_ie_contributions", [])
    if not rows:
        return
    fields = ["year", "n", "r2_base", "r2_ie", "delta_r2", "se_reduction"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in r.items()} for r in rows])
    print(f"  Wrote IE temporal analysis → {path.name}")

    # Also write IE coefficient comparisons
    stab = ie_temporal.get("ie_temporal_stability", {})
    if stab:
        path2 = OUTPUT / "phase1_ie_coef_stability.csv"
        fields2 = ["variable", "coef_early", "se_early", "coef_late", "se_late",
                   "diff", "temporal_stability"]
        rows2 = [{"variable": var, **vals} for var, vals in stab.items()]
        with open(path2, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields2)
            writer.writeheader()
            writer.writerows([{k: round(v, 6) if isinstance(v, float) else v
                               for k, v in r.items()} for r in rows2])
        print(f"  Wrote IE coefficient stability → {path2.name}")


def _write_cross_validation(cv_results: list[dict]):
    path = OUTPUT / "phase1_cross_validation.csv"
    if not cv_results:
        print(f"  No cross-validation data to write.")
        return
    fields = list(cv_results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: round(v, 6) if isinstance(v, float) else v
                           for k, v in r.items()} for r in cv_results])
    print(f"  Wrote cross-validation → {path.name}")


def _write_summary(
    base_fits, rolling_comparisons, interaction_result,
    recursive_results, cv_results, levels, ie_temporal=None
):
    path = OUTPUT / "phase1_regression_summary.txt"
    lines = []
    add = lines.append

    add("=" * 70)
    add("PHASE 1 REGRESSION SUMMARY")
    add("Texas Legislative Election Model — Historical Coefficients")
    add("=" * 70)
    add(f"Cycles analyzed: {YEARS}")
    add(f"Model levels run: {levels}")
    add("")

    # ---- Base model coefficients ----
    for fit in base_fits:
        if "error" in fit:
            add(f"\n[{fit['level']}] SKIPPED: {fit['error']}")
            continue
        add(f"\n{'─'*70}")
        add(f"BASE MODEL: {fit['level'].upper()}")
        add(f"  Formula:      {fit['formula']}")
        add(f"  Observations: {fit['n']}")
        add(f"  R²:           {fit['r2']:.4f}")
        add(f"  Adj-R²:       {fit['r2_adj']:.4f}")
        add(f"  Residual SE:  {fit['residual_se']:.4f}")
        add(f"  AIC:          {fit['aic']:.2f}")
        add(f"  Note: Residual SE = {fit['residual_se']:.4f} — this is the uncertainty parameter")
        add(f"        for converting vote share predictions to win probabilities in Phase 2.")
        add("")
        add(f"  {'Variable':<40} {'Coef':>8} {'SE':>7} {'p':>7}  Interpretation")
        add(f"  {'-'*80}")
        for c in fit["coefs"]:
            stars = "***" if c["p"] < 0.01 else "**" if c["p"] < 0.05 else "*" if c["p"] < 0.10 else ""
            add(f"  {c['variable']:<40} {c['coef']:+8.4f} {c['se']:7.4f} {c['p']:7.4f}  {stars}")
        add("")
        add("  Interpretation of key coefficients:")
        for c in fit["coefs"]:
            var = c["variable"]
            coef = c["coef"]
            if var == "national_env":
                add(f"    national_env: A D+1 shift in the national generic ballot is associated")
                add(f"      with a {abs(coef)*100:.2f} pp {'increase' if coef > 0 else 'decrease'} in Dem vote share in each district.")
            elif var == "dem_incumbent":
                add(f"    dem_incumbent: Democratic incumbents run {abs(coef)*100:.1f} pp "
                    f"{'better' if coef > 0 else 'worse'} than in open seat races.")
            elif var == "rep_incumbent":
                add(f"    rep_incumbent: Republican incumbents suppress Dem share by "
                    f"{abs(coef)*100:.1f} pp vs. open seat races.")
            elif var == "baseline_partisanship":
                add(f"    baseline_partisanship: A 1 pp higher governor Dem share → "
                    f"{coef:.4f} pp higher legislative Dem share.")
            elif var == "dem_fundraising_share":
                add(f"    dem_fundraising_share: A 10 pp increase in Dem fundraising share → "
                    f"{coef*0.10*100:.2f} pp increase in Dem vote share.")

    # ---- Temporal stability ----
    add(f"\n{'─'*70}")
    add("TEMPORAL STABILITY (2002-2010 vs 2014-2022)")
    add("─" * 70)
    if rolling_comparisons:
        unstable = [c for c in rolling_comparisons if c["temporal_stability"] == "UNSTABLE"]
        stable = [c for c in rolling_comparisons if c["temporal_stability"] == "STABLE"]
        add(f"  Stable coefficients ({len(stable)}):   "
            f"{[c['variable'] for c in stable]}")
        add(f"  Unstable coefficients ({len(unstable)}): "
            f"{[c['variable'] for c in unstable]}")
        add("")
        if unstable:
            add("  RECOMMENDATION: The following coefficients show meaningful temporal drift.")
            add("  Consider using only the 2014-2022 window for these predictors, or")
            add("  applying a decay-weighted regression that down-weights older cycles.")
            for c in unstable:
                add(f"    {c['variable']}: early={c['coef_early']:+.4f}, late={c['coef_late']:+.4f}")
        else:
            add("  RECOMMENDATION: All coefficients are temporally stable. Using the full")
            add("  2002-2022 sample is appropriate for coefficient estimation.")
    else:
        add("  Rolling window analysis could not be run (insufficient data).")

    # ---- Interaction test ----
    add(f"\n{'─'*70}")
    add("FUNDRAISING × TIME INTERACTION TEST")
    add("─" * 70)
    if interaction_result:
        f_stat = interaction_result.get("f_stat")
        p_val = interaction_result.get("p_value")
        if f_stat is not None:
            add(f"  F-statistic: {f_stat:.3f}, p-value: {p_val:.4f}")
            if p_val < 0.05:
                add("  FINDING: Fundraising effect is changing significantly over time (p<0.05).")
                add("  The fundraising coefficient has grown/shrunk across cycles.")
            else:
                add("  FINDING: No significant time trend in fundraising coefficients (p>0.05).")
                add("  Pooling all cycles for the fundraising estimate is appropriate.")
    else:
        add("  Interaction test not available (no finance data or insufficient coverage).")

    # ---- Recursive estimation ----
    add(f"\n{'─'*70}")
    add("RECURSIVE ESTIMATION (coefficient stability as cycles are added)")
    add("─" * 70)
    if recursive_results and len(recursive_results) >= 2:
        first = recursive_results[0]
        last = recursive_results[-1]
        add(f"  Key coefficients from earliest (thru {YEARS[0]}) to full sample (thru {YEARS[-1]}):")
        for key in [k for k in last.keys() if k.startswith("coef_")]:
            var = key[5:]
            c0 = first.get(key)
            cf = last.get(key)
            if c0 is not None and cf is not None:
                add(f"    {var}: {c0:+.4f} → {cf:+.4f}  (change: {cf-c0:+.4f})")
        add("")
        add("  If early cycles substantially shift the coefficients (|change| > 0.05),")
        add("  consider dropping 2002 and 2006 from the estimation sample.")
    else:
        add("  Recursive estimation not available.")

    # ---- Cross-validation ----
    add(f"\n{'─'*70}")
    add("LEAVE-ONE-CYCLE-OUT CROSS-VALIDATION")
    add("─" * 70)
    if cv_results:
        overall_mae = np.mean([r["mae"] for r in cv_results])
        overall_acc = np.mean([r["winner_accuracy_pct"] for r in cv_results])
        add(f"  Overall MAE: {overall_mae:.4f} ({overall_mae*100:.2f} percentage points)")
        add(f"  Overall winner prediction accuracy: {overall_acc:.1f}%")
        add("")
        add(f"  {'Year':6s} {'n_test':>8} {'MAE':>8} {'Winner%':>10}")
        for r in cv_results:
            add(f"  {r['hold_year']:<6d} {r['n_test']:>8d} {r['mae']:>8.4f} "
                f"{r['winner_accuracy_pct']:>9.1f}%")
        add("")
        add(f"  Interpretation: The model predicts Dem 2-party share within ±{overall_mae*100:.1f} pp")
        add(f"  on average in held-out cycles, and correctly calls the winner {overall_acc:.0f}% of the time.")
        add(f"  This out-of-sample accuracy is the key validity check for applying")
        add(f"  these coefficients to 2026 district projections in Phase 2.")
    else:
        add("  Cross-validation not available.")

    # ---- IE temporal analysis ----
    add(f"\n{'─'*70}")
    add("INDEPENDENT EXPENDITURE ANALYSIS")
    add("─" * 70)
    if ie_temporal:
        dr2   = ie_temporal.get("incremental_r2", 0)
        dse   = ie_temporal.get("se_reduction", 0)
        add(f"  Incremental R² from adding IE variables: {dr2:+.4f}")
        add(f"  Residual SE reduction from IEs:          {dse:+.4f} pp")
        add("")
        stab = ie_temporal.get("ie_temporal_stability", {})
        if stab:
            add("  IE coefficient temporal stability (Early 2002-2010 vs Late 2014-2022):")
            for var, vals in stab.items():
                flag = vals["temporal_stability"]
                add(f"    {var}: early={vals['coef_early']:+.4f}, "
                    f"late={vals['coef_late']:+.4f}  [{flag}]")
        yr_contribs = ie_temporal.get("yr_ie_contributions", [])
        if yr_contribs:
            early_yrs = [r for r in yr_contribs if r["year"] <= 2010]
            late_yrs  = [r for r in yr_contribs if r["year"] >= 2014]
            if early_yrs:
                avg_e = np.mean([r["delta_r2"] for r in early_yrs])
                add(f"  Average incremental R² — Early 2002-2010: {avg_e:+.4f}")
            if late_yrs:
                avg_l = np.mean([r["delta_r2"] for r in late_yrs])
                add(f"  Average incremental R² — Late  2014-2022: {avg_l:+.4f}")
                if early_yrs:
                    add(f"  IE effect is {'LARGER' if avg_l > avg_e else 'SMALLER'} "
                        f"in later cycles (consistent with post-Citizens United growth).")
        ie_coefs = ie_temporal.get("ie_coefs_full", {})
        dem_share_c = ie_coefs.get("ie_dem_share")
        if dem_share_c:
            add("")
            add(f"  INTERPRETATION (ie_dem_share coefficient = {dem_share_c['coef']:+.4f}):")
            add(f"    Moving from 50% D-IE-share to 100% (all IEs favor D) shifts")
            add(f"    Dem vote share by {dem_share_c['coef'] * 0.5 * 100:+.2f} pp.")
            add(f"    p-value = {dem_share_c['p']:.4f} "
                f"({'significant' if dem_share_c['p'] < 0.05 else 'not significant'} at 5%)")
        add("")
        add("  RECOMMENDATION:")
        if dr2 > 0.01 and (not ie_temporal.get("ie_coefs_full") or
                             min(c["p"] for c in ie_temporal.get("ie_coefs_full", {}).values()
                                 if isinstance(c, dict) and "p" in c) < 0.05):
            add("    IE variables are statistically significant and improve model fit.")
            add("    Include ie_dem_share + ie_log_total in Phase 2 model predictions.")
            add("    Update REGRESSION_COEFFICIENTS in model_config.py with full_ie values.")
        else:
            add("    IE variables do not significantly improve model fit (yet).")
            add("    Possible reasons: sparse pre-2010 data; IEs rare in early TX legislative races.")
            add("    Recommend re-running after July 2026 TEC deadline with 2022 full-cycle data.")
    else:
        add("  IE analysis not available. Run:")
        add("    python src/collect_ies_historical.py")
        add("    python src/build_phase1_dataset.py")
        add("  Then re-run this regression.")

    # ---- Final recommendation ----
    add(f"\n{'─'*70}")
    add("RECOMMENDED MODEL SPECIFICATION FOR PHASE 2")
    add("─" * 70)
    best_fit = None
    for level in reversed(levels):
        for fit in base_fits:
            if fit.get("level") == level and "error" not in fit:
                best_fit = fit
                break
        if best_fit:
            break

    if best_fit:
        add(f"  Use model level: {best_fit['level']}")
        add(f"  Formula: {best_fit['formula']}")
        add(f"  Key parameters for Phase 2 probability conversion:")
        add(f"    Residual SE = {best_fit['residual_se']:.4f}")
        add(f"    (Use as σ in: P(Dem wins) = Φ(predicted_dem_2p_share / σ))")
        add("")
        add(f"  To compute win probability for a district:")
        add(f"    1. Predict dem_2p_share using the coefficients above")
        add(f"    2. Convert: P(win) = norm.cdf((prediction - 0.5) / {best_fit['residual_se']:.4f})")
    else:
        add("  Could not determine a best model — check data collection and re-run.")

    add(f"\n{'='*70}")
    add("END OF SUMMARY")
    add(f"{'='*70}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Wrote summary → {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(min_level: str = "restricted"):
    df = load_dataset()

    levels = choose_completeness_level(df, min_level)
    print(f"\nModel levels to run: {levels}")
    if len(levels) < 3:
        missing = [l for l in ["restricted", "with_gov", "full"] if l not in levels]
        for m in missing:
            needed = {"with_gov": "baseline_partisanship", "full": "finance data"}
            print(f"  Skipping '{m}': insufficient {needed.get(m, 'data')}")
            print(f"    Run collect_gov_by_district.py / collect_finance.py and rebuild dataset.")

    base_fits = step1_base_models(df, levels)
    rolling = step2_rolling_window(df, levels)
    interaction = step3_interaction_test(df, levels)
    recursive = step4_recursive(df, levels)
    cv = step5_cross_validation(df, levels)
    ie_temporal = step6_ie_temporal(df)
    write_outputs(base_fits, rolling, interaction, recursive, cv, levels, ie_temporal)

    print(f"\nPhase 1 complete.")
    print(f"See output/phase1_regression_summary.txt for the full narrative.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 1 regression analysis")
    parser.add_argument(
        "--min-completeness",
        choices=["restricted", "with_gov", "full"],
        default="restricted",
        help="Minimum model level to attempt (default: restricted — always runs)",
    )
    args = parser.parse_args()
    main(min_level=args.min_completeness)
