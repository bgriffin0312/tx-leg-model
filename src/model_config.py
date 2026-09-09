"""
model_config.py

Manually-updated configuration for the Phase 2 TX legislative election model.

UPDATE TRIGGERS:
  1. TEC filing deadline passes (Jan, Apr, Jul, Oct) — re-run collect_finance_2026.py
     (use --force-download; the TEC cache has no TTL), then update
     FINANCE_DATA_THROUGH below. That constant means "last filing DEADLINE
     captured" and switches coefficient eras — don't bump it just because data
     was re-pulled.
  2. Generic ballot topline shifts > 1pp — run `python src/update_polling.py`
     (dry run) and then `--apply`. It rewrites RACE_GENERIC_BALLOT_D_SHARE,
     GENERIC_BALLOT_SOURCE/UPDATED and the topline below.

WHERE THE RACIAL CROSSTABS COME FROM (superseded Civiqs, 2026-08-15):
  Civiqs is no longer used — its by-race view sits behind an interactive
  dashboard. update_polling.py aggregates free published crosstabs instead:
  Economist/YouGov weekly (hand-maintained URL list) + anything added by hand
  to data/raw/racial_crosstab_inputs.csv. Most pollsters publish NO racial
  crosstabs at all — verified 2026-08-15: Quinnipiac, CNN/SSRS, Emerson,
  Ipsos (party-ID only), Marist (no national generic ballot since March),
  RMG (confidential), Cygnal (narrative deck only). Ones that DO: YouGov
  (adults universe — its race rows are NOT registered voters), Fox/Beacon-Shaw
  (RV, but no Black column), Pew, Big Data Poll (via MarketSight), and
  The Argument/Verasight (n=3,000 RV, best single source).

  Two-party share = D% / (D% + R%). Keep leaned/unleaned consistent across
  polls. A raw RCP margin understates the two-party margin (D+6.8 ≈ D+7.5 2p).

NATIONAL TOPLINE SOURCES (for detecting >1pp shifts):
  - Silver Bulletin generic ballot tracker
  - RealClearPolitics average
"""

# ---------------------------------------------------------------------------
# Generic Ballot by Race (D two-party share)
# ---------------------------------------------------------------------------
# These are the Democratic 2-party share of the generic congressional ballot
# disaggregated by racial/ethnic group.
#
# Source + last-updated date are recorded in GENERIC_BALLOT_SOURCE /
# GENERIC_BALLOT_UPDATED below — update_polling.py rewrites all of these
# together, so treat those two constants as authoritative rather than any
# comment here.
#
# !!! DO NOT HAND-EDIT — run `python src/update_polling.py --apply`  !!!
# !!! Refresh when the topline aggregate shifts by more than 1pp     !!!

RACE_GENERIC_BALLOT_D_SHARE: dict[str, float] = {
    # White non-Hispanic: historically R+15 to R+20 nationally; Trump era ~R+16
    "white_nh": 0.4789,

    # Black non-Hispanic: strongly Democratic, typically D+80 to D+90
    "black_nh": 0.8755,

    # Hispanic/Latino: shifted R in 2024 (nationally ~D+20 to D+30 vs. D+40+ in 2020)
    # TX Hispanics in 2024 were approximately even in some districts
    "hispanic": 0.6278,

    # Asian non-Hispanic + other: generally D-leaning, D+10 to D+20
    "other": 0.4508,
}

# Metadata — update these when you update the numbers above
GENERIC_BALLOT_SOURCE = "Multi-source 3-poll racial avg + 0 topline-only"
GENERIC_BALLOT_UPDATED = "2026-08-15"  # ISO date

# Topline D 2p share at last update — used by update_polling.py to compute shifts
# when Civiqs racial crosstabs aren't available. Computed as Σ(weight × D_share).
# Run update_polling.py to refresh automatically.
GENERIC_BALLOT_TOPLINE_D_2P: float = 0.5455  # D+9.1pp (implied by racial shares above)

# ---------------------------------------------------------------------------
# National Demographic Weights (2024 exit poll / electorate composition)
# ---------------------------------------------------------------------------
# Used to compute the national average D share from race-specific numbers.
# These reflect the 2024 presidential electorate composition nationally.
# Update after each election cycle.

NATIONAL_DEMO_WEIGHTS: dict[str, float] = {
    "white_nh":  0.61,   # ~61% of 2024 electorate
    "black_nh":  0.12,   # ~12%
    "hispanic":  0.15,   # ~15%
    "other":     0.12,   # Asian + AIAN + other
}

# ---------------------------------------------------------------------------
# Phase 1 Regression Coefficients (from run_phase1_regression.py)
# ---------------------------------------------------------------------------
# These come from the FULL model (with finance) in output/phase1_regression_summary.txt.
# Update after re-running the regression with the presidential baseline.
#
# Current values are from the RESTRICTED model (no presidential baseline yet).
# After Task 4 (refit with presidential baseline), update with new coefficients.

REGRESSION_COEFFICIENTS: dict[str, float] = {
    # From FULL model with presidential baseline (run_phase1_regression.py output)
    # n=268 contested races with full data (presidential + finance)
    # R²=0.7953, Residual SE=0.0742
    #
    # REFIT 2026-07-20: the wikilink parser fix restored 34 mislabeled
    # incumbency flags in phase1_dataset, so the regression was re-run.
    # Backtests old→new: 2022 Brier 0.0220→0.0214 (acc 99.0→97.9%),
    # 2018 Brier 0.0800→0.0859 (acc 87.9% both) — a wash; adopted because
    # the new fit uses corrected labels. Previous values: intercept 0.1520,
    # pass-through 0.6604, dem_inc 0.0600, rep_inc −0.0739, senate −0.0261,
    # viability 0.0393, share 0.0624, sigma 0.0785.
    # REFIT 2026-09-09 on CLEAN-VINTAGE MIDTERM cycles only (2014/2018/2022,
    # n=208, R2=0.9802, in-sample residual 0.0230).
    #
    # The previous values were fit on six cycles that all used the 2024
    # presidential result as their baseline. District numbers are not stable
    # across redistricting, so five of the six carried a baseline for the wrong
    # geography. Measurement error in the main regressor attenuates its
    # coefficient toward zero: that alone explains pass-through reading 0.5962
    # where clean cycles give ~0.98, with the intercept inflated to compensate.
    #
    # Old and new agree at the centre and diverge at the tails, where the old
    # fit pulled safe seats toward the middle. A 50%-Harris open seat: 0.4762
    # old vs 0.4686 new. A 0.35 baseline: 0.3868 old vs 0.3206 new.
    #
    # rep_incumbent is shipped as fitted (+0.0078, p=0.155) rather than zeroed.
    # It is NOT a claim that Republican incumbency helps Democrats -- with a
    # near-perfect baseline there is nothing left for it to explain, and at
    # 0.8pp it is noise around zero whichever way it is set.
    "intercept":                 -0.0322,
    "dem_pres_2p_baseline":       0.9759,
    "dem_incumbent":              0.0394,
    "rep_incumbent":              0.0078,
    "chamber_senate":             0.0069,
    # national_env: auto-selected based on FINANCE_DATA_THROUGH (see below).
    # Both branches refit on the clean midterms; the old gap between them was
    # mostly sample composition, and it largely closes once the baseline is
    # right (full 0.0036 vs with_pres 0.0033).
    "national_env":               None,  # set automatically by _auto_select_env_coef()
    # Both finance terms are now statistically indistinguishable from zero on
    # clean cycles (viability p=0.798, share p=0.208). The old +0.0731 share
    # coefficient was carried by early cycles where the "effect" was really a
    # TEC name-match artifact -- 143 of 268 training rows had the share pinned
    # at exactly 0 or 1, and in 53 of them the $0 side was the sitting
    # incumbent. Shipped as fitted; they now move a district by tenths of a pp.
    "challenger_viability_flag":  0.0015,
    "dem_fundraising_share":      0.0057,
    # FORECAST sigma, not the in-sample residual (0.0230). In-sample residual
    # excludes cross-cycle level uncertainty, which is most of the real error.
    # From leave-one-cycle-out over the five clean cycles (2012-2022):
    #     sigma_national (RMS of held-out cycle level shifts) 0.0339
    #     sigma_idio     (pooled within-cycle sd)             0.0280
    #     total sqrt(n^2 + i^2)                               0.0440
    # That decomposition also validates the "high-corr" 58% variance share
    # model.py has been using: the evidence puts it at 59%.
    "sigma":                      0.0440,
}

# ---------------------------------------------------------------------------
# Viability Thresholds (early-cycle)
# ---------------------------------------------------------------------------
# These will be calibrated by notebooks/calibrate_early_viability.ipynb
# after the windowed finance data (Task 1b) is collected.
# Placeholder values below are proportional to the full-cycle thresholds.

VIABILITY_THRESHOLD_POSTPRIMARY: dict[str, float] = {
    # Full-cycle thresholds: house=$100k, senate=$250k.
    # Calibrated against 2018 and 2022 TEC data: thresholds are roughly the
    # median dollar amount that ultimately-viable candidates had raised by
    # April 30 in those cycles.
    #   House: median apr30 was $40K (2018) / $53K (2022) → $40K
    #   Senate: median apr30 was $71K (2018) / $22K (2022) → $60K
    # Senate is more variable cycle-to-cycle so the threshold is set toward
    # the lower end of the historical range.
    "house":   40_000,
    "senate":  60_000,
}

VIABILITY_THRESHOLD_SEMIJUL: dict[str, float] = {
    # Calibrated 2026-07-20 from TEC cover.csv (all-cycles cache): median
    # election-year raised as of Jul 20 (reports with periodStart in the
    # election year AND filed by Jul 20 — the same quantity the 2026 pipeline
    # measures) among ultimately-viable candidates (full-cycle >= $100K house /
    # $250K senate). Off-year cycles are the comparators for 2026:
    #   House:  $82K (2018) / $115K (2022) → $80K (lower end, per convention)
    #   Senate: $305K (2018) / $135K (2022), n≈28 so highly variable → $135K
    # NOTE: the POSTPRIMARY apr30 medians documented above could NOT be
    # reproduced under this election-year-only window (2018 house median is
    # ~$0.3K, not $40K) — the original calibration evidently included prior-
    # year fundraising, a broader window than the pipeline applies. The
    # SEMIJUL numbers here are measurement-consistent with the pipeline.
    "house":   80_000,
    "senate": 135_000,
}

# Auto-select the viability threshold era from FINANCE_DATA_THROUGH, the same
# way the national_env coefficient switches (see below): July semi-annual data
# should not be judged against thresholds calibrated to April war chests.
def _auto_select_viability_threshold() -> dict[str, float]:
    try:
        month = int(FINANCE_DATA_THROUGH.replace("-", "")[4:6])
    except (ValueError, IndexError):
        month = 1
    return VIABILITY_THRESHOLD_SEMIJUL if month >= 7 else VIABILITY_THRESHOLD_POSTPRIMARY

# The dict collectors should import; resolved at import time (below, after
# FINANCE_DATA_THROUGH is defined).
VIABILITY_THRESHOLD: dict[str, float] = {}

# ---------------------------------------------------------------------------
# IE (Independent Expenditure) signal
# ---------------------------------------------------------------------------
# Coefficient from Phase 1 regression (full_ie model, n=102, p=0.012).
# ie_dem_share = D-favoring IEs / total IEs (0–1 scale; 0.5 = neutral/no IEs).
# Additive effect: COEF * IE_WEIGHT * (ie_dem_share − 0.5)
#   → full R-favor (0.0) shifts predicted share by −0.037pp
#   → full D-favor (1.0) shifts predicted share by +0.037pp
#
# IE_WEIGHT — scales signal based on how far along the cycle we are:
#   0.5  post-runoff, May–June: early-cycle targeting, high primary noise
#   0.75 post-July TEC filing:  semi-annual report, cleaner general-election signal
#   1.0  October/pre-election:  full-cycle IEs, most predictive
#
# IE_MIN_THRESHOLD — minimum total IE $ for the adjustment to apply.
# Below this, spending is likely routine PAC maintenance, not targeted competition.
# Based on analysis showing meaningful targeting starts around $50K–$200K.
#
# !!! UPDATE IE_WEIGHT AFTER EACH TEC FILING !!!
#   After July filing  → IE_WEIGHT = 0.75
#   After October filing → IE_WEIGHT = 1.0
# !!! UPDATE IE_DATA_THROUGH WHEN RUNNING collect_ies_2026.py !!!

IE_COEFFICIENT:   float = 0.074     # from full_ie regression (with_ie model p=0.000)
IE_MIN_THRESHOLD: float = 50_000    # $50K minimum total IEs for signal to apply
IE_WEIGHT:        float = 0.75      # post-July semi-annual filing (was 0.5 post-runoff)
IE_DATA_THROUGH:  str   = "2026-07-20"  # update when re-running collect_ies_2026.py

# ---------------------------------------------------------------------------
# Finance data currency
# ---------------------------------------------------------------------------
FINANCE_DATA_THROUGH = "2026-07-15"  # last TEC filing deadline captured (July semi-annual report)
FINANCE_CUTOFF_POSTPRIMARY = "20260815"  # include all reports filed through today (July semi-annual + Aug 5 monthly PAC/IE reports)

# ---------------------------------------------------------------------------
# Auto-select national_env coefficient based on finance data currency
# ---------------------------------------------------------------------------
# Pre-July: dem_fundraising_share is sparse/unreliable, so the full-model
# coefficient (0.0027) understates environment sensitivity due to collinearity
# with finance vars. Use with_pres coefficient (0.0052) instead.
# Post-July: TEC semi-annual filing populates dem_fundraising_share fully,
# so the full-model coefficient is appropriate.
#
# Set NATIONAL_ENV_COEF_OVERRIDE to force a specific value (bypasses auto).
NATIONAL_ENV_COEF_OVERRIDE: float | None = None

# ---------------------------------------------------------------------------
# No-finance coefficients, for the WAR baseline
# ---------------------------------------------------------------------------
# WAR is a residual against what a replacement-level candidate would do, so its
# baseline must exclude the finance terms -- a candidate's own fundraising is
# part of the quality being measured.
#
# compute_war used to take REGRESSION_COEFFICIENTS and simply drop the finance
# regressors while KEEPING the intercept. An intercept is not a constant of
# nature; it is whatever made the fitted line pass through the data GIVEN the
# other terms. Removing regressors without refitting left the baseline biased,
# and the bias landed on candidates: mean residual was +2.34pp instead of 0.
# Because party_sign flips WAR for Republicans, a uniform positive residual
# became a pro-D thumb on the scale across 110 of 166 districts.
#
# These are a genuine no-finance fit (with_pres) on the same clean-vintage
# midterms, n=259, R2=0.9766, in-sample residual 0.0255. Residuals are
# mean-zero by construction, which is the property WAR actually needs and the
# reason a refit is right where per-run demeaning is a patch: swapping one
# input file moved the old mean by half a point.
REGRESSION_COEFFICIENTS_NO_FINANCE: dict[str, float] = {
    "intercept":            -0.0245,
    "dem_pres_2p_baseline":  0.9861,
    "dem_incumbent":         0.0349,
    "rep_incumbent":         0.0064,
    "chamber_senate":        0.0072,
    "national_env":          0.0033,
    "sigma":                 0.0255,
}

# Refit 2026-09-09 on clean-vintage midterms. The old gap between these two
# (0.0049 vs 0.0025) was largely sample composition rather than collinearity,
# and it nearly closes once the presidential baseline is on the right lines.
_ENV_COEF_WITH_PRES = 0.0033   # with_pres, clean midterms n=259 (was 0.0049)
_ENV_COEF_FULL_MODEL = 0.0036  # full model, clean midterms n=208 (was 0.0025)

def _auto_select_env_coef() -> float:
    """Select national_env coefficient based on FINANCE_DATA_THROUGH date."""
    if NATIONAL_ENV_COEF_OVERRIDE is not None:
        print(f"  national_env coefficient: {NATIONAL_ENV_COEF_OVERRIDE} (manual override)")
        return NATIONAL_ENV_COEF_OVERRIDE

    try:
        month = int(FINANCE_DATA_THROUGH.replace("-", "")[4:6])
    except (ValueError, IndexError):
        month = 1

    if month >= 7:
        print(f"  Post-July: using full-model environment coefficient ({_ENV_COEF_FULL_MODEL})")
        if "dem_fundraising_share" not in REGRESSION_COEFFICIENTS:
            print("  WARNING: dem_fundraising_share not in REGRESSION_COEFFICIENTS "
                  "but using post-July coefficient")
        return _ENV_COEF_FULL_MODEL
    else:
        print(f"  Pre-July: using with_pres environment coefficient ({_ENV_COEF_WITH_PRES})")
        return _ENV_COEF_WITH_PRES

# Apply auto-selection at import time
REGRESSION_COEFFICIENTS["national_env"] = _auto_select_env_coef()
VIABILITY_THRESHOLD.update(_auto_select_viability_threshold())

# ---------------------------------------------------------------------------
# Model scenarios
# ---------------------------------------------------------------------------
# National environment dial: ABSOLUTE D-R generic ballot margin (percentage points).
# This is calibrated the same way the regression's national_env was estimated —
# the training data used absolute generic ballot values (R+4.6, D+8.6, etc.).
#
# Reference points:
#   2024 election: approx. R+1 to neutral (generic ballot polling in Nov 2024)
#   Current (Apr 2026): D+4.8 — see GENERIC_BALLOT_TOPLINE_D_2P above
#   2018 Dem wave: D+8.6
#
# The current environment is automatically added as a labeled scenario in model.py.
# ENV_SCENARIOS are the fixed reference points for comparison.
#
# e.g.:
#  -3 → R+3 national environment (worse than 2024)
#   0 → neutral / 2024-like environment
#  +3 → D+3 national environment (modest Dem lean)
#  +5 → approximately current environment (Apr 2026 ≈ D+4.8)
#  +8 → strong D wave (2018-level)

ENV_SCENARIOS: list[int] = [-3, 0, 3, 5, 8]

# ---------------------------------------------------------------------------
# WAR (Wins Above Replacement) persistence
# ---------------------------------------------------------------------------
# Empirically estimated from 91 candidate-pairs across 2018→2022 and 2022→2024.
#   Combined β = 0.432  (r=0.508, p<0.0001)
#   Competitive only β = 0.459  (r=0.531, p<0.0001)
#   2022→2024 only  β = 0.549  (more weight on recent cycle)
# Using 0.46 as the working estimate (rounds competitive β, skews toward recent).
#
# Applied only to incumbents found in data/processed/candidate_war.csv.
# For those districts, challenger_viability_flag and dem_fundraising_share
# are dropped from the baseline (WAR already incorporates fundraising ability).
WAR_PERSISTENCE_COEF: float = 0.46

# ---------------------------------------------------------------------------
# TX-specific Hispanic voting adjustment
# ---------------------------------------------------------------------------
# National racial crosstabs systematically overestimate Hispanic D support in TX.
# 2022 backtest regression: error ~ +0.068 * hispanic_cvap_pct (p=0.057).
# Full regression-implied adjustment is -0.07, but 2022 was peak Hispanic-R
# divergence (inflation frustration).
#
# 2018 backtest validation (Apr 11 2026, with proper 2014-2018 ACS CVAP under
# H2100/S2100 districts and Mar-Jun 2018 polling crosstabs from 4 polls):
#   adj=0.00  house_err=+5.9  brier=0.068  acc=90.1%
#   adj=-0.04 house_err=+3.5  brier=0.066  acc=92.1%
#   adj=-0.07 house_err=+1.6  brier=0.065  acc=91.1%
# A structural Hispanic gap was already present in 2018 — pre-realignment, in
# a Beto wave year. The pure-cyclical "LIFO" interpretation is not supported.
#
# -0.05 chosen as compromise: 2018 evidence supports a real structural gap
# beyond the 2022 cyclical spike, but going to -0.07 risks overcorrecting if
# 2026 sees Hispanic D reversion.
# At -0.05: 50% Hispanic district shifts -2.5pp, 80% district shifts -4.0pp.
# Set to 0.0 to disable.
TX_HISPANIC_ADJUSTMENT: float = -0.05

# Monte Carlo simulation count
N_SIMULATIONS: int = 10_000

# Chamber control thresholds (seats needed for majority)
HOUSE_MAJORITY: int = 76   # out of 150
SENATE_MAJORITY: int = 16  # out of 31
