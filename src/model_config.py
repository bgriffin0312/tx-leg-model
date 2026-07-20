"""
model_config.py

Manually-updated configuration for the Phase 2 TX legislative election model.

UPDATE TRIGGERS:
  1. TEC filing deadline passes (Jan, Apr, Jul, Oct) — re-run collect_finance.py,
     then update FINANCE_DATA_THROUGH below.
  2. Generic ballot topline shifts > 1pp — check Civiqs monthly for race-specific
     crosstabs, then update RACE_GENERIC_BALLOT_D_SHARE and GENERIC_BALLOT_UPDATED.

HOW TO READ CIVIQS RACE-SPECIFIC GENERIC BALLOT:
  Go to civiqs.com → "Congressional Generic Ballot" → filter by "Race/Ethnicity"
  Record the Democratic 2-party share for each group:
    D% / (D% + R%) for each racial category.
  Use Registered Voter (RV) numbers, not Likely Voter.

NATIONAL TOPLINE SOURCES (for detecting >1pp shifts):
  - FiveThirtyEight/Silver Bulletin generic ballot tracker
  - RealClearPolitics average
  - The Economist model
"""

# ---------------------------------------------------------------------------
# Generic Ballot by Race (D two-party share)
# ---------------------------------------------------------------------------
# These are the Democratic 2-party share of the generic congressional ballot
# disaggregated by racial/ethnic group.
#
# Source: Civiqs monthly tracking (civiqs.com), Registered Voter numbers
# Last updated: 2026-04-06
# Topline at time of update: D+4.8 (2026-04-06)
#
# !!! UPDATE THESE WHEN CIVIQS PUBLISHES A NEW MONTHLY RELEASE !!!
# !!! OR WHEN THE TOPLINE AGGREGATE SHIFTS BY MORE THAN 1PP     !!!

RACE_GENERIC_BALLOT_D_SHARE: dict[str, float] = {
    # White non-Hispanic: historically R+15 to R+20 nationally; Trump era ~R+16
    "white_nh": 0.4589,

    # Black non-Hispanic: strongly Democratic, typically D+80 to D+90
    "black_nh": 0.8685,

    # Hispanic/Latino: shifted R in 2024 (nationally ~D+20 to D+30 vs. D+40+ in 2020)
    # TX Hispanics in 2024 were approximately even in some districts
    "hispanic": 0.5796,

    # Asian non-Hispanic + other: generally D-leaning, D+10 to D+20
    "other": 0.4658,
}

# Metadata — update these when you update the numbers above
GENERIC_BALLOT_SOURCE = "Multi-source 2-poll racial avg + 0 topline-only"
GENERIC_BALLOT_UPDATED = "2026-07-20"  # ISO date

# Topline D 2p share at last update — used by update_polling.py to compute shifts
# when Civiqs racial crosstabs aren't available. Computed as Σ(weight × D_share).
# Run update_polling.py to refresh automatically.
GENERIC_BALLOT_TOPLINE_D_2P: float = 0.5270  # D+5.0pp (implied by racial shares above)

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
    "intercept":                  0.1781,
    "dem_pres_2p_baseline":       0.5962,
    "dem_incumbent":              0.0676,
    "rep_incumbent":             -0.0804,
    "chamber_senate":            -0.0263,
    # national_env: auto-selected based on FINANCE_DATA_THROUGH (see below).
    # Pre-July: 0.0049 (with_pres model, less suppressed by finance collinearity)
    # Post-July: 0.0025 (full model, when dem_fundraising_share is fully populated)
    "national_env":               None,  # set automatically by _auto_select_env_coef()
    "challenger_viability_flag":  0.0446,
    # dem_fundraising_share: D raised / (D+R raised). From full model = +0.0731 per unit (0–1).
    # NOTE: This coefficient is temporally unstable — early cycles (2002–2010): +0.22,
    # late cycles (2014–2022): ~0.00. The full-model average is used here.
    # Pre-primary party assignment is approximate (challenger_raised may include
    # same-party primary opponents). Treat with caution until post-July TEC data.
    "dem_fundraising_share":      0.0731,
    "sigma":                      0.0742,  # residual SE from FULL model (for win probability CDF)
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
FINANCE_CUTOFF_POSTPRIMARY = "20260720"  # include all reports filed through today (captures July semi-annual)

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

_ENV_COEF_WITH_PRES = 0.0049   # less suppressed by finance collinearity (2026-07-20 refit; was 0.0052)
_ENV_COEF_FULL_MODEL = 0.0025  # full model with finance vars active (2026-07-20 refit; was 0.0027)

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
