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
import os

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
    "white_nh": 0.4778,

    # Black non-Hispanic: strongly Democratic, typically D+80 to D+90
    "black_nh": 0.8569,

    # Hispanic/Latino: shifted R in 2024 (nationally ~D+20 to D+30 vs. D+40+ in 2020)
    # TX Hispanics in 2024 were approximately even in some districts
    "hispanic": 0.6327,

    # Asian non-Hispanic + other: generally D-leaning, D+10 to D+20
    "other": 0.4720,
}

# Metadata — update these when you update the numbers above
GENERIC_BALLOT_SOURCE = "Multi-source 11-poll racial avg + 0 topline-only"
GENERIC_BALLOT_UPDATED = "2026-10-07"  # ISO date

# Topline D 2p share at last update — used by update_polling.py to compute shifts
# when Civiqs racial crosstabs aren't available. Computed as Σ(weight × D_share).
# Run update_polling.py to refresh automatically.
GENERIC_BALLOT_TOPLINE_D_2P: float = 0.5458  # D+9.2pp (implied by racial shares above)

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
# Partisan baseline source
# ---------------------------------------------------------------------------
# Which 2024 statewide race supplies each district's partisan baseline.
#   "pres"  — 2024 presidential two-party share (districts_2026.csv column).
#   "rrc"   — 2024 Railroad Commissioner (Craddick v. Warford), from
#             data/raw/historical/tx_downballot_{chamber}_2024_plan*.csv.
#   "blend" — half presidential, half Railroad Commissioner.
#   "open"  — mean of the 2024 statewide downballot races with NO incumbent
#             on either side (Brennan's rule, 2026-09-10): CCA Presiding
#             Judge, CCA 7, CCA 8. The 2024 RRC race had an incumbent
#             (Craddick), so it is excluded here.
#   "blend_open" — half presidential, half open-race mean.
# Evidence: docs/statewide-proxy-findings.md and scripts/baseline_backtest.py
# (2026-09-10). Same-year the presidential vote is the worst of the nine 2024
# statewide races as a House proxy; leave-one-cycle-out on 2014/2018/2022 the
# blend has the lowest residual sd and RRC the lowest seat error and Brier,
# with RRC winning outright in 2022, the only post-straight-ticket cycle.
# The judicial mean was worse than both in every cycle and is not offered.
#
# ADOPTED 2026-10-07 (Brennan): "open". REGRESSION_COEFFICIENTS below are fit
# on this baseline (scripts/refit_clean_cycles.py), so the two go together.
BASELINE_SOURCE: str = "open"

# ---------------------------------------------------------------------------
# Phase 1 Regression Coefficients (from run_phase1_regression.py)
# ---------------------------------------------------------------------------
# CLEAN-CYCLE REFIT ON THE OPEN-RACE BASELINE, adopted 2026-10-07 (Brennan).
# scripts/refit_clean_cycles.py: contested races of the clean-vintage midterms
# 2014/2018/2022, each scored against the prior presidential year's open-race
# composite on the lines the race ran under; n=212 with finance. Training data
# rebuilt the same day with corrected TEC reports deduplicated and the August
# incumbent-filing fix applied to 2018/2022 (see NEXT-STEPS 2026-10-07).
#
# Why not the old values (intercept 0.178, pass-through 0.596, dem_inc +6.8,
# rep_inc -8.0, viability 4.5, share 7.3pp, sigma 0.074): they were fit with a
# baseline on the wrong district geography for every cycle before 2022, which
# attenuated the pass-through and pushed the slack into the intercept and the
# incumbency terms. Imposing those incumbency/finance values on the clean data
# gives a held-out RMSE of 10.5pp against 3.2pp for the refit.
#
# Each block was tested out of sample (leave-one-cycle-out), per Brennan:
#   incumbency      kept: held-out RMSE 3.64 -> 3.37pp on all 259 races. Small,
#                   and both signs D-ward: open seats run ~1.3pp more R than
#                   baseline in competitive districts; incumbents of either
#                   party a little more D.
#   viability flag  kept: neutral out of sample, +1.1pp (t 1.6). Fit SIGNED
#                   (+1 viable D vs R incumbent, -1 viable R vs D incumbent),
#                   matching how model.py applies it; training had stored it
#                   unsigned, which scored 4 viable R challengers as pro-D.
#   fundraising share  DROPPED (set 0): worse out of sample on every measure.
#   IE              kept at the refit estimate (IE_COEFFICIENT below, +0.9pp
#                   per unit share): neutral out of sample.
#
# sigma: 0.044 is the forecast sigma from the branch's leave-one-cycle-out over
# five clean cycles (national 0.0339 / idio 0.0280) -- Brennan's choice. The
# open-baseline LOO over three midterms gives ~0.031, but two of its three
# held-out "cycles" carry the national term, so 0.044 is kept as the safer
# value.

REGRESSION_COEFFICIENTS: dict[str, float] = {
    "intercept":                 -0.0334,
    "dem_pres_2p_baseline":       1.0583,  # pass-through on the BASELINE_SOURCE column
    "dem_incumbent":              0.0117,
    "rep_incumbent":              0.0128,
    "chamber_senate":             0.0095,
    # national_env: set by _auto_select_env_coef() (both eras 0.0046 after refit).
    "national_env":               None,
    "challenger_viability_flag":  0.0109,  # SIGNED in training and in model.py
    "dem_fundraising_share":      0.0,     # dropped 2026-10-07: fails out of sample
    "sigma":                      0.0440,
}

# No-finance coefficients, for the WAR baseline (compute_war.predict_dem_share).
# WAR is a residual against a replacement-level candidate, so the baseline must
# exclude finance -- and must be a genuine no-finance FIT, not the full model
# with terms deleted (that kept an intercept fit alongside them and left a
# +2.3pp mean residual that party_sign turned into a pro-D thumb).
# Same rows and baseline as above, no-finance sample, n=259.
REGRESSION_COEFFICIENTS_NO_FINANCE: dict[str, float] = {
    "intercept":            -0.0374,
    "dem_pres_2p_baseline":  1.0668,
    "dem_incumbent":         0.0102,
    "rep_incumbent":         0.0154,
    "chamber_senate":        0.0096,
    "national_env":          0.0046,
    "sigma":                 0.0288,
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

VIABILITY_THRESHOLD_OCT30: dict[str, float] = {
    # Calibrated 2026-10-07 by scripts/calibrate_viability_threshold.py, on a
    # classification criterion: the in-cycle threshold that best reproduces
    # the full-cycle flag the regression trained on (non-incumbent raised
    # >= $100K house / $250K senate), using election-year reports filed by
    # the 30-day-before-general deadline + 2 days, 2014/2018/2022 pooled.
    # The same criterion reproduces the SEMIJUL values above ($75-80K house,
    # $140K senate), which is why it is trusted here.
    #   House:  minimum errors (12 of 551) at $100K; $80K gives 16, $90K 15.
    #   Senate: minimum (2 of 118) at $220K; $200K and $250K give 3. n is
    #           small (~9-14 viable per cycle), so treat as approximate.
    "house":  100_000,
    "senate": 220_000,
}

# Auto-select the viability threshold era from FINANCE_DATA_THROUGH, the same
# way the national_env coefficient switches (see below): July semi-annual data
# should not be judged against thresholds calibrated to April war chests.
def _auto_select_viability_threshold() -> dict[str, float]:
    try:
        month = int(FINANCE_DATA_THROUGH.replace("-", "")[4:6])
    except (ValueError, IndexError):
        month = 1
    if month >= 10:
        return VIABILITY_THRESHOLD_OCT30
    return VIABILITY_THRESHOLD_SEMIJUL if month >= 7 else VIABILITY_THRESHOLD_POSTPRIMARY

# The dict collectors should import; resolved at import time (below, after
# FINANCE_DATA_THROUGH is defined).
VIABILITY_THRESHOLD: dict[str, float] = {}

# ---------------------------------------------------------------------------
# IE (Independent Expenditure) signal
# ---------------------------------------------------------------------------
# Coefficient history: 0.074 (102-row full_ie subsample) -> 0.026 (all 268
# contested midterms, master terms; scripts/ie_meaning_backtest.py) -> 0.0089
# (clean-cycle open-baseline refit, scripts/refit_clean_cycles.py), all on
# 2026-10-07. The backtest found IE money is not a weakness signal: defended
# favourites run at their baseline. On the refit terms it is close to zero.
# ie_dem_share = D-favoring IEs / total IEs (0–1 scale; 0.5 = neutral/no IEs).
# Additive effect: COEF * IE_WEIGHT * (ie_dem_share − 0.5)
#   → full R-favor (0.0) shifts predicted share by −0.4pp at weight 1.0
#   → full D-favor (1.0) shifts predicted share by +0.4pp
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

IE_COEFFICIENT:   float = 0.0089    # clean-cycle open-baseline refit, 2026-10-07 (t 1.1); was 0.026 earlier that day, 0.074 before
IE_MIN_THRESHOLD: float = 50_000    # $50K minimum total IEs for signal to apply
IE_WEIGHT:        float = 1.0       # October 30-day-before reports in (was 0.75 post-July)
IE_DATA_THROUGH:  str   = "2026-10-07"  # update when re-running collect_ies_2026.py

# ---------------------------------------------------------------------------
# Finance data currency
# ---------------------------------------------------------------------------
FINANCE_DATA_THROUGH = "2026-10-05"  # last TEC filing deadline captured (30-day-before-general report)
FINANCE_CUTOFF_POSTPRIMARY = "20261007"  # include all reports filed through today (Oct 5 30-day-before-general reports)

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

_ENV_COEF_WITH_PRES = 0.0046   # 2026-10-07 clean-cycle open-baseline refit (was 0.0049); the two eras now agree
_ENV_COEF_FULL_MODEL = 0.0046  # 2026-10-07 clean-cycle open-baseline refit (was 0.0025)

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

# ---------------------------------------------------------------------------
# Midterm education term — ADOPTED 2026-10-07 (Brennan)
# ---------------------------------------------------------------------------
# scripts/group_error_estimate.py --education / scripts/midterm_education_test.py:
# against the open-race baseline, every clean midterm runs more Democratic in
# districts with more college-educated white adults (+13.8/+17.1/+13.6 pp per
# unit centered share in 2014/2018/2022). A fixed term for it passes the
# out-of-sample test the race-only version failed: held-out RMSE 3.37→3.13pp on
# all 259 races, 3.20→2.72pp on the shipped spec; adding race terms on top makes
# it worse; on the PRESIDENTIAL baseline the effect is zero, i.e. it corrects a
# blind spot of the downballot open-race composite.
#     + white_col_centered × (white BA+ share of CVAP − Texas 0.201)
# white BA+ share = CVAP white share × ACS white-NH 25+ BA+ rate
# (src/collect_education_by_district.py). Whole spec refit with the term in.
# On by default since 2026-10-07; TXLEG_EDU=0 turns it off for one run.
REGRESSION_COEFFICIENTS_EDU: dict[str, float] = {
    "intercept":                 -0.0360,
    "dem_pres_2p_baseline":       1.0521,
    "dem_incumbent":              0.0244,
    "rep_incumbent":              0.0097,
    "chamber_senate":             0.0110,
    "national_env":               None,
    "challenger_viability_flag":  0.0042,
    "dem_fundraising_share":      0.0,
    "white_col_centered":         0.1554,
    "sigma":                      0.0440,
}
REGRESSION_COEFFICIENTS_NO_FINANCE_EDU: dict[str, float] = {
    "intercept":            -0.0416,
    "dem_pres_2p_baseline":  1.0648,
    "dem_incumbent":         0.0231,
    "rep_incumbent":         0.0116,
    "chamber_senate":        0.0097,
    "national_env":          0.0046,
    "white_col_centered":    0.1603,
    "sigma":                 0.0232,
}
IE_COEFFICIENT_EDU: float = 0.0122
EDUCATION_TERM: bool = os.environ.get("TXLEG_EDU", "1") != "0"
if EDUCATION_TERM:
    REGRESSION_COEFFICIENTS.clear()
    REGRESSION_COEFFICIENTS.update(REGRESSION_COEFFICIENTS_EDU)
    REGRESSION_COEFFICIENTS_NO_FINANCE = dict(REGRESSION_COEFFICIENTS_NO_FINANCE_EDU)
    IE_COEFFICIENT = IE_COEFFICIENT_EDU
    print("  Education term on (midterm white-college term)")

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
# Re-estimated 2026-10-07 on the rebuilt WAR (scripts/war_persistence_estimate.py,
# 238 race pairs 2018-2024). How much past WAR carries forward depends on how
# many races it averages: 1 race 0.16, 2 races 0.62, 3+ races ~1.0+ (n=18).
# Reliability form beta(n) = n / (n + k), k = 2.93 pooled:
#   beta(1) 0.25, beta(2) 0.41, beta(3) 0.51, beta(4) 0.58.
# Leave-one-year-out it beats a single beta in all three years and the old
# flat 0.46 (estimated on the pre-refit WAR) in 2020 and 2022, losing 2024
# narrowly (2.48 vs 2.45pp RMSE).
#
# Applied only to incumbents found in data/processed/candidate_war.csv.
# For those districts, challenger_viability_flag and dem_fundraising_share
# are dropped from the baseline (WAR already incorporates fundraising ability).
WAR_PERSISTENCE_K: float = 2.93

# (TX_HISPANIC_ADJUSTMENT, the -0.05 x Hispanic-CVAP constant, was deleted
# 2026-10-07 together with the demographic level term in model.py; it existed
# to cancel a residual that term created. See model.build_linear_predictions.)

# ---------------------------------------------------------------------------
# Firebrand stand-in for non-incumbents (2026-10-07, Brennan)
# ---------------------------------------------------------------------------
# scripts/firebrand_penalty_test.py: Republican legislators with visible
# firebrand markers (Freedom Caucus, Texas Monthly Worst, speaker revolts,
# insurgent-PAC backing, viral confrontations) ran -0.9pp behind fundamentals
# in contested general elections 2018-2024 (t -2.8). For incumbents WAR already
# carries it. Non-incumbents have no WAR, so a flagged Republican nominee gets
# a small stand-in: about half the raw effect, because a non-incumbent's signal
# is weaker evidence. Applied to the D share (+ = toward the Democrat).
# Democrats: no penalty found (+1.0pp raw, n=13), so no D flag.
# Flags and their evidence: config/firebrand_nonincumbents_2026.csv.
FIREBRAND_STANDIN_PP: float = 0.005

# ---------------------------------------------------------------------------
# Correlated group-error layer (Monte Carlo), built 2026-10-07
# ---------------------------------------------------------------------------
# docs/poll-integration-proposal.md §3e. Per simulation, one error per racial
# group; each district moves by its CENTERED composition:
#     group_err_d = Σ_r (share_dr − w_TX[r]) · ε_r,   ε_r ~ N(0, σ_r)
# so a Hispanic miss moves the Valley together ("we lost all of it at once")
# instead of district by district. Centering keeps the uniform part of any miss
# in the national layer. Shares are normalized CVAP (other = Asian + AIAN +
# other); w_TX is the statewide CVAP mix, computed in model.py from the 2024 ACS.
#
# σ per spec "estimate, report next to the prior, use the larger"
# (scripts/group_error_estimate.py, five clean cycles 2014–2024, net of
# sampling noise; only differences from white are identified):
#   Black − white    estimate 1.9   prior √(4²+2.5²)=4.7   → prior
#   Hispanic − white estimate 3.5   prior √(4.5²+2.5²)=5.1 → prior
#   other − white    estimate 8.9   prior √(6²+2.5²)=6.5   → estimate (2018's
#                    Asian-suburb swing); σ_other = √(8.9²−2.5²) = 8.6
# The variance the layer adds is CARVED OUT of σ_idio, not stacked on it, so
# total district σ stays at REGRESSION_COEFFICIENTS["sigma"] on average.
#
# A FIXED midterm composition term was tested alongside (scripts/
# midterm_composition_test.py: midterm residuals do lean R with Black/Hispanic
# share) and rejected: worse out of sample on all 259 races (RMSE 3.37→3.65pp).
GROUP_ERROR_ENABLED: bool = True
GROUP_ERROR_SIGMA: dict[str, float] = {
    "white":    0.025,
    "black":    0.040,
    "hispanic": 0.045,
    "other":    0.086,
    # Used only when GROUP_ERROR_EDUCATION splits white by BA+. Within midterms
    # the college/non-college gap barely varies cycle to cycle (net estimate
    # ~0), so these are priors: non-college whites are the group national
    # polls missed in 2016 and 2020, hence the larger σ.
    "white_col":    0.025,
    "white_noncol": 0.035,
}
# Split the white group by education in the error layer. Adopted 2026-10-07
# (Brennan); TXLEG_EDU_ERR=0 turns it off for one run.
GROUP_ERROR_EDUCATION: bool = os.environ.get("TXLEG_EDU_ERR", "1") != "0"

# Monte Carlo simulation count
N_SIMULATIONS: int = 10_000

# Chamber control thresholds (seats needed for majority)
HOUSE_MAJORITY: int = 76   # out of 150
SENATE_MAJORITY: int = 16  # out of 31
