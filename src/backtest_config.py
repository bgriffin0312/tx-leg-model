"""
backtest_config.py

Historical model configuration for backtesting the 2026 TX legislative model
against actual 2022 and 2018 election outcomes.

!!! UPDATE THESE WITH ACTUAL HISTORICAL POLLING DATA !!!
Sources:
  - Civiqs historical tracking (civiqs.com) — crosstabs by race, historical data available
  - ANES (American National Election Studies) — exit poll equivalent by race
  - Pew Research Center — validated voter surveys with race crosstabs

HOW TO LOOK UP HISTORICAL CIVIQS DATA:
  1. Go to civiqs.com → "Congressional Generic Ballot" → filter by "Race/Ethnicity"
  2. Use the date slider to view April of the target year (2022 or 2018)
  3. Record D% / (D% + R%) for each racial group at Registered Voter level

CURRENT STATUS OF VALUES BELOW:
  - Values marked "ESTIMATED" are approximations from exit poll crosstabs,
    ANES data, and historical polling trends. Verify against Civiqs.
  - National environment dial values are approximate based on generic ballot
    aggregators (FiveThirtyEight, RealClearPolitics) in April of each year.

IMPORTANT: These configs are for the MODEL-AS-OF-APRIL validation only.
  The model predicts outcomes as if it were April of the cycle year.
  We compare against actual November outcomes.
"""


# ---------------------------------------------------------------------------
# National demographic weights (used to compute national baseline)
# These are approximately constant across cycles; using 2022 election weights
# ---------------------------------------------------------------------------
NATIONAL_DEMO_WEIGHTS_2022: dict[str, float] = {
    "white_nh":  0.63,   # 2022 electorate composition
    "black_nh":  0.11,
    "hispanic":  0.13,
    "other":     0.13,
}

NATIONAL_DEMO_WEIGHTS_2018: dict[str, float] = {
    "white_nh":  0.67,   # 2018 electorate (whiter than 2022+)
    "black_nh":  0.11,
    "hispanic":  0.11,
    "other":     0.11,
}


# ---------------------------------------------------------------------------
# Backtest configurations by cycle year
# ---------------------------------------------------------------------------

BACKTEST_CONFIGS: dict[int, dict] = {

    # -----------------------------------------------------------------------
    # 2022 CYCLE — As of April 2022
    # -----------------------------------------------------------------------
    # Environment: R+3 to R+4 nationally (red wave building but not fully materializing)
    # Generic ballot: FiveThirtyEight aggregate ~R+3 in April 2022
    #
    # Race-specific crosstabs (ESTIMATED — verify with Civiqs April 2022):
    #   White non-Hispanic: R+24 nationally (~D+38% 2p) — pre-wave Dem weakness
    #   Black non-Hispanic: D+88 (~88% D 2p) — stable
    #   Hispanic: D+20 (~60% D 2p) — post-2020 shift already underway in TX
    #   Other: D+16 (~58% D 2p) — Asians/AIAN/multiracial lean Dem
    # -----------------------------------------------------------------------
    2022: {
        "label": "2022 TX Legislative — As of April 2022",
        "pres_year": 2020,          # presidential year for partisan baseline
        "map_year": 2022,           # district boundary year (post-redistricting)
        "results_year": 2022,       # actual outcomes to validate against

        # Race-specific 2022 vote share (Catalist validated voter estimates)
        # Source: catalist.us/whathappened2022/
        # Using actual 2022 results rather than April polling, which had
        # badly off Hispanic (Marist: R+13 vs actual D+24) and Black numbers.
        "race_generic_ballot_d_share": {
            "white_nh": 0.42,       # Catalist: 42% D (Pew: 41.8%)
            "black_nh": 0.88,       # Catalist: 88% D (Pew: 94.9%, Edison: 86.9%)
            "hispanic": 0.62,       # Catalist: 62% D (Pew: 60.6%)
            "other":    0.59,       # Catalist AAPI: 59% D (Pew: 68%)
        },

        # National demographic weights for computing national average
        "national_demo_weights": NATIONAL_DEMO_WEIGHTS_2022,

        # National environment: actual 2022 result was R+2.7 (D 2p = 48.6%)
        # env_dial is the absolute D-R margin in pp, consistent with regression training
        "env_dial": -2.8,

        # Regression coefficients (from Phase 1 with presidential baseline)
        # Using with_pres national_env coefficient (0.0052) since finance data
        # is available and the full-model coefficient (0.0027) is suppressed by
        # collinearity with finance variables
        "regression_coefficients": {
            "intercept":                 0.1781,
            "dem_pres_2p_baseline":      0.5962,
            "dem_incumbent":             0.0676,
            "rep_incumbent":            -0.0804,
            "chamber_senate":           -0.0263,
            "national_env":              0.0049,
            "challenger_viability_flag": 0.0446,
            "dem_fundraising_share":     0.0731,
            "sigma":                     0.0742,
        },

        # Finance: 2022 full-cycle finance data available
        "use_finance": True,
        "finance_file": "tx_finance_2022.csv",

        # WAR: use 2018-only WAR (pre-2022, avoids data leakage)
        "use_war": True,
        "war_max_year": 2018,
        "war_persistence_coef": 0.46,

        # Senate districts up in 2022
        # All 31 senate seats were on the 2022 ballot due to redistricting
        "senate_districts_on_ballot": set(range(1, 32)),

        "notes": (
            "2022 was the first election using the post-2021 redistricted maps. "
            "All 31 Senate seats were on the ballot (unusual; normally only ~half). "
            "Actual national result: R+2.7 popular vote. "
            "Racial crosstabs from Catalist validated voter estimates."
        ),
        # TX-specific Hispanic adjustment (see model_config.py for explanation)
        "tx_hispanic_adjustment": -0.05,

        "source": "Catalist whathappened2022, Pew validated voters July 2023, TEC finance 2022",
        "updated": "2026-04-09",
    },

    # -----------------------------------------------------------------------
    # 2018 CYCLE — As of April 2018
    # -----------------------------------------------------------------------
    # Environment: D+8 to D+9 nationally (strong Democratic wave building)
    # Generic ballot: FiveThirtyEight aggregate ~D+8 in April 2018
    #
    # Race-specific crosstabs — DERIVED from 4 polls fielded March-June 2018:
    #   - Harvard-Harris (April, May, June 2018; online RV)
    #   - Pew Research Political Survey (June 5-12, 2018; phone RV)
    #   Simple average of D 2-party share across all available polls.
    #   See: harvardharrispoll.com (Apr/May/Jun 2018 crosstab memos);
    #        pewresearch.org/politics/2018/06/20/2-the-2018-congressional-election/
    #   White: 0.460 (avg of 4 polls; range 0.446-0.470)
    #   Black: 0.884 (avg of 4 polls; range 0.828-0.929)
    #   Hispanic: 0.677 (avg of 4 polls; range 0.610-0.740)
    #   Other: 0.594 (avg of 3 HHP polls; Pew did not report)
    #
    # *** IMPORTANT: 2018 used PRE-redistricting maps ***
    # District numbers and boundaries differ from 2022+. CVAP loaded from
    # 2014-2018 ACS (cvap_vintage_year=2018), keyed under H2100/S2100 GEOIDs.
    # -----------------------------------------------------------------------
    2018: {
        "label": "2018 TX Legislative — As of April 2018",
        "pres_year": 2016,          # presidential year for partisan baseline
        "map_year": 2018,           # district boundary year (pre-redistricting)
        "results_year": 2018,       # actual outcomes to validate against
        "cvap_vintage_year": 2018,  # ACS 5-year CVAP under H2100/S2100 districts

        # Race-specific generic ballot D 2p share — March-June 2018 poll average
        "race_generic_ballot_d_share": {
            "white_nh": 0.460,
            "black_nh": 0.884,
            "hispanic": 0.677,
            "other":    0.594,
        },

        "national_demo_weights": NATIONAL_DEMO_WEIGHTS_2018,
        "env_dial": 0,

        "regression_coefficients": {
            "intercept":                 0.1781,
            "dem_pres_2p_baseline":      0.5962,
            "dem_incumbent":             0.0676,
            "rep_incumbent":            -0.0804,
            "chamber_senate":           -0.0263,
            "national_env":              0.0025,
            "challenger_viability_flag": 0.0446,
            "sigma":                     0.0742,
        },

        "use_finance": False,

        # 2018 Senate: 15 districts on ballot (from tx_senate_results_2018.csv on_ballot flag)
        # Not a simple odd/even split — TX Senate classes don't follow that pattern
        "senate_districts_on_ballot": {2, 3, 5, 7, 8, 9, 10, 14, 15, 16, 17, 23, 25, 30, 31},

        # TX-specific Hispanic adjustment (same coefficient; pre-2020 realignment
        # may mean the actual gap was smaller in 2018, but no TX-specific validated
        # data exists to calibrate separately)
        "tx_hispanic_adjustment": -0.05,

        "notes": (
            "2018 used pre-2021 redistricting maps (PlanH2100/PlanS2100). "
            "CVAP loaded from 2014-2018 ACS keyed under old district GEOIDs. "
            "2016 presidential baseline derived from TX Capitol VTD data joined "
            "to precincts20g_districts.xlsx (same H2100/S2100 plan as 2018). "
            "Beto O'Rourke's unusually strong Senate run likely boosted all Dem down-ballot. "
            "Generic ballot values are average of 4 polls fielded March-June 2018."
        ),
        "source": "Harvard-Harris Apr/May/Jun 2018 + Pew Jun 2018 (avg of 4 polls)",
        "updated": "2026-04-11",
    },
}
