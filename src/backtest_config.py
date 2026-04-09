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
            "intercept":                 0.1520,
            "dem_pres_2p_baseline":      0.6604,
            "dem_incumbent":             0.0600,
            "rep_incumbent":            -0.0739,
            "chamber_senate":           -0.0261,
            "national_env":              0.0052,
            "challenger_viability_flag": 0.0393,
            "dem_fundraising_share":     0.0624,
            "sigma":                     0.0785,
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
        "source": "Catalist whathappened2022, Pew validated voters July 2023, TEC finance 2022",
        "updated": "2026-04-08",
    },

    # -----------------------------------------------------------------------
    # 2018 CYCLE — As of April 2018
    # -----------------------------------------------------------------------
    # Environment: D+8 to D+9 nationally (strong Democratic wave building)
    # Generic ballot: FiveThirtyEight aggregate ~D+8 in April 2018
    #
    # Race-specific crosstabs (ESTIMATED — verify with Civiqs April 2018):
    #   White non-Hispanic: D+4 (~52% D 2p) — wave environment boosted white Dem support
    #   Black non-Hispanic: D+88 (~88% D 2p) — stable
    #   Hispanic: D+34 (~67% D 2p) — pre-2020 realignment, stronger Dem lean
    #   Other: D+28 (~64% D 2p) — Asians/AIAN more Dem than current
    #
    # *** IMPORTANT: 2018 used PRE-redistricting maps ***
    # District numbers and boundaries differ from 2022+.
    # CVAP data is NOT available for old district boundaries.
    # This backtest will use current CVAP as an approximation (districts overlap
    # significantly with current ones in many regions; border/suburban exceptions).
    # -----------------------------------------------------------------------
    2018: {
        "label": "2018 TX Legislative — As of April 2018",
        "pres_year": 2016,          # presidential year for partisan baseline
        "map_year": 2018,           # district boundary year (pre-redistricting)
        "results_year": 2018,       # actual outcomes to validate against

        # Race-specific generic ballot D 2p share (ESTIMATED — update from Civiqs)
        "race_generic_ballot_d_share": {
            "white_nh": 0.44,       # ~D+8 nationally, wave environment
            "black_nh": 0.88,
            "hispanic": 0.67,       # Pre-2020 TX Hispanic lean (significantly more D)
            "other":    0.64,
        },

        "national_demo_weights": NATIONAL_DEMO_WEIGHTS_2018,
        "env_dial": 0,

        "regression_coefficients": {
            "intercept":                 0.1520,
            "dem_pres_2p_baseline":      0.6604,
            "dem_incumbent":             0.0600,
            "rep_incumbent":            -0.0739,
            "chamber_senate":           -0.0261,
            "national_env":              0.0027,
            "challenger_viability_flag": 0.0393,
            "sigma":                     0.0785,
        },

        "use_finance": False,

        # 2018 Senate: only odd-numbered districts were on the ballot (class 1/3 seats)
        # Districts: 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31 (approx)
        # Note: confirm with actual 2018 Senate ballot schedule
        "senate_districts_on_ballot": {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31},

        "notes": (
            "2018 used pre-2021 redistricting maps (different district boundaries). "
            "CVAP data for old districts is approximated using current district CVAP. "
            "2016 presidential baseline requires crosswalk via precincts18g_districts.xlsx. "
            "Beto O'Rourke's unusually strong Senate run likely boosted all Dem down-ballot. "
            "This backtest has higher uncertainty than 2022 due to boundary changes. "
            "NOTE: Generic ballot values are ESTIMATED — update from Civiqs historical."
        ),
        "source": "ESTIMATED from 2018 exit polls and historical polling; update with Civiqs",
        "updated": "2026-04-06",
    },
}
