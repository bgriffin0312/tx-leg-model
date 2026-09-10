"""Falsification test for docs/poll-integration-proposal.md section 3f step 2 (results in section 5).

Run from the repo root. The R-rows use model_config.REGRESSION_COEFFICIENTS and only mean
what the doc says on branch refit-clean-cycles; on master they are the old coefficients.

Same backtest data, same coefficients, swap ONLY the demographic term:
  V0  level term + TX_HISPANIC_ADJUSTMENT (shipped)
  V1  level term, constant = 0
  V2  centered SHIFT term (national-only delta), constant = 0, centered on config national weights
  V3  centered SHIFT term, centered on approximate Texas electorate weights
  V4  no demographic term at all (null)
Reported: house seat error (deterministic CDF, no MC noise), Brier, accuracy,
mean signed residual on contested races, and the slope of residual on Hispanic
CVAP share -- the diagnostic the constant was originally fit on.
"""
import io
import contextlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import backtest as bt                                  # noqa: E402
from backtest_config import BACKTEST_CONFIGS          # noqa: E402
import model_config as mc                             # noqa: E402

# Catalist two-way presidential D support, national (What Happened 2024, figs 10/14/18/23)
CATALIST_PRES = {
    2016: {"white_nh": 0.41, "black_nh": 0.93, "hispanic": 0.70, "other": 0.70},
    2020: {"white_nh": 0.44, "black_nh": 0.89, "hispanic": 0.63, "other": 0.65},
    2024: {"white_nh": 0.42, "black_nh": 0.85, "hispanic": 0.54, "other": 0.61},
}
# Approximate Texas electorate by race for the prior presidential year (validated-ish)
TX_ELECTORATE = {
    2016: {"white_nh": 0.62, "black_nh": 0.12, "hispanic": 0.20, "other": 0.06},
    2020: {"white_nh": 0.58, "black_nh": 0.12, "hispanic": 0.24, "other": 0.06},
}
GROUPS = ["white_nh", "black_nh", "hispanic", "other"]
COLS = {"white_nh": "pct_white_nh", "black_nh": "pct_black_nh",
        "hispanic": "pct_hispanic", "other": "pct_other"}


def shares(df):
    """Normalized CVAP shares as in compute_demo_baseline."""
    m = np.array([[bt.safe_pct(row.get(COLS[g])) for g in GROUPS] for _, row in df.iterrows()])
    tot = m.sum(axis=1, keepdims=True)
    ok = tot[:, 0] > 0
    out = np.full_like(m, np.nan)
    out[ok] = m[ok] / tot[ok]
    return out, ok


def metrics(df, pred, label):
    contested = (df["contested"] == True) & df["actual_dem_2p_share"].notna()
    d = df[contested]
    p = pred[contested.values]
    resid = p - d["actual_dem_2p_share"].values
    win = norm.cdf(p, loc=0.5, scale=0.0785)
    actual_win = (d["actual_dem_2p_share"].values > 0.5).astype(float)
    brier = np.mean((win - actual_win) ** 2)
    acc = np.mean((win > 0.5) == (actual_win > 0.5))
    # seat error, deterministic
    is_house = (df["chamber_lower"] == "house").values
    wp_all = norm.cdf(pred, loc=0.5, scale=0.0785)
    exp_house = wp_all[is_house].sum()
    act_house = (df.loc[is_house, "actual_winner_party"] == "D").sum()
    hisp = np.array([bt.safe_pct(v) for v in d["pct_hispanic"]])
    slope = np.polyfit(hisp, resid, 1)[0] if len(hisp) > 5 else np.nan
    print("  %-46s house_err %+6.1f  brier %.4f  acc %5.1f%%  mean_resid %+.2fpp  resid~hisp slope %+.3f"
          % (label, exp_house - act_house, brier, acc * 100, resid.mean() * 100, slope))


def run_cycle(year):
    cfg = BACKTEST_CONFIGS[year]
    dfs, preds = [], []
    for chamber in ("house", "senate"):
        with contextlib.redirect_stdout(io.StringIO()):
            df = bt.build_backtest_df(cfg, chamber, verbose=False)
        if df is None or df.empty or df["dem_pres_2p_baseline"].notna().sum() == 0:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            lin = bt.build_linear_predictions(df, cfg)
        dfs.append(df); preds.append(lin)
    df = pd.concat(dfs, ignore_index=True)
    v0 = pd.concat(preds, ignore_index=True).values.astype(float)

    rg, nw = cfg["race_generic_ballot_d_share"], cfg["national_demo_weights"]
    nat_avg = sum(nw[g] * rg[g] for g in GROUPS)
    sh, ok = shares(df)
    level = np.where(ok, np.nansum(sh * np.array([rg[g] for g in GROUPS]), axis=1) - nat_avg, 0.0)
    hisp = np.array([bt.safe_pct(v) for v in df["pct_hispanic"]])
    const = cfg.get("tx_hispanic_adjustment", 0) * hisp

    base = v0 - level - const                      # structural prediction, no demo term
    prior = CATALIST_PRES[cfg["pres_year"]]
    delta = np.array([rg[g] - prior[g] for g in GROUPS])
    raw_shift = np.where(ok, np.nansum(sh * delta, axis=1), 0.0)
    w_nat = np.array([nw[g] for g in GROUPS])
    w_tx = np.array([TX_ELECTORATE[cfg["pres_year"]][g] for g in GROUPS])
    shift_nat = raw_shift - (w_nat * delta).sum()
    shift_tx = raw_shift - (w_tx * delta).sum()

    print("\n=== %d backtest  (pres %d, %d districts, %d contested) ===" % (
        year, cfg["pres_year"], len(df), int(((df["contested"] == True) & df["actual_dem_2p_share"].notna()).sum())))
    print("  group delta (poll/actual minus Catalist %d pres): %s" % (
        cfg["pres_year"], {g: round(float(x) * 100, 1) for g, x in zip(GROUPS, delta)}))
    print("  level term: mean %+.2fpp   shift(nat-centered): mean %+.2fpp sd %.2f   shift(TX-centered): mean %+.2fpp"
          % (level.mean() * 100, shift_nat.mean() * 100, shift_nat.std() * 100, shift_tx.mean() * 100))
    print("  -- config coefficients (intercept %.4f, pass-through %.4f) --" % (
        cfg["regression_coefficients"]["intercept"], cfg["regression_coefficients"]["dem_pres_2p_baseline"]))
    metrics(df, v0, "V0 level + constant %.2f (shipped)" % cfg.get("tx_hispanic_adjustment", 0))
    metrics(df, base + level, "V1 level, constant 0")
    metrics(df, base + shift_nat, "V2 shift, national-centered")
    metrics(df, base + shift_tx, "V3 shift, Texas-centered")
    metrics(df, base, "V4 no demo term")

    # Refit structural coefficients (branch refit-clean-cycles), swapping only the
    # structural terms; finance/WAR terms stay as the config computed them.
    c, r = cfg["regression_coefficients"], mc.REGRESSION_COEFFICIENTS
    pres = pd.to_numeric(df["dem_pres_2p_baseline"], errors="coerce").fillna(nat_avg).values
    sen = (df["chamber_lower"] == "senate").astype(int).values
    dinc, rinc = df["dem_incumbent"].values.astype(float), df["rep_incumbent"].values.astype(float)
    old = (c["intercept"] + c["dem_pres_2p_baseline"] * pres + c["dem_incumbent"] * dinc
           + c["rep_incumbent"] * rinc + c["chamber_senate"] * sen + c["national_env"] * cfg["env_dial"])
    new = (r["intercept"] + r["dem_pres_2p_baseline"] * pres + r["dem_incumbent"] * dinc
           + r["rep_incumbent"] * rinc + r["chamber_senate"] * sen + r["national_env"] * cfg["env_dial"])
    base_r = base - old + new
    print("  -- refit structural coefficients (intercept %.4f, pass-through %.4f); finance/WAR terms unchanged --"
          % (r["intercept"], r["dem_pres_2p_baseline"]))
    metrics(df, base_r + level + const, "R0 level + constant")
    metrics(df, base_r + level, "R1 level, constant 0")
    metrics(df, base_r + shift_nat, "R2 shift, national-centered")
    metrics(df, base_r + shift_tx, "R3 shift, Texas-centered")
    metrics(df, base_r, "R4 no demo term")


for y in (2018, 2022):
    run_cycle(y)

# Correction for the proposal doc: Catalist 2024 white is .42, not .46
print("\n2026 national-only delta with corrected Catalist 2024 white (.42):")
nat_now = {"white_nh": 0.475, "black_nh": 0.846, "hispanic": 0.618, "other": 0.60}
print("  ", {g: round((nat_now[g] - CATALIST_PRES[2024][g]) * 100, 1) for g in GROUPS})
