"""As-of-September-1 backtest of the demographic term (Brennan, 2026-09-09):
"we should be updating to late august/early september polls during that period
to compare to now."

The shipped backtests are "as of April". This re-runs the 2018 and 2022 cycles
with every polling input replaced by what was in hand on ~Sept 1 of that year:

  environment dial        RCP-style generic ballot average at Sept 1
  national race crosstabs polls fielded late Jul - mid Sep of that year
  Texas race crosstabs    UT/Texas Politics Project waves in hand on Sept 1
                          (Jun 2018 -- UT had no Aug 2018 wave; Aug 2022)
  same-pollster benchmark UT Oct 2016 / Oct 2020 likely-voter or RV banner

and then swaps ONLY the demographic term:
  L0  level + TX_HISPANIC_ADJUSTMENT (shipped formula, Sept-1 inputs)
  L1  level, constant 0
  N   no demographic term
  SN  centered shift, national-only delta  (national generic now - Catalist prior pres) [known ballot-mixed]
  SU  centered shift, same-pollster UT delta (UT generic now - UT generic prior Oct), other cell shrunk to 0
Everything else (presidential baseline, incumbency, finance/WAR, chamber) is
exactly what the shipped backtest computes. Run from the repo root.

Sources for every number are in data/raw/texas_crosstab_inputs.csv (Texas) and
the SOURCES dict below (national).
"""
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import backtest as bt                                  # noqa: E402
from backtest_config import BACKTEST_CONFIGS          # noqa: E402

GROUPS = ["white_nh", "black_nh", "hispanic", "other"]
COLS = {"white_nh": "pct_white_nh", "black_nh": "pct_black_nh",
        "hispanic": "pct_hispanic", "other": "pct_other"}

# Refit structural coefficients from branch refit-clean-cycles (STAGED, NOT ADOPTED).
REFIT = {"intercept": -0.0322, "dem_pres_2p_baseline": 0.9759, "dem_incumbent": 0.0394,
         "rep_incumbent": 0.0078, "chamber_senate": 0.0069, "national_env": 0.0036}

# Catalist two-way presidential D support, national (What Happened 2024 figs 10/14/18/23).
CATALIST_PRES = {2016: {"white_nh": 0.41, "black_nh": 0.93, "hispanic": 0.70, "other": 0.70},
                 2020: {"white_nh": 0.44, "black_nh": 0.89, "hispanic": 0.63, "other": 0.65}}

SOURCES = {
    2018: {
        # RCP average ran ~D+7 through Sept 2018 (RCP 2018-10-03 "Will 2018 Be a Wave
        # Election?"); 538 was ~1pp higher. Using +7.5 with a +/-0.5 caveat.
        "env_dial_sept1": 7.5,
        # National generic ballot by race in hand around Sept 1 2018 (2p D):
        #   Pew Sept 18-24 2018 RV: W 46/49, H 63/29, B 77/16  (pewresearch.org/politics/2018/09/26)
        #   4-poll Mar-Jun 2018 avg already in backtest_config (HH Apr/May/Jun + Pew Jun)
        # Harvard-Harris Aug/Sept 2018 books carry no generic-ballot table (checked).
        "national_polls": {
            "Pew Sep18-24 2018": {"white_nh": 0.484, "black_nh": 0.828, "hispanic": 0.685, "other": None},
            "Mar-Jun 2018 4-poll avg": {"white_nh": 0.460, "black_nh": 0.884, "hispanic": 0.677, "other": 0.594},
        },
        # Texas: UT Jun 2018 RV Texas Legislature generic (p314) -- the wave in hand on Sept 1.
        "tx_now": {"white_nh": 0.3758, "black_nh": 0.9339, "hispanic": 0.5879, "other": None},
        # Same-pollster benchmark: UT Oct 2016 LV generic (p301).
        "tx_prior": {"white_nh": 0.3101, "black_nh": 0.9081, "hispanic": 0.6505, "other": None},
        # Approximate 2016 Texas electorate for centering.
        "tx_weights": {"white_nh": 0.62, "black_nh": 0.12, "hispanic": 0.20, "other": 0.06},
        "prior_pres": 2016,
    },
    2022: {
        # RCP: Democrats retook the lead on Sept 1 2022 (i.e. ~D+0); 538 ~D+1.
        "env_dial_sept1": 0.5,
        # National generic ballot by race, late Jul - early Sep 2022 (2p D):
        #   Harvard-Harris Jul 27-28 RV: W 42/58, H 62/38, B 76/24, Other 52/48
        #   Pew Aug 1-14 RV:            W 38/51, B 70/6, H 53/28
        #   Economist/YouGov Aug 28-30: B 71/9, H 53/32; White from 4 white cells (N-weighted) ~41/44
        #   Harvard-Harris Sep 7-8 RV:  W 45/55, H 63/37, B 72/28, Other 65/35
        "national_polls": {
            "HH Jul27-28 2022": {"white_nh": 0.420, "black_nh": 0.760, "hispanic": 0.620, "other": 0.520},
            "Pew Aug1-14 2022": {"white_nh": 0.427, "black_nh": 0.921, "hispanic": 0.654, "other": None},
            "YouGov Aug28-30 2022": {"white_nh": 0.485, "black_nh": 0.888, "hispanic": 0.624, "other": None},
            "HH Sep7-8 2022": {"white_nh": 0.450, "black_nh": 0.720, "hispanic": 0.630, "other": 0.650},
        },
        # Texas: UT Aug 2022 RV Texas Legislature generic (Q21B) -- in window.
        "tx_now": {"white_nh": 0.3516, "black_nh": 0.8780, "hispanic": 0.5843, "other": 0.5366},
        # Same-pollster benchmark: UT Oct 2020 RV Texas Legislature generic (p62).
        "tx_prior": {"white_nh": 0.3511, "black_nh": 0.8485, "hispanic": 0.5824, "other": 0.4810},
        "tx_weights": {"white_nh": 0.58, "black_nh": 0.12, "hispanic": 0.24, "other": 0.06},
        "prior_pres": 2020,
    },
}


def avg_polls(polls, fallback):
    out = {}
    for g in GROUPS:
        vals = [p[g] for p in polls.values() if p.get(g) is not None]
        out[g] = float(np.mean(vals)) if vals else fallback[g]
    return out


def shares(df):
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
    is_house = (df["chamber_lower"] == "house").values
    wp_all = norm.cdf(pred, loc=0.5, scale=0.0785)
    exp_house = wp_all[is_house].sum()
    act_house = (df.loc[is_house, "actual_winner_party"] == "D").sum()
    hisp = np.array([bt.safe_pct(v) for v in d["pct_hispanic"]])
    slope = np.polyfit(hisp, resid, 1)[0]
    print("  %-40s house_err %+6.1f  brier %.4f  acc %5.1f%%  mean_resid %+.2fpp  resid~hisp slope %+.3f"
          % (label, exp_house - act_house, brier, acc * 100, resid.mean() * 100, slope))


def run_cycle(year):
    cfg = dict(BACKTEST_CONFIGS[year])
    src = SOURCES[year]
    nat_now = avg_polls(src["national_polls"], cfg["race_generic_ballot_d_share"])
    cfg["race_generic_ballot_d_share"] = nat_now
    cfg["env_dial"] = src["env_dial_sept1"]

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
    l0 = pd.concat(preds, ignore_index=True).values.astype(float)

    nw = cfg["national_demo_weights"]
    nat_avg = sum(nw[g] * nat_now[g] for g in GROUPS)
    sh, ok = shares(df)
    level = np.where(ok, np.nansum(sh * np.array([nat_now[g] for g in GROUPS]), axis=1) - nat_avg, 0.0)
    hisp = np.array([bt.safe_pct(v) for v in df["pct_hispanic"]])
    const = cfg.get("tx_hispanic_adjustment", 0) * hisp
    base = l0 - level - const

    # shift, national-only (ballot-mixed; expected to fail)
    prior = CATALIST_PRES[src["prior_pres"]]
    d_nat = np.array([nat_now[g] - prior[g] for g in GROUPS])
    w_tx = np.array([src["tx_weights"][g] for g in GROUPS])
    shift_nat = np.where(ok, np.nansum(sh * d_nat, axis=1), 0.0) - (w_tx * d_nat).sum()
    # shift, same-pollster UT (other cell -> 0 when missing or tiny-n)
    d_ut = np.array([(src["tx_now"][g] - src["tx_prior"][g])
                     if (src["tx_now"].get(g) is not None and src["tx_prior"].get(g) is not None and g != "other")
                     else 0.0 for g in GROUPS])
    shift_ut = np.where(ok, np.nansum(sh * d_ut, axis=1), 0.0) - (w_tx * d_ut).sum()

    n_c = int(((df["contested"] == True) & df["actual_dem_2p_share"].notna()).sum())
    print("\n=== %d AS OF SEPT 1  (env %+.1f; %d districts, %d contested) ===" % (year, cfg["env_dial"], len(df), n_c))
    print("  national by race (Sept-1 avg): %s" % {g: round(nat_now[g], 3) for g in GROUPS})
    print("  UT same-pollster delta (pp):   %s" % {g: round(float(x) * 100, 1) for g, x in zip(GROUPS, d_ut)})
    print("  national-only delta (pp):      %s" % {g: round(float(x) * 100, 1) for g, x in zip(GROUPS, d_nat)})
    print("  level term mean %+.2fpp | shift_ut mean %+.2f sd %.2f | shift_nat mean %+.2f sd %.2f"
          % (level.mean() * 100, shift_ut.mean() * 100, shift_ut.std() * 100, shift_nat.mean() * 100, shift_nat.std() * 100))

    c = cfg["regression_coefficients"]
    print("  -- config coefficients (intercept %.4f, pass %.4f) --" % (c["intercept"], c["dem_pres_2p_baseline"]))
    metrics(df, base + level + const, "L0 level + constant (shipped formula)")
    metrics(df, base + level, "L1 level, constant 0")
    metrics(df, base, "N  no demo term")
    metrics(df, base + shift_nat, "SN shift, national-only delta")
    metrics(df, base + shift_ut, "SU shift, same-pollster UT delta")

    pres = pd.to_numeric(df["dem_pres_2p_baseline"], errors="coerce").fillna(nat_avg).values
    sen = (df["chamber_lower"] == "senate").astype(int).values
    dinc, rinc = df["dem_incumbent"].values.astype(float), df["rep_incumbent"].values.astype(float)
    old = (c["intercept"] + c["dem_pres_2p_baseline"] * pres + c["dem_incumbent"] * dinc
           + c["rep_incumbent"] * rinc + c["chamber_senate"] * sen + c["national_env"] * cfg["env_dial"])
    r = REFIT
    new = (r["intercept"] + r["dem_pres_2p_baseline"] * pres + r["dem_incumbent"] * dinc
           + r["rep_incumbent"] * rinc + r["chamber_senate"] * sen + r["national_env"] * cfg["env_dial"])
    base_r = base - old + new
    print("  -- refit structural coefficients (branch refit-clean-cycles); finance/WAR terms unchanged --")
    metrics(df, base_r + level + const, "R-L0 level + constant")
    metrics(df, base_r + level, "R-L1 level, constant 0")
    metrics(df, base_r, "R-N  no demo term")
    metrics(df, base_r + shift_nat, "R-SN shift, national-only delta")
    metrics(df, base_r + shift_ut, "R-SU shift, same-pollster UT delta")


if __name__ == "__main__":
    for y in (2018, 2022):
        run_cycle(y)
