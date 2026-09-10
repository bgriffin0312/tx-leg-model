"""Baseline comparison: presidential vs Railroad Commissioner vs judicial mean.

For each candidate baseline, fit the model's structural regression
    dem_2p_share ~ baseline + dem_incumbent + rep_incumbent + chamber_senate + national_env
on the contested races of the clean midterm cycles (2014, 2018, 2022),
leave-one-cycle-out, and score the held-out cycle. Same rows, same covariates,
same sigma for every baseline, so the only thing that varies is the baseline.

Baselines:
    pres          dem_pres_2p_baseline as shipped in phase1_dataset.csv
    pres_spatial  President from the same spatial files as the others (method control)
    rrc           Railroad Commissioner
    judicial      mean of Supreme Court + Court of Criminal Appeals contests
    downballot    mean of judicial + RRC

Inputs: data/processed/phase1_dataset.csv plus
        data/raw/historical/tx_downballot_{chamber}_{year}_plan{plan}.csv
        (src/collect_downballot_spatial.py).
Output: output/baseline_backtest.csv and a printed table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
HIST = RAW / "historical"
OUT = ROOT / "output"

SIGMA = 0.0742  # backtest_config sigma; identical for every baseline
CYCLES = [2014, 2018, 2022]
DOWNBALLOT = {
    (2014, "house"): "tx_downballot_house_2012_planh358.csv",
    (2014, "senate"): "tx_downballot_senate_2012_plans172.csv",
    (2018, "house"): "tx_downballot_house_2016_planh2100.csv",
    (2018, "senate"): "tx_downballot_senate_2016_plans172.csv",
    (2022, "house"): "tx_downballot_house_2020_planh2316.csv",
    (2022, "senate"): "tx_downballot_senate_2020_plans2168.csv",
}
CVAP = {
    (2014, "house"): "historical/tx_cvap_house_2018.csv",
    (2018, "house"): "historical/tx_cvap_house_2018.csv",
    (2022, "house"): "tx_cvap_house.csv",
    (2014, "senate"): "historical/tx_cvap_senate_2018.csv",
    (2018, "senate"): "historical/tx_cvap_senate_2018.csv",
    (2022, "senate"): "tx_cvap_senate.csv",
}
BASELINES = {
    "pres": "dem_pres_2p_baseline",
    "pres_spatial": "pres_d2p",
    "rrc": "rrc_d2p",
    "judicial": "judicial_mean_d2p",
    "downballot": "downballot_mean_d2p",
    "blend": "blend_pres_rrc",
}
COVARS = ["dem_incumbent", "rep_incumbent", "chamber_senate", "national_env"]


def load():
    df = pd.read_csv(PROC / "phase1_dataset.csv")
    df["chamber_lower"] = df["chamber"].str.lower()
    df = df[df["year"].isin(CYCLES)].copy()
    parts = []
    for (year, chamber), fname in DOWNBALLOT.items():
        p = HIST / fname
        if not p.exists():
            print(f"  WARN missing {fname}; {year} {chamber} rows will lack downballot baselines")
            continue
        d = pd.read_csv(p)[["district", "rrc_d2p", "judicial_mean_d2p",
                            "downballot_mean_d2p", "pres_d2p", "n_judicial"]]
        d["year"] = year
        d["chamber_lower"] = chamber
        parts.append(d)
    db = pd.concat(parts, ignore_index=True)
    df = df.merge(db, on=["year", "chamber_lower", "district"], how="left")
    cv_parts = []
    for (year, chamber), rel in CVAP.items():
        c = pd.read_csv(RAW / rel)
        c = c[c["chamber"].astype(str).str.lower() == chamber][["district", "pct_hispanic"]].copy()
        c["pct_hispanic"] = pd.to_numeric(c["pct_hispanic"], errors="coerce")
        c["pct_hispanic"] = c["pct_hispanic"].where(c["pct_hispanic"] <= 1, c["pct_hispanic"] / 100)
        c["year"] = year
        c["chamber_lower"] = chamber
        cv_parts.append(c)
    df = df.merge(pd.concat(cv_parts, ignore_index=True), on=["year", "chamber_lower", "district"], how="left")
    for c in ["dem_2p_share", "dem_incumbent", "rep_incumbent", "chamber_senate", "national_env"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Cautious middle: half presidential, half Railroad Commissioner.
    df["blend_pres_rrc"] = 0.5 * (df["pres_d2p"] + df["rrc_d2p"])
    return df


def fit(train, xcol):
    X = np.column_stack([np.ones(len(train)), train[xcol].values] + [train[c].values for c in COVARS])
    y = train["dem_2p_share"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid.std(ddof=X.shape[1])


def predict(test, xcol, beta):
    X = np.column_stack([np.ones(len(test)), test[xcol].values] + [test[c].values for c in COVARS])
    return X @ beta


def score(test, pred, label, year, xcol):
    actual = test["dem_2p_share"].values
    resid = pred - actual
    win = norm.cdf(pred, loc=0.5, scale=SIGMA)
    actual_win = (actual > 0.5).astype(float)
    house = (test["chamber_lower"] == "house").values
    hisp = test["pct_hispanic"].values
    row = {
        "baseline": label, "cycle": year, "n": len(test),
        "resid_sd": resid.std(), "mean_resid_pp": resid.mean() * 100,
        "mae_pp": np.abs(resid).mean() * 100,
        "brier": np.mean((win - actual_win) ** 2),
        "house_seat_err": win[house].sum() - actual_win[house].sum(),
        "house_n": int(house.sum()),
        "resid_hisp_slope": np.polyfit(hisp[~np.isnan(hisp)], resid[~np.isnan(hisp)], 1)[0],
    }
    for lo, hi, name in [(0, .3, "h0_30"), (.3, .5, "h30_50"), (.5, .7, "h50_70"), (.7, 1.01, "h70_100")]:
        m = (hisp >= lo) & (hisp < hi)
        row[f"res_{name}_pp"] = resid[m].mean() * 100 if m.any() else np.nan
        row[f"n_{name}"] = int(m.sum())
    return row


def main():
    pd.set_option("display.width", 250)
    df = load()
    sample = df[(df["contested"] == True) & (df["on_ballot"] == True) & df["dem_2p_share"].notna()].copy()
    rows, coefs = [], []
    for label, xcol in BASELINES.items():
        s = sample[sample[xcol].notna()].copy()
        if len(s) < len(sample):
            print(f"  {label}: {len(sample) - len(s)} contested rows lack this baseline and are dropped")
        for year in CYCLES:
            train, test = s[s["year"] != year], s[s["year"] == year]
            if test.empty:
                continue
            beta, sig = fit(train, xcol)
            rows.append(score(test, predict(test, xcol, beta), label, year, xcol)
                        | {"train_pass_through": beta[1], "train_sigma": sig})
        beta, sig = fit(s, xcol)
        coefs.append({"baseline": label, "n": len(s), "intercept": beta[0], "pass_through": beta[1],
                      "dem_inc": beta[2], "rep_inc": beta[3], "senate": beta[4], "env": beta[5], "sigma": sig})
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "baseline_backtest.csv", index=False)

    print("\n=== Pooled fit on 2014+2018+2022 contested races (same rows for every baseline) ===")
    print(pd.DataFrame(coefs).round(4).to_string(index=False))

    show = ["baseline", "cycle", "n", "resid_sd", "mean_resid_pp", "mae_pp", "brier",
            "house_seat_err", "resid_hisp_slope", "res_h0_30_pp", "res_h30_50_pp",
            "res_h50_70_pp", "res_h70_100_pp", "n_h70_100", "train_pass_through"]
    for year in CYCLES:
        print(f"\n=== Held-out {year} (fit on the other two midterms) ===")
        print(res[res["cycle"] == year][show].round(4).to_string(index=False))

    print("\n=== Average over the three held-out cycles ===")
    avg = res.groupby("baseline")[["resid_sd", "mae_pp", "brier", "resid_hisp_slope"]].mean()
    avg["abs_seat_err"] = res.groupby("baseline")["house_seat_err"].apply(lambda s: s.abs().mean())
    avg["abs_res_h70"] = res.groupby("baseline")["res_h70_100_pp"].apply(lambda s: s.abs().mean())
    print(avg.round(4).sort_values("resid_sd").to_string())


if __name__ == "__main__":
    main()
