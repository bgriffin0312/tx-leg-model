"""Which statewide race best tracks Texas House results district by district?

Brennan's hypothesis (2026-09-10): a low-salience statewide race (AG 2026:
Middleton v. Johnson) should track generic downballot legislative races better
than marquee races (Senate, Governor) whose candidate effects are idiosyncratic.

Test on 2018, 2022 and 2024 using the Capitol Data Portal VTD returns
(2024-general-vtds-election-data.zip carries every general 2012-2024):
  * VTD -> House district is taken from the State Rep rows in the same file
    (TLC VTDs nest inside House districts; split VTDs are reported).
  * For every statewide office, D two-party share by House district.
  * House D two-party share by district for contested seats (both D and R).
  * Per office: mean gap (House - office), sd of the gap, correlation, OLS
    slope of House on office, residual sd after the fit, and the gap by
    Hispanic-CVAP bucket. Lower residual sd / higher corr = better proxy.
"""
import re
import sys
import zipfile

import numpy as np
import pandas as pd

import os
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "")
# The 2024 comprehensive zip carries every general 2012-2024, re-keyed onto
# 2024 VTDs. That is fine here because VTD -> district comes from the State Rep
# rows in the same file; it is NOT fine for joining to a year's own precinct
# shapefile (see statewide_proxy_cross_cycle.py).
ZIP = ROOT + "data/raw/_capitol_data_cache/2024-general-vtds-election-data.zip"

STATEWIDE = {
    2016: ["President", "RR Comm 1", "Sup Ct 3", "Sup Ct 5", "Sup Ct 9", "CCA 2", "CCA 5", "CCA 6"],
    2020: ["President", "U.S. Sen", "RR Comm 1", "Sup Ct Chief", "Sup Ct 6", "Sup Ct 7", "Sup Ct 8",
           "CCA 3", "CCA 4", "CCA 9"],
    2018: ["U.S. Sen", "Governor", "Lt. Governor", "Attorney Gen", "Comptroller",
           "Land Comm", "Ag Comm", "RR Comm 1", "Sup Ct 2", "Sup Ct 4", "Sup Ct 6",
           "CCA Pres Judge", "CCA 7", "CCA 8"],
    2022: ["Governor", "Lt. Governor", "Attorney Gen", "Comptroller", "Land Comm",
           "Ag Comm", "RR Comm 1", "Sup Ct 3", "Sup Ct 5", "Sup Ct 9", "CCA 5", "CCA 6"],
    2024: ["President", "U.S. Sen", "RR Comm 1", "Sup Ct 2", "Sup Ct 4", "Sup Ct 6",
           "CCA Pres Judge", "CCA 7", "CCA 8"],
}
CVAP = {2016: ROOT + "data/raw/historical/tx_cvap_house_2018.csv",
        2020: ROOT + "data/raw/historical/tx_cvap_house_2018.csv",
        2018: ROOT + "data/raw/historical/tx_cvap_house_2018.csv",
        2022: ROOT + "data/raw/tx_cvap_house.csv",
        2024: ROOT + "data/raw/tx_cvap_house.csv"}


def load_year(year):
    z = zipfile.ZipFile(ZIP)
    df = pd.read_csv(z.open(f"{year}_General_Election_Returns.csv"),
                     usecols=["cntyvtd", "Office", "Party", "Votes"])
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)
    return df


def vtd_to_house(df):
    sr = df[df["Office"].str.match(r"^State Rep \d+$", na=False)].copy()
    sr["dist"] = sr["Office"].str.extract(r"(\d+)").astype(int)
    per = sr.groupby(["cntyvtd", "dist"])["Votes"].sum().reset_index()
    per = per[per["Votes"] > 0]
    nmulti = per.groupby("cntyvtd")["dist"].nunique()
    top = per.sort_values("Votes", ascending=False).drop_duplicates("cntyvtd")
    mapping = dict(zip(top["cntyvtd"], top["dist"]))
    n_all = df["cntyvtd"].nunique()
    print(f"  VTDs in file {n_all:,}; mapped via State Rep rows {len(mapping):,}; "
          f"split across districts {(nmulti > 1).sum():,}")
    # House D/R totals by district, contested flag
    hr = sr[sr["Party"].isin(["D", "R"])].groupby(["dist", "Party"])["Votes"].sum().unstack(fill_value=0)
    hr["contested"] = (hr.get("D", 0) > 0) & (hr.get("R", 0) > 0)
    hr["house_d2p"] = hr["D"] / (hr["D"] + hr["R"])
    return mapping, hr


def office_by_district(df, mapping, office):
    o = df[(df["Office"] == office) & (df["Party"].isin(["D", "R"]))].copy()
    o["dist"] = o["cntyvtd"].map(mapping)
    o = o.dropna(subset=["dist"])
    g = o.groupby(["dist", "Party"])["Votes"].sum().unstack(fill_value=0)
    if "D" not in g.columns or "R" not in g.columns:
        return None  # e.g. CCA 8 in 2018 was R v. L only
    return g["D"] / (g["D"] + g["R"])


def main():
    years = [int(a) for a in sys.argv[1:]] or [2018, 2022, 2024]
    for year in years:
        print(f"\n{'=' * 78}\n{year}\n{'=' * 78}")
        df = load_year(year)
        mapping, hr = vtd_to_house(df)
        cv = pd.read_csv(CVAP[year])
        cv = cv[cv["chamber"].astype(str).str.lower() == "house"] if "chamber" in cv else cv
        hcol = "pct_hispanic"
        cv = cv[["district", hcol]].copy()
        cv["hisp"] = pd.to_numeric(cv[hcol], errors="coerce")
        cv["hisp"] = cv["hisp"].where(cv["hisp"] <= 1, cv["hisp"] / 100)
        cv = cv.set_index("district")["hisp"]

        contested = hr[hr["contested"]]
        print(f"  contested House seats: {len(contested)}  mean House D2p {contested['house_d2p'].mean():.4f}")
        rows = []
        buckets = [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]
        for office in STATEWIDE[year]:
            share = office_by_district(df, mapping, office)
            if share is None:
                print(f"  {office}: no D-v-R contest, skipped")
                continue
            m = pd.DataFrame({"house": contested["house_d2p"], "office": share}).dropna()
            m["hisp"] = cv.reindex(m.index).values
            gap = m["house"] - m["office"]
            slope, intercept = np.polyfit(m["office"], m["house"], 1)
            resid = m["house"] - (intercept + slope * m["office"])
            statewide = share.reindex(hr.index)  # all 150 for the statewide mean
            row = {"office": office, "n": len(m),
                   "office_mean_D2p_contested": m["office"].mean(),
                   "gap_mean": gap.mean(), "gap_sd": gap.std(),
                   "corr": m["house"].corr(m["office"]), "slope": slope,
                   "resid_sd": resid.std()}
            for lo, hi in buckets:
                s = gap[(m["hisp"] >= lo) & (m["hisp"] < hi)]
                row[f"gap_h{int(lo*100)}-{int(min(hi,1)*100)}"] = s.mean() if len(s) else np.nan
                row[f"n_h{int(lo*100)}"] = len(s)
            rows.append(row)
        out = pd.DataFrame(rows).sort_values("resid_sd")
        pd.set_option("display.width", 250)
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
