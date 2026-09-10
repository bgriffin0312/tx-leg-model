"""Cross-cycle test: which prior-presidential-year statewide race best predicts
the NEXT midterm's Texas House results, district by district?

This is the model's actual use of a baseline (2024 statewide -> 2026 House).
Pairs:
  2016 statewide -> 2018 House (both PlanH2100; 2016 precincts area-weighted
                    onto PlanH2100 via the repo's spatial collector)
  2020 statewide -> 2022 House (2020 precincts onto PlanH2316)
Offices: President, U.S. Senate (2020 only), Railroad Commissioner, Supreme
Court and Court of Criminal Appeals seats with a D-v-R contest, plus a
"judicial+RRC mean" composite.
Per office: OLS of House D2p on office D2p (slope = empirical pass-through),
correlation, residual sd, and mean residual by Hispanic-CVAP bucket.

Reuses collect_presidential_spatial.py for geometry and allocation.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import collect_presidential_spatial as cps  # noqa: E402

SCRATCH = ROOT / "output"  # cached per-office district allocations

PAIRS = [
    # (statewide year, plan, House result year, CVAP file)
    (2016, "H2100", 2018, ROOT / "data/raw/historical/tx_cvap_house_2018.csv"),
    (2020, "H2316", 2022, ROOT / "data/raw/tx_cvap_house.csv"),
]
OFFICE_RE = r"^(President|U\.S\. Sen|RR Comm 1|Sup Ct|CCA)"


def load_statewide_by_precinct(year):
    """D and R votes by precinct key for every statewide office of interest."""
    # The 2024 comprehensive zip re-keys every earlier election onto 2024 VTDs
    # (9,712 keys for all years), so it will not join to the year's own precinct
    # shapefile (81% attach for 2020). Read each year from its own zip, as the
    # repo's spatial collector does (98.7% attach).
    zpath = ROOT / "data/raw/_capitol_data_cache" / cps.RETURNS_ZIP.format(year=year)
    with zipfile.ZipFile(zpath) as zf:
        df = pd.read_csv(zf.open(f"{year}_General_Election_Returns.csv"),
                         usecols=["cntyvtd", "Office", "Party", "Votes"],
                         encoding="utf-8-sig", low_memory=False)
    df = df[df["Office"].str.match(OFFICE_RE, na=False) & df["Party"].isin(["D", "R"])].copy()
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)
    df["PCTKEY"] = df["cntyvtd"].astype(str).str.strip().str.upper()
    wide = (df.groupby(["PCTKEY", "Office", "Party"])["Votes"].sum()
              .unstack("Party", fill_value=0).reset_index())
    # keep offices with both parties somewhere
    ok = wide.groupby("Office")[["D", "R"]].sum()
    ok = ok[(ok["D"] > 0) & (ok["R"] > 0)].index
    return wide[wide["Office"].isin(ok)]


def allocate_all_offices(year, plan):
    """Area-weight every office's D/R precinct votes onto the plan's districts.
    Returns DataFrame: district, office, d2p."""
    cache = SCRATCH / f"statewide_{year}_plan{plan}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    votes = load_statewide_by_precinct(year)
    pg = cps.load_precinct_geometry(year)
    poly_keys = set(pg["PCTKEY"])
    votes["PCTKEY"] = votes["PCTKEY"].apply(
        lambda k: k if k in poly_keys else cps._strip_suffix(k, poly_keys))
    gpkg, dist_col, chamber, n_max = cps.PLAN_GEO[plan]
    if str(gpkg).lower().endswith(".zip"):
        with zipfile.ZipFile(gpkg) as zf:
            shp = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
        dg = gpd.read_file(f"zip://{gpkg}!{shp}")
    else:
        dg = gpd.read_file(gpkg)
    out = []
    for office, sub in votes.groupby("Office"):
        sub = sub.groupby("PCTKEY", as_index=False)[["D", "R"]].sum()
        sub = sub.rename(columns={"D": "dem_votes", "R": "rep_votes"})
        sub["other_pres_votes"] = 0.0
        in_file = (sub["dem_votes"] + sub["rep_votes"]).sum()
        merged = pg.merge(sub, on="PCTKEY", how="left")
        merged[["dem_votes", "rep_votes", "other_pres_votes"]] = (
            merged[["dem_votes", "rep_votes", "other_pres_votes"]].fillna(0))
        joined = (merged["dem_votes"] + merged["rep_votes"]).sum()
        print(f"  {year} {office}: {joined / in_file * 100:.2f}% of two-party vote attached")
        agg = cps.allocate(merged, dg, dist_col)
        agg["office"] = office
        agg["d2p"] = agg["dem_votes"] / (agg["dem_votes"] + agg["rep_votes"])
        out.append(agg[["district", "office", "d2p"]])
    res = pd.concat(out, ignore_index=True)
    res.to_csv(cache, index=False)
    return res


def house_results(year):
    h = pd.read_csv(ROOT / f"data/raw/historical/tx_house_results_{year}.csv")
    h = h[h["contested"] == True].copy()
    h["house_d2p"] = pd.to_numeric(h["dem_2p_share"], errors="coerce")
    return h.set_index("district")["house_d2p"].dropna()


def evaluate(pred, house, hisp, label):
    m = pd.DataFrame({"x": pred, "y": house}).dropna()
    m["hisp"] = hisp.reindex(m.index).values
    slope, intercept = np.polyfit(m["x"], m["y"], 1)
    resid = m["y"] - (intercept + slope * m["x"])
    row = {"office": label, "n": len(m), "corr": m["x"].corr(m["y"]),
           "slope": slope, "intercept": intercept, "resid_sd": resid.std(),
           "mae": resid.abs().mean()}
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
        s = resid[(m["hisp"] >= lo) & (m["hisp"] < hi)]
        row[f"res_h{int(lo*100)}"] = s.mean() if len(s) else np.nan
    # Uniform-swing version (slope fixed at 1): what the model does when the
    # environment dial carries the level. Residual sd after removing the mean.
    us = (m["y"] - m["x"]) - (m["y"] - m["x"]).mean()
    row["unif_resid_sd"] = us.std()
    return row


def main():
    pd.set_option("display.width", 250)
    for sw_year, plan, house_year, cvap_path in PAIRS:
        print(f"\n{'=' * 78}\n{sw_year} statewide -> {house_year} House  (plan {plan})\n{'=' * 78}")
        sw = allocate_all_offices(sw_year, plan)
        house = house_results(house_year)
        cv = pd.read_csv(cvap_path)
        cv = cv[cv["chamber"].astype(str).str.lower() == "house"]
        hisp = pd.to_numeric(cv.set_index("district")["pct_hispanic"], errors="coerce")
        hisp = hisp.where(hisp <= 1, hisp / 100)
        wide = sw.pivot(index="district", columns="office", values="d2p")
        rows = []
        for office in wide.columns:
            rows.append(evaluate(wide[office], house, hisp, office))
        jud = [c for c in wide.columns if re.match(r"^(Sup Ct|CCA|RR Comm)", c)]
        rows.append(evaluate(wide[jud].mean(axis=1), house, hisp, f"MEAN of {len(jud)} judicial+RRC"))
        out = pd.DataFrame(rows).sort_values("resid_sd")
        print(f"  contested House seats {house_year}: {len(house)}")
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
