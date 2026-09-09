"""
Put any past election onto any district plan, by geography instead of by key.

WHY THIS EXISTS
---------------
The key-join collectors pair an election's precinct returns with a precinct
file from the SAME election, which works perfectly (100% match) but only ever
yields the districts in force at the time. Scoring a 2022 legislative race needs
2020 presidential votes on PlanH2316 — a plan that did not exist in 2020 — and
joining 2020 returns to the 2022 precinct file silently loses ~15% of the
statewide vote plus two whole districts, because precincts were redrawn in
between and their keys simply do not appear on the other side.

Precinct keys churn. Geography does not. So: attach returns to the precinct
polygons of their own year (an exact key join), then intersect those polygons
with the target plan's districts and allocate each precinct's votes by the share
of its area falling in each district.

This is the standard method — VEST + a plan shapefile, spatially joined — and it
generalises: it is also the way to give the 2002-2014 regression cycles a
baseline on the lines they were actually run under.

SPLIT PRECINCTS
---------------
A precinct straddling a new district line has its votes split by AREA share, not
assigned whole to the largest piece. Area is a proxy for voters and is wrong at
the margin — a precinct that is half empty ranchland and half subdivision does
not distribute its voters evenly. The script reports how much of the statewide
vote sits in split precincts so the size of that assumption is visible rather
than assumed away. Whole-precinct assignment was rejected because it is biased
in a way area weighting is not: it hands every contested precinct to one side.

USAGE
    python src/collect_presidential_spatial.py --pres-year 2020 --plan H2316
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
HIST = RAW / "historical"
CACHE = RAW / "_capitol_data_cache"
GEO = RAW / "geo"

# Precinct geometry published by TLC, one file per general election.
PRECINCT_GEO = {
    2016: "precincts16g_2016.zip",
    2020: "precincts20g_2020.zip",
    2022: "precincts22g.zip",
    2024: "precincts24g.zip",
}

# Election-returns member inside the comprehensive VTD zip for that year.
RETURNS_ZIP = "{year}-general-vtds-election-data.zip"
RETURNS_CSV = "{year}_General_Election_Returns.csv"

# Target plans and where their district geometry lives. The .gpkg files are
# Census TIGER and carry the CURRENT (post-2021) legislative districts, which
# are PlanH2316 / PlanS2168.
PLAN_GEO = {
    "H2316": (GEO / "tx_house_districts.gpkg", "SLDLST", "house", 150),
    "S2168": (GEO / "tx_senate_districts.gpkg", "SLDUST", "senate", 31),
}

DEM = {2016: ["clinton"], 2020: ["biden"], 2024: ["harris"]}
REP = {2016: ["trump"], 2020: ["trump"], 2024: ["trump"]}


def load_returns(year: int) -> pd.DataFrame:
    """Presidential votes by precinct key, from the cached comprehensive zip."""
    zpath = CACHE / RETURNS_ZIP.format(year=year)
    if not zpath.exists():
        sys.exit(f"missing returns zip: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        member = next(f for f in zf.namelist()
                      if RETURNS_CSV.format(year=year) in f)
        with zf.open(member) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()), encoding="utf-8-sig",
                             low_memory=False)

    pres = df[df["Office"].str.lower().str.contains("president", na=False)].copy()
    pres["Votes"] = pd.to_numeric(pres["Votes"], errors="coerce").fillna(0)
    # Keys are strings. Coercing them to numbers is what silently dropped 3.56%
    # of the 2024 vote on letter-coded VTDs; do not reintroduce it.
    pres["PCTKEY"] = pres["cntyvtd"].astype(str).str.strip().str.upper()

    def side(name: str) -> str:
        nl = str(name).lower()
        if any(t in nl for t in DEM[year]):
            return "dem_votes"
        if any(t in nl for t in REP[year]):
            return "rep_votes"
        return "other_pres_votes"

    pres["side"] = pres["Name"].apply(side)
    wide = (pres.groupby(["PCTKEY", "side"])["Votes"].sum()
            .unstack(fill_value=0).reset_index())
    for c in ("dem_votes", "rep_votes", "other_pres_votes"):
        if c not in wide.columns:
            wide[c] = 0
    return wide


def _strip_suffix(key: str, valid: set[str]) -> str:
    """'850013A' -> '850013' when the parent polygon exists, else unchanged."""
    stripped = re.sub(r"[A-Z]+$", "", key)
    return stripped if stripped != key and stripped in valid else key


def load_precinct_geometry(year: int) -> gpd.GeoDataFrame:
    zpath = CACHE / PRECINCT_GEO[year]
    if not zpath.exists():
        sys.exit(f"missing precinct geometry: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        shp = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
    g = gpd.read_file(f"zip://{zpath}!{shp}")
    g["PCTKEY"] = g["PCTKEY"].astype(str).str.strip().str.upper()
    return g[["PCTKEY", "geometry"]]


def allocate(precincts: gpd.GeoDataFrame, districts: gpd.GeoDataFrame,
             dist_col: str) -> pd.DataFrame:
    """Area-weighted allocation of precinct votes to districts."""
    # Work in the precincts' projected CRS: areas in metres, not degrees.
    districts = districts.to_crs(precincts.crs)
    precincts = precincts.copy()
    precincts["_pct_area"] = precincts.geometry.area

    inter = gpd.overlay(precincts, districts[[dist_col, "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter["_share"] = inter.geometry.area / inter["_pct_area"]

    # A precinct wholly inside one district gets share ~1.0. Renormalise so
    # slivers from imperfect coincident boundaries do not leak or duplicate vote.
    tot = inter.groupby("PCTKEY")["_share"].transform("sum")
    inter["_share"] = inter["_share"] / tot.where(tot > 0, 1.0)

    for c in ("dem_votes", "rep_votes", "other_pres_votes"):
        inter[c] = inter[c] * inter["_share"]

    agg = (inter.groupby(dist_col)[["dem_votes", "rep_votes", "other_pres_votes"]]
           .sum().reset_index())
    agg = agg.rename(columns={dist_col: "district"})
    agg["district"] = agg["district"].astype(int)

    # How much vote actually depends on the area-weighting assumption?
    # Counting every piece with share < 0.999 overstates this badly: most such
    # pieces are topological slivers where a precinct edge and a district edge
    # are meant to be the same line but differ by metres. A precinct is only
    # genuinely split if its LARGEST piece is well short of the whole.
    largest = inter.groupby("PCTKEY")["_share"].transform("max")
    genuinely_split = inter[largest < 0.95]
    split_votes = (genuinely_split["dem_votes"] + genuinely_split["rep_votes"]).sum()
    total_votes = (agg["dem_votes"] + agg["rep_votes"]).sum()
    n_split = genuinely_split["PCTKEY"].nunique()
    print(f"  precinct pieces: {len(inter):,} from {precincts['PCTKEY'].nunique():,} precincts")
    print(f"  genuinely split precincts (largest piece < 95%): {n_split:,}")
    print(f"  vote depending on area weighting: {split_votes:,.0f} of {total_votes:,.0f} "
          f"({split_votes / total_votes * 100:.2f}%)")
    return agg.sort_values("district").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pres-year", type=int, required=True, choices=sorted(PRECINCT_GEO))
    ap.add_argument("--plan", required=True, choices=sorted(PLAN_GEO))
    ap.add_argument("--out", default=None, help="output CSV (default: derived)")
    args = ap.parse_args()

    gpkg, dist_col, chamber, n_max = PLAN_GEO[args.plan]
    print(f"=== {args.pres_year} presidential → PLAN{args.plan} ({chamber}) ===")

    print("Step 1: returns")
    votes = load_returns(args.pres_year)
    in_file = (votes["dem_votes"] + votes["rep_votes"]).sum()
    print(f"  {len(votes):,} precinct keys, {in_file:,.0f} two-party votes in file")

    print("Step 2: precinct geometry")
    pg = load_precinct_geometry(args.pres_year)
    print(f"  {len(pg):,} precinct polygons, crs={pg.crs.name}")

    # Two-pass key match, as in the other collectors: exact first, then strip a
    # trailing letter. Counties report split-precinct parts as '850013A',
    # '2090449A' etc. while the shapefile carries only the parent polygon
    # '850013'. Left alone these are 1.27% of the statewide vote.
    poly_keys = set(pg["PCTKEY"])
    votes = votes.copy()
    votes["PCTKEY"] = votes["PCTKEY"].apply(
        lambda k: k if k in poly_keys else _strip_suffix(k, poly_keys))
    # Parts of the same parent now share a key; sum them before the join.
    votes = (votes.groupby("PCTKEY", as_index=False)
             [["dem_votes", "rep_votes", "other_pres_votes"]].sum())

    merged = pg.merge(votes, on="PCTKEY", how="left")
    matched = merged["dem_votes"].notna().sum()
    merged[["dem_votes", "rep_votes", "other_pres_votes"]] = (
        merged[["dem_votes", "rep_votes", "other_pres_votes"]].fillna(0))
    joined = (merged["dem_votes"] + merged["rep_votes"]).sum()
    print(f"  key join: {matched:,}/{len(pg):,} polygons carry votes; "
          f"{joined:,.0f} of {in_file:,.0f} two-party votes attached "
          f"({joined / in_file * 100:.2f}%)")
    if joined / in_file < 0.99:
        sys.exit("key join lost >1% of the vote — refusing to continue")

    print("Step 3: districts")
    dg = gpd.read_file(gpkg)
    print(f"  {len(dg)} districts from {gpkg.name}")

    print("Step 4: area-weighted allocation")
    agg = allocate(merged, dg, dist_col)

    agg["total_pres_votes"] = (agg["dem_votes"] + agg["rep_votes"]
                               + agg["other_pres_votes"]).round(0)
    denom = agg["dem_votes"] + agg["rep_votes"]
    agg["dem_pres_2p_baseline"] = (agg["dem_votes"] / denom).round(4)
    agg["rep_2p_share"] = (1 - agg["dem_pres_2p_baseline"]).round(4)
    agg["dem_pct"] = (agg["dem_votes"] / agg["total_pres_votes"] * 100).round(2)
    agg["rep_pct"] = (agg["rep_votes"] / agg["total_pres_votes"] * 100).round(2)
    for c in ("dem_votes", "rep_votes", "other_pres_votes"):
        agg[c] = agg[c].round(0)
    agg["chamber"] = chamber
    agg["pres_year"] = args.pres_year
    agg["data_source"] = f"spatial_{args.pres_year}_precincts_x_plan{args.plan}"

    print("\nValidation")
    out_votes = (agg["dem_votes"] + agg["rep_votes"]).sum()
    print(f"  districts: {len(agg)}/{n_max}"
          + ("" if len(agg) == n_max else "  ** MISSING DISTRICTS **"))
    print(f"  two-party vote preserved: {out_votes:,.0f} of {in_file:,.0f} "
          f"({out_votes / in_file * 100:.3f}%)")
    print(f"  statewide Rep 2p: {agg['rep_votes'].sum() / out_votes * 100:.2f}%")
    low = agg[agg["total_pres_votes"] < 0.4 * agg["total_pres_votes"].median()]
    print(f"  districts under 40% of median turnout: {len(low)}"
          + ("" if low.empty else f"  {low['district'].tolist()}"))

    cols = ["chamber", "district", "pres_year", "dem_votes", "rep_votes",
            "other_pres_votes", "total_pres_votes", "dem_pct", "rep_pct",
            "rep_2p_share", "dem_pres_2p_baseline", "data_source"]
    out = Path(args.out) if args.out else (
        HIST / f"tx_presidential_{chamber}_{args.pres_year}_plan{args.plan.lower()}.csv")
    agg[cols].to_csv(out, index=False)
    print(f"\nWrote {out.name} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
