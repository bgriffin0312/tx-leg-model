"""
collect_presidential_by_district.py

Download and aggregate 2024 presidential results to TX state legislative districts.

Sources (TX Capitol Data Portal):
  1. Comprehensive Election Datasets ZIP — contains 2024_General_Election_Returns.csv
     https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format
  2. Precinct-to-District Mapping (precincts24g_districts.xlsx)
     https://data.capitol.texas.gov/dataset/precincts

Strategy:
  1. Load 2024_General_Election_Returns.csv from the comprehensive ZIP
  2. Load precincts24g_districts.xlsx (PCTKEY → PlanH2316/PlanS2168)
  3. Filter Office == 'President' rows for Trump vs Harris
  4. Join on cntyvtd == PCTKEY
  5. Aggregate by House / Senate district and compute 2p share

Output:
  data/raw/tx_presidential_house_2024.csv  — 150 rows
  data/raw/tx_presidential_senate_2024.csv — 31 rows

Columns:
  chamber, district, trump_votes, harris_votes, other_pres_votes,
  total_pres_votes, trump_pct, harris_pct,
  trump_2p_share, dem_pres_2p_baseline, data_source

Usage:
  python src/collect_presidential_by_district.py
  python src/collect_presidential_by_district.py --no-cache  # force re-download
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR = DATA_RAW / "_capitol_data_cache"

VTD_ZIP_URL = (
    "https://data.capitol.texas.gov/dataset/35b16aee-0bb0-4866-b1ec-859f1f044241"
    "/resource/e1cd6332-6a7a-4c78-ad2a-852268f6c7a2"
    "/download/2024-general-vtds-election-data.zip"
)
PRECINCT_MAP_URL = (
    "https://data.capitol.texas.gov/dataset/d04c72b9-16c4-4ab2-8c6d-c666d41e04b7"
    "/resource/1b8a3c29-931f-49be-a6c0-a092945f6679"
    "/download/precincts24g_districts.xlsx"
)
TARGET_CSV = "2024_General_Election_Returns.csv"

USER_AGENT = "TX-Legislature-Model/1.0 (research; contact via GitHub)"
TIMEOUT = 300


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_with_cache(url: str, cache_path: Path, force: bool = False,
                        min_size: int = 10_000) -> bytes | None:
    if not force and cache_path.exists() and cache_path.stat().st_size > min_size:
        print(f"  Using cached: {cache_path.name} ({cache_path.stat().st_size:,} bytes)")
        return cache_path.read_bytes()

    print(f"  Downloading: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=TIMEOUT, verify=False)
        r.raise_for_status()
        data = r.content
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        print(f"  Saved: {cache_path.name} ({len(data):,} bytes)")
        return data
    except requests.RequestException as exc:
        print(f"  ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_general_election_csv(zip_data: bytes) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            matches = [f for f in zf.namelist() if TARGET_CSV in f]
            if not matches:
                print(f"  ERROR: {TARGET_CSV} not found in ZIP")
                return None
            target = matches[0]
            print(f"  Reading: {target} ({zf.getinfo(target).file_size:,} bytes)")
            with zf.open(target) as f:
                df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
        print(f"  Loaded {len(df):,} rows")
        return df
    except zipfile.BadZipFile as exc:
        print(f"  ERROR: {exc}")
        return None


def load_precinct_mapping(map_data: bytes) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(io.BytesIO(map_data), engine="openpyxl")
        print(f"  Loaded {len(df):,} precinct rows")
        print(f"  Columns: {list(df.columns)}")
        return df
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_presidential(df_elec: pd.DataFrame,
                            df_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter presidential rows, join precinct mapping, aggregate by district.
    precincts24g_districts.xlsx columns:
      PCTKEY  = join key to cntyvtd
      PlanH2316 = House district (SLDL)
      PlanS2168 = Senate district (SLDU)
    """
    # Presidential rows only
    pres = df_elec[df_elec["Office"].str.lower().str.contains("president", na=False)].copy()
    print(f"  Presidential rows: {len(pres):,}")

    # Classify Trump vs Harris
    def classify(name: str) -> str:
        nl = str(name).lower()
        if "trump" in nl:
            return "trump"
        if "harris" in nl:
            return "harris"
        return "other"

    pres["candidate_type"] = pres["Name"].apply(classify)
    pres["Votes"] = pd.to_numeric(pres["Votes"], errors="coerce").fillna(0)
    pres["cntyvtd"] = pd.to_numeric(pres["cntyvtd"], errors="coerce")

    # Pivot to wide: one row per VTD with trump/harris/other votes
    vtd_pres = (pres.groupby(["cntyvtd", "candidate_type"])["Votes"]
                .sum().unstack(fill_value=0).reset_index())
    for col in ("trump", "harris", "other"):
        if col not in vtd_pres.columns:
            vtd_pres[col] = 0

    # Join precinct mapping
    df_map = df_map.copy()
    df_map["PCTKEY"] = pd.to_numeric(df_map["PCTKEY"], errors="coerce")
    df_map = df_map.dropna(subset=["PCTKEY", "PlanH2316", "PlanS2168"])
    df_map["PlanH2316"] = df_map["PlanH2316"].astype(int)
    df_map["PlanS2168"] = df_map["PlanS2168"].astype(int)

    merged = vtd_pres.merge(
        df_map[["PCTKEY", "PlanH2316", "PlanS2168"]],
        left_on="cntyvtd", right_on="PCTKEY", how="left"
    )

    unmatched = merged["PlanH2316"].isna().sum()
    unmatched_votes = merged[merged["PlanH2316"].isna()][["trump", "harris"]].sum().sum()
    total_votes = merged[["trump", "harris"]].sum().sum()
    print(f"  Join: {unmatched} VTDs unmatched ({unmatched_votes:,.0f} / {total_votes:,.0f} "
          f"presidential votes = {unmatched_votes/total_votes*100:.1f}% unassigned)")

    def make_district_df(dist_col: str, chamber: str, n_max: int) -> pd.DataFrame:
        sub = merged.dropna(subset=[dist_col]).copy()
        sub[dist_col] = sub[dist_col].astype(int)
        sub = sub[sub[dist_col].between(1, n_max)]

        agg = sub.groupby(dist_col)[["trump", "harris", "other"]].sum().reset_index()
        agg.columns = ["district", "trump_votes", "harris_votes", "other_pres_votes"]
        agg["total_pres_votes"] = agg["trump_votes"] + agg["harris_votes"] + agg["other_pres_votes"]
        agg["trump_pct"] = (agg["trump_votes"] / agg["total_pres_votes"] * 100).round(2)
        agg["harris_pct"] = (agg["harris_votes"] / agg["total_pres_votes"] * 100).round(2)
        denom = agg["trump_votes"] + agg["harris_votes"]
        agg["trump_2p_share"] = (agg["trump_votes"] / denom.replace(0, float("nan"))).round(4)
        agg["dem_pres_2p_baseline"] = (1 - agg["trump_2p_share"]).round(4)
        agg["chamber"] = chamber
        agg["data_source"] = "tx_capitol_vtd_2024"
        cols = ["chamber", "district", "trump_votes", "harris_votes", "other_pres_votes",
                "total_pres_votes", "trump_pct", "harris_pct",
                "trump_2p_share", "dem_pres_2p_baseline", "data_source"]
        return agg.sort_values("district").reset_index(drop=True)[cols]

    house_df = make_district_df("PlanH2316", "house", 150)
    senate_df = make_district_df("PlanS2168", "senate", 31)
    return house_df, senate_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect 2024 presidential results by TX legislative district")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("=== 2024 Presidential Results by TX Legislative District ===\n")

    print("Step 1: Load 2024 General Election data (comprehensive ZIP)")
    zip_data = download_with_cache(
        VTD_ZIP_URL, CACHE_DIR / "2024-general-vtds-election-data.zip",
        force=args.no_cache, min_size=1_000_000)
    if zip_data is None:
        sys.exit(1)
    df_elec = load_general_election_csv(zip_data)
    if df_elec is None:
        sys.exit(1)

    print("\nStep 2: Load precinct-to-district mapping")
    map_data = download_with_cache(
        PRECINCT_MAP_URL, CACHE_DIR / "precincts24g_districts.xlsx",
        force=args.no_cache, min_size=10_000)
    if map_data is None:
        sys.exit(1)
    df_map = load_precinct_mapping(map_data)
    if df_map is None:
        sys.exit(1)

    print("\nStep 3: Aggregate presidential votes by district")
    house_df, senate_df = aggregate_presidential(df_elec, df_map)

    print(f"\n  House districts: {len(house_df)} / 150")
    print(f"  Senate districts: {len(senate_df)} / 31")

    # Statewide validation
    total_trump = house_df["trump_votes"].sum()
    total_harris = house_df["harris_votes"].sum()
    if total_trump + total_harris > 0:
        statewide = total_trump / (total_trump + total_harris) * 100
        flag = "OK" if 54 < statewide < 58 else "CHECK THIS"
        print(f"\n  Statewide Trump 2p share: {statewide:.1f}%  (expected ~56.1%) [{flag}]")

    # Sample
    print("\n  Sample House districts:")
    for hd in [13, 47, 100, 130, 144]:
        row = house_df[house_df["district"] == hd]
        if not row.empty:
            r = row.iloc[0]
            print(f"    HD{hd:3d}: Trump {r['trump_pct']:.1f}%  "
                  f"Harris {r['harris_pct']:.1f}%  "
                  f"Dem 2p: {r['dem_pres_2p_baseline']:.3f}")

    print("\n  Sample Senate districts:")
    for sd in [9, 19, 26, 31]:
        row = senate_df[senate_df["district"] == sd]
        if not row.empty:
            r = row.iloc[0]
            print(f"    SD{sd:2d}: Trump {r['trump_pct']:.1f}%  "
                  f"Harris {r['harris_pct']:.1f}%  "
                  f"Dem 2p: {r['dem_pres_2p_baseline']:.3f}")

    print("\nStep 4: Write output")
    out_house = DATA_RAW / "tx_presidential_house_2024.csv"
    out_senate = DATA_RAW / "tx_presidential_senate_2024.csv"
    house_df.to_csv(out_house, index=False)
    senate_df.to_csv(out_senate, index=False)
    print(f"  Wrote {out_house.name} ({len(house_df)} rows)")
    print(f"  Wrote {out_senate.name} ({len(senate_df)} rows)")


if __name__ == "__main__":
    main()
