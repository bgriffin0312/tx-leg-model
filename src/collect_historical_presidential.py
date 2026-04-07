"""
collect_historical_presidential.py

Collect historical presidential election results by TX state legislative district
for backtesting the 2026 model against 2022 and 2018 actual election outcomes.

For 2022 backtest:
  2020 presidential (Biden/Trump) mapped to POST-redistricting district lines
  (same PlanH2316 / PlanS2168 plans used in 2022, 2024, and 2026 elections)
  → uses precincts22g_districts.xlsx (2022 general election precinct mapping)

For 2018 backtest:
  2016 presidential (Clinton/Trump) mapped to PRE-redistricting district lines
  (old House Plan H358 / Senate Plan S193 used in 2018 elections)
  → uses precincts18g_districts.xlsx (2018 general election precinct mapping)
  NOTE: 2018 used different district boundaries; backtest has limited CVAP support.

Data discovery:
  Uses TX Capitol Data Portal CKAN API (verify=False due to SSL chain issues on Windows):
    Comprehensive election datasets:
      https://data.capitol.texas.gov/api/3/action/package_show?id=comprehensive-election-datasets-compressed-format
    Precinct-to-district mappings:
      https://data.capitol.texas.gov/api/3/action/package_show?id=precincts

Output:
  data/raw/historical/tx_presidential_house_2020.csv   (150 rows, 2022 district lines)
  data/raw/historical/tx_presidential_senate_2020.csv  (31 rows, 2022 district lines)
  data/raw/historical/tx_presidential_house_2016.csv   (150 rows, 2018 district lines)
  data/raw/historical/tx_presidential_senate_2016.csv  (31 rows, 2018 district lines)

Columns (same as tx_presidential_*_2024.csv):
  chamber, district, dem_votes, rep_votes, other_pres_votes,
  total_pres_votes, dem_pct, rep_pct, rep_2p_share, dem_pres_2p_baseline, data_source

Usage:
  python src/collect_historical_presidential.py --pres-year 2020
  python src/collect_historical_presidential.py --pres-year 2016
  python src/collect_historical_presidential.py  # both years
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_HIST.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_RAW / "_capitol_data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "TX-Legislature-Model/1.0 (research; contact via GitHub)"
TIMEOUT = 300
CKAN_API = "https://data.capitol.texas.gov/api/3/action/package_show"

# ---------------------------------------------------------------------------
# Presidential candidate classification by year
# ---------------------------------------------------------------------------
CANDIDATE_CLASSIFIERS = {
    2020: {"dem": ["biden"], "rep": ["trump"]},
    2016: {"dem": ["clinton", "hillary"], "rep": ["trump"]},
}

# Precinct mapping info per election year
# map_year = the general election year whose precinct file we use for district assignment
# house_col / senate_col = column names in the XLSX for house/senate district assignment
PRECINCT_MAP_INFO = {
    2022: {
        "filename": "precincts22g_districts.xlsx",
        "house_col": "PlanH2316",
        "senate_col": "PlanS2168",
        "n_house": 150,
        "n_senate": 31,
        "note": "Post-2021 redistricting plans (same as 2024 and 2026)",
    },
    2018: {
        "filename": "precincts18g_districts.xlsx",
        "house_col": None,   # will be detected from file
        "senate_col": None,  # will be detected from file
        "n_house": 150,
        "n_senate": 31,
        "note": "Pre-2021 redistricting plans (old boundaries)",
    },
}

# Statewide validation targets (Trump 2p share)
STATEWIDE_VALIDATION = {
    2020: {"expected_trump_2p": 52.2, "tolerance": 2.0,
           "note": "TX 2020: Trump 52.2% 2p (52.1% official)"},
    2016: {"expected_trump_2p": 54.5, "tolerance": 2.0,
           "note": "TX 2016: Trump 54.5% 2p (52.6% total, ~54.5% 2p)"},
}


# ---------------------------------------------------------------------------
# CKAN API URL discovery
# ---------------------------------------------------------------------------

def ckan_get_resources(dataset_id_or_slug: str) -> list[dict]:
    """
    Query the TX Capitol CKAN API to get all resources for a dataset.
    Returns list of resource dicts with 'name', 'url', 'format', etc.
    """
    url = f"{CKAN_API}?id={dataset_id_or_slug}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=30, verify=False)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            print(f"  CKAN API error: {data.get('error', 'unknown')}")
            return []
        resources = data["result"].get("resources", [])
        print(f"  Found {len(resources)} resources in dataset '{dataset_id_or_slug}'")
        return resources
    except Exception as exc:
        print(f"  CKAN API request failed: {exc}")
        return []


def find_election_zip_url(pres_year: int) -> str | None:
    """
    Discover the download URL for the general election VTD ZIP for the year
    containing the target presidential election (pres_year).

    The TX Capitol portal has comprehensive election ZIPs by general election year.
    Presidential elections are in Nov of election year (2020 or 2016).
    """
    elec_year = pres_year  # presidential elections are in Nov of that year
    target_filename = f"{elec_year}-general-vtds-election-data.zip"

    print(f"  Searching CKAN for {elec_year} general election ZIP...")
    resources = ckan_get_resources("comprehensive-election-datasets-compressed-format")

    # Search by filename / URL pattern
    for r in resources:
        url = r.get("url", "")
        name = r.get("name", "")
        if str(elec_year) in url and "general" in url.lower() and "vtd" in url.lower():
            print(f"  Found via URL pattern: {url}")
            return url
        if str(elec_year) in name.lower() and "general" in name.lower():
            print(f"  Found via name: {name} → {url}")
            return url

    # Fallback: try common URL pattern
    fallback = (
        "https://data.capitol.texas.gov/dataset/"
        f"35b16aee-0bb0-4866-b1ec-859f1f044241/"
        f"resource/UNKNOWN/download/{target_filename}"
    )
    print(f"  CKAN discovery failed. Trying guessed URL pattern...")
    print(f"  Target file: {target_filename}")
    print(f"  Hint: Check https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format")
    return None


def find_precinct_map_url(map_year: int) -> str | None:
    """
    Discover the download URL for the precincts{YY}g_districts.xlsx file.
    map_year: the general election year (e.g. 2022 or 2018)
    """
    suffix = f"{str(map_year)[2:]}g"  # e.g. "22g" for 2022
    filename = f"precincts{suffix}_districts.xlsx"

    print(f"  Searching CKAN for {filename}...")
    resources = ckan_get_resources("precincts")

    for r in resources:
        url = r.get("url", "")
        name = r.get("name", "")
        if filename.lower() in url.lower() or filename.lower() in name.lower():
            print(f"  Found: {url}")
            return url
        if suffix in url.lower() and "district" in url.lower():
            print(f"  Found (partial match): {url}")
            return url

    print(f"  CKAN discovery failed for {filename}.")
    print(f"  Hint: Check https://data.capitol.texas.gov/dataset/precincts")
    return None


# ---------------------------------------------------------------------------
# Download helpers (same pattern as collect_presidential_by_district.py)
# ---------------------------------------------------------------------------

def download_with_cache(url: str, cache_path: Path, force: bool = False,
                        min_size: int = 10_000) -> bytes | None:
    if not force and cache_path.exists() and cache_path.stat().st_size > min_size:
        print(f"  Using cached: {cache_path.name} ({cache_path.stat().st_size:,} bytes)")
        return cache_path.read_bytes()

    if url is None:
        print(f"  ERROR: No URL provided for {cache_path.name}")
        return None

    print(f"  Downloading: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=TIMEOUT, verify=False)
        r.raise_for_status()
        data = r.content
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        print(f"  Saved: {cache_path.name} ({len(data):,} bytes)")
        return data
    except requests.RequestException as exc:
        print(f"  ERROR: {exc}")
        return None


# ---------------------------------------------------------------------------
# Load election data from ZIP
# ---------------------------------------------------------------------------

def find_general_election_csv(zip_data: bytes, year: int) -> pd.DataFrame | None:
    """
    Find and load the general election returns CSV from the ZIP.
    Tries several common filename patterns.
    """
    target_patterns = [
        f"{year}_General_Election_Returns.csv",
        f"{year}_general_election_returns.csv",
        f"{year}_General_Returns.csv",
    ]
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            names = zf.namelist()
            print(f"  ZIP contents ({len(names)} files): {names[:10]}")

            # Try explicit patterns first
            for pattern in target_patterns:
                matches = [f for f in names if pattern.lower() in f.lower()]
                if matches:
                    target = matches[0]
                    print(f"  Reading: {target} ({zf.getinfo(target).file_size:,} bytes)")
                    with zf.open(target) as f:
                        df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
                    print(f"  Loaded {len(df):,} rows")
                    return df

            # Fallback: find the largest CSV
            csvs = [(f, zf.getinfo(f).file_size) for f in names
                    if f.lower().endswith(".csv") and "general" in f.lower()]
            if csvs:
                csvs.sort(key=lambda x: x[1], reverse=True)
                target, size = csvs[0]
                print(f"  Reading largest general CSV: {target} ({size:,} bytes)")
                with zf.open(target) as f:
                    df = pd.read_csv(f, encoding="utf-8-sig", low_memory=False)
                print(f"  Loaded {len(df):,} rows")
                return df

            print(f"  ERROR: No matching CSV found in ZIP")
            print(f"  Available: {names}")
            return None

    except zipfile.BadZipFile as exc:
        print(f"  ERROR: Bad ZIP file: {exc}")
        return None


def load_precinct_mapping(map_data: bytes, map_info: dict) -> pd.DataFrame | None:
    """
    Load precinct-to-district mapping XLSX and detect district plan columns.
    """
    try:
        df = pd.read_excel(io.BytesIO(map_data), engine="openpyxl")
        print(f"  Loaded {len(df):,} precinct rows")
        print(f"  Columns: {list(df.columns)}")

        # Auto-detect house/senate plan columns if not specified
        house_col = map_info["house_col"]
        senate_col = map_info["senate_col"]

        if house_col is None or house_col not in df.columns:
            # Look for columns matching House plan patterns
            for col in df.columns:
                col_upper = str(col).upper()
                if "PLANH" in col_upper or ("PLAN" in col_upper and "H" in col_upper
                                             and "S" not in col_upper):
                    house_col = col
                    print(f"  Auto-detected house column: {house_col}")
                    break

        if senate_col is None or senate_col not in df.columns:
            for col in df.columns:
                col_upper = str(col).upper()
                if "PLANS" in col_upper or ("PLAN" in col_upper and "S" in col_upper
                                             and "H" not in col_upper):
                    senate_col = col
                    print(f"  Auto-detected senate column: {senate_col}")
                    break

        # Final fallback: look for any numeric plan columns
        if house_col is None or senate_col is None:
            plan_cols = [c for c in df.columns if str(c).upper().startswith("PLAN")]
            print(f"  Plan columns found: {plan_cols}")
            if len(plan_cols) >= 2 and house_col is None:
                house_col = plan_cols[0]
                print(f"  Using {house_col} for house")
            if len(plan_cols) >= 2 and senate_col is None:
                senate_col = plan_cols[1]
                print(f"  Using {senate_col} for senate")

        return df, house_col, senate_col

    except Exception as exc:
        print(f"  ERROR loading precinct mapping: {exc}")
        return None, None, None


# ---------------------------------------------------------------------------
# Aggregate presidential votes by district
# ---------------------------------------------------------------------------

def classify_candidate(name: str, classifiers: dict) -> str:
    """Classify a candidate as 'dem', 'rep', or 'other'."""
    nl = str(name).lower()
    for keyword in classifiers["dem"]:
        if keyword in nl:
            return "dem"
    for keyword in classifiers["rep"]:
        if keyword in nl:
            return "rep"
    return "other"


def aggregate_presidential(df_elec: pd.DataFrame, df_map: pd.DataFrame,
                            house_col: str, senate_col: str,
                            pres_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter presidential rows, join precinct mapping, aggregate by district.
    Mirrors the logic in collect_presidential_by_district.py.
    """
    classifiers = CANDIDATE_CLASSIFIERS[pres_year]

    # Filter for presidential race
    pres_mask = df_elec["Office"].str.lower().str.contains("president", na=False)
    pres = df_elec[pres_mask].copy()
    print(f"  Presidential rows: {len(pres):,}")

    if len(pres) == 0:
        print("  ERROR: No presidential rows found. Check 'Office' column values:")
        print(f"  Sample offices: {df_elec['Office'].value_counts().head(20).to_dict()}")
        return None, None

    # Classify candidates
    pres["candidate_type"] = pres["Name"].apply(
        lambda n: classify_candidate(n, classifiers)
    )
    pres["Votes"] = pd.to_numeric(pres["Votes"], errors="coerce").fillna(0)
    pres["cntyvtd"] = pd.to_numeric(pres["cntyvtd"], errors="coerce")

    # Print classification summary
    type_counts = pres.groupby("candidate_type")["Votes"].sum()
    print(f"  Vote totals by type: {type_counts.to_dict()}")

    # Pivot: one row per VTD
    vtd_pres = (pres.groupby(["cntyvtd", "candidate_type"])["Votes"]
                .sum().unstack(fill_value=0).reset_index())
    for col in ("dem", "rep", "other"):
        if col not in vtd_pres.columns:
            vtd_pres[col] = 0

    # Join precinct mapping
    df_map = df_map.copy()
    df_map["PCTKEY"] = pd.to_numeric(df_map["PCTKEY"], errors="coerce")
    df_map = df_map.dropna(subset=["PCTKEY"])
    if house_col:
        df_map = df_map.dropna(subset=[house_col])
        df_map[house_col] = df_map[house_col].astype(int)
    if senate_col:
        df_map = df_map.dropna(subset=[senate_col])
        df_map[senate_col] = df_map[senate_col].astype(int)

    keep_cols = ["PCTKEY"] + ([house_col] if house_col else []) + ([senate_col] if senate_col else [])
    merged = vtd_pres.merge(
        df_map[keep_cols],
        left_on="cntyvtd", right_on="PCTKEY", how="left"
    )

    unmatched = merged[house_col].isna().sum() if house_col else 0
    total_votes = merged[["dem", "rep"]].sum().sum()
    unmatched_votes = (merged[merged[house_col].isna()][["dem", "rep"]].sum().sum()
                       if house_col and unmatched > 0 else 0)
    pct_unmatched = unmatched_votes / total_votes * 100 if total_votes > 0 else 0
    print(f"  Join: {unmatched} VTDs unmatched ({unmatched_votes:,.0f} / {total_votes:,.0f} "
          f"votes = {pct_unmatched:.1f}% unassigned)")

    def make_district_df(dist_col: str, chamber_name: str, n_max: int) -> pd.DataFrame | None:
        if dist_col is None or dist_col not in merged.columns:
            print(f"  WARNING: No {chamber_name} district column found")
            return None

        sub = merged.dropna(subset=[dist_col]).copy()
        sub[dist_col] = sub[dist_col].astype(int)
        sub = sub[sub[dist_col].between(1, n_max)]

        agg = sub.groupby(dist_col)[["dem", "rep", "other"]].sum().reset_index()
        agg.columns = ["district", "dem_votes", "rep_votes", "other_pres_votes"]
        agg["total_pres_votes"] = agg["dem_votes"] + agg["rep_votes"] + agg["other_pres_votes"]
        agg["dem_pct"] = (agg["dem_votes"] / agg["total_pres_votes"] * 100).round(2)
        agg["rep_pct"] = (agg["rep_votes"] / agg["total_pres_votes"] * 100).round(2)
        denom = agg["dem_votes"] + agg["rep_votes"]
        agg["rep_2p_share"] = (agg["rep_votes"] / denom.replace(0, float("nan"))).round(4)
        agg["dem_pres_2p_baseline"] = (1 - agg["rep_2p_share"]).round(4)
        agg["chamber"] = chamber_name
        agg["pres_year"] = pres_year
        agg["data_source"] = f"tx_capitol_vtd_{pres_year}"
        cols = ["chamber", "district", "pres_year", "dem_votes", "rep_votes",
                "other_pres_votes", "total_pres_votes", "dem_pct", "rep_pct",
                "rep_2p_share", "dem_pres_2p_baseline", "data_source"]
        return agg.sort_values("district").reset_index(drop=True)[cols]

    map_info = PRECINCT_MAP_INFO.get(
        2022 if pres_year == 2020 else 2018, PRECINCT_MAP_INFO[2022])
    house_df = make_district_df(house_col, "house", map_info["n_house"])
    senate_df = make_district_df(senate_col, "senate", map_info["n_senate"])
    return house_df, senate_df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_statewide(house_df: pd.DataFrame, pres_year: int):
    """Print statewide validation vs expected Trump 2p share."""
    if house_df is None or house_df.empty:
        return
    val = STATEWIDE_VALIDATION.get(pres_year, {})
    total_dem = house_df["dem_votes"].sum()
    total_rep = house_df["rep_votes"].sum()
    denom = total_dem + total_rep
    if denom == 0:
        return
    rep_2p = total_rep / denom * 100
    expected = val.get("expected_trump_2p", 0)
    tol = val.get("tolerance", 2.0)
    diff = abs(rep_2p - expected)
    flag = "OK" if diff <= tol else "CHECK THIS"
    print(f"\n  Statewide Rep 2p share: {rep_2p:.1f}%  "
          f"(expected ~{expected:.1f}%)  [{flag}]")
    print(f"  Note: {val.get('note', '')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_pres_year(pres_year: int, force: bool = False):
    """
    Collect presidential results for one presidential year and write output CSVs.
    """
    map_year = 2022 if pres_year == 2020 else 2018
    map_info = PRECINCT_MAP_INFO[map_year]
    map_filename = map_info["filename"]

    print(f"\n{'='*60}")
    print(f"{pres_year} Presidential Results → {map_year} District Boundaries")
    print(f"{'='*60}")

    # Step 1: Discover and download election ZIP
    print(f"\nStep 1: Find {pres_year} general election ZIP")
    zip_cache = CACHE_DIR / f"{pres_year}-general-vtds-election-data.zip"

    zip_url = None
    if not zip_cache.exists() or force:
        zip_url = find_election_zip_url(pres_year)

    zip_data = download_with_cache(zip_url, zip_cache, force=force, min_size=1_000_000)
    if zip_data is None:
        print(f"\n  FAILED: Could not obtain {pres_year} election ZIP.")
        print(f"  Manual steps:")
        print(f"    1. Go to: https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format")
        print(f"    2. Download the {pres_year} general election ZIP")
        print(f"    3. Save to: {zip_cache}")
        print(f"    4. Re-run this script")
        return False

    # Step 2: Load election CSV
    print(f"\nStep 2: Load {pres_year} election returns from ZIP")
    df_elec = find_general_election_csv(zip_data, pres_year)
    if df_elec is None:
        return False

    # Step 3: Discover and download precinct mapping
    print(f"\nStep 3: Load precinct-to-district mapping ({map_filename})")
    map_cache = CACHE_DIR / map_filename

    map_url = None
    if not map_cache.exists() or force:
        map_url = find_precinct_map_url(map_year)

    map_data = download_with_cache(map_url, map_cache, force=force, min_size=10_000)
    if map_data is None:
        print(f"\n  FAILED: Could not obtain {map_filename}.")
        print(f"  Manual steps:")
        print(f"    1. Go to: https://data.capitol.texas.gov/dataset/precincts")
        print(f"    2. Download: {map_filename}")
        print(f"    3. Save to: {map_cache}")
        print(f"    4. Re-run this script")
        return False

    df_map, house_col, senate_col = load_precinct_mapping(map_data, map_info)
    if df_map is None:
        return False

    # Step 4: Aggregate
    print(f"\nStep 4: Aggregate presidential votes by district")
    house_df, senate_df = aggregate_presidential(
        df_elec, df_map, house_col, senate_col, pres_year)

    if house_df is not None:
        print(f"  House: {len(house_df)} / {map_info['n_house']} districts")
    if senate_df is not None:
        print(f"  Senate: {len(senate_df)} / {map_info['n_senate']} districts")

    # Statewide validation
    if house_df is not None:
        validate_statewide(house_df, pres_year)

    # Sample output
    if house_df is not None and len(house_df) > 0:
        print(f"\n  Sample House districts:")
        sample_districts = [1, 47, 100, 130, 144]
        for hd in sample_districts:
            row = house_df[house_df["district"] == hd]
            if not row.empty:
                r = row.iloc[0]
                print(f"    HD{hd:3d}: Dem {r['dem_pct']:.1f}%  "
                      f"Rep {r['rep_pct']:.1f}%  "
                      f"Dem 2p: {r['dem_pres_2p_baseline']:.3f}")

    # Step 5: Write output
    print(f"\nStep 5: Write output")
    if house_df is not None:
        out_house = DATA_HIST / f"tx_presidential_house_{pres_year}.csv"
        house_df.to_csv(out_house, index=False)
        print(f"  Wrote {out_house.name} ({len(house_df)} rows)")

    if senate_df is not None:
        out_senate = DATA_HIST / f"tx_presidential_senate_{pres_year}.csv"
        senate_df.to_csv(out_senate, index=False)
        print(f"  Wrote {out_senate.name} ({len(senate_df)} rows)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical presidential results by TX legislative district")
    parser.add_argument("--pres-year", type=int, choices=[2020, 2016],
                        help="Presidential year to collect (2020 or 2016). Default: both")
    parser.add_argument("--no-cache", action="store_true", help="Force re-download")
    args = parser.parse_args()

    years = [args.pres_year] if args.pres_year else [2020, 2016]

    print("=== Historical Presidential Results by TX Legislative District ===")
    print(f"Target years: {years}")
    print(f"Output: {DATA_HIST}\n")

    results = {}
    for pres_year in years:
        ok = collect_pres_year(pres_year, force=args.no_cache)
        results[pres_year] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for year, ok in results.items():
        status = "OK" if ok else "FAILED — manual download needed"
        map_year = 2022 if year == 2020 else 2018
        print(f"  {year} presidential → {map_year} districts: {status}")

    if not all(results.values()):
        print("\nFor failed years, download the ZIP manually from:")
        print("  https://data.capitol.texas.gov/dataset/comprehensive-election-datasets-compressed-format")
        print(f"  and save to: {CACHE_DIR}/{{year}}-general-vtds-election-data.zip")


if __name__ == "__main__":
    main()
