"""
collect_cvap_by_district.py

Download ACS 5-year Citizen Voting-Age Population (CVAP) data for
Texas state legislative districts and compute racial/ethnic composition.

Uses CVAP not total population because in heavily Hispanic TX border districts,
total population significantly overstates the voting-eligible electorate due
to non-citizen residents.

Source:
  Census Bureau ACS CVAP Special Tabulation
  https://www.census.gov/programs-surveys/decennial-census/about/rdo/summary-files.html

  The CVAP Special Tabulation is published as ZIP files containing CSV tables.
  We use the "BlockGr.csv" or state legislative district table.

  Alternative direct URL (StateLegLower and StateLegUpper tables):
  https://www2.census.gov/programs-surveys/decennial/rdo/datasets/{year}/CVAP_{year}_ACS{N}YR_csv_files.zip

Output:
  data/raw/tx_cvap_house.csv   — 150 rows (House districts 1-150)
  data/raw/tx_cvap_senate.csv  — 31 rows  (Senate districts 1-31)

Columns:
  chamber, district, cvap_total,
  cvap_white_nh, cvap_black_nh, cvap_hispanic, cvap_asian_nh, cvap_aian_nh, cvap_other,
  pct_white_nh, pct_black_nh, pct_hispanic, pct_asian_nh, pct_other,
  acs_year, data_source

Usage:
  python src/collect_cvap_by_district.py
  python src/collect_cvap_by_district.py --inspect   # show raw file structure
  python src/collect_cvap_by_district.py --no-cache  # force re-download
  python src/collect_cvap_by_district.py --year 2022 # use 2022 5-year ACS (2018-2022)
"""

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
CACHE_DIR = DATA_RAW / "_cvap_cache"

# ACS CVAP Special Tabulation ZIP URL pattern
# {year} = last year of the 5-year period (e.g. 2023 = 2019-2023 ACS)
# URL format: .../datasets/{year}/{year}-cvap/CVAP_{year-4}-{year}_ACS_csv_files.zip
def _cvap_url(year: int) -> str:
    return (
        f"https://www2.census.gov/programs-surveys/decennial/rdo/datasets/"
        f"{year}/{year}-cvap/CVAP_{year - 4}-{year}_ACS_csv_files.zip"
    )

# Texas FIPS code
TX_FIPS = "48"

# CVAP table file names inside the ZIP (Census naming)
# SLDLC = State Legislative District Lower Chamber (House)
# SLDUC = State Legislative District Upper Chamber (Senate)
TABLE_FILES = {
    "house": "SLDLC.csv",
    "senate": "SLDUC.csv",
}

# CVAP lntitle values that identify racial/ethnic groups
# Actual Census CVAP 2019-2023 lntitle strings (from SLDLC.csv inspection)
CVAP_GROUPS = {
    "total":      "Total",
    "not_hisp":   "Not Hispanic or Latino",       # non-Hispanic universe
    "white_nh":   "White Alone",                   # non-Hisp white (within Not Hispanic rows)
    "black_nh":   "Black or African American Alone",
    "hispanic":   "Hispanic or Latino",
    "asian_nh":   "Asian Alone",
    "aian_nh":    "American Indian or Alaska Native Alone",
    "nhpi_nh":    "Native Hawaiian or Other Pacific Islander Alone",
    "two_plus":   "Remainder of Two or More Race Responses",
}

USER_AGENT = "TX-Legislature-Model/1.0 (research; contact via GitHub)"
TIMEOUT = 300


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_with_cache(url: str, cache_path: Path, force: bool = False) -> bytes | None:
    if not force and cache_path.exists() and cache_path.stat().st_size > 10_000:
        print(f"  Using cached: {cache_path.name} ({cache_path.stat().st_size:,} bytes)")
        return cache_path.read_bytes()

    print(f"  Downloading: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT},
                         timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        data = r.content
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        print(f"  Saved: {cache_path.name} ({len(data):,} bytes)")
        return data
    except requests.RequestException as exc:
        print(f"  ERROR downloading {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Load CVAP data for TX
# ---------------------------------------------------------------------------

def load_cvap_table(zip_data: bytes, table_file: str) -> pd.DataFrame | None:
    """Extract and read a single CVAP table CSV from the ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            names = zf.namelist()
            # Find the target file (may be in a subdirectory)
            matches = [n for n in names if n.endswith(table_file)]
            if not matches:
                print(f"  ERROR: {table_file} not found in ZIP")
                print(f"  Available files: {names[:20]}")
                return None
            target = matches[0]
            print(f"  Reading: {target}")
            with zf.open(target) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
            print(f"  Loaded {len(df):,} rows, columns: {list(df.columns)}")
            return df
    except zipfile.BadZipFile as exc:
        print(f"  ERROR: bad ZIP: {exc}")
        return None


def filter_texas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Texas state legislative district rows.
    TX state FIPS = 48. GEOID format: '620L800US48NNN' (SLDL) or '620U800US48NN' (SLDU).
    We filter using 'US48' which is unambiguous for Texas.
    """
    geoid_col = next((c for c in df.columns if c.lower() == "geoid"), None)
    if geoid_col is None:
        print("  WARNING: 'geoid' column not found — returning all rows")
        return df

    # 'US48' appears in TX geoids but not in other states (48 = Texas FIPS)
    tx_mask = df[geoid_col].astype(str).str.contains("US48", na=False)
    result = df[tx_mask].copy()
    print(f"  Texas rows: {len(result):,} (from {len(df):,} total)")
    return result


# ---------------------------------------------------------------------------
# Parse CVAP groups
# ---------------------------------------------------------------------------

def parse_cvap_for_chamber(zip_data: bytes, table_file: str, chamber: str) -> pd.DataFrame | None:
    """
    Load CVAP table, filter to TX, pivot racial groups to columns.
    Returns one row per district with CVAP totals and percentages.

    GEOID format (SLDLC): '620L800US48NNN' where NNN = district number (zero-padded)
    GEOID format (SLDUC): '620U800US48NN' where NN = district number (zero-padded)
    """
    df = load_cvap_table(zip_data, table_file)
    if df is None:
        return None

    df = filter_texas(df)
    if df.empty:
        print(f"  ERROR: no TX rows found in {table_file}")
        return None

    # CVAP format: geoid, geoname, lntitle, cvap_est (plus tot_est, adu_est, cit_est)
    lntitle_col = "lntitle"
    cvap_col = "cvap_est"
    geoid_col = "geoid"

    df[cvap_col] = pd.to_numeric(df[cvap_col], errors="coerce").fillna(0)

    # Extract district number: last 3 digits of geoid (SLDLC) or last 2 (SLDUC)
    # e.g. '620L800US48047' → 47, '620U800US4809' → 9
    def geoid_to_district(geoid: str) -> int | None:
        try:
            suffix = geoid.split("US48")[-1]  # everything after 'US48'
            return int(suffix.lstrip("0") or "0")
        except Exception:
            return None

    df["_district"] = df[geoid_col].astype(str).apply(geoid_to_district)

    # Build records: {district: {lntitle: cvap_est}}
    records: dict[int, dict[str, float]] = {}
    n_districts = 150 if chamber == "house" else 31
    for _, row in df.iterrows():
        dist = row["_district"]
        if dist is None or dist < 1 or dist > n_districts:
            continue
        if dist not in records:
            records[dist] = {}
        records[dist][str(row[lntitle_col]).strip()] = float(row[cvap_col])

    if not records:
        print("  ERROR: no valid district rows parsed")
        return None

    print(f"  Parsed {len(records)} districts")

    # Build output DataFrame
    rows = []
    for district in range(1, n_districts + 1):
        grp = records.get(district, {})
        total = grp.get(CVAP_GROUPS["total"], 0)
        white_nh = grp.get(CVAP_GROUPS["white_nh"], 0)
        black_nh = grp.get(CVAP_GROUPS["black_nh"], 0)
        hispanic = grp.get(CVAP_GROUPS["hispanic"], 0)
        asian_nh = grp.get(CVAP_GROUPS["asian_nh"], 0)
        aian_nh = grp.get(CVAP_GROUPS["aian_nh"], 0)
        nhpi_nh = grp.get(CVAP_GROUPS["nhpi_nh"], 0)
        two_plus = grp.get(CVAP_GROUPS["two_plus"], 0)

        # "other" = Asian + AIAN + NHPI + two-or-more (everything non-white, non-Black, non-Hispanic)
        other_combined = asian_nh + aian_nh + nhpi_nh + two_plus

        def pct(num: float) -> float | None:
            return round(num / total * 100, 2) if total > 0 else None

        rows.append({
            "chamber": chamber,
            "district": district,
            "cvap_total": int(total),
            "cvap_white_nh": int(white_nh),
            "cvap_black_nh": int(black_nh),
            "cvap_hispanic": int(hispanic),
            "cvap_asian_nh": int(asian_nh),
            "cvap_aian_nh": int(aian_nh),
            "cvap_other": int(other_combined),
            "pct_white_nh": pct(white_nh),
            "pct_black_nh": pct(black_nh),
            "pct_hispanic": pct(hispanic),
            "pct_asian_nh": pct(asian_nh),
            "pct_other": pct(other_combined),
            "data_source": "ACS_CVAP_5yr",
            "MANUAL_NEEDED": total == 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inspect mode
# ---------------------------------------------------------------------------

def inspect_files(zip_data: bytes):
    """Print CVAP ZIP structure to help diagnose column names."""
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        print(f"\nZIP contents ({len(zf.namelist())} files):")
        for name in zf.namelist():
            print(f"  {name} ({zf.getinfo(name).file_size:,} bytes)")

    for chamber, table_file in TABLE_FILES.items():
        print(f"\n=== {chamber.upper()} ({table_file}) ===")
        df = load_cvap_table(zip_data, table_file)
        if df is None:
            continue
        print(f"Columns: {list(df.columns)}")
        print(f"First 5 rows:\n{df.head().to_string()}")
        lntitle_col = next((c for c in df.columns if "lntitle" in c.lower()), None)
        if lntitle_col:
            unique_lntitles = df[lntitle_col].dropna().unique()[:20]
            print(f"\nUnique lntitle values: {list(unique_lntitles)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect ACS CVAP by TX legislative district")
    parser.add_argument("--year", type=int, default=2023,
                        help="Last year of ACS 5-year period (default: 2023 = 2019-2023)")
    parser.add_argument("--inspect", action="store_true",
                        help="Print ZIP structure and sample data then exit")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download even if cached")
    args = parser.parse_args()

    url = _cvap_url(args.year)
    cache_path = CACHE_DIR / f"CVAP_{args.year - 4}-{args.year}_ACS_csv_files.zip"

    print(f"=== ACS CVAP Collection ({args.year} 5-year) ===\n")
    print(f"Step 1: Download CVAP ZIP")
    zip_data = download_with_cache(url, cache_path, force=args.no_cache)
    if zip_data is None:
        print("FAILED: could not download CVAP data")
        sys.exit(1)

    if args.inspect:
        inspect_files(zip_data)
        return

    results = {}
    for chamber, table_file in TABLE_FILES.items():
        print(f"\nStep 2: Parse {chamber} districts")
        df = parse_cvap_for_chamber(zip_data, table_file, chamber)
        if df is None:
            print(f"  FAILED for {chamber}")
            continue

        n_missing = df["MANUAL_NEEDED"].sum()
        print(f"  {chamber.capitalize()}: {len(df)} districts, {n_missing} missing CVAP data")
        results[chamber] = df

        # Sample output
        sample_dists = [13, 47, 100, 130] if chamber == "house" else [9, 19, 26]
        sample = df[df["district"].isin(sample_dists)]
        for _, row in sample.iterrows():
            print(f"    {chamber[0].upper()}D{row['district']:3d}: "
                  f"total={row['cvap_total']:,}  "
                  f"White={row['pct_white_nh']:.0f}%  "
                  f"Black={row['pct_black_nh']:.0f}%  "
                  f"Hispanic={row['pct_hispanic']:.0f}%  "
                  f"Other={row['pct_other']:.0f}%")

    print("\nStep 3: Write output")
    for chamber, df in results.items():
        out_path = DATA_RAW / f"tx_cvap_{chamber}.csv"
        df["acs_year"] = args.year
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path.name}")

    # Statewide sanity check
    if "house" in results:
        h = results["house"]
        total_cvap = h["cvap_total"].sum()
        if total_cvap > 0:
            print(f"\nStalewide TX CVAP composition (from House districts):")
            for grp in ("white_nh", "black_nh", "hispanic", "other"):
                col = f"cvap_{grp}"
                if col in h.columns:
                    share = h[col].sum() / total_cvap * 100
                    print(f"  {grp:12s}: {share:.1f}%")


if __name__ == "__main__":
    main()
