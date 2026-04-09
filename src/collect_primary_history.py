"""
collect_primary_history.py

Download and aggregate TX legislative primary results (2018, 2022, 2024)
from the OpenElections project (GitHub) for empirical analysis of whether
competitive primaries affect general election performance.

For each primary cycle and district, computes:
  - n_candidates     : how many candidates ran in that party's primary
  - winner_pct       : first-round winner's % of total primary votes
  - runner_up_pct    : second-place finisher's %  (0 if uncontested)
  - primary_margin   : winner_pct - runner_up_pct
  - runoff_needed    : True if winner_pct < 50 (TX 50% threshold)
  - had_runoff       : True if runoff-round data exists for this district/party

Source: OpenElections TX county-level precinct files
  https://github.com/openelections/openelections-data-tx

Output: data/raw/tx_primary_history.csv

USAGE:
  python src/collect_primary_history.py           # all years
  python src/collect_primary_history.py --year 2022
  python src/collect_primary_history.py --no-cache  # re-download even if cached
"""

import argparse
import csv
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CACHE    = DATA_RAW / "_primary_cache"
OUTPUT   = DATA_RAW / "tx_primary_history.csv"
CACHE.mkdir(parents=True, exist_ok=True)

GITHUB_API  = "https://api.github.com/repos/openelections/openelections-data-tx/contents"
HEADERS     = {"User-Agent": "TXLegPrimaryModel/1.0", "Accept": "application/vnd.github.v3+json"}

# Office strings in OpenElections TX data (varies by year)
HOUSE_LABELS  = {"State House", "State Representative"}
SENATE_LABELS = {"State Senate", "State Senator"}

# Target years and their primary dates (YYYYMMDD)
YEAR_PRIMARY_DATE = {
    2024: "20240305",
    2022: "20220301",
    2018: "20180306",
}
YEAR_RUNOFF_DATE = {
    2024: "20240528",
    2022: "20220524",
    2018: "20180522",
}

OUTPUT_COLS = [
    "year", "chamber", "district", "party",
    "n_candidates", "winner_name",
    "winner_votes", "total_votes", "winner_pct",
    "runner_up_pct", "primary_margin",
    "runoff_needed", "had_runoff",
]


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def _gh_list_files(year: int, primary_date: str, election_type: str = "primary") -> list[dict]:
    """Return list of county file metadata for a given year/election type."""
    url = f"{GITHUB_API}/{year}/counties"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    files = resp.json()

    prefix = f"{primary_date}__tx__{election_type}__"
    return [f for f in files if f["name"].startswith(prefix)]


def _download_file(file_meta: dict, use_cache: bool = True) -> str | None:
    """Download a single county file, using local cache if available."""
    fname = file_meta["name"]
    cache_path = CACHE / fname

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = file_meta["download_url"]
    try:
        resp = requests.get(url, headers={"User-Agent": "TXLegPrimaryModel/1.0"},
                            timeout=30)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8", errors="replace")
        return text
    except requests.RequestException as exc:
        print(f"    Warning: failed to download {fname}: {exc}", file=sys.stderr)
        return None


def _download_all(file_list: list[dict], use_cache: bool,
                  max_workers: int = 8) -> list[str]:
    """Download all county files in parallel, return list of CSV text strings."""
    results = [None] * len(file_list)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_download_file, f, use_cache): i
            for i, f in enumerate(file_list)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(file_list)} files downloaded...")
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_county_csv(text: str) -> list[dict]:
    """Parse one county CSV, return rows for State House and State Senate only."""
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        office = row.get("office", "").strip()
        if office not in HOUSE_LABELS and office not in SENATE_LABELS:
            continue
        candidate = row.get("candidate", "").strip()
        if not candidate:
            continue
        try:
            votes = int(row.get("votes", 0) or 0)
        except ValueError:
            votes = 0
        district_str = row.get("district", "").strip()
        try:
            district = int(district_str)
        except ValueError:
            continue
        party_raw = row.get("party", "").strip().upper()
        party = "R" if party_raw in ("REP", "R") else "D" if party_raw in ("DEM", "D") else None
        if party is None:
            continue
        chamber = "House" if office in HOUSE_LABELS else "Senate"
        rows.append({
            "chamber":   chamber,
            "district":  district,
            "party":     party,
            "candidate": candidate,
            "votes":     votes,
        })
    return rows


def _aggregate_to_district(all_rows: list[dict]) -> pd.DataFrame:
    """
    Aggregate precinct/county rows to (chamber, district, party, candidate) totals,
    then compute competitiveness metrics per (chamber, district, party).
    """
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    cand_totals = (
        df.groupby(["chamber", "district", "party", "candidate"], as_index=False)
        ["votes"].sum()
    )

    records = []
    for (ch, dist, pty), grp in cand_totals.groupby(["chamber", "district", "party"]):
        grp = grp.sort_values("votes", ascending=False).reset_index(drop=True)
        total     = grp["votes"].sum()
        n_cands   = len(grp)
        winner    = grp.iloc[0]
        w_votes   = int(winner["votes"])
        w_pct     = (w_votes / total * 100) if total > 0 else 0.0
        ru_pct    = float(grp.iloc[1]["votes"] / total * 100) if n_cands > 1 and total > 0 else 0.0
        margin    = w_pct - ru_pct
        records.append({
            "chamber":       ch,
            "district":      dist,
            "party":         pty,
            "n_candidates":  n_cands,
            "winner_name":   winner["candidate"],
            "winner_votes":  w_votes,
            "total_votes":   int(total),
            "winner_pct":    round(w_pct, 2),
            "runner_up_pct": round(ru_pct, 2),
            "primary_margin": round(margin, 2),
            "runoff_needed": w_pct < 50.0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Collect one year
# ---------------------------------------------------------------------------

def collect_year(year: int, use_cache: bool = True) -> pd.DataFrame:
    primary_date = YEAR_PRIMARY_DATE[year]
    runoff_date  = YEAR_RUNOFF_DATE[year]

    print(f"\n  Year {year} — primary {primary_date}")

    # --- Primary round ---
    try:
        primary_files = _gh_list_files(year, primary_date, "primary")
    except Exception as exc:
        print(f"    Could not list primary files: {exc}")
        return pd.DataFrame()

    print(f"    Downloading {len(primary_files)} county primary files...")
    primary_texts = _download_all(primary_files, use_cache)
    print(f"    Parsing {len(primary_texts)} files...")

    all_rows = []
    for text in primary_texts:
        all_rows.extend(_parse_county_csv(text))

    df = _aggregate_to_district(all_rows)
    if df.empty:
        print(f"    No legislative races found for {year}")
        return df

    print(f"    {year} primary: {len(df)} (chamber × district × party) records")

    # --- Runoff round (detect had_runoff) ---
    try:
        runoff_files = _gh_list_files(year, runoff_date, "runoff")
    except Exception:
        runoff_files = []

    runoff_districts: set[tuple] = set()
    if runoff_files:
        print(f"    Downloading {len(runoff_files)} county runoff files...")
        runoff_texts = _download_all(runoff_files, use_cache)
        runoff_rows = []
        for text in runoff_texts:
            runoff_rows.extend(_parse_county_csv(text))
        if runoff_rows:
            runoff_df = pd.DataFrame(runoff_rows)
            for _, row in runoff_df.iterrows():
                runoff_districts.add((row["chamber"], row["district"], row["party"]))
        print(f"    Runoff races found: {len(runoff_districts)}")
    else:
        # Fall back: infer from winner_pct < 50
        print(f"    No runoff files found — inferring had_runoff from winner_pct < 50%")
        for _, row in df.iterrows():
            if row["runoff_needed"]:
                runoff_districts.add((row["chamber"], row["district"], row["party"]))

    df["had_runoff"] = df.apply(
        lambda r: (r["chamber"], r["district"], r["party"]) in runoff_districts,
        axis=1
    )
    df["year"] = year

    # Filter to only on-ballot chambers/districts
    if year in (2022, 2018):
        # Senate: all 31 on ballot (redistricting year)
        pass  # keep all
    elif year == 2024:
        # Senate: only 2024-cycle seats (even-numbered districts, 2028 cycle; so 2024 had
        # districts 2,4,6,...30 on ballot — but we only care about TX House primaries for 2024
        # since the 2026-cycle Senate seats weren't up in 2024)
        pass

    return df[OUTPUT_COLS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect TX legislative primary history")
    parser.add_argument("--year", type=int, choices=[2018, 2022, 2024],
                        help="Collect only this year (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download even if cached locally")
    args = parser.parse_args()

    years = [args.year] if args.year else [2018, 2022, 2024]
    use_cache = not args.no_cache

    print("=" * 60)
    print("  TX Legislative Primary History — OpenElections")
    print("=" * 60)

    all_dfs = []
    for yr in years:
        df = collect_year(yr, use_cache)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["year", "chamber", "district", "party"])
    combined.to_csv(OUTPUT, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Written {len(combined)} rows → {OUTPUT.name}")

    # Summary
    for yr in combined["year"].unique():
        ydf = combined[combined["year"] == yr]
        contested   = (ydf["n_candidates"] > 1).sum()
        runoffs     = ydf["runoff_needed"].sum()
        had_runoff  = ydf["had_runoff"].sum()
        print(f"\n  {yr}:  {len(ydf)} party-districts  |  "
              f"{contested} contested  |  "
              f"{runoffs} runoff-eligible  |  "
              f"{had_runoff} confirmed runoffs")

    print()


if __name__ == "__main__":
    main()
