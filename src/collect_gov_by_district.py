"""
collect_gov_by_district.py

Collect Texas governor race results by state legislative district for
historical off-year cycles: 2002, 2006, 2010, 2014, 2018, 2022.

These are used as the "baseline partisanship" variable in the Phase 1
regression model.

Strategy (three-tier fallback):
  Tier 1 — Texas Legislative Council WRM PDFs
    URL pattern: https://wrm.capitol.texas.gov/fyiwebdocs/PDF/{chamber}/dist{N}/r{X}.pdf
    We try report numbers r4 and r5 (r8 is presidential; governor is likely r4 or r5).
    Uses pdfplumber to extract the table row containing "Governor".
    Caches each PDF to data/raw/historical/_tlc_pdf_{chamber}_{year}_dist{N}.pdf

  Tier 2 — Statewide fallback
    If TLC PDFs are unavailable or don't parse, record the statewide two-party
    governor share as a constant for all districts (weak but keeps the pipeline
    running). Flag with gov_source = "statewide_fallback".

  Tier 3 — Placeholder
    Write rows with MANUAL_NEEDED = True so build_phase1_dataset.py can flag
    incomplete rows. Provides the exact URL pattern to try manually.

OUTPUT: data/raw/historical/tx_gov_by_house_dist_{year}.csv
        data/raw/historical/tx_gov_by_senate_dist_{year}.csv
Columns:
  year, chamber, district, gov_r_candidate, gov_d_candidate,
  gov_r_pct, gov_d_pct, gov_dem_2p_share, gov_2p_note, gov_source, MANUAL_NEEDED

MANUAL DATA ENTRY NOTES:
  If TLC PDFs don't work, the best manual source is:
  - 2022: Texas Tribune district-level data or TLC redistricting analysis
  - 2018: TLC analysis for Plan H3067/S2100
  - 2014/2010/2006/2002: TLC historical election results by district
  See: https://wrm.capitol.texas.gov/ and https://redistricting.capitol.texas.gov/

  For each row where MANUAL_NEEDED=True, fill in:
    gov_r_pct, gov_d_pct, gov_dem_2p_share
  Then set MANUAL_NEEDED=False and gov_source="manual"

Usage:
  python src/collect_gov_by_district.py
  python src/collect_gov_by_district.py --year 2022
"""

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("WARNING: pdfplumber not installed. TLC PDF parsing unavailable.")
    print("  Install with: pip install pdfplumber")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_HIST.mkdir(parents=True, exist_ok=True)
PDF_CACHE = DATA_HIST / "_tlc_pdfs"
PDF_CACHE.mkdir(exist_ok=True)

TLC_BASE = "https://wrm.capitol.texas.gov/fyiwebdocs/PDF"
TLC_REPORT_CANDIDATES = ["r4", "r5", "r6"]  # try these report numbers for governor
TLC_CHAMBER_DIRS = {"house": "house", "senate": "senate"}
USER_AGENT = "TXLegislativeModel/1.0 (academic research; non-commercial)"

# ---------------------------------------------------------------------------
# Statewide governor two-party results (fallback constants)
# Source: Wikipedia / official TX SoS results
# 2006: 4-way race — using Perry/Bell two-party share only
# ---------------------------------------------------------------------------
STATEWIDE_GOV = {
    2002: {"r": "Rick Perry", "d": "Tony Sanchez", "r_pct": 57.8, "d_pct": 40.0,
           "note": "statewide_2party"},
    2006: {"r": "Rick Perry", "d": "Chris Bell", "r_pct": 39.0, "d_pct": 29.8,
           "note": "four_way_race_2006_rbell_only",
           "dem_2p_override": 29.8 / (39.0 + 29.8)},
    2010: {"r": "Rick Perry", "d": "Bill White", "r_pct": 55.0, "d_pct": 42.3,
           "note": "statewide_2party"},
    2014: {"r": "Greg Abbott", "d": "Wendy Davis", "r_pct": 59.3, "d_pct": 38.9,
           "note": "statewide_2party"},
    2018: {"r": "Greg Abbott", "d": "Lupe Valdez", "r_pct": 55.8, "d_pct": 42.5,
           "note": "statewide_2party"},
    2022: {"r": "Greg Abbott", "d": "Beto O'Rourke", "r_pct": 54.8, "d_pct": 43.8,
           "note": "statewide_2party"},
}

MAX_DISTRICTS = {"house": 150, "senate": 31}

# ---------------------------------------------------------------------------
# TLC PDF fetch + parse
# ---------------------------------------------------------------------------

def tlc_pdf_url(chamber: str, district: int, report: str) -> str:
    """Build TLC WRM PDF URL for a given district and report number."""
    return f"{TLC_BASE}/{TLC_CHAMBER_DIRS[chamber]}/dist{district}/{report}.pdf"


def fetch_tlc_pdf(chamber: str, district: int, report: str) -> bytes | None:
    """
    Fetch a TLC PDF (with caching). Returns raw bytes or None on failure.
    """
    if not PDF_AVAILABLE:
        return None

    cache_file = PDF_CACHE / f"{chamber}_{report}_dist{district:03d}.pdf"
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file.read_bytes()

    url = tlc_pdf_url(chamber, district, report)
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
            cache_file.write_bytes(resp.content)
            time.sleep(0.5)  # rate limit
            return resp.content
        return None
    except requests.RequestException:
        return None


def parse_gov_from_pdf(pdf_bytes: bytes, year: int) -> dict | None:
    """
    Extract governor race R/D percentages from a TLC district report PDF.

    TLC reports list races as rows in a table. We look for a row containing
    "Governor" and extract R/D vote totals or percentages from adjacent cells.

    Returns dict with r_pct, d_pct, r_candidate, d_candidate or None.
    """
    if not PDF_AVAILABLE or not pdf_bytes:
        return None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = ""
            all_tables = []
            for page in pdf.pages:
                full_text += page.extract_text() or ""
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)

        # Strategy 1: look for "Governor" in table rows
        for table in all_tables:
            for row in table:
                if row and any("governor" in str(cell).lower() for cell in row if cell):
                    # Find numeric cells in this row
                    nums = []
                    cells_clean = [str(c).strip() if c else "" for c in row]
                    for cell in cells_clean:
                        m = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', cell)
                        if m:
                            nums.append(float(m.group(1).replace(",", "")))

                    if len(nums) >= 2:
                        # Determine if these are vote counts or percentages
                        # Percentages are typically < 100; counts are larger
                        if all(n <= 100 for n in nums[:2]):
                            r_pct, d_pct = nums[0], nums[1]
                        else:
                            # Vote counts — compute percentages
                            total = sum(nums[:2])
                            r_pct = round(nums[0] / total * 100, 2) if total > 0 else None
                            d_pct = round(nums[1] / total * 100, 2) if total > 0 else None
                        return {"r_pct": r_pct, "d_pct": d_pct, "source": "tlc_pdf_table"}

        # Strategy 2: regex on full text
        gov_m = re.search(
            r'Governor.*?(\d{1,3}(?:,\d{3})*)\s+(\d{1,3}(?:,\d{3})*)',
            full_text,
            re.DOTALL | re.IGNORECASE,
        )
        if gov_m:
            r_votes = int(gov_m.group(1).replace(",", ""))
            d_votes = int(gov_m.group(2).replace(",", ""))
            total = r_votes + d_votes
            if total > 0:
                return {
                    "r_pct": round(r_votes / total * 100, 2),
                    "d_pct": round(d_votes / total * 100, 2),
                    "source": "tlc_pdf_text",
                }

        return None

    except Exception:
        return None


def try_tlc_district(chamber: str, district: int, year: int) -> dict | None:
    """
    Attempt to fetch and parse TLC PDF for one district across report number variants.
    Returns parsed result dict or None if all attempts fail.
    """
    for report in TLC_REPORT_CANDIDATES:
        pdf_bytes = fetch_tlc_pdf(chamber, district, report)
        if pdf_bytes:
            result = parse_gov_from_pdf(pdf_bytes, year)
            if result:
                return result
    return None


# ---------------------------------------------------------------------------
# Build result rows
# ---------------------------------------------------------------------------

def build_placeholder_row(year: int, chamber: str, district: int) -> dict:
    """Create a row flagged MANUAL_NEEDED with statewide fallback values."""
    gov = STATEWIDE_GOV.get(year, {})
    r_pct = gov.get("r_pct")
    d_pct = gov.get("d_pct")
    total = (r_pct or 0) + (d_pct or 0)

    if "dem_2p_override" in gov:
        dem_2p = round(gov["dem_2p_override"], 6)
    elif total > 0:
        dem_2p = round(d_pct / total, 6)
    else:
        dem_2p = None

    return {
        "year": year,
        "chamber": chamber.title(),
        "district": district,
        "gov_r_candidate": gov.get("r", ""),
        "gov_d_candidate": gov.get("d", ""),
        "gov_r_pct": r_pct,
        "gov_d_pct": d_pct,
        "gov_dem_2p_share": dem_2p,
        "gov_2p_note": gov.get("note", "") + "|statewide_fallback",
        "gov_source": "statewide_fallback",
        "MANUAL_NEEDED": True,
    }


def collect_year_chamber(year: int, chamber: str, use_tlc: bool = True) -> list[dict]:
    """
    Collect governor-by-district data for one year/chamber combination.
    Returns list of row dicts.
    """
    gov = STATEWIDE_GOV.get(year, {})
    max_d = MAX_DISTRICTS[chamber]
    rows = []

    tlc_success = 0
    tlc_fail = 0

    # Test TLC availability with district 1 before attempting all districts
    tlc_available = False
    if use_tlc and PDF_AVAILABLE:
        print(f"  Testing TLC PDF availability for {chamber} district 1...")
        test_result = try_tlc_district(chamber, 1, year)
        tlc_available = test_result is not None
        if tlc_available:
            print(f"  TLC PDFs available — fetching all {max_d} districts.")
        else:
            print(f"  TLC PDFs unavailable for {chamber} {year}.")
            print(f"  Manual source: {tlc_pdf_url(chamber, 1, 'r4')} (try r4, r5, r6)")
            print(f"  Falling back to statewide governor constants.")

    for district in range(1, max_d + 1):
        if tlc_available:
            result = try_tlc_district(chamber, district, year)
            if result:
                r_pct = result["r_pct"]
                d_pct = result["d_pct"]
                total = r_pct + d_pct
                dem_2p = round(d_pct / total, 6) if total > 0 else None
                # Special handling for 2006 four-way race
                note = ""
                if year == 2006:
                    note = "four_way_race_2006_rbell_only"
                rows.append({
                    "year": year,
                    "chamber": chamber.title(),
                    "district": district,
                    "gov_r_candidate": gov.get("r", ""),
                    "gov_d_candidate": gov.get("d", ""),
                    "gov_r_pct": r_pct,
                    "gov_d_pct": d_pct,
                    "gov_dem_2p_share": dem_2p,
                    "gov_2p_note": note,
                    "gov_source": result["source"],
                    "MANUAL_NEEDED": False,
                })
                tlc_success += 1
                continue
            else:
                tlc_fail += 1

        # Statewide fallback
        rows.append(build_placeholder_row(year, chamber, district))

    if tlc_available and tlc_fail > 0:
        print(f"  TLC: {tlc_success} succeeded, {tlc_fail} fell back to statewide constant")

    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fields = ["year", "chamber", "district", "gov_r_candidate", "gov_d_candidate",
              "gov_r_pct", "gov_d_pct", "gov_dem_2p_share", "gov_2p_note",
              "gov_source", "MANUAL_NEEDED"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manual_count = sum(1 for r in rows if r["MANUAL_NEEDED"])
    print(f"  Wrote {len(rows)} rows → {path.name}  ({manual_count} MANUAL_NEEDED)")


def summarize_all(year: int, chambers: list[str]):
    """Print a note about what to fill in manually if needed."""
    print(f"\n  DATA GAP NOTE for {year}:")
    gov = STATEWIDE_GOV.get(year, {})
    print(f"    Statewide governor: {gov.get('r','?')} (R) {gov.get('r_pct','?')}%"
          f" vs {gov.get('d','?')} (D) {gov.get('d_pct','?')}%")
    if year == 2006:
        print(f"    *** 2006 was a 4-way race. Two-party share uses Perry/Bell only.")
        print(f"        District-level data especially needed here (Strayhorn/Friedman varied by district).")
    print(f"    Best source for district-level data:")
    print(f"      TLC WRM: https://wrm.capitol.texas.gov/")
    print(f"      TLC Redistricting: https://redistricting.capitol.texas.gov/")
    print(f"      Texas Tribune election results by district")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
CHAMBERS = ["house", "senate"]


def main(years=None, chambers=None, use_tlc: bool = True):
    years = years or YEARS
    chambers = chambers or CHAMBERS

    for year in years:
        print(f"\n{'='*60}")
        print(f"Collecting governor by district: TX {year}")
        print(f"{'='*60}")
        for chamber in chambers:
            rows = collect_year_chamber(year, chamber, use_tlc=use_tlc)
            out = DATA_HIST / f"tx_gov_by_{chamber}_dist_{year}.csv"
            write_csv(rows, out)
        summarize_all(year, chambers)

    print(f"\n{'='*60}")
    print("Governor data collection complete.")
    print(f"Output directory: {DATA_HIST}")
    print("\nIMPORTANT: Rows with MANUAL_NEEDED=True use statewide constants,")
    print("not district-level values. Fill these in for accurate baseline partisanship.")
    print("The regression will still run using statewide fallbacks, but accuracy")
    print("will be lower for districts with atypical partisan lean.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect TX governor results by district")
    parser.add_argument("--year", type=int, choices=YEARS)
    parser.add_argument("--chamber", choices=CHAMBERS)
    parser.add_argument("--no-tlc", action="store_true", help="Skip TLC PDF attempts")
    args = parser.parse_args()

    years = [args.year] if args.year else None
    chambers = [args.chamber] if args.chamber else None
    main(years=years, chambers=chambers, use_tlc=not args.no_tlc)
