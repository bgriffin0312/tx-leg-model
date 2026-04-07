"""
collect_ies_pac.py

Extract "true" independent expenditure (IE) data for TX legislative races from
PAC expenditures with district-specific descriptions in the TEC expend_*.csv files.

Unlike collect_ies_historical.py (which uses TEC's spacs.csv SPAC declarations),
this script targets specific PACs known to make large election-related expenditures
in TX legislative races, parsing district numbers and direction from expenditure
description text (expendDescr).

WHY THIS MATTERS:
  TEC "SPACs" are committees that officially declare SUPPORT/OPPOSE for a candidate.
  Many SPACs are candidate-affiliated "Friends of X" committees — not outside groups.
  True outside-group IEs (party caucus PACs, business advocacy groups) often file
  as regular PACs and identify their targets only in the expenditure description.

KEY PAC FILERS TARGETED:
  Known directional (direction from PAC identity):
    00058081 — Texas Republican Legislative Campaign Committee   [R-favor]
    00084976 — RSLC Grassroots Account (Republican)             [R-favor]
    00055005 — House Democratic Campaign Committee              [D-favor]
    00068897 — Battleground Texas                               [D-favor]
    00054804 — Texans for Insurance Reform                      [parse]
  Parsed directional (direction from description text):
    00028135 — Texans for Lawsuit Reform PAC
    00016623 — Texas Farm Bureau AGFUND
    00015487 — Texas REALTORS PAC

PARSING STRATEGY:
  1. Extract district number from expendDescr via regex:
       HD/SD patterns: "HD 54", "HD-54", "H.D. 54", "HOUSE-DIST 10"
  2. Determine direction:
       - For party caucus PACs: direction fixed by PAC identity
       - For issue PACs: parse "oppose"/"support"/"in-kind" from description
         + candidate party from "(R)"/"(D)" or name→district lookup
  3. Filter to election cycle date windows
  4. Aggregate D_favor / R_favor dollars by (year, chamber, district)

OUTPUT (same schema as tx_ies_{year}.csv, data_source="tec_pac_expend"):
  data/raw/historical/tx_ies_pac_{year}.csv
  data/raw/historical/tx_ies_pac_summary.csv

USAGE:
  python src/collect_ies_pac.py              # all years 2002-2022
  python src/collect_ies_pac.py --year 2022  # single year
  python src/collect_ies_pac.py --scan       # scan for high-value PAC descriptions
  python src/collect_ies_pac.py --merge      # after running, merge with SPAC files
"""

import argparse
import csv
import io
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_HIST.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from collect_finance import (
    _tec_zip_central_dir,
    _tec_extract_file,
    TEC_ZIP_URL,
    TEC_ENCODING,
)

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]

# Election cycle windows (inclusive): Jan 1 → day after Election Day
CYCLE_WINDOWS = {
    2002: ("20020101", "20021106"),
    2006: ("20060101", "20061108"),
    2010: ("20100101", "20101103"),
    2014: ("20140101", "20141105"),
    2018: ("20180101", "20181107"),
    2022: ("20220101", "20221109"),
}

# IE significance threshold
IE_FLAG_THRESHOLD = 25_000

# ---------------------------------------------------------------------------
# PAC Registry
# ---------------------------------------------------------------------------
# default_direction: "R_favor" | "D_favor" | "parse"
#   "parse" = extract from expendDescr (need both direction + party markers)
#
# For "parse" PACs:
#   - Explicit "oppose/support" keywords + "(R)"/"(D)" party markers needed
#   - Rows with parseable district but ambiguous direction → "unknown" (skipped)
# ---------------------------------------------------------------------------
PAC_REGISTRY = {
    # Party caucus PACs — direction fixed by PAC identity
    "00058081": {
        "name":              "Texas Republican Legislative Campaign Committee",
        "default_direction": "R_favor",
        "party":             "R",
    },
    "00084976": {
        "name":              "RSLC Grassroots Account",
        "default_direction": "R_favor",
        "party":             "R",
    },
    "00055005": {
        "name":              "House Democratic Campaign Committee",
        "default_direction": "D_favor",
        "party":             "D",
    },
    "00068897": {
        "name":              "Battleground Texas",
        "default_direction": "D_favor",
        "party":             "D",
    },
    # Issue PACs — direction parsed from description
    "00028135": {
        "name":              "Texans for Lawsuit Reform PAC",
        "default_direction": "parse",
        "party":             None,
    },
    "00016623": {
        "name":              "Texas Farm Bureau AGFUND",
        "default_direction": "parse",
        "party":             None,
    },
    "00054804": {
        "name":              "Texans for Insurance Reform",
        "default_direction": "parse",
        "party":             None,
    },
    "00015487": {
        "name":              "Texas REALTORS PAC",
        "default_direction": "parse",
        "party":             None,
    },
}

PAC_FILER_IDS = set(PAC_REGISTRY.keys())

OUTPUT_FIELDS = [
    "chamber", "district",
    "ie_d_favor", "ie_r_favor", "ie_total",
    "ie_dem_share",
    "ie_log_total",
    "ie_flag",
    "n_spacs",       # kept for schema compatibility; here = n_pacs
    "data_source",
]

# ---------------------------------------------------------------------------
# Regex patterns for description parsing
# ---------------------------------------------------------------------------

# House district: "HD 54", "HD-54", "HD54", "H.D. 54", "(HD43)", "DIST 54 H"
_HD_PATS = [
    re.compile(r'\bH\.?D\.?\s*[-#]?\s*(\d{1,3})\b', re.I),
    re.compile(r'\bHOUSE[-\s]+DIST(?:RICT)?\s+(\d{1,3})\b', re.I),
    re.compile(r'\bHOUSE\s+DISTRICT\s+(\d{1,3})\b', re.I),
    re.compile(r'\bSTATEREP\s+DIST(?:RICT)?\s+(\d{1,3})\b', re.I),
]

# Senate district: "SD 25", "SD-25", "S.D. 25", "SENATE-DIST 23"
_SD_PATS = [
    re.compile(r'\bS\.?D\.?\s*[-#]?\s*(\d{1,2})\b', re.I),
    re.compile(r'\bSENATE[-\s]+DIST(?:RICT)?\s+(\d{1,2})\b', re.I),
    re.compile(r'\bSENATE\s+DISTRICT\s+(\d{1,2})\b', re.I),
    re.compile(r'\bSTATESEN\s+DIST(?:RICT)?\s+(\d{1,2})\b', re.I),
]

# Direction
_OPPOSE_PAT  = re.compile(r'\boppos', re.I)
_SUPPORT_PAT = re.compile(
    r'\b(support|in.?kind|inkind|contrib(?:ution|uted|ute|ing|ibut)?|donated?)\b', re.I
)

# Party from description: "(R)", "- R)", "(D)", "- D", "party R", "party D"
_PARTY_R_PAT = re.compile(r'[\(\-\s]R[\)\s,;]|\bparty\s+R\b', re.I)
_PARTY_D_PAT = re.compile(r'[\(\-\s]D[\)\s,;]|\bparty\s+D\b', re.I)

# TFB "SUPPORT OFFICEHOLDER" / "OPPOSE OFFICEHOLDER" format
_TFB_DIR_PAT = re.compile(r'\b(SUPPORT|OPPOSE)\s+(?:OFFICEHOLDER|CANDIDATE|INCUMBENT)', re.I)


def extract_district_from_descr(descr: str) -> tuple[str | None, int | None]:
    """
    Parse chamber and district number from an expenditure description.
    Returns ("house"|"senate"|None, district_number|None).
    """
    for pat in _HD_PATS:
        m = pat.search(descr)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 150:
                return "house", num

    for pat in _SD_PATS:
        m = pat.search(descr)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 31:
                return "senate", num

    return None, None


def extract_direction_from_descr(descr: str, pac_default: str, pac_party: str | None) -> str:
    """
    Determine if expenditure is D_favor or R_favor.

    For party caucus PACs: return pac_default directly.
    For issue PACs: parse opposition/support keywords + candidate party markers.

    Returns "D_favor", "R_favor", or "unknown".
    """
    if pac_default in ("D_favor", "R_favor"):
        return pac_default

    # Check for TFB-style explicit SUPPORT/OPPOSE in description
    tfb = _TFB_DIR_PAT.search(descr)
    if tfb:
        action = tfb.group(1).upper()
        # TFB supports incumbents; need party from description or district fallback
        has_r = bool(_PARTY_R_PAT.search(descr))
        has_d = bool(_PARTY_D_PAT.search(descr))
        if action == "OPPOSE" and has_r:
            return "D_favor"
        if action == "OPPOSE" and has_d:
            return "R_favor"
        if action == "SUPPORT" and has_r:
            return "R_favor"
        if action == "SUPPORT" and has_d:
            return "D_favor"
        # TFB often doesn't mark party; direction ambiguous
        return "unknown"

    # Parse oppose vs. support
    is_oppose  = bool(_OPPOSE_PAT.search(descr))
    is_support = bool(_SUPPORT_PAT.search(descr)) and not is_oppose

    if not is_oppose and not is_support:
        return "unknown"

    # Parse candidate party from description
    has_r = bool(_PARTY_R_PAT.search(descr))
    has_d = bool(_PARTY_D_PAT.search(descr))

    if is_oppose and has_r:
        return "D_favor"
    if is_oppose and has_d:
        return "R_favor"
    if is_support and has_r:
        return "R_favor"
    if is_support and has_d:
        return "D_favor"

    # Support something but no party marker — ambiguous
    return "unknown"


# ---------------------------------------------------------------------------
# Build candidate name → (chamber, district, party) lookup from election results
# ---------------------------------------------------------------------------

def build_name_district_map() -> dict[str, tuple[str, int, str]]:
    """
    Build a fuzzy name → (chamber, district, party) mapping from historical
    election results. Used to resolve descriptions like
    "direct mail in kind donation Nathan Macias HD 73" where HD is already
    in the description, but also for names without district numbers.

    Returns {normalized_last_name: (chamber, district, winner_party)}.
    Only includes names that are unique across all years/districts.
    """
    name_map: dict[str, list[tuple[str, int, str]]] = defaultdict(list)

    for year in YEARS:
        for ch in ("house", "senate"):
            path = DATA_HIST / f"tx_{ch}_results_{year}.csv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dist = row.get("district", "").strip()
                    if not dist.isdigit():
                        continue
                    for col_prefix, party in [("r_candidate", "R"), ("d_candidate", "D")]:
                        name = row.get(col_prefix, "").strip()
                        if not name:
                            continue
                        # Normalize: lowercase last name only
                        last = name.split()[-1].lower() if name.split() else ""
                        if last:
                            name_map[last].append((ch, int(dist), party))

    # Keep only names that uniquely identify a single (chamber, district, party)
    unique: dict[str, tuple[str, int, str]] = {}
    for last, entries in name_map.items():
        # Deduplicate by (chamber, district, party)
        deduped = list({(ch, d, p) for ch, d, p in entries})
        if len(deduped) == 1:
            unique[last] = deduped[0]

    return unique


def lookup_name_in_descr(descr: str, name_map: dict) -> tuple[str | None, int | None, str | None]:
    """
    Try to match candidate last name from description to known candidates.
    Returns (chamber, district, party) or (None, None, None).
    """
    words = re.findall(r'[A-Za-z]+', descr)
    for word in words:
        key = word.lower()
        if key in name_map and len(key) >= 4:  # avoid short common words
            return name_map[key]
    return None, None, None


# ---------------------------------------------------------------------------
# Load PAC expenditures from expend_*.csv
# ---------------------------------------------------------------------------

def load_pac_expenditures(cd: dict) -> list[dict]:
    """
    Extract expenditure rows from expend_*.csv where filerIdent is in PAC_FILER_IDS.
    Includes filerIdent, expendDt, expendAmount, and expendDescr.
    Caches files after first download (reuses collect_finance.py cache).
    """
    expend_files = sorted(f for f in cd if re.match(r"expend_\d+\.csv", f, re.IGNORECASE))
    if not expend_files:
        print("  WARNING: No expend_*.csv files found in TEC ZIP")
        return []

    print(f"\nExtracting expenditures from {len(expend_files)} expend files "
          f"for {len(PAC_FILER_IDS)} target PAC filers...")

    all_rows = []
    for fname in expend_files:
        data = _tec_extract_file(TEC_ZIP_URL, cd[fname], fname)
        if not data:
            print(f"  WARNING: Could not extract {fname}")
            continue

        text   = data.decode(TEC_ENCODING, errors="replace")
        reader = csv.DictReader(io.StringIO(text))

        file_count = 0
        for row in reader:
            filer_id = row.get("filerIdent", "").strip()
            if filer_id not in PAC_FILER_IDS:
                continue

            expend_dt = re.sub(r"\D", "", row.get("expendDt", "").strip())[:8]
            if len(expend_dt) < 8:
                continue

            amount_raw = (row.get("expendAmount", "") or "0").strip().replace(",", "").replace("$", "")
            try:
                amount = float(amount_raw)
            except ValueError:
                continue
            if amount <= 0:
                continue

            descr = (row.get("expendDescr", "") or "").strip()

            all_rows.append({
                "filer_id":    filer_id,
                "expend_date": expend_dt,
                "amount":      amount,
                "descr":       descr,
            })
            file_count += 1

        if file_count:
            print(f"  {fname}: {file_count} rows from target PAC filers")

    print(f"  Total PAC rows across all expend files: {len(all_rows):,}")
    return all_rows


# ---------------------------------------------------------------------------
# Parse and aggregate
# ---------------------------------------------------------------------------

def _date_to_election_year(date_str: str) -> int | None:
    if len(date_str) < 8:
        return None
    for year, (start, end) in CYCLE_WINDOWS.items():
        if start <= date_str <= end:
            return year
    return None


def parse_and_aggregate(all_rows: list[dict], name_map: dict) -> dict[int, list[dict]]:
    """
    For each expenditure row:
      1. Extract district from description
      2. Determine direction (D_favor / R_favor)
      3. Map date to election year
      4. Aggregate by (year, chamber, district)

    Returns {year: [row_dict, ...]}
    """
    year_dist: dict[tuple, dict] = defaultdict(
        lambda: {"D_favor": 0.0, "R_favor": 0.0, "n_pacs": set()}
    )

    stats = {
        "total": 0, "no_district": 0, "unknown_direction": 0,
        "out_of_window": 0, "included": 0,
    }
    parse_log: dict[str, dict] = defaultdict(
        lambda: {"included": 0, "no_district": 0, "unknown_direction": 0, "dollars": 0.0}
    )

    for row in all_rows:
        stats["total"] += 1
        fid     = row["filer_id"]
        descr   = row["descr"]
        pac     = PAC_REGISTRY[fid]

        # Step 1: extract district from description
        chamber, district = extract_district_from_descr(descr)

        # Fallback for party caucus PACs: try name lookup
        # (some descriptions have candidate name but no district number)
        cand_party = None
        if chamber is None and pac["default_direction"] in ("D_favor", "R_favor"):
            ch2, dist2, party2 = lookup_name_in_descr(descr, name_map)
            if ch2 and dist2:
                chamber, district, cand_party = ch2, dist2, party2

        if chamber is None or district is None:
            stats["no_district"] += 1
            parse_log[pac["name"]]["no_district"] += 1
            continue

        # Step 2: determine direction
        direction = extract_direction_from_descr(
            descr, pac["default_direction"], pac["party"]
        )

        # For party caucus PACs with known direction but unknown candidate party
        # (e.g., HDCC spending in HD 33 — direction is D_favor regardless)
        if direction == "unknown":
            stats["unknown_direction"] += 1
            parse_log[pac["name"]]["unknown_direction"] += 1
            continue

        # Step 3: map to election year
        year = _date_to_election_year(row["expend_date"])
        if year is None:
            stats["out_of_window"] += 1
            continue

        key = (year, chamber, district)
        year_dist[key][direction] += row["amount"]
        year_dist[key]["n_pacs"].add(fid)

        stats["included"] += 1
        parse_log[pac["name"]]["included"] += 1
        parse_log[pac["name"]]["dollars"] += row["amount"]

    # Print parse stats
    print(f"\n  Row classification:")
    print(f"    Total rows:            {stats['total']:>6,}")
    print(f"    No district parseable: {stats['no_district']:>6,}")
    print(f"    Unknown direction:     {stats['unknown_direction']:>6,}")
    print(f"    Outside cycle window:  {stats['out_of_window']:>6,}")
    print(f"    Included in output:    {stats['included']:>6,}")
    print(f"\n  By PAC:")
    for pac_name, s in sorted(parse_log.items(), key=lambda x: -x[1]["dollars"]):
        print(f"    {pac_name[:45]:45s}  "
              f"incl={s['included']:>4}  no_dist={s['no_district']:>4}  "
              f"no_dir={s['unknown_direction']:>4}  ${s['dollars']:>12,.0f}")

    # Flatten to per-year lists
    results_by_year: dict[int, list[dict]] = defaultdict(list)
    for (year, chamber, district), totals in sorted(year_dist.items()):
        d_fav = totals["D_favor"]
        r_fav = totals["R_favor"]
        total = d_fav + r_fav
        n_pacs = len(totals["n_pacs"])

        ie_dem_share = d_fav / total if total > 0 else 0.5
        ie_log_total = math.log(total + 1)
        ie_flag = 1 if total >= IE_FLAG_THRESHOLD else 0

        results_by_year[year].append({
            "chamber":      chamber.title(),
            "district":     district,
            "ie_d_favor":   round(d_fav, 2),
            "ie_r_favor":   round(r_fav, 2),
            "ie_total":     round(total, 2),
            "ie_dem_share": round(ie_dem_share, 4),
            "ie_log_total": round(ie_log_total, 4),
            "ie_flag":      ie_flag,
            "n_spacs":      n_pacs,
            "data_source":  "tec_pac_expend",
        })

    return dict(results_by_year)


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_pac_ie_csv(rows: list[dict], year: int):
    path = DATA_HIST / f"tx_ies_pac_{year}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    total = sum(r["ie_total"] for r in rows)
    flagged = sum(r["ie_flag"] for r in rows)
    print(f"  {year}: wrote {len(rows)} districts, "
          f"${total:,.0f} total, {flagged} flagged → {path.name}")


def write_pac_summary(yearly_stats: list[dict]):
    path = DATA_HIST / "tx_ies_pac_summary.csv"
    fields = ["year", "n_districts", "total_dollars", "ie_d_favor", "ie_r_favor",
              "n_flagged", "coverage_note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(yearly_stats)
    print(f"  Summary → {path.name}")


# ---------------------------------------------------------------------------
# Merge PAC + SPAC data into combined IE files
# ---------------------------------------------------------------------------

def merge_with_spac_files(years: list[int]):
    """
    Merge tx_ies_pac_{year}.csv with tx_ies_{year}.csv (SPAC-based).
    For districts in both: sum D_favor and R_favor.
    Output: tx_ies_combined_{year}.csv (same schema, data_source="combined").
    """
    print(f"\n{'='*70}")
    print("  Merging PAC + SPAC IE files...")

    for year in years:
        spac_path = DATA_HIST / f"tx_ies_{year}.csv"
        pac_path  = DATA_HIST / f"tx_ies_pac_{year}.csv"
        out_path  = DATA_HIST / f"tx_ies_combined_{year}.csv"

        combined: dict[tuple, dict] = {}

        def _load(path, source_label):
            if not path.exists():
                return
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ch = row["chamber"].strip().title()
                    d  = int(row["district"])
                    key = (ch, d)
                    if key not in combined:
                        combined[key] = {
                            "chamber": ch, "district": d,
                            "ie_d_favor": 0.0, "ie_r_favor": 0.0,
                            "n_spacs": 0, "sources": set(),
                        }
                    combined[key]["ie_d_favor"] += float(row.get("ie_d_favor") or 0)
                    combined[key]["ie_r_favor"] += float(row.get("ie_r_favor") or 0)
                    combined[key]["n_spacs"]    += int(row.get("n_spacs") or 0)
                    combined[key]["sources"].add(source_label)

        _load(spac_path, "spac")
        _load(pac_path, "pac")

        rows = []
        for key, data in sorted(combined.items()):
            d_fav = data["ie_d_favor"]
            r_fav = data["ie_r_favor"]
            total = d_fav + r_fav
            rows.append({
                "chamber":      data["chamber"],
                "district":     data["district"],
                "ie_d_favor":   round(d_fav, 2),
                "ie_r_favor":   round(r_fav, 2),
                "ie_total":     round(total, 2),
                "ie_dem_share": round(d_fav / total if total > 0 else 0.5, 4),
                "ie_log_total": round(math.log(total + 1), 4),
                "ie_flag":      1 if total >= IE_FLAG_THRESHOLD else 0,
                "n_spacs":      data["n_spacs"],
                "data_source":  "combined_" + "+".join(sorted(data["sources"])),
            })

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        total_dollars = sum(r["ie_total"] for r in rows)
        print(f"  {year}: {len(rows)} districts, ${total_dollars:,.0f} → {out_path.name}")

    print("\n  To use combined files in regression:")
    print("  In build_phase1_dataset.py, change IE_SOURCE to 'combined'")


# ---------------------------------------------------------------------------
# Scan mode: show description samples from top PAC filers
# ---------------------------------------------------------------------------

def scan_mode(cd: dict):
    """Print sample descriptions from target PACs for quality-checking."""
    print("\n" + "="*70)
    print("  SCAN MODE: Sample expenditure descriptions from target PACs")
    print("="*70)

    all_rows = load_pac_expenditures(cd)
    if not all_rows:
        print("  No rows found.")
        return

    # Group by PAC
    by_pac: dict[str, list] = defaultdict(list)
    for row in all_rows:
        by_pac[row["filer_id"]].append(row)

    for fid, rows in sorted(by_pac.items(), key=lambda x: -sum(r["amount"] for r in x[1])):
        pac = PAC_REGISTRY[fid]
        total = sum(r["amount"] for r in rows)
        print(f"\n{fid}  ${total:>12,.0f}  {len(rows):>5} rows  {pac['name']}")
        # Sample description parsing
        parsed = 0
        for row in rows[:200]:
            ch, dist = extract_district_from_descr(row["descr"])
            if ch:
                direction = extract_direction_from_descr(
                    row["descr"], pac["default_direction"], pac["party"]
                )
                year = _date_to_election_year(row["expend_date"])
                if year and direction != "unknown":
                    parsed += 1
                    if parsed <= 5:
                        print(f"  [{year}] {ch.upper()} {dist:3d}  {direction}  "
                              f"${row['amount']:>10,.0f}  \"{row['descr'][:60]}\"")
        print(f"  (parseable in first 200: {parsed})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect true PAC IE data for TX legislative Phase 1 regression")
    parser.add_argument("--year", type=int, action="append", dest="years",
                        help="Run for specific year(s) only")
    parser.add_argument("--scan", action="store_true",
                        help="Scan PAC descriptions (no output files written)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge PAC + SPAC files into combined IE files")
    args = parser.parse_args()

    target_years = args.years if args.years else YEARS

    print("=" * 70)
    print("  TX Legislature — PAC IE Collection for Phase 1 Regression")
    print(f"  Target PACs: {len(PAC_REGISTRY)}  |  Years: {target_years}")
    print("=" * 70)

    print("\nReading TEC ZIP central directory...")
    cd = _tec_zip_central_dir(TEC_ZIP_URL)
    if not cd:
        print("ERROR: Could not read TEC ZIP central directory.")
        sys.exit(1)

    if args.scan:
        scan_mode(cd)
        return

    # Build candidate name → district lookup for name-based fallback
    print("\nBuilding candidate name → district lookup from election results...")
    name_map = build_name_district_map()
    print(f"  {len(name_map):,} unique candidate last names mapped to districts")

    # Load all PAC expenditure rows (cached)
    all_rows = load_pac_expenditures(cd)
    if not all_rows:
        print("ERROR: No PAC expenditure rows found.")
        sys.exit(1)

    # Parse and aggregate
    print("\nParsing descriptions and aggregating by year/district...")
    results_by_year = parse_and_aggregate(all_rows, name_map)

    # Print and write
    print(f"\n{'='*70}")
    print("  PAC IE SUMMARY BY ELECTION YEAR")
    print(f"{'='*70}")

    yearly_stats = []
    for year in target_years:
        rows = results_by_year.get(year, [])
        total  = sum(r["ie_total"] for r in rows)
        d_fav  = sum(r["ie_d_favor"] for r in rows)
        r_fav  = sum(r["ie_r_favor"] for r in rows)
        flagged = sum(r["ie_flag"] for r in rows)

        print(f"\n  {year}: {len(rows)} districts  |  "
              f"${total:>12,.0f} total  |  D: ${d_fav:>10,.0f}  R: ${r_fav:>10,.0f}")
        if rows:
            print(f"         Flagged (>${IE_FLAG_THRESHOLD:,}): {flagged}")

        yearly_stats.append({
            "year":           year,
            "n_districts":    len(rows),
            "total_dollars":  round(total, 2),
            "ie_d_favor":     round(d_fav, 2),
            "ie_r_favor":     round(r_fav, 2),
            "n_flagged":      flagged,
            "coverage_note":  (
                "no_data"   if not rows else
                "sparse"    if len(rows) < 10 else
                "moderate"  if len(rows) < 30 else
                "good"
            ),
        })

        write_pac_ie_csv(rows, year)

    write_pac_summary(yearly_stats)

    if args.merge or True:  # always merge after writing
        merge_with_spac_files(target_years)

    print(f"\n{'='*70}")
    print("  COVERAGE SUMMARY")
    print(f"{'='*70}")
    for stat in yearly_stats:
        print(f"  {stat['year']}: {stat['coverage_note']:10s}  "
              f"{stat['n_districts']:3d} districts  ${stat['total_dollars']:>12,.0f}")

    print(f"\nNext steps:")
    print(f"  1. Check tx_ies_pac_summary.csv for coverage by year")
    print(f"  2. Edit build_phase1_dataset.py: set IE_SOURCE = 'combined'")
    print(f"  3. python src/build_phase1_dataset.py")
    print(f"  4. python src/run_phase1_regression.py")


if __name__ == "__main__":
    main()
