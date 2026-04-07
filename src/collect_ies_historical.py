"""
collect_ies_historical.py

Collect Texas legislative independent expenditure (IE) data for historical
election cycles (2002, 2006, 2010, 2014, 2018, 2022) from the TEC master ZIP.

STRATEGY:
  1. Extract spacs.csv  — SPAC → candidate mappings with SUPPORT/OPPOSE positions
                          (all years in one file; filter by TX legislative office)
  2. Extract cover.csv  — candidate filer → politicalPartyCd lookup
                          (for historical years, TEC DID populate politicalPartyCd)
  3. Extract expend_*.csv — itemized SPAC expenditures by date; filter to election windows
  4. Join: spacs → cover (party) → expend (amounts) → aggregate by district-year

OUTPUT (one per cycle):
  data/raw/historical/tx_ies_{year}.csv
  Columns: chamber, district, ie_dem_share, ie_rep_share, ie_log_total,
           ie_total, ie_d_favor, ie_r_favor, ie_flag, n_spacs, data_source

Then update data/raw/historical/tx_ies_summary.csv with coverage statistics.

IMPORTANT CAVEATS:
  - 2002-2006: Citizens United (2010) hadn't happened yet. IE activity was minimal
    and TEC electronic filing was sparse. Expect little to no data for early cycles.
  - 2010+: IE activity grew significantly; TEC data increasingly complete.
  - spacs.csv covers all years in TEC's system, but early-year SPACs may be absent.
  - This is fine: the regression will fit IEs where data exists and flag data gaps.

USAGE:
  python src/collect_ies_historical.py             # all years 2002-2022
  python src/collect_ies_historical.py --year 2022 # single year
  python src/collect_ies_historical.py --summary   # print only, no write
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

OFFICE_CODES = {"STATEREP": "house", "STATESEN": "senate"}

# IE significance threshold (total IE dollars to set ie_flag=1)
IE_FLAG_THRESHOLD = 25_000

OUTPUT_FIELDS = [
    "chamber", "district",
    "ie_d_favor", "ie_r_favor", "ie_total",
    "ie_dem_share",   # D-favoring IEs / total IEs; 0.5 if no IEs
    "ie_log_total",   # log(total + 1)
    "ie_flag",        # 1 if total >= IE_FLAG_THRESHOLD
    "n_spacs",
    "data_source",
]

SUMMARY_FIELDS = [
    "year", "n_legislative_spacs", "n_districts_with_ie",
    "total_ie_dollars", "ie_d_favor", "ie_r_favor",
    "n_party_resolved", "n_party_unknown",
    "earliest_expend_date", "coverage_note",
]


# ---------------------------------------------------------------------------
# Step 1: Parse spacs.csv — SPAC → candidate mapping
# ---------------------------------------------------------------------------

def load_spacs_all_years(cd: dict) -> list[dict]:
    """
    Extract spacs.csv and return all TX legislative SPAC rows.

    TEC spacs.csv schema (confirmed from file inspection):
      spacFilerIdent         — the SPAC's own filer ID (used to find expenditures in expend_*.csv)
      spacFilerName          — SPAC committee name
      spacPositionCd         — SUPPORT | OPPOSE | ASSIST | UNKNOWN
      candidateFilerIdent    — the target candidate's filer ID (used for party lookup)
      candidateFilerName     — candidate name
      candidateSeekOfficeCd  — office code (STATEREP/STATESEN) — may be blank
      candidateSeekOfficeDistrict — district if seeking — may be blank
      candidateHoldOfficeCd  — office code if holding — fallback for incumbents
      candidateHoldOfficeDistrict — district if holding

    NOTE: TX "SPACs" include both candidate-affiliated committees ("Friends of X")
    and true outside-group committees. All declare SUPPORT/OPPOSE for a specific candidate.
    """
    spac_fname = next((f for f in cd if "spac" in f.lower() and f.lower().endswith(".csv")), None)
    if not spac_fname:
        print("  WARNING: spacs.csv not found in TEC ZIP")
        return []

    print(f"\nExtracting {spac_fname}...")
    data = _tec_extract_file(TEC_ZIP_URL, cd[spac_fname], spac_fname)
    if not data:
        print("  Failed to extract spacs.csv")
        return []

    text = data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    spacs = []
    skipped = 0
    for row in reader:
        # Try seeking office first, then holding office (for incumbents)
        seek_cd   = row.get("candidateSeekOfficeCd", "").strip().upper()
        hold_cd   = row.get("candidateHoldOfficeCd", "").strip().upper()
        seek_dist = re.sub(r"\D", "", row.get("candidateSeekOfficeDistrict", "").strip())
        hold_dist = re.sub(r"\D", "", row.get("candidateHoldOfficeDistrict", "").strip())

        if seek_cd in OFFICE_CODES and seek_dist:
            chamber  = OFFICE_CODES[seek_cd]
            district = int(seek_dist)
        elif hold_cd in OFFICE_CODES and hold_dist:
            chamber  = OFFICE_CODES[hold_cd]
            district = int(hold_dist)
        else:
            skipped += 1
            continue

        position = row.get("spacPositionCd", "").strip().upper()
        # ASSIST ≈ SUPPORT (helping the candidate's campaign)
        if position == "ASSIST":
            position = "SUPPORT"
        if position not in ("SUPPORT", "OPPOSE"):
            skipped += 1
            continue

        # CRITICAL: use spacFilerIdent (the SPAC's own ID, matches filerIdent in expend_*.csv)
        spac_filer_id = row.get("spacFilerIdent", "").strip()
        cand_filer_id = row.get("candidateFilerIdent", "").strip()
        cand_name     = row.get("candidateFilerName", "").strip()

        if not spac_filer_id:
            skipped += 1
            continue

        spacs.append({
            "filer_id":      spac_filer_id,   # key for joining to expend_*.csv
            "position":      position,
            "cand_filer_id": cand_filer_id,   # key for party lookup in cover.csv
            "cand_name":     cand_name,
            "chamber":       chamber,
            "district":      district,
        })

    print(f"  Found {len(spacs)} TX legislative SPAC → candidate rows "
          f"(skipped {skipped} without TX legislative office/district)")
    return spacs


# ---------------------------------------------------------------------------
# Step 2: Parse cover.csv — candidate filer → party lookup
# ---------------------------------------------------------------------------

def load_candidate_parties(cd: dict, cand_filer_ids: set[str]) -> dict[str, str]:
    """
    Parse cover.csv to build a party map for the target candidates in our SPAC set.
    Returns {candidateFilerIdent: "D"|"R"}.

    Strategy:
      1. Match cover.csv filerIdent to our candidateFilerIdent set
      2. Use politicalPartyCd if populated (works for historical filers)
      3. Fall back to filerSeekOfficeCd context + historical election results
         (election results will be used later if cover.csv party is empty)

    Note: politicalPartyCd is often empty — TEC doesn't require party registration
    in cover reports. The field is populated selectively. We get party for ~50% of cases.
    """
    if not cand_filer_ids:
        return {}

    cover_fname = next((f for f in cd if f.lower().endswith("cover.csv")), None)
    if not cover_fname:
        print("  WARNING: cover.csv not found")
        return {}

    print(f"\nExtracting {cover_fname} for candidate party lookup ({len(cand_filer_ids)} candidates)...")
    data = _tec_extract_file(TEC_ZIP_URL, cd[cover_fname], cover_fname)
    if not data:
        return {}

    text = data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    party_map: dict[str, str] = {}
    for row in reader:
        filer_id = row.get("filerIdent", "").strip()
        if filer_id not in cand_filer_ids:
            continue

        # politicalPartyCd: "R" or "D" for historical; often empty
        party_raw = row.get("politicalPartyCd", "").strip().upper()
        if party_raw in ("R", "REPUBLICAN"):
            party_map[filer_id] = "R"
        elif party_raw in ("D", "DEMOCRAT", "DEMOCRATIC"):
            party_map[filer_id] = "D"
        # Blank: will fall back to historical election results

    d_count = sum(1 for v in party_map.values() if v == "D")
    r_count = sum(1 for v in party_map.values() if v == "R")
    print(f"  Party from cover.csv: {len(party_map)}/{len(cand_filer_ids)} resolved "
          f"({r_count} R, {d_count} D)  ({len(cand_filer_ids)-len(party_map)} unknown)")
    return party_map


# ---------------------------------------------------------------------------
# Step 3: Extract expenditures for SPAC filers from expend_*.csv
# ---------------------------------------------------------------------------

def load_party_from_election_results(spacs: list[dict]) -> dict[str, str]:
    """
    Fallback party lookup: use historical election results to assign party
    to candidates in districts where cover.csv didn't give us party.

    Logic: for each (chamber, district, election_year), we know who won and
    with what party. If a SPAC SUPPORTS candidate X in district Y, and X is
    the winner in year Z, X's party = winner_party.

    Returns {candidateFilerIdent: "D"|"R"} — same structure as cover.csv lookup.
    Note: this can't distinguish the candidate from the winner if name-matching isn't done.
    We can't reliably match filerIdent to election_results without names. Instead,
    we use the district's incumbent_party as a proxy for the SPAC's target.

    Returns {spac_filer_id: "D"|"R"} keyed by the SPAC's own ID.
    """
    # Load districts_2026.csv for incumbent party (current snapshot)
    dist_party: dict[tuple, str] = {}
    dist_path = Path(__file__).parent.parent / "data" / "processed" / "districts_2026.csv"
    if dist_path.exists():
        with open(dist_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                party = row.get("incumbent_party", "").strip().upper()
                if party in ("D", "R"):
                    ch   = row["chamber"].strip().lower()
                    dist = int(row["district"])
                    dist_party[(ch, dist)] = party

    # Load historical election winners for more accurate per-year party assignment
    hist_winner: dict[tuple, str] = {}  # (year, chamber, district) → winner_party
    hist_dir = Path(__file__).parent.parent / "data" / "raw" / "historical"
    for year in YEARS:
        for ch in ("house", "senate"):
            path = hist_dir / f"tx_{ch}_results_{year}.csv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    wp = row.get("winner_party", "").strip().upper()
                    if wp in ("D", "R"):
                        hist_winner[(year, ch, int(row["district"]))] = wp

    # For each SPAC, assign the party of the candidate they SUPPORT in that district
    # For SUPPORT SPACs: candidate party = the party they're supporting
    # For OPPOSE SPACs: candidate party = whatever party is being opposed
    spac_party: dict[str, str] = {}
    for s in spacs:
        # Use historical winners from any cycle that has data for this district
        # The incumbent's party is the best proxy for the SPAC's target
        ch       = s["chamber"]
        district = s["district"]
        fid      = s["filer_id"]

        # Try to find party from any election year for this district
        for year in reversed(YEARS):
            party = hist_winner.get((year, ch, district))
            if party:
                # This is the WINNER's party — not necessarily the SPAC's target
                # For SUPPORT SPACs: likely supporting the winner (or incumbent)
                # For OPPOSE SPACs: likely opposing the winner
                # This is approximate; use as fallback only
                spac_party[fid] = party
                break

    assigned = len(spac_party)
    print(f"  Party from election history: {assigned}/{len(spacs)} SPACs assigned "
          f"(approximate — uses district winner as proxy)")
    return spac_party


def load_all_spac_expenditures(cd: dict, spac_filer_ids: set[str]) -> list[dict]:
    """
    Extract all expenditure rows from expend_*.csv where filerIdent is in the
    SPAC set. Returns full list — caller filters by election year date window.

    Caches each file after first download.
    """
    if not spac_filer_ids:
        print("  No SPAC filers to look up")
        return []

    expend_files = sorted(f for f in cd if re.match(r"expend_\d+\.csv", f, re.IGNORECASE))
    if not expend_files:
        print("  WARNING: No expend_*.csv files found in TEC ZIP")
        return []

    print(f"\nExtracting expenditures from {len(expend_files)} expend files "
          f"for {len(spac_filer_ids)} SPAC filers...")

    all_expend = []
    for fname in expend_files:
        data = _tec_extract_file(TEC_ZIP_URL, cd[fname], fname)
        if not data:
            print(f"  WARNING: Could not extract {fname}")
            continue

        text = data.decode(TEC_ENCODING, errors="replace")
        reader = csv.DictReader(io.StringIO(text))

        file_count = 0
        for row in reader:
            # In expend_*.csv, filerIdent is the spender's ID
            # For SPACs: this equals spacFilerIdent from spacs.csv
            filer_id = row.get("filerIdent", "").strip()
            if filer_id not in spac_filer_ids:
                continue

            # expendDt: date of expenditure (confirmed column name)
            expend_dt = re.sub(r"\D", "", row.get("expendDt", "").strip())[:8]
            if len(expend_dt) < 8:
                continue

            # expendAmount: dollar amount (confirmed column name)
            amount_raw = (row.get("expendAmount", "") or "0").strip().replace(",", "").replace("$", "")
            try:
                amount = float(amount_raw)
            except ValueError:
                continue
            if amount <= 0:
                continue

            all_expend.append({
                "filer_id":   filer_id,
                "expend_date": expend_dt,
                "amount":     amount,
            })
            file_count += 1

        if file_count:
            print(f"  {fname}: {file_count} rows matched SPAC filers")

    print(f"  Total SPAC expenditure rows across all years: {len(all_expend):,}")
    return all_expend


# ---------------------------------------------------------------------------
# Step 4: Aggregate by (year, chamber, district)
# ---------------------------------------------------------------------------

def _resolve_direction(position: str, cand_party: str) -> str:
    """SUPPORT D → D_favor; OPPOSE D → R_favor; vice versa for R."""
    if cand_party == "D":
        return "D_favor" if position == "SUPPORT" else "R_favor"
    elif cand_party == "R":
        return "R_favor" if position == "SUPPORT" else "D_favor"
    return "unknown"


def aggregate_by_year_district(
    spacs: list[dict],
    candidate_party: dict[str, str],
    all_expenditures: list[dict],
) -> dict[tuple, list[dict]]:
    """
    For each election year, aggregate IE totals by (chamber, district).

    Returns: {year: [{chamber, district, ie_d_favor, ie_r_favor, ...}]}
    """
    # Build spac lookup: spac_filer_id → {position, chamber, district, cand_party}
    # candidate_party is keyed by cand_filer_id
    spac_info: dict[str, dict] = {}
    for s in spacs:
        cand_party = candidate_party.get(s["cand_filer_id"], "unknown")
        spac_info[s["filer_id"]] = {
            "position":   s["position"],
            "chamber":    s["chamber"],
            "district":   s["district"],
            "cand_party": cand_party,
        }

    # Group expenditures by (year, chamber, district, direction)
    # Key: (year, chamber, district) → {D_favor: $, R_favor: $, unknown: $}
    year_dist_totals: dict[tuple, dict] = defaultdict(
        lambda: {"D_favor": 0.0, "R_favor": 0.0, "unknown": 0.0, "n_spacs": set()}
    )

    party_resolved = 0
    party_unknown = 0

    for exp in all_expenditures:
        fid = exp["filer_id"]
        info = spac_info.get(fid)
        if not info:
            continue

        # Determine election year from expenditure date
        year = _date_to_election_year(exp["expend_date"])
        if year is None:
            continue

        chamber  = info["chamber"]
        district = info["district"]
        position = info["position"]
        cand_party = info["cand_party"]

        direction = _resolve_direction(position, cand_party)
        if cand_party != "unknown":
            party_resolved += 1
        else:
            party_unknown += 1

        key = (year, chamber, district)
        year_dist_totals[key][direction] += exp["amount"]
        year_dist_totals[key]["n_spacs"].add(fid)

    print(f"\n  Expenditure rows: {party_resolved} with party, {party_unknown} unknown party")

    # Flatten to per-year lists
    results_by_year: dict[int, list[dict]] = defaultdict(list)
    for (year, chamber, district), totals in sorted(year_dist_totals.items()):
        d_fav = totals["D_favor"]
        r_fav = totals["R_favor"]
        total = d_fav + r_fav  # exclude unknown from totals
        n_spacs = len(totals["n_spacs"])

        ie_dem_share = d_fav / total if total > 0 else 0.5  # 0.5 = neutral
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
            "n_spacs":      n_spacs,
            "data_source":  "tec_spacs",
        })

    return dict(results_by_year)


def _date_to_election_year(date_str: str) -> int | None:
    """
    Map an expenditure date (YYYYMMDD) to the nearest election year.
    Returns the election year this expenditure belongs to, or None if outside all cycles.
    """
    if len(date_str) < 8:
        return None
    for year, (start, end) in CYCLE_WINDOWS.items():
        if start <= date_str <= end:
            return year
    return None  # Outside all cycle windows (e.g., pre-2002 or post-2022)


# ---------------------------------------------------------------------------
# Step 5: Write output CSVs
# ---------------------------------------------------------------------------

def write_ie_csv(rows: list[dict], year: int):
    path = DATA_HIST / f"tx_ies_{year}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    total_ie = sum(r["ie_total"] for r in rows)
    flagged = sum(r["ie_flag"] for r in rows)
    print(f"  {year}: wrote {len(rows)} districts with IE activity, "
          f"${total_ie:,.0f} total, {flagged} flagged → {path.name}")


def write_summary(yearly_stats: list[dict]):
    path = DATA_HIST / "tx_ies_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(yearly_stats)
    print(f"\n  Summary written → {path.name}")


# ---------------------------------------------------------------------------
# Print yearly summary
# ---------------------------------------------------------------------------

def print_year_summary(year: int, rows: list[dict], spac_count: int):
    if not rows:
        print(f"\n  {year}: No IE data found (data_source=empty)")
        return

    total  = sum(r["ie_total"] for r in rows)
    d_fav  = sum(r["ie_d_favor"] for r in rows)
    r_fav  = sum(r["ie_r_favor"] for r in rows)
    flagged = sum(r["ie_flag"] for r in rows)

    print(f"\n  {year}: {len(rows)} districts  |  "
          f"${total:>12,.0f} total  |  D-favor: ${d_fav:>10,.0f}  R-favor: ${r_fav:>10,.0f}")
    print(f"         Flagged districts (>${IE_FLAG_THRESHOLD:,}): {flagged}  |  "
          f"SPACs in window: {spac_count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect historical TX legislative IE data from TEC for Phase 1 regression")
    parser.add_argument("--year", type=int, action="append", dest="years",
                        help="Run for specific year(s) only (default: all 2002-2022)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only; don't write files")
    args = parser.parse_args()

    target_years = args.years if args.years else YEARS
    for y in target_years:
        if y not in CYCLE_WINDOWS:
            print(f"ERROR: {y} is not a supported cycle year. Choose from {YEARS}")
            sys.exit(1)

    print("=" * 70)
    print("  TX Legislature — Historical IE Collection for Phase 1 Regression")
    print(f"  Years: {target_years}")
    print("=" * 70)

    print("\nReading TEC ZIP central directory...")
    cd = _tec_zip_central_dir(TEC_ZIP_URL)
    if not cd:
        print("ERROR: Could not read TEC ZIP central directory.")
        sys.exit(1)

    # Step 1: Load all SPAC → candidate mappings (TX legislative only)
    spacs = load_spacs_all_years(cd)
    if not spacs:
        print("\nNo TX legislative SPAC data found. TEC spacs.csv has 504 rows but none")
        print("matched TX legislative offices with populated district numbers.")
        print("Proceeding with empty IE dataset.")
        spacs = []

    print(f"\n  {len(spacs)} TX legislative SPACs found")

    # Step 2: Party lookup — cover.csv + historical election results fallback
    cand_filer_ids = {s["cand_filer_id"] for s in spacs if s["cand_filer_id"]}
    candidate_party_cover = load_candidate_parties(cd, cand_filer_ids)

    # For SPACs where cover.csv didn't resolve party, use election history
    candidate_party_hist = load_party_from_election_results(spacs)

    # Merge: cover.csv takes priority, then election history
    # Key mapping: we need spac_filer_id → candidate party
    # cover gives us: cand_filer_id → party
    # Convert to: spac_filer_id → party
    candidate_party: dict[str, str] = {}
    for s in spacs:
        # Try cover.csv via candidate's filer ID
        party = candidate_party_cover.get(s["cand_filer_id"])
        if not party:
            # Fall back to election-history proxy (keyed by spac filer ID)
            party = candidate_party_hist.get(s["filer_id"])
        if party:
            candidate_party[s["cand_filer_id"]] = party

    print(f"  Total candidate party assignments: {len(candidate_party)}/{len(spacs)} SPACs")

    # Step 3: Extract expenditures for ALL SPAC filers (cached after first run)
    spac_filer_ids = {s["filer_id"] for s in spacs}
    all_expenditures = load_all_spac_expenditures(cd, spac_filer_ids) if spac_filer_ids else []

    # Step 4: Aggregate by year-district
    print("\nAggregating by year and district...")
    if spacs and all_expenditures:
        results_by_year = aggregate_by_year_district(spacs, candidate_party, all_expenditures)
    else:
        print("  No SPAC expenditure data found — writing empty IE files.")
        results_by_year = {year: [] for year in target_years}

    # Step 5: Print and write
    print(f"\n{'='*70}")
    print("  IE COLLECTION SUMMARY BY ELECTION YEAR")
    print(f"{'='*70}")

    yearly_stats = []
    for year in target_years:
        rows = results_by_year.get(year, [])

        # Count SPACs that had expenditures in this cycle window
        start, end = CYCLE_WINDOWS[year]
        spacs_in_window = sum(
            1 for exp in all_expenditures
            if start <= exp["expend_date"] <= end
            and exp["filer_id"] in spac_filer_ids
        )

        print_year_summary(year, rows, spacs_in_window)

        # Party resolution stats
        n_resolved = sum(1 for r in rows if r["ie_d_favor"] + r["ie_r_favor"] > 0)
        total_ie = sum(r["ie_total"] for r in rows)

        yearly_stats.append({
            "year":                  year,
            "n_legislative_spacs":   spacs_in_window,
            "n_districts_with_ie":   len(rows),
            "total_ie_dollars":      round(total_ie, 2),
            "ie_d_favor":            round(sum(r["ie_d_favor"] for r in rows), 2),
            "ie_r_favor":            round(sum(r["ie_r_favor"] for r in rows), 2),
            "n_party_resolved":      n_resolved,
            "n_party_unknown":       len(rows) - n_resolved,
            "earliest_expend_date":  min(
                (e["expend_date"] for e in all_expenditures
                 if start <= e["expend_date"] <= end and e["filer_id"] in spac_filer_ids),
                default=""
            ),
            "coverage_note": (
                "no_data" if not rows else
                "sparse_pre_citizens_united" if year <= 2006 else
                "partial" if year <= 2014 else
                "comprehensive"
            ),
        })

        if not args.summary:
            write_ie_csv(rows, year)

    if not args.summary:
        write_summary(yearly_stats)

    print(f"\n{'='*70}")
    print("  Coverage assessment:")
    for stat in yearly_stats:
        note = stat["coverage_note"]
        total = stat["total_ie_dollars"]
        n = stat["n_districts_with_ie"]
        print(f"  {stat['year']}: {note:35s} | {n:3d} districts | ${total:>12,.0f}")

    print(f"\nNext step: python src/build_phase1_dataset.py")
    print(f"           python src/run_phase1_regression.py")


if __name__ == "__main__":
    main()
