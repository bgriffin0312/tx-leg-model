"""
collect_finance_2026.py

Pull 2026 campaign finance data from the TEC master ZIP for current-cycle
TX legislative candidates.

Key differences from the historical collect_finance.py:
  - The 2026 election cycle is incomplete (partial year through today).
  - TEC does not populate politicalPartyCd for 2026 filers.
  - Party assignment is inferred:
      Incumbents   → filerHoldOfficeCd/District matches sought office/district;
                     party from districts_2026.csv.
      Challengers  → all other filers; party unknown pre-primary.
  - For the viability flag we ask: "does the NON-INCUMBENT side have
    meaningful early fundraising?" This catches viable opposition candidates
    without requiring party assignment.
  - Uses early-cycle viability thresholds (lower than full-cycle):
      House  > $40,000  (VIABILITY_THRESHOLD_POSTPRIMARY["house"])
      Senate > $100,000 (VIABILITY_THRESHOLD_POSTPRIMARY["senate"])
  - Output merged into data/processed/districts_2026.csv as new columns.

COVERAGE NOTES (as of April 2026):
  - Covers Jan–Apr 2026 primary pre-election reports (30-day and 8-day before).
  - Re-run after July semi-annual TEC deadline to pick up post-primary data.
  - Only 16 Senate districts are on the 2026 ballot; all 150 House districts are.

Usage:
  python src/collect_finance_2026.py              # fetch, write CSV, merge into districts
  python src/collect_finance_2026.py --no-merge   # write CSV only, don't touch districts_2026.csv
  python src/collect_finance_2026.py --summary    # print district summary only, no write
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

ROOT = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

# Reuse TEC range-extraction utilities from the historical script
sys.path.insert(0, str(Path(__file__).parent))
from collect_finance import (
    _tec_zip_central_dir,
    _tec_extract_file,
    _normalize_name,
    _name_match,
    TEC_ZIP_URL,
    TEC_ENCODING,
)
from model_config import (
    VIABILITY_THRESHOLD_POSTPRIMARY as THRESHOLD,
    FINANCE_CUTOFF_POSTPRIMARY,      # "20260430"
    FINANCE_DATA_THROUGH,
)

# 2026 cycle: include reports whose period started on or after this date
CYCLE_START = "20260101"
# Don't include reports filed after the cutoff (future data)
CYCLE_END   = FINANCE_CUTOFF_POSTPRIMARY  # "20260430"

# Senate districts actually on the 2026 ballot (16 of 31)
SENATE_2026_BALLOT = {1, 2, 3, 4, 5, 9, 11, 13, 18, 19, 21, 22, 24, 26, 28, 31}


# ---------------------------------------------------------------------------
# Step 1: Pull and aggregate TEC 2026 filings
# ---------------------------------------------------------------------------

def load_2026_cover(verbose: bool = False) -> dict:
    """
    Extract cover.csv from TEC master ZIP and aggregate 2026 legislative filings.

    Returns by_district:
      {(chamber, district): {filer_id: {"name": str, "total": float, "is_incumbent": bool}}}
    """
    print("Reading TEC ZIP central directory...")
    cd = _tec_zip_central_dir(TEC_ZIP_URL)
    if not cd:
        raise RuntimeError("Could not read TEC ZIP central directory.")

    cover_fname = next((f for f in cd if f.lower().endswith("cover.csv")), None)
    if not cover_fname:
        raise RuntimeError("cover.csv not found in TEC ZIP.")

    cover_data = _tec_extract_file(TEC_ZIP_URL, cd[cover_fname], cover_fname)
    if not cover_data:
        raise RuntimeError("Failed to download cover.csv from TEC ZIP.")

    text = cover_data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    OFFICE_CODES = {"STATEREP": "house", "STATESEN": "senate"}

    by_district: dict = defaultdict(dict)
    rows_kept = 0
    rows_skipped_date = 0

    for row in reader:
        seek_office = row.get("filerSeekOfficeCd", "").strip().upper()
        chamber = OFFICE_CODES.get(seek_office)
        if chamber is None:
            continue

        # District
        dist_raw = re.sub(r"\D", "", row.get("filerSeekOfficeDistrict", "").strip())
        if not dist_raw:
            continue
        district = int(dist_raw)
        if district < 1 or district > (150 if chamber == "house" else 31):
            continue

        # Date filter: period must start within 2026 cycle
        period_start = re.sub(r"\D", "", row.get("periodStartDt", "").strip())[:8]
        if len(period_start) < 8 or not (CYCLE_START <= period_start <= CYCLE_END):
            rows_skipped_date += 1
            continue

        # Contribution amount
        total_raw = row.get("totalContribAmount", "0").strip().replace(",", "").replace("$", "")
        try:
            total = float(total_raw)
        except ValueError:
            continue
        if total <= 0:
            continue

        filer_id   = row.get("filerIdent", "").strip() or row.get("filerName", "").strip()
        filer_name = row.get("filerName", "").strip()

        # Is this an incumbent? They hold the same office+district they're seeking.
        hold_office = row.get("filerHoldOfficeCd", "").strip().upper()
        hold_dist   = re.sub(r"\D", "", row.get("filerHoldOfficeDistrict", "").strip())
        seek_dist   = re.sub(r"\D", "", row.get("filerSeekOfficeDistrict", "").strip())
        is_incumbent = (hold_office == seek_office and hold_dist == seek_dist and bool(hold_dist))

        dk = (chamber, district)
        if filer_id in by_district[dk]:
            by_district[dk][filer_id]["total"] += total
        else:
            by_district[dk][filer_id] = {
                "name": filer_name,
                "total": total,
                "is_incumbent": is_incumbent,
            }
        rows_kept += 1

    print(f"  2026 cover rows kept: {rows_kept:,}  (skipped {rows_skipped_date:,} outside date window)")
    house_count  = sum(1 for (ch, _) in by_district if ch == "house")
    senate_count = sum(1 for (ch, _) in by_district if ch == "senate")
    print(f"  Districts with any filing: {house_count} house, {senate_count} senate")
    return dict(by_district)


# ---------------------------------------------------------------------------
# Step 2: Load districts_2026.csv for incumbent party lookup and name matching
# ---------------------------------------------------------------------------

def load_districts_2026() -> dict:
    """
    Load districts_2026.csv.
    Returns dict: {(chamber_lower, district_int): {incumbent, incumbent_party, up_in_2026, open_seat}}
    """
    path = DATA_PROC / "districts_2026.csv"
    if not path.exists():
        raise FileNotFoundError(f"districts_2026.csv not found at {path}")

    districts = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ch = row["chamber"].strip().lower()
            dist = int(row["district"])
            districts[(ch, dist)] = {
                "incumbent":       row.get("incumbent", "").strip(),
                "incumbent_party": row.get("incumbent_party", "").strip(),
                "up_in_2026":      str(row.get("up_in_2026", "")).strip().lower() in ("true", "1", "yes"),
                "open_seat":       str(row.get("open_seat", "")).strip().lower() in ("true", "1", "yes"),
            }
    return districts


# ---------------------------------------------------------------------------
# Step 3: Assign parties and compute per-district finance summary
# ---------------------------------------------------------------------------

def assign_parties_and_aggregate(by_district: dict, districts_info: dict) -> list[dict]:
    """
    For each district with TEC data:
      - Identify the incumbent filer (via hold-office flag OR name match to districts_2026)
      - Aggregate incumbent_raised and challenger_raised
      - Compute early-cycle viability flag

    Returns list of row dicts ready for CSV output.
    """
    rows = []

    # Build a combined set: House (all 150) + Senate on 2026 ballot (16)
    all_districts = []
    for dist in range(1, 151):
        all_districts.append(("house", dist))
    for dist in SENATE_2026_BALLOT:
        all_districts.append(("senate", dist))

    for (chamber, district) in sorted(all_districts):
        info = districts_info.get((chamber, district), {})
        incumbent_name   = info.get("incumbent", "")
        incumbent_party  = info.get("incumbent_party", "")
        open_seat        = info.get("open_seat", False)

        filers = by_district.get((chamber, district), {})

        if not filers:
            rows.append(_placeholder_row(chamber, district, incumbent_name, incumbent_party))
            continue

        # Classify each filer as incumbent or challenger
        inc_total  = 0.0
        chal_total = 0.0
        inc_name_found   = ""
        chal_names_found = []

        for fid, fdata in filers.items():
            name  = fdata["name"]
            total = fdata["total"]
            is_inc_by_hold = fdata["is_incumbent"]

            # Secondary check: name match against known incumbent
            is_inc_by_name = (
                bool(incumbent_name)
                and _name_match(_normalize_name(name), _normalize_name(incumbent_name))
            )

            if is_inc_by_hold or is_inc_by_name:
                inc_total += total
                if not inc_name_found:
                    inc_name_found = name
            else:
                chal_total += total
                chal_names_found.append(name)

        # If open seat, whoever raised most is treated as "leading" (no incumbent)
        if open_seat:
            inc_total  = 0.0
            chal_total = sum(f["total"] for f in filers.values())

        threshold = THRESHOLD.get(chamber, 100_000)
        viability_flag = int(chal_total >= threshold)

        # dem_fundraising_share: D raised / (D raised + R raised)
        # Assignment: incumbent party tells us which side is D vs R.
        # Caveats: challenger_raised may include same-party primary opponents,
        # so this is an approximation. Open/vacant seats get None.
        if incumbent_party == "D" and (inc_total + chal_total) > 0:
            dem_fundraising_share = round(inc_total / (inc_total + chal_total), 4)
        elif incumbent_party == "R" and (inc_total + chal_total) > 0:
            dem_fundraising_share = round(chal_total / (inc_total + chal_total), 4)
        else:
            dem_fundraising_share = None  # open seat, vacant, or no data

        rows.append({
            "year":                          2026,
            "chamber":                       chamber.title(),
            "district":                      district,
            "incumbent_name_tec":            inc_name_found,
            "incumbent_party":               incumbent_party,
            "incumbent_raised":              round(inc_total, 2),
            "challenger_raised":             round(chal_total, 2),
            "dem_fundraising_share":         dem_fundraising_share,
            "challenger_names":              "; ".join(chal_names_found[:5]),
            "party_assignment_method":       "hold_office_and_name_match",
            "challenger_viability_flag_early": viability_flag,
            "viability_threshold_used":      threshold,
            "open_seat":                     open_seat,
            "data_source":                   "tec_cover_2026",
            "MANUAL_NEEDED":                 False,
        })

    return rows


def _placeholder_row(chamber: str, district: int,
                     incumbent_name: str, incumbent_party: str) -> dict:
    return {
        "year":                          2026,
        "chamber":                       chamber.title(),
        "district":                      district,
        "incumbent_name_tec":            "",
        "incumbent_party":               incumbent_party,
        "incumbent_raised":              None,
        "challenger_raised":             None,
        "dem_fundraising_share":         None,
        "challenger_names":              "",
        "party_assignment_method":       "no_filing",
        "challenger_viability_flag_early": 0,
        "viability_threshold_used":      THRESHOLD.get(chamber, 100_000),
        "open_seat":                     False,
        "data_source":                   "no_filing",
        "MANUAL_NEEDED":                 True,
    }


# ---------------------------------------------------------------------------
# Step 4: Write CSV
# ---------------------------------------------------------------------------

FIELDS = [
    "year", "chamber", "district", "incumbent_name_tec", "incumbent_party",
    "incumbent_raised", "challenger_raised", "dem_fundraising_share",
    "challenger_names", "party_assignment_method",
    "challenger_viability_flag_early", "viability_threshold_used",
    "open_seat", "data_source", "MANUAL_NEEDED",
]


def write_finance_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    viable = sum(1 for r in rows if r["challenger_viability_flag_early"])
    manual = sum(1 for r in rows if r["MANUAL_NEEDED"])
    print(f"  Wrote {len(rows)} rows → {path.name}")
    print(f"  Challenger viability flags: {viable} districts")
    print(f"  No-filing placeholders:     {manual} districts")


# ---------------------------------------------------------------------------
# Step 5: Merge into districts_2026.csv
# ---------------------------------------------------------------------------

def merge_into_districts(finance_rows: list[dict]):
    """
    Add/update challenger_viability_flag_early, incumbent_raised, challenger_raised
    columns in districts_2026.csv.
    """
    path = DATA_PROC / "districts_2026.csv"
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        orig_rows   = list(reader)

    # Build finance lookup
    finance_by_key = {
        (r["chamber"].lower(), int(r["district"])): r
        for r in finance_rows
    }

    # New columns to add / update
    new_cols = ["challenger_viability_flag_early", "incumbent_raised", "challenger_raised", "dem_fundraising_share"]
    out_fields = list(orig_fields) + [c for c in new_cols if c not in orig_fields]

    updated = 0
    for row in orig_rows:
        ch   = row["chamber"].strip().lower()
        dist = int(row["district"])
        fin  = finance_by_key.get((ch, dist))
        if fin:
            row["challenger_viability_flag_early"] = fin["challenger_viability_flag_early"]
            row["incumbent_raised"]      = fin.get("incumbent_raised", "")
            row["challenger_raised"]     = fin.get("challenger_raised", "")
            row["dem_fundraising_share"] = fin.get("dem_fundraising_share", "")
            updated += 1
        else:
            row.setdefault("challenger_viability_flag_early", 0)
            row.setdefault("incumbent_raised", "")
            row.setdefault("challenger_raised", "")
            row.setdefault("dem_fundraising_share", "")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(orig_rows)

    print(f"  districts_2026.csv updated: {updated} rows merged, {len(orig_rows)} total rows")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict]):
    print(f"\n{'='*60}")
    print("  2026 FINANCE SUMMARY — Early-Cycle")
    print(f"{'='*60}")

    viable_rows = [r for r in rows if r["challenger_viability_flag_early"]]
    print(f"\n  Challenger viability flags set: {len(viable_rows)}")

    house_viable  = [r for r in viable_rows if r["chamber"].lower() == "house"]
    senate_viable = [r for r in viable_rows if r["chamber"].lower() == "senate"]
    print(f"    House: {len(house_viable)}    Senate: {len(senate_viable)}")

    if viable_rows:
        print(f"\n  Districts with viable challengers:")
        print(f"  {'Chamber':7s}  {'Dist':4s}  {'Inc Party':9s}  {'Inc $':>10s}  "
              f"{'Chal $':>10s}  {'Challengers'}")
        print(f"  {'-'*7}  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*30}")
        for r in sorted(viable_rows, key=lambda x: (-x["challenger_raised"] if x["challenger_raised"] else 0,)):
            inc_r   = r["incumbent_raised"]   or 0
            chal_r  = r["challenger_raised"]  or 0
            challengers = (r["challenger_names"] or "")[:40]
            print(f"  {r['chamber']:7s}  {r['district']:4d}  {r['incumbent_party']:9s}  "
                  f"${inc_r:>9,.0f}  ${chal_r:>9,.0f}  {challengers}")

    # Coverage stats
    total_with_data = sum(1 for r in rows if not r["MANUAL_NEEDED"])
    print(f"\n  Districts with any 2026 TEC filings: {total_with_data} / {len(rows)}")
    print(f"  Data through: {FINANCE_DATA_THROUGH}")
    print(f"  Viability thresholds: House=${THRESHOLD['house']:,.0f}  "
          f"Senate=${THRESHOLD['senate']:,.0f}")
    print(f"\n  NOTE: Challenger party not directly available from TEC pre-primary data.")
    print(f"  Viability flag = 1 if any non-incumbent raises above threshold.")
    print(f"  This may include same-party primary challengers.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect 2026 TX legislative campaign finance from TEC master ZIP")
    parser.add_argument("--no-merge", action="store_true",
                        help="Write finance CSV but don't update districts_2026.csv")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only; don't write any files")
    args = parser.parse_args()

    print("=" * 60)
    print("  TX Legislature 2026 — Finance Collection")
    print(f"  Cycle window: {CYCLE_START} → {CYCLE_END}")
    print("=" * 60)
    print()

    by_district = load_2026_cover()
    districts_info = load_districts_2026()
    rows = assign_parties_and_aggregate(by_district, districts_info)

    print_summary(rows)

    if args.summary:
        return

    out_path = DATA_RAW / "tx_finance_2026.csv"
    print(f"\nWriting finance CSV...")
    write_finance_csv(rows, out_path)

    if not args.no_merge:
        print(f"\nMerging into districts_2026.csv...")
        merge_into_districts(rows)
        print("\nRe-run projections:  python src/model.py")


if __name__ == "__main__":
    main()
