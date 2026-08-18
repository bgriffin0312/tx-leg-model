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
    VIABILITY_THRESHOLD as THRESHOLD,  # era auto-selected from FINANCE_DATA_THROUGH
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
    rows_straddling = 0  # reports where start is outside window but end is inside
    rows_hold_fallback = 0  # matched via filerHoldOffice* because seek fields were blank

    for row in reader:
        seek_office = row.get("filerSeekOfficeCd", "").strip().upper()
        hold_office_cd = row.get("filerHoldOfficeCd", "").strip().upper()
        dist_raw = re.sub(r"\D", "", row.get("filerSeekOfficeDistrict", "").strip())
        hold_dist_raw = re.sub(r"\D", "", row.get("filerHoldOfficeDistrict", "").strip())

        # TEC leaves the *seek* fields blank on many July semi-annual reports
        # filed by sitting legislators (they file as officeholders, with no
        # candidacy designated). Keying only on seek fields silently dropped
        # those reports — 36 reports / $2.7M / 35 districts in the 2026 window,
        # and it made 5 districts look like they had no filings at all.
        #
        # Fall back to the *hold* fields ONLY when seek is entirely blank.
        # If seek names a DIFFERENT office, the filer is running statewide
        # (Middleton→ATTYGEN, Hinojosa→GOVERNOR, Goodwin→LTGOVERNOR,
        # Huffines→COMPTROLLER) and that money must NOT be attributed to the
        # legislative seat they currently hold — $24M across 12 districts,
        # including SD 11, which is in the competitive set.
        chamber = OFFICE_CODES.get(seek_office)
        if chamber is None:
            if seek_office:
                continue  # seeking a different office — not this district's race
            chamber = OFFICE_CODES.get(hold_office_cd)
            if chamber is None:
                continue
            rows_hold_fallback += 1

        # District: seek district if present, else the seat they hold.
        # (A member seeking a different district with a blank seekDist would be
        # mis-assigned to their current seat; no such case in the 2026 window.)
        if not dist_raw:
            dist_raw = hold_dist_raw
        if not dist_raw:
            continue
        district = int(dist_raw)
        if district < 1 or district > (150 if chamber == "house" else 31):
            continue

        # 2026 partial-cycle: use periodStartDt because we want reports whose
        # coverage period begins within our window. Historical pipeline uses
        # periodEndDt because full-cycle aggregation needs the end date.
        # A report starting Dec 2025 and ending Jan 2026 will be missed here
        # but this is acceptable for early-cycle data; re-run after July filing
        # will catch full-cycle reports.
        period_start = re.sub(r"\D", "", row.get("periodStartDt", "").strip())[:8]
        period_end   = re.sub(r"\D", "", row.get("periodEndDt",   "").strip())[:8]
        if len(period_start) < 8 or not (CYCLE_START <= period_start <= CYCLE_END):
            # Track straddling reports: start outside window but end inside
            if (len(period_end) >= 8 and CYCLE_START <= period_end <= CYCLE_END
                    and seek_office in OFFICE_CODES):
                rows_straddling += 1
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
    if rows_hold_fallback:
        print(f"  {rows_hold_fallback} report(s) matched via filerHoldOffice* "
              f"(seek fields blank — officeholder semi-annual filings)")
    if rows_straddling:
        print(f"  NOTE: {rows_straddling} legislative reports straddled the cycle boundary "
              f"(periodEnd in window but periodStart outside). These are excluded; "
              f"re-run after July filing to capture full-cycle reports.")
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

def _load_candidates_2026() -> dict[tuple, dict]:
    """
    Load data/processed/candidates_2026.csv → {(chamber_lower, district): {r: name, d: name}}.
    Names are normalized for fuzzy matching against TEC filer names.
    """
    path = DATA_PROC / "candidates_2026.csv"
    if not path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        ch = str(row["chamber"]).strip().lower()
        dist = int(row["district"])
        r_cand = str(row.get("r_candidate") or "").strip() if pd.notna(row.get("r_candidate")) else ""
        d_cand = str(row.get("d_candidate") or "").strip() if pd.notna(row.get("d_candidate")) else ""
        lookup[(ch, dist)] = {"r": r_cand, "d": d_cand}
    return lookup


def _match_filer_to_candidate(filer_name: str, candidate_name: str) -> bool:
    """Check if a TEC filer name matches a candidate name (fuzzy last-name match)."""
    if not candidate_name:
        return False
    return _name_match(_normalize_name(filer_name), _normalize_name(candidate_name))


def _match_strength(filer_name: str, candidate_name: str) -> int:
    """
    Graded match between a TEC filer and a nominee, so a filer who loosely
    matches BOTH nominees can be assigned to the stronger side.
      3 = exact normalized equality or >=2 significant common tokens
          ("LONGORIA OSCAR L" vs "OSCAR LONGORIA")
      1 = weaker fuzzy match (single long surname token, prefix rule) —
          enough when only one side matches ("MORALES HERIBERTO" vs
          "EDDIE MORALES"), never enough to beat a 3 on the other side
          ("LONGORIA OSCAR L" vs "OSCAR ROSA" scores 1 and loses)
      0 = no match
    """
    if not candidate_name:
        return 0
    a = _normalize_name(filer_name)
    b = _normalize_name(candidate_name)
    if not _name_match(a, b):
        return 0
    if a == b:
        return 3
    tokens_a = set(t for t in a.split() if len(t) >= 4)
    tokens_b = set(t for t in b.split() if len(t) >= 4)
    return 3 if len(tokens_a & tokens_b) >= 2 else 1


def assign_parties_and_aggregate(by_district: dict, districts_info: dict) -> list[dict]:
    """
    For each district with TEC data:
      - For districts with known 2026 nominees (from candidates_2026.csv):
        match filers to R/D nominees by name, compute per-party fundraising
      - For incumbent-held seats without candidate data:
        use incumbent vs challenger classification
      - Compute early-cycle viability flag based on opposition fundraising

    Returns list of row dicts ready for CSV output.
    """
    rows = []
    candidates = _load_candidates_2026()

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

        # Check if we have known 2026 nominees for this district
        cand = candidates.get((chamber, district))

        r_total = 0.0
        d_total = 0.0
        r_name_found = ""
        d_name_found = ""
        unmatched_names = []

        if cand and (cand["r"] or cand["d"]):
            # --- Nominee-based matching: match filers to known R/D nominees ---
            for fid, fdata in filers.items():
                name = fdata["name"]
                total = fdata["total"]

                # Score against BOTH nominees and take the stronger side.
                # First-match-wins with R checked first booked Oscar Longoria's
                # (D inc, HD35) own filings to R challenger "Oscar Rosa" via a
                # shared-first-name fuzzy hit; strength comparison fixes that
                # while keeping loose single-surname matches when unambiguous.
                sr = _match_strength(name, cand["r"]) if cand["r"] else 0
                sd = _match_strength(name, cand["d"]) if cand["d"] else 0
                if sr > sd:
                    r_total += total
                    if not r_name_found:
                        r_name_found = name
                elif sd > sr:
                    d_total += total
                    if not d_name_found:
                        d_name_found = name
                else:
                    # sr == sd: either no match, or an ambiguous tie — leave out
                    unmatched_names.append(name)

            # Viability: does the opposition party nominee have meaningful funding?
            if incumbent_party == "R" or (open_seat and incumbent_party == "R"):
                opp_total = d_total  # D is the opposition
            elif incumbent_party == "D" or (open_seat and incumbent_party == "D"):
                opp_total = r_total  # R is the opposition
            else:
                opp_total = max(r_total, d_total)

            threshold = THRESHOLD.get(chamber, 100_000)
            viability_flag = int(opp_total >= threshold)

            # Signed flag: +1 = viable D opposition, -1 = viable R opposition,
            # 0 = no viable opposition / both parties viable in open seat /
            # opposition party undetermined. Sign carries the direction of
            # the partisan signal so the model coefficient (always positive)
            # pushes the prediction the right way.
            if viability_flag == 0:
                viable_opp_signed = 0
            elif incumbent_party == "R":
                viable_opp_signed = +1  # viable D opposition to R incumbent
            elif incumbent_party == "D":
                viable_opp_signed = -1  # viable R opposition to D incumbent
            else:
                # Open seat: sign by which party is viable.
                d_viable = d_total >= threshold
                r_viable = r_total >= threshold
                if d_viable and not r_viable:
                    viable_opp_signed = +1
                elif r_viable and not d_viable:
                    viable_opp_signed = -1
                else:
                    viable_opp_signed = 0  # both viable, or neither cleanly assigned

            # dem_fundraising_share from actual nominee totals
            # Minimum $10K total raised to compute a meaningful ratio;
            # below that, one-sided filings produce 0.0 or 1.0 from noise
            _MIN_TOTAL_FOR_SHARE = 10_000
            if (r_total + d_total) >= _MIN_TOTAL_FOR_SHARE:
                dem_fundraising_share = round(d_total / (r_total + d_total), 4)
            else:
                # None → filled with 0.5 (neutral) in model.py line 270 via .fillna(0.5)
                dem_fundraising_share = None

            # For the model: incumbent_raised = party-of-seat nominee,
            # challenger_raised = opposition nominee
            if incumbent_party == "R":
                inc_raised = r_total
                chal_raised = d_total
            elif incumbent_party == "D":
                inc_raised = d_total
                chal_raised = r_total
            else:
                inc_raised = 0.0
                chal_raised = max(r_total, d_total)

            method = "nominee_match"
            chal_display = "; ".join(
                [n for n in [r_name_found, d_name_found] if n]
                + unmatched_names[:3]
            )
        else:
            # --- Legacy: incumbent vs challenger classification ---
            inc_total  = 0.0
            chal_total = 0.0
            inc_name_found   = ""
            chal_names_found = []

            for fid, fdata in filers.items():
                name  = fdata["name"]
                total = fdata["total"]
                is_inc_by_hold = fdata["is_incumbent"]

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

            if open_seat:
                # Legacy path can't determine party affiliation of filers
                # without nominee data. Setting viability=0 and dem_share=None
                # (neutral) avoids misclassifying all filer money as one party.
                # These districts need entries in candidates_2026.csv for
                # proper nominee-matched finance.
                inc_total  = 0.0
                chal_total = sum(f["total"] for f in filers.values())
                viability_flag = 0
                viable_opp_signed = 0
                dem_fundraising_share = None
            else:
                threshold = THRESHOLD.get(chamber, 100_000)
                viability_flag = int(chal_total >= threshold)

                # Legacy path: no party data on challengers. Assume well-funded
                # opposition is the opposite-party general nominee. This is
                # mostly true post-primary but produces noise when a same-party
                # primary challenger raises money — those races are typically
                # safe enough that the noise doesn't move win probabilities.
                if viability_flag == 0:
                    viable_opp_signed = 0
                elif incumbent_party == "R":
                    viable_opp_signed = +1
                elif incumbent_party == "D":
                    viable_opp_signed = -1
                else:
                    viable_opp_signed = 0

                _MIN_TOTAL_FOR_SHARE = 10_000
                if incumbent_party == "D" and (inc_total + chal_total) >= _MIN_TOTAL_FOR_SHARE:
                    dem_fundraising_share = round(inc_total / (inc_total + chal_total), 4)
                elif incumbent_party == "R" and (inc_total + chal_total) >= _MIN_TOTAL_FOR_SHARE:
                    dem_fundraising_share = round(chal_total / (inc_total + chal_total), 4)
                else:
                    dem_fundraising_share = None

            inc_raised = inc_total
            chal_raised = chal_total
            method = "hold_office_and_name_match"
            chal_display = "; ".join(chal_names_found[:5])

        rows.append({
            "year":                          2026,
            "chamber":                       chamber.title(),
            "district":                      district,
            "incumbent_name_tec":            r_name_found or d_name_found or "",
            "incumbent_party":               incumbent_party,
            "incumbent_raised":              round(inc_raised, 2),
            "challenger_raised":             round(chal_raised, 2),
            "dem_fundraising_share":         dem_fundraising_share,
            "challenger_names":              chal_display,
            "party_assignment_method":       method,
            "challenger_viability_flag_early": viability_flag,
            "viable_opposition_signed":      viable_opp_signed,
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
        "viable_opposition_signed":      0,
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
    new_cols = ["challenger_viability_flag_early", "viable_opposition_signed",
                "incumbent_raised", "challenger_raised", "dem_fundraising_share"]
    out_fields = list(orig_fields) + [c for c in new_cols if c not in orig_fields]

    updated = 0
    for row in orig_rows:
        ch   = row["chamber"].strip().lower()
        dist = int(row["district"])
        fin  = finance_by_key.get((ch, dist))
        if fin:
            row["challenger_viability_flag_early"] = fin["challenger_viability_flag_early"]
            row["viable_opposition_signed"] = fin.get("viable_opposition_signed", 0)
            row["incumbent_raised"]      = fin.get("incumbent_raised", "")
            row["challenger_raised"]     = fin.get("challenger_raised", "")
            row["dem_fundraising_share"] = fin.get("dem_fundraising_share", "")
            updated += 1
        else:
            # Explicitly clear — don't preserve stale data from prior runs
            row["challenger_viability_flag_early"] = 0
            row["viable_opposition_signed"] = 0
            row["incumbent_raised"] = ""
            row["challenger_raised"] = ""
            row["dem_fundraising_share"] = ""

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
    parser.add_argument("--force-download", action="store_true",
                        help="Bypass the on-disk TEC cache and re-download "
                             "(the cache has no TTL; without this a stale "
                             "tec_*.csv is reused and looks like 'no new filings')")
    args = parser.parse_args()

    if args.force_download:
        import collect_finance
        collect_finance.FORCE_TEC_DOWNLOAD = True

    print("=" * 60)
    print("  TX Legislature 2026 — Finance Collection")
    print(f"  Cycle window: {CYCLE_START} → {CYCLE_END}")
    print(f"  TEC cache: {'BYPASSED (--force-download)' if args.force_download else 'enabled'}")
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
