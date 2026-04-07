"""
collect_ies_2026.py

Collect 2026 Texas legislative independent expenditure (IE) data from the Texas
Ethics Commission (TEC) master ZIP, then classify each expenditure as FOR or
AGAINST a specific candidate using:
  1. TEC spacs.csv — SPAC (Super PAC) filings with explicit SUPPORT/OPPOSE position codes
  2. Claude API — classify free-text purpose descriptions for non-SPAC expenditures

OUTPUT:
  data/raw/tx_ies_2026.csv — one row per IE filing with direction, amount, district
  Also merges ie_dem_total, ie_rep_total, ie_net into districts_2026.csv

WHY IEs MATTER:
  Independent expenditures (IEs) are spending by outside groups FOR or AGAINST
  candidates. In competitive TX legislative districts, IEs can dwarf candidate
  fundraising. Adding IE totals improves the model's dem_fundraising_share proxy
  and may reduce Brier score in backtests.

TEC DATA SOURCES:
  spacs.csv    — 504 rows: spacPositionCd (SUPPORT/OPPOSE), candidateFilerIdent,
                 candidateSeekOfficeCd, candidateSeekOfficeDistrict, filerIdent.
                 This directly encodes IE direction without needing Claude.
  expend_*.csv — 13 files of itemized expenditures. Contains all PAC + candidate
                 spending. Filter for filerIdent matching SPAC filers.
  purpose.csv  — Free-text expenditure purpose descriptions. Used with Claude
                 for non-SPAC PAC spending on legislative races.

USAGE:
  python src/collect_ies_2026.py                  # full run (spacs + Claude classify)
  python src/collect_ies_2026.py --spac-only      # only process spacs.csv (no Claude)
  python src/collect_ies_2026.py --no-merge       # don't update districts_2026.csv
  python src/collect_ies_2026.py --summary        # print summary only

ENVIRONMENT:
  ANTHROPIC_API_KEY — required for Claude classification (set in .env)
  Skip Claude with --spac-only if key not available.

COST ESTIMATE:
  Claude classification: ~200-500 purpose strings × ~100 tokens each
  ≈ 50K tokens → ~$0.05-0.15 at Haiku pricing
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).parent))
from collect_finance import (
    _tec_zip_central_dir,
    _tec_extract_file,
    TEC_ZIP_URL,
    TEC_ENCODING,
)

OUTPUT_PATH    = DATA_RAW / "tx_ies_2026.csv"
DISTRICTS_PATH = DATA_PROC / "districts_2026.csv"

USER_AGENT = "TXLegislativeModel/1.0 (academic research; non-commercial)"

# TEC office codes for TX legislative races
OFFICE_CODES = {"STATEREP": "house", "STATESEN": "senate"}

# Senate districts on the 2026 ballot
SENATE_2026_BALLOT = {1, 2, 3, 4, 5, 9, 11, 13, 18, 19, 21, 22, 24, 26, 28, 31}

# Date window: 2026 cycle
CYCLE_START = "20260101"
CYCLE_END   = "20261231"  # full cycle (IEs happen through general election)

OUTPUT_FIELDS = [
    "chamber", "district", "direction", "filer_name", "filer_id",
    "candidate_name", "candidate_filer_id", "amount", "expenditure_date",
    "purpose_description", "classification_method", "source_file",
]


# ---------------------------------------------------------------------------
# Step 1: Extract and parse spacs.csv
# ---------------------------------------------------------------------------

def load_spacs(cd: dict, verbose: bool = False) -> dict[str, dict]:
    """
    Extract spacs.csv from TEC ZIP. Returns dict of:
      {spac_filer_id: {
          "position": "SUPPORT"|"OPPOSE",
          "candidate_filer_id": str,
          "chamber": "house"|"senate",
          "district": int,
          "candidate_name": str,
      }}

    A SPAC (political action committee) files a SPAC report declaring whether they
    SUPPORT or OPPOSE a specific candidate. This is the cleanest IE direction signal.
    """
    spac_fname = next((f for f in cd if "spac" in f.lower() and f.lower().endswith(".csv")), None)
    if not spac_fname:
        print("  spacs.csv not found in TEC ZIP central directory")
        return {}

    print(f"\nExtracting {spac_fname}...")
    data = _tec_extract_file(TEC_ZIP_URL, cd[spac_fname], spac_fname)
    if not data:
        print("  Failed to extract spacs.csv")
        return {}

    text = data.decode(TEC_ENCODING, errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    if verbose:
        # Print first row to confirm columns
        rows = list(reader)
        if rows:
            print(f"  spacs.csv columns: {list(rows[0].keys())}")
        reader_data = rows
    else:
        reader_data = reader

    spacs: dict[str, dict] = {}
    skipped = 0

    for row in reader_data:
        # Key fields (TEC column names confirmed from ZIP inspection)
        office_cd = row.get("candidateSeekOfficeCd", "").strip().upper()
        chamber = OFFICE_CODES.get(office_cd)
        if chamber is None:
            skipped += 1
            continue

        dist_raw = re.sub(r"\D", "", row.get("candidateSeekOfficeDistrict", "").strip())
        if not dist_raw:
            skipped += 1
            continue
        district = int(dist_raw)

        if chamber == "senate" and district not in SENATE_2026_BALLOT:
            skipped += 1
            continue

        position = row.get("spacPositionCd", "").strip().upper()
        if position not in ("SUPPORT", "OPPOSE"):
            # Try alternative column names
            position = row.get("positionCd", "").strip().upper()
            if position not in ("SUPPORT", "OPPOSE"):
                skipped += 1
                continue

        filer_id          = row.get("filerIdent", "").strip()
        cand_filer_id     = row.get("candidateFilerIdent", "").strip()
        cand_name         = row.get("candidateName", "").strip()

        if not filer_id:
            skipped += 1
            continue

        spacs[filer_id] = {
            "position":           position,
            "candidate_filer_id": cand_filer_id,
            "candidate_name":     cand_name,
            "chamber":            chamber,
            "district":           district,
        }

    print(f"  Parsed {len(spacs)} SPAC → candidate mappings for TX legislative races")
    if skipped:
        print(f"  Skipped {skipped} rows (other offices or missing data)")

    return spacs


# ---------------------------------------------------------------------------
# Step 2: Extract expenditures from expend_*.csv files
# ---------------------------------------------------------------------------

def load_expenditures_for_spacs(
    cd: dict,
    spac_filer_ids: set[str],
    verbose: bool = False,
) -> list[dict]:
    """
    Extract expenditure rows from expend_01.csv through expend_13.csv
    where filerIdent is in the SPAC set and date is in the 2026 cycle.

    Returns list of expenditure dicts.
    """
    expend_files = sorted(f for f in cd if re.match(r"expend_\d+\.csv", f, re.IGNORECASE))
    if not expend_files:
        print("  No expend_*.csv files found in TEC ZIP")
        return []

    print(f"\nFound {len(expend_files)} expenditure files: {expend_files[:3]}{'...' if len(expend_files) > 3 else ''}")

    expenditures = []
    for fname in expend_files:
        print(f"  Scanning {fname}...")
        data = _tec_extract_file(TEC_ZIP_URL, cd[fname], fname)
        if not data:
            print(f"  Failed to extract {fname}, skipping")
            continue

        text = data.decode(TEC_ENCODING, errors="replace")
        reader = csv.DictReader(io.StringIO(text))

        file_count = 0
        for row in reader:
            filer_id = row.get("filerIdent", "").strip()
            if filer_id not in spac_filer_ids:
                continue

            # Date filter
            expend_date = re.sub(r"\D", "", row.get("expendDt", "").strip())[:8]
            if len(expend_date) < 8 or not (CYCLE_START <= expend_date <= CYCLE_END):
                continue

            amount_raw = row.get("expendAmount", "0").strip().replace(",", "").replace("$", "")
            try:
                amount = float(amount_raw)
            except ValueError:
                continue
            if amount <= 0:
                continue

            purpose = row.get("expendPurposeDesc", "").strip() or row.get("purposeOfExpenditure", "").strip()

            expenditures.append({
                "filer_id":           filer_id,
                "filer_name":         row.get("filerName", "").strip(),
                "amount":             amount,
                "expenditure_date":   expend_date,
                "purpose_description": purpose,
                "source_file":        fname,
            })
            file_count += 1

        print(f"    {fname}: {file_count} rows matched SPAC filers in 2026")

    print(f"  Total expenditure rows for SPAC filers: {len(expenditures)}")
    return expenditures


# ---------------------------------------------------------------------------
# Step 3: Classify non-SPAC PAC expenditures using Claude API
# ---------------------------------------------------------------------------

def classify_with_claude(purpose_texts: list[str], districts_hint: list[dict]) -> list[str | None]:
    """
    Use Claude Haiku to classify expenditure purpose descriptions as:
      "SUPPORT" — money spent to help a candidate win
      "OPPOSE"  — money spent to hurt a candidate
      None      — cannot determine or not a legislative IE

    Batches requests for efficiency.

    Args:
        purpose_texts: list of free-text purpose descriptions
        districts_hint: list of dicts with {district, chamber, candidate_name} for context

    Returns:
        list of "SUPPORT" | "OPPOSE" | None, same length as purpose_texts
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — skipping Claude classification")
        return [None] * len(purpose_texts)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    batch_size = 20  # classify 20 at a time to reduce API calls

    for i in range(0, len(purpose_texts), batch_size):
        batch_texts = purpose_texts[i:i + batch_size]
        batch_hints = districts_hint[i:i + batch_size]

        # Build prompt
        items = []
        for j, (text, hint) in enumerate(zip(batch_texts, batch_hints)):
            hint_str = ""
            if hint.get("candidate_name"):
                hint_str = f" (candidate: {hint['candidate_name']}, {hint.get('chamber', '')} district {hint.get('district', '')})"
            items.append(f"{j+1}. [{text[:200]}]{hint_str}")

        prompt = (
            "You are classifying Texas campaign expenditures.\n\n"
            "For each expenditure purpose below, determine if it represents:\n"
            "  SUPPORT — spending that helps a candidate (pro-candidate advertising, GOTV for candidate, etc.)\n"
            "  OPPOSE  — spending that hurts a candidate (negative ads, opposition research, etc.)\n"
            "  UNKNOWN — cannot determine or not a candidate-specific IE\n\n"
            "Respond with a JSON array of strings, one per item: [\"SUPPORT\", \"OPPOSE\", \"UNKNOWN\", ...]\n"
            "No explanation needed.\n\n"
            "Expenditures:\n" + "\n".join(items)
        )

        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text.strip()

            # Parse JSON array
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                classifications = json.loads(json_match.group())
                for c in classifications:
                    if c.upper() in ("SUPPORT", "OPPOSE"):
                        results.append(c.upper())
                    else:
                        results.append(None)
            else:
                results.extend([None] * len(batch_texts))

        except Exception as exc:
            print(f"  Claude API error (batch {i//batch_size + 1}): {exc}")
            results.extend([None] * len(batch_texts))

        time.sleep(0.5)  # rate limiting

    return results


# ---------------------------------------------------------------------------
# Step 4: Assemble final IE rows
# ---------------------------------------------------------------------------

def assemble_ie_rows(
    spacs: dict[str, dict],
    expenditures: list[dict],
    candidate_party: dict[str, str],  # spac_filer_id -> "D"|"R"|"unknown"
) -> list[dict]:
    """
    Join SPAC position data with expenditure amounts to produce final IE rows.

    Direction mapping:
      SPAC SUPPORT D candidate → D_favor (IE helps D)
      SPAC SUPPORT R candidate → R_favor (IE helps R)
      SPAC OPPOSE  D candidate → R_favor (hurts D = helps R)
      SPAC OPPOSE  R candidate → D_favor (hurts R = helps D)
    """
    rows = []

    for exp in expenditures:
        filer_id = exp["filer_id"]
        spac_info = spacs.get(filer_id)
        if not spac_info:
            continue

        chamber   = spac_info["chamber"]
        district  = spac_info["district"]
        position  = spac_info["position"]  # SUPPORT or OPPOSE
        cand_name = spac_info["candidate_name"]

        cand_party = candidate_party.get(filer_id, "unknown")
        direction  = _resolve_direction(position, cand_party, cand_name)

        rows.append({
            "chamber":               chamber.title(),
            "district":              district,
            "direction":             direction,
            "filer_name":            exp["filer_name"],
            "filer_id":              filer_id,
            "candidate_name":        cand_name,
            "candidate_filer_id":    spac_info["candidate_filer_id"],
            "amount":                round(exp["amount"], 2),
            "expenditure_date":      exp["expenditure_date"],
            "purpose_description":   exp["purpose_description"],
            "classification_method": "spac_position_code",
            "source_file":           exp["source_file"],
        })

    return rows


def _resolve_direction(position: str, cand_party: str | None, cand_name: str) -> str:
    """
    Convert SPAC SUPPORT/OPPOSE + candidate party to net direction.

    Returns: "D_favor" | "R_favor" | "unknown"
      D_favor = IE net helps Democrats
      R_favor = IE net helps Republicans
    """
    if not cand_party:
        # Can't resolve without knowing candidate party
        return f"{position.lower()}_unknown_party"

    if position == "SUPPORT":
        return "D_favor" if cand_party == "D" else "R_favor"
    elif position == "OPPOSE":
        # Opposing D → helps R; opposing R → helps D
        return "R_favor" if cand_party == "D" else "D_favor"
    return "unknown"


# ---------------------------------------------------------------------------
# Step 5: District-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_district(ie_rows: list[dict]) -> list[dict]:
    """
    Sum IE totals by (chamber, district, direction).
    Returns list: {chamber, district, ie_d_favor, ie_r_favor, ie_net_d}
    """
    totals: dict[tuple, dict] = {}
    for row in ie_rows:
        key = (row["chamber"], row["district"])
        if key not in totals:
            totals[key] = {"ie_d_favor": 0.0, "ie_r_favor": 0.0}

        direction = row["direction"]
        amount = float(row["amount"])
        if "d_favor" in direction.lower():
            totals[key]["ie_d_favor"] += amount
        elif "r_favor" in direction.lower():
            totals[key]["ie_r_favor"] += amount
        # "unknown" directions excluded from totals

    result = []
    for (chamber, district), vals in sorted(totals.items()):
        ie_net_d = vals["ie_d_favor"] - vals["ie_r_favor"]
        result.append({
            "chamber":     chamber,
            "district":    district,
            "ie_d_favor":  round(vals["ie_d_favor"], 2),
            "ie_r_favor":  round(vals["ie_r_favor"], 2),
            "ie_net_d":    round(ie_net_d, 2),
        })
    return result


# ---------------------------------------------------------------------------
# Step 6: Merge into districts_2026.csv
# ---------------------------------------------------------------------------

def merge_ies_into_districts(aggregated: list[dict]):
    """Add ie_d_favor, ie_r_favor, ie_net_d columns to districts_2026.csv."""
    if not DISTRICTS_PATH.exists():
        print(f"  {DISTRICTS_PATH.name} not found")
        return

    ie_lookup = {(r["chamber"].lower(), int(r["district"])): r for r in aggregated}

    with open(DISTRICTS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        orig_rows   = list(reader)

    new_cols = ["ie_d_favor", "ie_r_favor", "ie_net_d"]
    out_fields = list(orig_fields) + [c for c in new_cols if c not in orig_fields]

    for row in orig_rows:
        ch   = row["chamber"].strip().lower()
        dist = int(row["district"])
        ie   = ie_lookup.get((ch, dist))
        if ie:
            row["ie_d_favor"] = ie["ie_d_favor"]
            row["ie_r_favor"] = ie["ie_r_favor"]
            row["ie_net_d"]   = ie["ie_net_d"]
        else:
            row.setdefault("ie_d_favor", 0)
            row.setdefault("ie_r_favor", 0)
            row.setdefault("ie_net_d",  0)

    with open(DISTRICTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(orig_rows)

    print(f"  districts_2026.csv updated with IE totals ({len(aggregated)} districts)")


# ---------------------------------------------------------------------------
# Write output CSV
# ---------------------------------------------------------------------------

def write_ie_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} IE rows to {path.name}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(ie_rows: list[dict], aggregated: list[dict]):
    print(f"\n{'='*65}")
    print("  2026 INDEPENDENT EXPENDITURE SUMMARY")
    print(f"{'='*65}")

    total_d  = sum(r["ie_d_favor"] for r in aggregated)
    total_r  = sum(r["ie_r_favor"] for r in aggregated)
    total_all = total_d + total_r

    print(f"\n  Total IEs classified: ${total_all:>12,.0f}")
    print(f"    Favoring Democrats: ${total_d:>12,.0f}")
    print(f"    Favoring Republicans: ${total_r:>10,.0f}")

    districts_with_ie = len(aggregated)
    print(f"\n  Districts with IE activity: {districts_with_ie}")

    # Top districts by total IE spending
    top = sorted(aggregated, key=lambda r: r["ie_d_favor"] + r["ie_r_favor"], reverse=True)[:10]
    if top:
        print(f"\n  Top 10 districts by IE activity:")
        print(f"  {'Chamber':7s}  {'Dist':4s}  {'D-Favor':>12s}  {'R-Favor':>12s}  {'Net D':>10s}")
        print(f"  {'-'*7}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}")
        for r in top:
            print(f"  {r['chamber']:7s}  {r['district']:4d}  "
                  f"${r['ie_d_favor']:>11,.0f}  ${r['ie_r_favor']:>11,.0f}  "
                  f"${r['ie_net_d']:>+10,.0f}")

    # Classification breakdown
    methods = {}
    for row in ie_rows:
        m = row["classification_method"]
        methods[m] = methods.get(m, 0) + 1
    print(f"\n  Classification methods: {dict(methods)}")


# ---------------------------------------------------------------------------
# Load candidate party from primary results or districts
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip punctuation, single spaces."""
    name = name.lower()
    name = re.sub(r"[,\.\-']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    for suffix in [" jr", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def _names_match(a: str, b: str) -> bool:
    a_tok = {t for t in _norm_name(a).split() if len(t) > 1}
    b_tok = {t for t in _norm_name(b).split() if len(t) > 1}
    if not a_tok or not b_tok:
        return False
    longest_a = max(a_tok, key=len)
    longest_b = max(b_tok, key=len)
    return longest_a == longest_b or len(a_tok & b_tok) >= 2


def load_candidate_party(spacs: dict[str, dict]) -> dict[str, str]:
    """
    Build a filer-ID → party map for SPAC target candidates.

    Strategy (in priority order):
      1. Primary results (tx_primary_2026.csv) — name-match against spacs candidate names
      2. districts_2026.csv incumbent party — for incumbents running in the general
      3. Fallback: "unknown"

    Returns {spac_filer_id: "D"|"R"|"unknown"} indicating whether the *target* candidate
    is the Democratic or Republican candidate in that district.
    """
    # Build: (chamber, district) → {D: winner_name, R: winner_name}
    primary_winners: dict[tuple, dict] = {}
    primary_path = DATA_RAW / "tx_primary_2026.csv"
    if primary_path.exists():
        with open(primary_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("runoff_needed", "")).lower() in ("true", "1", "yes"):
                    continue  # skip pending runoffs
                key = (row["chamber"].strip().lower(), int(row["district"]))
                party = row["party"].strip().upper()
                winner = row.get("winner_name", "").strip()
                if party in ("D", "R") and winner:
                    primary_winners.setdefault(key, {})[party] = winner
        print(f"  Loaded primary winners for {len(primary_winners)} districts")

    # Load incumbent party from districts_2026.csv
    incumbent_party: dict[tuple, str] = {}
    if DISTRICTS_PATH.exists():
        with open(DISTRICTS_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                party = row.get("incumbent_party", "").strip().upper()
                if party in ("D", "R"):
                    key = (row["chamber"].strip().lower(), int(row["district"]))
                    incumbent_party[key] = party

    # Resolve each SPAC → target party
    result: dict[str, str] = {}
    for spac_id, info in spacs.items():
        key     = (info["chamber"], info["district"])
        cand_nm = info.get("candidate_name", "")

        # Try primary results first: does the candidate name match D or R winner?
        pri = primary_winners.get(key, {})
        if pri and cand_nm:
            if pri.get("D") and _names_match(cand_nm, pri["D"]):
                result[spac_id] = "D"
                continue
            if pri.get("R") and _names_match(cand_nm, pri["R"]):
                result[spac_id] = "R"
                continue

        # Fallback: if only one primary winner is known, use incumbent party
        inc_p = incumbent_party.get(key)
        if inc_p:
            # The SPAC's target may be incumbent or challenger
            # Without name-match, assume target is the general election candidate
            # whose party we can infer from incumbent + primary
            result[spac_id] = inc_p  # imperfect but best fallback
        else:
            result[spac_id] = "unknown"

    resolved = sum(1 for v in result.values() if v in ("D", "R"))
    print(f"  SPAC target party resolved: {resolved}/{len(result)} "
          f"({'from primary' if primary_winners else 'from incumbent data'})")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect 2026 TX legislative independent expenditures from TEC")
    parser.add_argument("--spac-only", action="store_true",
                        help="Only process spacs.csv (no Claude classification)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Don't update districts_2026.csv")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary only, don't write files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed column names and row samples")
    args = parser.parse_args()

    print("=" * 65)
    print("  TX Legislature 2026 — Independent Expenditure Collection")
    print(f"  Cycle window: {CYCLE_START} → {CYCLE_END}")
    print("=" * 65)

    print("\nReading TEC ZIP central directory...")
    cd = _tec_zip_central_dir(TEC_ZIP_URL)
    if not cd:
        print("ERROR: Could not read TEC ZIP central directory.")
        sys.exit(1)

    # Step 1: Parse spacs.csv for SUPPORT/OPPOSE mappings
    spacs = load_spacs(cd, verbose=args.verbose)
    if not spacs:
        print("\nNo SPAC data found. Ensure spacs.csv is present in TEC ZIP.")
        if args.spac_only:
            return

    # Step 2: Extract expenditure amounts for SPAC filers
    spac_filer_ids = set(spacs.keys())
    expenditures = []
    if spac_filer_ids:
        expenditures = load_expenditures_for_spacs(cd, spac_filer_ids, verbose=args.verbose)

    # Step 3: Claude classification for non-SPAC expenditures (future enhancement)
    # Currently, SPAC filings cover the main IEs. Non-SPAC PAC spending on legislative
    # races would require purpose.csv extraction + Claude. Skip for now with --spac-only.
    if not args.spac_only:
        print("\nNon-SPAC PAC classification via Claude: not yet implemented.")
        print("  Run with --spac-only to process SPAC data only.")
        print("  See purpose.csv for free-text descriptions (42MB uncompressed).")

    # Step 4: Load candidate party for direction resolution
    candidate_party = load_candidate_party(spacs)

    # Step 5: Assemble IE rows
    ie_rows = assemble_ie_rows(spacs, expenditures, candidate_party)
    print(f"\nAssembled {len(ie_rows)} IE rows with direction classification")

    # Step 6: Aggregate by district
    aggregated = aggregate_by_district(ie_rows)

    print_summary(ie_rows, aggregated)

    if args.summary:
        return

    # Step 7: Write IE CSV
    print(f"\nWriting IE data...")
    write_ie_csv(ie_rows, OUTPUT_PATH)

    # Step 8: Merge into districts_2026.csv
    if not args.no_merge:
        print(f"\nMerging IE totals into districts_2026.csv...")
        merge_ies_into_districts(aggregated)
        print("\nRe-run projections:  python src/model.py")


if __name__ == "__main__":
    main()
