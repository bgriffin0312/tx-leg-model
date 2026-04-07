"""
collect_primary_2026.py

Collect 2026 TX legislative primary election results (March 4, 2026) to fix
party assignment in campaign finance data.

WHY THIS MATTERS:
  TEC's politicalPartyCd column is empty for all 2026 filers. In collect_finance_2026.py,
  challenger party is unknown, so dem_fundraising_share is approximated using only the
  incumbent's known party. This script provides actual primary winners so we can
  assign D/R to every TEC filer by name-matching primary winners.

DATA SOURCES (tried in order):
  1. TX SOS Civix API — the official TX election results system (goelect.txelections.civixapps.com)
  2. TX SOS legacy results pages
  3. Manual override CSV — data/raw/tx_primary_2026_manual.csv
     (If auto-collection fails, fill this manually and re-run with --manual-only)

OUTPUT: data/raw/tx_primary_2026.csv
  Columns: chamber, district, party, winner_name, winner_votes, runoff_needed, source

USAGE:
  python src/collect_primary_2026.py               # try all sources
  python src/collect_primary_2026.py --manual-only # use manual CSV only
  python src/collect_primary_2026.py --update-finance # also update finance data

AFTER RUNNING:
  python src/collect_finance_2026.py  # re-run with primary party data

MANUAL OVERRIDE FORMAT (data/raw/tx_primary_2026_manual.csv):
  chamber,district,party,winner_name,winner_votes,runoff_needed
  House,32,D,Jane Smith,4521,False
  House,32,R,John Doe,7823,False
  ...
  (runoff_needed=True for May 27 runoff races)
"""

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

ROOT = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

USER_AGENT = "TXLegislativeModel/1.0 (academic research; non-commercial)"

OUTPUT_PATH   = DATA_RAW / "tx_primary_2026.csv"
MANUAL_PATH   = DATA_RAW / "tx_primary_2026_manual.csv"
FINANCE_PATH  = DATA_RAW / "tx_finance_2026.csv"
DISTRICTS_PATH = DATA_PROC / "districts_2026.csv"

SENATE_2026_BALLOT = {1, 2, 3, 4, 5, 9, 11, 13, 18, 19, 21, 22, 24, 26, 28, 31}

OUTPUT_FIELDS = [
    "chamber", "district", "party", "winner_name",
    "winner_votes", "runoff_needed", "source",
]


# ---------------------------------------------------------------------------
# Source 1: TX SOS Civix API
# ---------------------------------------------------------------------------

CIVIX_BASE = "https://goelect.txelections.civixapps.com"

def _civix_get(path: str, params: dict = None) -> dict | list | None:
    """GET from Civix API. Returns parsed JSON or None."""
    url = f"{CIVIX_BASE}{path}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Referer": f"{CIVIX_BASE}/ivis-enr-ui/",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"  Civix API {path}: HTTP {r.status_code}")
        return None
    except (requests.RequestException, json.JSONDecodeError) as exc:
        print(f"  Civix API {path}: {exc}")
        return None


def _find_2026_primary_election_id() -> str | None:
    """Try to find the 2026 primary election ID in the Civix API."""
    # Try common Civix API paths for election list
    for path in ["/api/elections", "/api/v1/elections", "/ivis-enr-ui/api/elections"]:
        data = _civix_get(path)
        if isinstance(data, list):
            for election in data:
                name = str(election.get("name", "") or election.get("electionName", "")).lower()
                if "2026" in name and ("primary" in name or "march" in name):
                    eid = election.get("id") or election.get("electionId")
                    print(f"  Found 2026 primary: id={eid} name={name!r}")
                    return str(eid)
        elif isinstance(data, dict):
            elections = data.get("elections", data.get("data", []))
            for election in elections:
                name = str(election.get("name", "") or election.get("electionName", "")).lower()
                if "2026" in name and ("primary" in name or "march" in name):
                    eid = election.get("id") or election.get("electionId")
                    print(f"  Found 2026 primary: id={eid} name={name!r}")
                    return str(eid)
    return None


def _fetch_civix_legislative_results(election_id: str) -> list[dict]:
    """
    Fetch state legislative race results from Civix API for a given election.
    Returns list of result rows: {chamber, district, party, winner_name, winner_votes, runoff_needed}
    """
    results = []

    # Civix typically has endpoints like /api/elections/{id}/races or /api/races
    for path_template in [
        f"/api/elections/{election_id}/races",
        f"/api/v1/elections/{election_id}/races",
        f"/api/races?electionId={election_id}",
    ]:
        data = _civix_get(path_template)
        if data is None:
            continue

        races = data if isinstance(data, list) else data.get("races", data.get("data", []))

        for race in races:
            race_name = str(race.get("raceName", "") or race.get("name", "")).strip()

            # Filter for state legislative races
            house_m = re.search(r"state representative.*?district\s*(\d+)", race_name, re.IGNORECASE)
            senate_m = re.search(r"state senator.*?district\s*(\d+)", race_name, re.IGNORECASE)
            if not house_m and not senate_m:
                continue

            chamber  = "House" if house_m else "Senate"
            district = int((house_m or senate_m).group(1))

            if chamber == "Senate" and district not in SENATE_2026_BALLOT:
                continue

            # Extract party from race name or field
            party_cd = str(race.get("partyCode", "") or race.get("party", "")).strip().upper()
            if not party_cd:
                if "democrat" in race_name.lower():
                    party_cd = "D"
                elif "republican" in race_name.lower():
                    party_cd = "R"

            if party_cd not in ("D", "R", "DEM", "REP"):
                continue
            party = "D" if party_cd.startswith("D") else "R"

            # Find winner
            candidates = race.get("candidates", [])
            winner = None
            runoff_needed = False
            for cand in candidates:
                if cand.get("winner") or cand.get("isWinner") or cand.get("status", "").lower() == "winner":
                    winner = cand
                    break
            if not winner and candidates:
                # Fallback: highest vote getter
                candidates_sorted = sorted(
                    candidates,
                    key=lambda c: int(c.get("votes", 0) or c.get("totalVotes", 0) or 0),
                    reverse=True
                )
                if candidates_sorted:
                    winner = candidates_sorted[0]

            # Check if runoff needed (no majority in primary)
            runoff_needed = bool(race.get("runoff") or race.get("runoffNeeded") or
                                  race.get("status", "").lower() == "runoff")

            if winner:
                name = str(winner.get("name", "") or winner.get("candidateName", "")).strip()
                votes = int(winner.get("votes", 0) or winner.get("totalVotes", 0) or 0)
                results.append({
                    "chamber":       chamber,
                    "district":      district,
                    "party":         party,
                    "winner_name":   name,
                    "winner_votes":  votes,
                    "runoff_needed": runoff_needed,
                    "source":        "civix_api",
                })

        if results:
            print(f"  Civix API: {len(results)} legislative race results fetched")
            return results

    return results


def try_civix(verbose: bool = False) -> list[dict]:
    """Try to collect 2026 primary results from the TX SOS Civix system."""
    print("\nTrying TX SOS Civix API...")
    election_id = _find_2026_primary_election_id()
    if not election_id:
        print("  Could not find 2026 primary election ID in Civix API.")
        return []
    return _fetch_civix_legislative_results(election_id)


# ---------------------------------------------------------------------------
# Source 2: TX SOS legacy HTML scraping
# ---------------------------------------------------------------------------

def try_sos_legacy() -> list[dict]:
    """Try TX SOS legacy election results page for 2026 primary."""
    print("\nTrying TX SOS legacy results...")
    urls_to_try = [
        "https://www.sos.state.tx.us/elections/historical/2026primary.shtml",
        "https://www.sos.state.tx.us/elections/results/2026-03-04.shtml",
        "https://elections.sos.state.tx.us/elchist.do",
    ]

    for url in urls_to_try:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code == 200 and "state representative" in r.text.lower():
                print(f"  Got content from {url}")
                return _parse_sos_html(r.text, "sos_html")
        except requests.RequestException:
            pass

    print("  TX SOS legacy results not accessible.")
    return []


def _parse_sos_html(html: str, source: str) -> list[dict]:
    """Very basic HTML parser for TX SOS results pages."""
    results = []
    # Look for patterns like "State Representative, District 32" with vote totals
    pat = re.compile(
        r"State\s+Representative.*?District\s+(\d+).*?"
        r"(Republican|Democratic)\s+Party.*?"
        r"([A-Z][a-zA-Z\s,]+?)\s+([\d,]+)\s+votes?",
        re.IGNORECASE | re.DOTALL
    )
    for m in pat.finditer(html):
        district = int(m.group(1))
        party_raw = m.group(2).lower()
        party = "R" if "republican" in party_raw else "D"
        name = m.group(3).strip().rstrip(",")
        votes = int(m.group(4).replace(",", ""))
        results.append({
            "chamber": "House", "district": district, "party": party,
            "winner_name": name, "winner_votes": votes,
            "runoff_needed": False, "source": source,
        })
    return results


# ---------------------------------------------------------------------------
# Source 3: Manual override CSV
# ---------------------------------------------------------------------------

def load_manual_override() -> list[dict]:
    """Load manually-entered primary results from data/raw/tx_primary_2026_manual.csv."""
    if not MANUAL_PATH.exists():
        return []

    results = []
    with open(MANUAL_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                results.append({
                    "chamber":       row["chamber"].strip().title(),
                    "district":      int(row["district"]),
                    "party":         row["party"].strip().upper(),
                    "winner_name":   row["winner_name"].strip(),
                    "winner_votes":  int(str(row.get("winner_votes", 0)).replace(",", "") or 0),
                    "runoff_needed": str(row.get("runoff_needed", "")).strip().lower() in ("true", "1", "yes"),
                    "source":        "manual",
                })
            except (KeyError, ValueError) as exc:
                print(f"  Manual CSV row error: {exc} — {row}")

    print(f"  Loaded {len(results)} rows from manual override CSV")
    return results


# ---------------------------------------------------------------------------
# Merge: deduplicate and combine sources
# ---------------------------------------------------------------------------

def merge_results(sources: list[list[dict]]) -> list[dict]:
    """
    Merge results from multiple sources. Priority: manual > civix > html.
    Deduplicates by (chamber, district, party).
    """
    # Source priority
    priority = {"manual": 0, "civix_api": 1, "sos_html": 2}

    best: dict[tuple, dict] = {}
    for source_rows in sources:
        for row in source_rows:
            key = (row["chamber"], row["district"], row["party"])
            existing = best.get(key)
            if existing is None:
                best[key] = row
            elif priority.get(row["source"], 99) < priority.get(existing["source"], 99):
                best[key] = row

    return sorted(best.values(), key=lambda r: (r["chamber"], r["district"], r["party"]))


# ---------------------------------------------------------------------------
# Write output CSV
# ---------------------------------------------------------------------------

def write_primary_csv(results: list[dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    d_count = sum(1 for r in results if r["party"] == "D")
    r_count = sum(1 for r in results if r["party"] == "R")
    runoffs  = sum(1 for r in results if r["runoff_needed"])
    print(f"\n  Wrote {len(results)} rows to {path.name}")
    print(f"    D primary winners: {d_count}, R primary winners: {r_count}")
    print(f"    Races with May runoff pending: {runoffs}")


# ---------------------------------------------------------------------------
# Update finance data with primary party info
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    name = name.lower()
    name = re.sub(r"[,\.\-']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Remove common suffixes
    for suffix in [" jr", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def _name_match(a: str, b: str) -> bool:
    """Fuzzy name match: check if any token sequence overlaps."""
    a_tokens = set(_normalize_name(a).split())
    b_tokens = set(_normalize_name(b).split())
    # Remove single-char tokens (initials)
    a_tokens = {t for t in a_tokens if len(t) > 1}
    b_tokens = {t for t in b_tokens if len(t) > 1}
    if not a_tokens or not b_tokens:
        return False
    shared = a_tokens & b_tokens
    # Match if last name (longest token) is in both
    longest_a = max(a_tokens, key=len)
    longest_b = max(b_tokens, key=len)
    return longest_a == longest_b or len(shared) >= 2


def update_finance_with_primary(primary_results: list[dict]):
    """
    Re-assign dem_fundraising_share in tx_finance_2026.csv using primary winners
    as authoritative party assignment for TEC filers.

    For each district with primary data, we now know:
      - D primary winner name → match to TEC filer → classify as 'dem_raised'
      - R primary winner name → match to TEC filer → classify as 'rep_raised'

    This replaces the incumbent-party approximation with actual primary results.
    """
    if not FINANCE_PATH.exists():
        print(f"  {FINANCE_PATH.name} not found — run collect_finance_2026.py first")
        return

    # Build primary lookup: (chamber, district) → {D: name, R: name}
    primary_by_dist: dict[tuple, dict] = {}
    for r in primary_results:
        if r.get("runoff_needed"):
            continue  # skip pending runoffs — party winner not yet determined
        key = (r["chamber"].lower(), r["district"])
        if key not in primary_by_dist:
            primary_by_dist[key] = {}
        primary_by_dist[key][r["party"]] = r["winner_name"]

    # Read finance CSV
    with open(FINANCE_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        print(f"  {FINANCE_PATH.name} is empty or malformed — skipping update")
        return

    updated_count = 0
    for row in rows:
        chamber  = row["chamber"].strip().lower()
        district = int(row["district"])
        key = (chamber, district)

        primary = primary_by_dist.get(key)
        if not primary:
            continue

        inc_raised   = float(row.get("incumbent_raised") or 0)
        chal_raised  = float(row.get("challenger_raised") or 0)
        inc_name_tec = row.get("incumbent_name_tec", "").strip()
        chal_names   = row.get("challenger_names", "")

        d_name = primary.get("D", "")
        r_name = primary.get("R", "")

        # Determine if we can reassign more accurately
        # Case 1: D name matches incumbent → incumbent is D
        # Case 2: R name matches incumbent → incumbent is R
        # Case 3: Open seat — both are challengers
        dem_raised = None
        rep_raised = None

        # Try to match primary winners to TEC incumbents/challengers
        inc_is_d = d_name and _name_match(inc_name_tec, d_name) if inc_name_tec else False
        inc_is_r = r_name and _name_match(inc_name_tec, r_name) if inc_name_tec else False

        if inc_is_d:
            dem_raised = inc_raised
            rep_raised = chal_raised
        elif inc_is_r:
            dem_raised = chal_raised
            rep_raised = inc_raised
        elif not inc_name_tec:
            # Open seat: check if D/R primary winner names appear in challenger_names
            chal_name_list = [n.strip() for n in chal_names.split(";") if n.strip()]
            d_in_challengers = any(_name_match(n, d_name) for n in chal_name_list) if d_name else False
            r_in_challengers = any(_name_match(n, r_name) for n in chal_name_list) if r_name else False

            if d_in_challengers or r_in_challengers:
                # Approximate: split challenger total between D and R by name matching
                # (imperfect but better than None)
                pass  # leave as None for open seats without clear match

        if dem_raised is not None and rep_raised is not None and (dem_raised + rep_raised) > 0:
            new_share = round(dem_raised / (dem_raised + rep_raised), 4)
            old_share = row.get("dem_fundraising_share")
            row["dem_fundraising_share"] = new_share
            row["party_assignment_method"] = "primary_results"
            updated_count += 1
            if old_share and abs(float(old_share or 0) - new_share) > 0.05:
                print(f"  {chamber.title()} {district}: share {old_share} -> {new_share:.4f} "
                      f"(D={d_name} / R={r_name})")

    # Write back
    with open(FINANCE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Updated {updated_count} districts in {FINANCE_PATH.name} with primary party data")


# ---------------------------------------------------------------------------
# Write template manual CSV if it doesn't exist
# ---------------------------------------------------------------------------

def write_manual_template():
    """Create a blank template for manual data entry."""
    if MANUAL_PATH.exists():
        print(f"  {MANUAL_PATH.name} already exists — not overwriting")
        return

    # Pre-fill with all district/party combinations needing data
    districts = load_districts_2026_basic()
    rows = []
    for chamber, district in sorted(districts):
        for party in ("D", "R"):
            rows.append({
                "chamber": chamber.title(),
                "district": district,
                "party": party,
                "winner_name": "",
                "winner_votes": "",
                "runoff_needed": "False",
            })

    with open(MANUAL_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chamber", "district", "party",
                                                "winner_name", "winner_votes", "runoff_needed"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Created manual template: {MANUAL_PATH.name} ({len(rows)} rows)")
    print(f"  Fill in winner names and re-run with --manual-only")


def load_districts_2026_basic() -> list[tuple]:
    """Return list of (chamber, district) tuples for all 2026 ballot districts."""
    result = []
    if DISTRICTS_PATH.exists():
        with open(DISTRICTS_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                up = str(row.get("up_in_2026", "")).strip().lower() in ("true", "1", "yes")
                if up:
                    result.append((row["chamber"].strip().lower(), int(row["district"])))
    else:
        # Fallback: hardcode 150 house + 16 senate districts
        for d in range(1, 151):
            result.append(("house", d))
        for d in SENATE_2026_BALLOT:
            result.append(("senate", d))
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    print(f"\n{'='*60}")
    print("  2026 PRIMARY RESULTS SUMMARY")
    print(f"{'='*60}")

    by_chamber = {}
    for r in results:
        ch = r["chamber"]
        by_chamber.setdefault(ch, {"D": 0, "R": 0, "runoffs": 0})
        by_chamber[ch][r["party"]] += 1
        if r["runoff_needed"]:
            by_chamber[ch]["runoffs"] += 1

    for chamber, stats in sorted(by_chamber.items()):
        print(f"\n  {chamber}:")
        print(f"    D primary winners: {stats['D']}")
        print(f"    R primary winners: {stats['R']}")
        print(f"    Pending May runoffs: {stats['runoffs']}")

    manual_count = sum(1 for r in results if r["source"] == "manual")
    auto_count   = len(results) - manual_count
    print(f"\n  Sources: {auto_count} auto-collected, {manual_count} manual")

    if len(results) < 200:
        print(f"\n  WARNING: Only {len(results)} results — expected ~332 (166 districts x 2 parties).")
        print(f"  Consider filling in {MANUAL_PATH.name} for missing districts.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect 2026 TX legislative primary results for party assignment")
    parser.add_argument("--manual-only", action="store_true",
                        help="Use manual override CSV only, skip API calls")
    parser.add_argument("--update-finance", action="store_true",
                        help="After collecting, update tx_finance_2026.csv with primary party data")
    parser.add_argument("--template", action="store_true",
                        help="Create blank manual entry template at data/raw/tx_primary_2026_manual.csv")
    args = parser.parse_args()

    print("=" * 60)
    print("  TX Legislature 2026 — Primary Results Collection")
    print("  Primary date: March 4, 2026 | Runoffs: May 27, 2026")
    print("=" * 60)

    if args.template:
        write_manual_template()
        return

    source_results = []

    if not args.manual_only:
        civix = try_civix()
        if civix:
            source_results.append(civix)

        if not civix:
            legacy = try_sos_legacy()
            if legacy:
                source_results.append(legacy)

    manual = load_manual_override()
    if manual:
        source_results.append(manual)

    results = merge_results(source_results)

    print_summary(results)

    if not results:
        print(f"\n  No results collected from any source.")
        print(f"  To enter data manually:")
        print(f"    1. Run: python src/collect_primary_2026.py --template")
        print(f"    2. Fill in {MANUAL_PATH.name}")
        print(f"    3. Run: python src/collect_primary_2026.py --manual-only")
        return

    write_primary_csv(results, OUTPUT_PATH)

    if args.update_finance:
        print(f"\nUpdating finance data with primary party assignments...")
        update_finance_with_primary(results)
        print("\nRe-run model:  python src/model.py")


if __name__ == "__main__":
    main()
