"""
build_phase1_dataset.py

Merge historical election results, governor-by-district data, and campaign
finance into a single analysis-ready dataset for the Phase 1 regression.

Join key: (year, chamber, district)

Data sources (all in data/raw/historical/):
  tx_{chamber}_results_{year}.csv       — from collect_historical_results.py
  tx_gov_by_{chamber}_dist_{year}.csv   — from collect_gov_by_district.py
  tx_finance_{year}.csv                 — from collect_finance.py

Incumbency:
  Derived automatically from sequential election results: the winner of
  district D in year Y is coded as the incumbent in year Y+4 (or Y+2 for
  the 2002-2006 Senate transition). After redistricting (2002, 2022), all
  seats are treated as open for incumbency purposes unless an explicit
  "(incumbent)" tag was captured in the election results.

  For override/manual incumbency, create:
    data/raw/historical/tx_incumbents_manual.csv
  with columns: year, chamber, district, incumbent_party
  Values: R, D, or OPEN

National environment (generic congressional ballot D-R margin, final):
  2002: R+4.6 → -4.6
  2006: D+7.9 → +7.9
  2010: R+6.8 → -6.8
  2014: R+5.7 → -5.7
  2018: D+8.6 → +8.6
  2022: R+2.8 → -2.8

Output: data/processed/phase1_dataset.csv
Schema:
  year, chamber, district
  dem_2p_share              — dependent variable
  baseline_partisanship     — governor dem 2p share (or MANUAL_NEEDED)
  national_env              — D-R generic ballot margin
  dem_incumbent             — 1/0
  rep_incumbent             — 1/0
  open_seat                 — 1/0
  dem_fundraising_share     — 0-1 (or null)
  log_challenger_fundraising — log(challenger_raised + 1) (or null)
  challenger_viability_flag  — 0/1 (or null)
  chamber_senate            — 1 if Senate, 0 if House
  uncontested               — 1 if only one party ran
  contested                 — 1 if both R and D ran
  on_ballot                 — 1 if district was on the ballot this year
  data_completeness         — full / no_gov / no_finance / no_gov_no_finance
  two_party_calc_note       — notes from election parse
  gov_2p_note               — notes from governor parse
  MANUAL_NEEDED             — True if any required field is unfilled

Usage:
  python src/build_phase1_dataset.py
"""

import csv
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_PROC = ROOT / "data" / "processed"
DATA_PROC.mkdir(parents=True, exist_ok=True)

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
CHAMBERS = ["house", "senate"]
MAX_DISTRICTS = {"house": 150, "senate": 31}

NATIONAL_ENV = {
    2002: -4.6,   # R+4.6
    2006:  7.9,   # D+7.9
    2010: -6.8,   # R+6.8
    2014: -5.7,   # R+5.7
    2018:  8.6,   # D+8.6
    2022: -2.8,   # R+2.8
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv_by_key(path: Path, key_fields: list[str]) -> dict:
    """Load a CSV, returning a dict keyed on tuple of key_fields values."""
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(
                row[f].strip().lower() if f in ("chamber",) else
                int(row[f]) if f in ("year", "district") else
                row[f].strip()
                for f in key_fields
            )
            result[key] = row
    return result


def load_election_results() -> dict:
    """
    Load all election result CSVs.
    Returns dict keyed (year, chamber_lower, district_int).
    """
    data = {}
    for year in YEARS:
        for chamber in CHAMBERS:
            path = DATA_HIST / f"tx_{chamber}_results_{year}.csv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    district = int(row["district"])
                    key = (year, chamber, district)
                    data[key] = row
    return data


def load_governor_data() -> dict:
    """
    Load all governor-by-district CSVs.
    Returns dict keyed (year, chamber_lower, district_int).
    """
    data = {}
    for year in YEARS:
        for chamber in CHAMBERS:
            path = DATA_HIST / f"tx_gov_by_{chamber}_dist_{year}.csv"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    district = int(row["district"])
                    key = (year, chamber, district)
                    data[key] = row
    return data


def load_presidential_data() -> dict:
    """
    Load 2024 presidential results by district (static baseline for all years).
    Returns dict keyed (chamber_lower, district_int) → dem_pres_2p_baseline float.
    """
    data = {}
    for chamber in CHAMBERS:
        path = DATA_RAW / f"tx_presidential_{chamber}_2024.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                district = int(row["district"])
                val = safe_float(row.get("dem_pres_2p_baseline"))
                if val is not None:
                    data[(chamber, district)] = val
    return data


def load_finance_data() -> dict:
    """
    Load all finance CSVs.
    Returns dict keyed (year, chamber_lower, district_int).
    """
    data = {}
    for year in YEARS:
        path = DATA_HIST / f"tx_finance_{year}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chamber = row["chamber"].strip().lower()
                district = int(row["district"])
                key = (int(row["year"]), chamber, district)
                data[key] = row
    return data


def load_ie_data(source: str = "combined") -> dict:
    """
    Load historical IE CSVs. Returns dict keyed (year, chamber_lower, district_int).

    source options:
      "combined"  — prefer tx_ies_combined_{year}.csv (PAC + SPAC merged);
                    fall back to tx_ies_{year}.csv if combined not found
      "pac"       — use tx_ies_pac_{year}.csv only (PAC expenditure parsing)
      "spac"      — use tx_ies_{year}.csv only (original SPAC-based)
    """
    data = {}
    for year in YEARS:
        if source == "combined":
            primary   = DATA_HIST / f"tx_ies_combined_{year}.csv"
            fallback  = DATA_HIST / f"tx_ies_{year}.csv"
            path = primary if primary.exists() else fallback
        elif source == "pac":
            path = DATA_HIST / f"tx_ies_pac_{year}.csv"
        else:
            path = DATA_HIST / f"tx_ies_{year}.csv"

        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chamber = row["chamber"].strip().lower()
                district = int(row["district"])
                key = (year, chamber, district)
                data[key] = row
    n_years = len({k[0] for k in data})
    print(f"  {len(data)} IE district-year records loaded across {n_years} cycles "
          f"(source={source})")
    return data


def load_manual_incumbents() -> dict:
    """
    Load optional manual incumbency overrides.
    CSV: year, chamber, district, incumbent_party
    Returns dict keyed (year, chamber_lower, district_int) → party str.
    """
    path = DATA_RAW / "historical" / "tx_incumbents_manual.csv"
    if not path.exists():
        return {}
    data = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["year"]), row["chamber"].strip().lower(), int(row["district"]))
            data[key] = row["incumbent_party"].strip().upper()
    return data


# ---------------------------------------------------------------------------
# Incumbency derivation
# ---------------------------------------------------------------------------

def build_prior_winners(election_data: dict) -> dict:
    """
    Build a lookup: (next_year, chamber, district) → winner_party of previous cycle.
    Used to infer incumbency when the wikitext didn't flag it.

    The "prior cycle" for incumbency:
      2006 uses 2002 winners
      2010 uses 2006 winners
      2014 uses 2010 winners
      2018 uses 2014 winners
      2022 uses 2018 winners
    2002 is treated as post-redistricting (no strong prior; mark as unknown)
    """
    prior_map = {2006: 2002, 2010: 2006, 2014: 2010, 2018: 2014, 2022: 2018}
    lookup = {}
    for curr_year, prev_year in prior_map.items():
        for chamber in CHAMBERS:
            for district in range(1, MAX_DISTRICTS[chamber] + 1):
                prev_key = (prev_year, chamber, district)
                prev = election_data.get(prev_key)
                if prev and prev.get("winner_party") in ("R", "D"):
                    lookup[(curr_year, chamber, district)] = prev["winner_party"]
    return lookup


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def parse_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def safe_float(val) -> float | None:
    try:
        return float(val) if val not in (None, "", "None", "nan") else None
    except (ValueError, TypeError):
        return None


def build_row(year: int, chamber: str, district: int,
              election: dict | None,
              governor: dict | None,
              finance: dict | None,
              prior_winner: str | None,
              manual_inc: str | None,
              dem_pres_2p_baseline: float | None = None,
              ie: dict | None = None) -> dict:
    """
    Construct one analysis-ready row.
    Returns a dict with all output columns.
    """
    row = {
        "year": year,
        "chamber": chamber.title(),
        "district": district,
        "dem_2p_share": None,
        "baseline_partisanship": None,
        "dem_pres_2p_baseline": dem_pres_2p_baseline,
        "national_env": NATIONAL_ENV.get(year, 0.0),
        "dem_incumbent": 0,
        "rep_incumbent": 0,
        "open_seat": 0,
        "dem_fundraising_share": None,
        "log_challenger_fundraising": None,
        "challenger_viability_flag": None,
        # IE columns (from collect_ies_historical.py; None if no data for this year)
        "ie_dem_share": None,    # D-favoring IEs / total IEs (0.5 if none)
        "ie_log_total": None,    # log(total_ie_dollars + 1)
        "ie_flag": None,         # 1 if total IEs >= $25k
        "ie_total": None,        # raw total IE dollars
        "ie_d_favor": None,
        "ie_r_favor": None,
        "chamber_senate": 1 if chamber == "senate" else 0,
        "uncontested": 0,
        "contested": 0,
        "on_ballot": 0,
        "data_completeness": "no_data",
        "two_party_calc_note": "",
        "gov_2p_note": "",
        "MANUAL_NEEDED": False,
    }

    # ---- Election results ----
    if election is None:
        row["MANUAL_NEEDED"] = True
        return row

    row["on_ballot"] = 1 if parse_bool(election.get("on_ballot", False)) else 0
    if not row["on_ballot"]:
        row["two_party_calc_note"] = "not_on_ballot"
        return row

    contested = parse_bool(election.get("contested", False))
    row["contested"] = 1 if contested else 0
    row["two_party_calc_note"] = election.get("notes", "") or election.get("two_party_calc_note", "")

    dem_2p = safe_float(election.get("dem_2p_share"))
    row["dem_2p_share"] = dem_2p

    uncontested_notes = ("uncontested_r", "uncontested_d")
    row["uncontested"] = 1 if (
        row["two_party_calc_note"].lower() in uncontested_notes
        or (dem_2p is not None and dem_2p in (0.0, 1.0) and not contested)
    ) else 0

    # ---- Incumbency ----
    # Priority: manual override > wikitext flag > prior winner inference
    r_inc_wiki = parse_bool(election.get("r_incumbent", False))
    d_inc_wiki = parse_bool(election.get("d_incumbent", False))

    if manual_inc:
        if manual_inc == "D":
            row["dem_incumbent"] = 1
        elif manual_inc == "R":
            row["rep_incumbent"] = 1
        elif manual_inc == "OPEN":
            row["open_seat"] = 1
    elif r_inc_wiki or d_inc_wiki:
        row["dem_incumbent"] = 1 if d_inc_wiki else 0
        row["rep_incumbent"] = 1 if r_inc_wiki else 0
    elif prior_winner and year not in (2002, 2022):
        # Use prior cycle winner as incumbent (redistricting years have no valid prior)
        if prior_winner == "D":
            row["dem_incumbent"] = 1
        elif prior_winner == "R":
            row["rep_incumbent"] = 1
    else:
        # 2002 / 2022 redistricting years or no prior data: treat as open
        row["open_seat"] = 1

    # If neither incumbent flag is set and open_seat wasn't explicitly set, mark open
    if not (row["dem_incumbent"] or row["rep_incumbent"] or row["open_seat"]):
        row["open_seat"] = 1

    # ---- Governor (baseline partisanship) ----
    gov_needs_manual = True
    if governor is not None:
        gov_manual = parse_bool(governor.get("MANUAL_NEEDED", True))
        gov_dem_2p = safe_float(governor.get("gov_dem_2p_share"))
        gov_source = governor.get("gov_source", "")
        gov_note = governor.get("gov_2p_note", "")
        row["gov_2p_note"] = gov_note

        if not gov_manual and gov_dem_2p is not None:
            row["baseline_partisanship"] = gov_dem_2p
            gov_needs_manual = False
        elif gov_source == "statewide_fallback" and gov_dem_2p is not None:
            # Accept statewide fallback with a note
            row["baseline_partisanship"] = gov_dem_2p
            row["gov_2p_note"] = (gov_note + "|statewide").strip("|")
            gov_needs_manual = False  # usable, just imprecise

    if gov_needs_manual:
        row["MANUAL_NEEDED"] = True

    # ---- Finance ----
    fin_needs_manual = True
    if finance is not None:
        fin_manual = parse_bool(finance.get("MANUAL_NEEDED", True))
        if not fin_manual:
            row["dem_fundraising_share"] = safe_float(finance.get("dem_fundraising_share"))
            row["log_challenger_fundraising"] = safe_float(finance.get("log_challenger_fundraising"))
            cf = finance.get("challenger_viability_flag")
            row["challenger_viability_flag"] = int(cf) if cf not in (None, "", "None") else None
            fin_needs_manual = False

    if fin_needs_manual:
        row["MANUAL_NEEDED"] = True

    # ---- Data completeness ----
    has_gov = row["baseline_partisanship"] is not None
    has_fin = row["dem_fundraising_share"] is not None
    if has_gov and has_fin:
        row["data_completeness"] = "full"
    elif has_gov:
        row["data_completeness"] = "no_finance"
    elif has_fin:
        row["data_completeness"] = "no_gov"
    else:
        row["data_completeness"] = "no_gov_no_finance"

    # ---- IE data ----
    # Districts with no IE data are left as None (not zero) so the regression
    # can correctly distinguish "no IEs" from "IEs = 0" and handle missing data.
    # For districts with IEs, populate all four IE columns.
    if ie is not None:
        row["ie_dem_share"] = safe_float(ie.get("ie_dem_share"))
        row["ie_log_total"] = safe_float(ie.get("ie_log_total"))
        ie_flag_raw = ie.get("ie_flag")
        row["ie_flag"] = int(ie_flag_raw) if ie_flag_raw not in (None, "", "None") else None
        row["ie_total"]  = safe_float(ie.get("ie_total"))
        row["ie_d_favor"] = safe_float(ie.get("ie_d_favor"))
        row["ie_r_favor"] = safe_float(ie.get("ie_r_favor"))

    return row


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def build_dataset() -> list[dict]:
    print("Loading election results...")
    election_data = load_election_results()
    print(f"  {len(election_data)} district-year records loaded")

    print("Loading governor data...")
    governor_data = load_governor_data()
    print(f"  {len(governor_data)} governor records loaded")

    print("Loading finance data...")
    finance_data = load_finance_data()
    print(f"  {len(finance_data)} finance records loaded")

    print("Loading IE data (independent expenditures)...")
    ie_data = load_ie_data(source="combined")

    print("Loading presidential data (2024 static baseline)...")
    presidential_data = load_presidential_data()
    print(f"  {len(presidential_data)} presidential records loaded")

    print("Loading manual incumbency overrides...")
    manual_inc = load_manual_incumbents()
    print(f"  {len(manual_inc)} manual override entries")

    print("Building prior winner lookup for incumbency inference...")
    prior_winners = build_prior_winners(election_data)
    print(f"  {len(prior_winners)} prior winner entries")

    rows = []
    for year in YEARS:
        for chamber in CHAMBERS:
            for district in range(1, MAX_DISTRICTS[chamber] + 1):
                key = (year, chamber, district)
                election = election_data.get(key)
                governor = governor_data.get(key)
                finance = finance_data.get(key)
                ie = ie_data.get(key)
                prior_winner = prior_winners.get(key)
                manual = manual_inc.get(key)
                pres_baseline = presidential_data.get((chamber, district))

                row = build_row(year, chamber, district,
                                election, governor, finance,
                                prior_winner, manual, pres_baseline, ie)
                rows.append(row)

    return rows


def write_csv(rows: list[dict], path: Path):
    fields = [
        "year", "chamber", "district",
        "dem_2p_share", "baseline_partisanship", "dem_pres_2p_baseline", "national_env",
        "dem_incumbent", "rep_incumbent", "open_seat",
        "dem_fundraising_share", "log_challenger_fundraising", "challenger_viability_flag",
        "ie_dem_share", "ie_log_total", "ie_flag", "ie_total", "ie_d_favor", "ie_r_favor",
        "chamber_senate", "uncontested", "contested", "on_ballot",
        "data_completeness", "two_party_calc_note", "gov_2p_note", "MANUAL_NEEDED",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {path.name}")


def print_summary(rows: list[dict]):
    on_ballot = [r for r in rows if r["on_ballot"]]
    contested = [r for r in rows if r["contested"]]
    print(f"\n{'='*60}")
    print("PHASE 1 DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows:     {len(rows)}")
    print(f"On ballot:      {len(on_ballot)}")
    print(f"Contested:      {len(contested)}")

    print("\nContested races by year:")
    for year in YEARS:
        yr_rows = [r for r in contested if r["year"] == year]
        full = sum(1 for r in yr_rows if r["data_completeness"] == "full")
        no_fin = sum(1 for r in yr_rows if r["data_completeness"] == "no_finance")
        no_gov = sum(1 for r in yr_rows if r["data_completeness"] == "no_gov")
        no_both = sum(1 for r in yr_rows if r["data_completeness"] == "no_gov_no_finance")
        print(f"  {year}: {len(yr_rows):3d} contested  "
              f"[full={full}, no_finance={no_fin}, no_gov={no_gov}, neither={no_both}]")

    print("\nData completeness (contested races only):")
    for level in ("full", "no_finance", "no_gov", "no_gov_no_finance"):
        count = sum(1 for r in contested if r["data_completeness"] == level)
        pct = count / len(contested) * 100 if contested else 0
        label = {
            "full": "Full data",
            "no_finance": "Missing finance",
            "no_gov": "Missing governor",
            "no_gov_no_finance": "Missing both",
        }[level]
        print(f"  {label:25s}: {count:4d}  ({pct:.1f}%)")

    print("\nIncumbency coding (contested races):")
    dem_inc = sum(1 for r in contested if r["dem_incumbent"])
    rep_inc = sum(1 for r in contested if r["rep_incumbent"])
    open_s = sum(1 for r in contested if r["open_seat"])
    unknown = sum(1 for r in contested if not (r["dem_incumbent"] or r["rep_incumbent"] or r["open_seat"]))
    print(f"  D incumbent: {dem_inc}")
    print(f"  R incumbent: {rep_inc}")
    print(f"  Open seat:   {open_s}")
    print(f"  Unknown:     {unknown}")

    print("\nRegression model eligibility:")
    restricted_eligible = [r for r in contested
                           if r["uncontested"] == 0 and r["dem_2p_share"] is not None]
    full_eligible = [r for r in restricted_eligible
                     if r["baseline_partisanship"] is not None
                     and r["dem_fundraising_share"] is not None]
    print(f"  Restricted model (env + incumbency only): {len(restricted_eligible)}")
    print(f"  Full model (+ baseline + finance):         {len(full_eligible)}")

    if len(restricted_eligible) < 200:
        print("\n  WARNING: <200 observations for regression. Check that")
        print("  collect_historical_results.py ran successfully for all years.")

    if full_eligible:
        # Show distribution of dem_2p_share
        shares = [r["dem_2p_share"] for r in full_eligible if r["dem_2p_share"] is not None]
        if shares:
            avg = sum(shares) / len(shares)
            print(f"\n  Dem 2-party share (full model sample): "
                  f"mean={avg:.3f}, n={len(shares)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = build_dataset()
    out = DATA_PROC / "phase1_dataset.csv"
    write_csv(rows, out)
    print_summary(rows)
    print(f"\nNext step: python src/run_phase1_regression.py")
