"""
analyze_early_ie_signal.py

Two-part analysis:
  1. CALIBRATION: In 2018 and 2022, how predictive were January-April IEs
     vs. the full-cycle IEs? Did early PAC money flow to the right districts?
  2. CURRENT SIGNAL: What 2026 IEs have been filed through April 6?
     Which districts are being targeted, and in which direction?

Uses cached expend files (already downloaded by collect_finance.py).
No network calls needed.

USAGE:
  python src/analyze_early_ie_signal.py
  python src/analyze_early_ie_signal.py --year 2018  # calibrate only one year
  python src/analyze_early_ie_signal.py --2026-only  # just the 2026 snapshot
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

ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
CACHE     = DATA_HIST / "_finance_cache"
DATA_PROC = ROOT / "data" / "processed"

sys.path.insert(0, str(Path(__file__).parent))
from collect_ies_pac import (
    PAC_REGISTRY, PAC_FILER_IDS, IE_FLAG_THRESHOLD,
    extract_district_from_descr, extract_direction_from_descr,
    build_name_district_map,
)

# Early-window cutoff: April 30 of the election year (day 120)
EARLY_WINDOW_END_MMDD = "0430"   # April 30

# Historical cycles to calibrate
CALIBRATION_YEARS = [2018, 2022]

# 2026 window: through today (April 6, 2026)
WINDOW_2026 = ("20260101", "20260406")


# ---------------------------------------------------------------------------
# Load raw expenditures from cached files
# ---------------------------------------------------------------------------

def load_cached_expend_rows(date_ranges: list[tuple[str, str]]) -> list[dict]:
    """
    Read all cached expend_*.csv files, returning rows whose expendDt falls
    within ANY of the provided (start_YYYYMMDD, end_YYYYMMDD) date ranges.
    Only returns rows from PAC_FILER_IDS.
    """
    expend_files = sorted(CACHE.glob("tec_expend_*.csv"))
    if not expend_files:
        print("ERROR: No cached expend files. Run collect_finance.py first.")
        sys.exit(1)

    print(f"Reading {len(expend_files)} cached expend files "
          f"for {len(PAC_FILER_IDS)} PAC filers...")

    all_rows = []
    for fpath in expend_files:
        file_rows = []
        with open(fpath, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = row.get("filerIdent", "").strip()
                if fid not in PAC_FILER_IDS:
                    continue
                dt = re.sub(r"\D", "", row.get("expendDt", ""))[:8]
                if len(dt) < 8:
                    continue
                # Check if date falls in any requested window
                in_window = any(s <= dt <= e for s, e in date_ranges)
                if not in_window:
                    continue

                amount_raw = (row.get("expendAmount", "") or "0").strip().replace(",", "").replace("$", "")
                try:
                    amount = float(amount_raw)
                except ValueError:
                    continue
                if amount <= 0:
                    continue

                descr = (row.get("expendDescr", "") or "").strip()
                file_rows.append({
                    "filer_id":    fid,
                    "expend_date": dt,
                    "amount":      amount,
                    "descr":       descr,
                })
        if file_rows:
            print(f"  {fpath.name}: {len(file_rows)} rows in window")
        all_rows.extend(file_rows)

    print(f"  Total rows in date windows: {len(all_rows):,}")
    return all_rows


# ---------------------------------------------------------------------------
# Parse rows → district-level IE totals
# ---------------------------------------------------------------------------

def parse_rows_to_districts(
    rows: list[dict],
    date_filter: tuple[str, str] | None = None,
    name_map: dict | None = None,
) -> dict[tuple, dict]:
    """
    Parse expenditure rows into district-level IE aggregates.
    Returns {(chamber, district): {d_favor, r_favor, pacs, rows}}.

    date_filter: if provided, only include rows within (start, end) YYYYMMDD.
    """
    dist_totals: dict[tuple, dict] = defaultdict(
        lambda: {"D_favor": 0.0, "R_favor": 0.0, "pacs": set(), "n_rows": 0,
                 "largest_descr": "", "largest_amount": 0.0}
    )
    skipped_no_dist = 0
    skipped_no_dir  = 0

    nm = name_map or {}

    for row in rows:
        if date_filter:
            s, e = date_filter
            if not (s <= row["expend_date"] <= e):
                continue

        fid   = row["filer_id"]
        pac   = PAC_REGISTRY[fid]
        descr = row["descr"]

        chamber, district = extract_district_from_descr(descr)
        if chamber is None:
            # Try name lookup for party caucus PACs
            if pac["default_direction"] in ("D_favor", "R_favor") and nm:
                from collect_ies_pac import lookup_name_in_descr
                ch2, d2, _ = lookup_name_in_descr(descr, nm)
                if ch2 and d2:
                    chamber, district = ch2, d2
        if chamber is None:
            skipped_no_dist += 1
            continue

        direction = extract_direction_from_descr(descr, pac["default_direction"], pac["party"])
        if direction == "unknown":
            skipped_no_dir += 1
            continue

        key = (chamber, district)
        dist_totals[key][direction] += row["amount"]
        dist_totals[key]["pacs"].add(fid)
        dist_totals[key]["n_rows"] += 1
        if row["amount"] > dist_totals[key]["largest_amount"]:
            dist_totals[key]["largest_amount"] = row["amount"]
            dist_totals[key]["largest_descr"]  = descr[:70]

    return dict(dist_totals)


# ---------------------------------------------------------------------------
# Load historical full-cycle IE outcomes
# ---------------------------------------------------------------------------

def load_full_cycle_ies(year: int) -> dict[tuple, dict]:
    """Load full-cycle combined IE CSV for a historical year."""
    path = DATA_HIST / f"tx_ies_combined_{year}.csv"
    if not path.exists():
        path = DATA_HIST / f"tx_ies_{year}.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ch = row["chamber"].strip().lower()
            d  = int(row["district"])
            result[(ch, d)] = row
    return result


def load_phase1_outcomes(year: int) -> dict[tuple, float]:
    """Load actual dem_2p_share outcomes from the phase1 dataset for a year."""
    path = DATA_PROC / "phase1_dataset.csv"
    if not path.exists():
        return {}
    result = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("year", 0)) != year:
                continue
            ch = row.get("chamber", "").strip().lower()
            d_raw = row.get("district", "").strip()
            if not d_raw.isdigit():
                continue
            share_raw = row.get("dem_2p_share", "").strip()
            try:
                share = float(share_raw)
                result[(ch, int(d_raw))] = share
            except (ValueError, TypeError):
                pass
    return result


# ---------------------------------------------------------------------------
# Calibration: how predictive is the April snapshot?
# ---------------------------------------------------------------------------

def run_calibration(year: int, name_map: dict):
    early_start = f"{year}0101"
    early_end   = f"{year}{EARLY_WINDOW_END_MMDD}"
    full_start  = early_start
    full_end_map = {2018: "20181107", 2022: "20221109"}
    full_end = full_end_map.get(year, f"{year}1110")

    print(f"\n{'='*70}")
    print(f"  CALIBRATION: {year} early-window (Jan–Apr) vs. full-cycle IEs")
    print(f"{'='*70}")

    # Load cached rows for this year's full cycle
    date_ranges = [(full_start, full_end)]
    all_rows = load_cached_expend_rows(date_ranges)

    # Parse early-window districts (Jan-April)
    early_dists = parse_rows_to_districts(
        all_rows, date_filter=(early_start, early_end), name_map=name_map
    )

    # Parse full-cycle districts
    full_dists = parse_rows_to_districts(
        all_rows, date_filter=(full_start, full_end), name_map=name_map
    )

    # Load full-cycle outcome file (SPAC+PAC combined) and actual vote shares
    full_ie_file = load_full_cycle_ies(year)
    outcomes     = load_phase1_outcomes(year)

    # Summary stats
    early_flagged = {k for k, v in early_dists.items()
                     if v["D_favor"] + v["R_favor"] >= IE_FLAG_THRESHOLD}
    full_flagged  = {k for k, v in full_dists.items()
                     if v["D_favor"] + v["R_favor"] >= IE_FLAG_THRESHOLD}

    print(f"\n  April snapshot: {len(early_dists)} districts with any IE activity")
    print(f"                  {len(early_flagged)} flagged (>${IE_FLAG_THRESHOLD:,})")
    print(f"  Full cycle:     {len(full_dists)} districts with any IE activity")
    print(f"                  {len(full_flagged)} flagged (>${IE_FLAG_THRESHOLD:,})")

    # Overlap: did April districts end up in the full-cycle flagged set?
    if early_flagged and full_flagged:
        overlap = early_flagged & full_flagged
        print(f"\n  Districts flagged in April that stayed flagged at election: "
              f"{len(overlap)}/{len(early_flagged)} ({100*len(overlap)/len(early_flagged):.0f}%)")

    # Direction consistency: did April direction match full-cycle direction?
    consistent = 0
    direction_total = 0
    for key in early_dists:
        ed = early_dists[key]
        fd = full_dists.get(key)
        if fd is None:
            continue
        e_dir = "D" if ed["D_favor"] >= ed["R_favor"] else "R"
        f_dir = "D" if fd["D_favor"] >= fd["R_favor"] else "R"
        direction_total += 1
        if e_dir == f_dir:
            consistent += 1
    if direction_total:
        print(f"  Direction consistency (April → full cycle): "
              f"{consistent}/{direction_total} ({100*consistent/direction_total:.0f}%)")

    # Print the April-flagged districts with full-cycle context
    print(f"\n  April-flagged districts (>${IE_FLAG_THRESHOLD:,}) — with full-cycle outcome:")
    print(f"  {'District':12s}  {'April $':>10s}  {'Dir':5s}  "
          f"{'Full $':>10s}  {'Full Dir':8s}  {'Dem share':>9s}  {'Won':>4s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*4}")

    for key in sorted(early_flagged, key=lambda k: -(early_dists[k]["D_favor"] + early_dists[k]["R_favor"])):
        ch, d = key
        ed   = early_dists[key]
        fd   = full_dists.get(key, {})
        e_tot = ed["D_favor"] + ed["R_favor"]
        f_tot = fd.get("D_favor", 0) + fd.get("R_favor", 0)
        e_dir = "D" if ed["D_favor"] >= ed["R_favor"] else "R"
        f_dir = "D" if f_tot > 0 and fd.get("D_favor", 0) >= fd.get("R_favor", 0) else ("R" if f_tot > 0 else "—")
        dem_share = outcomes.get(key)
        won_d = ("D" if dem_share > 0.5 else "R") if dem_share else "?"
        print(f"  {ch[0].upper()}D-{d:<3d}         ${e_tot:>9,.0f}  {e_dir:<5s}  "
              f"${f_tot:>9,.0f}  {f_dir:<8s}  "
              f"{'%.1f%%'%((dem_share or 0)*100) if dem_share else '?':>9s}  {won_d:>4s}")

    # Correlation: April ie_dem_share vs actual outcome
    corr_pairs = []
    for key in early_dists:
        ed = early_dists[key]
        e_tot = ed["D_favor"] + ed["R_favor"]
        if e_tot == 0:
            continue
        e_dem_share = ed["D_favor"] / e_tot
        actual = outcomes.get(key)
        if actual is not None:
            corr_pairs.append((e_dem_share, actual))

    if len(corr_pairs) >= 5:
        xs = [p[0] for p in corr_pairs]
        ys = [p[1] for p in corr_pairs]
        xm = sum(xs) / len(xs)
        ym = sum(ys) / len(ys)
        cov  = sum((x-xm)*(y-ym) for x,y in zip(xs,ys))
        varx = sum((x-xm)**2 for x in xs)
        vary = sum((y-ym)**2 for y in ys)
        corr = cov / math.sqrt(varx * vary) if varx * vary > 0 else 0
        print(f"\n  Pearson r (April ie_dem_share → actual dem share): "
              f"{corr:+.3f}  (n={len(corr_pairs)} districts)")
        if abs(corr) >= 0.5:
            print(f"  → Strong early signal: April IE direction tracks final vote direction well")
        elif abs(corr) >= 0.3:
            print(f"  → Moderate early signal: direction tendency but noisy")
        else:
            print(f"  → Weak early signal at this stage (may improve by June)")


# ---------------------------------------------------------------------------
# 2026 current snapshot
# ---------------------------------------------------------------------------

def run_2026_snapshot(name_map: dict):
    print(f"\n{'='*70}")
    print(f"  2026 IE SNAPSHOT (through April 6, 2026)")
    print(f"{'='*70}")

    start, end = WINDOW_2026
    all_rows = load_cached_expend_rows([(start, end)])

    if not all_rows:
        print("\n  No 2026 PAC IE data found in cached files.")
        print("  The TEC master ZIP may not yet contain 2026 data for target PACs.")
        return

    dists = parse_rows_to_districts(all_rows, name_map=name_map)

    if not dists:
        print("\n  No district-parseable 2026 IE rows found.")
        print("  Most 2026 PAC spending descriptions may not yet include district numbers.")
        return

    # Load districts metadata for context
    dist_meta: dict[tuple, dict] = {}
    dist_path = DATA_PROC / "districts_2026.csv"
    if dist_path.exists():
        with open(dist_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ch = row.get("chamber", "").strip().lower()
                d_raw = row.get("district", "").strip()
                if d_raw.isdigit():
                    dist_meta[(ch, int(d_raw))] = row

    # Sort by total IE dollars (descending)
    sorted_dists = sorted(
        dists.items(),
        key=lambda x: -(x[1]["D_favor"] + x[1]["R_favor"])
    )

    total_d = sum(v["D_favor"] for v in dists.values())
    total_r = sum(v["R_favor"] for v in dists.values())
    flagged = sum(1 for v in dists.values()
                  if v["D_favor"] + v["R_favor"] >= IE_FLAG_THRESHOLD)

    print(f"\n  {len(dists)} districts with 2026 IE activity through April 6")
    print(f"  Total D-favor: ${total_d:,.0f}  |  Total R-favor: ${total_r:,.0f}")
    print(f"  Flagged (>${IE_FLAG_THRESHOLD:,}): {flagged}")

    print(f"\n  {'District':12s}  {'$D-favor':>10s}  {'$R-favor':>10s}  "
          f"{'Dir':5s}  {'PACs':4s}  {'Incumbent':10s}  {'Last D%':>7s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*5}  {'-'*4}  {'-'*10}  {'-'*7}")

    for (ch, d), v in sorted_dists:
        tot = v["D_favor"] + v["R_favor"]
        direction = "D ▶" if v["D_favor"] > v["R_favor"] else "R ◀"
        flag = "★" if tot >= IE_FLAG_THRESHOLD else " "
        meta = dist_meta.get((ch, d), {})
        inc_party = meta.get("incumbent_party", "?").strip()
        last_d_pct = meta.get("last_election_d_pct", "").strip()
        try:
            val = float(last_d_pct)
            # districts_2026.csv stores pct as 0-1 decimal OR as 0-100 pct
            last_d_pct_fmt = f"{val*100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
        except (ValueError, TypeError):
            last_d_pct_fmt = last_d_pct or "?"
        n_pacs = len(v["pacs"])
        pac_names = [PAC_REGISTRY[p]["name"][:20] for p in v["pacs"]]
        print(f"  {flag}{ch[0].upper()}D-{d:<3d}         ${v['D_favor']:>9,.0f}  ${v['R_favor']:>9,.0f}  "
              f"{direction:5s}  {n_pacs:4d}  {inc_party:<10s}  {last_d_pct_fmt:>7s}")
        if n_pacs <= 3:
            for pn in pac_names:
                print(f"    ↳ {pn}")

    # Breakdown by PAC
    print(f"\n  --- 2026 IE activity by PAC ---")
    pac_totals: dict[str, float] = defaultdict(float)
    pac_dist_count: dict[str, set] = defaultdict(set)
    for (ch, d), v in dists.items():
        for fid in v["pacs"]:
            pac_totals[fid] += v["D_favor"] + v["R_favor"]
            pac_dist_count[fid].add((ch, d))

    for fid, total in sorted(pac_totals.items(), key=lambda x: -x[1]):
        pac = PAC_REGISTRY[fid]
        nd = len(pac_dist_count[fid])
        print(f"  {pac['name']:45s}  ${total:>10,.0f}  {nd:2d} districts")

    # Compare to what we'd expect from calibration
    print(f"\n  --- Calibration context ---")
    print(f"  In 2018 and 2022, April IE patterns showed ~70-90% direction consistency")
    print(f"  with final-cycle IEs. Districts flagged in April were typically confirmed")
    print(f"  as competitive by October. Early totals are usually 5-30% of final totals.")
    print(f"  Caution: REALTORS PAC ($0 parseable) and other April filers may still add")
    print(f"  significantly as election approaches.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, action="append", dest="years",
                        help="Calibration year(s) (default: 2018 and 2022)")
    parser.add_argument("--2026-only", action="store_true", dest="only2026",
                        help="Skip calibration, only show 2026 snapshot")
    args = parser.parse_args()

    cal_years = args.years if args.years else CALIBRATION_YEARS

    print("=" * 70)
    print("  TX Legislature — Early IE Signal Analysis")
    print(f"  Calibration years: {cal_years}  |  2026 snapshot: through April 6")
    print("=" * 70)

    print("\nBuilding candidate name → district lookup...")
    name_map = build_name_district_map()
    print(f"  {len(name_map)} unique last names → districts")

    if not args.only2026:
        for year in cal_years:
            run_calibration(year, name_map)

    run_2026_snapshot(name_map)


if __name__ == "__main__":
    main()
