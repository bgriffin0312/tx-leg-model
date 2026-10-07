"""
calibrate_viability_threshold.py

Calibrate the in-cycle challenger viability threshold for a given point in the
filing calendar, the way VIABILITY_THRESHOLD_SEMIJUL was set on 2026-07-20:

  Among candidates who ended the cycle VIABLE by the training definition
  (election-year raised, Jan 1 through election day, >= $100K House / $250K
  Senate), what had they raised by the time the cutoff report was filed?
  The median of that is the in-cycle threshold.

Same window the 2026 pipeline applies (election-year reports only), so the
numbers are measurement-consistent with collect_finance_2026.py.

Usage:
  python scripts/calibrate_viability_threshold.py                     # duplicates dropped
  python scripts/calibrate_viability_threshold.py --keep-duplicates   # as the training data was built
"""

import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
csv.field_size_limit(10**9)

ROOT = Path(__file__).parent.parent
COVER = ROOT / "data" / "raw" / "historical" / "_finance_cache" / "tec_cover.csv"

sys.path.insert(0, str(ROOT / "src"))
from collect_finance_2026 import latest_reports_only  # noqa: E402

OFFICE = {"STATEREP": "house", "STATESEN": "senate"}
FULL_CYCLE_VIABLE = {"house": 100_000, "senate": 250_000}

# Cycle: (start, election-day end, {era: filed-by cutoff})
# July semi-annual due Jul 15/16; 30-day-before-general due ~Oct 5-11.
# Cutoffs allow a few days of late filers, matching how the 2026 runs land.
CYCLES = {
    2014: ("20140101", "20141105", {"SEMIJUL": "20140720", "OCT30": "20141008"}),
    2018: ("20180101", "20181107", {"SEMIJUL": "20180720", "OCT30": "20181011"}),
    2022: ("20220101", "20221109", {"SEMIJUL": "20220720", "OCT30": "20221013"}),
}


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")[:8]


def main():
    # (yr, chamber, district, filer) -> {"full": $, era: $}, plus incumbency flag
    tot = defaultdict(lambda: defaultdict(float))
    inc = {}
    dedup = "--keep-duplicates" not in sys.argv
    with open(COVER, encoding="utf-8", errors="replace") as f:
        rows = csv.DictReader(f)
        if dedup:
            rows = latest_reports_only(rows)
        for row in rows:
            chamber = OFFICE.get(row.get("filerSeekOfficeCd", "").strip().upper())
            if not chamber:
                continue
            dist = digits(row.get("filerSeekOfficeDistrict", ""))
            if not dist:
                continue
            pe, filed = digits(row.get("periodEndDt")), digits(row.get("filedDt"))
            try:
                amt = float((row.get("totalContribAmount") or "0").replace(",", ""))
            except ValueError:
                continue
            if amt <= 0 or len(pe) < 8:
                continue
            for yr, (start, end, eras) in CYCLES.items():
                if not (start <= pe <= end):
                    continue
                key = (yr, chamber, int(dist), row.get("filerIdent", "").strip())
                tot[key]["full"] += amt
                for era, cutoff in eras.items():
                    if filed and filed <= cutoff:
                        tot[key][era] += amt
                hold = OFFICE.get(row.get("filerHoldOfficeCd", "").strip().upper())
                hold_d = digits(row.get("filerHoldOfficeDistrict", ""))
                if hold == chamber and hold_d and int(hold_d) == int(dist):
                    inc[key] = True
                break

    print("corrected-report duplicates:", "dropped (latest filing kept)" if dedup else "KEPT")
    print(f"{'yr':>5} {'chamber':<7} {'era':<8} {'group':<11} {'n':>4} "
          f"{'median':>10} {'p25':>10} {'share_of_full':>13}")
    for era in ("SEMIJUL", "OCT30"):
        for chamber in ("house", "senate"):
            for yr in CYCLES:
                for group in ("all", "challenger"):
                    vals, shares = [], []
                    for key, d in tot.items():
                        if key[0] != yr or key[1] != chamber:
                            continue
                        if d["full"] < FULL_CYCLE_VIABLE[chamber]:
                            continue
                        if group == "challenger" and inc.get(key):
                            continue
                        vals.append(d[era])
                        shares.append(d[era] / d["full"])
                    if not vals:
                        continue
                    q = statistics.quantiles(vals, n=4) if len(vals) > 1 else [vals[0]] * 3
                    print(f"{yr:>5} {chamber:<7} {era:<8} {group:<11} {len(vals):>4} "
                          f"{statistics.median(vals):>10,.0f} {q[0]:>10,.0f} "
                          f"{statistics.median(shares):>13.2f}")

    # Classification view: which in-cycle threshold best reproduces the
    # full-cycle flag the regression was trained on? Non-incumbents only (the
    # flag is about the challenger side), every filer with money counted.
    print("\nBest-classifying threshold (non-incumbents; errors = false flags + missed flags)")
    print(f"{'yr':>5} {'chamber':<7} {'era':<8} {'n':>4} {'viable':>6} "
          f"{'best_T':>9} {'errors':>6}  errors at candidate T")
    grid = {"house": range(40_000, 160_001, 5_000), "senate": range(100_000, 400_001, 10_000)}
    show = {"house": (60_000, 80_000, 90_000, 100_000), "senate": (135_000, 200_000, 250_000)}
    pooled = defaultdict(lambda: defaultdict(int))
    for era in ("SEMIJUL", "OCT30"):
        for chamber in ("house", "senate"):
            for yr in CYCLES:
                pts = [(d[era], d["full"] >= FULL_CYCLE_VIABLE[chamber])
                       for key, d in tot.items()
                       if key[0] == yr and key[1] == chamber and not inc.get(key)]
                errs = {T: sum((v >= T) != y for v, y in pts) for T in grid[chamber]}
                for T, e in errs.items():
                    pooled[(era, chamber)][T] += e
                best = min(errs, key=lambda T: (errs[T], T))
                at = "  ".join(f"${T // 1000}K:{sum((v >= T) != y for v, y in pts)}"
                               for T in show[chamber])
                print(f"{yr:>5} {chamber:<7} {era:<8} {len(pts):>4} "
                      f"{sum(y for _, y in pts):>6} {best:>9,} {errs[best]:>6}  {at}")
    print("\nPooled 2014+2018+2022 — threshold range tied at the minimum:")
    for (era, chamber), errs in pooled.items():
        lo = min(errs.values())
        tied = [T for T, e in errs.items() if e == lo]
        print(f"  {era:<8} {chamber:<7} min errors {lo:>3} at ${min(tied):,}–${max(tied):,}")


if __name__ == "__main__":
    main()
