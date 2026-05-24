"""
Build a Flourish-ready CSV of the top-10 WAR candidates who are currently
running in a 2026 Texas legislative race.

Matches candidates_2026.csv (R/D nominees + unopposed) against historical
candidate_war.csv by normalized name + party, then ranks by weighted_avg_war
(time-decayed average WAR per cycle, the methodology preferred since May 17).

Output: output/war_top10_2026.csv

Columns:
  Candidate | Party | 2026 Race | Role | Weighted Avg WAR | Cycles

Run after `python src/compute_war.py` so candidate_war.csv is fresh.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compute_war import normalize_name  # same logic the WAR table uses

DATA   = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"

TOP_N = 10
# Candidate statuses that mean "this is the party's 2026 nominee (or running uncontested)".
# Excludes "runoff" (winner TBD) and "none_filed".
NOMINEE_STATUSES = {"won_primary", "nominee", "unopposed", "incumbent"}


def main() -> None:
    cand_2026 = pd.read_csv(DATA / "candidates_2026.csv")
    war       = pd.read_csv(DATA / "candidate_war.csv")

    # Build {(norm_name, party): {chamber, district, role}} from candidates_2026.csv
    in_race: dict[tuple[str, str], dict] = {}
    for _, row in cand_2026.iterrows():
        chamber  = str(row["chamber"]).strip()
        district = int(row["district"])
        for party_letter, name_col, status_col in (
            ("R", "r_candidate", "r_status"),
            ("D", "d_candidate", "d_status"),
        ):
            name   = str(row.get(name_col) or "").strip()
            status = str(row.get(status_col) or "").strip().lower()
            if not name or status not in NOMINEE_STATUSES:
                continue
            key = (normalize_name(name), party_letter)
            if not key[0]:
                continue
            in_race[key] = {
                "chamber":  chamber,
                "district": district,
                "role":     "Incumbent" if status == "incumbent" else "Challenger",
                "display":  name,
            }

    # Inner-join against historical WAR
    matched = []
    for _, w in war.iterrows():
        key = (str(w["candidate_norm"]).strip(), str(w["party"]).strip().upper())
        race = in_race.get(key)
        if race is None:
            continue
        if pd.isna(w["weighted_avg_war"]):
            continue
        matched.append({
            "Candidate":        race["display"],
            "Party":            key[1],
            "2026 Race":        f"{'HD' if race['chamber'] == 'House' else 'SD'} {race['district']}",
            "Role":             race["role"],
            "Weighted Avg WAR": round(float(w["weighted_avg_war"]), 2),
            "Cycles":           int(w["n_races"]),
        })

    matched.sort(key=lambda r: r["Weighted Avg WAR"], reverse=True)
    top = matched[:TOP_N]
    out_df = pd.DataFrame(top)

    out_path = OUTPUT / "war_top10_2026.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Matched 2026 nominees with WAR history: {len(matched)}")
    print(f"  Wrote top {len(top)}  →  {out_path.relative_to(ROOT)}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
