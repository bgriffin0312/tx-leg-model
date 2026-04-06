"""
Build the master district-level data table for the 2026 TX legislative model.

Merges:
  - Current incumbents (89th Legislature)
  - Most recent general election results for each seat:
      House: 2024 results (all 150 seats contested in 2024)
      Senate (2026 seats, districts 1/2/3/4/5/9/11/13/18/19/21/22/24/26/28/31):
             2022 results (last time these were on the ballot)
      Senate (2028 seats): 2024 results
  - 2026 race flags

Output:
  data/processed/districts_2026.csv
"""

import csv
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROC = Path(__file__).parent.parent / "data" / "processed"

# Senate districts up in 2026
SENATE_2026 = {1, 2, 3, 4, 5, 9, 11, 13, 18, 19, 21, 22, 24, 26, 28, 31}


def load_csv(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as f:
        return {int(r["district"]): r for r in csv.DictReader(f)}


def result_row(res: dict, incumbent_name: str) -> dict:
    """Extract result fields, backfilling blank candidate names from incumbent."""
    r_cand = res.get("r_candidate", "").strip()
    d_cand = res.get("d_candidate", "").strip()
    winner = res.get("winner_party", "")
    # Backfill incumbent name into winner's slot if blank
    if not r_cand and winner == "R":
        r_cand = incumbent_name
    if not d_cand and winner == "D":
        d_cand = incumbent_name
    return {
        "last_election_r_candidate": r_cand,
        "last_election_r_pct": res.get("r_pct", ""),
        "last_election_d_candidate": d_cand,
        "last_election_d_pct": res.get("d_pct", ""),
        "last_election_contested": res.get("contested", ""),
        "last_election_winner_party": winner,
        "last_election_notes": res.get("notes", ""),
    }


def build_house() -> list[dict]:
    members = load_csv(DATA_RAW / "tx_house_members_89th.csv")
    results = load_csv(DATA_RAW / "tx_house_results_2024.csv")
    rows = []
    for district in range(1, 151):
        m = members.get(district, {})
        r = results.get(district, {})
        res = result_row(r, m.get("member", ""))
        rows.append({
            "chamber": "House",
            "district": district,
            "incumbent": m.get("member", ""),
            "incumbent_party": m.get("party", ""),
            "last_election_year": 2024,
            **res,
            "up_in_2026": True,
            "open_seat": "",
            "notes_2026": "",
        })
    return rows


def build_senate() -> list[dict]:
    members = load_csv(DATA_RAW / "tx_senate_members_89th.csv")
    results_2022 = load_csv(DATA_RAW / "tx_senate_2022_results.csv") \
        if (DATA_RAW / "tx_senate_2022_results.csv").exists() else {}
    results_2024 = load_csv(DATA_RAW / "tx_senate_results_2024.csv")
    rows = []
    for district in range(1, 32):
        m = members.get(district, {})
        up = district in SENATE_2026
        # Use 2022 results for seats up in 2026, 2024 results for 2028 seats
        if up:
            r = results_2022.get(district, {})
            last_yr = 2022
        else:
            r = results_2024.get(district, {})
            last_yr = 2024
        res = result_row(r, m.get("member", ""))
        rows.append({
            "chamber": "Senate",
            "district": district,
            "incumbent": m.get("member", ""),
            "incumbent_party": m.get("party", ""),
            "last_election_year": last_yr,
            **res,
            "up_in_2026": up,
            "open_seat": "True" if m.get("member", "") in ("Vacant", "") else "",
            "notes_2026": m.get("notes", ""),
        })
    return rows


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path.name}")


if __name__ == "__main__":
    house = build_house()
    senate = build_senate()
    all_rows = house + senate

    out = DATA_PROC / "districts_2026.csv"
    write_csv(all_rows, out)

    # Summaries
    h_cont = sum(1 for r in house if r["last_election_contested"] == "True")
    h_blank = sum(1 for r in house if r["up_in_2026"] and not r["last_election_r_pct"] and not r["last_election_d_pct"])
    s_up = [r for r in senate if r["up_in_2026"]]
    s_blank = sum(1 for r in s_up if not r["last_election_r_pct"] and not r["last_election_d_pct"])
    print(f"\nHouse: 150 seats | {h_cont} contested in 2024 | {h_blank} missing result data")
    print(f"Senate: {len(s_up)} seats up in 2026 | {s_blank} missing 2022 result data")

    print("\nSample — competitive House seats:")
    competitive = sorted(
        [r for r in house if r["last_election_contested"] == "True"
         and r["last_election_r_pct"] and r["last_election_d_pct"]
         and abs(float(r["last_election_r_pct"]) - float(r["last_election_d_pct"])) < 10],
        key=lambda x: abs(float(x["last_election_r_pct"]) - float(x["last_election_d_pct"]))
    )
    for r in competitive[:8]:
        margin = float(r["last_election_r_pct"]) - float(r["last_election_d_pct"])
        winner = "R" if margin > 0 else "D"
        print(f"  HD{r['district']:3d}: {r['incumbent']:30s} ({r['incumbent_party']}) "
              f"R {r['last_election_r_pct']}% vs D {r['last_election_d_pct']}% "
              f"[{winner}+{abs(margin):.1f}]")
