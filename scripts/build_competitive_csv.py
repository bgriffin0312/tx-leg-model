"""
Build clean CSVs of competitive 2026 TX legislative races at the current
environment for upload to Flourish.

Output:
  output/competitive_house.csv
  output/competitive_senate.csv

Columns mirror src/build_table.py (the flip-prob HTML table):
  District | Current Party | R Candidate | D Candidate | Flip Prob | Rating

Flip prob = P(seat changes party from current holder).
Filter:   0.25 <= flip_prob <= 0.75 (the actually-competitive band).

Run after `python src/model.py` so model_2026_scenarios.csv is fresh.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_competitive_map import load_candidate_lookup
from model_config import GENERIC_BALLOT_TOPLINE_D_2P

OUTPUT = ROOT / "output"
# Matches the flip table's "Competitive" pill threshold: any race rated
# Likely holds or better. No upper bound — a high-flip-prob seat like SD 9
# (D-held, model puts R at ~84% to flip) is still very much a contested race.
MIN_FLIP = 0.20


def _env_label(env_dial: float) -> str:
    s = "D" if env_dial >= 0 else "R"
    return f"{s}+{abs(env_dial):.0f}"


def rating(flip_prob: float) -> str:
    if flip_prob >= 0.75: return "Likely flip"
    if flip_prob >= 0.55: return "Lean flip"
    if flip_prob >= 0.40: return "Competitive"
    if flip_prob >= 0.20: return "Likely holds"
    return "Safe"


def main() -> None:
    current_env = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)
    target_label = _env_label(current_env)

    scenarios = pd.read_csv(OUTPUT / "model_2026_scenarios.csv")
    if target_label in set(scenarios["scenario"]):
        df = scenarios[scenarios["scenario"] == target_label].copy()
    else:
        all_envs = scenarios["env_dial"].unique()
        closest = min(all_envs, key=lambda e: abs(e - current_env))
        df = scenarios[scenarios["env_dial"] == closest].copy()
        target_label = _env_label(closest)

    cand = load_candidate_lookup()

    rows_by_chamber: dict[str, list[dict]] = {"House": [], "Senate": []}
    for _, r in df.iterrows():
        wp_d = float(r["win_prob_d"])
        chamber = str(r["chamber"]).strip()
        district = int(r["district"])
        inc_party = str(r["incumbent_party"]).strip().upper()

        if inc_party == "R":
            flip_prob = wp_d
        elif inc_party == "D":
            flip_prob = 1 - wp_d
        else:
            continue  # no current holder → no flip prob

        if flip_prob < MIN_FLIP:
            continue

        info = cand.get((chamber, district), {})
        d_name = info.get("d") or "TBD"
        r_name = info.get("r") or "TBD"

        rows_by_chamber[chamber].append({
            "District":      f"{'HD' if chamber == 'House' else 'SD'} {district}",
            "Current Party": inc_party,
            "R Candidate":   r_name,
            "D Candidate":   d_name,
            "Flip Prob":     round(flip_prob * 100, 1),
            "Rating":        rating(flip_prob),
        })

    print(f"  Scenario: {target_label} (current env)")
    for chamber, rows in rows_by_chamber.items():
        rows.sort(key=lambda x: x["Flip Prob"], reverse=True)
        out_df = pd.DataFrame(rows)
        suffix = chamber.lower()
        out_path = OUTPUT / f"competitive_{suffix}.csv"
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {chamber}: {len(out_df)} rows  →  {out_path.relative_to(ROOT)}")
        if len(out_df):
            print()
            print(out_df.to_string(index=False))
            print()


if __name__ == "__main__":
    main()
