"""
Build a clean CSV of competitive 2026 TX legislative races at the current
environment (D win prob in [25%, 75%]) for upload to Flourish.

Output: output/competitive_races.csv

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
LOW, HIGH = 0.25, 0.75


def _env_label(env_dial: float) -> str:
    s = "D" if env_dial >= 0 else "R"
    return f"{s}+{abs(env_dial):.0f}"


def main() -> None:
    current_env = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)
    target_label = _env_label(current_env)

    scenarios = pd.read_csv(OUTPUT / "model_2026_scenarios.csv")
    # Pick the scenario whose label matches current env; fall back to closest env_dial
    if target_label in set(scenarios["scenario"]):
        df = scenarios[scenarios["scenario"] == target_label].copy()
    else:
        all_envs = scenarios["env_dial"].unique()
        closest = min(all_envs, key=lambda e: abs(e - current_env))
        df = scenarios[scenarios["env_dial"] == closest].copy()
        target_label = _env_label(closest)

    cand = load_candidate_lookup()

    # Pull open_seat flag from districts_2026.csv (not in scenarios CSV)
    dists = pd.read_csv(ROOT / "data" / "processed" / "districts_2026.csv")
    open_lookup = {
        (str(r["chamber"]).strip(), int(r["district"])):
            str(r.get("open_seat", "")).strip().lower() in ("true", "1", "yes")
        for _, r in dists.iterrows()
    }

    rows = []
    for _, r in df.iterrows():
        wp = float(r["win_prob_d"])
        if not (LOW <= wp <= HIGH):
            continue
        chamber = str(r["chamber"]).strip()
        district = int(r["district"])
        info = cand.get((chamber, district), {})
        d_name = info.get("d") or "TBD"
        r_name = info.get("r") or "TBD"
        inc_party = str(r["incumbent_party"]).strip().upper()
        is_open = open_lookup.get((chamber, district), False)
        if is_open:
            status = "Open seat"
        elif inc_party == "D":
            status = "D incumbent"
        elif inc_party == "R":
            status = "R incumbent"
        else:
            status = "Open seat"
        pres = float(r["dem_pres_2p_baseline"]) * 100 if pd.notna(r["dem_pres_2p_baseline"]) else None
        rows.append({
            "District":      f"{'HD' if chamber == 'House' else 'SD'} {district}",
            "Status":        status,
            "D candidate":   d_name,
            "R candidate":   r_name,
            "Harris 2024 %": round(pres, 1) if pres is not None else "",
            "D win prob %":  round(wp * 100, 1),
        })

    rows.sort(key=lambda x: x["D win prob %"], reverse=True)
    out_df = pd.DataFrame(rows)
    out_path = OUTPUT / "competitive_races.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Scenario: {target_label} (current env)")
    print(f"  Competitive rows: {len(out_df)}")
    print(f"  Wrote {out_path.relative_to(ROOT)}")
    print()
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
