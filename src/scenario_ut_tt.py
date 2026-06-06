"""
scenario_ut_tt.py

What-if: run the model on the straight UT/Texas Politics Project (UT/TT) poll
instead of the national racial-crosstab aggregate, with the TX Hispanic
adjustment turned OFF (because UT/TT already measures Texas Hispanics directly,
so the -0.05 national->TX correction would double-count).

Source: UT/Texas Politics Project April 2026 poll (fielded Apr 10-20, 2026;
1,200 RV). Generic U.S. Congress ballot (Q25A):
  Topline   R 43 / D 41                 -> D 2p 0.4881  (R+2.4 two-party)
  White     R 53 / D 33                 -> D 2p 0.3837
  Black     R  9 / D 73                 -> D 2p 0.8902
  Hispanic  R 38 / D 44                 -> D 2p 0.5366
  Asian     R 35 / D 31  | Other R42/D32 -> blended "other" ~0.455

Three runs, to separate the two channels by which UT/TT differs from the
national aggregate (demographic STRUCTURE vs topline LEVEL):

  1. PRODUCTION         national crosstabs + Hisp adj -0.05 + env D+6.4
                        (reproduces the committed headline as a sanity check)
  2. UT/TT structure    UT/TT crosstabs + NO Hisp adj + env held at D+6.4
                        (isolates the effect of UT/TT's redder Hispanic mix)
  3. STRAIGHT UT/TT     UT/TT crosstabs + NO Hisp adj + env = UT/TT topline R+2.4
                        (believe UT/TT wholesale: structure AND level)

This script does NOT modify model_config.py. It patches the in-memory
TX_HISPANIC_ADJUSTMENT global per run and passes crosstabs/env explicitly.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import model  # so we can patch model.TX_HISPANIC_ADJUSTMENT
from model import (
    load_districts,
    run_monte_carlo,
    _SENATE_D_HOLDOVER,
    HOUSE_MAJORITY,
    N_SIMULATIONS,
    CURRENT_ENV,
)
from model_config import RACE_GENERIC_BALLOT_D_SHARE

# --- UT/TT April 2026 racial crosstabs (D two-party shares) ---
UT_TT = {
    "white_nh": 33 / 86,   # 0.3837
    "black_nh": 73 / 82,   # 0.8902
    "hispanic": 44 / 82,   # 0.5366
    "other":    0.455,     # Asian (31/66=.470) / Other (32/74=.432) blend
}
# UT/TT topline two-party margin: 41/(41+43) = 0.4881 -> R+2.38
UT_TT_ENV = (41 / 84 - 0.5) * 2 * 100   # = -2.38


def run(df, label, race_generic, env_dial, hisp_adj):
    """Run one Monte Carlo config and return a result dict + the env/label."""
    model.TX_HISPANIC_ADJUSTMENT = hisp_adj
    res = run_monte_carlo(df, env_dial, race_generic, N_SIMULATIONS)
    res["_label"] = label
    res["_env"] = env_dial
    res["_hisp"] = hisp_adj
    return res


def env_str(env_dial):
    return f"D+{env_dial:.1f}" if env_dial >= 0 else f"R+{-env_dial:.1f}"


def line(res):
    h = res["expected_house_seats"]
    hp = res["house_control_prob"] * 100
    s = res["expected_senate_seats"] + _SENATE_D_HOLDOVER
    sp = res["senate_control_prob"] * 100
    return (f"House {h:5.1f}/150  ({hp:4.1f}% maj)    "
            f"Senate {s:4.1f}/31  ({sp:4.1f}% maj)")


def main():
    print("=" * 78)
    print("  TX Legislature 2026 — STRAIGHT UT/TT what-if (no Hispanic adjustment)")
    print("=" * 78)
    print(f"  UT/TT April 2026 crosstabs (D 2p):  "
          f"W={UT_TT['white_nh']:.3f}  B={UT_TT['black_nh']:.3f}  "
          f"H={UT_TT['hispanic']:.3f}  O={UT_TT['other']:.3f}")
    print(f"  UT/TT topline env:  {env_str(UT_TT_ENV)}   (national aggregate: {env_str(CURRENT_ENV)})")
    print(f"  Simulations: {N_SIMULATIONS:,}")

    df = load_districts()

    runs = [
        run(df, "1. PRODUCTION  (national xtabs, Hisp -0.05, D+6.4)",
            RACE_GENERIC_BALLOT_D_SHARE, CURRENT_ENV, -0.05),
        run(df, "2. UT/TT structure  (UT/TT xtabs, no Hisp adj, env held D+6.4)",
            UT_TT, CURRENT_ENV, 0.0),
        run(df, "3. STRAIGHT UT/TT  (UT/TT xtabs, no Hisp adj, env R+2.4)",
            UT_TT, UT_TT_ENV, 0.0),
    ]

    print(f"\n{'-'*78}")
    for r in runs:
        print(f"  {r['_label']}")
        print(f"      {line(r)}")
    print(f"{'-'*78}")

    prod, struct, straight = runs
    d_struct = struct["expected_house_seats"] - prod["expected_house_seats"]
    d_level  = straight["expected_house_seats"] - struct["expected_house_seats"]
    d_total  = straight["expected_house_seats"] - prod["expected_house_seats"]
    print(f"\n  House-seat decomposition (vs production {prod['expected_house_seats']:.1f}):")
    print(f"    swap to UT/TT Hispanic structure (drop -0.05 adj):  {d_struct:+.1f} seats")
    print(f"    drop topline level D+6.4 -> R+2.4:                  {d_level:+.1f} seats")
    print(f"    combined (straight UT/TT):                          {d_total:+.1f} seats")

    # Competitive districts under the straight-UT/TT run
    wp = straight["district_win_probs"]
    comp = df.copy()
    comp["wp"] = wp
    comp = comp[(comp["wp"] >= 0.25) & (comp["wp"] <= 0.75)].sort_values("wp", ascending=False)
    print(f"\n  Competitive districts under STRAIGHT UT/TT (25-75% D win):  {len(comp)}")
    for _, r in comp.iterrows():
        opn = "open " if str(r.get("open_seat","")).strip().lower() in ("true","1","yes") else \
              f"{str(r['incumbent_party'])[:1]}-inc"
        print(f"    {r['chamber'][:1]}D{int(r['district']):>3}  {str(r['incumbent'])[:24]:24}  "
              f"({opn})  D {r['wp']*100:4.1f}%")


if __name__ == "__main__":
    main()
