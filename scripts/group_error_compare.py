"""
group_error_compare.py

What the correlated group-error layer changes, at the current environment:
same seed, layer on vs off. Marginal district odds barely move (total sigma is
held fixed); what changes is how seats move together.

Usage:
  python scripts/group_error_compare.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model  # noqa: E402

# Majority-Hispanic seats on the competitive edge, and a DFW/Houston suburban set.
VALLEY = [("house", d) for d in (34, 37, 74, 41, 35, 31, 80, 118)]
SUBURB = [("house", d) for d in (112, 121, 52, 108, 138, 63, 94, 96)]


def run(enabled: bool, df: pd.DataFrame):
    model.GROUP_ERROR_ENABLED = enabled
    model.RNG = np.random.default_rng(42)
    return model.run_monte_carlo(df, model.CURRENT_ENV, model.RACE_GENERIC_BALLOT_D_SHARE,
                                 keep_wins=True)


def block(df, res, seats):
    idx = [i for i, (c, d) in enumerate(zip(df.chamber_lower, df.district)) if (c, int(d)) in seats]
    w = res["wins"][idx, :]
    p = w.mean(axis=1)
    k = w.sum(axis=0)
    live = w.std(axis=1) > 0          # near-certain seats have no variance
    corr = np.corrcoef(w[live].astype(float))
    off = corr[~np.eye(int(live.sum()), dtype=bool)]
    return {"expected_D": p.sum(), "sd_D": k.std(), "P(D wins none)": (k == 0).mean(),
            "P(D wins all)": (k == len(idx)).mean(), "mean pairwise corr": np.nanmean(off)}


def main():
    df = model.load_districts()
    on, off = run(True, df), run(False, df)
    print(f"Environment D{model.CURRENT_ENV:+.1f}, {len(df)} districts, same seed both runs\n")
    for name, r in (("layer OFF", off), ("layer ON", on)):
        hs = r["house_seat_dist"]
        print(f"{name:<10} House E={hs.mean():.1f}  sd={hs.std():.2f}  10-90 "
              f"{np.percentile(hs, 10):.0f}-{np.percentile(hs, 90):.0f}  P(maj)={r['house_control_prob']:.3f}  "
              f"Senate P(maj)={r['senate_control_prob']:.3f}")
    dp = (on["district_win_probs"] - off["district_win_probs"]).abs()
    print(f"\nLargest change in any district's D win probability: {dp.max() * 100:.1f}pp "
          f"(mean {dp.mean() * 100:.2f}pp)")
    for label, seats in (("VALLEY / majority-Hispanic", VALLEY), ("SUBURBAN", SUBURB)):
        print(f"\n{label}: {', '.join(f'HD {d}' for _, d in seats)}")
        rows = {n: block(df, r, seats) for n, r in (("OFF", off), ("ON", on))}
        print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
