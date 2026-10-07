"""
war_persistence_estimate.py

How much of a candidate's past WAR carries into their next race?
The model carries a candidate's avg_war (mean of their past race WARs) into
the 2026 prediction. The old flat 0.46 was estimated on
the pre-2026-10-07 WAR (91 consecutive-race pairs). WAR has since been rebuilt
(open-race baseline, genuine no-finance fit, cycle fixed effect, education
term), so re-estimate on data/processed/race_war.csv.

Two estimates:
  pairs   next race WAR ~ previous race WAR (consecutive races, same person/party)
  avg     race WAR ~ mean of ALL earlier race WARs   <- what the model applies
Both through the origin (WAR is mean-zero by cycle). Also by number of prior
races, since an average of more races should carry more.

Result (2026-10-07): persistence rises with races on record; adopted as
model_config.WAR_PERSISTENCE_K, beta(n) = n / (n + 2.93).

Usage:
  python scripts/war_persistence_estimate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent


def slope(x, y):
    b = (x * y).sum() / (x * x).sum()
    e = y - b * x
    se = np.sqrt((e ** 2).sum() / (len(x) - 1) / (x * x).sum())
    return b, se, np.corrcoef(x, y)[0, 1]


def main():
    r = pd.read_csv(ROOT / "data" / "processed" / "race_war.csv")
    r = r.sort_values("year")
    rows = []
    for (who, party), g in r.groupby(["candidate_norm", "party"]):
        g = g.drop_duplicates("year")
        for i in range(1, len(g)):
            prior = g.iloc[:i]
            cur = g.iloc[i]
            rows.append({"who": who, "party": party, "year": int(cur.year),
                         "war": cur.war_race, "prev": prior.war_race.iloc[-1],
                         "avg_prior": prior.war_race.mean(), "n_prior": len(prior),
                         "competitive": bool(cur.competitive_race),
                         "incumbent": bool(cur.is_incumbent)})
    p = pd.DataFrame(rows)
    print(f"Race pairs with a prior WAR: {len(p)} ({p.who.nunique()} people); "
          f"by year {p.groupby('year').size().to_dict()}")

    def show(d, label):
        if len(d) < 10:
            return
        b1, s1, r1 = slope(d.prev, d.war)
        b2, s2, r2 = slope(d.avg_prior, d.war)
        print(f"  {label:<34} n={len(d):>3}   pairs beta {b1:.3f} (se {s1:.3f}, r {r1:.2f})   "
              f"avg-of-prior beta {b2:.3f} (se {s2:.3f}, r {r2:.2f})")

    print()
    show(p, "all")
    show(p[p.competitive], "competitive races (30-70%)")
    show(p[p.incumbent], "incumbents")
    for party in ("D", "R"):
        show(p[p.party == party], f"party {party}")
    for y in sorted(p.year.unique()):
        show(p[p.year == y], f"target year {y}")
    for n in (1, 2, 3):
        show(p[p.n_prior == n] if n < 3 else p[p.n_prior >= n], f"n_prior {'>=' if n == 3 else '='}{n}")

    # Old-definition check: the 91-pair estimate used 2018->2022 and 2022->2024.
    old = p[((p.year == 2022) | (p.year == 2024)) & (p.n_prior >= 1)]
    show(old, "target 2022+2024 (old design)")


if __name__ == "__main__":
    main()
