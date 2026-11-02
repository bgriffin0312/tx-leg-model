"""Backtest score of a mean-zero error on R-held and open House seats within 30 points. No center
moves. Same seat log score as the shared-error grid, plus coverage and the shipped 6.7 for scale."""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

import whole_election_backtest as b

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
rows = []
envs = {y: (b.GB.final_538[y] + b.GB.final_rcp[y]) / 2 for y in b.YEARS}
for y in b.YEARS:
    x = b.Cycle(y, envs[y])
    di = x.te[x.is_h].dem_incumbent.astype(float).values == 1
    for v in ("statewide", "legislative"):
        for label, st, sd in (("thin", None, 0.0), ("group 1.5", None, 1.5), ("group 2.5", None, 2.5), ("group 3.5", None, 3.5), ("group 4.5", None, 4.5), ("shared 6.7", 6.7, 0.0)):
            r = x.run(v, shared_total=st)
            m = r["kind"] == "state_house"
            L = r["M"][:, m].copy()
            act = int((r["actual"][m] > 0).sum()) + x.fixed_h
            g = (~di) & (np.abs(L.mean(0)) < 30)
            for seed in range(4):
                z = np.random.default_rng(100 + seed).normal(0, sd, len(L))[:, None] * g[None, :]
                s = ((L + z) > 0).sum(1) + x.fixed_h
                rows.append({"year": y, "version": v, "err": label, "score": b.seat_score(s, act), "mean": s.mean(), "q10": np.percentile(s, 10), "q90": np.percentile(s, 90), "in80": np.percentile(s, 10) <= act <= np.percentile(s, 90), "actual": act})
t = pd.DataFrame(rows).groupby(["version", "err", "year"], as_index=False).mean(numeric_only=True)
print(t.pivot_table(index=["version", "err"], columns="year", values="score").round(2).assign(mean=lambda q: q.mean(axis=1).round(3)).to_string())
print(t.pivot_table(index=["version", "err"], columns="year", values="q90").round(1).to_string())
