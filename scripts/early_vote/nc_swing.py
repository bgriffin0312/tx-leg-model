"""North Carolina 2026 early vote vs 2024 early vote, same counties, same vote modes (election night).

Reads NCSBE ENR county files (results_<cid>.txt) for 2026-11-03 and 2024-11-05 saved under a folder.
Early = one-stop (ovc) + mail (avc). Compares the down-ballot judicial races (Supreme Court + Court of
Appeals, both parties) and the top of the ticket, county by county, only in counties whose 2026 early
votes are in, weighted by 2026 early votes.
    python scripts/early_vote/nc_swing.py <dir26> <dir24>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

d26, d24 = Path(sys.argv[1]), Path(sys.argv[2])


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def agg(rows, pred):
    out = {"DEM": [0, 0, 0], "REP": [0, 0, 0]}   # early (one-stop + mail), eday, total
    for x in rows:
        if pred(x["cnm"]) and x["pty"] in out:
            out[x["pty"]][0] += int(x.get("ovc") or 0) + int(x.get("avc") or 0)
            out[x["pty"]][1] += int(x.get("evc") or 0)
            out[x["pty"]][2] += int(x.get("vct") or 0)
    return out


def mg(a, i):
    d, r = a["DEM"][i], a["REP"][i]
    return (d - r) / (d + r) * 100 if d + r else np.nan


jud = lambda n: ("SUPREME COURT" in n or "COURT OF APPEALS" in n)
rows = []
for cid in range(1, 101):
    a, b = load(d26 / f"{cid}.txt"), load(d24 / f"{cid}.txt")
    if not a or not b:
        continue
    j26, j24 = agg(a, jud), agg(b, jud)
    s26 = agg(a, lambda n: n.startswith("US SENATE"))
    p24 = agg(b, lambda n: n.startswith("US PRESIDENT"))
    n26 = j26["DEM"][0] + j26["REP"][0]
    if n26 == 0:
        continue
    rows.append({"cid": cid, "n26": n26, "j26_early": mg(j26, 0), "j24_early": mg(j24, 0), "j24_final": mg(j24, 2),
                 "sen26_early": mg(s26, 0), "pres24_early": mg(p24, 0), "pres24_final": mg(p24, 2)})
t = pd.DataFrame(rows).dropna()
w = t.n26
print(f"counties with 2026 early votes in: {len(t)} of 100; early judicial votes {int(w.sum()):,}")
for lab, a, b in (("judicial, early vs 2024 early", "j26_early", "j24_early"), ("judicial 2026 early vs 2024 final", "j26_early", "j24_final"),
                  ("Senate 2026 early vs 2024 president early", "sen26_early", "pres24_early"), ("Senate 2026 early vs 2024 president final", "sen26_early", "pres24_final")):
    s = t[a] - t[b]
    print(f"  swing {lab}: weighted {np.average(s, weights=w):+.1f}  median {s.median():+.1f}  (10-90 {s.quantile(.1):+.1f} to {s.quantile(.9):+.1f})")
print("  reporting counties' 2024 judicial final margin, weighted:", round(np.average(t.j24_final, weights=w), 1))
