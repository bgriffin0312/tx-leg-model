"""Label-neutral reading of the statewide early vote.

Within one voter file, turnout rates by party and vote history do not depend on how the other file
labelled people. So: apply 2022's rates (party x history) to the 2026 file, and compare the net that
gives with the actual 2026 net. The gap is how far the turnout pattern alone moved the electorate.
Also the pool still left, by party, for a range of Election Day sizes.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
D = json.load(open(ROOT / "data" / "raw" / "early_vote" / "coldspark_2026-10-30.json"))
pd.set_option("display.width", 220)


def ag(y, counties=None):
    h = D["heat"][y]
    cols = h["cols"]
    o = {}
    for c, p in h["data"].items():
        if counties is not None and c not in counties:
            continue
        for k, rows in p.items():
            for r in rows:
                o.setdefault((k, r[0]), np.zeros(len(cols) - 1))
                o[(k, r[0])] += np.array(r[1:], float)
    return pd.DataFrame(o, index=cols[1:]).T


def net(s):
    return float((s["R"] - s["D"]) / s.sum() * 100)


def report(lab, counties=None):
    a22, a26 = ag("2022", counties), ag("2026", counties)
    r22 = a22.ev_now / a22.reg
    cf = (a26.reg * r22).groupby(level=0).sum()
    # scale 2022 rates so the counterfactual has the 2026 total, keeping the pattern
    cf2 = (a26.reg * r22 * a26.ev_now.sum() / (a26.reg * r22).sum()).groupby(level=0).sum()
    ac = a26.ev_now.groupby(level=0).sum()
    e22 = a22.ev_now.groupby(level=0).sum()
    print(f"{lab}: 2022 actual net {net(e22):+.1f}; 2026 file at 2022 rates {net(cf):+.1f} (scaled {net(cf2):+.1f}); 2026 actual {net(ac):+.1f}; turnout-pattern shift {net(ac) - net(cf2):+.1f}; file shift {net(cf2) - net(e22):+.1f}")
    # the same, one party at a time: rate ratios
    t = pd.DataFrame({"r22": r22 * 100, "r26": a26.ev_now / a26.reg * 100})
    t["ratio"] = t.r26 / t.r22
    return t, a26


t, a26 = report("state")
print(t.round(1).unstack(0).to_string())
BORDER = {"HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"}
allc = set(D["heat"]["2026"]["data"])
report("no border", allc - BORDER)
report("border", BORDER)
for c in ("HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON", "MONTGOMERY", "HIDALGO", "EL PASO"):
    report(c, {c})

print("\nStill left, by party and history (registered minus early voted, 2026 file; includes people who moved or will not vote)")
left = (a26.reg - a26.ev_now).unstack(0)
print(left.round(0).astype(int).to_string())
print("rates by top two history groups")

# Election Day electorate: take 2022 ED voters' propensity by party x history as the share of each
# remaining pool that votes, scaled to a target ED size.
a22 = ag("2022")
print("\ncheck: what the page's 2022 ED voter pool says")
Ed = json.load(open(ROOT / "data" / "raw" / "early_vote" / "coldspark_eday_2026-10-30.json"))
for y in ("2022", "2024"):
    p = Ed["p"][y]["_"]
    print(y, "ED voters R,D,O", p[:3], "net", round((p[0] - p[1]) / sum(p[:3]) * 100, 1), "| of those, voted early in 2026", p[3:], [round(p[3 + i] / p[i] * 100, 1) for i in range(3)])
p = Ed["p"]["2022"]["_"]
q = Ed["p"]["2024"]["_"]
EV = 7.75e6
Enet = 3.2
for lab, pp, in (("2022 ED voters not yet voted", p), ("2024 ED voters not yet voted", q)):
    rem = np.array(pp[:3]) - np.array(pp[3:])
    print(lab, rem, "net", round((rem[0] - rem[1]) / rem.sum() * 100, 1))
print("\nFinal electorate net for ED size and ED net")
rows = []
for ed in (2.2e6, 2.6e6, 3.0e6):
    for n in (0, 5, 10, 15, 20):
        rows.append({"ED_m": ed / 1e6, "ED_net": n, "total_m": round((EV + ed) / 1e6, 2), "final_net": round((EV * Enet + ed * n) / (EV + ed), 1)})
print(pd.DataFrame(rows).pivot_table(index="ED_net", columns="ED_m", values="final_net").to_string())
