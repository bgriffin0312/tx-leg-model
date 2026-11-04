"""Election-night watch sheet: what each big county's EARLY vote should look like if the model is right.

Final benchmark for a party-line race (AG / Lt Gov, gap to T about 0) = county's 2024 judicial lean
relative to the state + T, plus the early-vote county shift (0.5 per point, non-border, the T -6.0 read).
Early-only benchmark = final benchmark minus the drift from early to final electorate composition
(ColdSpark modeled party: early voters vs the 2022 / 2024 Election Day voters who have not voted yet).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
S = ROOT / "data" / "raw" / "early_vote"
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)

e = pd.read_csv(S / "county_panel2.csv", index_col=0)
Ed = json.load(open(S / "coldspark_eday_2026-10-30.json"))

# state judicial margins (D minus R), vote weighted
for y in (2022, 2024):
    jj = e[f"j{y}"]; nn = e[f"n{y}"]
    ok = jj.notna() & nn.notna()
    print(y, "state judicial", round(np.average(jj[ok], weights=nn[ok]), 2), "top", round(np.average(e[f"top{y}"][ok], weights=nn[ok]), 2))
J24 = np.average(e.j2024.dropna(), weights=e.n2024[e.j2024.notna()])
J22 = np.average(e.j2022.dropna(), weights=e.n2022[e.j2022.notna()])

BORDER = ["HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"]
nb = e.sort_values("n2024").tail(60).index.difference(BORDER)
w = e.t2026[nb]
sig = {}
for name in ("d24", "adj"):
    v = e.loc[nb, name]
    sig[name] = -(v - np.average(v, weights=w))
shift = (0.5 * (sig["d24"] + sig["adj"]) / 2).reindex(e.index).fillna(0.0)

TOP = ["HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON",
       "MONTGOMERY", "EL PASO", "HIDALGO", "CAMERON", "WEBB", "NUECES", "HAYS", "BRAZORIA", "GALVESTON", "BELL", "LUBBOCK"]

rows = []
for c in TOP:
    r = e.loc[c]
    # Election Day pools: each past year's ED voters, minus those who already voted early in 2026
    pools = {}
    for y in ("2022", "2024"):
        p = Ed["p"][y].get(c)
        rem = np.array(p[:3], float) - np.array(p[3:], float)
        pools[y] = (rem[0] - rem[1]) / rem.sum() * 100       # R minus D, percent
    ed_share = {y: Ed["y"][y][c][1] / Ed["y"][y][c][0] for y in ("2022", "2024")}
    ev_net = r.E2026                                           # R minus D among 2026 early voters
    out = {"county": c, "ev26_votes_k": r.t2026 / 1000, "ed_share22": ed_share["2022"] * 100, "ed_share24": ed_share["2024"] * 100,
           "ev_net26": ev_net, "ev_net22": r.E2022, "ev_net24": r.E2024, "ed_pool22": pools["2022"], "ed_pool24": pools["2024"],
           "j22": r.j2022, "j24": r.j2024, "top24": r.top2024, "lean24": r.j2024 - J24, "cshift": shift[c]}
    # final composition: ED share midway between 2022's and 2024's (statewide that is ~25.7%, the 2.6M-of-10.35M estimate is ~25%)
    eds = (ed_share["2022"] + ed_share["2024"]) / 2
    for y in ("2022", "2024"):
        fin = (1 - eds) * ev_net + eds * pools[y]
        out[f"drift_{y}"] = fin - ev_net                       # + means the final electorate is redder than the early one
    out["ed_share_used"] = eds * 100
    rows.append(out)
t = pd.DataFrame(rows).set_index("county")

K = 0.9   # margin points per point of composition (labels are not perfectly loyal)
for lab, T in (("T-6.0", -6.0), ("SW-4.8", -4.78), ("LEG-7.2", -7.2)):
    t[f"final_{lab}"] = t.lean24 + T + t.cshift
t["drift"] = t[["drift_2022", "drift_2024"]].mean(axis=1)
# margins are D minus R; drift + = final electorate redder, so the early-only count runs BLUER than final by K*drift
for lab in ("T-6.0", "SW-4.8", "LEG-7.2"):
    t[f"ev_{lab}"] = t[f"final_{lab}"] + K * t.drift
t["ev_d22"] = K * t.drift_2022
t["ev_d24"] = K * t.drift_2024
t["swing_vs_j24"] = t["final_T-6.0"] - t.j24
print(t.round(1).to_string())
show = t[["ev26_votes_k", "j22", "j24", "lean24", "cshift", "ed_share_used", "ev_d22", "ev_d24", "final_SW-4.8", "final_T-6.0", "final_LEG-7.2", "ev_SW-4.8", "ev_T-6.0", "ev_LEG-7.2"]]
print("\nSHEET\n" + show.round(1).to_string())
# composite of the nine big metro counties, weighted by 2026 early votes
big9 = ["HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON"]
wv = t.loc[big9, "ev26_votes_k"]
for col in ("j24", "final_T-6.0", "ev_SW-4.8", "ev_T-6.0", "ev_LEG-7.2"):
    print("big9", col, round(np.average(t.loc[big9, col], weights=wv), 2))
print("big9 share of statewide early vote", round(wv.sum() * 1000 / e.t2026.sum(), 3))
t.round(2).to_csv(ROOT / "output" / "ev_night_benchmarks_2026.csv")

# competitive races from the T -6.0 county-shift read
rs = pd.read_csv(S / "race_shift.csv")
comp = rs[(rs.p1 > 0.08) & (rs.p1 < 0.92)].sort_values("p1", ascending=False)
print("\nRaces 8-92% at T -6.0 with county shifts")
print(comp.round(3).to_string(index=False))
print("\nStatewide / other")
print(rs[rs.kind == "other"].round(3).to_string(index=False))
