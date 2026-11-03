"""Williamson County Commissioner, Precinct 4: Boles (R, inc) v Rhea (D), write-in Flores.

Only contested commissioner race in the county in 2026: Precinct 2 is McDonald (R) unopposed, Precincts
1 and 3 are not up. Commissioner races are not in the model, so this reads Precinct 4 off the county race.

Footprint: voting precincts with Precinct 4 commissioner votes in the 2022 OpenElections file, on the
2021 lines. Offset: Precinct 4 minus the county on the judicial vote, 2022 and 2024. Margins are D minus R.
Two estimates of the 2026 mean:
  A  county race at each T, minus Jefts's money term (Rhea has no finance on file), plus the offset.
     Boles and Snell carry the same R-seat term.
  B  Boles's 2022 margin plus the county's modeled swing from 2022.
  C  T plus the county's gap to the state on the judicial vote, plus the offset, plus the mean
     gap between these commissioner races and the judicial vote on their own precincts.
Error: shared with the county race through T, plus a race error sized from every Precinct 2 and 4 race,
2014-2022, against the judicial vote on the same precincts. Small n.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import statewide_results as sr

pd.set_option("display.width", 200, "display.max_rows", 80)
OE = ROOT / "data" / "raw" / "county_judge" / "oe"
JUD = r"supreme court|criminal appeals"
BETA = 0.95          # races per point of T
M_T6 = 6.53          # Williamson county judge mean at T -6.0, data/raw/early_vote/race_shift.csv
MONEY = 1.98 * np.log((339480 + 25000) / (288939 + 25000))
SHARED_SD, COUNTY_SD = 3.4, 4.2    # county P(D) .885 at +6.5 implies about 5.4 in total


def load(y, f):
    d = pd.read_csv(OE / f"{f}__tx__general__williamson__precinct.csv", dtype=str)
    d["votes"] = pd.to_numeric(d.votes.str.replace(",", ""), errors="coerce").fillna(0)
    d["office"] = d.office.fillna("") + " " + d.district.fillna("")
    d["p"] = d.party.fillna("").str.upper().str[:3]
    return d[d.p.isin(["DEM", "REP"])]


def margin(x):
    d, r = x[x.p == "DEM"].votes.sum(), x[x.p == "REP"].votes.sum()
    return 100 * (d - r) / (d + r) if d + r else np.nan


def jud(d):
    o = d[d.office.str.contains(JUD, case=False)]
    keep = o.groupby("office").p.nunique()
    return o[o.office.isin(keep[keep == 2].index)]


rows = []
sets = {}
for y, f in ((2014, "20141104"), (2018, "20181106"), (2022, "20221108")):
    d = load(y, f)
    J = jud(d)
    print(y, "judicial races", J.office.nunique(), "county J", round(margin(J), 2))
    for n in ("2", "4"):
        c = d[d.office.str.contains(rf"Commissioner.*Precinct(?: No\.)?\s*{n}\b", case=False, regex=True) & ~d.office.str.contains("Agric|Land|Railroad", case=False)]
        pr = set(c[c.votes > 0].precinct)
        if y == 2022:
            sets[n] = pr
        Jp = J[J.precinct.isin(pr)]
        names = c.groupby(["p", "candidate"]).votes.sum().reset_index()
        rows.append({"year": y, "pct": n, "precincts": len(pr), "race": margin(c), "J_here": margin(Jp), "J_county": margin(J), "offset": margin(Jp) - margin(J), "race_minus_J": margin(c) - margin(Jp),
                     "who": " v ".join(names.candidate)})
r = pd.DataFrame(rows)
print(r.round(1).to_string(index=False))
sd_race = float(np.sqrt((r.race_minus_J - r.groupby("year").race_minus_J.transform("mean")).pow(2).sum() / max(len(r) - 3, 1)))
print("race error sd, within year", round(sd_race, 2), "mean race minus J, R-held", round(r.race_minus_J.mean(), 2))

print("\n2024 offset, TLC VTD returns")
t = sr.load(2024)
print(t.columns.tolist()[:10])
w = t[t.County.str.upper().str.strip() == "WILLIAMSON"].copy()
w["pr"] = w.cntyvtd.astype(str).str[3:].str.lstrip("0")
w["party"] = w.Party.str[0].str.upper()
jo = [o for o in w.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)]
ok = [o for o in jo if w[w.Office == o].party.nunique() >= 2 and {"D", "R"} <= set(w[w.Office == o].party)]
J = w[w.Office.isin(ok) & w.party.isin(["D", "R"])]


def m2(x):
    d, r_ = x[x.party == "D"].Votes.sum(), x[x.party == "R"].Votes.sum()
    return 100 * (d - r_) / (d + r_)


s4 = {str(p).lstrip("0") for p in sets["4"]}
print("VTDs", J.pr.nunique(), "in Pct 4 footprint", J[J.pr.isin(s4)].pr.nunique(), "of", len(s4), "; Pct 4 share of county J votes", round(J[J.pr.isin(s4)].Votes.sum() / J.Votes.sum(), 3))
off24 = m2(J[J.pr.isin(s4)]) - m2(J)
print("county J 2024", round(m2(J), 2), "Pct 4", round(m2(J[J.pr.isin(s4)]), 2), "offset", round(off24, 2))
off22 = float(r[(r.year == 2022) & (r.pct == "4")].offset.iloc[0])
offset = (off22 + off24) / 2
b22 = float(r[(r.year == 2022) & (r.pct == "4")].race.iloc[0])
g22 = 100 * (103180 - 108778) / (103180 + 108778)
print("offset used", round(offset, 2), "; Jefts money term removed", round(MONEY, 2), "; Boles 2022", round(b22, 1), "Gravell 2022", round(g22, 1))

CTY = ((-3.56 + 13.96) + (m2(J) + 16.35)) / 2     # county J minus state J, 2022 and 2024
GAP = float(r.race_minus_J.mean())
print("county J minus state J", round(CTY, 2), "commissioner minus J", round(GAP, 2))

print("\nForecast")
rng = np.random.default_rng(4)
n = 400000
S = rng.normal(0, SHARED_SD, n)
Uc = rng.normal(0, COUNTY_SD, n)
Ur = rng.normal(0, 1, n)
out = []
for lab, T in (("R+7.15", -7.15), ("legislative", -7.2), ("R+6 read", -6.0), ("statewide", -4.78), ("D+3.1", 3.1)):
    mc = M_T6 + BETA * (T + 6.0)
    ma = mc - MONEY + offset
    mb = b22 + (mc - g22)
    mcc = T + CTY + offset + GAP
    for name, m in (("A", ma), ("B", mb), ("C", mcc), ("mean", (ma + mb + mcc) / 3)):
        c = mc + S + Uc
        p4 = m + S + 0.5 * Uc + Ur * max(sd_race, 3.0)
        out.append({"T": lab, "est": name, "county": mc, "Pct4_mean": m, "P_Jefts": (c > 0).mean(), "P_Rhea": (p4 > 0).mean(), "both": ((c > 0) & (p4 > 0)).mean(), "Jefts_only": ((c > 0) & (p4 <= 0)).mean()})
print(pd.DataFrame(out).round(3).to_string(index=False))
