"""Statewide early-vote arithmetic: nets by file, registered nets, Election Day pools, and
county totals for the panel. Writes county_panel.csv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "src"))
import statewide_results as sr

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
EV = ROOT / "data" / "raw" / "early_vote"
D = json.load(open(EV / "coldspark_2026-10-30.json"))
Ed = json.load(open(EV / "coldspark_eday_2026-10-30.json"))
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 40)


def res(year):
    df = sr.load(year)
    df = df[df.Party.isin(["D", "R"])].copy()
    df["c"] = df.County.str.upper().str.strip()
    df["race"] = df.Office.map(sr.classify)
    out = {}
    top = {2018: r"^U\.S\. Sen", 2020: "^President", 2022: "^Governor", 2024: "^President", 2014: "^Governor", 2016: "^President", 2012: "^President"}[year]
    t = df[df.Office.str.contains(top, regex=True)].pivot_table(index="c", columns="Party", values="Votes", aggfunc="sum")
    out["top"] = (t.D - t.R) / (t.D + t.R) * 100
    out["n"] = t.D + t.R
    j = df[df.race == "judicial"]
    both = j.groupby("Office").Party.nunique()
    j = j[j.Office.isin(both[both == 2].index)]
    p = j.pivot_table(index="c", columns="Party", values="Votes", aggfunc="sum")
    out["j"] = (p.D - p.R) / (p.D + p.R) * 100
    print(year, "offices in J", both.eq(2).sum(), "state J", round(float((p.D.sum() - p.R.sum()) / (p.D.sum() + p.R.sum()) * 100), 2), "top", round(float((t.D.sum() - t.R.sum()) / (t.D.sum() + t.R.sum()) * 100), 2), "votes", int(t.values.sum()))
    return pd.DataFrame(out)


R = {y: res(y) for y in (2012, 2014, 2016, 2018, 2020, 2022, 2024)}


def tally(rows, cut=-99):
    Rr = sum(x[4] + x[7] for x in rows if x[0] >= cut)
    Dd = sum(x[5] + x[8] for x in rows if x[0] >= cut)
    O = sum(x[6] + x[9] for x in rows if x[0] >= cut)
    return Rr, Dd, O


print("\n=== Early electorate by file (R-D net of all early votes) ===")
ASOF = D["asof"]["d"]
for y in ("2022", "2024", "2026"):
    a = tally(D["daily"][y])
    s = tally(D["daily"][y], ASOF)
    t = sum(a)
    print(y, "final", t, [round(v / t * 100, 1) for v in a], "net", round((a[0] - a[1]) / t * 100, 1), "| at d>=4 net", round((s[0] - s[1]) / sum(s) * 100, 1))


def regnet(y):
    h = D["heat"][y]
    ri = h["cols"].index("reg")
    tot = {"R": 0, "D": 0, "O": 0}
    out = {}
    for c, p in h["data"].items():
        v = {k: sum(r[ri] for r in rows) for k, rows in p.items()}
        out[c] = v
        for k in tot:
            tot[k] += v[k]
    return tot, out


T22, C22 = regnet("2022")
T26, C26 = regnet("2026")
for lab, t in (("2022", T22), ("2026", T26)):
    n = sum(t.values())
    print("registered file", lab, n, {k: round(v / n * 100, 1) for k, v in t.items()}, "net", round((t["R"] - t["D"]) / n * 100, 1))

print("\nheat cols", D["heat"]["2026"]["cols"], "row meaning: party x history bucket 0-4")
for y in ("2022", "2026"):
    h = D["heat"][y]
    cols = h["cols"]
    ag = {}
    for c, p in h["data"].items():
        for k, rows in p.items():
            for r in rows:
                key = (k, r[0])
                ag.setdefault(key, np.zeros(len(cols) - 1))
                ag[key] += np.array(r[1:], float)
    a = pd.DataFrame(ag, index=cols[1:]).T
    a["rate"] = a.ev_now / a.reg * 100
    print(y)
    print(a[["reg", "ev_now", "rate"]].round(1).unstack(0).to_string())

print("\n=== Election Day pools from the page's static table ===")
print("keys", list(Ed.keys()), "lab" in Ed and Ed.get("lab"))
for y in ("2024", "2022"):
    r = Ed["y"][y]["_"]
    p = Ed["p"][y]["_"]
    E, V = p[:3], p[3:]
    print(y, "voted", r[0], "ED", r[1], f"{r[1] / r[0] * 100:.1f}%", "E pool", E, "net", round((E[0] - E[1]) / sum(E) * 100, 1), "share", [round(v / sum(E) * 100, 1) for v in E], "V", V, sum(V))

rows = []
for c in D["county"]["2026"]:
    r = {"county": c}
    for y in ("2022", "2024", "2026"):
        a = tally(D["county"][y].get(c, []))
        r[f"E{y}"] = (a[0] - a[1]) / sum(a) * 100 if sum(a) else np.nan
        r[f"t{y}"] = sum(a)
        r[f"oshare{y}"] = a[2] / sum(a) * 100 if sum(a) else np.nan
    for yy, cc in (("22", C22), ("26", C26)):
        v = cc.get(c)
        r[f"reg{yy}"] = (v["R"] - v["D"]) / sum(v.values()) * 100 if v else np.nan
        r[f"nreg{yy}"] = sum(v.values()) if v else np.nan
    for y in (2022, 2024):
        v = Ed["y"][str(y)].get(c, [np.nan, np.nan])
        r[f"tot{y}"], r[f"ed{y}"] = v
    rows.append(r)
e = pd.DataFrame(rows).set_index("county")
for y, t in R.items():
    e = e.join(t.rename(columns={"top": f"top{y}", "n": f"n{y}", "j": f"j{y}"}))
print(e.isna().sum()[lambda s: s > 0].to_dict())
e.to_csv(S / "county_panel.csv")
print(len(e))
