"""How much did the 2025 congressional redraw (PlanC2333) gain Republicans in 2026, against the 2021 map (PlanC2193)?

Method: every 2024 VTD's judicial composite (Supreme Court + CCA, equal-weight mean), shifted by its county's measured
2024 -> 2026 judicial swing and scaled by its county's turnout ratio, then summed into
  - old-map districts (a VTD belongs to the district whose 2024 U.S. House contest it voted in), and
  - new-map districts (exact TLC Red-110 assignment, as in src/statewide_results.py).
County swing comes from the election-night projection (fully counted counties' actual 2026 judicial margin).
Check: the new-map estimate against the actual 2026 U.S. House results from the state feed.
Limit: swing is uniform within a county, so a county whose Hispanic precincts swung more than its others is smoothed.

    python scripts/congress_map_counterfactual.py  -> output/congress_map_counterfactual_2026.csv
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import statewide_results as SR  # noqa: E402

EN = ROOT / "data" / "raw" / "election_night"

# 1. 2024 VTD judicial composite
df24 = SR.load(2024)
jud = sorted(o for o in df24.Office.unique() if SR.classify(o) == "judicial")
sel = df24[df24.Office.isin(jud) & df24.Party.isin(["D", "R"])]
# contested offices only: an office with no D candidate would drag the composite
contested = [o for o, g in sel.groupby("Office") if g.Party.nunique() == 2]
v = sel[sel.Office.isin(contested)].groupby(["cntyvtd", "Party"]).Votes.sum().unstack(fill_value=0) / len(contested)
v = v.reset_index()
print(f"2024 judicial offices used: {len(contested)}; VTDs {len(v)}")
fips_of = {k.strip().upper(): int(f) for k, f in df24.groupby("County").FIPS.first().items()}
cty_of_vtd = df24.groupby("cntyvtd").County.first().str.strip().str.upper()
v["county"] = v.cntyvtd.map(cty_of_vtd)

# 2. county swing and turnout ratio, 2024 -> 2026 (newest projection file)
pf = max(glob.glob(str(EN / "projection_*.csv")), key=os.path.getmtime)
p = pd.read_csv(pf)
p["county"] = p.county.str.upper().replace({"LASALLE": "LA SALLE"})
c24 = v.groupby("county")[["D", "R"]].sum()
c24["m24"] = (c24.D - c24.R) / (c24.D + c24.R) * 100
c24["n24"] = c24.D + c24.R
p = p.set_index("county")
c24["m26"] = p.m_total.reindex(c24.index)
c24["n26"] = (p.n_total / 7).reindex(c24.index)     # n_total sums the 7 judicial races
c24["status"] = p.status.reindex(c24.index)
c24["swing"] = c24.m26 - c24.m24
c24["ratio"] = c24.n26 / c24.n24
miss = c24[c24.m26.isna()]
print(f"projection {Path(pf).name}; counties without a 2026 read: {list(miss.index)}")
print(f"county swing: median {c24.swing.median():+.1f}, vote-weighted {np.average(c24.swing.dropna(), weights=c24.n24[c24.swing.notna()]):+.1f}; "
      f"turnout ratio median {c24.ratio.median():.2f}; status {c24.status.value_counts().to_dict()}")
print(f"  check: 2024 composite vs projection j2024 (vote-weighted gap) "
      f"{np.average((c24.m24 - p.j2024.reindex(c24.index)).dropna(), weights=c24.n24[p.j2024.reindex(c24.index).notna()]):+.2f}")

v = v.merge(c24[["swing", "ratio"]], left_on="county", right_index=True, how="left")
v["m24"] = (v.D - v.R) / (v.D + v.R).replace(0, np.nan) * 100
v["n26"] = (v.D + v.R) * v.ratio
v["m26"] = (v.m24 + v.swing).clip(-100, 100)
v["D26"] = v.n26 * (1 + v.m26 / 100) / 2
v["R26"] = v.n26 * (1 - v.m26 / 100) / 2

# 3. district assignments
# old map: TLC's 2024 precinct -> PlanC2193 crosswalk (exact; PCTKEY = FIPS + 4-digit precinct, the returns' cntyvtd).
# Precincts it misses fall back to the district whose 2024 U.S. House contest they voted in (uncontested districts
# TX-1, 9, 11, 13, 20 have no contest in the returns, which is why the crosswalk comes first).
xw = pd.read_excel(ROOT / "data" / "raw" / "_capitol_data_cache" / "precincts24g_districts.xlsx")
xw["cntyvtd"] = xw.PCTKEY.astype(str)
xw = xw.groupby(["cntyvtd", "PlanC2193"]).size().reset_index(name="n").rename(columns={"PlanC2193": "dist"})
xw["w"] = xw.n / xw.groupby("cntyvtd").n.transform("sum")
cd = df24[df24.Office.str.match(r"^U\.S\. Rep")].copy()
cd["dist"] = cd.Office.str.extract(r"(\d+)").astype(int)
con = cd.groupby(["cntyvtd", "dist"]).Votes.sum().reset_index()
con["w"] = con.Votes / con.groupby("cntyvtd").Votes.transform("sum")
hit = v.cntyvtd.isin(xw.cntyvtd)
print(f"old-map crosswalk matches {hit.mean():.1%} of VTDs, {(v.D + v.R)[hit].sum() / (v.D + v.R).sum():.2%} of the 2024 judicial vote")
old = pd.concat([xw[["cntyvtd", "dist", "w"]], con[~con.cntyvtd.isin(xw.cntyvtd)][["cntyvtd", "dist", "w"]]], ignore_index=True)
new = SR.vtd_assignment(df24)


def agg(assign, label):
    m = v.merge(assign, on="cntyvtd", how="inner")
    lost = 1 - (m.D * m.w + m.R * m.w).sum() / (v.D + v.R).sum()
    for a, b in (("D", "R"), ("D26", "R26")):
        m[a + "w"], m[b + "w"] = m[a] * m.w, m[b] * m.w
    g = m.groupby("dist")[["Dw", "Rw", "D26w", "R26w"]].sum()
    out = pd.DataFrame({"map": label, "dist": g.index,
                        "jud24": (g.Dw - g.Rw) / (g.Dw + g.Rw) * 100,
                        "jud26": (g.D26w - g.R26w) / (g.D26w + g.R26w) * 100})
    print(f"{label}: {len(out)} districts, 2024 judicial vote unmatched {lost:.2%}")
    return out.reset_index(drop=True)


O, N = agg(old, "2021 map (PlanC2193)"), agg(new, "2025 map (PlanC2333)")

# 4. actual 2026 U.S. House results (newest state-feed snapshot; Fort Bend replaced with its Clarity count)
snap = sorted(EN.glob("countyInfo_*.json.gz"), key=lambda q: q.stat().st_mtime)[-1]
ci = json.load(gzip.open(snap, "rt", encoding="utf-8"))
RX = re.compile(r"(?:U\. ?S\. ?|UNITED STATES )REPRESENTATIVE\D*(\d+)")
act = defaultdict(lambda: {"D": 0, "R": 0, "dn": "", "rn": ""})
for cty in ci.values():
    cn = cty.get("N", "").upper()
    for race in cty.get("Races", {}).values():
        mm = RX.search((race.get("N") or "").upper())
        if not mm or cn == "FORT BEND":
            continue
        a = act[int(mm.group(1))]
        for c in race.get("C", {}).values():
            if c.get("P") == "DEM":
                a["D"] += c.get("V") or 0; a["dn"] = c.get("N")
            elif c.get("P") == "REP":
                a["R"] += c.get("V") or 0; a["rn"] = c.get("N")
fb = json.loads((EN / "local_FORT_BEND_latest.json").read_text())["contests"]
for name, c in fb.items():
    mm = re.match(r"UNITED STATES REPRESENTATIVE, DISTRICT (\d+)", name)
    if mm:
        for ch in c["choices"]:
            if ch["party"] in ("DEM", "REP"):
                act[int(mm.group(1))][ch["party"][0]] += ch["total"]
A = pd.DataFrame([{"dist": d, "actual26": (x["D"] - x["R"]) / (x["D"] + x["R"]) * 100, "d_name": x["dn"], "r_name": x["rn"]}
                  for d, x in act.items()])
N = N.merge(A, on="dist", how="left")
fc = pd.read_csv(ROOT / "output" / "congress_forecast_2026.csv")[["dist", "pres24", "mean", "inc_d", "inc_r"]]
N = N.merge(fc.rename(columns={"mean": "model_mean"}), on="dist", how="left")
N["cand_gap"] = N.actual26 - N.jud26

r24 = pd.read_csv(ROOT / "data" / "raw" / "statewide" / "results_us_house.csv")
r24 = r24[r24.year == 2024][["dist", "d_name", "r_name", "d_inc", "r_inc", "pres_margin", "d_margin"]]
O = O.merge(r24.rename(columns={"d_name": "d_name_2024", "r_name": "r_name_2024", "d_margin": "actual24"}), on="dist", how="left")

pd.concat([O, N], ignore_index=True).to_csv(ROOT / "output" / "congress_map_counterfactual_2026.csv", index=False)

# 5. report
f = lambda x: f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}"
print("\nCHECK, 2025 map: estimated 2026 judicial margin vs actual 2026 U.S. House result")
print(f"  corr {N.jud26.corr(N.actual26):.3f}; actual minus estimate: mean {N.cand_gap.mean():+.1f}, sd {N.cand_gap.std():.1f}; "
      f"same winner in {int(((N.jud26 > 0) == (N.actual26 > 0)).sum())} of {len(N)}")
for r in N.sort_values("jud26").itertuples():
    if abs(r.jud26) < 20 or (r.jud26 > 0) != (r.actual26 > 0):
        print(f"  TX-{r.dist:<2} est {f(r.jud26):>7}  actual {f(r.actual26):>7}  gap {r.cand_gap:+5.1f}  (2024 judicial {f(r.jud24)}, Trump-Harris {f(r.pres24)})  {r.r_name} v {r.d_name}")
print("\nSEATS on the 2026 judicial vote (no candidate effects), D-R:")
for lab, X, col in (("2021 map", O, "jud26"), ("2025 map", N, "jud26")):
    nd = int((X[col] > 0).sum())
    print(f"  {lab}: {nd} D - {len(X) - nd} R   (2024 judicial vote: {int((X.jud24 > 0).sum())} D)")
print(f"  2025 map, actual 2026 result: {int((N.actual26 > 0).sum())} D - {int((N.actual26 <= 0).sum())} R")
print("\n2021 map districts within 15 points on the 2026 estimate:")
for r in O.sort_values("jud26").itertuples():
    if abs(r.jud26) < 15:
        print(f"  old TX-{r.dist:<2} est 2026 {f(r.jud26):>7}  (2024 judicial {f(r.jud24)}; 2024 result {f(r.actual24)} {r.r_name_2024} v {r.d_name_2024})")
