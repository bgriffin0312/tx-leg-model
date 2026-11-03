"""SD-9 slice of Tarrant's 2026 early vote (ColdSpark modeled party, through Oct 30)
against the same precincts' 2024 presidential and judicial vote."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "src"))
import statewide_results as sr

D = json.load(open(ROOT / "data/raw/early_vote/coldspark_2026-10-30.json"))
cols = ["regR", "regD", "regO", "mR", "mD", "mO", "m2R", "m2D", "m2O", "eR", "eD", "eO"]
p = pd.DataFrame([r[1:] for r in D["precinct"]["TARRANT"]], index=[r[0] for r in D["precinct"]["TARRANT"]], columns=cols)
p["R"] = p.mR + p.eR; p["D"] = p.mD + p.eD; p["O"] = p.mO + p.eO; p["N"] = p.R + p.D + p.O
p.index = p.index.astype(str).str.zfill(4)

x = pd.read_excel(ROOT / "data/raw/_capitol_data_cache/precincts24g_districts.xlsx", dtype=str)
x = x[x.FIPS == "439"].set_index("PREC")
p["sd"] = x.PlanS2168.reindex(p.index)
p["hd"] = x.PlanH2316.reindex(p.index)
print("precincts", len(p), "unmapped", p.sd.isna().sum(), "EV votes unmapped", p.N[p.sd.isna()].sum(), "of", p.N.sum())

v = sr.load(2024).copy()
v = v[v.County.str.upper().str.strip() == "TARRANT"]
v["prec"] = v.cntyvtd.astype(str).str[-4:].str.zfill(4) if v.cntyvtd.astype(str).str.len().max() > 4 else v.cntyvtd.astype(str).str.zfill(4)
dr = v[v.Party.isin(["D", "R"])]
pres = dr[dr.Office.str.contains("President", case=False)].pivot_table(index="prec", columns="Party", values="Votes", aggfunc="sum")
jo = [o for o in dr.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)]
jud = dr[dr.Office.isin(jo)].pivot_table(index="prec", columns="Party", values="Votes", aggfunc="sum")
print("sample cntyvtd", v.cntyvtd.astype(str).head(3).tolist(), "pres precincts", len(pres), "match", p.index.isin(pres.index).sum())
p["pR"], p["pD"] = pres.R.reindex(p.index), pres.D.reindex(p.index)
p["jR"], p["jD"] = jud.R.reindex(p.index), jud.D.reindex(p.index)


def summ(g, lab):
    T = g.N.sum()
    ev = (g.R.sum() - g.D.sum()) / T * 100
    pm = (g.pR.sum() - g.pD.sum()) / (g.pR.sum() + g.pD.sum()) * 100
    jm = (g.jR.sum() - g.jD.sum()) / (g.jR.sum() + g.jD.sum()) * 100
    reg = (g.regR.sum() - g.regD.sum()) / (g.regR.sum() + g.regD.sum() + g.regO.sum()) * 100
    turn = T / (g.regR.sum() + g.regD.sum() + g.regO.sum()) * 100
    tR = g.R.sum() / g.regR.sum() * 100; tD = g.D.sum() / g.regD.sum() * 100; tO = g.O.sum() / g.regO.sum() * 100
    print(f"{lab:<14} EV {T:>8,.0f}  EVnet R{ev:+5.1f}  reg-net R{reg:+5.1f}  pres24 R{pm:+5.1f}  jud24 R{jm:+5.1f}  EV-pres {ev-pm:+5.1f}  turnout {turn:4.1f}%  R {tR:4.1f} D {tD:4.1f} O {tO:4.1f}")


m = p.dropna(subset=["sd", "pR"])
summ(m, "Tarrant all")
for s, g in m.groupby("sd"):
    summ(g, f"SD {s}")
summ(m[m.sd != "9"], "Tarrant not 9")

# precinct-level: EV net vs pres margin within Tarrant, SD9 residual
m = m[m.N >= 50].copy()
m["evn"] = (m.R - m.D) / m.N * 100
m["pm"] = (m.pR - m.pD) / (m.pR + m.pD) * 100
w = m.N
X = np.column_stack([np.ones(len(m)), m.pm])
b = np.linalg.lstsq(X * np.sqrt(w.values)[:, None], m.evn * np.sqrt(w), rcond=None)[0]
m["res"] = m.evn - X @ b
print("precinct fit EVnet = %.2f + %.3f * pres24" % tuple(b))
for s, g in m.groupby("sd"):
    print(f"SD {s}: weighted residual {np.average(g.res, weights=g["N"]):+.2f}  (n {len(g)})")
print("SD 9 by HD:")
for h, g in m[m.sd == "9"].groupby("hd"):
    print(f"  HD {h}: EV {g["N"].sum():>7,.0f}  EVnet R{(g.R.sum()-g.D.sum())/g["N"].sum()*100:+5.1f}  pres R{(g.pR.sum()-g.pD.sum())/(g.pR.sum()+g.pD.sum())*100:+5.1f}  resid {np.average(g.res, weights=g["N"]):+.2f}")

print("not-yet-voted pool (registered minus early, modeled party):")
for lab, g in (("SD 9", p[p.sd == "9"]), ("Tarrant not 9", p[p.sd.notna() & (p.sd != "9")]), ("Tarrant all", p)):
    r, d, o = g.regR.sum() - g.R.sum(), g.regD.sum() - g.D.sum(), g.regO.sum() - g.O.sum()
    print(f"  {lab:<14} R {r:>8,} D {d:>8,} O {o:>8,}  partisan net R{(r-d)/(r+d+o)*100:+.1f}")
