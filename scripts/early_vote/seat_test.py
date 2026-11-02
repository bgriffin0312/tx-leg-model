"""Does a county early signal explain the House model's own seat misses?

Residual = actual margin minus simulated mean with T fixed at the actual judicial vote (held-out
formula), so the statewide level is out. Signal = county early change, averaged over each district's
counties by J votes, relative to the state. Border counties are flagged because party labels move
most there.

2024: early-net change 2022-24 (all counties), and early turnout growth 2022-24.
2022: early turnout growth 2018-22 (top 30 counties, SOS 2018 table).
2018: early turnout growth 2014-18 (top 15 counties, SOS 2014 table).
A district is scored only if at least 80% of its J votes sit in covered counties.
"""
import io
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import statsmodels.api as sm

import whole_election_backtest as b

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 40)
e = pd.read_csv(S / "county_panel2.csv", index_col=0)
sr = b.sr
BORDER = ["HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"]


def hd_county(y):
    df = sr.load(y).copy()
    df["cntyvtd"] = df.cntyvtd.astype(str)
    df["c"] = df.County.str.upper().str.strip()
    dr = df[df.Party.isin(["D", "R"])]
    rep = dr[dr.Office.str.match(r"^State Rep \d+$")]
    hd = rep.assign(hd=rep.Office.str.extract(r"(\d+)$")[0].astype(int)).groupby(["cntyvtd", "hd"]).Votes.sum().reset_index().sort_values("Votes").drop_duplicates("cntyvtd", keep="last").set_index("cntyvtd").hd
    if y in b.PRECINCTS:
        pf, col = b.PRECINCTS[y]
        pm = pd.read_excel(ROOT / "data" / "raw" / "_capitol_data_cache" / pf, dtype=str)
        pm = pd.to_numeric(pm.set_index("PCTKEY")[col], errors="coerce").dropna().astype(int)
        pm = pm[(pm >= 1) & (pm <= 150)]
        pm = pm[~pm.index.duplicated()]
        pm.index = pm.index.astype(str)
        hd.index = hd.index.astype(str)
        hd = pm.combine_first(hd).astype(int)
    jo = [o for o in dr.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)]
    j = dr[dr.Office.isin(jo)].groupby(["cntyvtd", "c"]).Votes.sum().reset_index()
    j["hd"] = j.cntyvtd.map(hd)
    w = j.dropna(subset=["hd"]).groupby(["hd", "c"]).Votes.sum().reset_index()
    w["hd"] = w.hd.astype(int)
    w["w"] = w.Votes / w.groupby("hd").Votes.transform("sum")
    return w[["hd", "c", "w"]]


def sos(y):
    f = next((S / "sos").glob(f"{y}_*.html"))
    t = max(pd.read_html(io.StringIO(f.read_text(encoding="utf-8"))), key=lambda q: q.shape[0])
    t.columns = [re.sub(r"\s+", " ", str(c)) for c in t.columns]
    cum = [c for c in t.columns if "In-Person And Mail" in c][0]
    t = t[t.County.str.lower() != "total"].assign(c=lambda d: d.County.str.upper().str.strip()).set_index("c")
    return (t[cum].astype(float) / t["Reg Voters"].astype(float))


def to_hd(sig, w):
    """County signal to district, with coverage."""
    s = sig.dropna()
    t = w.assign(v=w.c.map(s))
    g = t.dropna(subset=["v"]).groupby("hd")
    out = pd.DataFrame({"sig": g.apply(lambda q: np.average(q.v, weights=q.w), include_groups=False), "cov": g.w.sum()})
    b_ = w.assign(b=w.c.isin(BORDER)).groupby("hd").apply(lambda q: (q.w * q.b).sum(), include_groups=False)
    out["border"] = b_
    return out


SIG = {}
g1418 = np.log(sos(2018) / sos(2014)).dropna()
r22 = (e.t2022 / e.nreg22)
g1822 = np.log(r22 / sos(2018)).dropna()
SIG[2018] = {"turnout": (g1418, e.n2018.reindex(g1418.index))}
SIG[2022] = {"turnout": (g1822, e.n2022.reindex(g1822.index))}
SIG[2024] = {"early net": (-(e.E2024 - e.E2022), e.n2024), "turnout": (np.log(e.t2024 / e.t2022), e.n2024)}

envs = {y: (b.GB.final_538[y] + b.GB.final_rcp[y]) / 2 for y in b.YEARS}
rows = []
store = {}
for y in b.YEARS:
    w = hd_county(y)
    x = b.Cycle(y, envs[y])
    r = x.run("actual_T")
    m = r["kind"] == "state_house"
    mean = r["M"][:, m].mean(0)
    act = r["actual"][m]
    te = x.te[x.is_h]
    ids = te.district.astype(int).values
    d = pd.DataFrame({"hd": ids, "resid": act - mean, "mean": mean, "act": act, "rinc": te.rep_incumbent.astype(float).values, "dinc": te.dem_incumbent.astype(float).values}).set_index("hd")
    for name, (sig, wt) in SIG[y].items():
        k = sig.dropna().index.intersection(wt.dropna().index)
        c = sig[k] - np.average(sig[k], weights=wt[k])
        h = to_hd(c, w)
        q = d.join(h, how="inner")
        q = q[(q["cov"] >= .8) & (q["mean"].abs() < 40)]
        store[(y, name)] = q
        for lab, kk in (("all", q.index), ("no border", q.index[q.border < .2]), ("R-held close", q.index[(q.rinc == 1) & (q["mean"].abs() < 30) & (q.border < .2)]), ("border", q.index[q.border >= .2])):
            t = q.loc[kk]
            if len(t) < 6:
                continue
            f = sm.OLS(t.resid, sm.add_constant(t[["sig"]])).fit()
            # leave one district out; counties are shared, so this flatters a little
            er0, er1 = [], []
            for i in t.index:
                tr = t.drop(i)
                ff = sm.OLS(tr.resid, sm.add_constant(tr[["sig"]])).fit()
                er1.append(t.resid[i] - ff.params["const"] - ff.params["sig"] * t.sig[i])
                er0.append(t.resid[i] - tr.resid.mean())
            rows.append({"year": y, "signal": name, "group": lab, "n": len(t), "sd_sig": t.sig.std(), "slope": f.params["sig"], "se": f.bse["sig"], "r2": f.rsquared, "rmse0": float(np.sqrt(np.mean(np.square(er0)))), "rmse1": float(np.sqrt(np.mean(np.square(er1))))})
    del x, r
out = pd.DataFrame(rows)
out["gain_pct"] = 100 * (1 - out.rmse1 / out.rmse0)
print(out.round(3).to_string(index=False))
pd.to_pickle(store, S / "hd_store.pkl")

print("\nDistrict-county clustered slope (counties are the real unit)")
for (y, name), q in store.items():
    q = q[q.border < .2]
    # dominant county cluster
    w = hd_county(y)
    dom = w.sort_values("w").drop_duplicates("hd", keep="last").set_index("hd").c
    c = q.index.map(dom)
    f = sm.OLS(q.resid, sm.add_constant(q[["sig"]])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(c)[0]})
    print(y, name, "n", len(q), "slope", round(f.params["sig"], 2), "cluster se", round(f.bse["sig"], 2), "clusters", len(set(c)))
