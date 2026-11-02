"""Does the place a swing lands scale with the size of the swing?

Local judicial margin on each House district's own VTDs, consecutive cycles, same map. Within a
pair, regress the district's J swing (statewide mean removed) on centered white-college share
and centered Hispanic share. Then ask whether those loadings move with the statewide J swing
across pairs, or are a steady trend.

Pairs on PlanH358: 2012-14, 14-16, 16-18, 18-20. On PlanH2316: 2022-24, and 2020-22 through the
2020 open composite, which is on the 2022 lines (composite, not J; mean removed so level is out).
"""
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\scripts")
sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import whole_election_backtest as b

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
sw, rc, sr, ge = b.sw, b.rc, b.sr, b.ge
SCR = Path(__file__).resolve().parent


def j_by_hd(y):
    df = sr.load(y).copy()
    df["cntyvtd"] = df.cntyvtd.astype(str)
    dr = df[df.Party.isin(["D", "R"])]
    rep = dr[dr.Office.str.match(r"^State Rep \d+$")]
    jo = [o for o in dr.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)]
    jw = dr[dr.Office.isin(jo)].pivot_table(index=["Office", "cntyvtd"], columns="Party", values="Votes", aggfunc="sum")
    both = jw.groupby("Office").sum()
    both = both[(both.D > 0) & (both.R > 0)].index
    j = jw[jw.index.get_level_values(0).isin(both)].groupby("cntyvtd").sum()
    # any State Rep row places the VTD, contested or not; one-party seats still have rows
    hd = rep.assign(hd=rep.Office.str.extract(r"(\d+)$")[0].astype(int)).groupby(["cntyvtd", "hd"]).Votes.sum().reset_index().sort_values("Votes").drop_duplicates("cntyvtd", keep="last").set_index("cntyvtd").hd
    if y in b.PRECINCTS:
        pf, col = b.PRECINCTS[y]
        pm = pd.read_excel(b.ROOT / "data" / "raw" / "_capitol_data_cache" / pf, dtype=str)
        pm = pd.to_numeric(pm.set_index("PCTKEY")[col], errors="coerce").dropna().astype(int)
        pm = pm[(pm >= 1) & (pm <= 150)]
        pm = pm[~pm.index.duplicated()]
        pm.index = pm.index.astype(str)
        hd.index = hd.index.astype(str)
        hd = pm.combine_first(hd).astype(int)
    j["hd"] = j.index.map(hd)
    s = j.dropna(subset=["hd"]).groupby("hd")[["D", "R"]].sum()
    return ((s.D - s.R) / (s.D + s.R) * 100).rename(y), float((j.D.sum() - j.R.sum()) / (j.D.sum() + j.R.sum()) * 100), s.D + s.R


J, Jst, V = {}, {}, {}
for y in (2012, 2014, 2016, 2018, 2020, 2022, 2024):
    J[y], Jst[y], V[y] = j_by_hd(y)
print("districts with a J reading:", {y: len(J[y]) for y in J}, "statewide J:", {k: round(v, 1) for k, v in Jst.items()})

ge_w = ge.texas_weights()
a18 = b.leg_frame(2018)
a22 = b.leg_frame(2022)
d18 = a18[a18.chamber_lower == "house"].set_index(a18[a18.chamber_lower == "house"].district.astype(int))
d22 = a22[a22.chamber_lower == "house"].set_index(a22[a22.chamber_lower == "house"].district.astype(int))
print("demographic coverage", d18.c_white_col.notna().sum(), d22.c_white_col.notna().sum())
print("all-seat frames", len(d18), len(d22))

base20 = pd.read_csv(b.HIST / "tx_downballot_house_2020_planh2316.csv").set_index("district").open_mean_d2p * 200 - 100

PAIRS = [(2012, 2014, d18), (2014, 2016, d18), (2016, 2018, d18), (2018, 2020, d18), (2020, 2022, d22), (2022, 2024, d22)]
rows = []
for a, c, dem in PAIRS:
    s0 = base20 if (a, c) == (2020, 2022) else J[a]
    sw_ = (J[c] - s0).dropna()
    sw_ = sw_[sw_.index.isin(dem.index)]
    k = pd.DataFrame({"s": sw_, "wc": dem.c_white_col.reindex(sw_.index), "his": dem.c_hispanic.reindex(sw_.index), "blk": dem.c_black.reindex(sw_.index), "wt": V[c].reindex(sw_.index)}).dropna()
    k["s"] = k.s - np.average(k.s, weights=k.wt)
    for n in ("wc", "his", "blk"):
        k[n] = k[n] - np.average(k[n], weights=k.wt)
    f = sm.WLS(k.s, sm.add_constant(k[["wc", "his", "blk"]]), weights=k.wt).fit()
    dJ = Jst[c] - Jst[a]
    rows.append({"pair": f"{a}-{c}", "n": len(k), "dJ_state": dJ, "dNat": sw.NAT[c] - sw.NAT[a] if c in sw.NAT and a in sw.NAT else np.nan, "wc": f.params["wc"], "wc_se": f.bse["wc"], "his": f.params["his"], "his_se": f.bse["his"], "blk": f.params["blk"], "r2": f.rsquared, "sd_swing": k.s.std()})
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))
t.to_pickle(SCR / "j_panel.pkl")

print("\nloading on statewide J swing, across pairs (all, and without each pair)")
for term in ("wc", "his"):
    f = sm.OLS(t[term], sm.add_constant(t[["dJ_state"]])).fit()
    print(f"  {term}: const {f.params['const']:+.1f} (se {f.bse['const']:.1f}), per J point {f.params['dJ_state']:+.2f} (se {f.bse['dJ_state']:.2f}), r2 {f.rsquared:.2f}")
    for i in range(len(t)):
        g = sm.OLS(t[term].drop(i), sm.add_constant(t[["dJ_state"]].drop(i))).fit()
        print(f"      without {t.pair[i]}: const {g.params['const']:+.1f}, per J point {g.params['dJ_state']:+.2f}")
pickle.dump({"J": J, "Jst": Jst}, open(SCR / "j_hd.pkl", "wb"))
