"""Are the seats that decide the chamber more elastic than the state?

For each pair: the local J swing in bands of prior J margin, net of the steady white-college and
race loadings, against the statewide swing. Elastic bands have |band swing| above state swing in
pairs of either sign, so the test is sign-symmetric: a wave-only hump would show up in D-up pairs
only. Then the same question in the legislative rows: slope by who holds the seat.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\scripts")
sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import whole_election_backtest as b

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
sw, rc, met = b.sw, b.rc, b.met
SCR = Path(__file__).resolve().parent
P = pickle.load(open(SCR / "j_hd.pkl", "rb"))
J, Jst = P["J"], P["Jst"]

a18, a22 = b.leg_frame(2018), b.leg_frame(2022)
d18 = a18[a18.chamber_lower == "house"].set_index(a18[a18.chamber_lower == "house"].district.astype(int))
d22 = a22[a22.chamber_lower == "house"].set_index(a22[a22.chamber_lower == "house"].district.astype(int))
base20 = pd.read_csv(b.HIST / "tx_downballot_house_2020_planh2316.csv").set_index("district").open_mean_d2p * 200 - 100
PAIRS = [(2012, 2014, d18), (2014, 2016, d18), (2016, 2018, d18), (2018, 2020, d18), (2020, 2022, d22), (2022, 2024, d22)]
EDGES = [-100, -30, -15, -5, 5, 15, 100]
rows, fits = [], []
for a, c, dem in PAIRS:
    s0 = base20 if (a, c) == (2020, 2022) else J[a]
    k = pd.DataFrame({"s0": s0, "s1": J[c]}).dropna()
    k = k[k.index.isin(dem.index)]
    k["wc"], k["his"], k["blk"] = dem.c_white_col.reindex(k.index), dem.c_hispanic.reindex(k.index), dem.c_black.reindex(k.index)
    k = k.dropna()
    k["sw"] = k.s1 - k.s0
    dJ = Jst[c] - Jst[a]
    X = sm.add_constant(k[["s0", "wc", "his", "blk"]])
    f = sm.OLS(k.sw, X).fit()
    fits.append({"pair": f"{a}-{c}", "dJ": dJ, "b_prior": f.params["s0"], "se": f.bse["s0"], "b_prior_nocontrols": sm.OLS(k.sw, sm.add_constant(k[["s0"]])).fit().params["s0"]})
    # net of demographic loadings, keep prior margin free
    k["net"] = k.sw - f.params["wc"] * k.wc - f.params["his"] * k.his - f.params["blk"] * k.blk
    k["band"] = pd.cut(k.s0, EDGES)
    g = k.groupby("band", observed=False).agg(n=("sw", "size"), raw=("sw", "mean"), net=("net", "mean"))
    rows.append(pd.Series({f"{i}": f"{r.raw:+.1f} ({int(r.n)})" for i, r in g.iterrows()}, name=f"{a}-{c} dJ {dJ:+.1f}"))
print("raw mean J swing by band of prior J margin (n)")
print(pd.DataFrame(rows).to_string())
ft = pd.DataFrame(fits)
print(ft.round(3).to_string(index=False))
f = sm.OLS(ft.b_prior, sm.add_constant(ft[["dJ"]])).fit()
print(f"loading on prior margin vs statewide swing: const {f.params['const']:+.3f}, slope {f.params['dJ']:+.4f} (se {f.bse['dJ']:.4f}), r2 {f.rsquared:.2f}")

print("\n=== Legislative rows: environment slope by who holds the seat, margin per national point ===")
d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
d["dE"] = d.national_env - d.year.map(b.ENV_BASE)
d["dem_inc"], d["rep_inc"] = d.dem_incumbent.astype(float), d.rep_incumbent.astype(float)
d["open"] = 1 - d.dem_inc - d.rep_inc
a24 = b.leg_frame(2024)
a24 = a24[a24.contested & a24.x.notna() & a24.dem_2p_share.between(.001, .999)].copy()
a24["dE"] = sw.NAT[2024] - 3.1
a24["dem_inc"], a24["rep_inc"], a24["open"] = a24.dem_incumbent.astype(float), a24.rep_incumbent.astype(float), 1 - a24.dem_incumbent.astype(float) - a24.rep_incumbent.astype(float)
OTHER = ["x", "chamber_senate", "challenger_viability_flag", "ie_c", "c_white_col", "dem_inc", "rep_inc"]


def spec(rows, label):
    r = rows.copy()
    for s in ("dem_inc", "rep_inc", "open"):
        r[f"dE_{s}"] = r.dE * r[s]
    X = sm.add_constant(r[OTHER + ["dE_dem_inc", "dE_rep_inc", "dE_open"]].astype(float))
    f = sm.OLS(r.dem_2p_share, X).fit(cov_type="cluster", cov_kwds={"groups": r.year})
    g = sm.OLS(r.dem_2p_share, X).fit()
    print(f"\n{label}: n {len(r)}, by year {r.groupby('year').size().to_dict()}")
    for k in ("dE_dem_inc", "dE_rep_inc", "dE_open"):
        print(f"  {k:12s} {g.params[k] * 200:+.3f} margin per point (se {g.bse[k] * 200:.3f})")
    print(f"  dem_inc cushion {g.params['dem_inc'] * 200:+.2f}, rep_inc {g.params['rep_inc'] * 200:+.2f}, const {g.params['const']:.4f}, x {g.params['x']:.3f}")
    # close seats only
    return g


spec(d[d.chamber_lower == "house"], "House midterms")
spec(pd.concat([d[d.chamber_lower == "house"], a24[a24.chamber_lower == "house"]]), "House midterms + 2024")
spec(d, "House + Senate midterms")
print("\nclose seats only |x margin|<30")
c_ = d[(d.chamber_lower == "house") & ((d.x * 200 - 100).abs() < 30)]
spec(c_, "House midterms close")
c2 = pd.concat([c_, a24[(a24.chamber_lower == "house") & ((a24.x * 200 - 100).abs() < 30)]])
spec(c2, "House midterms + 2024 close")
