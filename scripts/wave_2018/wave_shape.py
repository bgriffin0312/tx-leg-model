"""Two shape explanations for 2018, each scored held out on every midterm plus 2024.

  linear    shipped: env moves share by 0.0034 per point everywhere
  logit     same formula fit on the logit of share; env moves logit by 0.0034/0.25 per point,
            so the same average push lands hardest near 50%. No new fitted term.

Also per-cycle incumbency, and residual-by-band under each. Seats use Phi(lin/race sd) so no
Monte Carlo noise, with the held-out environment at the actual national number.
"""
import sys

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\scripts")
sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import whole_election_backtest as b

pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 40)
sw, rc, met = b.sw, b.rc, b.met
TERMS = ["x", "chamber_senate", "dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_c", "c_white_col"]
LOGIT_SLOPE = b.LEG_SLOPE / 0.25
SD = 8.8  # lg.SIGMA_IDIO 0.044 in margin


def lg(p):
    p = np.clip(p, .02, .98)
    return np.log(p / (1 - p))


def il(z):
    return 1 / (1 + np.exp(-z))


d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
d["dE"] = d.national_env - d.year.map(b.ENV_BASE)
print("training rows", d.groupby("year").size().to_dict(), "dE", d.groupby("year").dE.first().round(1).to_dict())


def fit(tr, mode):
    X = sm.add_constant(tr[TERMS].astype(float))
    if mode == "linear":
        return sm.OLS(tr.dem_2p_share - b.LEG_SLOPE * tr.dE, X).fit()
    Xl = X.copy()
    Xl["x"] = lg(tr.x)
    return sm.OLS(lg(tr.dem_2p_share) - LOGIT_SLOPE * tr.dE, Xl).fit()


def pred(f, te, dE, mode):
    X = sm.add_constant(te[TERMS].astype(float), has_constant="add")
    if mode == "linear":
        return f.predict(X).values + b.LEG_SLOPE * dE
    Xl = X.copy()
    Xl["x"] = lg(te.x)
    return il(f.predict(Xl).values + LOGIT_SLOPE * dE)


rows, bands = [], []
for y in (2014, 2018, 2022, 2024):
    tr = d[d.year != y]
    if y == 2024:
        a = b.leg_frame(2024)
        a = a[a.contested & a.x.notna()].copy()
        te = a[(a.dem_2p_share > 0.001) & (a.dem_2p_share < .999)]
    else:
        te = d[d.year == y].copy()
    h = (te.chamber_lower == "house").values
    dE = sw.NAT[y] - b.ENV_BASE[y]
    for mode in ("linear", "logit"):
        f = fit(tr, mode)
        p = pred(f, te, dE, mode)
        m = p * 200 - 100
        act = te.dem_2p_share.values * 200 - 100
        res = act - m
        close = h & (np.abs(m) < 20)
        rows.append({"year": y, "mode": mode, "n": int(h.sum()), "rmse": float(np.sqrt((res[h] ** 2).mean())), "rmse_close": float(np.sqrt((res[close] ** 2).mean())), "miss_close": float(res[close].mean()),
                     "miss_Rheld_close": float(res[close & (te.rep_incumbent.values == 1)].mean()), "seats_sign": int((m[h] > 0).sum()), "seats_prob": float(norm.cdf(m[h] / SD).sum()), "actual": int((act[h] > 0).sum()),
                     "brier": float(((norm.cdf(m[h] / SD) - (act[h] > 0)) ** 2).mean())})
        t = pd.DataFrame({"m": m[h], "res": res[h]})
        t["band"] = pd.cut(t.m, [-100, -30, -20, -12, -6, 0, 6, 12, 20, 100])
        g = t.groupby("band", observed=True).res.mean().round(1)
        bands.append(pd.Series(g, name=f"{y} {mode}"))
print(pd.DataFrame(rows).round(2).to_string(index=False))
print(pd.concat(bands, axis=1).to_string())

print("\nincumbency by cycle (house only, fit within cycle, margin points), with the rest of the formula free")
out = []
for y in (2014, 2018, 2022):
    g = d[(d.year == y) & (d.chamber_lower == "house")]
    f = sm.OLS(g.dem_2p_share, sm.add_constant(g[["x", "dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_c", "c_white_col"]].astype(float))).fit()
    out.append({"year": y, "n": len(g), **{k: f.params[k] * 200 for k in ("dem_incumbent", "rep_incumbent")}, "se_d": f.bse["dem_incumbent"] * 200, "se_r": f.bse["rep_incumbent"] * 200, "x": f.params["x"]})
a = b.leg_frame(2024)
a = a[(a.chamber_lower == "house") & a.contested & a.x.notna() & (a.dem_2p_share.between(.001, .999))]
f = sm.OLS(a.dem_2p_share, sm.add_constant(a[["x", "dem_incumbent", "rep_incumbent", "c_white_col"]].astype(float))).fit()
out.append({"year": 2024, "n": len(a), **{k: f.params[k] * 200 for k in ("dem_incumbent", "rep_incumbent")}, "se_d": f.bse["dem_incumbent"] * 200, "se_r": f.bse["rep_incumbent"] * 200, "x": f.params["x"]})
print(pd.DataFrame(out).round(2).to_string(index=False))
