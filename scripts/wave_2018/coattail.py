"""Is the 2018 hump coattails, a local partisan swing, or a trend you could have seen coming?

For every contested House race, on the VTDs that voted in it: top-of-ticket margin, judicial
margin, race margin. Then, per cycle:
  pass-through   (race - J) on (top - J), with incumbency: do crossovers for the top carry down?
  residual       formula residual (held out, actual env) on (top - J), local J swing, and the
                 prior presidential trend, which was knowable before the election.
"""
import re
import sys

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\scripts")
sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import numpy as np
import pandas as pd
import statsmodels.api as sm

import whole_election_backtest as b

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
sw, rc, sr = b.sw, b.rc, b.sr
HIST = b.HIST
TOPS = {2012: "President", 2014: "U.S. Sen", 2016: "President", 2018: "U.S. Sen", 2020: "U.S. Sen", 2022: "Governor", 2024: "U.S. Sen"}
ST = {2012: 1, 2014: 1, 2016: 1, 2018: 1, 2020: 0, 2022: 0, 2024: 0}


def race_table(y):
    df = sr.load(y).copy()
    df["cntyvtd"] = df.cntyvtd.astype(str)
    dr = df[df.Party.isin(["D", "R"])]
    rep = dr[dr.Office.str.match(r"^State Rep \d+$")]
    w = rep.pivot_table(index=["Office", "cntyvtd"], columns="Party", values="Votes", aggfunc="sum").fillna(0)
    t = dr[dr.Office == TOPS[y]].pivot_table(index="cntyvtd", columns="Party", values="Votes", aggfunc="sum").fillna(0)
    jo = [o for o in dr.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)]
    jw = dr[dr.Office.isin(jo)].pivot_table(index=["Office", "cntyvtd"], columns="Party", values="Votes", aggfunc="sum")
    both = jw.groupby("Office").sum()
    both = both[(both.D > 0) & (both.R > 0)].index
    j = jw[jw.index.get_level_values(0).isin(both)].groupby("cntyvtd").sum()
    rows = []
    for o, g in w.groupby(level=0):
        if g.D.sum() == 0 or g.R.sum() == 0:
            continue
        v = g.index.get_level_values(1)
        T, J = t.reindex(v).fillna(0), j.reindex(v).fillna(0)
        rows.append({"year": y, "district": int(o.split()[-1]), "m_race": (g.D.sum() - g.R.sum()) / (g.D.sum() + g.R.sum()) * 100,
                     "m_top": (T.D.sum() - T.R.sum()) / (T.D.sum() + T.R.sum()) * 100, "m_j": (J.D.sum() - J.R.sum()) / (J.D.sum() + J.R.sum()) * 100,
                     "county": df[df.cntyvtd.isin(v)].County.value_counts().index[0]})
    return pd.DataFrame(rows)


print("=== Pass-through of top-of-ticket crossover to the House race, contested seats |race|<40 ===")
tabs = {}
out = []
for y in TOPS:
    t = race_table(y)
    tabs[y] = t
    k = t[t.m_race.abs() < 40].copy()
    k["gtop"], k["gh"] = k.m_top - k.m_j, k.m_race - k.m_j
    f = sm.OLS(k.gh, sm.add_constant(k[["gtop"]])).fit()
    out.append({"year": y, "straight": ST[y], "top": TOPS[y], "n": len(k), "mean_top_minus_J": k.gtop.mean(), "sd_top_minus_J": k.gtop.std(), "mean_race_minus_J": k.gh.mean(), "sd_race_minus_J": k.gh.std(),
                "pass": f.params["gtop"], "se": f.bse["gtop"], "r2": f.rsquared})
print(pd.DataFrame(out).round(2).to_string(index=False))

# the prior presidential trend, knowable before the election
pres = {}
for name, f in (("2012", "tx_presidential_house_2012_planh358.csv"), ("2016", "tx_presidential_house_2016_planh358.csv"), ("2020", "tx_presidential_house_2020_planh2316.csv")):
    v = pd.read_csv(HIST / f)
    pres[name] = v.set_index("district").assign(m=lambda q: (q.dem_votes - q.rep_votes) / (q.dem_votes + q.rep_votes) * 100).m
trend18 = pres["2016"] - pres["2012"]

print("\n=== Formula residual on what might explain it, House contested, |formula|<30 ===")
for y in (2018, 2022, 2024):
    c = b.leg_fit(y)
    a = b.leg_frame(y)
    a = a[(a.chamber_lower == "house") & a.contested & a.x.notna()].copy()
    a["lin"] = (rc.predict(a, c) + b.LEG_SLOPE * (sw.NAT[y] - b.ENV_BASE[y])) * 200 - 100
    a["district"] = a.district.astype(int)
    m = a.merge(tabs[y], on="district")
    m["res"] = m.m_race - m.lin
    m["gtop"] = m.m_top - m.m_j
    prev = {2018: 2016, 2022: 2018, 2024: 2022}[y]
    if y == 2018:
        pv = tabs[2016].set_index("district")
        m["dJ_local"] = m.m_j - m.district.map(pv.m_j)
        m["trend"] = m.district.map(trend18)
    else:
        m["dJ_local"] = np.nan
        m["trend"] = np.nan
    k = m[m.lin.abs() < 30].copy()
    print(f"\n--- {y}: n {len(k)}, resid mean {k.res.mean():+.2f} sd {k.res.std():.2f}; R-held n {int((k.rep_incumbent == 1).sum())} mean {k.res[k.rep_incumbent == 1].mean():+.2f}")
    for term in ("gtop", "dJ_local", "trend", "c_white_col"):
        z = k.dropna(subset=[term])
        if len(z) < 10:
            continue
        ctr = z[term] - z[term].mean()
        f = sm.OLS(z.res, sm.add_constant(ctr.rename(term))).fit()
        print(f"  res on {term:12s} coef {f.params[term]:+.3f} se {f.bse[term]:.3f} t {f.tvalues[term]:+.2f} r2 {f.rsquared:.3f}  sd(term) {z[term].std():.2f}")
    if y == 2018:
        print(k.sort_values("lin")[k.rep_incumbent == 1][["district", "county", "lin", "m_race", "res", "m_j", "m_top", "gtop", "dJ_local", "trend", "c_white_col"]].round(1).to_string(index=False))
        print(k.groupby("county").agg(n=("res", "size"), res=("res", "mean"), gtop=("gtop", "mean")).query("n>=2").round(1).sort_values("res").to_string())
        print("level of local J vs formula (m_j - lin), close R-held:", (k.m_j - k.lin)[k.rep_incumbent == 1].mean().round(2), " race minus J:", (k.m_race - k.m_j)[k.rep_incumbent == 1].mean().round(2))
