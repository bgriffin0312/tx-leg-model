"""County tests of the early vote, all with two out-of-sample directions where the data allow.

Outcome: county margin (D minus R) relative to the state, on the judicial composite and the top race.
A. Swing test, 2022 to 2024: change in early net against margin change.
B. Levels test: this year's margin on last result plus this year's early net, needs no prior early
   vote. Fit 2022, test 2024, and the reverse. Prior = last same-type result and last result.
C. Registered-net version: early net minus registered net in the same file, 2022 only for testing.
D. Turnout: relative growth in early votes, no party labels at all. 2022 to 2024 full county set,
   then SOS top counties for 2012-16, 2014-18, 2018-22.
All weighted by votes; large counties and no-border cuts reported separately because a pooled number
can hide a group that fails.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
e = pd.read_csv(S / "county_panel.csv", index_col=0)
pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 40)
BORDER = ["HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"]
top60 = e.sort_values("n2024").tail(60).index
top30 = e.sort_values("n2024").tail(30).index
GROUPS = {"all": e.index, "top 60": top60, "top 60 no border": top60.difference(BORDER), "top 30": top30, "rest (outside top 60)": e.index.difference(top60), "border": e.index.intersection(BORDER)}


def rel(s, w):
    return s - np.average(s, weights=w)


def wfit(y, X, w):
    return sm.WLS(y, sm.add_constant(X), weights=w).fit()


def wrmse(r, w):
    return float(np.sqrt(np.average(np.square(r), weights=w)))


print("=== A. Swing 2022 to 2024, leave one county out ===")
for out in ("j", "top"):
    e["dm"] = e[f"{out}2024"] - e[f"{out}2022"]
    e["dE"] = e.E2024 - e.E2022
    rows = []
    for g, idx in GROUPS.items():
        x = e.loc[idx]
        w = x.n2024.values
        f = wfit(x.dm, x[["dE"]], w)
        e0, e1 = [], []
        for c in x.index:
            tr = x.drop(c)
            ft = wfit(tr.dm, tr[["dE"]], tr.n2024)
            e1.append(x.dm[c] - ft.params["const"] - ft.params["dE"] * x.dE[c])
            e0.append(x.dm[c] - np.average(tr.dm, weights=tr.n2024))
        rows.append({"group": g, "n": len(x), "slope": f.params["dE"], "se": f.bse["dE"], "r2": f.rsquared, "uniform": wrmse(e0, w), "early": wrmse(e1, w), "gain_pct": 100 * (1 - wrmse(e1, w) / wrmse(e0, w))})
    print(out)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== B. Levels: margin on prior margin, then add this year's early net; fit one year, test the other ===")
PAIRS = {"midterm prior": {2022: 2018, 2024: 2020}, "last result": {2022: 2020, 2024: 2022}}
out_rows = []
for out in ("j", "top"):
    for pname, pr in PAIRS.items():
        for tr_y, te_y in ((2022, 2024), (2024, 2022)):
            for g, idx in GROUPS.items():
                x = e.loc[idx]
                a = x.assign(y=x[f"{out}{tr_y}"], p=x[f"{out}{pr[tr_y]}"], ev=-x[f"E{tr_y}"], w=x[f"n{tr_y}"])
                b = x.assign(y=x[f"{out}{te_y}"], p=x[f"{out}{pr[te_y]}"], ev=-x[f"E{te_y}"], w=x[f"n{te_y}"])
                f0 = wfit(a.y, a[["p"]], a.w)
                f1 = wfit(a.y, a[["p", "ev"]], a.w)
                # score county position against the state, so the statewide level does not count
                r0 = rel(b.y, b.w) - rel(f0.predict(sm.add_constant(b[["p"]])), b.w)
                r1 = rel(b.y, b.w) - rel(f1.predict(sm.add_constant(b[["p", "ev"]])), b.w)
                out_rows.append({"out": out, "prior": pname, "fit": tr_y, "test": te_y, "group": g, "n": len(x), "b_prior": f1.params["p"], "b_early": f1.params["ev"], "se": f1.bse["ev"], "rmse_prior": wrmse(r0, b.w), "rmse_early": wrmse(r1, b.w)})
B = pd.DataFrame(out_rows)
B["gain_pct"] = 100 * (1 - B.rmse_early / B.rmse_prior)
print(B[B.group.isin(["all", "top 60 no border", "top 30", "rest (outside top 60)", "border"])].round(2).to_string(index=False))

print("\n=== C. Early net minus registered net, same file, 2020 to 2022 only ===")
e["gap22"] = -(e.E2022 - e.reg22)
e["dm"] = rel(e.j2022 - e.j2020, e.n2022)
for g, idx in GROUPS.items():
    x = e.loc[idx]
    f = wfit(x.dm, x[["gap22"]], x.n2022)
    print(f"{g:<24} n {len(x):>3} slope {f.params['gap22']:+.3f} se {f.bse['gap22']:.3f} r2 {f.rsquared:.3f}")

print("\n=== D1. Turnout only: relative early-vote growth, 2022 to 2024 ===")
e["g2224"] = np.log(e.t2024 / e.t2022)
for out in ("j", "top"):
    e["dm"] = e[f"{out}2024"] - e[f"{out}2022"]
    for g, idx in GROUPS.items():
        x = e.loc[idx]
        f = wfit(x.dm, x[["g2224"]], x.n2024)
        print(f"{out:<4}{g:<24} n {len(x):>3} slope {f.params['g2224']:+.2f} se {f.bse['g2224']:.2f} r2 {f.rsquared:.3f}")

print("\n=== D2. SOS top-county tables: early votes per registered voter, change vs margin change ===")
tabs = {}
for f in sorted((S / "sos").glob("*.html")):
    y = int(f.name[:4])
    t = max(pd.read_html(io.StringIO(f.read_text(encoding="utf-8"))), key=lambda q: q.shape[0])
    t.columns = [re.sub(r"\s+", " ", str(c)) for c in t.columns]
    cum = [c for c in t.columns if "In-Person And Mail" in c][0]
    t = t[t.County.str.lower() != "total"].assign(c=lambda d: d.County.str.upper().str.strip())
    tabs[y] = t.set_index("c")[[cum, "Reg Voters"]].astype(float).set_axis(["ev", "reg"], axis=1)
tabs[2022] = e[["t2022", "nreg22"]].set_axis(["ev", "reg"], axis=1)
Dpanel = []
for a_, b_, lab in ((2012, 2016, "pres"), (2014, 2018, "mid"), (2018, 2022, "mid")):
    k = tabs[a_].index.intersection(tabs[b_].index)
    p, q = tabs[a_].loc[k], tabs[b_].loc[k]
    g = np.log((q.ev / q.reg) / (p.ev / p.reg))
    for out in ("j", "top"):
        dm = (e.loc[k, f"{out}{b_}"] - e.loc[k, f"{out}{a_}"])
        w = e.loc[k, f"n{b_}"]
        f = wfit(rel(dm, w), rel(g, w).rename("g"), w)
        print(f"{a_}-{b_} {out:<4} n {len(k)} slope {f.params['g']:+.2f} se {f.bse['g']:.2f} r2 {f.rsquared:.2f}  corr {np.corrcoef(rel(dm, w), rel(g, w))[0, 1]:+.2f}")
    Dpanel.append(pd.DataFrame({"pair": f"{a_}-{b_}", "g": rel(g, e.loc[k, f'n{b_}']), "dm": rel(e.loc[k, f"j{b_}"] - e.loc[k, f"j{a_}"], e.loc[k, f"n{b_}"])}))
pool = pd.concat(Dpanel)
f = sm.WLS(pool.dm, pool[["g"]]).fit()
print("pooled three pairs, j, no intercept (both sides relative):", round(f.params["g"], 2), "se", round(f.bse["g"], 2), "n", len(pool))

print("\n=== E. Is the 2022-24 early-net signal momentum? Control for the prior swing ===")
for out in ("j", "top"):
    e["dm"] = e[f"{out}2024"] - e[f"{out}2022"]
    e["lag2"] = e[f"{out}2022"] - e[f"{out}2020"]
    e["lag4"] = e[f"{out}2022"] - e[f"{out}2018"]
    e["dE"] = e.E2024 - e.E2022
    for g_, idx in GROUPS.items():
        if g_ not in ("all", "top 60 no border", "top 30"):
            continue
        x = e.loc[idx]
        for ctl in (["lag2"], ["lag4"], ["lag2", "g2224"]):
            f = wfit(x.dm, x[["dE"] + ctl], x.n2024)
            print(f"{out:<4}{g_:<20} dE {f.params['dE']:+.2f} (se {f.bse['dE']:.2f}) " + " ".join(f"{c} {f.params[c]:+.2f} (se {f.bse[c]:.2f})" for c in ctl) + f"  r2 {f.rsquared:.2f}")

print("\n=== F. 2026 signals across the top 30 ===")
e["d22"] = e.E2026 - e.E2022
e["d24"] = e.E2026 - e.E2024
e["dreg"] = e.reg26 - e.reg22
e["adj"] = e.d22 - e.dreg
e["g2226"] = np.log((e.t2026 / e.nreg26) / (e.t2022 / e.nreg22))
e["g2426"] = np.log(e.t2026 / e.t2024)
w = e.t2026
for c in ("d22", "d24", "dreg", "adj", "g2226"):
    print(c, "state weighted", round(np.average(e[c], weights=w), 2), "sd", round(np.sqrt(np.average(rel(e[c], w) ** 2, weights=w)), 2))
t = e.loc[top30]
print(t[["d22", "d24", "dreg", "adj", "g2226"]].corr().round(2).to_string())
show = e.sort_values("t2026", ascending=False).head(30)[["E2022", "E2024", "E2026", "d22", "d24", "dreg", "adj", "g2226", "t2026", "tot2022", "ed2022"]].copy()
for c in ("d22", "d24", "adj", "g2226"):
    show[c + "_r"] = rel(e[c], w)[show.index]
print(show.round(2).to_string())
e.to_csv(S / "county_panel2.csv")
