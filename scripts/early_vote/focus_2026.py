"""A swing one point bigger than 2018's, spread evenly and then where the polls put it.

Size. Texas judicial vote: 2016 to 2018 +8.2, 2014 to 2018 +16.1. Plus one point on the 2024 and 2022
results gives T -7.15 and T +3.1. T -6.3 is 2018's own level plus one.

Where. Zero-sum race pattern, s = sum_g (district share - Texas share) * delta_g, as in
scripts/race_shift_test.py, doubled to margin. delta_g is the D two-party change by race:
  UT    same-pollster, UT Oct 2026 LV minus UT Oct 2024 LV, the one version with a held-out test
  polls every LV crosstab since September minus UT Oct 2024 LV: larger, and crosses pollsters
  edu   UT's education gap put on white voters only, white college against white non-college
  UT+edu both. UT publishes no race-by-education row, so half of white voters are taken as college.
The formula already carries a midterm white-college term, so edu sits on top of that term.
Applied to legislative and county races. Statewide races are unchanged by construction. U.S. House
races are left alone: they are already conditioned on their own polls, tested in part A.
Nothing here changes a model file.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

pd.set_option("display.width", 240, "display.max_rows", 400, "display.max_columns", 40)


def load_patched(name, path, a, b_):
    s = Path(path).read_text(encoding="utf-8")
    assert s.count(a) == 1, a
    m = types.ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(s.replace(a, b_), str(path), "exec"), m.__dict__)
    return m


wm = load_patched("whole_election_model", ROOT / "src" / "whole_election_model.py", 'center = self.Tc if env == "statewide" else T_LEGISLATIVE', 'center = self.Tc if env == "statewide" else (T_LEGISLATIVE if env == "legislative" else float(env))')
DRAWS = 40000    # small enough to run under the broker's floor; the full set is 200,000
_load = wm.load_layers


def thin():
    L = _load()
    f = L["f"]
    n = len(f["T"])
    for k, v in f.items():
        if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == n:
            f[k] = v[:DRAWS]
    return L


wm.load_layers = thin
wh = wm.Whole()
f, df, lg = wh.L["f"], wh.L["df"], wh.L["lg"]
print("draws", wh.n, "T center", round(wh.Tc, 2), "Hispanic slope h mean", round(float(f["h"].mean()), 2), "sd", round(float(f["h"].std()), 2))

print("\n=== A. District polls against the 2024 judicial vote ===")
b = pd.read_csv(ROOT / "data" / "raw" / "statewide" / "cd_baseline_2026.csv").set_index("dist")
jc = [c for c in b.columns if c.startswith("m2024_") and any(k in c for k in ("cca", "sup_ct"))]
p = pd.read_csv(ROOT / "data" / "raw" / "statewide" / "cd_polls_2026.csv")
p = p[pd.to_datetime(p.field_end) >= "2026-08-15"].copy()
p["m"] = p.d_pct - p.r_pct
adj = p.partisan_flag.map({"D": -2.0, "R": 2.0}).fillna(0.0)    # the usual sponsor discount; the sign of the slope does not depend on it
p["madj"] = p.m + adj
r0 = wh.run(-6.0)
ix = {k: i for i, k in enumerate(r0["names"])}
C = f["C"]
rows = []
for d, g in p.groupby("dist"):
    if d not in b.index or d not in C.index:
        continue
    j24 = float(b.loc[d, jc].mean())
    j24 = j24 * 100 if abs(j24) <= 1 else j24
    rows.append({"dist": d, "n_polls": len(g), "poll_adj": g.madj.mean(), "j2024": j24, "swing": g.madj.mean() - j24, "model_T-6": float(r0["M"][:, ix[f"tx{d}"]].mean()), "ss": float(C.ss[d])})
a = pd.DataFrame(rows).set_index("dist")
a["model_swing"] = a["model_T-6"] - a.j2024
print(a.round(1).to_string())
sl = np.polyfit(a.ss, a.swing, 1)
print("poll swing on Hispanic exposure: slope", round(sl[0], 2), "per unit, intercept", round(sl[1], 1), "; model minus poll mean", round((a["model_T-6"] - a.poll_adj).mean(), 1))

print("\n=== B. Race deltas ===")
t = pd.read_csv(ROOT / "data" / "raw" / "texas_crosstab_inputs.csv")
ut = t[t.pollster.str.startswith("UT/") & (t.ballot == "tx_legislature_generic") & (t.population == "LV")].sort_values("poll_date")
prior, now = ut[ut.poll_date.str.startswith("2024-10")].iloc[-1], ut.iloc[-1]
print("prior", prior.poll_date, "now", now.poll_date)
G = ["white", "black", "hispanic", "other"]
d_ut = {g: float(now[f"{g}_d_2p"] - prior[f"{g}_d_2p"]) if g != "other" else 0.0 for g in G}
lv = t[(t.population == "LV") & (t.poll_date >= "2026-09-01")]
print(lv[["poll_date", "pollster", "ballot", "white_d_2p", "black_d_2p", "hispanic_d_2p", "sample_size"]].to_string(index=False))
d_po = {g: float(lv[f"{g}_d_2p"].mean() - prior[f"{g}_d_2p"]) if g != "other" else 0.0 for g in G}
print("UT delta", {g: round(v * 100, 1) for g, v in d_ut.items()})
print("polls delta", {g: round(v * 100, 1) for g, v in d_po.items()})

# UT Texas Legislature generic, LV, R and D: Oct 2024 Q24B p.93, Oct 2026 Q27B p.136. Weights are rough
# shares within each side of the college line; subgroups run about 100-200 respondents.
E24 = {"lths": (47, 41), "hs": (53, 38), "some": (41, 47), "two": (39, 41), "four": (52, 38), "post": (47, 51)}
E26 = {"lths": (58, 40), "hs": (50, 37), "some": (46, 41), "two": (40, 49), "four": (50, 46), "post": (45, 49)}
WN, WC = {"lths": .06, "hs": .31, "some": .39, "two": .24}, {"four": 2 / 3, "post": 1 / 3}


def d2p(tab, w):
    return sum(v * 100 * tab[k][1] / sum(tab[k]) for k, v in w.items())


dcol, dnon = d2p(E26, WC) - d2p(E24, WC), d2p(E26, WN) - d2p(E24, WN)
gap = (dcol - dnon) / 100
print("UT college delta", round(dcol, 1), "non-college", round(dnon, 1), "gap", round(gap * 100, 1))
print("UT place delta: urban", round(100 * (49 / 95 - 51 / 90), 1), "suburban", round(100 * (46 / 90 - 41 / 89), 1), "rural", round(100 * (27 / 87 - 29 / 93), 1))
print("formula white-college term, D 2p per unit share", lg.COEFS.get("white_col_centered"), "= margin per unit", round(200 * lg.COEFS.get("white_col_centered", 0), 1), "; UT gap, margin per unit", round(200 * gap, 1))

cen = lg.centered_group_shares(df, education=False)    # white, black, hispanic, other, Texas-centered
cen5 = lg.centered_group_shares(df, education=True)    # white_noncol, white_col, black, hispanic, other
top = wh.E["top"]
counties_hd = wh.E["weights"](top, "county", wh.cj_names)[1]
share = counties_hd / counties_hd.sum(1, keepdims=True)
house = wh.is_house


def pattern(dl, edu=0.0):
    s = (cen5 @ np.array([dl["white"] - edu / 2, dl["white"] + edu / 2, dl["black"], dl["hispanic"], dl["other"]])) * 200
    out = {(f"hd{int(k)}" if c == "house" else f"sd{int(k)}"): v for c, k, v in zip(df.chamber_lower, df.district, s)}
    byhd = pd.Series(s[house], index=wh.h_ids).reindex(wh.E["hds"]).fillna(0).values
    cty = share @ byhd
    for nm, v in zip(wh.cj_names, cty):
        out[nm.replace(" ", "_")] = float(v)
    out["bexar_da"] = float(cty[wh.cj_names.index("bexar")])
    return out


PAT = {"even": {}, "UT": pattern(d_ut), "polls": pattern(d_po), "edu": pattern(dict.fromkeys(G, 0.0), gap), "UT+edu": pattern(d_ut, gap)}
s = pd.Series(PAT["UT"])
print("UT pattern on House seats: sd", round(s[s.index.str.match(r"hd")].std(), 2), "min", round(s[s.index.str.match(r"hd")].min(), 1), "max", round(s[s.index.str.match(r"hd")].max(), 1))

forced = {}
for k, (ch, dd) in enumerate(zip(df.chamber_lower, df.district)):
    forced[f"hd{int(dd)}" if ch == "house" else f"sd{int(dd)}"] = wh.leg_forced[k]


def apply(r, d):
    i = {k: n for n, k in enumerate(r["names"])}
    M, W = r["M"].copy(), r["W"].copy()
    for k, v in d.items():
        if k in i and v:
            M[:, i[k]] += v
            if forced.get(k, np.nan) == forced.get(k, np.nan) and k in forced:
                continue
            W[:, i[k]] = M[:, i[k]] > 0
    return M, W, i


CENTERS = {"current read": -6.0, "2018 swing +1, from 2024": -7.15, "2018 swing +1, from 2022": 3.1}
rows, keep = [], {}
for lab, T in CENTERS.items():
    r = wh.run(T)
    for pat, d in PAT.items():
        M, W, i = apply(r, d)
        h = W[:, [i[f"hd{x}"] for x in wh.h_ids]].sum(1)
        sn = W[:, [i[f"sd{x}"] for x in wh.s_ids]].sum(1) + r["hold"]
        c = W[:, [i[f"tx{x}"] for x in C.index]].sum(1)
        rows.append({"case": lab, "T": T, "where": pat, "Senate_m": M[:, i["us_senate"]].mean(), "P_Sen": W[:, i["us_senate"]].mean(), "Gov_m": M[:, i["governor"]].mean(), "P_Gov": W[:, i["governor"]].mean(),
                     "House": h.mean(), "q10": np.percentile(h, 10), "q90": np.percentile(h, 90), "P76": (h >= 76).mean(), "TXSen": sn.mean(), "USH": c.mean()})
        keep[(lab, pat)] = (W.mean(0), M.mean(0), i)
res = pd.DataFrame(rows)
print("\n=== C. Results ===")
print(res.round(3).to_string(index=False))

print("\n=== D. Seats that move 5+ points of win chance, T -7.15 ===")
w0, m0, i = keep[("2018 swing +1, from 2024", "even")]
names = list(i)
tab = pd.DataFrame({"race": names, "even": w0})
for pat in ("UT", "polls", "edu", "UT+edu"):
    tab[pat] = keep[("2018 swing +1, from 2024", pat)][0]
tab["s_UT"] = tab.race.map(PAT["UT"]).fillna(0)
tab["s_edu"] = tab.race.map(PAT["edu"]).fillna(0)
tab["s_both"] = tab.race.map(PAT["UT+edu"]).fillna(0)
tab = tab[~tab.race.str.startswith("tx")]
mv = tab[((tab.UT - tab.even).abs() >= .05) | ((tab.polls - tab.even).abs() >= .05) | ((tab["UT+edu"] - tab.even).abs() >= .05)].sort_values("s_both")
print(mv.round(2).to_string(index=False))
print("\nCounty races")
print(tab[~tab.race.str.match(r"hd|sd|us_|gov|lt_|att|comp|land|ag_|rail")].round(2).to_string(index=False))
res.to_csv(ROOT / "output" / "focus_2026.csv", index=False)
