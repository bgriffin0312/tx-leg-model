"""Early-vote readings of 2026, run through the whole-election model.

Part 1 moves T's center only. Part 2 adds county shifts, center held, outside the border counties,
applied to each simulated race margin by its county weights: 0.5 per point of relative early change,
the 2024 seat slope (0.55) and the 2022-24 county J slope (0.48). Border counties get none because
party labels move most there. Nothing here changes a model file.
"""
import sys
import types
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
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
wh = wm.Whole()
print("T center", round(wh.Tc, 2), "T sd", round(float(wh.L["f"]["T"].std()), 2), "beta", round(wh.beta, 3))


def seats(r, lab):
    i = {k: n for n, k in enumerate(r["names"])}
    return {"run": lab, "T": float(r["T"].mean()), "Senate_m": float(r["M"][:, i["us_senate"]].mean()), "P_Sen": float(r["W"][:, i["us_senate"]].mean()), "Gov_m": float(r["M"][:, i["governor"]].mean()), "P_Gov": float(r["W"][:, i["governor"]].mean()),
            "House": float(r["house"].mean()), "q10": float(np.percentile(r["house"], 10)), "q90": float(np.percentile(r["house"], 90)), "P76": float((r["house"] >= 76).mean()), "TXSen": float(r["senate"].mean()), "USH": float(r["seats"].mean())}


rows = []
runs = {}
for lab, env in (("statewide", "statewide"), ("legislative", "legislative"), ("T -6.0", -6.0), ("T -3.5", -3.5), ("T -8.0", -8.0), ("T -9.0", -9.0)):
    r = wh.run(env)
    runs[lab] = r
    rows.append(seats(r, lab))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# county shifts
e = pd.read_csv(S / "county_panel2.csv", index_col=0)
BORDER = ["HIDALGO", "CAMERON", "WEBB", "STARR", "MAVERICK", "EL PASO", "NUECES", "ZAPATA", "WILLACY", "VAL VERDE", "JIM HOGG", "BROOKS", "DUVAL", "KLEBERG", "ZAVALA", "DIMMIT", "KINNEY", "PRESIDIO", "REEVES", "LA SALLE", "JIM WELLS", "SAN PATRICIO", "BEE", "UVALDE", "HUDSPETH", "CULBERSON"]
nb = e.sort_values("n2024").tail(60).index.difference(BORDER)   # where the county test held; rural slope is weak
print("counties carrying a shift", len(nb), "share of early vote", round(e.t2026[nb].sum() / e.t2026.sum(), 3))
w = e.t2026[nb]
sig = {}
for name in ("d24", "adj", "d22"):
    v = e.loc[nb, name]
    sig[name] = -(v - np.average(v, weights=w))
print("\ncorrelation of county shifts, no border, votes weighted-ish (top 60)")
top = e.loc[nb].sort_values("t2026").tail(60).index
print(pd.DataFrame(sig).loc[top].corr().round(2).to_string())

E = wh.E
top_v = E["top"]
cn = [c.upper() for c in E["counties"]]
df = wh.L["df"]


def share(frame, key, labels):
    t = frame.groupby([key, "county"]).v.sum().reset_index()
    t = t[t[key].isin(labels)]
    t["w"] = t.v / t.groupby(key).v.transform("sum")
    return t


hd_w = share(top_v, "hd", wh.h_ids)
sd_w = share(top_v, "sd", wh.s_ids)
cd_w = share(E["pieces"], "dist", list(wh.L["f"]["C"].index))


def race_shift(signal):
    s = signal.reindex([c.upper() for c in E["counties"]]).fillna(0.0)
    s.index = [c.lower() for c in s.index]
    d = {}
    for lab, t, key, pre in ((0, hd_w, "hd", "hd"), (1, sd_w, "sd", "sd"), (2, cd_w, "dist", "tx")):
        x = t.assign(v2=t.county.map(s) * t.w).groupby(key).v2.sum()
        d.update({f"{pre}{int(k)}": float(v) for k, v in x.items()})
    for nm in wh.cj_names:
        key = "bexar_da" if nm == "bexar_da" else nm.replace(" ", "_")
        d[key] = float(s.get(nm, 0.0))
    d["bexar_da"] = float(s.get("bexar", 0.0))
    return d


SLOPE = 0.5
idx = {k: i for i, k in enumerate(runs["statewide"]["names"])}
house_cols = [idx[f"hd{d}"] for d in wh.h_ids]
sen_cols = [idx[f"sd{d}"] for d in wh.s_ids]
cd_cols = [idx[f"tx{d}"] for d in wh.L["f"]["C"].index]
forced = wh.leg_forced
ff = {}
for k, (ch, dd) in enumerate(zip(df.chamber_lower, df.district)):
    nm = f"hd{int(dd)}" if ch == "house" else f"sd{int(dd)}"
    ff[nm] = forced[k]


def apply(r, d):
    M = r["M"].copy()
    W = r["W"].copy()
    for k, v in d.items():
        if k in idx and v:
            j = idx[k]
            before = W[:, j].copy()
            M[:, j] += SLOPE * v
            if ff.get(k, np.nan) == ff.get(k, np.nan) and k in ff:
                continue
            W[:, j] = M[:, j] > 0
    return M, W


rows = []
store = {}
for run in ("statewide", "legislative", "T -6.0"):
    for name in ("none", "d24", "adj", "d24+adj"):
        d = {} if name == "none" else race_shift(sig[name]) if "+" not in name else race_shift((sig["d24"] + sig["adj"]) / 2)
        M, W = apply(runs[run], d)
        h, s, c = W[:, house_cols].sum(1), W[:, sen_cols].sum(1) + runs[run]["hold"], W[:, cd_cols].sum(1)
        rows.append({"T": run, "signal": name, "House": h.mean(), "q10": np.percentile(h, 10), "q90": np.percentile(h, 90), "P76": (h >= 76).mean(), "TXSen": s.mean(), "USH": c.mean()})
        store[(run, name)] = (M, W, d)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# the mean-zero error on R-held and open seats within 30 points from the 2018 work, sd 2.5
p_ = df.incumbent_party.astype(str).str.strip().str.upper()
op = df.open_seat.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
di = ((p_ == "D") & ~op).values
ish = wh.is_house
lin = wh.L["linear"] * 200 - 100
g_ = ((~di) & (np.abs(lin) < 30))[ish]
print("\nseats in the group error", int(g_.sum()))
L_forced = wh.leg_forced[ish]
rows = []
for run in ("statewide", "legislative", "T -6.0"):
    for name in ("none", "d24+adj"):
        M, W, d = store[(run, name)]
        L = M[:, house_cols]
        for sd in (0.0, 2.5):
            z = np.random.default_rng(11).normal(0, sd, len(L))[:, None] * g_[None, :]
            w_ = (L + z) > 0
            w_[:, L_forced == 1.0] = True
            w_[:, L_forced == 0.0] = False
            h = w_.sum(1)
            rows.append({"T": run, "signal": name, "group_sd": sd, "House": h.mean(), "q10": np.percentile(h, 10), "q90": np.percentile(h, 90), "P76": (h >= 76).mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

M0, W0, _ = store[("T -6.0", "none")]
M1, W1, dd = store[("T -6.0", "d24+adj")]
names = runs["statewide"]["names"]
t = pd.DataFrame({"race": names, "m0": M0.mean(0), "p0": W0.mean(0), "m1": M1.mean(0), "p1": W1.mean(0)})
t["shift"] = t.m1 - t.m0
t["kind"] = t.race.str.extract(r"^(hd|sd|tx)")[0].fillna("other")
print("\nRaces where the county signal moves the chance of a D win by 4+ points (T -6.0)")
print(t[(t.p1 - t.p0).abs() >= .04].sort_values("p1", ascending=False).round(3).to_string(index=False))
print("\nCounty and DA races")
print(t[t.kind == "other"].sort_values("m1").round(3).to_string(index=False))
t.to_csv(S / "race_shift.csv", index=False)
print("\nCounty shifts, top 25 by early vote, no border (margin points at slope 0.5)")
c = pd.DataFrame({k: SLOPE * v for k, v in sig.items()}).join(e.t2026).sort_values("t2026", ascending=False).head(25)
print(c.round(1).to_string())
