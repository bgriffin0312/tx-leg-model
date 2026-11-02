"""Split the environment slope by who holds the seat, keep every other term, and see what it does.

  A. Outside check that does not use the formula or 2018: same seat, consecutive cycles, race
     swing in D-held and R-held seats against the national swing.
  B. Backtest 2018/2022/2024 with the split, slopes fixed the way 0.0034 is (fit on all rows).
     The seat's response to shared T scales the same way, so the split also reaches the error.
  C. 2026, both versions.
"""
import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_patched(name, path, a, b_):
    s = Path(path).read_text(encoding="utf-8")
    assert s.count(a) == 1, a
    m = types.ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(s.replace(a, b_), str(path), "exec"), m.__dict__)
    return m


wm = load_patched("whole_election_model", ROOT / "src" / "whole_election_model.py", "leg = lin + self.beta * (T - T_LEGISLATIVE)[:, None]", "leg = lin + self.beta * getattr(self, 'bmul', 1.0) * (T - T_LEGISLATIVE)[:, None]")
b = load_patched("wbt", ROOT / "scripts" / "whole_election_backtest.py", "leg = lin + beta * (T - self.TL + delta)[:, None]", "leg = lin + beta * getattr(self, 'bmul', 1.0) * (T - self.TL + delta)[:, None]")

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
sw, rc, met = b.sw, b.rc, b.met
OTHER = b.OTHER

print("=== A. Same seat, consecutive cycles: race swing by who held it, |prior margin|<40, no formula ===")
r = pd.read_csv(b.OUT / "whole_election_rolloff.csv")
h = r[r.tier == "state_house"]
rows = []
for a_, c_ in ((2012, 2014), (2014, 2016), (2016, 2018), (2018, 2020), (2022, 2024)):
    p, q = h[h.year == a_].set_index("office"), h[h.year == c_].set_index("office")
    k = p.index.intersection(q.index)
    k = [o for o in k if abs(p.m_race[o]) < 40]
    s = (q.m_race[k] - p.m_race[k])
    held = p.m_race[k] > 0
    rows.append({"pair": f"{a_}-{c_}", "dNat": sw.NAT[c_] - sw.NAT[a_], "n_D": int(held.sum()), "D_held": s[held].mean(), "n_R": int((~held).sum()), "R_held": s[~held].mean(), "gap_R_minus_D": s[~held].mean() - s[held].mean()})
A = pd.DataFrame(rows)
print(A.round(2).to_string(index=False))
for lab, t in (("all five", A), ("without 2016-18 and 2018-20", A[~A.pair.isin(["2016-2018", "2018-2020"])])):
    for col in ("D_held", "R_held"):
        f = sm.OLS(t[col], sm.add_constant(t[["dNat"]])).fit()
        print(f"  {lab}: {col} slope {f.params['dNat']:+.2f} se {f.bse['dNat']:.2f} const {f.params['const']:+.2f}")

d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
d["env_diff"] = d.national_env - d.year.map(b.ENV_BASE)
d["dE_d"], d["dE_o"] = d.env_diff * d.dem_incumbent.astype(float), d.env_diff * (1 - d.dem_incumbent.astype(float))
f = sm.OLS(d.dem_2p_share, sm.add_constant(d[["dE_d", "dE_o"] + OTHER].astype(float))).fit()
S_D, S_O = float(f.params["dE_d"]), float(f.params["dE_o"])
print(f"\nfit on the 212 rows: D-incumbent seats {S_D * 200:.3f}, every other seat {S_O * 200:.3f} margin per point (shipped 0.68); se {f.bse['dE_d'] * 200:.3f}, {f.bse['dE_o'] * 200:.3f}; effective cycles 2")


def fit_split(year, sd=S_D, so=S_O):
    t = d[d.year != year].copy() if year else d.copy()
    t["dem_2p_share"] = t.dem_2p_share - sd * t.dE_d - so * t.dE_o
    c, _, _ = rc.ols(t, OTHER)
    return c


print("\n=== B. Backtest ===")
rows = []
envs = {y: (b.GB.final_538[y] + b.GB.final_rcp[y]) / 2 for y in b.YEARS}
SPECS = {"shipped": None, "split": (S_D, S_O), "split, close-seat slopes": (0.591 / 200, 0.854 / 200)}
for y in b.YEARS:
    x = b.Cycle(y, envs[y])
    a = x.a
    house = (a.chamber_lower == "house").values
    lin0, TL0 = x.lin.copy(), x.TL
    for name, s in SPECS.items():
        if s is None:
            x.lin, x.TL = lin0, TL0
            if hasattr(x, "bmul"):
                del x.bmul
        else:
            c = fit_split(y, *s)
            di = a.dem_incumbent.astype(float).values
            sl = s[0] * di + s[1] * (1 - di)
            x.lin = (rc.predict(a, c) + sl * (x.env - b.ENV_BASE[y])) * 200 - 100
            x.TL = float(np.average(x.lin[house], weights=b.hd_weights(y, a.district[house])))
            x.bmul = (sl[x.m] / b.LEG_SLOPE)[None, :]
        for v in ("statewide", "legislative", "actual_T"):
            res = x.run(v)
            m = res["kind"] == "state_house"
            sc = res["W"][:, m].sum(1) + x.fixed_h
            act = int((res["actual"][m] > 0).sum()) + x.fixed_h
            mean, p = res["M"][:, m].mean(0), res["W"][:, m].mean(0)
            yy = res["actual"][m]
            close = np.abs(mean) < 20
            rows.append({"year": y, "spec": name, "version": v, "TL": x.TL, "mean": sc.mean(), "q10": np.percentile(sc, 10), "q90": np.percentile(sc, 90), "actual": act, "pit": float((sc < act).mean() + .5 * (sc == act).mean()),
                         "logscore": b.seat_score(sc, act), "brier": float(((p - (yy > 0)) ** 2).mean()), "rmse": float(np.sqrt(((mean - yy) ** 2).mean())), "miss_close": float((yy - mean)[close].mean())})
B = pd.DataFrame(rows)
print(B.round(2).to_string(index=False))
print(B.groupby(["version", "spec"])[["logscore", "brier", "rmse"]].mean().round(3).to_string())
B["abs_miss"] = (B["mean"] - B.actual).abs()
print(B.groupby(["version", "spec"]).abs_miss.mean().round(2).to_string())

print("\n=== C. 2026 ===")
wh = wm.Whole()
lg_ = wh.L["lg"]
df = wh.L["df"]
EB = -2.7
COEF_KEYS = {"intercept": "intercept", "dem_pres_2p_baseline": "x", "dem_incumbent": "dem_incumbent", "rep_incumbent": "rep_incumbent", "chamber_senate": "chamber_senate", "challenger_viability_flag": "challenger_viability_flag", "white_col_centered": "c_white_col"}
print("COEFS keys", {k: round(v, 4) for k, v in lg_.COEFS.items()})
pv = pd.read_csv(ROOT / "data" / "raw" / "tx_presidential_house_2024.csv")
vw = dict(zip(pv.district, pv.trump_votes + pv.harris_votes))
ish = (df.chamber_lower == "house").values
wts = df.district[ish].astype(int).map(vw).values
ship_lin = wh.L["linear"].copy()
ship_tl = float(np.average((ship_lin * 200 - 100)[ish], weights=wts))
env_coef = lg_.COEFS["national_env"]
saved, saved_ie = dict(lg_.COEFS), lg_.IE_COEFFICIENT


def dem_inc_col(df):
    p = df.incumbent_party.astype(str).str.strip().str.upper()
    op = df.open_seat.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    return ((p == "D") & ~op).values.astype(float)


# reproduce shipped from the same fit first, so any move below is the split and nothing else
c0 = None
dt = d.copy()
dt["dem_2p_share"] = dt.dem_2p_share - b.LEG_SLOPE * dt.env_diff
c0, _, _ = rc.ols(dt, OTHER)
print("rc.ols at the shipped slope:", {k: round(v, 4) for k, v in c0.items()})
DI = dem_inc_col(df)
out = []
for name, s in (("shipped", None), ("split", (S_D, S_O)), ("split, close-seat slopes", (0.591 / 200, 0.854 / 200))):
    if s is None:
        wh.L["linear"] = ship_lin
        wm.T_LEGISLATIVE = -7.2
        if hasattr(wh, "bmul"):
            del wh.bmul
    else:
        c = fit_split(None, *s)
        pair = []
        for slope in s:
            lg_.COEFS.update(saved)
            lg_.COEFS.update({k: c[v] for k, v in COEF_KEYS.items()})
            lg_.COEFS["national_env"] = slope
            lg_.IE_COEFFICIENT = c["ie_c"]
            # the formula's own intercept already sits on the differenced scale
            with contextlib.redirect_stdout(io.StringIO()):
                pair.append(np.asarray(lg_.build_linear_predictions(df, lg_.CURRENT_ENV, lg_.RACE_GENERIC_BALLOT_D_SHARE).values, float))
        lg_.COEFS.clear(); lg_.COEFS.update(saved); lg_.IE_COEFFICIENT = saved_ie
        new = pair[0] * DI + pair[1] * (1 - DI)
        tl = float(np.average((new * 200 - 100)[ish], weights=wts))
        wh.L["linear"] = new
        wm.T_LEGISLATIVE = -7.2 + (tl - ship_tl)
        wh.bmul = (s[0] * DI + s[1] * (1 - DI))[None, :] / b.LEG_SLOPE
        dm = (new - ship_lin) * 200
        print(f"{name}: legislative center {ship_tl:.2f} -> {tl:.2f}; R-held/open House mean move {dm[ish & (DI == 0)].mean():+.2f}, D-incumbent {dm[ish & (DI == 1)].mean():+.2f}")
    for env in ("statewide", "legislative"):
        r_ = wh.run(env)
        out.append({"spec": name, "version": env, "T_center": wh.Tc if env == "statewide" else wm.T_LEGISLATIVE, "house": r_["house"].mean(), "q10": np.percentile(r_["house"], 10), "q90": np.percentile(r_["house"], 90),
                    "P76": (r_["house"] >= 76).mean(), "senate": r_["senate"].mean(), "P16": (r_["senate"] >= 16).mean(), "gain_R_seats": r_["house"].mean() - 62})
print(pd.DataFrame(out).round(3).to_string(index=False))
