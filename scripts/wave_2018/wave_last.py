"""Last three: a higher uniform slope; 2018's own seat pattern replayed on 2026; whether races
followed the local judicial swing less once straight-ticket voting ended."""
import contextlib
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
d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
d["env_diff"] = d.national_env - d.year.map(b.ENV_BASE)
SL = (0.0034, 0.0040, 0.0047)


def fit_u(year, s):
    t = d[d.year != year].copy() if year else d.copy()
    t["dem_2p_share"] = t.dem_2p_share - s * t.env_diff
    return rc.ols(t, OTHER)[0]


print("=== Straight ticket: same seat, race swing on local J swing, cycle mean removed ===")
r = pd.read_csv(b.OUT / "whole_election_rolloff.csv")
h = r[(r.tier == "state_house") & r.m_j.notna() & (r.m_race.abs() < 60)]
out = []
for a_, c_ in ((2012, 2014), (2014, 2016), (2016, 2018), (2018, 2020), (2022, 2024)):
    p, q = h[h.year == a_].set_index("office"), h[h.year == c_].set_index("office")
    k = p.index.intersection(q.index)
    x_, y_ = q.m_j[k] - p.m_j[k], q.m_race_j[k] - p.m_race_j[k]
    x_, y_ = x_ - x_.mean(), y_ - y_.mean()
    f = sm.OLS(y_, x_).fit()
    out.append({"pair": f"{a_}-{c_}", "n": len(k), "slope": f.params.iloc[0], "se": f.bse.iloc[0], "r2": f.rsquared, "resid_sd": np.sqrt(f.scale)})
print(pd.DataFrame(out).round(2).to_string(index=False))

print("\n=== Uniform slope, backtest ===")
rows = []
envs = {y: (b.GB.final_538[y] + b.GB.final_rcp[y]) / 2 for y in b.YEARS}
for y in b.YEARS:
    x = b.Cycle(y, envs[y])
    a = x.a
    house = (a.chamber_lower == "house").values
    for s in SL:
        c = fit_u(y, s)
        x.lin = (rc.predict(a, c) + s * (x.env - b.ENV_BASE[y])) * 200 - 100
        x.TL = float(np.average(x.lin[house], weights=b.hd_weights(y, a.district[house])))
        x.bmul = s / b.LEG_SLOPE
        for v in ("statewide", "legislative"):
            res = x.run(v)
            m = res["kind"] == "state_house"
            sc = res["W"][:, m].sum(1) + x.fixed_h
            act = int((res["actual"][m] > 0).sum()) + x.fixed_h
            mean, p = res["M"][:, m].mean(0), res["W"][:, m].mean(0)
            yy = res["actual"][m]
            rows.append({"year": y, "slope": round(s * 200, 2), "version": v, "TL": x.TL, "J": x.S["J"], "mean": sc.mean(), "q10": np.percentile(sc, 10), "q90": np.percentile(sc, 90), "actual": act, "logscore": b.seat_score(sc, act), "brier": float(((p - (yy > 0)) ** 2).mean()), "rmse": float(np.sqrt(((mean - yy) ** 2).mean()))})
B = pd.DataFrame(rows)
print(B.round(2).to_string(index=False))
B["abs_miss"] = (B["mean"] - B.actual).abs()
print(B.groupby(["version", "slope"])[["logscore", "brier", "rmse", "abs_miss"]].mean().round(3).to_string())

print("\n=== 2026 ===")
wh = wm.Whole()
lg_ = wh.L["lg"]
df = wh.L["df"]
pv = pd.read_csv(ROOT / "data" / "raw" / "tx_presidential_house_2024.csv")
vw = dict(zip(pv.district, pv.trump_votes + pv.harris_votes))
ish = (df.chamber_lower == "house").values
wts = df.district[ish].astype(int).map(vw).values
ship_lin = wh.L["linear"].copy()
ship_tl = float(np.average((ship_lin * 200 - 100)[ish], weights=wts))
saved, saved_ie = dict(lg_.COEFS), lg_.IE_COEFFICIENT
KEYS = {"intercept": "intercept", "dem_pres_2p_baseline": "x", "dem_incumbent": "dem_incumbent", "rep_incumbent": "rep_incumbent", "chamber_senate": "chamber_senate", "challenger_viability_flag": "challenger_viability_flag", "white_col_centered": "c_white_col"}
out = []


def seats(label, mode=None):
    for env in ("statewide", "legislative"):
        r_ = wh.run(env)
        out.append({"spec": label, "version": env, "T_center": wh.Tc if env == "statewide" else wm.T_LEGISLATIVE, "house": r_["house"].mean(), "q10": np.percentile(r_["house"], 10), "q90": np.percentile(r_["house"], 90), "P76": (r_["house"] >= 76).mean(), "senate": r_["senate"].mean()})


for s in SL:
    c = fit_u(None, s)
    lg_.COEFS.update({k: c[v] for k, v in KEYS.items()})
    lg_.COEFS["national_env"] = s
    lg_.IE_COEFFICIENT = c["ie_c"]
    with contextlib.redirect_stdout(io.StringIO()):
        new = np.asarray(lg_.build_linear_predictions(df, lg_.CURRENT_ENV, lg_.RACE_GENERIC_BALLOT_D_SHARE).values, float)
    lg_.COEFS.clear(); lg_.COEFS.update(saved); lg_.IE_COEFFICIENT = saved_ie
    tl = float(np.average((new * 200 - 100)[ish], weights=wts))
    wh.L["linear"] = new
    wm.T_LEGISLATIVE = -7.2 + (tl - ship_tl)
    wh.bmul = s / b.LEG_SLOPE
    seats(f"uniform slope {s * 200:.2f}")

# 2018's residual by band, R-held and open seats only, read off the held-out formula at the
# actual environment; nothing here is fitted to 2026
wh.L["linear"] = ship_lin
wm.T_LEGISLATIVE = -7.2
del wh.bmul
p = df.incumbent_party.astype(str).str.strip().str.upper()
op = df.open_seat.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
di = ((p == "D") & ~op).values
lin = ship_lin * 200 - 100
for label, edges, vals in (("2018 pattern, center unchanged by band", [-30, -20, -12, -6, 0, 6], [3.5, 5.3, 2.4, 2.1, 4.1]), ("flat +3.2 on R-held seats within 30", [-30, 30], [3.2]), ("half the 2018 pattern", [-30, -20, -12, -6, 0, 6], [1.75, 2.65, 1.2, 1.05, 2.05])):
    bump = np.zeros(len(df))
    for lo, hi, v in zip(edges[:-1], edges[1:], vals):
        bump[(lin > lo) & (lin <= hi) & ~di & ish] = v
    wh.L["linear"] = ship_lin + bump / 200
    print(label, "seats touched", int((bump > 0).sum()))
    seats(label)
wh.L["linear"] = ship_lin
print(pd.DataFrame(out).round(3).to_string(index=False))

print("\nR-held/open House seats by formula margin at the legislative center, 2026 vs 2018 held out at actual env")
a = b.leg_frame(2018)
a = a[a.chamber_lower == "house"]
l18 = (rc.predict(a, b.leg_fit(2018)) + b.LEG_SLOPE * (sw.NAT[2018] - b.ENV_BASE[2018])) * 200 - 100
E = [-100, -30, -20, -12, -6, 0, 6, 100]
print("2018", pd.Series(pd.cut(l18[a.dem_incumbent.astype(float).values == 0], E)).value_counts().sort_index().to_dict())
print("2026", pd.Series(pd.cut(lin[ish & ~di], E)).value_counts().sort_index().to_dict())
