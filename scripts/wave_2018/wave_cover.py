"""Does the shipped error already cover a 2018-sized run in the R-held close seats? And what would
a mean-zero error on that seat group do to 2026: tails only, no change to any center."""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

import whole_election_backtest as b
import whole_election_model as wm

pd.set_option("display.width", 240, "display.max_rows", 300, "display.max_columns", 40)
rows = []
envs = {y: (b.GB.final_538[y] + b.GB.final_rcp[y]) / 2 for y in b.YEARS}
for y in b.YEARS:
    x = b.Cycle(y, envs[y])
    te = x.te
    held = (te[x.is_h].rep_incumbent.astype(float).values == 1)
    for v in ("statewide", "legislative", "actual_T"):
        r = x.run(v)
        m = r["kind"] == "state_house"
        M = r["M"][:, m]
        a = r["actual"][m]
        mean = M.mean(0)
        k = held & (np.abs(mean) < 30)
        sim = (M[:, k] - mean[k]).mean(1)
        real = float((a[k] - mean[k]).mean())
        # same seats against the D-held and open remainder: where the seat group runs against the rest
        o = ~k & (np.abs(mean) < 30)
        gap = sim - (M[:, o] - mean[o]).mean(1)
        rg = real - float((a[o] - mean[o]).mean())
        rows.append({"year": y, "version": v, "n_close_R": int(k.sum()), "real": real, "sim_sd": float(sim.std()), "pit": float((sim < real).mean()), "n_rest": int(o.sum()), "gap_real": rg, "gap_sim_sd": float(gap.std()), "gap_pit": float((gap < rg).mean())})
print(pd.DataFrame(rows).round(2).to_string(index=False))

wh = wm.Whole()
df = wh.L["df"]
ish = (df.chamber_lower == "house").values
p = df.incumbent_party.astype(str).str.strip().str.upper()
op = df.open_seat.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
di = ((p == "D") & ~op).values
ship = wh.L["linear"].copy()
lin = ship * 200 - 100
grp = (~di) & ish & (np.abs(lin) < 30)
print("seats in the group", int(grp.sum()))
base = {}
for env in ("statewide", "legislative"):
    r = wh.run(env)
    base[env] = r
    M = r["M"]
    names = list(r["names"]) if "names" in r else None
    print(env, "shipped", round(r["house"].mean(), 2), "P76", round((r["house"] >= 76).mean(), 3), "keys", [k for k in r.keys()][:14])
    if names:
        idx = [names.index(f"hd{int(d)}") for d, g in zip(df.district, grp) if g]
        W = r["W"][:, [names.index(f"hd{int(d)}") for d in df.district[ish]]].astype(int).copy()
        g_ = grp[ish]
        L = M[:, [names.index(f"hd{int(d)}") for d in df.district[ish]]]
        forced = wh.leg_forced[ish]
        rng = np.random.default_rng(11)
        for sd in (0.0, 1.5, 2.5, 3.5):
            z = rng.normal(0, sd, len(L))[:, None] * g_[None, :]
            w = (L + z) > 0
            w[:, forced == 1.0] = True
            w[:, forced == 0.0] = False
            s = w.sum(1)
            print(f"   mean-zero group sd {sd}: seats {s.mean():.2f}, 10-90 {np.percentile(s, 10):.0f}-{np.percentile(s, 90):.0f}, P76 {(s >= 76).mean():.3f}")
