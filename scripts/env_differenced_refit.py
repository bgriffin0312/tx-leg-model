"""
env_differenced_refit.py

Refit the shipped spec (open baseline, education, incumbency, viability, IE) with the
national environment measured as a change from the baseline's presidential year, and
compare it out of sample with the absolute form it replaces.

    absolute:     share = a + b*x + e * env                      + ...
    differenced:  share = a + b*x + e * (env - env_baseline_yr)  + ...

The baseline x is a presidential-year composite, so the differenced form says the
district starts where that year left it and moves with the change in environment.

Usage:  python scripts/env_differenced_refit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402
import midterm_education_test as met  # noqa: E402

ENV_BASE = {2014: 1.2, 2018: -1.1, 2022: 3.1}      # national House margin, 2012/2016/2020
ENV_BASE_2026, ENV_2026 = -2.7, 9.2
pd.set_option("display.width", 200)

d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
d["env_abs"] = d.national_env
d["env_diff"] = d.national_env - d.year.map(ENV_BASE)
OTHER = ["x", "chamber_senate", "dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_c", "c_white_col"]

rows = []
for name, env in (("absolute", "env_abs"), ("differenced", "env_diff")):
    terms = [env] + OTHER
    res, resid = rc.loo(d, terms)
    c, se, s = rc.ols(d, terms)
    print(f"\n=== {name}, n={len(d)} ===")
    print(res.round(4).to_string(index=False))
    print("coef: " + ", ".join(f"{k} {v:+.4f} (se {se[k]:.4f})" for k, v in c.items()))
    env_val = ENV_2026 if env == "env_abs" else ENV_2026 - ENV_BASE_2026
    print(f"  environment contribution 2026 {c[env] * env_val * 100:+.2f}pp share, intercept {c['intercept']:+.4f}, sum {(c[env] * env_val + c['intercept']) * 100:+.2f}pp")
    rows.append({"spec": name, "rmse": np.sqrt((resid ** 2).mean()), "abs_level": res.mean_resid.abs().mean() * 100,
                 "abs_seat_err": res.seat_err.abs().mean(), "brier": res.brier.mean(),
                 "sum_2026": (c[env] * env_val + c["intercept"]) * 100})
t = pd.DataFrame(rows)
print("\n" + t.round(4).to_string(index=False))
print(f"\n2026 statewide margin effect, before pass-through differences: {(t.sum_2026[1] - t.sum_2026[0]) * 2:+.2f} points")


def fixed_slope_loo(env, beta):
    """Hold the environment slope fixed, so held-out cycles test levels only. The plain
    LOO above cannot judge the differenced form: without 2018 the training changes are
    -6.9 and -5.9, the slope is unidentified, and the 2018 projection runs away."""
    errs, seat = [], []
    for yr in rc.CYCLES:
        tr, te = d[d.year != yr].copy(), d[d.year == yr].copy()
        tr["dem_2p_share"] = tr.dem_2p_share - beta * tr[env]
        c, _, _ = rc.ols(tr, OTHER)
        p = rc.predict(te, c) + beta * te[env].values
        s = rc.score(te, p)
        errs.append(s["mean_resid"] * 100)
        seat.append(s["seat_err"])
    return errs, seat


print("\n=== Fixed-slope LOO: level miss by held-out cycle (pp share) and House seat error ===")
for env, beta in (("env_abs", 0.0046), ("env_diff", 0.0034), ("env_diff", 0.0046)):
    e, st = fixed_slope_loo(env, beta)
    print(f"  {env:<9} beta {beta:.4f}: levels 14/18/22 {e[0]:+.2f} {e[1]:+.2f} {e[2]:+.2f}  rms {np.sqrt(np.mean(np.square(e))):.2f}; "
          f"seat err {st[1]:+.1f} (2018) {st[2]:+.1f} (2022)")

print("\n=== Differenced coefficients for model_config ===")
for label, terms in (("with education", ["env_diff"] + OTHER), ("no education", ["env_diff"] + [t for t in OTHER if t != "c_white_col"])):
    c, _, _ = rc.ols(d, terms)
    print(f"  {label}: " + ", ".join(f"{k}={v:.4f}" for k, v in c.items()))
