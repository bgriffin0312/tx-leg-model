"""Second try: per-cycle intercepts with the env coefficient FIXED, absolute vs differenced env.

With the env slope fixed, each spec leaves one level per cycle. The spec whose
cycle levels agree better is the one whose env term explains the cycle-to-cycle
moves; a 2026 forecast then rests on the mean of those levels.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402

ENV_BASE = {2014: 1.2, 2018: -1.1, 2022: 3.1}
OTHER = ["chamber_senate", "dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_c"]

d = rc.load(rc.bb.PROC / "phase1_dataset.csv", "open")
d["challenger_viability_flag"] = d["challenger_viability_flag"].fillna(0)
d = d.dropna(subset=["dem_incumbent", "rep_incumbent"])
d["env_abs"] = d["national_env"]
d["env_diff"] = d["national_env"] - d["year"].map(ENV_BASE)

# within-cycle coefficients (cycle fixed effects) for x and the other terms
dummies = np.column_stack([(d.year == y).astype(float) for y in rc.CYCLES])
X = np.column_stack([dummies, d["x"]] + [d[t] for t in OTHER])
b, *_ = np.linalg.lstsq(X, d["dem_2p_share"].values, rcond=None)
fe = dict(zip(rc.CYCLES, b[:3]))
print("cycle fixed effects (pp):", {y: round(100 * v, 2) for y, v in fe.items()}, " x slope", round(b[3], 4))

for beta in [0.0034, 0.0043, 0.0046, 0.0050]:
    for col in ["env_abs", "env_diff"]:
        env = {y: d.loc[d.year == y, col].iloc[0] for y in rc.CYCLES}
        lv = {y: 100 * (fe[y] - beta * env[y]) for y in rc.CYCLES}
        vals = np.array(list(lv.values()))
        print(f"beta {beta:.4f} {col:8s} levels {{{', '.join(f'{y}: {v:+.2f}' for y, v in lv.items())}}}"
              f"  spread sd {vals.std(ddof=1):.2f}pp")
# best-fitting beta for each spec: regress cycle FE on env (3 points)
for col in ["env_abs", "env_diff"]:
    e = np.array([d.loc[d.year == y, col].iloc[0] for y in rc.CYCLES])
    f = np.array([fe[y] for y in rc.CYCLES])
    A = np.column_stack([np.ones(3), e])
    (a0, a1), res, *_ = np.linalg.lstsq(A, f, rcond=None)
    resid = f - A @ np.array([a0, a1])
    print(f"{col}: FE ~ env slope {a1:.5f}, resid pp {np.round(100 * resid, 2)}, rmse {100 * np.sqrt((resid**2).mean()):.2f}")
