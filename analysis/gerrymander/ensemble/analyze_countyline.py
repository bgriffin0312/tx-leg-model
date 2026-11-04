"""Compare the enacted Texas House map (PlanH2316) with three neutral ensembles on 2024 presidential lean:
  county-line   county groups fixed at the enacted plan's, whole counties kept whole, +/-5%  (county_line_chain.py)
  plain +/-5%   same chain settings, no county rule                                        (county_line_chain.py --plain)
  plain +/-2%   the July run                                                               (run_chain.py)
Drops the first BURN steps of each chain. Positive metric = pro-Republican.
    python analysis/gerrymander/ensemble/analyze_countyline.py   (main Python; reads output/gerrymander/)
"""
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as gm  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "output" / "gerrymander"
BURN = 2000
det = pd.read_csv(OUT / "district_detail_2024.csv")
en = gm.run_all(det[det.chamber == "house"].dem_share_pres_2024.dropna().to_numpy())

ens = {}
for f in sorted(glob.glob(str(OUT / "ensemble_house_countyline_steps*_seed*.csv"))):
    ens[f"county-line (seed {re.search(r'seed(\d+)', f).group(1)})"] = pd.read_csv(f)
for f in sorted(glob.glob(str(OUT / "ensemble_house_plain_eps05_steps*_seed*.csv"))):
    ens[f"plain +/-5% (seed {re.search(r'seed(\d+)', f).group(1)})"] = pd.read_csv(f)
jul = OUT / "ensemble_house_steps20000.csv"
if jul.exists():
    j = pd.read_csv(jul)
    j = j[~j.is_enacted.astype(bool)]
    ens["plain +/-2% (July)"] = j.rename(columns=lambda c: c.replace("y24_", "") if c.startswith("y24_") else c)

M = [("eg_efficiency_gap", "Efficiency gap"), ("mm_mean_median", "Mean-median"), ("pb_partisan_bias", "Partisan bias"),
     ("decl_declination", "Declination"), ("pack_n_dem_seats", "Dem seats")]
rows = []
for name, df in ens.items():
    df = df[df.step > BURN]
    for k, lab in M:
        s = df[k].dropna()
        e = en[k]
        if k == "pack_n_dem_seats":
            beyond = (s > e).mean() * 100 + (s == e).mean() * 50     # plans giving Democrats more seats than enacted
        else:
            beyond = (s < e).mean() * 100 + (s == e).mean() * 50     # plans less pro-R than enacted
        rows.append({"ensemble": name, "metric": lab, "enacted": e, "mean": s.mean(), "p05": s.quantile(.05),
                     "p95": s.quantile(.95), "enacted_more_pro_R_than_pct": beyond, "n": len(s)})
T = pd.DataFrame(rows)
T.to_csv(OUT / "outlier_summary_house_countyline.csv", index=False)
pd.set_option("display.width", 200)
for lab in [m[1] for m in M]:
    t = T[T.metric == lab]
    print(f"\n{lab} (enacted {t.enacted.iloc[0]:+.4f})" if lab != "Dem seats" else f"\n{lab} (enacted {int(t.enacted.iloc[0])})")
    for r in t.itertuples():
        fmtv = (lambda x: f"{x:6.2f}") if lab == "Dem seats" else (lambda x: f"{x:+.4f}")
        print(f"  {r.ensemble:26} mean {fmtv(r.mean)}  5-95% {fmtv(r.p05)} to {fmtv(r.p95)}  enacted more pro-R than {r.enacted_more_pro_R_than_pct:5.1f}% (n {r.n})")

for name, df in ens.items():
    if "county_crossing" in df:
        cc = df[df.step > BURN].county_crossing.dropna()
        print(f"\n{name}: county-crossing pieces mean {cc.mean():.1f} (5-95% {cc.quantile(.05):.0f}-{cc.quantile(.95):.0f}); enacted 20")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, df in ens.items():
    df = df[df.step > BURN]
    ax[0].hist(df.pack_n_dem_seats, bins=np.arange(45.5, 70.5, 1), alpha=.45, label=name, density=True)
    ax[1].hist(df.eg_efficiency_gap, bins=60, alpha=.45, label=name, density=True)
ax[0].axvline(en["pack_n_dem_seats"], color="k", lw=2, label="enacted")
ax[1].axvline(en["eg_efficiency_gap"], color="k", lw=2, label="enacted")
ax[0].set_title("Democratic seats at 2024 presidential vote")
ax[1].set_title("Efficiency gap (positive = pro-R)")
ax[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "outlier_house_countyline_2024.png", dpi=130)
print(f"\nwrote {OUT / 'outlier_summary_house_countyline.csv'} and outlier_house_countyline_2024.png")
