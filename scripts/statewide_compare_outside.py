"""
statewide_compare_outside.py

Compare the frozen forecast (output/congress_forecast_2026.csv, statewide_forecast_2026.csv)
with Split Ticket / The Argument's published table. Held out until our numbers were
fixed; nothing here feeds the model.

The article's Datawrapper embeds are pinned to June versions. Resolve the current one
from the unversioned https://datawrapper.dwcdn.net/<id>/ page, then fetch
<id>/<version>/dataset.csv. House table Cy0fK, Senate table Hus3x.

Usage: python scripts/statewide_compare_outside.py <dir holding the downloaded tables>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 60); pd.set_option("display.max_colwidth", 26)
ROOT = Path(__file__).resolve().parent.parent
ST = Path(sys.argv[1])


def margin(s):
    """'D+4.2' / 'R+3' / 'Even' -> D margin."""
    t = str(s).strip()
    if t.lower().startswith("even"):
        return 0.0
    m = re.search(r"([DR])\s*\+\s*([\d.]+)", t)
    if m:
        return float(m.group(2)) * (1 if m.group(1) == "D" else -1)
    try:
        return float(t)
    except ValueError:
        return np.nan


for name in ("u9EZf", "Cy0fK", "g4hHM", "Hus3x"):
    d = pd.read_csv(ST / f"{name}.csv")
    print(name, d.shape, list(d.columns))
    print(d[d.iloc[:, 1].astype(str).str.contains("TX|Texas")].head(3).to_string(index=False))

h = pd.read_csv(ST / "Cy0fK.csv")
h = h[h.cd.astype(str).str.startswith("TX")].copy()
h["dist"] = h.cd.str.extract(r"(\d+)").astype(int)
h["st_margin"] = h.prediction.map(margin)
h["st_raw"] = pd.to_numeric(h.raw_prediction, errors="coerce")
h["st_p"] = pd.to_numeric(h.win_prob_d.astype(str).str.rstrip("%"), errors="coerce")
c = pd.read_csv(ROOT / "output" / "congress_forecast_2026.csv").set_index("dist")
j = h.set_index("dist").join(c[["r_name", "d_name", "pres24", "mean", "sd", "p_d", "cook", "silver"]], rsuffix="_ours")
j["gap"] = j["mean"] - j.st_margin
print("\n=== Texas House: ours vs Split Ticket (D margin) ===")
print(j[["d_name", "r_name", "dem_name", "rep_name", "pres24", "mean", "p_d", "st_margin", "st_p", "gap", "bucket", "cook", "silver"]].round(2).to_string())
print(f"\nmean gap {j.gap.mean():+.2f}, sd {j.gap.std():.2f}, corr {j['mean'].corr(j.st_margin):.3f}")
comp = j[(j["mean"].abs() < 15) | (j.st_margin.abs() < 15)]
print(f"competitive subset n={len(comp)}: mean gap {comp.gap.mean():+.2f}, sd {comp.gap.std():.2f}, corr {comp['mean'].corr(comp.st_margin):.3f}")
pc = j.st_p.max()
scale = 100 if pc > 1.5 else 1
print(f"expected D seats: ours {j.p_d.sum():.1f}, Split Ticket {(j.st_p / scale).sum():.1f}")
print("disagree on favourite:", j[np.sign(j['mean']) != np.sign(j.st_margin)].index.tolist())

s = pd.read_csv(ST / "Hus3x.csv")
print("\n=== Senate ===")
print(s[s.State.astype(str).str.contains("Texas")].to_string(index=False))
r = pd.read_csv(ROOT / "output" / "statewide_forecast_2026.csv")
print(r[r.race == "us_senate"][["race", "mean", "sd", "p_d"]].round(2).to_string(index=False))
