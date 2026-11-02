"""How far does a top-15 or top-30 county sample sit from the whole state on early net? Needed to
put older county-sample primary-history figures on a statewide footing."""
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
e = pd.read_csv(S / "county_panel2.csv", index_col=0)
for y in (2022, 2024, 2026):
    st = np.average(e[f"E{y}"], weights=e[f"t{y}"])
    for n in (15, 30):
        k = e.sort_values(f"t{y}").tail(n)
        s = np.average(k[f"E{y}"], weights=k[f"t{y}"])
        print(y, "top", n, "net", round(s, 1), "state", round(st, 1), "state minus sample", round(st - s, 1), "share of early", round(k[f"t{y}"].sum() / e[f"t{y}"].sum(), 3))
