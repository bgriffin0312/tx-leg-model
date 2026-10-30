"""Down-ballot vs presidential D 2p share in heavily Hispanic TX House districts, 2012-2024.

If the post-2016 shift were Trump-specific, the low-salience judicial races would lag the
presidential line. Unweighted district means within CVAP bands; national context alongside.
"""
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent.parent / "data" / "raw"
H = R / "historical"
RUNS = [
    (2012, H / "tx_downballot_house_2012_planh358.csv", H / "tx_cvap_house_2018.csv"),
    (2016, H / "tx_downballot_house_2016_planh2100.csv", H / "tx_cvap_house_2018.csv"),
    (2020, H / "tx_downballot_house_2020_planh2316.csv", R / "tx_cvap_house.csv"),
    (2024, H / "tx_downballot_house_2024_planh2316.csv", R / "tx_cvap_house.csv"),
]
rows = []
for y, db, cv in RUNS:
    d = pd.read_csv(db).merge(pd.read_csv(cv)[["district", "pct_hispanic"]], on="district")
    d["band"] = pd.cut(d["pct_hispanic"], [0, 30, 50, 70, 101],
                       labels=["<30% Hisp", "30-50%", "50-70%", "70%+ Hisp"], right=False)
    for band, g in d.groupby("band", observed=True):
        rows.append({"year": y, "band": band, "n": len(g),
                     "pres": round(100 * g["pres_d2p"].mean(), 1),
                     "judicial": round(100 * g["judicial_mean_d2p"].mean(), 1),
                     "rrc": round(100 * g["rrc_d2p"].mean(), 1)})
t = pd.DataFrame(rows)
for col in ["pres", "judicial", "rrc"]:
    print(f"\n{col} D 2p (%)")
    print(t.pivot(index="band", columns="year", values=col).to_string())
print("\nn districts:")
print(t.pivot(index="band", columns="year", values="n").to_string())
