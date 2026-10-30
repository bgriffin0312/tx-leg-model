"""How far right of the nation does Texas run in TX House races, 2012-2024?

Per cycle: legislative D 2p share in contested districts vs the district's presidential
D 2p share (same year in presidential years, prior presidential in midterms). The mean
gap (turnout-weighted) is used to impute uncontested districts, giving a statewide
"every seat contested" TX House vote. Compared to the national US House popular vote.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
H = ROOT / "historical"

PRES = {
    2012: H / "tx_presidential_house_2012_planh309.csv",
    2014: H / "tx_presidential_house_2012_planh358.csv",
    2016: H / "tx_presidential_house_2016_planh358.csv",
    2018: H / "tx_presidential_house_2016_planh358.csv",
    2020: H / "tx_presidential_house_2020_h2100.csv",
    2022: H / "tx_presidential_house_2020_planh2316.csv",
    2024: ROOT / "tx_presidential_house_2024.csv",
}
RESULTS = {y: H / f"tx_house_results_{y}.csv" for y in PRES if y != 2024}

# National US House popular vote, D-R margin (pp). 2012-2022 from build_phase1_dataset.py;
# 2020 and 2024 approximate.
NAT = {2012: 1.2, 2014: -5.7, 2016: -1.1, 2018: 8.6, 2020: 3.1, 2022: -2.8, 2024: -2.7}
# Statewide presidential D-R margin: TX and nation (pres years only).
TX_PRES = {2012: -15.8, 2016: -9.0, 2020: -5.6, 2024: -13.7}
NAT_PRES = {2012: 3.9, 2016: 2.1, 2020: 4.5, 2024: -1.5}


def load_results(y):
    if y == 2024:
        r = pd.read_csv(ROOT / "tx_house_results_2024.csv")
        r["contested"] = r["contested"].astype(str).str.lower() == "true"
        r["dem_2p_share"] = r["d_votes"] / (r["d_votes"] + r["r_votes"])
        return r[["district", "contested", "dem_2p_share"]]
    r = pd.read_csv(RESULTS[y], encoding="utf-8-sig")
    r["contested"] = r["contested"].astype(str).str.lower() == "true"
    return r[["district", "contested", "dem_2p_share"]]


rows = []
for y, pf in PRES.items():
    p = pd.read_csv(pf)
    dv = "harris_votes" if y == 2024 else "dem_votes"
    rv = "trump_votes" if y == 2024 else "rep_votes"
    p["w"] = p[dv] + p[rv]
    p["pres"] = p[dv] / p["w"]
    d = load_results(y).merge(p[["district", "pres", "w"]], on="district", how="inner")
    assert len(d) == 150, (y, len(d))
    c = d[d["contested"] & d["dem_2p_share"].between(0.001, 0.999)]
    gap = np.average(c["dem_2p_share"] - c["pres"], weights=c["w"])
    gap_eq = (c["dem_2p_share"] - c["pres"]).mean()
    d["leg_equiv"] = np.where(d.index.isin(c.index), d["dem_2p_share"], d["pres"] + gap)
    state_leg = np.average(d["leg_equiv"], weights=d["w"])
    state_pres = np.average(d["pres"], weights=d["w"])
    contested_only = np.average(c["dem_2p_share"], weights=c["w"])
    leg_margin = (2 * state_leg - 1) * 100
    rows.append({
        "year": y, "n_contested": len(c),
        "gap_vs_pres_pp": gap * 100, "gap_unweighted_pp": gap_eq * 100,
        "tx_pres_margin_districts": (2 * state_pres - 1) * 100,
        "tx_leg_equiv_margin": leg_margin,
        "tx_contested_only_margin": (2 * contested_only - 1) * 100,
        "nat_house_margin": NAT[y],
        "tx_leg_lean_vs_nat": leg_margin - NAT[y],
        "tx_pres_lean_vs_nat_pres": (TX_PRES[y] - NAT_PRES[y]) if y in TX_PRES else np.nan,
    })

out = pd.DataFrame(rows)
print(out.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
