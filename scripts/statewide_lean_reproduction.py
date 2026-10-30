"""Does the model's frame reproduce Texas's statewide House lean in the training midterms?

Fit the shipped spec (abs env, no edu) on contested 2014/2018/2022 House+Senate rows, then
predict EVERY House district in each cycle and turnout-weight to a statewide margin.
Compare with tx_lean.py's figure (contested actual + uncontested imputed from presidential).
Also: the same statewide figure counting only contested districts, where both use actual
results, to isolate how much of any gap is the uncontested-seat imputation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402

H = ROOT / "data" / "raw" / "historical"
PRESW = {2014: H / "tx_presidential_house_2012_planh358.csv",
         2018: H / "tx_presidential_house_2016_planh358.csv",
         2022: H / "tx_presidential_house_2020_planh2316.csv"}
NAT = {2014: -5.7, 2018: 8.6, 2022: -2.8}
TERMS = ["x", "national_env", "chamber_senate", "dem_incumbent", "rep_incumbent",
         "challenger_viability_flag", "ie_c"]

train = rc.load(rc.bb.PROC / "phase1_dataset.csv", "open")
train["challenger_viability_flag"] = train["challenger_viability_flag"].fillna(0)
train = train.dropna(subset=["dem_incumbent", "rep_incumbent"])
Xt = np.column_stack([np.ones(len(train))] + [train[t] for t in TERMS])
b, *_ = np.linalg.lstsq(Xt, train["dem_2p_share"].values, rcond=None)

allr = rc.bb.load()
allr = allr[allr.chamber_lower == "house"].copy()
allr["x"] = allr["open_mean_d2p"]
for c in ["dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_dem_share"]:
    allr[c] = pd.to_numeric(allr[c], errors="coerce").fillna(0 if c != "ie_dem_share" else 0.5)
allr["ie_c"] = allr["ie_dem_share"] - 0.5
allr["contested"] = allr["contested"].astype(str).str.lower().isin(["true", "1", "1.0"])

for y in rc.CYCLES:
    d = allr[allr.year == y].copy()
    p = pd.read_csv(PRESW[y])
    p["w"] = p["dem_votes"] + p["rep_votes"]
    d = d.merge(p[["district", "w"]], on="district", how="left")
    X = np.column_stack([np.ones(len(d))] + [d[t] for t in TERMS])
    d["pred"] = X @ b
    m = lambda s, w: (2 * np.average(s, weights=w) - 1) * 100
    c = d[d.contested & d.dem_2p_share.between(0.001, 0.999)]
    u = d[~d.index.isin(c.index)]
    print(f"{y}: n={len(d)} contested={len(c)} | model all-district {m(d.pred, d.w):+.1f} "
          f"(lean {m(d.pred, d.w) - NAT[y]:+.1f}) | contested actual {m(c.dem_2p_share, c.w):+.1f} "
          f"vs model {m(c.pred, c.w):+.1f} | uncontested model-imputed {m(u.pred, u.w):+.1f}"
          f" [{(u.dem_2p_share > 0.5).sum()} D-held, {(u.dem_2p_share < 0.5).sum()} R-held]")
