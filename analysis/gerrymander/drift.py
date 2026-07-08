"""
Phase B: has demographic/voting-pattern change since the 2021 maps were drawn
made the gerrymander stronger or weaker?

Two independent lines of evidence, both at the district level on CURRENT
(2021-drawn) lines so they're apples-to-apples:

  1. Partisan drift: district Dem two-party presidential share, 2020 -> 2024.
     2020 is used as the "as-drawn" baseline (the maps were built with 2020
     census population and the most recent full statewide election available
     at drawing time was 2020's), not 2016, because 2016 predates the current
     lines entirely (any 2016 vs current-map comparison silently mixes old
     and new district shapes). The authoritative answer to "did the
     gerrymander's structural bias grow or shrink" is not any single
     district's trend — it's whether metrics.run_all() output over the WHOLE
     MAP is larger or smaller under 2024 lean than under 2020 lean. That
     recomputation is what carries the claim in this module; the per-district
     classification below is descriptive color, not the load-bearing evidence.

  2. Demographic drift: CVAP composition 2018 vs current (2024 ACS vintage)
     by district — where Hispanic/white/Black/Asian CVAP share moved.

A "dummymander" is a map that was gerrymandered based on a stale partisan
snapshot and decays as that snapshot ages — safe seats for the drawing party
quietly become competitive as the underlying electorate shifts. The opposite
failure mode also exists: a district drawn to pack the opposing party can
become packed even further if that population's voting behavior shifts
toward the packed party's opponent (net effect: fewer wasted votes for the
packed party, i.e. slightly less efficient packing) OR the reverse, so this
is genuinely two-directional and must be measured, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import metrics as gm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
HIST = RAW / "historical"


def load_pres_year(chamber: str, year: int) -> pd.DataFrame:
    """Uniform loader across the different historical/current file schemas.
    Common output columns: district, dem_share_<year>."""
    if year == 2024:
        path = RAW / f"tx_presidential_{chamber}_2024.csv"
    else:
        path = HIST / f"tx_presidential_{chamber}_{year}.csv"
    df = pd.read_csv(path)
    return df[["district", "dem_pres_2p_baseline"]].rename(
        columns={"dem_pres_2p_baseline": f"dem_share_{year}"}
    )


def load_cvap(chamber: str, current: bool) -> pd.DataFrame:
    if current:
        path = RAW / f"tx_cvap_{chamber}.csv"
        suffix = "_now"
    else:
        path = HIST / f"tx_cvap_{chamber}_2018.csv"
        suffix = "_2018"
    df = pd.read_csv(path)
    cols = ["district", "cvap_total", "pct_white_nh", "pct_black_nh", "pct_hispanic", "pct_asian_nh"]
    df = df[cols].rename(columns={c: f"{c}{suffix}" for c in cols if c != "district"})
    return df


def partisan_drift_table(chamber: str) -> pd.DataFrame:
    d2016 = load_pres_year(chamber, 2016)
    d2020 = load_pres_year(chamber, 2020)
    d2024 = load_pres_year(chamber, 2024)

    out = d2020.merge(d2024, on="district", how="inner").merge(d2016, on="district", how="left")
    out["chamber"] = chamber
    out["shift_2020_2024_pp"] = (out["dem_share_2024"] - out["dem_share_2020"]) * 100
    out["shift_2016_2020_pp"] = (out["dem_share_2020"] - out["dem_share_2016"]) * 100
    return out


def demographic_drift_table(chamber: str) -> pd.DataFrame:
    now = load_cvap(chamber, current=True)
    then = load_cvap(chamber, current=False)
    out = then.merge(now, on="district", how="inner")
    out["chamber"] = chamber
    for group in ["white_nh", "black_nh", "hispanic", "asian_nh"]:
        out[f"delta_pct_{group}_pp"] = out[f"pct_{group}_now"] - out[f"pct_{group}_2018"]
    return out


def bucket_2020(dem_share_2020: float) -> str:
    if dem_share_2020 < 0.40:
        return "safe_R"
    if dem_share_2020 < 0.50:
        return "lean_R"
    if dem_share_2020 < 0.60:
        return "lean_D"
    return "safe_D"


def classify_dummymander(drift: pd.DataFrame, flat_threshold_pp: float = 1.0) -> pd.DataFrame:
    drift = drift.copy()
    drift["bucket_2020"] = drift["dem_share_2020"].apply(bucket_2020)

    def direction(pp):
        if pp > flat_threshold_pp:
            return "trending_D"
        if pp < -flat_threshold_pp:
            return "trending_R"
        return "flat"

    drift["trend_2020_2024"] = drift["shift_2020_2024_pp"].apply(direction)
    drift["dummymander_class"] = drift["bucket_2020"] + "_" + drift["trend_2020_2024"]

    # Interpretive flag: does this district's movement weaken or strengthen
    # the map's pro-R structure? R-held seats trending D = weakening
    # (cracked R margins eroding). D-leaning seats trending D = strengthening
    # (deeper packing, more wasted D votes). D-leaning seats trending R =
    # ambiguous at the district level (could reduce packing OR flip the seat
    # to R outright) -- resolved only by the map-level metric recomputation,
    # not asserted here.
    def effect(row):
        b, t = row["bucket_2020"], row["trend_2020_2024"]
        if b in ("safe_R", "lean_R") and t == "trending_D":
            return "weakens_gerrymander"
        if b in ("safe_R", "lean_R") and t == "trending_R":
            return "strengthens_gerrymander"
        if b in ("safe_D", "lean_D") and t == "trending_D":
            return "strengthens_gerrymander"
        if b in ("safe_D", "lean_D") and t == "trending_R":
            return "ambiguous_see_map_level"
        return "flat"

    drift["effect_label"] = drift.apply(effect, axis=1)
    return drift


def map_level_trajectory(chamber: str, drift: pd.DataFrame) -> pd.DataFrame:
    """The authoritative before/after: recompute the full metric battery
    using 2020 lean vs 2024 lean on the SAME (current) district lines.

    Known upstream data gap: the project's 2020 historical presidential
    collector is missing TX House districts 10 and 63 entirely, and recorded
    zero votes (NaN share) for district 65. Those districts are dropped from
    this 2020-vs-2024 comparison (not fabricated) with a logged warning;
    they remain in the per-district drift table as NaN so the gap stays
    visible rather than silently patched."""
    complete = drift.dropna(subset=["dem_share_2020", "dem_share_2024"])
    dropped = set(drift["district"]) - set(complete["district"])
    if dropped:
        print(
            f"  [data gap] {chamber}: dropping {len(dropped)} district(s) with missing 2020 baseline "
            f"from map-level trajectory: {sorted(dropped)} ({len(complete)}/{len(drift)} districts used)"
        )

    rows = []
    for year, col in [(2020, "dem_share_2020"), (2024, "dem_share_2024")]:
        share = complete[col].to_numpy()
        row = {"chamber": chamber, "year": year, "n_districts_used": len(share)}
        row.update(gm.run_all(share))
        rows.append(row)
    return pd.DataFrame(rows)
