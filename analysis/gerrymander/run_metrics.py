"""
Phase A driver: run the standard partisan-symmetry metric battery on the
enacted (2021-drawn) Texas House and Senate maps.

Two parallel indices are computed for each chamber, because either one alone
has a known weakness:

  - "pres_2024": 2024 presidential two-party share by district
    (data/raw/tx_presidential_{house,senate}_2024.csv). Defined for every
    district (no uncontested-race problem), but measures presidential
    preference, not legislative voting behavior.

  - "actual_last": each district's own most recent legislative election
    result (districts_2026.csv: last_election_r_pct/d_pct/contested).
    Reflects real legislative voting, but many TX legislative races are
    uncontested; uncontested districts are imputed using that district's
    2024 presidential two-party share as a proxy (standard practice to avoid
    the false 0%/100% signal an uncontested race would otherwise inject —
    see metrics.py module docstring).

Agreement between the two indices is itself a robustness check: if EG/MM/PB/
declination point the same direction and similar magnitude under both, the
finding isn't an artifact of which index was chosen.

Outputs (all under output/gerrymander/):
  - metrics_by_year.csv       one row per (chamber, index)
  - district_detail_2024.csv  per-district dem_share (both indices) for reuse
  - seats_votes_house.png, seats_votes_senate.png
  - margin_distribution.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metrics as gm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "output" / "gerrymander"
OUT.mkdir(parents=True, exist_ok=True)


def load_pres_2024(chamber: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"tx_presidential_{chamber}_2024.csv")
    return df[["district", "dem_pres_2p_baseline"]].rename(
        columns={"dem_pres_2p_baseline": "dem_share_pres_2024"}
    )


def load_actual_last(chamber: str, districts: pd.DataFrame) -> pd.DataFrame:
    sub = districts[districts["chamber"].str.lower() == chamber].copy()
    contested = sub["last_election_contested"].astype(str) == "True"
    actual = sub["last_election_d_pct"] / 100.0
    imputed = sub["dem_pres_2p_baseline"]
    sub["dem_share_actual_last"] = np.where(contested, actual, imputed)
    sub["actual_imputed"] = ~contested
    return sub[["district", "dem_share_actual_last", "actual_imputed", "last_election_year"]]


def build_district_table(chamber: str, districts: pd.DataFrame) -> pd.DataFrame:
    pres = load_pres_2024(chamber)
    actual = load_actual_last(chamber, districts)
    out = pres.merge(actual, on="district", how="outer")
    out["chamber"] = chamber
    return out


def battery_row(chamber: str, index_name: str, dem_share: np.ndarray) -> dict:
    row = {"chamber": chamber, "index": index_name, "n_districts": len(dem_share)}
    row.update(gm.run_all(dem_share))
    return row


def plot_seats_votes(chamber: str, dem_share: np.ndarray, path: Path) -> None:
    svc = gm.seats_votes_curve(dem_share)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(svc["statewide_dem_share"], svc["dem_seat_share"], color="#2f6fed", lw=2)
    ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1, label="Proportional (no bias)")
    ax.axvline(0.5, color="#cccccc", lw=1)
    ax.axhline(0.5, color="#cccccc", lw=1)
    statewide = dem_share.mean()
    actual_seats = (dem_share >= 0.5).mean()
    ax.scatter([statewide], [actual_seats], color="#e0453c", zorder=5, label="Enacted map, actual vote")
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Statewide Democratic two-party vote share")
    ax.set_ylabel("Democratic seat share")
    ax.set_title(f"Seats-votes curve — TX {chamber.title()} (enacted map, uniform swing)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_margin_distribution(house_share: np.ndarray, senate_share: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, share, label in zip(axes, [house_share, senate_share], ["House (150)", "Senate (31)"]):
        margin = share - 0.5  # positive = Dem win, negative = Rep win
        ax.hist(margin, bins=30, color="#2f6fed", edgecolor="white")
        ax.axvline(0, color="#e0453c", lw=1.5)
        ax.set_title(f"TX {label}: district margin distribution (2024 pres)")
        ax.set_xlabel("Dem margin (negative = R-won)")
    axes[0].set_ylabel("Number of districts")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sanity_check(chamber: str, dem_share_pres: np.ndarray, known_r_seats: int, known_d_seats: int) -> None:
    thresholded_d = int((dem_share_pres >= 0.5).sum())
    thresholded_r = len(dem_share_pres) - thresholded_d
    print(
        f"  [sanity] {chamber}: 2024-pres-lean would elect {thresholded_r}R-{thresholded_d}D "
        f"vs actual current composition {known_r_seats}R-{known_d_seats}D "
        f"(gap expected from incumbency/candidate effects; large gap flags a data problem)"
    )


def main() -> None:
    gm.self_test(verbose=False)  # re-verify sign conventions before trusting real output
    districts = pd.read_csv(PROCESSED / "districts_2026.csv")

    all_rows = []
    detail_frames = []
    share_by_chamber = {}

    for chamber in ["house", "senate"]:
        det = build_district_table(chamber, districts)
        detail_frames.append(det)

        pres_share = det["dem_share_pres_2024"].to_numpy()
        actual_share = det["dem_share_actual_last"].to_numpy()
        share_by_chamber[chamber] = pres_share

        all_rows.append(battery_row(chamber, "pres_2024", pres_share))
        all_rows.append(battery_row(chamber, "actual_last_election", actual_share))

        plot_seats_votes(chamber, pres_share, OUT / f"seats_votes_{chamber}.png")

    # Known current composition per CLAUDE.md: House 88R-62D, Senate 20R-11D.
    sanity_check("house", share_by_chamber["house"], known_r_seats=88, known_d_seats=62)
    sanity_check("senate", share_by_chamber["senate"], known_r_seats=20, known_d_seats=11)

    plot_margin_distribution(share_by_chamber["house"], share_by_chamber["senate"], OUT / "margin_distribution.png")

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(OUT / "metrics_by_year.csv", index=False)

    detail_df = pd.concat(detail_frames, ignore_index=True)
    detail_df.to_csv(OUT / "district_detail_2024.csv", index=False)

    print("\n=== Phase A summary (positive = pro-Republican, negative = pro-Democratic) ===")
    print(
        metrics_df[
            ["chamber", "index", "n_districts", "eg_efficiency_gap", "mm_mean_median",
             "pb_partisan_bias", "decl_declination", "responsiveness_at_status_quo"]
        ].to_string(index=False)
    )
    print(f"\nWrote: {OUT / 'metrics_by_year.csv'}")
    print(f"Wrote: {OUT / 'district_detail_2024.csv'}")
    print(f"Wrote: {OUT / 'seats_votes_house.png'}, seats_votes_senate.png, margin_distribution.png")


if __name__ == "__main__":
    main()
