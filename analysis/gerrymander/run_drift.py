"""
Phase B driver: quantify whether demographic/voting-pattern change since the
2021 maps were drawn has weakened or strengthened the partisan gerrymander,
for both TX House and Senate.

Outputs (output/gerrymander/):
  - drift_by_district.csv       per-district partisan + demographic drift,
                                 dummymander classification
  - map_trajectory.csv          map-level metric battery under 2020 vs 2024
                                 lean (the authoritative before/after)
  - drift_scatter_house.png, drift_scatter_senate.png
  - metrics_trajectory.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import drift as gd
import metrics as gm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "output" / "gerrymander"
OUT.mkdir(parents=True, exist_ok=True)


def statewide_shift_check(chamber: str) -> dict:
    """Sum district-level presidential vote totals back up to a statewide
    two-party share for 2020 and 2024, as an internal consistency check —
    the district file totals should reproduce the well-known statewide TX
    presidential shift toward Republicans between 2020 and 2024."""
    d2020 = pd.read_csv(RAW / "historical" / f"tx_presidential_{chamber}_2020.csv")
    d2024 = pd.read_csv(RAW / f"tx_presidential_{chamber}_2024.csv")

    dem_2020 = d2020["dem_votes"].sum()
    rep_2020 = d2020["rep_votes"].sum()
    share_2020 = dem_2020 / (dem_2020 + rep_2020)

    dem_2024 = d2024["harris_votes"].sum()
    rep_2024 = d2024["trump_votes"].sum()
    share_2024 = dem_2024 / (dem_2024 + rep_2024)

    return {
        "chamber": chamber,
        "statewide_dem_2p_share_2020": share_2020,
        "statewide_dem_2p_share_2024": share_2024,
        "statewide_shift_pp": (share_2024 - share_2020) * 100,
    }


def plot_drift_scatter(drift: pd.DataFrame, chamber: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = drift["effect_label"].map({
        "weakens_gerrymander": "#2f6fed",
        "strengthens_gerrymander": "#e0453c",
        "ambiguous_see_map_level": "#c98a1f",
        "flat": "#999999",
    })
    ax.scatter(drift["dem_share_2020"] * 100, drift["shift_2020_2024_pp"], c=colors, s=28, alpha=0.85)
    ax.axvline(50, color="#cccccc", lw=1)
    ax.axhline(0, color="#cccccc", lw=1)
    ax.set_xlabel("District Dem two-party share, 2020 (as-drawn baseline)")
    ax.set_ylabel("Shift toward Dem, 2020→2024 (pp)")
    ax.set_title(f"TX {chamber.title()}: drawn lean vs. subsequent trend")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=lbl)
        for lbl, c in [
            ("Weakens gerrymander (R seat trending D)", "#2f6fed"),
            ("Strengthens gerrymander (D seat trending D)", "#e0453c"),
            ("Ambiguous (D seat trending R)", "#c98a1f"),
            ("Flat (<1pp shift)", "#999999"),
        ]
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_trajectory(traj_all: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    metric_cols = {
        "eg_efficiency_gap": "Efficiency gap",
        "mm_mean_median": "Mean-median",
        "pb_partisan_bias": "Partisan bias",
        "decl_declination": "Declination",
    }
    for ax, chamber in zip(axes, ["house", "senate"]):
        sub = traj_all[traj_all["chamber"] == chamber].sort_values("year")
        for col, label in metric_cols.items():
            ax.plot(sub["year"], sub[col], marker="o", label=label)
        ax.axhline(0, color="#cccccc", lw=1)
        ax.set_title(f"TX {chamber.title()}")
        ax.set_xlabel("Presidential-lean baseline year")
        ax.set_xticks([2020, 2024])
    axes[0].set_ylabel("Metric value (positive = pro-R)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Has the map's structural pro-R bias grown or shrunk since 2021? (higher = more pro-R)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def spot_check_high_hispanic_districts(chamber: str, drift: pd.DataFrame) -> None:
    demo = gd.demographic_drift_table(chamber)
    merged = drift.merge(demo[["district", "pct_hispanic_now", "delta_pct_hispanic_pp"]], on="district")
    high_hisp = merged[merged["pct_hispanic_now"] > 80].sort_values("pct_hispanic_now", ascending=False)
    if high_hisp.empty:
        print(f"  [spot-check] {chamber}: no districts with >80% Hispanic CVAP found")
        return
    n_trending_r = (high_hisp["shift_2020_2024_pp"] < -1).sum()
    print(
        f"  [spot-check] {chamber}: {len(high_hisp)} districts with >80% Hispanic CVAP; "
        f"{n_trending_r} trended toward Republicans 2020->2024 (median shift "
        f"{high_hisp['shift_2020_2024_pp'].median():.1f} pp) -- expected direction given the "
        f"well-documented Rio Grande Valley Hispanic shift toward Trump."
    )


def main() -> None:
    all_drift = []
    all_traj = []

    for chamber in ["house", "senate"]:
        pdrift = gd.partisan_drift_table(chamber)
        classified = gd.classify_dummymander(pdrift)
        demo = gd.demographic_drift_table(chamber)
        merged = classified.merge(
            demo.drop(columns=["chamber"]), on="district", how="left"
        )
        all_drift.append(merged)

        traj = gd.map_level_trajectory(chamber, pdrift)
        all_traj.append(traj)

        plot_drift_scatter(classified, chamber, OUT / f"drift_scatter_{chamber}.png")

        check = statewide_shift_check(chamber)
        print(
            f"[consistency check] {chamber}: statewide Dem 2p share {check['statewide_dem_2p_share_2020']:.3f} "
            f"(2020) -> {check['statewide_dem_2p_share_2024']:.3f} (2024), shift {check['statewide_shift_pp']:+.1f}pp "
            f"(public reporting puts the actual TX statewide presidential shift 2020->2024 solidly Republican; "
            f"this should be in that ballpark as a sanity check on the district file aggregation)"
        )

        spot_check_high_hispanic_districts(chamber, classified)

        counts = classified["effect_label"].value_counts()
        print(f"  [dummymander summary] {chamber}:\n{counts.to_string()}\n")

    drift_df = pd.concat(all_drift, ignore_index=True)
    drift_df.to_csv(OUT / "drift_by_district.csv", index=False)

    traj_df = pd.concat(all_traj, ignore_index=True)
    traj_df.to_csv(OUT / "map_trajectory.csv", index=False)
    plot_metrics_trajectory(traj_df, OUT / "metrics_trajectory.png")

    print("=== Map-level trajectory: 2020 lean vs 2024 lean, same (current) district lines ===")
    print(
        traj_df[
            ["chamber", "year", "eg_efficiency_gap", "mm_mean_median", "pb_partisan_bias", "decl_declination"]
        ].to_string(index=False)
    )
    for chamber in ["house", "senate"]:
        sub = traj_df[traj_df["chamber"] == chamber].set_index("year")
        eg_delta = sub.loc[2024, "eg_efficiency_gap"] - sub.loc[2020, "eg_efficiency_gap"]
        direction = "STRENGTHENED (more pro-R)" if eg_delta > 0 else "WEAKENED (less pro-R)"
        print(f"  {chamber}: efficiency gap moved {eg_delta:+.4f} from 2020->2024 lean -> gerrymander {direction}")

    print(f"\nWrote: {OUT / 'drift_by_district.csv'}")
    print(f"Wrote: {OUT / 'map_trajectory.csv'}")
    print(f"Wrote: drift_scatter_house.png, drift_scatter_senate.png, metrics_trajectory.png")


if __name__ == "__main__":
    main()
