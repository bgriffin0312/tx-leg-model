"""
Phase C, step 3: is the enacted map a statistical outlier relative to the
ReCom ensemble of neutrally-drawn alternative maps?

Run with the dedicated Python 3.12 venv:
  .venv-ensemble\\Scripts\\python.exe analyze.py --chamber house --steps 1000

Reads output/gerrymander/ensemble_{chamber}_steps{N}.csv (written by
run_chain.py, which includes the enacted plan as step 0 / is_enacted=True)
and reports where the enacted map falls in the ensemble distribution for
Dem seats and efficiency gap, under both 2020 and 2024 presidential lean.

CAVEATS this analysis does NOT control for (state clearly in any writeup):
  - Voting Rights Act minority-opportunity districts: this is an
    UNCONSTRAINED ensemble. It has no preference for preserving
    Hispanic-/Black-opportunity districts, so its "neutral" Dem-seat
    baseline may run higher than a VRA-compliant neutral map would, which
    would make the enacted map look like less of an outlier than it
    actually is relative to a legally realistic alternative.
  - Texas House county-line rule (Tex. Const. art. III, sec. 26): county
    boundaries must be respected where population allows. This ensemble
    does not enforce that constraint, so it is a looser neutral baseline
    than the House's actual legal constraints -- again biasing toward
    UNDERSTATING how much of an outlier the enacted House map is.
  Both caveats point the same direction: this v1 ensemble is a conservative
  (lower-bound) test of how extreme the enacted maps are.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "output" / "gerrymander"


def percentile_of_enacted(ensemble: pd.Series, enacted_value: float) -> float:
    """% of ensemble plans with a value <= the enacted plan's value."""
    return float((ensemble <= enacted_value).mean() * 100)


def plot_outlier_histogram(df: pd.DataFrame, col: str, chamber: str, year: str, path: Path, xlabel: str) -> None:
    enacted = df[df["is_enacted"]].iloc[0]
    ensemble = df[~df["is_enacted"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(ensemble[col], bins=30, color="#2f6fed", edgecolor="white", alpha=0.85, label="Ensemble (neutral ReCom plans)")
    ax.axvline(enacted[col], color="#e0453c", lw=2.5, label="Enacted map")
    pct = percentile_of_enacted(ensemble[col], enacted[col])
    ax.set_title(f"TX {chamber.title()}, {year} lean: enacted map is at the {pct:.0f}th percentile\n({len(ensemble)}-plan ensemble)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of ensemble plans")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {chamber} {year} {col}: enacted={enacted[col]:.4f}, ensemble mean={ensemble[col].mean():.4f}, "
          f"enacted percentile={pct:.1f} (0=most pro-D end, 100=most pro-R end of the ensemble spread)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chamber", choices=["house", "senate"], required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=None,
                         help="Match the --epsilon used in run_chain.py if it was non-default (filename suffix)")
    args = parser.parse_args()

    eps_suffix = "" if args.epsilon is None else f"_eps{args.epsilon}"
    path = OUT / f"ensemble_{args.chamber}_steps{args.steps}{eps_suffix}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path} -- run run_chain.py first with matching --steps")
    df = pd.read_csv(path)

    n_ensemble = (~df["is_enacted"]).sum()
    print(f"=== {args.chamber.title()}: enacted map vs {n_ensemble}-plan ensemble ===")

    summary_rows = []
    for year, seat_col, eg_col in [("2024", "y24_pack_n_dem_seats", "y24_eg_efficiency_gap"),
                                     ("2020", "y20_pack_n_dem_seats", "y20_eg_efficiency_gap")]:
        plot_outlier_histogram(df, seat_col, args.chamber, year, OUT / f"outlier_{args.chamber}_{year}_seats.png", f"Democratic seats ({year} presidential lean)")
        plot_outlier_histogram(df, eg_col, args.chamber, year, OUT / f"outlier_{args.chamber}_{year}_eg.png", f"Efficiency gap ({year} lean, positive = pro-R)")

        enacted = df[df["is_enacted"]].iloc[0]
        ensemble = df[~df["is_enacted"]]
        summary_rows.append({
            "chamber": args.chamber, "year": year,
            "enacted_dem_seats": enacted[seat_col], "ensemble_mean_dem_seats": ensemble[seat_col].mean(),
            "enacted_eg": enacted[eg_col], "ensemble_mean_eg": ensemble[eg_col].mean(),
            "eg_percentile": percentile_of_enacted(ensemble[eg_col], enacted[eg_col]),
            "n_ensemble_plans": n_ensemble,
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT / f"outlier_summary_{args.chamber}.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote: {summary_path}")
    print(f"Wrote: outlier_{args.chamber}_2024_seats.png, outlier_{args.chamber}_2024_eg.png, outlier_{args.chamber}_2020_seats.png, outlier_{args.chamber}_2020_eg.png")


if __name__ == "__main__":
    main()
