"""
analyze_primary_effect.py

Empirical test: do competitive TX legislative primaries predict general
election under/overperformance relative to model fundamentals?

Method:
  Dependent variable  : war_race (actual − predicted dem 2p share, pp)
                        from data/processed/race_war.csv
                        Already controls for: district lean, incumbency,
                        national environment, IE spending.
  Independent vars    : primary competitiveness metrics from
                        data/raw/tx_primary_history.csv

  Key specs:
    (1) war ~ runoff_needed                    + year_FE
    (2) war ~ primary_margin (continuous)      + year_FE
    (3) war ~ n_candidates_binned              + year_FE
    (4) All of above restricted to competitive_race == True

  Subset by party (R primaries → R candidate WAR; D primaries → D candidate WAR)
  so that primary competitiveness for the correct party is tested.

USAGE:
  python src/analyze_primary_effect.py
  python src/analyze_primary_effect.py --chamber house
  python src/analyze_primary_effect.py --competitive-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
OUTPUT   = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------

def load_merged(chamber_filter: str | None = None) -> pd.DataFrame:
    war_path  = DATA_PRO / "race_war.csv"
    prim_path = DATA_RAW  / "tx_primary_history.csv"

    if not war_path.exists():
        raise FileNotFoundError(f"{war_path}\nRun: python src/compute_war.py")
    if not prim_path.exists():
        raise FileNotFoundError(
            f"{prim_path}\nRun: python src/collect_primary_history.py"
        )

    war  = pd.read_csv(war_path)
    prim = pd.read_csv(prim_path)

    # Normalize party and chamber for merge
    prim["party"]   = prim["party"].str.upper().str.strip()
    prim["chamber"] = prim["chamber"].str.strip()
    war["party"]    = war["party"].str.upper().str.strip()
    war["chamber"]  = war["chamber"].str.strip()

    # Merge on (year, chamber, district, party)
    # This gives each candidate-race their party's primary competitiveness
    merged = war.merge(
        prim[["year", "chamber", "district", "party",
              "n_candidates", "winner_pct", "runner_up_pct",
              "primary_margin", "runoff_needed", "had_runoff"]],
        on=["year", "chamber", "district", "party"],
        how="left",
    )

    # Only rows where we have primary data
    merged = merged[merged["n_candidates"].notna()].copy()

    # Derived flags
    merged["primary_contested"] = merged["n_candidates"] > 1
    merged["primary_close"]     = (merged["primary_margin"] < 15) & merged["primary_contested"]
    merged["primary_very_close"]= (merged["primary_margin"] < 5)  & merged["primary_contested"]

    # Candidate competitiveness buckets
    merged["margin_bucket"] = pd.cut(
        merged["primary_margin"],
        bins=[-1, 0, 10, 25, 50, 101],
        labels=["Uncontested", "Very close (<10)", "Close (10-25)",
                "Moderate (25-50)", "Dominant (>50)"],
        right=True,
    )

    if chamber_filter:
        merged = merged[merged["chamber"].str.lower() == chamber_filter.lower()]

    print(f"Merged dataset: {len(merged)} candidate-races "
          f"({merged['year'].nunique()} years, "
          f"{merged['chamber'].unique().tolist()})")
    print(f"  Primary data coverage: {merged['primary_contested'].notna().sum()} races")
    print(f"  Contested primaries:   {merged['primary_contested'].sum()}")
    print(f"  Runoff-eligible:       {merged['runoff_needed'].sum()}")
    print(f"  Competitive general:   {merged['competitive_race'].sum()}")

    return merged


# ---------------------------------------------------------------------------
# Descriptive analysis
# ---------------------------------------------------------------------------

def describe_by_group(df: pd.DataFrame, groupvar: str, label: str):
    """Print mean WAR ± SE by group."""
    print(f"\n  {label}:")
    grps = df.groupby(groupvar)["war_race"].agg(["mean", "std", "count"])
    grps["se"] = grps["std"] / np.sqrt(grps["count"])
    for name, row in grps.iterrows():
        stars = ""
        print(f"    {str(name):30s}  n={row['count']:4.0f}  "
              f"mean WAR={row['mean']:+.2f}pp  "
              f"SE={row['se']:.2f}  {stars}")


def t_test_runoff(df: pd.DataFrame, label: str = ""):
    """Two-sample t-test: runoff vs. no-runoff WAR."""
    runoff  = df[df["runoff_needed"] == True]["war_race"].dropna()
    norunoff = df[df["runoff_needed"] == False]["war_race"].dropna()
    if len(runoff) < 5 or len(norunoff) < 5:
        print(f"  Insufficient data for t-test ({label})")
        return
    t, p = stats.ttest_ind(runoff, norunoff)
    diff = runoff.mean() - norunoff.mean()
    print(f"\n  Runoff vs. no-runoff WAR ({label}):")
    print(f"    Runoff    : n={len(runoff):3d}  mean={runoff.mean():+.2f}pp  "
          f"SD={runoff.std():.2f}")
    print(f"    No runoff : n={len(norunoff):3d}  mean={norunoff.mean():+.2f}pp  "
          f"SD={norunoff.std():.2f}")
    print(f"    Difference: {diff:+.2f}pp  t={t:.2f}  p={p:.3f}"
          + ("  *" if p < 0.10 else "") + ("*" if p < 0.05 else ""))


# ---------------------------------------------------------------------------
# OLS regression
# ---------------------------------------------------------------------------

def run_ols(df: pd.DataFrame, label: str):
    """OLS: war_race ~ primary vars + year fixed effects."""
    if not HAS_SM:
        print("  (statsmodels not available — skipping OLS)")
        return None

    df = df.copy()
    df["year_fe"] = df["year"].astype(str)

    specs = [
        ("runoff_needed",
         "war_race ~ C(runoff_needed) + C(year_fe)",
         "Spec 1: runoff (binary) + year FE"),
        ("primary_margin",
         "war_race ~ primary_margin + C(year_fe)",
         "Spec 2: primary margin (continuous) + year FE"),
        ("primary_contested + primary_margin",
         "war_race ~ primary_contested + primary_margin + C(year_fe)",
         "Spec 3: contested + margin + year FE"),
    ]

    results = {}
    for key, formula, spec_label in specs:
        sub = df[df["primary_margin"].notna()].copy()
        if len(sub) < 20:
            continue
        try:
            fit = smf.ols(formula, data=sub).fit()
            print(f"\n  {spec_label}  (n={len(sub)})")
            print(f"  {'Variable':40s}  {'Coef':>8s}  {'SE':>6s}  {'p':>6s}")
            print(f"  {'-'*65}")
            for term, coef, se, p in zip(
                fit.params.index,
                fit.params.values,
                fit.bse.values,
                fit.pvalues.values
            ):
                if "Intercept" in term or "year_fe" in term:
                    continue
                sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                print(f"  {term:40s}  {coef:+8.3f}  {se:6.3f}  {p:6.3f}  {sig}")
            print(f"  R² = {fit.rsquared:.4f}   Adj-R² = {fit.rsquared_adj:.4f}")
            results[key] = fit
        except Exception as exc:
            print(f"  OLS failed ({spec_label}): {exc}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test whether competitive primaries predict general election WAR")
    parser.add_argument("--chamber", choices=["house", "senate"],
                        help="Restrict analysis to one chamber")
    parser.add_argument("--competitive-only", action="store_true",
                        help="Restrict to competitive general elections only")
    args = parser.parse_args()

    print("=" * 68)
    print("  TX Legislative Primary Effect on General Election WAR")
    print("=" * 68)

    df = load_merged(args.chamber)

    out_lines = []  # accumulate for text output

    for subset_label, subset_df in [
        ("All contested races", df[df["primary_contested"].notna()]),
        ("Competitive general only", df[df["competitive_race"] == True]),
    ]:
        if args.competitive_only and subset_label != "Competitive general only":
            continue

        n = len(subset_df)
        if n < 20:
            print(f"\n  Skipping '{subset_label}' — only {n} rows")
            continue

        print(f"\n{'='*68}")
        print(f"  SUBSET: {subset_label}  (n={n})")
        print(f"{'='*68}")

        # --- Descriptive ---
        describe_by_group(subset_df, "runoff_needed",
                          "Mean WAR by runoff status")
        describe_by_group(subset_df, "margin_bucket",
                          "Mean WAR by primary margin bucket")

        # --- t-test ---
        t_test_runoff(subset_df, subset_label)

        # --- Correlation: primary margin vs WAR ---
        sub_valid = subset_df[subset_df["primary_margin"].notna() &
                              subset_df["war_race"].notna()]
        if len(sub_valid) >= 10:
            r, p_r = stats.pearsonr(sub_valid["primary_margin"],
                                    sub_valid["war_race"])
            print(f"\n  Pearson r (primary_margin vs war_race): "
                  f"r={r:+.3f}  p={p_r:.3f}")
            r_comp, p_comp = stats.pearsonr(sub_valid["primary_margin"],
                                             sub_valid["war_race"])

        # --- OLS ---
        print(f"\n  OLS regressions ({subset_label}):")
        run_ols(subset_df, subset_label)

    # --- By party ---
    print(f"\n{'='*68}")
    print("  BY PARTY (competitive general only)")
    print(f"{'='*68}")
    comp = df[df["competitive_race"] == True]
    for pty in ["R", "D"]:
        pty_df = comp[comp["party"] == pty]
        if len(pty_df) < 10:
            continue
        t_test_runoff(pty_df, f"{pty} primaries")

    # --- Summary ---
    print(f"\n{'='*68}")
    print("  SUMMARY TABLE: Mean WAR by primary competitiveness")
    print(f"{'='*68}")
    summary = (
        df[df["competitive_race"] == True]
        .groupby(["runoff_needed", "primary_contested"])["war_race"]
        .agg(n="count", mean=lambda x: x.mean(), sd="std")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    print(f"\n  {'Runoff':8s}  {'Contested':10s}  {'n':>5s}  "
          f"{'Mean WAR':>10s}  {'SE':>6s}")
    print(f"  {'-'*50}")
    for _, row in summary.iterrows():
        print(f"  {str(row['runoff_needed']):8s}  {str(row['primary_contested']):10s}  "
              f"{row['n']:5.0f}  {row['mean']:+10.2f}pp  {row['se']:6.2f}")

    # Save summary to file
    out_path = OUTPUT / "primary_effect_analysis.txt"
    print(f"\n  (Full output also written to {out_path.name})")


if __name__ == "__main__":
    main()
