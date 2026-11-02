"""
statewide_pollster_history.py

Which pollsters have been right about Texas, this close to an election?

Uses every Texas statewide general-election poll 2014-2024 fielded in the last 35
days. All margins two-party. Three questions:
  1. Per-pollster signed bias and absolute error, raw and relative to the other
     polls of the same race (so a pollster is not credited for a cycle's shared miss).
  2. Walk-forward: using only earlier cycles, would house-effect correction or
     weighting by record have beaten a plain average in the next cycle?
  3. Does a high undecided share mean a pro-D or pro-R miss? (Matters for 2026
     down-ballot polls, which carry ~10% undecided.)

Writes data/raw/statewide/pollster_history.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "raw" / "historical"
OUT = ROOT / "data" / "raw" / "statewide"
ELECTION = {2014: "2014-11-04", 2016: "2016-11-08", 2018: "2018-11-06", 2020: "2020-11-03", 2022: "2022-11-08", 2024: "2024-11-05"}
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 120)


def canon(s: str) -> str:
    k = str(s).lower()
    for pat, name in [
        (r"univision", "Univision/YouGov"), (r"texas politics|university of texas/|uot/|texas tribune|^ut/", "UT/TPP (YouGov)"),
        (r"cbs", "CBS/YouGov"), (r"umass lowell", "UMass Lowell/YouGov"), (r"yougov blue", "YouGov Blue"), (r"yougov", "YouGov"),
        (r"ut tyler|tyler", "UT Tyler"), (r"siena|nyt|new york times", "NYT/Siena"), (r"quinnipiac", "Quinnipiac"),
        (r"emerson", "Emerson"), (r"data for progress", "Data for Progress"), (r"morning consult", "Morning Consult"),
        (r"public policy polling", "PPP"), (r"univ.* of houston|^uh|hobby", "U of Houston"),
        (r"texas southern|tsu", "Texas Southern"), (r"marist", "Marist"), (r"beacon|fox", "Fox (Beacon/Shaw)"),
        (r"lyceum", "Texas Lyceum"), (r"dixie", "Dixie Strategies"), (r"cnn|ssrs", "CNN/SSRS"), (r"rasmussen", "Rasmussen"),
        (r"activote", "ActiVote"), (r"atlas", "AtlasIntel"), (r"cygnal", "Cygnal"), (r"gravis", "Gravis"),
        (r"ipsos", "Ipsos"), (r"swayable", "Swayable"), (r"rmg", "RMG"), (r"civiqs", "Civiqs"), (r"trafalgar", "Trafalgar"),
        (r"insideradvantage", "InsiderAdvantage"), (r"quantus", "Quantus"), (r"big data", "Big Data Poll"),
    ]:
        if re.search(pat, k):
            return name
    return s


def load_polls() -> pd.DataFrame:
    p = pd.read_csv(HIST / "tx_statewide_polls_2014_2024.csv")
    r = pd.read_csv(HIST / "tx_statewide_results_2014_2024.csv")
    r["actual"] = (r.d_pct - r.r_pct) / (r.d_pct + r.r_pct) * 100
    p = p.dropna(subset=["r_pct", "d_pct", "field_end"]).merge(r[["cycle", "race", "actual"]], on=["cycle", "race"])
    p["end"] = pd.to_datetime(p.field_end, errors="coerce")
    p["ed"] = pd.to_datetime(p.cycle.map(ELECTION))
    p["days"] = (p.ed - p.end).dt.days
    p = p[(p.days >= 0) & (p.days <= 35)]
    p["margin"] = (p.d_pct - p.r_pct) / (p.d_pct + p.r_pct) * 100
    p["err"] = p.margin - p.actual
    p["und"] = 100 - p.r_pct - p.d_pct
    p["who"] = p.pollster.map(canon)
    p["rc"] = p.cycle.astype(str) + p.race
    p["partisan"] = p.partisan_flag.notna() & (p.partisan_flag.astype(str).str.strip() != "")
    # one poll per pollster per race: the latest
    return p.sort_values("end").drop_duplicates(["rc", "who"], keep="last")


def walk_forward(p: pd.DataFrame):
    rows = []
    for c in sorted(p.cycle.unique()):
        train, test = p[p.cycle < c], p[(p.cycle == c) & ~p.partisan]
        if train.empty:
            continue
        rel = train.err - train.groupby("rc").err.transform("mean")
        tau2 = 2.0 ** 2   # prior variance of a true house effect, margin points
        s = pd.DataFrame({"who": train.who, "rel": rel}).groupby("who").rel.agg(["mean", "count", "var"])
        s["shr"] = s["count"] / (s["count"] + 9.0 / tau2) * s["mean"]
        ab = train.assign(ae=rel.abs()).groupby("who").ae.mean()
        for rc, g in test.groupby("rc"):
            if len(g) < 3:
                continue
            base = g.margin.mean()
            house = (g.margin - g.who.map(s.shr).fillna(0)).mean()
            w = 1 / (g.who.map(ab).fillna(ab.median() if len(ab) else 4) ** 2)
            weighted = (g.margin * w).sum() / w.sum()
            rows.append({"cycle": c, "rc": rc, "n": len(g), "actual": g.actual.iloc[0], "plain": base - g.actual.iloc[0],
                         "house_adj": house - g.actual.iloc[0], "record_wtd": weighted - g.actual.iloc[0]})
    return pd.DataFrame(rows)


def main():
    p = load_polls()
    print(f"{len(p)} polls in the last 35 days, {p.rc.nunique()} race-cycles, {p.who.nunique()} pollsters, days median {p.days.median():.0f}")
    p["rel"] = p.err - p.groupby("rc").err.transform("mean")
    g = p.groupby("who").agg(polls=("err", "size"), cycles=("cycle", "nunique"), bias=("err", "mean"), mae=("err", lambda x: x.abs().mean()),
                             rel_bias=("rel", "mean"), rel_sd=("rel", "std"), years=("cycle", lambda x: ",".join(str(v)[2:] for v in sorted(set(x)))))
    g["shrunk_rel"] = g.polls / (g.polls + 9.0 / 4.0) * g.rel_bias
    g = g.sort_values("polls", ascending=False)
    print("\n=== Pollsters, Texas statewide, last 35 days (+ = too Democratic) ===")
    print(g[g.polls >= 2].round(1).to_string())
    g.reset_index().to_csv(OUT / "pollster_history.csv", index=False)

    # is the spread between pollsters more than sampling noise?
    q = p[p.groupby("who").err.transform("size") >= 3]
    obs = q.groupby("who").rel.mean().abs().mean()
    rng = np.random.default_rng(11)
    perm = []
    for _ in range(3000):
        shuffled = q.groupby("rc").rel.transform(lambda x: rng.permutation(x.values))
        perm.append(shuffled.groupby(q.who).mean().abs().mean())
    print(f"\nmean |house effect| among pollsters with 3+ polls: {obs:.2f}; permutation p = {(np.array(perm) >= obs).mean():.3f} (shuffling within race)")

    print("\n=== Walk-forward: average error of the race average by method ===")
    w = walk_forward(p)
    print(w.groupby("cycle")[["plain", "house_adj", "record_wtd"]].mean().round(2).to_string())
    for c in ("plain", "house_adj", "record_wtd"):
        print(f"  {c:<11} rms {np.sqrt((w[c] ** 2).mean()):.2f}  mean {w[c].mean():+.2f}  (races {len(w)})")

    print("\n=== Cycle and race error of the final-35-day plain average ===")
    avg = p[~p.partisan].groupby(["cycle", "race"]).agg(n=("err", "size"), err=("err", "mean"), und=("und", "mean")).reset_index()
    print(avg.round(1).to_string(index=False))
    a = avg[avg.n >= 2]
    cyc = a.groupby("cycle").err.mean()
    print("cycle means:", cyc.round(1).to_dict(), f"mean {cyc.mean():+.2f} sd {cyc.std():.2f}")
    print(f"within-cycle race sd {(a.err - a.groupby('cycle').err.transform('mean')).std():.2f}")
    mt = cyc[[c for c in cyc.index if c % 4 == 2]]
    print(f"midterms mean {mt.mean():+.2f}, presidential {cyc.drop(mt.index).mean():+.2f}")

    print("\n=== Undecided share vs error ===")
    b = p[~p.partisan].copy()
    b["und_rel"] = b.und - b.groupby("rc").und.transform("mean")
    b["err_rel"] = b.err - b.groupby("rc").err.transform("mean")
    slope = np.polyfit(b.und_rel, b.err_rel, 1)[0]
    print(f"within a race, +1pt undecided -> {slope:+.3f} margin err (n={len(b)}, r {np.corrcoef(b.und_rel, b.err_rel)[0, 1]:+.2f})")
    print(f"across race-cycles, err on und: {np.polyfit(a.und, a.err, 1)[0]:+.3f} (n={len(a)}, r {np.corrcoef(a.und, a.err)[0, 1]:+.2f})")
    print(f"down-ballot races: mean und {a[a.race.isin(['lt_governor', 'attorney_general', 'comptroller', 'railroad_commissioner'])].und.mean():.1f}, "
          f"mean err {a[a.race.isin(['lt_governor', 'attorney_general', 'comptroller', 'railroad_commissioner'])].err.mean():+.2f}")


if __name__ == "__main__":
    main()
