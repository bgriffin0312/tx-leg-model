"""
whole_election_estimates.py

Measures what the whole-election model needs from TLC VTD returns: the same precincts, every
office. Total votes in a race cannot change because of ticket splitting, only because of
incomplete ballots, so drop-off is separable from splitting on aggregate data.

  1. Roll-off from the top of the ticket, and whether it falls on D or R top-of-ticket voters
  2. What that does to a race's margin; the rest of the gap to the top of the ticket is splitting
  3. Does a race move one for one with its own precincts' party vote between cycles?

Output: output/whole_election_rolloff.csv, output/whole_election_swing.csv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import statewide_results as sr  # noqa: E402

YEARS = [2012, 2014, 2016, 2018, 2020, 2022, 2024]
TOP = {2012: "President", 2014: "Governor", 2016: "President", 2018: "Governor", 2020: "President", 2022: "Governor", 2024: "President"}
STATEWIDE = r"^(Lt\.? ?Gov|Attorney Gen|Comptroller|Land Comm|Ag(riculture)? Comm|RR Comm|U\.S\. Sen|Governor)"
JUDICIAL = r"^(Sup Ct|CCA)"
LOCAL = r"^(U\.S\. Rep|State Sen|State Rep) \d+$|^Harris Co Judge$"


def tier(o):
    if o.startswith("U.S. Rep"):
        return "us_house"
    if o.startswith("State Sen"):
        return "state_senate"
    if o.startswith("State Rep"):
        return "state_house"
    if o.startswith("Sup Ct") or o.startswith("CCA"):
        return "judicial"
    if o == "Harris Co Judge":
        return "county_judge"
    return "statewide"


def wide(df, office):
    g = df[df.Office == office]
    p = g.pivot_table(index="cntyvtd", columns="Party", values="Votes", aggfunc="sum").fillna(0)
    out = pd.DataFrame(index=p.index)
    out["d"] = p["D"] if "D" in p else 0.0
    out["r"] = p["R"] if "R" in p else 0.0
    out["tot"] = p.sum(axis=1)
    return out


def judicial_vtd(df):
    """Equal-weight mean margin of the contested Supreme Court and CCA races, same VTDs."""
    offices = [o for o in df.Office.unique() if re.match(JUDICIAL, o)]
    parts = []
    for o in offices:
        w = wide(df, o)
        if w.d.sum() > 0 and w.r.sum() > 0:
            parts.append(w[["d", "r"]])
    if not parts:
        return None
    s = sum(p.reindex(parts[0].index.union(p.index)).fillna(0) for p in parts) / len(parts)
    return s


def rolloff(year, df):
    top = wide(df, TOP[year]).rename(columns={"d": "D", "r": "R", "tot": "T"})
    j = judicial_vtd(df)
    rows = []
    main_county = df.groupby(["Office", "County"]).Votes.sum().reset_index().sort_values("Votes").groupby("Office").County.last()
    offices = [o for o in df.Office.unique() if re.match(STATEWIDE, o) or re.match(LOCAL, o) or re.match(JUDICIAL, o)]
    for o in offices:
        if o == TOP[year]:
            continue
        w = wide(df, o)
        if w.d.sum() == 0 or w.r.sum() == 0:
            continue
        x = w.join(top, how="inner")
        x = x[x["T"] > 0]
        if len(x) < 5:
            continue
        missing = (x["T"] - x.tot).values
        O = (x["T"] - x.D - x.R).values
        X = np.column_stack([x.D.values, x.R.values, O])
        wts = 1.0 / np.maximum(x["T"].values, 25.0)
        f = sm.WLS(missing, X, weights=wts).fit()
        a_d, a_r = f.params[0], f.params[1]
        D, R = x.D.sum(), x.R.sum()
        m_top = (D - R) / (D + R) * 100
        D2, R2 = D * (1 - a_d), R * (1 - a_r)
        m_kept = (D2 - R2) / (D2 + R2) * 100
        m_race = (x.d.sum() - x.r.sum()) / (x.d.sum() + x.r.sum()) * 100
        rec = {"year": year, "office": o, "tier": tier(o), "county": main_county[o], "vtds": len(x), "votes_top": x["T"].sum(), "rolloff": 1 - x.tot.sum() / x["T"].sum(),
               "a_d": a_d, "a_r": a_r, "m_top": m_top, "m_race": m_race, "rolloff_effect": m_kept - m_top, "split": m_race - m_kept}
        if j is not None:
            y = x.join(j.rename(columns={"d": "jd", "r": "jr"}), how="inner")
            if y.jd.sum() + y.jr.sum() > 0:
                rec["m_j"] = (y.jd.sum() - y.jr.sum()) / (y.jd.sum() + y.jr.sum()) * 100
                rec["m_race_j"] = (y.d.sum() - y.r.sum()) / (y.d.sum() + y.r.sum()) * 100
        rows.append(rec)
    return rows


def county_judge_test():
    """Same county, consecutive cycles: county judge margin against the county's judicial vote."""
    import contextlib
    import io
    import county_judge_model as cj
    with contextlib.redirect_stdout(io.StringIO()):
        t, _ = cj.race_table()
    c = t[t.office == "county_judge"].copy()
    c["m"] = (2 * c.y - 1) * 100
    jr = {}
    for yr in (2014, 2018, 2022):
        df = sr.load(yr)
        j = judicial_vtd(df)
        cty = df.groupby("cntyvtd").County.first().str.lower().str.strip()
        g = j.join(cty).groupby("County")[["d", "r"]].sum()
        jr[yr] = (g.d - g.r) / (g.d + g.r) * 100
    rows = []
    for k, g in c.groupby("county"):
        g = g.set_index("year")
        for a, b in ((2014, 2018), (2018, 2022)):
            if a in g.index and b in g.index and k in jr[a].index:
                rows.append({"county": k, "pair": f"{a}-{b}", "dm": g.m[b] - g.m[a], "dj": jr[b][k] - jr[a][k]})
    x = pd.DataFrame(rows)
    f = sm.OLS(x.dm, sm.add_constant(x.dj)).fit()
    print(f"county judge on county judicial vote, same county: n {len(x)} slope {f.params['dj']:.2f} (se {f.bse['dj']:.2f}) resid sd {np.sqrt(f.scale):.2f}")
    x = x.assign(a=x.dm - x.groupby("pair").dm.transform("mean"), b=x.dj - x.groupby("pair").dj.transform("mean"))
    f = sm.OLS(x.a, x.b).fit()
    print(f"  within pair: slope {f.params['b']:.2f} (se {f.bse['b']:.2f})")


def main():
    rows = []
    for y in YEARS:
        print("loading", y)
        rows += rolloff(y, sr.load(y))
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "output" / "whole_election_rolloff.csv", index=False)

    v = r[r.tier != "judicial"]
    print("\n=== Roll-off and margin effects, contested races, by year and office tier ===")
    g = v.groupby(["year", "tier"])
    t = g.apply(lambda d: pd.Series({
        "n": len(d),
        "rolloff_pct": np.average(d.rolloff, weights=d.votes_top) * 100,
        "a_d_pct": np.average(d.a_d, weights=d.votes_top) * 100,
        "a_r_pct": np.average(d.a_r, weights=d.votes_top) * 100,
        "rolloff_effect": np.average(d.rolloff_effect, weights=d.votes_top),
        "rolloff_eff_sd": d.rolloff_effect.std(),
        "split_mean": np.average(d.split, weights=d.votes_top),
        "split_sd": d.split.std()}), include_groups=False)
    print(t.round(2).to_string())

    print("\n=== Statewide offices in midterms ===")
    s = v[(v.tier == "statewide") & v.year.isin([2014, 2018, 2022])].copy()
    print(s[["year", "office", "rolloff", "a_d", "a_r", "m_top", "m_race", "rolloff_effect", "split"]].round(3).to_string(index=False))

    # Same seat, consecutive cycles: does the race move with the judicial vote on its own VTDs?
    d = r[r.tier.isin(["us_house", "state_senate", "state_house"]) & r.m_j.notna()].copy()
    d = d[d.m_race.abs() < 60].sort_values(["office", "year"])
    swing = []
    for o, g in d.groupby("office"):
        g = g.set_index("year")
        for a, b in ((2012, 2014), (2014, 2016), (2016, 2018), (2018, 2020), (2022, 2024)):
            if a in g.index and b in g.index:
                swing.append({"office": o, "tier": g.tier.iloc[0], "county": g.county.iloc[0], "pair": f"{a}-{b}", "drace": g.m_race_j[b] - g.m_race_j[a], "dj": g.m_j[b] - g.m_j[a]})
    w = pd.DataFrame(swing)
    w.to_csv(ROOT / "output" / "whole_election_swing.csv", index=False)
    print("\n=== Race change on change in judicial vote, same VTDs (slope 1 = one electorate) ===")
    for lab, x in [("all", w)] + [(t_, w[w.tier == t_]) for t_ in w.tier.unique()] + [(p, w[w.pair == p]) for p in sorted(w.pair.unique())]:
        f = sm.OLS(x.drace, sm.add_constant(x.dj)).fit()
        print(f"{lab:<14} n {len(x):>4} slope {f.params['dj']:.2f} (se {f.bse['dj']:.2f}) resid sd {np.sqrt(f.scale):.2f}")
    x = w.assign(dm=w.drace - w.groupby("pair").drace.transform("mean"), dj_=w.dj - w.groupby("pair").dj.transform("mean"))
    f = sm.OLS(x.dm, x.dj_).fit()
    print(f"within pair    n {len(x):>4} slope {f.params['dj_']:.2f} (se {f.bse['dj_']:.2f})")
    # How much of a race's own error is its place, not its candidates: the local share of the
    # variance, and the part of that a whole county shares.
    v_local = (f.params["dj_"] * x.dj_).var()
    v_split = f.resid.var()
    multi = x[x.groupby(["pair", "county"]).office.transform("count") >= 3]
    cm = multi.groupby(["pair", "county"]).dj_.transform("mean")
    k = multi.groupby(["pair", "county"]).office.transform("count")
    v_within = ((multi.dj_ - cm) ** 2 * (k / (k - 1))).mean()
    icc = max(0.0, 1 - v_within / multi.dj_.var())
    est = {"slope": float(sm.OLS(w.drace, sm.add_constant(w.dj)).fit().params["dj"]), "slope_within": float(f.params["dj_"]),
           "local_share": float(v_local / (v_local + v_split)), "county_icc": float(icc), "gap_tier_sd": {}}
    print(f"local share of a race's change {est['local_share']:.2f}; county share of that {icc:.2f} (n {len(multi)}); lambda {est['local_share'] * icc:.3f}")
    y = r[r.tier.isin(["us_house", "state_house", "state_senate"]) & r.m_j.notna() & (r.m_race.abs() < 60)].copy()
    y["gap"] = y.m_race_j - y.m_j
    for t_, g in y.groupby("tier"):
        est["gap_tier_sd"][t_] = float(g.groupby("year").gap.mean().std())
    print("tier cycle sd of the gap to the judicial vote:", {k_: round(v_, 2) for k_, v_ in est["gap_tier_sd"].items()})
    # Competitive House seats share a swing the state does not: suburbs in 2018, say. Seats
    # within 20 at the first election of each pair; the level sd is the rms change over root 2.
    sw = pd.read_csv(ROOT / "data" / "raw" / "statewide" / "results_statewide.csv")
    jy = sw[(sw.race == "judicial") & sw.contested].groupby("year").d_margin.mean()
    h = y[y.tier == "state_house"]
    ex = []
    for a_, b_ in ((2012, 2014), (2014, 2016), (2016, 2018), (2018, 2020), (2022, 2024)):
        p_, q_ = h[h.year == a_].set_index("office"), h[h.year == b_].set_index("office")
        k_ = [o for o in p_.index.intersection(q_.index) if abs(p_.m_race[o]) < 20]
        ex.append((q_.m_race_j[k_] - p_.m_race_j[k_]).mean() - (jy[b_] - jy[a_]))
    est["competitive_sd"] = float(np.sqrt(np.mean(np.square(ex))) / np.sqrt(2))
    print("competitive-seat excess swing by pair:", np.round(ex, 2), "level sd", round(est["competitive_sd"], 2))
    (ROOT / "output" / "whole_election_estimates.json").write_text(json.dumps(est, indent=1))
    county_judge_test()
    b = w.groupby("pair")[["drace", "dj"]].mean()
    print("pair means:\n", b.round(2).to_string())
    f = sm.OLS(b.drace, sm.add_constant(b.dj)).fit()
    print(f"between cycles n {len(b)} slope {f.params['dj']:.2f} (se {f.bse['dj']:.2f})")


if __name__ == "__main__":
    main()
