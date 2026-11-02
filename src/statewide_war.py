"""
statewide_war.py

Candidate strength for statewide and congressional races, on one yardstick.

A candidate's raw overperformance is their margin minus the Supreme Court / CCA
vote on the SAME VTDs in the SAME election (the party vote with the candidate
stripped out). This reaches back to 2002 because TLC returns carry surnames for
every State Rep, State Senate and U.S. House contest.

  district WAR  = overperformance, signed toward the candidate's party, after
                  removing cycle, incumbency and district profile (lean, lean
                  squared, Spanish-surname share)
  statewide dev = statewide race margin minus that year's judicial composite,
                  signed toward the candidate's party, after race-tier and
                  incumbency means

Then the transfer tests: does district WAR survive a move to a different district
(movers), and does it predict statewide dev (legislators who ran statewide)?

Writes data/raw/statewide/district_war_long.csv, statewide_dev.csv.
Prints the coefficients src/statewide_model.py uses.
"""

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statewide_results import CACHE, OUT, ZIPS, load  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)

# (year statewide, race, party, TLC surname, prior office regex)  - nominees who held
# or ran for a district office in the 2002+ returns
FIRST_TIMERS = [
    (2006, "governor", "D", "Bell", "U.S. Rep"), (2006, "ag_commissioner", "R", "Staples", "State Sen"),
    (2008, "us_senate", "D", "Noriega", "State Rep"), (2012, "us_senate", "D", "Sadler", "State Rep"),
    (2014, "governor", "D", "Davis", "State Sen"), (2014, "lt_governor", "D", "Van De Putte", "State Sen"),
    (2014, "lt_governor", "R", "Patrick", "State Sen"), (2014, "attorney_general", "R", "Paxton", "State (Rep|Sen)"),
    (2014, "comptroller", "R", "Hegar", "State Sen"), (2014, "ag_commissioner", "R", "Miller", "State Rep"),
    (2016, "railroad_commissioner", "R", "Christian", "State Rep"), (2018, "us_senate", "D", "O'Rourke", "U.S. Rep"),
    (2022, "land_commissioner", "R", "Buckingham", "State Sen"), (2024, "us_senate", "D", "Allred", "U.S. Rep"),
]
# 2026 nominees making a first statewide run, with their TLC surname and office
NOMINEES_2026 = [
    ("governor", "D", "Hinojosa", "State Rep", 49), ("lt_governor", "D", "Goodwin", "State Rep", 47),
    ("attorney_general", "D", "Johnson", "State Sen", 16), ("attorney_general", "R", "Middleton", "State (Rep|Sen)", None),
    ("comptroller", "D", "Eckhardt", "State Sen", 14), ("comptroller", "R", "Huffines", "State Sen", 16),
    ("railroad_commissioner", "D", "Rosenthal", "State Rep", 135), ("us_senate", "D", "Talarico", "State Rep", None),
]


def district_rows(year: int) -> pd.DataFrame:
    df = load(year)
    jud = [o for o in df.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)
           and {"D", "R"} <= set(df[df.Office == o].Party)]
    jv = df[df.Office.isin(jud) & df.Party.isin(["D", "R"])]
    jv = jv.groupby(["cntyvtd", "Party"]).Votes.sum().unstack(fill_value=0) / len(jud)
    with zipfile.ZipFile(CACHE / ZIPS[year]) as z:
        vr = pd.read_csv(io.BytesIO(z.read(f"{year}_General_Election_VRTO.csv")), encoding="utf-8-sig", low_memory=False)
    vr.columns = [c.lower() for c in vr.columns]
    vr["cntyvtd"] = vr.cntyvtd.astype(str).str.strip()
    ssv = next((c for c in vr.columns if "spanish" in c), None)
    tot = next((c for c in vr.columns if c in ("totalvr", "vr")), None)
    vr = vr.set_index("cntyvtd")
    rows = []
    d = df[df.Office.str.match(r"^(State Rep|State Sen|U\.S\. Rep) \d+")]
    for office, g in d.groupby("Office"):
        t = g.groupby(["Party", "Name", "Incumbent"], dropna=False).Votes.sum().reset_index()
        dd, rr = t[t.Party == "D"], t[t.Party == "R"]
        if len(dd) != 1 or len(rr) != 1 or dd.Votes.iloc[0] <= 0 or rr.Votes.iloc[0] <= 0:
            continue
        vv = g.groupby("cntyvtd").Votes.sum()
        keep = vv[vv > 0].index
        j = jv.reindex(keep).fillna(0).sum()
        jm = (j.get("D", 0) - j.get("R", 0)) / max(j.get("D", 0) + j.get("R", 0), 1) * 100
        ss = np.nan
        if ssv and tot:
            x = vr.reindex(keep)
            w = pd.to_numeric(x[tot], errors="coerce").fillna(0)
            ss = float((pd.to_numeric(x[ssv], errors="coerce").fillna(0) * w).sum() / max(w.sum(), 1))
        m = (dd.Votes.iloc[0] - rr.Votes.iloc[0]) / (dd.Votes.iloc[0] + rr.Votes.iloc[0]) * 100
        rows.append({"year": year, "office": office, "type": office.rsplit(" ", 1)[0], "dist": int(office.rsplit(" ", 1)[1]),
                     "d_name": dd.Name.iloc[0], "r_name": rr.Name.iloc[0],
                     "d_inc": str(dd.Incumbent.iloc[0]).upper().startswith("Y"),
                     "r_inc": str(rr.Incumbent.iloc[0]).upper().startswith("Y"),
                     "margin": m, "jud": jm, "ssvr": ss})
    return pd.DataFrame(rows)


def fit_district_war(d: pd.DataFrame) -> pd.DataFrame:
    """Overperformance net of cycle, incumbency and profile, in the D direction.
    The residual is split between the two candidates by who is in the race: the
    same residual is the D's WAR and minus the R's."""
    d = d.copy()
    d["over"] = d.margin - d.jud
    d["inc"] = d.d_inc.astype(int) - d.r_inc.astype(int)
    d["lean"] = d.jud / 10
    d["lean2"] = d.lean ** 2
    d["ss"] = d.ssvr.fillna(d.ssvr.median()) / 10
    X = pd.concat([d[["inc", "lean", "lean2", "ss"]],
                   pd.get_dummies(d.year.astype(str) + d.type, dtype=float)], axis=1)
    fit = sm.OLS(d.over, X).fit()
    print("\n=== District overperformance model (margin pts; n=%d, resid sd %.2f) ===" % (len(d), fit.resid.std()))
    print(fit.params[["inc", "lean", "lean2", "ss"]].round(2).to_string())
    d["resid"] = fit.resid
    d["raw"] = d.over
    return d


def long_candidates(d: pd.DataFrame) -> pd.DataFrame:
    a = d.assign(party="D", name=d.d_name, war=d.resid, raw_war=d.raw, is_inc=d.d_inc)
    b = d.assign(party="R", name=d.r_name, war=-d.resid, raw_war=-d.raw, is_inc=d.r_inc)
    c = pd.concat([a, b], ignore_index=True)
    return c[["year", "office", "type", "dist", "party", "name", "war", "raw_war", "is_inc", "jud", "ssvr"]]


def movers(c: pd.DataFrame):
    """Same surname + party in consecutive records at a different office TYPE. Surnames
    that appear twice for one party in a single year are dropped: they cannot be told apart."""
    amb = c.groupby(["year", "party", "name"]).size()
    amb = set(amb[amb > 1].reset_index().apply(lambda r: (r["party"], r["name"]), axis=1))
    k = c[~c.apply(lambda r: (r["party"], r["name"]) in amb, axis=1)].sort_values(["party", "name", "year"])
    rows = []
    for (party, name), g in k.groupby(["party", "name"]):
        g = g.reset_index(drop=True)
        for i in range(1, len(g)):
            if 0 < g.year[i] - g.year[i - 1] <= 8:
                before = g.iloc[:i].tail(2)
                rows.append({"party": party, "name": name, "y0": g.year[i - 1], "y1": g.year[i],
                             "from": g.office[i - 1], "to": g.office[i], "moved": g.office[i - 1] != g.office[i],
                             "w0": before.war.mean(), "w1": g.war[i], "dlean": abs(g.jud[i] - g.jud[i - 1]),
                             "n0": len(before)})
    m = pd.DataFrame(rows)
    # real moves only: office type changed, not a renumbered seat in the same chamber
    typ = lambda s: s.rsplit(" ", 1)[0]
    m["type_change"] = m["from"].map(typ) != m["to"].map(typ)
    # surname matches include strangers; a candidate who held the first office is far
    # more likely to be the same person, and a real up-or-across move
    held = c[c.is_inc].groupby(["party", "name"]).year.apply(set).to_dict()
    m["held"] = [any(y <= r.y0 for y in held.get((r.party, r.name), ())) for r in m.itertuples()]
    for lbl, g in (("same seat next race", m[~m.moved]), ("different office type", m[m.type_change]),
                   ("  of those, held office", m[m.type_change & m.held])):
        if len(g) > 3:
            f = sm.OLS(g.w1, sm.add_constant(g.w0)).fit()
            print(f"  {lbl:<24} n={len(g):>4}  slope {f.params['w0']:+.2f} (se {f.bse['w0']:.2f})  r {np.corrcoef(g.w0, g.w1)[0, 1]:+.2f}")
    return m


def statewide_dev() -> pd.DataFrame:
    sw = pd.read_csv(OUT / "results_statewide.csv")
    sw = sw[sw.contested]
    j = sw[sw.race == "judicial"].groupby("year").d_margin.mean().rename("J")
    s = sw[sw.race != "judicial"].merge(j, on="year")
    s["dev"] = s.d_margin - s.J
    s["tier"] = s.race.map(lambda r: "top" if r in ("president", "us_senate", "governor") else
                           "mid" if r in ("lt_governor", "attorney_general") else "low")
    s["inc"] = s.d_inc.astype(int) - s.r_inc.astype(int)
    return s


def main():
    d = pd.concat([district_rows(y) for y in sorted(ZIPS)], ignore_index=True)
    print("contested district races:", d.groupby("type").size().to_dict())
    d = fit_district_war(d)
    c = long_candidates(d)
    c.to_csv(OUT / "district_war_long.csv", index=False)

    print("\n=== Does district WAR travel? (WAR after on WAR before, avg of last 2 races) ===")
    m = movers(c)
    tc = m[m.type_change]
    print(tc.sort_values("w0", ascending=False)[["party", "name", "y0", "from", "y1", "to", "w0", "w1"]].round(1).to_string(index=False))

    s = statewide_dev()
    s.to_csv(OUT / "statewide_dev.csv", index=False)
    X = pd.concat([s[["inc"]], pd.get_dummies(s.tier, dtype=float)], axis=1)
    f = sm.OLS(s.dev, X).fit()
    print("\n=== Statewide race minus judicial composite (n=%d, resid sd %.2f) ===" % (len(s), f.resid.std()))
    print(f.params.round(2).to_string())
    s["resid"] = f.resid

    print("\n=== First statewide run vs prior district WAR ===")
    rows = []
    for year, race, party, name, pat in FIRST_TIMERS:
        h = c[(c.party == party) & (c.name.str.lower() == name.lower()) & c.office.str.match(pat) & (c.year < year)]
        r = s[(s.year == year) & (s.race == race)]
        if r.empty:
            continue
        sign = 1 if party == "D" else -1
        rows.append({"year": year, "race": race, "party": party, "name": name, "n_prior": len(h),
                     "war_prior": h.sort_values("year").war.tail(3).mean() if len(h) else np.nan,
                     "raw_prior": h.sort_values("year").raw_war.tail(3).mean() if len(h) else np.nan,
                     "seat_lean": h.jud.mean() if len(h) else np.nan,
                     "dev_resid": sign * r.resid.iloc[0]})
    t = pd.DataFrame(rows)
    print(t.round(1).to_string(index=False))
    u = t.dropna(subset=["war_prior"])
    for col in ("war_prior", "raw_prior"):
        fit = sm.OLS(u.dev_resid, u[[col]]).fit()
        print(f"  no-intercept slope on {col}: {fit.params[col]:+.3f} (se {fit.bse[col]:.3f}), n={len(u)}, r {np.corrcoef(u[col], u.dev_resid)[0, 1]:+.2f}")
    rng = np.random.default_rng(7)
    obs = abs(np.corrcoef(u.war_prior, u.dev_resid)[0, 1])
    perm = [abs(np.corrcoef(rng.permutation(u.war_prior.values), u.dev_resid)[0, 1]) for _ in range(5000)]
    print(f"  permutation p for |r|: {np.mean(np.array(perm) >= obs):.3f}")

    print("\n=== 2026 first-time statewide candidates: district record ===")
    out = []
    for race, party, name, pat, dist in NOMINEES_2026:
        h = c[(c.party == party) & (c.name.str.lower() == name.lower()) & c.office.str.match(pat)]
        if dist is not None:
            h = h[h.dist == dist]
        h = h.sort_values("year")
        print(f"{race:<22}{party} {name:<11} races {len(h)}: " +
              ", ".join(f"{r.year} {r.office.replace('State ', '')} {r.war:+.1f}" for r in h.itertuples()))
        out.append({"race": race, "party": party, "name": name, "n": len(h), "war": h.war.tail(3).mean() if len(h) else np.nan})
    pd.DataFrame(out).to_csv(OUT / "first_timer_war_2026.csv", index=False)


if __name__ == "__main__":
    main()
