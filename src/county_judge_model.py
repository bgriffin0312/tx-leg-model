"""
county_judge_model.py

Nine 2026 county judge races and the Bexar County DA, each priced as if the county were
a legislative district: the shipped model_config formula (open-race baseline from the
prior presidential year, differenced environment, incumbency, white-college term), then
tested on the county judge races of the same three midterms that formula was fit on.

Polls enter only where a public head-to-head exists (Harris, Fort Bend). Market prices
are never an input.

Run after county_judge_collect.py:  python src/county_judge_model.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_config as mc  # noqa: E402
import statewide_results as sr  # noqa: E402

RAW = ROOT / "data" / "raw" / "county_judge"
CVAP_ZIP = {2014: "CVAP_2014-2018_ACS_csv_files.zip", 2018: "CVAP_2014-2018_ACS_csv_files.zip",
            2022: "CVAP_2020-2024_ACS_csv_files.zip", 2026: "CVAP_2020-2024_ACS_csv_files.zip"}
ACS = {2014: 2021, 2018: 2021, 2022: 2024, 2026: 2024}    # same vintages as the district term
ENV = {2012: 1.2, 2014: -5.7, 2016: -1.1, 2018: 8.6, 2020: 3.1, 2022: -2.8, 2024: -2.7, 2026: 9.2}
OPEN_OFFICES = {2012: ["RR Comm 1"], 2016: ["CCA 5", "RR Comm 1"], 2020: ["RR Comm 1"],
                2024: ["CCA Pres Judge", "CCA 7", "CCA 8"]}
PRIOR = {2014: 2012, 2018: 2016, 2022: 2020, 2026: 2024}
TEXAS_WHITE_COL = 0.201
MIN_COVERAGE = 0.85       # county race votes as a share of the county's governor votes
POLL_SD = 6.0             # district-poll noise convention, as in the U.S. House layer
INTERNAL_HAIRCUT = 4.0
T_SD = 2.7                # statewide model's posterior sd on the Texas environment
MODEL_GAP = 2.5           # statewide minus legislative environment, margin points

# Rows the OpenElections files lack or carry incomplete: (year, county, office): d, r, names, source
HAND = {
    (2022, "harris", "county_judge"): (546745, 530788, "Lina Hidalgo", "Alexandra del Moral Mealer", "Houston Chronicle via The Texan"),
    (2022, "hays", "county_judge"): (44214, 43439, "Ruben Becerra", "Mark Jones", "Community Impact; the OE file is partial"),
    (2018, "hays", "county_judge"): (39904, 37267, "Ruben Becerra", "Will Conley", "county report via Hays Free Press"),
    (2022, "denton", "county_judge"): (126700, 185232, "Fabian Thomas", "Andy Eads", "Ballotpedia, 59.4% R; D derived"),
    (2014, "bexar", "county_judge"): (5210, 4420, "Nelson Wolff", "Carlton Soules", "Ballotpedia percentages only"),
    (2022, "travis", "county_judge"): (7100, 2900, "Andy Brown", "(R)", "KUT, 71% D; percentages only"),
    (2022, "galveston", "county_judge"): (3500, 6500, "William King III", "Mark Henry", "abc13, 65% R; percentages only"),
}
PARTYLESS = {(2014, "travis"): ("Sarah Eckhardt", "Mike McNamara")}     # OE file has no party column

# Sitting officeholder: D, R or O (open or defeated in the primary). County judge.
INC = {("harris", 2018): "R", ("harris", 2022): "D",
       ("dallas", 2014): "D", ("dallas", 2018): "D", ("dallas", 2022): "D",
       ("tarrant", 2018): "R", ("tarrant", 2022): "O",
       ("bexar", 2014): "D", ("bexar", 2018): "D", ("bexar", 2022): "O",
       ("travis", 2014): "O", ("travis", 2022): "D",
       ("collin", 2018): "O", ("collin", 2022): "R", ("denton", 2018): "O", ("denton", 2022): "R",
       ("fort bend", 2014): "R", ("fort bend", 2018): "R", ("fort bend", 2022): "D",
       ("williamson", 2014): "R", ("williamson", 2018): "O", ("williamson", 2022): "R",
       ("montgomery", 2018): "O", ("hidalgo", 2018): "O",
       ("brazoria", 2014): "R", ("brazoria", 2018): "R", ("brazoria", 2022): "R",
       ("hays", 2014): "R", ("hays", 2018): "O", ("hays", 2022): "D",
       ("nueces", 2014): "R", ("nueces", 2018): "O", ("nueces", 2022): "D",
       ("cameron", 2018): "D", ("cameron", 2022): "D",
       ("jefferson", 2018): "R", ("jefferson", 2022): "R",
       ("mclennan", 2018): "R", ("galveston", 2022): "R"}
INC_DA = {("dallas", 2014): "D", ("dallas", 2018): "R", ("dallas", 2022): "D",
          ("tarrant", 2018): "R", ("tarrant", 2022): "O",
          ("bexar", 2018): "O", ("bexar", 2022): "D",
          ("fort bend", 2014): "R", ("fort bend", 2018): "O",
          ("hays", 2022): "O", ("mclennan", 2022): "O"}
BORDER = {"hidalgo", "cameron", "nueces", "el paso"}
NINE = ["tarrant", "williamson", "collin", "denton", "fort bend", "harris", "bexar", "montgomery", "el paso"]
# Wong funds his race with loans. History excludes loans because they are small there;
# leaving out his $1.505M would score the race as a 12-to-1 money gap.
LOANS_2026 = {("fort bend", "R"): 1505000.0}

# 2026 nominees. inc: sitting officeholder. repeat: name fragment of a candidate with a
# past county judge residual in this county. Fort Bend's Wong is acting since April.
RACES_2026 = [
    {"county": "tarrant", "office": "county_judge", "d": "Simmons", "r": "O'Hare", "inc": "R", "repeat": "O'Hare"},
    {"county": "williamson", "office": "county_judge", "d": "Jefts", "r": "Snell", "inc": "R", "repeat": None},
    {"county": "collin", "office": "county_judge", "d": "Brown", "r": "Hill", "inc": "R", "repeat": "Chris Hill"},
    {"county": "denton", "office": "county_judge", "d": "Pappas", "r": "Eads", "inc": "R", "repeat": "Eads"},
    {"county": "fort bend", "office": "county_judge", "d": "McCoy", "r": "Wong", "inc": "R", "repeat": None},
    {"county": "harris", "office": "county_judge", "d": "Plummer", "r": "Sanchez", "inc": "O", "repeat": None},
    {"county": "bexar", "office": "county_judge", "d": "Nirenberg", "r": "Von Dohlen", "inc": "O", "repeat": None},
    {"county": "montgomery", "office": "county_judge", "d": "Graf", "r": "Keough", "inc": "R", "repeat": "Keough"},
    {"county": "el paso", "office": "county_judge", "d": "Samaniego", "r": "Torres Shelton", "inc": "D", "repeat": None},
    {"county": "bexar", "office": "district_attorney", "d": "Chapa", "r": "Foster", "inc": "O", "repeat": None},
]
# Public head-to-heads inside the window. Margin D-R, flagged if a campaign paid for it.
POLLS = {("harris", "county_judge"): [("UH Hobby Oct 3-8, n=2750", 47 - 40, False)],
         ("fort bend", "county_judge"): [("TSU Sep 9-Oct 2, n=1200", 49 - 40, False)]}


def log_ratio(d, r):
    """Same form as the statewide and U.S. House layers."""
    return np.clip(np.log((np.asarray(d, float) + 25e3) / (np.asarray(r, float) + 25e3)), -3.5, 3.5)


def money_table() -> pd.DataFrame:
    """Receipts by race from data/raw/county_judge/finance.csv. money needs filings on both
    sides. money_rec also treats a Collin Democrat with nothing on file as zero, and is used
    only to net candidate records, never in the fit."""
    f = pd.read_csv(RAW / "finance.csv")
    w = f.pivot_table(index=["county", "year"], columns="party", values="receipts", aggfunc="first").reset_index()
    for (c, party), v in LOANS_2026.items():
        w.loc[(w.county == c) & (w.year == 2026), party] += v
    w["money"] = log_ratio(w.D, w.R)
    w.loc[w.D.isna() | w.R.isna(), "money"] = np.nan
    w["money_rec"] = w.money
    m = (w.county == "collin") & w.D.isna()
    w.loc[m, "money_rec"] = log_ratio(0.0, w.R[m])
    return w


def county_baselines() -> pd.DataFrame:
    rows = []
    for y, offices in OPEN_OFFICES.items():
        df = sr.load(y)
        df = df[df.Office.isin(offices) & df.Party.isin(["D", "R"])].copy()
        df["county"] = df.County.astype(str).str.strip().str.lower()
        t = df.groupby(["county", "Office", "Party"]).Votes.sum().unstack(fill_value=0)
        t["s"] = t.D / (t.D + t.R)
        x = t.s.groupby("county").mean().rename("x").reset_index().assign(base_year=y)
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def governor_votes() -> pd.DataFrame:
    rows = []
    for y in (2014, 2018, 2022):
        df = sr.load(y)
        df = df[df.Office.str.match("^Governor") & df.Party.isin(["D", "R"])].copy()
        df["county"] = df.County.astype(str).str.strip().str.lower()
        t = df.groupby(["county", "Party"]).Votes.sum().unstack(fill_value=0)
        t = t.assign(gov_total=t.D + t.R, gov_d2p=t.D / (t.D + t.R), year=y).reset_index()
        rows.append(t[["county", "year", "gov_total", "gov_d2p"]])
    return pd.concat(rows, ignore_index=True)


def white_college() -> pd.DataFrame:
    """White BA+ share of CVAP by county: CVAP white share x ACS white-NH BA+ rate."""
    rows = []
    for y, zname in CVAP_ZIP.items():
        with zipfile.ZipFile(ROOT / "data" / "raw" / "_cvap_cache" / zname) as z:
            name = next(n for n in z.namelist() if n.lower().endswith("county.csv"))
            c = pd.read_csv(io.BytesIO(z.read(name)), encoding="latin-1", low_memory=False)
        c = c[c.geoid.astype(str).str.contains("US48")]
        c["fips"] = c.geoid.astype(str).str[-5:]
        tot = c[c.lntitle == "Total"].set_index("fips").cvap_est
        wh = c[c.lntitle == "White Alone"].set_index("fips").cvap_est
        a = pd.read_csv(ROOT / "data" / "raw" / "_acs_cache" / f"acsdt5y{ACS[y]}-c15002h.dat", sep="|", dtype={"GEO_ID": str})
        a = a[a.GEO_ID.str.match(r"^0500000US48\d{3}$")]
        a = a.assign(fips=a.GEO_ID.str[-5:]).set_index("fips")
        rate = (a.C15002H_E006 + a.C15002H_E011) / a.C15002H_E001
        names = c[c.lntitle == "Total"].set_index("fips").geoname.str.replace(" County, Texas", "", regex=False).str.lower()
        rows.append(pd.DataFrame({"county": names, "wc": (wh / tot * rate).reindex(names.index), "year": y}))
    return pd.concat(rows, ignore_index=True)


def race_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(RAW / "local_races_raw.csv")
    raw["county"] = raw.county.str.lower()
    raw = raw.set_index(["year", "county", "office"])
    for k, (d, r, dn, rn, src) in HAND.items():
        raw.loc[k, ["d_votes", "r_votes", "d_name", "r_name", "source"]] = [d, r, dn, rn, src]
    raw = raw.reset_index()
    for (y, c), (dn, rn) in PARTYLESS.items():
        src = pd.read_csv(RAW / "oe" / f"{y}1104__tx__general__{c}__precinct.csv", dtype=str)
        src["votes"] = pd.to_numeric(src.votes, errors="coerce").fillna(0)
        s = src[src.office.str.upper() == "COUNTY JUDGE"].groupby("candidate").votes.sum()
        m = (raw.year == y) & (raw.county == c) & (raw.office == "county_judge")
        raw.loc[m, ["d_votes", "r_votes", "d_name", "r_name"]] = [s[dn], s[rn], dn, rn]
    keys = {**{(c, y, "county_judge"): v for (c, y), v in INC.items()},
            **{(c, y, "district_attorney"): v for (c, y), v in INC_DA.items()}}
    raw["inc"] = [keys.get((c, y, o)) for c, y, o in zip(raw.county, raw.year, raw.office)]
    t = raw[raw.inc.notna() & (raw.d_votes > 0) & (raw.r_votes > 0)].copy()
    gov = governor_votes()
    t = t.merge(gov, on=["county", "year"], how="left")
    by_hand = t.source != "openelections"
    t["cover"] = (t.d_votes + t.r_votes + t.other_votes) / t.gov_total
    print("coverage failures dropped:", t[(t.cover < MIN_COVERAGE) & ~by_hand][["year", "county", "office", "cover"]].round(2).values.tolist())
    t = t[(t.cover >= MIN_COVERAGE) | by_hand].copy()
    t["y"] = t.d_votes / (t.d_votes + t.r_votes)
    return attach(t), raw


def attach(t: pd.DataFrame) -> pd.DataFrame:
    base = county_baselines()
    wc = white_college()
    t = t.copy()
    t["base_year"] = t.year.map(PRIOR)
    t = t.merge(base, on=["county", "base_year"], how="left").merge(wc, on=["county", "year"], how="left")
    t["d_inc"] = (t.inc == "D").astype(float)
    t["r_inc"] = (t.inc == "R").astype(float)
    t["wc_c"] = t.wc - TEXAS_WHITE_COL
    t["env_d"] = t.year.map(ENV) - t.base_year.map(ENV)
    c = mc.REGRESSION_COEFFICIENTS
    t["pred"] = (c["intercept"] + c["dem_pres_2p_baseline"] * t.x + c["national_env"] * t.env_d
                 + c["dem_incumbent"] * t.d_inc + c["rep_incumbent"] * t.r_inc + c["white_col_centered"] * t.wc_c)
    if "y" in t:
        t["res"] = t.y - t.pred
    return t


def loco(t: pd.DataFrame, terms: list[str] | None) -> dict:
    """Leave one cycle out on the residual from the shipped formula. None = no fit at all;
    [] = a constant only."""
    r = []
    for y in sorted(t.year.unique()):
        tr, te = t[t.year != y], t[t.year == y]
        if terms is None:
            r.append(te.res)
        else:
            f = sm.OLS(tr.res, sm.add_constant(tr[terms], has_constant="add")).fit()
            r.append(te.res - f.predict(sm.add_constant(te[terms], has_constant="add")))
    e = pd.concat(r).reindex(t.index)
    cyc = e.groupby(t.year).mean()
    return {"rmse": float(np.sqrt((e**2).mean())), "cycle_rms": float(np.sqrt((cyc**2).mean())),
            "within": float((e - e.groupby(t.year).transform("mean")).std()), "cycles": cyc.round(4).to_dict()}


def own_record(t: pd.DataFrame) -> pd.DataFrame:
    """Persistence: same candidate, same county, next race. Residual net of cycle."""
    d = t[t.office == "county_judge"].copy()
    d["res_c"] = d.res - d.groupby("year").res.transform("mean")
    rows = []
    for side, col in (("d", "d_name"), ("r", "r_name")):
        for (c, n), g in d.groupby(["county", col]):
            g = g.sort_values("year")
            for i in range(1, len(g)):
                sign = 1 if side == "d" else -1
                rows.append({"county": c, "name": n, "year": g.year.iloc[i], "prev": sign * g.res_c.iloc[i - 1], "now": sign * g.res_c.iloc[i]})
    return pd.DataFrame(rows)


def run(r, k_open, k_held, nat, within, shift=0.0, open_ids=(), save=False, shared_draws=None, seed=2026, keep=None, money_mult=1.0,
        local=None, keep_margin=None):
    """shared_draws (share units) replaces the county simulation's own shared error, so the
    joint run can tie these races to the statewide draws; keep collects each race's wins.
    local gives the whole-election model's county draws: {county: (standard normal, share of
    the race error, extra margin)}; keep_margin collects each race's margin."""
    rng = np.random.default_rng(seed)
    n = 200000
    shared = rng.normal(0, nat, n) if shared_draws is None else shared_draws
    n = len(shared)
    rows = []
    for _, a in r.iterrows():
        k = k_held if a.inc == "R" and a.county not in open_ids else k_open
        m0 = (a.pred + k + a.war + money_mult * a.money_term) * 200 - 100 + shift
        s0 = np.hypot(nat, within) * 200
        m, s = m0, s0
        polls = POLLS.get((a.county, a.office), [])
        for _, v, internal in polls:
            v = v - INTERNAL_HAIRCUT if internal else v
            g = s**2 / (s**2 + POLL_SD**2)
            m, s = m + g * (v - m), np.sqrt((1 - g) * s**2)
        # the shared error keeps its share of the variance after any poll update
        idio = rng.standard_t(5, n) * within * 200 * np.sqrt(3 / 5)
        extra = 0.0
        if local is not None and a.county in local:
            u, lam, extra = local[a.county]
            idio = np.sqrt(1 - lam) * idio + np.sqrt(lam) * u * within * 200
        margin = m + (shared * 200 + idio) * s / s0 + extra
        if a.office == "district_attorney":
            w = rng.beta(3, 17, n)              # Wolff, the ex-Republican independent: mean 15%
            frac = rng.uniform(0.55, 0.9, n)    # share of his vote taken from the Republican
            dv, rv = (100 + margin) / 2, (100 - margin) / 2
            dv3, rv3 = dv - w * 100 * (1 - frac), rv - w * 100 * frac
            won = (dv3 > rv3) & (dv3 > w * 100)
        else:
            won = margin > 0
        p = float(won.mean())
        if keep is not None:
            keep[(a.county, a.office)] = won
        if keep_margin is not None:
            keep_margin[(a.county, a.office)] = margin
        rows.append({"county": a.county, "office": a.office, "d": a.d, "r": a.r, "inc": a.inc, "x_2024": a.x, "wc_c": a.wc_c,
                     "formula": a.pred * 200 - 100, "seat_term": k * 200, "money_term": money_mult * a.money_term * 200, "candidate": a.war * 200, "prior": m0,
                     "polls": "; ".join(p_[0] for p_ in polls), "mean": m, "sd": float(margin.std()), "p_d": p,
                     "lo10": float(np.quantile(margin, .1)), "hi90": float(np.quantile(margin, .9))})
    o = pd.DataFrame(rows)
    if save:
        o.to_csv(ROOT / "output" / "county_judge_forecast_2026.csv", index=False)
    return o


def main():
    t, raw = race_table()
    cj = t[t.office == "county_judge"].copy()
    da = t[t.office == "district_attorney"].copy()
    cj["two"] = cj.d_votes + cj.r_votes
    show = cj.sort_values(["year", "county"])
    print(show.assign(y_m=(2 * show.y - 1) * 100, p_m=(2 * show.pred - 1) * 100, r_m=2 * show.res * 100)[
        ["year", "county", "d_name", "r_name", "inc", "x", "wc_c", "y_m", "p_m", "r_m"]].round(3).to_string(index=False))

    print(f"\nCounty judge races n={len(cj)}; residual in margin points")
    for name, m in (("all", cj.index == cj.index), ("excl. border", ~cj.county.isin(BORDER)), ("border", cj.county.isin(BORDER)),
                    ("R officeholder", cj.inc == "R"), ("D officeholder", cj.inc == "D"), ("open", cj.inc == "O")):
        g = cj[m]
        print(f"  {name:<16} n {len(g):>2} mean {2 * g.res.mean() * 100:+5.1f} sd {2 * g.res.std() * 100:4.1f}  by cycle "
              + str((2 * g.groupby('year').res.mean() * 100).round(1).to_dict()))
    print("DA races as an out-of-office check: n", len(da), "mean", round(2 * da.res.mean() * 100, 1), "sd", round(2 * da.res.std() * 100, 1),
          (2 * da.groupby("year").res.mean() * 100).round(1).to_dict())

    print("\nLeave-one-cycle-out on the shipped formula (share units)")
    cj["open"] = (cj.inc == "O").astype(float)
    specs = {"formula as shipped": None, "+constant": [], "+R officeholder": ["r_inc"],
             "+open seat": ["open"], "+D and R officeholder": ["d_inc", "r_inc"]}
    use = cj[~cj.county.isin(BORDER)]
    out = {}
    for name, terms in specs.items():
        out[name] = loco(use, terms)
        print(f"  {name:<24}", {k: (round(v, 4) if not isinstance(v, dict) else v) for k, v in out[name].items()})
    f = sm.OLS(use.res, sm.add_constant(use[["open"]])).fit()
    print("  open seat, in sample (share): ", f.params.round(4).to_dict(), "se", f.bse.round(4).to_dict())
    rng_p = np.random.default_rng(5)
    obs = use[use.open == 1].res.mean() - use[use.open == 0].res.mean()
    perm = [rng_p.permutation(use.res.values) for _ in range(5000)]
    m = (use.open == 1).values
    print(f"  open-minus-held gap {obs * 200:+.1f} pts of margin, permutation p {np.mean([abs(x[m].mean() - x[~m].mean()) >= abs(obs) for x in perm]):.3f}")
    for lab, g in (("R-held open", use[(use.open == 1) & (use.y < .5)]), ("open won by D", use[(use.open == 1) & (use.y >= .5)])):
        print(f"   {lab}: n {len(g)}, mean {g.res.mean() * 200:+.1f}")
    da = da.assign(open=(da.inc == "O").astype(float))
    print("\nDA races, never used to find the open-seat term:")
    print(da.assign(r_m=da.res * 200)[["year", "county", "d_name", "r_name", "inc", "r_m"]].round(1).to_string(index=False))
    print(f"  open {da[da.open == 1].res.mean() * 200:+.1f} (n {int(da.open.sum())}), held {da[da.open == 0].res.mean() * 200:+.1f} (n {int((1 - da.open).sum())}); "
          f"rmse formula {np.sqrt((da.res**2).mean()) * 200:.1f}, with the term {np.sqrt(((da.res - f.predict(sm.add_constant(da[['open']], has_constant='add')))**2).mean()) * 200:.1f}")
    print("Open cases by year:", use[use.open == 1].assign(r_m=use.res * 200)[["year", "county", "r_m"]].round(1).values.tolist())
    cj["res2"] = cj.res - f.predict(sm.add_constant(cj[["open"]], has_constant="add"))
    t = t.assign(res=cj.res2.reindex(t.index).fillna(t.res))
    pers = own_record(t)
    pf = sm.OLS(pers.now, pers[["prev"]]).fit()
    print(f"\nSame candidate, same county, next race: slope {pf.params.iloc[0]:.2f} (se {pf.bse.iloc[0]:.2f}, n {len(pers)})")

    # Error sizes: within-cycle spread of county races, and the legislative model's national term
    within = float((use.res - use.groupby("year").res.transform("mean")).std())
    # The backtest knew each year's environment; 2026 does not. Shared error is the formula's
    # cycle miss, the statewide posterior sd on Texas, and half the 2.5-point gap between
    # the two models. The shipped legislative national sd is the sensitivity case.
    cyc = float(out["formula as shipped"]["cycle_rms"]) * 200
    nat = np.sqrt(cyc**2 + T_SD**2 + (MODEL_GAP / 2)**2) / 200
    print(f"sd used: shared {nat * 200:.1f} pts of margin (cycle {cyc:.1f}, T {T_SD}, gap {MODEL_GAP / 2}), county {within * 200:.1f} pts")

    # Nine-county factors, fit together. The legislative incumbency values come out of the
    # formula first, so these replace them. Open and D-held seats share a level: the
    # D-seat gap is one case (Wolff), and the R-seat-only form was best out of sample.
    c = mc.REGRESSION_COEFFICIENTS
    cj["r0"] = cj.y - (cj.pred - c["dem_incumbent"] * cj.d_inc - c["rep_incumbent"] * cj.r_inc)
    cj = cj.merge(money_table()[["county", "year", "D", "R", "money", "money_rec"]], on=["county", "year"], how="left")
    nine = cj[cj.county.isin(NINE)]
    hist = nine.dropna(subset=["money"])
    fs = sm.OLS(nine.r0, sm.add_constant(nine[["r_inc"]])).fit()
    fm = sm.OLS(hist.r0, sm.add_constant(hist[["r_inc", "money"]])).fit()
    print(f"\nSeat term, nine counties, n {len(nine)} (margin): other seats {fs.params['const'] * 200:+.2f}, R seat {fs.params['r_inc'] * 200:+.2f} (se {fs.bse['r_inc'] * 200:.2f})")
    print(f"With money, n {len(hist)}: other {fm.params['const'] * 200:+.2f}, R seat {fm.params['r_inc'] * 200:+.2f} (se {fm.bse['r_inc'] * 200:.2f}), "
          f"money {fm.params['money'] * 200:+.2f} per log unit (se {fm.bse['money'] * 200:.2f})")
    hist = hist.assign(r_p=hist.r0 * 200)
    print(hist[["county", "year", "inc", "D", "R", "money", "r_p"]].round(2).to_string(index=False))
    for label, terms in (("constant", []), ("R seat", ["r_inc"]), ("money", ["money"]), ("R seat + money", ["r_inc", "money"])):
        e = {}
        for key in ("year", "county"):
            parts = []
            for v in hist[key].unique():
                tr, te = hist[hist[key] != v], hist[hist[key] == v]
                h = sm.OLS(tr.r_p, sm.add_constant(tr[terms], has_constant="add")).fit()
                parts.append(te.r_p - h.predict(sm.add_constant(te[terms], has_constant="add")))
            e[key] = round(float(np.sqrt((pd.concat(parts)**2).mean())), 2)
        print(f"  {label:<16} out-of-sample rmse {e}")
    print("money coefficient dropping each county:", {v: round(float(sm.OLS(hist[hist.county != v].r_p, sm.add_constant(hist[hist.county != v][["r_inc", "money"]])).fit().params["money"]), 2) for v in hist.county.unique()})
    rng_m = np.random.default_rng(1)
    sims = [sm.OLS(hist.r_p, sm.add_constant(hist[["r_inc"]].assign(money=rng_m.permutation(hist.money.values)))).fit().params["money"] for _ in range(5000)]
    print(f"money permutation p {np.mean(np.abs(sims) >= abs(fm.params['money'] * 200)):.3f}")

    k_open, k_held = float(fm.params["const"]), float(fm.params["const"] + fm.params["r_inc"])
    g = float(fm.params["money"])
    r = pd.DataFrame(RACES_2026)
    r["year"] = 2026
    r = attach(r)
    r["pred"] = r.pred - c["dem_incumbent"] * r.d_inc - c["rep_incumbent"] * r.r_inc
    mt = money_table()
    mt = mt[mt.year == 2026].set_index("county").money
    r["money"] = [mt.get(x) if o_ == "county_judge" else np.nan for x, o_ in zip(r.county, r.office)]
    r["money_term"] = g * r.money.fillna(0.0).astype(float)
    cj["res2"] = cj.r0 - fm.predict(sm.add_constant(cj[["r_inc"]].assign(money=cj.money_rec.fillna(0.0)), has_constant="add"))
    d = cj.assign(res_c=cj.res2 - cj.groupby("year").res2.transform("mean"))
    rows = []
    for _, a in r.iterrows():
        war, n = 0.0, 0
        if isinstance(a["repeat"], str):
            g_ = d[(d.county == a.county) & (d.r_name.astype(str).str.contains(a["repeat"], regex=False))]
            n = len(g_)
            war = float(g_.res_c.mean()) * n / (n + mc.WAR_PERSISTENCE_K)
        rows.append({"war": war, "n_prior": n})
    r = pd.concat([r, pd.DataFrame(rows)], axis=1)
    o = run(r, k_open, k_held, nat, within, save=True)
    print()
    print(o.round(2).drop(columns=["polls", "x_2024", "wc_c", "lo10", "hi90"]).to_string(index=False))

    print("\nSensitivity: P(D wins), mean in brackets")
    legs = np.sqrt(0.58) * c["sigma"]
    no_loans = r.copy()
    no_loans.loc[no_loans.county == "fort bend", "money_term"] = g * float(log_ratio(1085876.37, 63508.0))
    cases = {"base": {},
             "statewide environment (+2.5)": {"shift": MODEL_GAP},
             "no money, n=18 seat term": {"k_open": float(fs.params["const"]), "k_held": float(fs.params["const"] + fs.params["r_inc"]), "money_mult": 0.0},
             "Snell + Wong as open": {"open_ids": {"williamson", "fort bend"}},
             "Wong loans excluded": {"r": no_loans},
             "legislative national sd": {"nat": legs}}
    tbl = {}
    for name, kw in cases.items():
        a = dict(r=r, k_open=k_open, k_held=k_held, nat=nat, within=within)
        a.update(kw)
        x = run(a.pop("r"), a.pop("k_open"), a.pop("k_held"), a.pop("nat"), a.pop("within"), **a)
        tbl[name] = [f"{p_:.0%} ({m_:+.1f})" for p_, m_ in zip(x.p_d, x["mean"])]
    print(pd.DataFrame(tbl, index=[f"{c_} {o_[:2]}" for c_, o_ in zip(o.county, o.office)]).to_string())
    return {"r": r, "o": o, "k_open": k_open, "k_held": k_held, "nat": nat, "within": within}


if __name__ == "__main__":
    main()
