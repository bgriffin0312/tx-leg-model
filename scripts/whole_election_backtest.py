"""
whole_election_backtest.py

Re-runs the whole-election model on 2018, 2022 and 2024: one electorate across every statewide,
U.S. House, Texas House and Senate race on the ballot, and the county judge races in the nine
forecast counties. Each cycle is left out of every fit that can leave it out, then scored on
what the model is actually for: seat counts, whether races missed together the way it says
they should, and joint outcomes.

    statewide      T centered on that cycle's statewide posterior, polls included
    legislative    T centered on what the legislative formula implies
    actual_T       T fixed at the judicial vote that happened, which separates a miss in
                   the environment from a miss in how races are read from it

Held out by cycle: statewide and U.S. House fits, T prior, poll miss mean and sd, legislative
formula (environment slope fixed at 0.0034, as in env_differenced_refit: without 2018 it
cannot be identified), county seat terms, one-electorate slope, place shares, competitive sd.

Not held out: the legislative error sizes (fit on 2014/2018/2022; 2024 is clean), POLL_RACE_SD,
group error sizes. 2024 is also a midterm formula on a presidential year. No district polls
exist for these years, no 2024 viability or IE data (set to 0), and county races skip the
candidate-record and money terms, because the money pairs fall to about five without the
held-out year.

The environment is the final generic ballot, not the result: the mean of FiveThirtyEight's
last daily average (Internet Archive copy of generic_ballot_averages.csv, 2024-11-10) and
RealClearPolitics' final average, from data/raw/statewide/generic_ballot_finals.csv. --env
takes either alone, or the national House vote as the bracket.

Usage:  python scripts/whole_election_backtest.py [--env poll|538|rcp|actual]
Output: output/whole_election_backtest_{races,summary,joint}.csv
"""
import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import county_judge_model as cj  # noqa: E402
import midterm_education_test as met  # noqa: E402  (sets --education for group_error_estimate)
import model as lg  # noqa: E402
import model_config as mc  # noqa: E402
import refit_clean_cycles as rc  # noqa: E402
import statewide_model as sw  # noqa: E402
import statewide_results as sr  # noqa: E402
import whole_election_model as wm  # noqa: E402
from statewide_pollster_history import load_polls  # noqa: E402

ge = met.ge
OUT = ROOT / "output"
HIST = ROOT / "data" / "raw" / "historical"
YEARS = (2018, 2022, 2024)
GB = pd.read_csv(ROOT / "data" / "raw" / "statewide" / "generic_ballot_finals.csv").set_index("cycle")
ENV_BASE = {2014: 1.2, 2018: -1.1, 2022: 3.1, 2024: 3.1}
LEG_SLOPE = 0.0034
SHIPPED_SHARED = 6.7       # the legislative shared error as shipped, margin
TOP = {2014: "Governor", 2018: "Governor", 2022: "Governor", 2024: "President"}
PRES_VOTES = {2014: "tx_presidential_house_2012_planh358.csv", 2018: "tx_presidential_house_2016_planh358.csv",
              2022: "tx_presidential_house_2020_planh2316.csv", 2024: "tx_presidential_house_2020_planh2316.csv"}
# TLC precinct-to-district files. VTD returns carry no row for an uncontested race, so the race
# rows alone map only about 100 House seats. 2018 ran on the lines in the 2020 file.
PRECINCTS = {2018: ("precincts20g_districts.xlsx", "PlanH2100"), 2022: ("precincts22g_districts.xlsx", "PlanH2316"), 2024: ("precincts24g_districts.xlsx", "PlanH2316")}
PAIRS = ((2012, 2014), (2014, 2016), (2016, 2018), (2018, 2020), (2022, 2024))
CD_D = {2018: 13, 2022: 13, 2024: 13}                  # D seats won, all districts
OTHER = ["x", "chamber_senate", "dem_incumbent", "rep_incumbent", "challenger_viability_flag", "ie_c", "c_white_col"]
GROUP_COLS = ["c_white_noncol", "c_white_col", "c_black", "c_hispanic", "c_other"]
N = 60000


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def estimates_without(year):
    """whole_election_estimates.py with every cycle pair touching the held-out year removed."""
    r = pd.read_csv(OUT / "whole_election_rolloff.csv")
    d = r[r.tier.isin(["us_house", "state_senate", "state_house"]) & r.m_j.notna() & (r.m_race.abs() < 60)]
    swing = []
    for o, g in d.groupby("office"):
        g = g.set_index("year")
        for a, b in PAIRS:
            if year in (a, b) or a not in g.index or b not in g.index:
                continue
            swing.append({"office": o, "county": g.county.iloc[0], "pair": f"{a}-{b}", "drace": g.m_race_j[b] - g.m_race_j[a], "dj": g.m_j[b] - g.m_j[a]})
    w = pd.DataFrame(swing)
    x = w.assign(dm=w.drace - w.groupby("pair").drace.transform("mean"), dj_=w.dj - w.groupby("pair").dj.transform("mean"))
    f = sm.OLS(x.dm, x.dj_).fit()
    v_local, v_split = (f.params["dj_"] * x.dj_).var(), f.resid.var()
    multi = x[x.groupby(["pair", "county"]).office.transform("count") >= 3]
    cm = multi.groupby(["pair", "county"]).dj_.transform("mean")
    k = multi.groupby(["pair", "county"]).office.transform("count")
    v_within = ((multi.dj_ - cm) ** 2 * (k / (k - 1))).mean()
    icc = max(0.0, 1 - v_within / multi.dj_.var())
    sw_ = pd.read_csv(sw.RAW / "results_statewide.csv")
    jy = sw_[(sw_.race == "judicial") & sw_.contested].groupby("year").d_margin.mean()
    h = d[d.tier == "state_house"]
    ex = []
    for a, b in PAIRS:
        if year in (a, b):
            continue
        p_, q_ = h[h.year == a].set_index("office"), h[h.year == b].set_index("office")
        k_ = [o for o in p_.index.intersection(q_.index) if abs(p_.m_race[o]) < 20]
        ex.append((q_.m_race_j[k_] - p_.m_race_j[k_]).mean() - (jy[b] - jy[a]))
    return {"slope": float(sm.OLS(w.drace, sm.add_constant(w.dj)).fit().params["dj"]), "local_share": float(v_local / (v_local + v_split)),
            "county_icc": float(icc), "competitive_sd": float(np.sqrt(np.mean(np.square(ex))) / np.sqrt(2))}


def poll_avg(p, race, year):
    """statewide_model.poll_average on the history file: same window, same weights."""
    g = p[(p.cycle == year) & (p.race == race) & ~p.partisan & (p.days <= 30)].copy()
    if g.empty:
        return np.nan, 0
    w = np.sqrt(g.sample_size.fillna(800).clip(upper=2000) / 800) * 0.5 ** (g.days / 14)
    return float((g.margin * w).sum() / w.sum()), len(g)


def statewide_layer(year, env):
    s0 = sw.statewide_table()
    b = sw.fit_dev(s0[s0.year != year], ["inc"])
    tiers = {t: b.params[t] for t in ("top", "mid", "low")}
    s = sw.add_prior_dev(s0, b.params["inc"], tiers)
    terms = ["inc", "money", "prior_own"]
    tr, te = s[s.year != year], s[s.year == year].copy()
    f = sw.fit_dev(tr, terms)
    res = tr.dropna(subset=terms).assign(res=f.resid)
    sds = res.groupby("tier").res.std().to_dict()
    te["no_money"] = te.money.isna()
    te["money"] = te.money.fillna(0.0)
    p = load_polls()
    avg = {r: poll_avg(p, r, year) for r in te.race}
    R = pd.DataFrame({"tier": te.tier.values, "dev_hat": f.predict(te[["top", "mid", "low"] + terms]).values, "actual": te.d_margin.values,
                      "poll_avg": [avg[r][0] for r in te.race], "n_polls": [avg[r][1] for r in te.race], "no_money": te.no_money.values}, index=te.race.values)

    _, _, t = quiet(sw.t_prior, s)
    dt = t.dropna().drop(index=[y for y in (year, year + 2) if y in t.dropna().index])
    fit = sm.OLS(dt.dJ, dt[["dE"]]).fit()
    loo = [dt.dJ[y] - float(sm.OLS(dt.dJ.drop(y), dt[["dE"]].drop(y)).fit().params["dE"] * dt.dE[y]) for y in dt.index]
    Tmu = float(t.J[year - 2] + fit.params["dE"] * (env - sw.NAT[year - 2]))
    Tsd = float(np.sqrt(np.mean(np.square(loo)) + (fit.params["dE"] * sw.ENV_SD_2026) ** 2))
    errs = p[~p.partisan].groupby("cycle").err.mean().drop(year)
    sw.POLL_B_SD = max(float(errs.std()), 2.5)

    h = sw.house_table()
    f_cd, c_sd, h_sd, e_sd = quiet(sw.congress_layer, h[h.year != year].copy())
    C = h[h.year == year].set_index("dist").copy()
    C["ss"] = C.ss.fillna(0.0)
    C["rel_hat"] = f_cd.predict(sm.add_constant(C[sw.CD_TERMS], has_constant="add"))
    dp = pd.DataFrame(columns=["dist", "end", "pollster", "d_pct", "r_pct", "partisan_flag"])
    f = sw.forecast(R, C[["ss", "rel_hat"]], dp, Tmu, Tsd, c_sd, h_sd, e_sd, sds, float(errs.mean()), quiet=True)
    return {"f": f, "R": R, "C": C, "Tprior": Tmu, "Tprior_sd": Tsd, "J": float(t.J[year])}


def leg_frame(year):
    """Every House seat and each Senate seat up that year, with the formula's inputs. Contested
    rows carry their result; the rest are known at filing and counted at the winner."""
    w = ge.texas_weights()
    if year == 2024:
        rows = []
        for ch, base in (("house", "tx_downballot_house_2020_planh2316.csv"), ("senate", "tx_downballot_senate_2020_plans2168.csv")):
            r = pd.read_csv(ROOT / "data" / "raw" / f"tx_{ch}_results_2024.csv")
            r = r[r.winner_party.isin(["D", "R"])].copy()
            prev = pd.read_csv(HIST / f"tx_{ch}_results_2022.csv", encoding="utf-8-sig").set_index("district")
            last = lambda s: str(s).split()[-1].lower() if isinstance(s, str) and s.strip() else ""   # noqa: E731
            r["dem_incumbent"] = [float(d in prev.index and prev.winner_party[d] == "D" and last(prev.d_candidate[d]) == last(c)) for d, c in zip(r.district, r.d_candidate)]
            r["rep_incumbent"] = [float(d in prev.index and prev.winner_party[d] == "R" and last(prev.r_candidate[d]) == last(c)) for d, c in zip(r.district, r.r_candidate)]
            b = pd.read_csv(HIST / base)[["district", "open_mean_d2p"]]
            r = r.merge(b, on="district", how="left").rename(columns={"open_mean_d2p": "x"})
            r["chamber_lower"], r["chamber_senate"] = ch, float(ch == "senate")
            r["dem_2p_share"] = r.d_votes / (r.d_votes + r.r_votes)
            r["contested"] = r.contested.astype(bool)
            rows.append(r)
        a = pd.concat(rows, ignore_index=True).assign(year=2022)
        a = ge.attach_shares(a).assign(year=2024)
        a["challenger_viability_flag"], a["ie_c"] = 0.0, 0.0
    else:
        a = rc.bb.load()
        a = a[a.year == year].copy()
        a["x"] = a.open_mean_d2p
        a = ge.attach_shares(a)
        d = met.prep("open")
        a = a.drop(columns=["challenger_viability_flag"]).merge(d[d.year == year][["chamber_lower", "district", "challenger_viability_flag", "ie_c"]], on=["chamber_lower", "district"], how="left")
        a[["challenger_viability_flag", "ie_c"]] = a[["challenger_viability_flag", "ie_c"]].fillna(0.0)
        res = []
        for ch in ("house", "senate"):
            r = pd.read_csv(HIST / f"tx_{ch}_results_{year}.csv", encoding="utf-8-sig")
            r = r[r.winner_party.isin(["D", "R"])]
            if "on_ballot" in r:
                r = r[r.on_ballot.astype(bool)]
            res.append(r[["district", "winner_party"]].assign(chamber_lower=ch))
        a = a.drop(columns=["contested"], errors="ignore").merge(pd.concat(res), on=["chamber_lower", "district"], how="inner")
        a["contested"] = (pd.to_numeric(a.dem_2p_share, errors="coerce").between(0.001, 0.999))
    for g in ("white_col", "white_noncol", "black", "hispanic", "other"):
        a[f"c_{g}"] = (a[g] - w[g]).fillna(0.0)
    for c in ("dem_incumbent", "rep_incumbent", "chamber_senate"):
        a[c] = pd.to_numeric(a[c], errors="coerce").fillna(0.0)
    return a.reset_index(drop=True)


def leg_fit(year):
    d = met.prep("open").dropna(subset=["challenger_viability_flag"]).copy()
    d["env_diff"] = d.national_env - d.year.map(ENV_BASE)
    tr = d[d.year != year].copy()
    tr["dem_2p_share"] = tr.dem_2p_share - LEG_SLOPE * tr.env_diff
    c, _, _ = rc.ols(tr, OTHER)
    return c


def county_frame(year, c, env):
    t, _ = quiet(cj.race_table)
    t = t[(t.office == "county_judge") & t.county.isin(cj.NINE)].copy()
    t["open"] = (t.inc == "O").astype(float)

    def base(g, e):
        return c["intercept"] + c["x"] * g.x + c["c_white_col"] * g.wc_c + LEG_SLOPE * e

    t["env_act"] = t.year.map(sw.NAT) - t.year.map(ENV_BASE)
    tr, te = t[t.year != year].copy(), t[t.year == year].copy()
    tr["r0"] = tr.y - base(tr, tr.env_act)
    f = sm.OLS(tr.r0, sm.add_constant(tr[["r_inc"]])).fit()
    within = float((tr.r0 - tr.groupby("year").r0.transform("mean")).std())
    te["m0"] = (base(te, env - ENV_BASE[year]) + f.params["const"] + f.params["r_inc"] * te.r_inc) * 200 - 100
    te["actual"] = (2 * te.y - 1) * 100
    return te.reset_index(drop=True), within


def geography(year):
    df = sr.load(year).copy()
    df["county"] = df.County.astype(str).str.lower().str.strip()
    counties = sorted(df.county.unique())
    ci = {c: i for i, c in enumerate(counties)}
    rep = df[df.Office.str.match(r"^State Rep \d+$")]
    hdv = rep.assign(hd=rep.Office.str.extract(r"(\d+)$")[0].astype(int)).groupby(["cntyvtd", "hd"]).Votes.sum().reset_index().sort_values("Votes").drop_duplicates("cntyvtd", keep="last").set_index("cntyvtd").hd
    pf, col = PRECINCTS[year]
    pm = pd.read_excel(ROOT / "data" / "raw" / "_capitol_data_cache" / pf, dtype=str)
    pm = pd.to_numeric(pm.set_index("PCTKEY")[col], errors="coerce").dropna().astype(int)
    pm = pm[(pm >= 1) & (pm <= 150)]
    pm.index = pm.index.astype(str)
    pm = pm[~pm.index.duplicated()]
    hdv.index = hdv.index.astype(str)
    hdv = pm.combine_first(hdv).astype(int)
    df["cntyvtd"] = df.cntyvtd.astype(str)

    def weights(frame, n_rows):
        wc, wh = np.zeros((n_rows, len(counties))), np.zeros((n_rows, 150))
        np.add.at(wc, (frame.i.values, frame.county.map(ci).values), frame.Votes.values)
        h = frame.cntyvtd.map(hdv)
        m = h.notna().values
        np.add.at(wh, (frame.i.values[m], h[m].astype(int).values - 1), frame.Votes.values[m])
        return wm.unit(wc), wm.unit(wh), wh

    def office(prefix, ids):
        idx = {f"{prefix} {d}": i for i, d in enumerate(ids)}
        g = df[df.Office.isin(idx)].groupby(["Office", "cntyvtd", "county"]).Votes.sum().reset_index()
        return weights(g.assign(i=g.Office.map(idx)), len(ids))

    def county(names):
        idx = {c: i for i, c in enumerate(names)}
        g = df[(df.Office == TOP[year]) & df.county.isin(idx)].groupby(["county", "cntyvtd"]).Votes.sum().reset_index()
        return weights(g.assign(i=g.county.map(idx)), len(names))

    return {"office": office, "county": county, "n_counties": len(counties)}


class Cycle:
    def __init__(self, year, env):
        self.year, self.env = year, env
        self.est = estimates_without(year)
        self.ro = wm.rolloff_table()
        self.S = quiet(statewide_layer, year, env)
        self.c = leg_fit(year)
        self.a = leg_frame(year)
        a = self.a
        self.lin_share = rc.predict(a, self.c) + LEG_SLOPE * (env - ENV_BASE[year])
        self.lin = self.lin_share * 200 - 100
        house = (a.chamber_lower == "house").values
        self.G = geography(year)
        # same reading as the 2026 center: every House seat treated as contested, weighted by
        # the prior presidential two-party vote (scripts/majority_threshold_bias.py sw_latent)
        self.TL = float(np.average(self.lin[house], weights=hd_weights(year, a.district[house])))
        self.m = a.contested.values & a.x.notna().values
        self.te = a[self.m].reset_index(drop=True)
        self.fixed_h = int((a[house & ~self.m].winner_party == "D").sum()) if "winner_party" in a else 0
        self.fixed_s = int((a[~house & ~self.m].winner_party == "D").sum())
        self.county, self.within = (pd.DataFrame(), 0.0) if year == 2024 else county_frame(year, self.c, env)

        rng = np.random.default_rng(year)
        lc = self.est["local_share"] * self.est["county_icc"]
        lh = self.est["local_share"] * (1 - self.est["county_icc"])
        self.lc, self.lh = lc, lh
        self.Uc = rng.standard_normal((N, self.G["n_counties"]), dtype=np.float32)
        self.Uh = rng.standard_normal((N, 150), dtype=np.float32)
        self.kappa = rng.normal(0, wm.ROLL_SD, N)
        self.lay = rng.normal(0, self.est["competitive_sd"], N)
        self.lay_c = rng.normal(0, wm.LAYER_COUNTY, N)
        self.groups = rng.standard_normal((N, 5))
        self.extra = rng.standard_normal(N)
        self.eps = rng.standard_normal((N, len(self.te)), dtype=np.float32)
        self.t5 = rng.standard_t(5, (N, max(len(self.county), 1))) * np.sqrt(3 / 5)

        ids_h = self.te[self.te.chamber_lower == "house"].district.astype(int).tolist()
        ids_s = self.te[self.te.chamber_lower == "senate"].district.astype(int).tolist()
        hc, hh, hraw = self.G["office"]("State Rep", ids_h)
        sc, sh, _ = self.G["office"]("State Sen", ids_s)
        self.is_h = (self.te.chamber_lower == "house").values
        self.leg_c = np.zeros((len(self.te), hc.shape[1])); self.leg_h = np.zeros((len(self.te), 150))
        self.leg_c[self.is_h], self.leg_h[self.is_h] = hc, hh
        self.leg_c[~self.is_h], self.leg_h[~self.is_h] = sc, sh
        self.cd_c, self.cd_h, _ = self.G["office"]("U.S. Rep", list(self.S["C"].index))
        self.cen = self.te[GROUP_COLS].values
        gnames = [g.replace("c_", "") for g in GROUP_COLS]
        self.sig = np.array([mc.GROUP_ERROR_SIGMA[g] for g in gnames]) * 200
        self.gi = gnames.index("hispanic")
        hisp = pd.Series(self.cen[self.is_h, self.gi] * 10, index=ids_h).reindex(range(1, 151)).fillna(0).values
        if len(self.county):
            self.cj_c, self.cj_h, raw = self.G["county"](list(self.county.county))
            self.cj_ss = (raw / raw.sum(1, keepdims=True)) @ hisp

    def local(self, wc, wh):
        return np.sqrt(self.lc) * (self.Uc @ wc.T.astype(np.float32)) + np.sqrt(self.lh) * (self.Uh @ wh.T.astype(np.float32))

    def run(self, version, shared_total=None, delta=0.0):
        f, ro = self.S["f"], self.ro
        T0, c_, h_ = f["T"][:N], f["c"][:N], f["h"][:N]
        Tc = float(f["T"].mean())
        T = {"statewide": T0, "legislative": T0 - Tc + self.TL, "actual_T": np.full(N, self.S["J"])}[version]
        hd = h_ - h_.mean()
        kind, names, actual, cols = [], [], [], []

        dev = f["sw_y"][:N] - T0[:, None]
        for i, r in enumerate(self.S["R"].index):
            names.append(r); kind.append("statewide"); actual.append(self.S["R"].actual[r])
            cols.append(T + dev[:, i] + self.kappa * ro["statewide"][r])

        C = self.S["C"]
        rest = f["cd_y"][:N] - T0[:, None] - c_[:, None] - h_[:, None] * C.ss.values[None, :]
        mu, sd = rest.mean(0), rest.std(0)
        loc = self.local(self.cd_c, self.cd_h)
        share = 1 - self.lc - self.lh
        cd = T[:, None] + c_[:, None] + h_[:, None] * C.ss.values[None, :] + mu + (rest - mu) * np.sqrt(share) + loc * sd + self.kappa[:, None] * ro["tier"]["us_house"]
        for j, d in enumerate(C.index):
            names.append(f"tx{d}"); kind.append("us_house"); actual.append(C.d_margin[d]); cols.append(cd[:, j])

        lin = self.lin[self.m]
        beta = self.est["slope"]
        ge_ = self.groups * self.sig
        ge_[:, self.gi] = hd * 10
        gerr = ge_ @ self.cen.T
        ig = float(lg.SIGMA_IDIO ** 2 - ((self.cen ** 2) @ (self.sig / 200) ** 2).mean())
        idio_sd = np.sqrt(max(ig, 0.0)) * 200
        idio = idio_sd * (np.sqrt(share) * self.eps + self.local(self.leg_c, self.leg_h))
        rl = np.where(self.is_h, ro["tier"]["state_house"], ro["tier"]["state_senate"])
        extra = 0.0
        if shared_total:
            e = np.sqrt(max(shared_total ** 2 - float(T0.std() * beta) ** 2 - self.est["competitive_sd"] ** 2, 0.0))
            extra = (e * self.extra)[:, None]
        leg = lin + beta * (T - self.TL + delta)[:, None] + self.lay[:, None] + extra + gerr + idio + self.kappa[:, None] * rl
        for k, r in enumerate(self.te.itertuples()):
            names.append(f"{'hd' if r.chamber_lower == 'house' else 'sd'}{int(r.district)}")
            kind.append("state_house" if r.chamber_lower == "house" else "state_senate")
            actual.append(2 * r.dem_2p_share * 100 - 100)
            cols.append(leg[:, k])

        if len(self.county):
            lam = self.lc + self.lh
            loc = self.local(self.cj_c, self.cj_h)
            shift = beta * (T - self.TL + delta)
            for i, r in enumerate(self.county.itertuples()):
                e = self.within * 200 * (np.sqrt(1 - lam) * self.t5[:, i] + np.sqrt(lam) * loc[:, i])
                names.append(r.county.replace(" ", "_")); kind.append("county"); actual.append(r.actual)
                cols.append(r.m0 + shift + self.lay_c + e + hd * self.cj_ss[i] + self.kappa * ro["tier"]["county_judge"])
        M = np.column_stack(cols).astype(np.float32)
        return {"T": T, "M": M, "W": M > 0, "names": names, "kind": np.array(kind), "actual": np.array(actual, float)}


def race_rows(x, res, version):
    M, a = res["M"], res["actual"]
    return pd.DataFrame({"year": x.year, "version": version, "race": res["names"], "kind": res["kind"], "actual": a, "mean": M.mean(0), "sd": M.std(0),
                         "p_d": res["W"].mean(0), "q10": np.percentile(M, 10, axis=0), "q90": np.percentile(M, 90, axis=0), "pit": (M < a[None, :]).mean(0)})


def seat_rows(x, res, version, shared=None):
    k, W, a = res["kind"], res["W"], res["actual"] > 0
    H = x.a.chamber_lower == "house"
    rows = []
    for name, m, fixed, tot in (("TX House", k == "state_house", x.fixed_h, int(H.sum())), ("TX Senate up", k == "state_senate", x.fixed_s, int((~H).sum())),
                                ("U.S. House", k == "us_house", CD_D[x.year] - int(a[k == "us_house"].sum()), None)):
        s = W[:, m].sum(1) + fixed
        act = int(a[m].sum()) + fixed
        rows.append({"year": x.year, "version": version, "shared": shared or "", "chamber": name, "mean": s.mean(), "q10": np.percentile(s, 10), "q90": np.percentile(s, 90),
                     "actual": act, "miss": s.mean() - act, "inside80": bool(np.percentile(s, 10) <= act <= np.percentile(s, 90)),
                     "pit": float((s < act).mean() + 0.5 * (s == act).mean()), "fixed_d": fixed, "scored": int(m.sum()), "seats": tot})
    return rows


def joint_rows(x, res, version):
    """Do races miss together the way the model says? Three checks, each against its own
    simulated distribution: shared level by office type, overlap between U.S. House and Texas
    House residuals sharing voters, and joint outcome pairs."""
    M, W, a, k = res["M"], res["W"], res["actual"], res["kind"]
    mean, sd = M.mean(0), M.std(0)
    out = []
    for kd in pd.unique(k):
        m = k == kd
        sim = (M[:, m] - mean[m]).mean(1)
        real = float((a[m] - mean[m]).mean())
        out.append({"year": x.year, "version": version, "check": f"level {kd}", "actual": real, "sim_sd": float(sim.std()), "pit": float((sim < real).mean())})
    idx = np.random.default_rng(1).choice(N, 2000, replace=False)
    il, ic = (k == "state_house"), (k == "us_house")
    O = x.leg_h[x.is_h] @ x.cd_h.T
    O = O / O.sum()

    def stat(rl, rc_):
        rl = (rl - rl.mean(-1, keepdims=True)) / sd[il]
        rc_ = (rc_ - rc_.mean(-1, keepdims=True)) / sd[ic]
        return rl, rc_

    rl, rc_ = stat(M[idx][:, il] - mean[il], M[idx][:, ic] - mean[ic])
    sim = np.einsum("ni,ij,nj->n", rl, O, rc_)
    rl, rc_ = stat(a[il] - mean[il], a[ic] - mean[ic])
    real = float(rl @ O @ rc_)
    out.append({"year": x.year, "version": version, "check": "overlap TX House x U.S. House", "actual": real, "sim_sd": float(sim.std()), "sim_mean": float(sim.mean()), "pit": float((sim < real).mean())})

    p = W.mean(0)
    keep = np.where((p > 0.1) & (p < 0.9))[0]
    w = W[:, keep].astype(np.float32)
    both = (w.T @ w) / N
    iu = np.triu_indices(len(keep), 1)
    y = (a[keep] > 0).astype(float)
    pi = p[keep]
    ev_b = np.outer(y, y)[iu]
    ev_d = (y[:, None] != y[None, :])[iu].astype(float)
    pj_b, pi_b = both[iu], np.outer(pi, pi)[iu]
    pj_d = (pi[:, None] + pi[None, :] - 2 * both)[iu]
    pi_d = (pi[:, None] + pi[None, :] - 2 * np.outer(pi, pi))[iu]
    out.append({"year": x.year, "version": version, "check": "pairs", "n_races": len(keep), "n_pairs": len(iu[0]),
                "brier_both_joint": float(((pj_b - ev_b) ** 2).mean()), "brier_both_indep": float(((pi_b - ev_b) ** 2).mean()),
                "brier_split_joint": float(((pj_d - ev_d) ** 2).mean()), "brier_split_indep": float(((pi_d - ev_d) ** 2).mean()),
                "pred_split": float(pj_d.mean()), "indep_split": float(pi_d.mean()), "actual_split": float(ev_d.mean())})
    return out


def hd_weights(year, districts):
    """Prior presidential two-party vote by House district, on the lines the race ran under."""
    v = pd.read_csv(HIST / PRES_VOTES[year])
    return districts.astype(int).map(dict(zip(v.district, v.dem_votes + v.rep_votes))).values


def formula_center(year, c):
    """The legislative center for a cycle, read at the environment that actually happened."""
    a = leg_frame(year)
    house = (a.chamber_lower == "house").values
    lin = (rc.predict(a, c) + LEG_SLOPE * (sw.NAT[year] - ENV_BASE[year])) * 200 - 100
    return float(np.average(lin[house], weights=hd_weights(year, a.district[house])))


def judicial():
    s = pd.read_csv(sw.RAW / "results_statewide.csv")
    return s[(s.race == "judicial") & s.contested].groupby("year").d_margin.mean()


def seat_score(s, act):
    """Log score of the House seat count, with a three-seat kernel so one draw does not decide it."""
    return float(np.log(max((np.abs(s - act) <= 1).mean() / 3, 1e-4)))


def probe(envs):
    """Three follow-ups the first pass raised. Is the formula's center on the same scale as the
    judicial vote? How large should the shared legislative error be? Is the place share too high?"""
    J = judicial()
    print("=== Formula center minus judicial vote, midterms, each read with the formula fit without that cycle ===")
    gap = {y: formula_center(y, leg_fit(y)) - J[y] for y in (2014, 2018, 2022)}
    print({k: round(v, 2) for k, v in gap.items()})
    all_c = leg_fit(None)
    full = {y: formula_center(y, all_c) - J[y] for y in (2014, 2018, 2022, 2024)}
    print("same, formula fit on all three midterms:", {k: round(v, 2) for k, v in full.items()})
    print(f"2026 delta from the three midterms: {np.mean([full[y] for y in (2014, 2018, 2022)]):.2f}; held-out gaps mean {np.mean(list(gap.values())):.2f}")

    rows, grid, place = [], [], []
    for y in YEARS:
        x = Cycle(y, envs[y])
        k_ = (2014, 2018, 2022)
        c = leg_fit(y)
        d_ = float(np.mean([formula_center(t, c) - J[t] for t in k_ if t != y]))
        house_fixed, act = x.fixed_h, None
        for name, ver, delta in (("statewide", "statewide", 0.0), ("statewide, calibrated", "statewide", d_), ("legislative", "legislative", 0.0)):
            for st in (None, 4.5, 5.5, SHIPPED_SHARED, 8.0):
                r = x.run(ver, shared_total=st, delta=delta)
                m = r["kind"] == "state_house"
                s = r["W"][:, m].sum(1) + house_fixed
                act = int((r["actual"][m] > 0).sum()) + house_fixed
                grid.append({"year": y, "version": name, "shared": st or 0, "score": seat_score(s, act), "mean": s.mean(), "q10": np.percentile(s, 10), "q90": np.percentile(s, 90), "actual": act,
                             "pit": float((s < act).mean() + .5 * (s == act).mean())})
                if st is None:
                    miss = (r["M"][:, :].mean(0) - r["actual"])
                    rows.append({"year": y, "version": name, "delta": delta, "house": s.mean(), "actual": act, "house_miss": s.mean() - act,
                                 **{f"miss_{kd}": float(miss[r["kind"] == kd].mean()) for kd in ("state_house", "state_senate", "county") if (r["kind"] == kd).any()},
                                 "brier_leg": float(((r["W"][:, m].mean(0) - (r["actual"][m] > 0)) ** 2).mean())})
        lc0, lh0 = x.lc, x.lh
        for sc in (1.0, 0.7, 0.5, 0.35, 0.2):
            x.lc, x.lh = lc0 * sc, lh0 * sc
            r = x.run("statewide")
            j = [q for q in joint_rows(x, r, "statewide") if q["check"].startswith("overlap")][0]
            m = r["kind"] == "state_house"
            s = r["W"][:, m].sum(1) + house_fixed
            place.append({"year": y, "scale": sc, "place_share": (lc0 + lh0) * sc, "real": j["actual"], "sim": j["sim_mean"], "pit": j["pit"], "seat_sd": s.std()})
        x.lc, x.lh = lc0, lh0
    print("\n=== Statewide version with the center put on the judicial scale, held out ===")
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    g = pd.DataFrame(grid)
    g.to_csv(OUT / "whole_election_backtest_shared_grid.csv", index=False)
    print("\n=== TX House seat count against shared error: mean log score, higher is better; 0 = T + 1.65 as in the model ===")
    print(g.pivot_table(index="version", columns="shared", values="score").round(2).to_string())
    print(g[g.shared.isin([0, SHIPPED_SHARED])].round(2).to_string(index=False))
    print("\n=== Place share against the overlap between TX House and U.S. House residuals ===")
    print(pd.DataFrame(place).pivot_table(index="scale", values=["real", "sim", "pit", "seat_sd"]).round(3).to_string())
    print(pd.DataFrame(place).round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="poll", choices=["poll", "538", "rcp", "actual"])
    ap.add_argument("--probe", action="store_true")
    a_ = ap.parse_known_args()[0]
    envs = {y: {"poll": (GB.final_538[y] + GB.final_rcp[y]) / 2, "538": GB.final_538[y], "rcp": GB.final_rcp[y], "actual": sw.NAT[y]}[a_.env] for y in YEARS}
    if a_.probe:
        probe(envs)
        return
    races, seats, joint = [], [], []
    for y in YEARS:
        x = Cycle(y, envs[y])
        S = x.S
        f = S["f"]
        print(f"\n===== {y}: env {envs[y]:+.1f}; T prior {S['Tprior']:+.1f} ± {S['Tprior_sd']:.1f}, posterior {f['T'].mean():+.1f} ± {f['T'].std():.1f}; "
              f"legislative center {x.TL:+.1f}; actual J {S['J']:+.1f} =====")
        print("  held-out estimates:", {k: round(v, 3) for k, v in x.est.items()}, f"; scored: {len(S['R'])} statewide, {len(S['C'])} U.S. House, {int(x.is_h.sum())} TX House, "
              f"{int((~x.is_h).sum())} TX Senate, {len(x.county)} county; polls {S['R'].n_polls.to_dict()}; no money {list(S['R'].index[S['R'].no_money])}")
        for v in ("statewide", "legislative", "actual_T"):
            r = x.run(v)
            races.append(race_rows(x, r, v))
            seats += seat_rows(x, r, v)
            joint += joint_rows(x, r, v)
            seats += seat_rows(x, x.run(v, shared_total=SHIPPED_SHARED), v, shared=SHIPPED_SHARED)
    R = pd.concat(races, ignore_index=True)
    Se = pd.DataFrame(seats)
    Jn = pd.DataFrame(joint)
    tag = "" if a_.env == "poll" else f"_{a_.env}_env"
    R.to_csv(OUT / f"whole_election_backtest_races{tag}.csv", index=False)
    Se.to_csv(OUT / f"whole_election_backtest_summary{tag}.csv", index=False)
    Jn.to_csv(OUT / f"whole_election_backtest_joint{tag}.csv", index=False)

    print("\n=== Seat counts, D seats (10-90 band) ===")
    print(Se[Se.shared == ""][["year", "version", "chamber", "mean", "q10", "q90", "actual", "miss", "inside80", "pit", "scored", "fixed_d"]].round(2).to_string(index=False))
    print("\n=== Same, legislative shared error at the shipped 6.7 ===")
    print(Se[(Se.shared != "") & (Se.chamber != "U.S. House")][["year", "version", "chamber", "mean", "q10", "q90", "actual", "inside80", "pit"]].round(2).to_string(index=False))
    print("\n=== Statewide races ===")
    s = R[(R.kind == "statewide") & (R.version != "actual_T")]
    print(s[["year", "version", "race", "actual", "mean", "sd", "p_d", "pit"]].round(2).to_string(index=False))
    print("\n=== Race scoring by version and office ===")
    R["miss"] = R["mean"] - R.actual
    R["won"] = (R.actual > 0).astype(float)
    R["in80"] = (R.actual >= R.q10) & (R.actual <= R.q90)
    g = R.groupby(["version", "year", "kind"])
    t = g.apply(lambda d: pd.Series({"n": len(d), "mean_miss": d.miss.mean(), "rmse": np.sqrt((d.miss ** 2).mean()), "within_sd": d.miss.std(), "stated_sd": d.sd.mean(), "in80": d.in80.mean(),
                                     "brier": ((d.p_d - d.won) ** 2).mean(), "called": ((d.p_d > .5) == (d.won == 1)).mean()}), include_groups=False)
    print(t.round(3).to_string())
    print("\n=== Pooled by version: competitive contested races (P between 10% and 90%) ===")
    c = R[(R.p_d > .1) & (R.p_d < .9)]
    print(c.groupby("version").apply(lambda d: pd.Series({"n": len(d), "mean_p": d.p_d.mean(), "won": d.won.mean(), "brier": ((d.p_d - d.won) ** 2).mean(), "in80": d.in80.mean()}), include_groups=False).round(3).to_string())
    print("\n=== Joint checks ===")
    print(Jn[Jn.check != "pairs"][["year", "version", "check", "actual", "sim_mean", "sim_sd", "pit"]].round(2).to_string(index=False))
    print(Jn[Jn.check == "pairs"].drop(columns=["check", "actual", "sim_sd", "sim_mean", "pit"]).round(3).to_string(index=False))
    # reliability of the probabilities, pooled
    print("\n=== Reliability, contested races, statewide version ===")
    d = R[R.version == "statewide"].copy()
    d["bin"] = pd.cut(d.p_d, [-.01, .02, .1, .3, .5, .7, .9, .98, 1.01])
    print(d.groupby("bin", observed=True).agg(n=("won", "size"), p=("p_d", "mean"), won=("won", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
