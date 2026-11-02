"""
statewide_model.py

Joint 2026 forecast for Texas statewide races and the 38 congressional seats.

Every race is priced off one latent number, T: the Texas party vote margin with
candidates stripped out, measured historically by the Supreme Court / CCA composite.

  statewide race r:   y_r = T + dev_r
  congressional d:    y_d = T + c + L_d + inc_d + fin_d + war_d + h * ssvr_d + e_d
  poll of race x:     p_x = y_x + b + noise     (b = the cycle's shared poll miss)

Priors come from Texas history only, by race type: statewide effects from the 61
statewide races 2002-2024 (55 without presidential races), congressional effects from
the 210 two-candidate contested Texas U.S. House races 2012-2024. Nothing is borrowed from the legislative model's coefficients.
Conditioning on the polls is exact (the model is linear-Gaussian), so the amount of
weight polls get is a consequence of the measured error sizes, not a chosen dial.

Usage: python src/statewide_model.py            full fit, backtest and forecast
Outputs: output/statewide_forecast_2026.csv, output/congress_forecast_2026.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statewide_finance import load_fec  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "statewide"
OUTDIR = ROOT / "output"

# National House popular vote, D margin (two-party). 2026 = the legislative model's
# current generic-ballot input.
NAT = {2002: -4.6, 2004: -2.6, 2006: 8.0, 2008: 10.6, 2010: -6.8, 2012: 1.2, 2014: -5.7, 2016: -1.1,
       2018: 8.6, 2020: 3.1, 2022: -2.8, 2024: -2.7}
ENV_2026, ENV_SD_2026 = 9.2, 2.5
ELECTION_2026 = pd.Timestamp("2026-11-03")
SHRINK_K = 2.93          # same form as the legislative WAR persistence, n/(n+k)

# Sponsors with a party stake. Excluded from the average, matching the history the
# error sizes were measured on (partisan polls were excluded there too).
PARTISAN = r"Quantus|Stratus|Overton|SoCal|Trafalgar|Cor Strategies|Pulse Decision|Guidant|GBAO|YouGov Blue|Texas Public Opinion|GS Strategy"
RACES_2026 = {
    "governor": dict(r="Abbott", d="Hinojosa", tier="top", inc=-1, r_prev="Abbott", d_prev=None),
    "us_senate": dict(r="Paxton", d="Talarico", tier="top", inc=0, r_prev="Paxton", d_prev=None),
    "lt_governor": dict(r="Patrick", d="Goodwin", tier="mid", inc=-1, r_prev="Patrick", d_prev=None),
    "attorney_general": dict(r="Middleton", d="Johnson", tier="mid", inc=0, r_prev=None, d_prev=None),
    "comptroller": dict(r="Huffines", d="Eckhardt", tier="low", inc=0, r_prev=None, d_prev=None),
    "land_commissioner": dict(r="Buckingham", d="Flores", tier="low", inc=-1, r_prev="Buckingham", d_prev=None),
    "ag_commissioner": dict(r="Sheets", d="Tucker", tier="low", inc=0, r_prev=None, d_prev=None),
    "railroad_commissioner": dict(r="French", d="Rosenthal", tier="low", inc=0, r_prev=None, d_prev=None),
}


# Controversy in the candidate's first general after it broke, D-margin points added to
# dev_hat. scripts/scandal_tx23_tests.py: statewide cases alone give -3.6 per person
# (p 0.06), but adding Cuellar, who ran +7.9 after his raid and indictment, gives -1.3
# (p 0.27); a quarter of unflagged races run as far behind too. The evidence does not
# establish a penalty. Set at 2 for French only as Brennan's judgment, not an estimate.
# Paxton is off: his record already carries it through prior_own.
SCANDAL_2026 = {"railroad_commissioner": 2.0}


def log_ratio(d, r):
    return np.clip(np.log((np.asarray(d, float) + 25e3) / (np.asarray(r, float) + 25e3)), -3.5, 3.5)


# ---------------------------------------------------------------- statewide layer
def senate_money() -> pd.DataFrame:
    rows = []
    for y in (2012, 2014, 2018, 2020, 2024):
        f = load_fec(y)
        rows.append(f[f.office == "S"][["last", "pty", "receipts"]].assign(year=y))
    return pd.concat(rows)


def statewide_table() -> pd.DataFrame:
    s = pd.read_csv(RAW / "statewide_dev.csv")
    s = s[s.race != "president"].copy()
    fin = pd.read_csv(RAW / "finance_statewide.csv")
    s = s.merge(fin[["year", "office", "d_raised", "r_raised"]], on=["year", "office"], how="left")
    m = senate_money()
    for i, r in s[s.race == "us_senate"].iterrows():
        for p, nm in (("D", r.d_name), ("R", r.r_name)):
            g = m[(m.year == r.year) & (m.pty == p) & m["last"].str.contains(str(nm).lower().split()[-1], regex=False)]
            if len(g):
                s.loc[i, f"{p.lower()}_raised"] = g.receipts.max()
    s["money"] = np.nan
    has = s.d_raised.notna() & s.r_raised.notna()
    s.loc[has, "money"] = log_ratio(s.d_raised[has], s.r_raised[has])
    s["top"] = (s.tier == "top").astype(float)
    s["mid"] = (s.tier == "mid").astype(float)
    s["low"] = (s.tier == "low").astype(float)
    # the candidate's own earlier statewide record, in D-margin terms
    s = s.sort_values("year").reset_index(drop=True)
    return s


def add_prior_dev(s: pd.DataFrame, coef_inc: float, tiers: dict) -> pd.DataFrame:
    s = s.copy()
    s["base"] = s.tier.map(tiers) + coef_inc * s.inc
    s["own"] = s.dev - s.base
    prior = []
    for r in s.itertuples():
        v = 0.0
        for party, nm, sign in (("d", r.d_name, 1), ("r", r.r_name, -1)):
            col = f"{party}_name"
            h = s[(s.year < r.year) & (s[col] == nm)]
            if len(h):
                n = len(h)
                v += sign * (n / (n + SHRINK_K)) * (sign * h.own).tail(3).mean()
        prior.append(v)
    s["prior_own"] = prior
    return s


def fit_dev(s: pd.DataFrame, terms: list[str]):
    d = s.dropna(subset=terms + ["dev"])
    return sm.OLS(d.dev, d[["top", "mid", "low"] + terms]).fit()


def statewide_layer(s: pd.DataFrame) -> tuple:
    base = fit_dev(s, ["inc"])
    tiers = {t: base.params[t] for t in ("top", "mid", "low")}
    s = add_prior_dev(s, base.params["inc"], tiers)
    specs = {"tier only": [], "+inc": ["inc"], "+inc+money": ["inc", "money"], "+inc+prior": ["inc", "prior_own"],
             "+inc+money+prior": ["inc", "money", "prior_own"]}
    print("\n=== Statewide: race minus judicial composite, leave-one-cycle-out (margin pts) ===")
    cyc = sorted(s.year.unique())
    best = None
    for name, terms in specs.items():
        err = []
        for c in cyc:
            tr, te = s[s.year != c], s[s.year == c].dropna(subset=terms)
            if te.empty:
                continue
            f = fit_dev(tr, terms)
            err.append(te.dev - f.predict(te[["top", "mid", "low"] + terms]))
        e = pd.concat(err)
        print(f"  {name:<19} rms {np.sqrt((e ** 2).mean()):.2f}  n {len(e)}")
    final_terms = ["inc", "money", "prior_own"]
    f = fit_dev(s, final_terms)
    print(f.summary2().tables[1].round(3)[["Coef.", "Std.Err.", "t"]].to_string())
    resid = s.dropna(subset=final_terms).assign(res=f.resid)
    sds = resid.groupby("tier").res.std().to_dict()
    print("  residual sd by tier:", {k: round(v, 2) for k, v in sds.items()})
    return s, f, sds


# ---------------------------------------------------------------- T prior
def t_prior(s: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    sw = pd.read_csv(RAW / "results_statewide.csv")
    j = sw[(sw.race == "judicial") & sw.contested].groupby("year").d_margin.mean()
    t = pd.DataFrame({"J": j, "env": pd.Series(NAT)}).dropna()
    t["dJ"], t["dE"] = t.J.diff(), t.env.diff()
    # +1 stepping into a midterm, -1 stepping into a presidential year
    t["mid"] = [1.0 if y % 4 == 2 else -1.0 for y in t.index]
    d = t.dropna()
    res = {}
    for name, cols in (("national change only", ["dE"]), ("+ midterm step", ["dE", "mid"])):
        b = sm.OLS(d.dJ, d[cols]).fit()
        loo = []
        for y in d.index:
            f = sm.OLS(d.dJ.drop(y), d[cols].drop(y)).fit()
            loo.append(d.dJ[y] - float((f.params * d.loc[y, cols]).sum()))
        res[name] = (b, float(np.sqrt(np.mean(np.square(loo)))))
        print(f"\n=== T prior, {name}: " + ", ".join(f"{c} {b.params[c]:+.2f} (se {b.bse[c]:.2f})" for c in cols) +
              f", LOO rms {res[name][1]:.2f}, n {len(d)} ===")
    print(t.round(1).T.to_string())
    last = t.J.iloc[-1]
    # The midterm step does not earn its place: LOO 2.79 vs 2.74, joint backtest 3.07 vs 2.88.
    b, sd = res["national change only"]
    mu = last + b.params["dE"] * (ENV_2026 - NAT[2024])
    sd_total = float(np.sqrt(sd ** 2 + (b.params["dE"] * ENV_SD_2026) ** 2))
    print(f"  2024 J {last:.1f} -> structural T 2026 {mu:.1f} (sd {sd_total:.1f})")
    return float(mu), sd_total, t


# ---------------------------------------------------------------- congressional layer
def house_table() -> pd.DataFrame:
    h = pd.read_csv(RAW / "results_us_house.csv")
    h = h[(h.year >= 2012) & h.contested & (h.n_d == 1) & (h.n_r == 1)].copy()
    sw = pd.read_csv(RAW / "results_statewide.csv")
    J = sw[(sw.race == "judicial") & sw.contested].groupby("year").d_margin.mean()
    P = sw[(sw.race == "president") & sw.contested].set_index("year").d_margin
    h["J"] = h.year.map(J)
    py = h.year.map(lambda y: y - 2 if y % 4 == 2 else y - 4)
    h["lean_p"] = h.prevpres_margin - py.map(P)
    h["lean_j"] = h.prevjud_margin - py.map(J)
    h["rel"] = h.d_margin - h.J
    h["inc_d"], h["inc_r"] = h.d_inc.astype(float), h.r_inc.astype(float)
    fin = pd.read_csv(RAW / "finance_us_house.csv")
    h = h.merge(fin, on=["year", "dist"], how="left")
    h["money"] = log_ratio(h.d_raised.fillna(0), h.r_raised.fillna(0))
    w = pd.read_csv(RAW / "district_war_long.csv")
    us = w[w.type == "U.S. Rep"]
    ss = us.groupby(["year", "dist"]).ssvr.first().rename("ssvr").reset_index()
    h = h.merge(ss, on=["year", "dist"], how="left")
    state_ss = h.groupby("year").ssvr.transform("mean")
    h["ss"] = (h.ssvr - state_ss) / 10
    pri = []
    for r in h.itertuples():
        v = 0.0
        for party, nm, sign in (("D", r.d_name, 1), ("R", r.r_name, -1)):
            g = us[(us.year < r.year) & (us.party == party) & (us.name == nm) & (us.dist == r.dist)].sort_values("year").tail(2)
            if len(g):
                v += sign * g.war.mean()
        pri.append(v)
    h["prior_war"] = pri
    return h.dropna(subset=["lean_p", "lean_j"])


CD_TERMS = ["lean_p", "lean_j", "inc_d", "inc_r", "money", "prior_war"]


def congress_layer(h: pd.DataFrame):
    print(f"\n=== U.S. House relative to J, leave-one-cycle-out (n={len(h)}, cycles {sorted(h.year.unique())}) ===")
    specs = {"lean only": ["lean_p", "lean_j"], "+inc": ["lean_p", "lean_j", "inc_d", "inc_r"],
             "+inc+money": ["lean_p", "lean_j", "inc_d", "inc_r", "money"],
             "+inc+prior": ["lean_p", "lean_j", "inc_d", "inc_r", "prior_war"], "+inc+money+prior": CD_TERMS,
             "+all+ssvr": CD_TERMS + ["ss"]}
    for name, t in specs.items():
        err = []
        for c in sorted(h.year.unique()):
            tr, te = h[h.year != c], h[h.year == c]
            f = sm.OLS(tr.rel, sm.add_constant(tr[t])).fit()
            e = te.rel - f.predict(sm.add_constant(te[t], has_constant="add"))
            err.append(e)
        e = pd.concat(err)
        print(f"  {name:<19} rms {np.sqrt((e ** 2).mean()):.2f}  bias by cycle " +
              " ".join(f"{str(c)[2:]}:{v:+.1f}" for c, v in e.groupby(h.year).mean().items()))
    tot = sm.OLS(h.rel, sm.add_constant(h[["lean_p", "lean_j", "inc_d", "inc_r"]])).fit()
    print(f"  total incumbency (no money/WAR controls): D {tot.params['inc_d']:+.2f} (se {tot.bse['inc_d']:.2f}), R {tot.params['inc_r']:+.2f} (se {tot.bse['inc_r']:.2f})"
          f"; among races within 15: {sm.OLS(h[abs(h.d_margin) < 15].rel, sm.add_constant(h[abs(h.d_margin) < 15][['lean_p', 'lean_j', 'inc_d', 'inc_r']])).fit().params[['inc_d', 'inc_r']].round(2).to_dict()}")
    print(f"  incumbents per cycle D/R: {h.groupby('year')[['inc_d', 'inc_r']].sum().astype(int).to_dict('index')}")
    f = sm.OLS(h.rel, sm.add_constant(h[CD_TERMS])).fit()
    print(f.summary2().tables[1].round(3)[["Coef.", "Std.Err.", "t"]].to_string())
    h = h.assign(res=f.resid)
    cyc = h.groupby("year").res.mean()
    slopes = {y: np.polyfit(g.ss.fillna(0), g.res - g.res.mean(), 1)[0] for y, g in h.groupby("year")}
    close = h[abs(h.d_margin) < 15]
    print(f"  cycle mean residual sd {cyc.std():.2f}: {cyc.round(1).to_dict()}")
    print(f"  Hispanic slope by cycle (pts per 10pt SSVR): { {k: round(v, 2) for k, v in slopes.items()} }  sd {np.std(list(slopes.values()), ddof=1):.2f}")
    within = (h.res - h.groupby("year").res.transform("mean"))
    print(f"  residual sd within cycle {within.std():.2f}; races within 15 pts {within[close.index].std():.2f} (n={len(close)})")
    return f, float(cyc.std()), float(np.std(list(slopes.values()), ddof=1)), float(within[close.index].std())


def parse_ssvr() -> pd.Series:
    import pdfplumber
    out = {}
    with pdfplumber.open(RAW / "map" / "r119_24.pdf") as p:
        for pg in p.pages:
            for ln in (pg.extract_text() or "").split("\n"):
                m = re.match(r"^(\d{1,2}) [\d,]+ [\d,]+ [\d.]+ [\d,]+ \(.*?\) [\d.]+ \(.*?\) [\d,]+ ([\d.]+) ", ln)
                if m:
                    out[int(m.group(1))] = float(m.group(2))
    return pd.Series(out, name="ssvr")


def congress_2026(f) -> pd.DataFrame:
    b = pd.read_csv(RAW / "cd_baseline_2026.csv").set_index("dist")
    r = pd.read_csv(RAW / "cd_races_2026.csv").set_index("dist")
    fin = pd.read_csv(RAW / "finance_us_house.csv").query("year == 2026").set_index("dist")
    w = pd.read_csv(RAW / "district_war_long.csv")
    us = w[w.type == "U.S. Rep"]
    sw = pd.read_csv(RAW / "results_statewide.csv")
    jud24 = sw[(sw.race == "judicial") & (sw.year == 2024) & sw.contested]
    keys = ["m2024_" + re.sub(r"[^a-z0-9]+", "_", o.lower()).strip("_") for o in jud24.office]
    jmean = jud24.d_margin.mean()
    pres = sw[(sw.race == "president") & (sw.year == 2024)].d_margin.iloc[0]
    d = pd.DataFrame(index=b.index)
    d["lean_p"] = b.m2024_president - pres
    d["lean_j"] = b[keys].mean(axis=1) - jmean
    d["inc_d"], d["inc_r"] = r.d_inc.astype(float), r.r_inc.astype(float)
    d["money"] = log_ratio(fin.d_raised, fin.r_raised)
    pri = []
    for x in r.itertuples():
        v = 0.0
        for party, nm, sign in (("D", x.d_name, 1), ("R", x.r_name, -1)):
            if not isinstance(nm, str) or not nm:
                continue
            last = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", nm).split()[-1]
            g = us[(us.party == party) & (us.name.str.lower() == last.lower())].sort_values("year").tail(2)
            inc = bool(x.d_inc if party == "D" else x.r_inc)
            if len(g) and inc:
                v += sign * g.war.mean()
        pri.append(v)
    d["prior_war"] = pri
    ssvr = parse_ssvr()
    d["ss"] = (ssvr - ssvr.mean()) / 10
    d["r_name"], d["d_name"] = r.r_name, r.d_name
    d["pres24"] = b.m2024_president
    d["rel_hat"] = f.predict(sm.add_constant(d[CD_TERMS], has_constant="add"))
    d["d_raised"], d["r_raised"] = fin.d_raised, fin.r_raised
    for c in ("cook", "silver"):
        d[c] = r[c] if c in r else ""
    return d


# ---------------------------------------------------------------- polls
def poll_average(p: pd.DataFrame, race: str, include_partisan=False) -> dict:
    g = p[p.race == race].copy()
    g["end"] = pd.to_datetime(g.field_end)
    g = g[g.end >= ELECTION_2026 - pd.Timedelta(days=30)]
    if not include_partisan:
        g = g[~g.pollster.str.contains(PARTISAN, regex=True, na=False)]
    if g.empty:
        return {"avg": np.nan, "n": 0, "und": np.nan}
    g["rank"] = g.population.map({"LV": 0, "RV": 1}).fillna(2)
    g = g.sort_values(["pollster", "end", "rank"], ascending=[True, False, True]).drop_duplicates("pollster")
    g["margin"] = (g.d_pct - g.r_pct) / (g.d_pct + g.r_pct) * 100
    days = (ELECTION_2026 - g.end).dt.days
    w = np.sqrt(g.sample_size.fillna(800).clip(upper=2000) / 800) * 0.5 ** (days / 14)
    return {"avg": float((g.margin * w).sum() / w.sum()), "n": len(g), "und": float((100 - g.r_pct - g.d_pct).mean()),
            "simple": float(g.margin.mean())}


# scripts/statewide_pollster_history.py: cycle mean miss +1.47 all six cycles, -0.55 in
# midterms, +3.50 with Trump or a presidential race on the ballot. n is too small to
# trust the split or to ignore it, so the mean is set between the two; sd is measured.
POLL_B_MEAN, POLL_B_SD, POLL_RACE_SD = 0.5, 3.69, 1.44


def condition(mu0, cov0, H, y, R):
    S = H @ cov0 @ H.T + R
    K = cov0 @ H.T @ np.linalg.inv(S)
    return mu0 + K @ (y - H @ mu0), cov0 - K @ H @ cov0


def backtest(s, f_dev, t_tbl, sds):
    """Leave-one-cycle-out on the race-cycles with polls: polls alone, structure alone, joint."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from statewide_pollster_history import load_polls
    p = load_polls()
    avg = p[~p.partisan].groupby(["cycle", "race"]).agg(avg=("margin", "mean"), n=("margin", "size"), actual=("actual", "first")).reset_index()
    avg = avg[avg.race != "president"]
    avg = avg[avg.race.isin(s.race.unique())]
    errs = p[~p.partisan].groupby("cycle").err.mean()
    rows = []
    base_terms = ["inc", "money", "prior_own"]
    for c in sorted(avg.cycle.unique()):
        tr = s[s.year != c]
        f = fit_dev(tr, base_terms)
        dt = t_tbl.dropna()
        dt2 = dt.drop(c) if c in dt.index else dt
        if c not in dt.index:
            continue
        tf = sm.OLS(dt2.dJ, dt2[["dE"]]).fit()
        Tp = t_tbl.J.shift(1)[c] + float(tf.params["dE"] * dt.dE[c])
        Tsd = float(np.sqrt(np.mean(tf.resid ** 2)))
        bm = errs.drop(c).mean()
        bs = max(errs.drop(c).std(), 2.5)
        a = avg[avg.cycle == c]
        ss = s[s.year == c].set_index("race")
        rc = [r for r in a.race if r in ss.index and pd.notna(ss.loc[r, "money"])]
        if not rc:
            continue
        k = len(rc)
        mu = np.zeros(k + 2); cov = np.zeros((k + 2, k + 2))
        mu[0], cov[0, 0] = Tp, Tsd ** 2
        mu[1], cov[1, 1] = bm, bs ** 2
        for i, r in enumerate(rc):
            x = ss.loc[[r], ["top", "mid", "low"] + base_terms]
            mu[2 + i] = float(f.predict(x).iloc[0])
            cov[2 + i, 2 + i] = sds.get(ss.loc[r, "tier"], 5.0) ** 2
        H = np.zeros((k, k + 2)); H[:, 0] = 1; H[:, 1] = 1
        for i in range(k):
            H[i, 2 + i] = 1
        y = a.set_index("race").loc[rc, "avg"].values
        n = a.set_index("race").loc[rc, "n"].values
        R = np.diag(POLL_RACE_SD ** 2 + 16 / n)
        mp, cp = condition(mu, cov, H, y, R)
        for i, r in enumerate(rc):
            act = ss.loc[r, "d_margin"]
            v = np.zeros(k + 2); v[0] = 1; v[2 + i] = 1
            rows.append({"cycle": c, "race": r, "n": int(n[i]), "actual": act, "polls": y[i] - act,
                         "structure": mu[0] + mu[2 + i] - act, "joint": mp[0] + mp[2 + i] - act,
                         "sd": float(np.sqrt(v @ cp @ v))})
    b = pd.DataFrame(rows)
    print("\n=== Backtest, leave-one-cycle-out, race-cycles with polls (error = forecast - actual) ===")
    print(b.round(1).to_string(index=False))
    for c in ("polls", "structure", "joint"):
        print(f"  {c:<10} rms {np.sqrt((b[c] ** 2).mean()):.2f}  mean {b[c].mean():+.2f}")
    z = (b.joint / b.sd).abs()
    print(f"  joint stated sd mean {b.sd.mean():.2f}; inside 80% band {(z < 1.2816).mean():.0%}, inside 1 sd {(z < 1).mean():.0%}; "
          f"called winner right: joint {(np.sign(b.joint + b.actual) == np.sign(b.actual)).mean():.0%}")
    return b


def build():
    """Fit every layer and assemble the 2026 inputs."""
    s = statewide_table()
    s, f_dev, sds = statewide_layer(s)
    Tmu, Tsd, t_tbl = t_prior(s)
    backtest(s, f_dev, t_tbl, sds)
    h = house_table()
    f_cd, c_sd, h_sd, e_sd = congress_layer(h)

    # First-timer district record at a heavily reduced rate. Measured transfer: 0.18
    # (se 0.12) across office types, -0.18 (se 0.25) among those who held the first
    # office, 0.37 (se 0.33) into a first statewide run, n=11. Set at 0.2, the middle of
    # a range that cannot be told from zero; it moves no race by more than a point.
    ft = pd.read_csv(RAW / "first_timer_war_2026.csv")
    KAPPA = 0.2
    polls = pd.read_csv(RAW / "polls_2026.csv")
    fin = pd.read_csv(RAW / "finance_statewide.csv").query("year == 2026").set_index("race")
    f26 = load_fec(2026)
    rows = []
    for race, c in RACES_2026.items():
        if race == "us_senate":
            dm = f26[f26["last"].str.contains("talarico")].receipts.max()
            rm = f26[f26["last"].str.contains("paxton") & (f26.office == "S")].receipts.max()
        else:
            dm, rm = fin.loc[race, "d_raised"], fin.loc[race, "r_raised"]
        prior = 0.0
        if c["r_prev"]:
            hist = s[s.r_name == c["r_prev"]]
            if len(hist):
                n = len(hist)
                prior += (n / (n + SHRINK_K)) * hist.own.tail(3).mean()
        ftw = 0.0
        for party, sign in (("D", 1), ("R", -1)):
            g = ft[(ft.race == race) & (ft.party == party) & (ft.n > 0)]
            if len(g):
                g = g.iloc[0]
                ftw += sign * KAPPA * (g.n / (g.n + SHRINK_K)) * g.war
        x = pd.DataFrame([{"top": float(c["tier"] == "top"), "mid": float(c["tier"] == "mid"), "low": float(c["tier"] == "low"),
                           "inc": c["inc"], "money": float(log_ratio(dm, rm)), "prior_own": prior}])
        devhat = float(f_dev.predict(x[["top", "mid", "low", "inc", "money", "prior_own"]]).iloc[0]) + ftw + SCANDAL_2026.get(race, 0.0)
        pa = poll_average(polls, race)
        pa2 = poll_average(polls, race, include_partisan=True)
        rows.append({"race": race, "r": c["r"], "d": c["d"], "tier": c["tier"], "dev_hat": devhat, "money": float(log_ratio(dm, rm)),
                     "prior_own": prior, "ftw": ftw, "d_raised": dm, "r_raised": rm, "poll_avg": pa["avg"], "poll_simple": pa.get("simple"),
                     "poll_all": pa2["avg"], "n_polls": pa["n"], "und": pa["und"]})
    R = pd.DataFrame(rows).set_index("race")
    print("\n=== 2026 statewide inputs ===")
    print(R.drop(columns=["d_raised", "r_raised"]).round(1).to_string())

    C = congress_2026(f_cd)
    dp = pd.read_csv(RAW / "cd_polls_2026.csv")
    dp["end"] = pd.to_datetime(dp.field_end)
    dp = dp[dp.end >= ELECTION_2026 - pd.Timedelta(days=75)]

    return dict(R=R, C=C, dp=dp, Tmu=Tmu, Tsd=Tsd, c_sd=c_sd, h_sd=h_sd, e_sd=e_sd, sds=sds, f_dev=f_dev, s=s, f_cd=f_cd)


def main():
    b = build()
    R, C, dp, Tmu, Tsd, c_sd, h_sd, e_sd, sds, f_dev = (b[k] for k in ("R", "C", "dp", "Tmu", "Tsd", "c_sd", "h_sd", "e_sd", "sds", "f_dev"))
    forecast(R, C, dp, Tmu, Tsd, c_sd, h_sd, e_sd, sds, POLL_B_MEAN, save=True)
    print("\n=== Sensitivity (same seed) ===")
    print(f"{'scenario':<34}{'Gov':>7}{'Sen':>7}{'LtG':>7}{'AG':>7}{'Comp':>7}{'RRC':>7}{'seats':>7}{'P Sen':>7}{'P Gov':>7}")
    for name, kw in (("base", {}), ("polls miss as all-cycle mean +1.47", dict(bm=1.47)), ("polls miss as midterm mean -0.55", dict(bm=-0.55)),
                     ("2018 repeat: polls too R by 4.9", dict(bm=-4.9)), ("2024 repeat: polls too D by 5.7", dict(bm=5.7)),
                     ("T prior at legislative R+8.3", dict(tm=-8.3)), ("polls only (flat T prior)", dict(tsd=30.0)),
                     ("with partisan polls", dict(partisan=True)), ("no district polls", dict(nodp=True)),
                     ("no Senate money edge", dict(nomoney=True))):
        rr = R.copy()
        if kw.get("partisan"):
            rr["poll_avg"] = rr["poll_all"]
        if kw.get("nomoney"):
            rr.loc["us_senate", "dev_hat"] -= f_dev.params["money"] * rr.loc["us_senate", "money"]
        o = forecast(rr, C.copy(), dp.iloc[0:0] if kw.get("nodp") else dp, kw.get("tm", Tmu), kw.get("tsd", Tsd), c_sd, h_sd, e_sd, sds, kw.get("bm", POLL_B_MEAN), save=False, quiet=True)
        m = o["R"]["mean"]
        print(f"{name:<34}{m['governor']:>7.1f}{m['us_senate']:>7.1f}{m['lt_governor']:>7.1f}{m['attorney_general']:>7.1f}{m['comptroller']:>7.1f}{m['railroad_commissioner']:>7.1f}"
              f"{o['seats']:>7.1f}{o['R']['p_d']['us_senate']:>7.2f}{o['R']['p_d']['governor']:>7.2f}")


def forecast(R, C, dp, Tmu, Tsd, c_sd, h_sd, e_sd, sds, bmean, save=False, quiet=False):
    """Joint Gaussian over [T, b, c, h, dev_r..., e_d...], conditioned on all polls.
    District residuals are drawn with t(5) tails: a single seat can blow up in ways
    a normal understates, and this is where a 1% would otherwise be false precision."""
    nr, nd = len(R), len(C)
    N = 4 + nr + nd
    mu = np.zeros(N); cov = np.zeros((N, N))
    mu[0], cov[0, 0] = Tmu, Tsd ** 2
    mu[1], cov[1, 1] = bmean, POLL_B_SD ** 2
    mu[2], cov[2, 2] = 0.0, c_sd ** 2
    mu[3], cov[3, 3] = 0.0, h_sd ** 2
    for i, r in enumerate(R.itertuples()):
        mu[4 + i] = r.dev_hat
        cov[4 + i, 4 + i] = sds[r.tier] ** 2
    for j, dist in enumerate(C.index):
        mu[4 + nr + j] = C.loc[dist, "rel_hat"]
        cov[4 + nr + j, 4 + nr + j] = e_sd ** 2
    H, y, Rv = [], [], []
    for i, r in enumerate(R.itertuples()):
        if r.n_polls == 0:
            continue
        v = np.zeros(N); v[0] = 1; v[1] = 1; v[4 + i] = 1
        H.append(v); y.append(r.poll_avg); Rv.append(POLL_RACE_SD ** 2 + 16 / r.n_polls)
    for j, dist in enumerate(C.index):
        g = dp[dp.dist == dist]
        if g.empty:
            continue
        g = g.sort_values("end").drop_duplicates("pollster", keep="last")
        marg = (g.d_pct - g.r_pct) / (g.d_pct + g.r_pct) * 100
        # internals run about 4 points toward their sponsor (a convention, not estimated
        # here: Texas district polls have no usable back file)
        adj = g.partisan_flag.map({"D": -4.0, "R": 4.0}).fillna(0.0)
        for mm, a in zip(marg, adj):
            v = np.zeros(N); v[0] = 1; v[1] = 1; v[2] = 1; v[3] = C.loc[dist, "ss"]; v[4 + nr + j] = 1
            H.append(v); y.append(mm + a); Rv.append(6.0 ** 2)
    H = np.array(H); y = np.array(y); Rm = np.diag(Rv)
    post, pcov = condition(mu, cov, H, y, Rm)
    rng = np.random.default_rng(2026)
    L = np.linalg.cholesky(pcov + 1e-9 * np.eye(N))
    draws = 200000
    z = post + (L @ rng.standard_normal((N, draws))).T
    idx = 4 + nr
    tmix = np.sqrt(3.0 / 5.0) * np.sqrt(5.0 / rng.chisquare(5, (draws, nd)))
    z[:, idx:] = post[idx:] + (z[:, idx:] - post[idx:]) * tmix
    sw_y = z[:, [0]] + z[:, 4:idx]
    cd_y = z[:, [0]] + z[:, [2]] + z[:, [3]] * C.ss.values[None, :] + z[:, idx:]
    R = R.copy(); C = C.copy()
    R["mean"] = sw_y.mean(0); R["sd"] = sw_y.std(0)
    R["lo10"], R["hi90"] = np.percentile(sw_y, 10, axis=0), np.percentile(sw_y, 90, axis=0)
    R["p_d"] = (sw_y > 0).mean(0)
    C["mean"] = cd_y.mean(0); C["sd"] = cd_y.std(0); C["p_d"] = (cd_y > 0).mean(0)
    C["lo10"], C["hi90"] = np.percentile(cd_y, 10, axis=0), np.percentile(cd_y, 90, axis=0)
    C["prior_mean"] = Tmu + C.rel_hat
    C["n_polls"] = [len(dp[dp.dist == d]) for d in C.index]
    seats = (cd_y > 0).sum(1)
    if not quiet:
        print(f"\nT prior {Tmu:.1f} ± {Tsd:.1f} -> posterior {post[0]:.1f} ± {np.sqrt(pcov[0, 0]):.1f}; poll miss b {post[1]:+.1f} ± {np.sqrt(pcov[1, 1]):.1f}; "
              f"congressional offset c {post[2]:+.1f} ± {np.sqrt(pcov[2, 2]):.1f}; Hispanic slope h {post[3]:+.2f} ± {np.sqrt(pcov[3, 3]):.2f} per 10pt SSVR")
        print("\n=== 2026 statewide forecast (D margin) ===")
        print(R[["r", "d", "poll_avg", "n_polls", "dev_hat", "mean", "sd", "lo10", "hi90", "p_d"]].round(2).to_string())
        print("\n=== Congress (D margin) ===")
        print(C[["r_name", "d_name", "pres24", "prior_mean", "mean", "sd", "lo10", "hi90", "p_d", "n_polls"]].round(2).to_string())
        print(f"\nD seats: mean {seats.mean():.1f}, median {np.median(seats):.0f}, 10-90 {np.percentile(seats, 10):.0f}-{np.percentile(seats, 90):.0f}")
        print("seat distribution:", pd.Series(seats).value_counts(normalize=True).sort_index().round(3).to_dict())
        print(f"P(D 10 or fewer) {(seats <= 10).mean():.3f}; P(D 12+) {(seats >= 12).mean():.3f}; P(D 13+, i.e. no net R gain on the 13 held at the 2024 election) {(seats >= 13).mean():.3f}")
        print(f"P(any D statewide win) {(sw_y > 0).any(1).mean():.3f}; P(Senate + Gov) {((sw_y[:, 1] > 0) & (sw_y[:, 0] > 0)).mean():.3f}; "
              f"P(all 8) {(sw_y > 0).all(1).mean():.3f}")
        print("corr(T-driven) Senate vs TX-35 win:", round(np.corrcoef(sw_y[:, 1] > 0, cd_y[:, list(C.index).index(35)] > 0)[0, 1], 2))
    if save:
        OUTDIR.mkdir(exist_ok=True)
        R.to_csv(OUTDIR / "statewide_forecast_2026.csv")
        C.to_csv(OUTDIR / "congress_forecast_2026.csv")
    return {"R": R, "C": C, "seats": float(seats.mean()), "sw_y": sw_y, "cd_y": cd_y, "T": z[:, 0], "c": z[:, 2], "h": z[:, 3]}


if __name__ == "__main__":
    main()
