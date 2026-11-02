"""
scandal_tx23_tests.py

Two questions from 2026-10-30.

1. Does a controversy cost a statewide candidate votes the first general after it
   breaks, and does that fade afterward (Brennan: Paxton's are baked into his record)?
   Cases are coded by hand below; residual = statewide race minus judicial composite
   minus tier and incumbency means, signed so negative is bad for the flagged candidate.

2. What separates our TX-23 from Split Ticket's? Pull each input and re-run the joint
   forecast with it changed.

Usage: python scripts/scandal_tx23_tests.py
"""
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import statewide_pollster_history  # noqa: E402,F401  (reconfigures stdout at import)
import statewide_model as sm_  # noqa: E402

pd.set_option("display.width", 200); pd.set_option("display.max_rows", 80)

# (year, race, party flagged, first general after it broke?, what)
CASES = [
    (2014, "attorney_general", "R", True, "Paxton: securities violation admitted, fined May 2014"),
    (2018, "attorney_general", "R", True, "Paxton: indicted 2015, trial pending"),
    (2022, "attorney_general", "R", True, "Paxton: aides' 2020 bribery allegation, FBI probe"),
    (2018, "ag_commissioner", "R", True, "Miller: taxpayer-funded 'Jesus shot' trip, offensive posts"),
    (2022, "ag_commissioner", "R", False, "Miller: same record, repeat run"),
    (2018, "governor", "D", True, "Valdez: unpaid property taxes, parking fines"),
    (2024, "us_senate", "R", True, "Cruz: Cancun during 2021 storm, first general since"),
]


def scandal_test(b):
    s = b["s"].copy()
    f = b["f_dev"]
    # residual WITHOUT the candidate's prior record, so the first test sees the raw effect
    x = sm_.fit_dev(s, ["inc", "money"])
    d = s.dropna(subset=["inc", "money"]).copy()
    d["res"] = x.resid
    rows = []
    for year, race, party, first, what in CASES:
        r = d[(d.year == year) & (d.race == race)]
        if r.empty:
            print("no row", year, race)
            continue
        sign = 1 if party == "D" else -1
        rows.append({"year": year, "race": race, "first": first, "own_res": sign * r.res.iloc[0],
                     "own_res_prior": sign * (r.res.iloc[0] - f.params["prior_own"] * r.prior_own.iloc[0]),
                     "what": what})
    t = pd.DataFrame(rows)
    print("\n=== Flagged candidates: own residual (negative = ran behind expectation) ===")
    print(t.round(1).to_string(index=False))
    a = t[t["first"]]
    print(f"\nfirst general after it broke: mean {a.own_res.mean():+.2f}, se {a.own_res.std() / np.sqrt(len(a)):.2f}, n {len(a)}")
    rng = np.random.default_rng(5)
    allr = np.concatenate([d.res.values, -d.res.values])
    sims = [rng.choice(allr, len(a), replace=False).mean() for _ in range(20000)]
    print(f"permutation p (one-sided, as low as this by chance): {(np.array(sims) <= a.own_res.mean()).mean():.3f}")
    r_only = a[a.race.isin(["attorney_general", "ag_commissioner"])]
    print(f"down-ballot R only (closest to French): mean {r_only.own_res.mean():+.2f}, n {len(r_only)}")
    p = t[t.race == "attorney_general"].sort_values("year")
    print("Paxton by run:", dict(zip(p.year, p.own_res.round(1))))

    # Stress tests: the six cases are four people, picked by hand with the outcomes
    # roughly known, and a race residual belongs to both candidates.
    who = {"attorney_general": "Paxton", "ag_commissioner": "Miller", "governor": "Valdez", "us_senate": "Cruz"}
    person = a.assign(p=a.race.map(who)).groupby("p").own_res.mean()
    print(f"by person: {person.round(1).to_dict()}, mean {person.mean():+.2f}")
    cuellar_case(b, person.to_dict())
    sims = [rng.choice(allr, len(person), replace=False).mean() for _ in range(20000)]
    print(f"permutation p with one draw per person: {(np.array(sims) <= person.mean()).mean():.3f}")
    print(f"without Cruz (a top-ticket race against heavy money): {person.drop('Cruz').mean():+.2f}")
    flagged = {(y, r) for y, r, *_ in CASES}
    rest = d[[(y, r) not in flagged for y, r in zip(d.year, d.race)]]
    print("unflagged R candidates running 2.4+ behind expectation (own residual <= -2.4), as a placebo:")
    bad = rest[rest.res >= 2.4].sort_values("res", ascending=False)
    print("  " + "; ".join(f"{r.r_name} {r.year} {r.race} {-r.res:+.1f}" for r in bad.head(12).itertuples()))
    print(f"  {len(bad)} of {len(rest)} unflagged races ({len(bad) / len(rest):.0%}); flagged R cases at that level: "
          f"{(t[t.what.str.contains('Paxton|Miller|Cruz') & t['first']].own_res <= -2.4).mean():.0%}")
    return float(a.own_res.mean())


def cuellar_case(b, statewide_mean_by_person):
    """Cuellar as a backtest case only (Brennan): FBI raid Jan 2022, indicted May 2024.
    Measured on the U.S. House layer, without prior WAR so his own record does not
    absorb the effect, net of each cycle's mean. Signed so negative is bad for him."""
    h = sm_.house_table()
    terms = [t for t in sm_.CD_TERMS if t != "prior_war"]
    f = sm.OLS(h.rel, sm.add_constant(h[terms])).fit()
    h = h.assign(res=f.resid)
    h["res"] = h.res - h.groupby("year").res.transform("mean")
    print(f"\n=== Cuellar in the U.S. House layer (n={len(h)}, residual sd {h.res.std():.1f}) ===")
    g = h[(h.dist == 28) & (h.d_name == "Cuellar")].sort_values("year")
    print(g[["year", "d_margin", "res"]].round(1).to_string(index=False))
    before = g[g.year < 2022].res
    after = g[g.year >= 2022].res
    print(f"before the raid {before.mean():+.1f} (n {len(before)}), 2022 and 2024 {after.mean():+.1f}, change {after.mean() - before.mean():+.1f}")
    rio = h[(h.year >= 2022) & (h.ss > 2.5) & (h.d_inc) & (h.dist != 28)]
    print("other Hispanic-seat D incumbents, 2022-24 own residual:", rio[["year", "dist", "d_name", "res"]].round(1).values.tolist())
    rng = np.random.default_rng(9)
    pooled = dict(statewide_mean_by_person, Cuellar=float(after.mean()))
    vals = np.array(list(pooled.values()))
    d = b["s"].dropna(subset=["inc", "money"])
    x = sm_.fit_dev(b["s"], ["inc", "money"]).resid.values
    allr = np.concatenate([x, -x, h.res.values, -h.res.values])
    sims = [rng.choice(allr, len(vals), replace=False).mean() for _ in range(20000)]
    print(f"with Cuellar: {pd.Series(pooled).round(1).to_dict()}, mean {vals.mean():+.2f}, permutation p {(np.array(sims) <= vals.mean()).mean():.3f}")
    print(f"without Cruz: {np.mean([v for k, v in pooled.items() if k != 'Cruz']):+.2f}")


def run(b, R=None, C=None, dp=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return sm_.forecast(b["R"] if R is None else R, b["C"] if C is None else C, b["dp"] if dp is None else dp,
                            b["Tmu"], b["Tsd"], b["c_sd"], b["h_sd"], b["e_sd"], b["sds"], sm_.POLL_B_MEAN, quiet=True)


def tx23(b):
    C = b["C"]
    x = C.loc[23]
    f = b["f_cd"].params
    print("\n=== TX-23 inputs ===")
    print(f"{x.r_name} v. {x.d_name}: Trump 2024 {x.pres24:+.1f}, lean_p {x.lean_p:+.1f}, lean_j {x.lean_j:+.1f}, SSVR centered {x.ss:+.2f}, "
          f"raised D ${x.d_raised:,.0f} R ${x.r_raised:,.0f}, polls in window {len(b['dp'][b['dp'].dist == 23])}")
    print(f"pieces of the prior: T {b['Tmu']:+.1f} + const {f['const']:+.1f} + lean {f['lean_p'] * x.lean_p + f['lean_j'] * x.lean_j:+.1f} "
          f"+ money {f['money'] * x.money:+.1f} + prior WAR {f['prior_war'] * x.prior_war:+.1f} = {b['Tmu'] + x.rel_hat:+.1f}")
    base = run(b)["C"]
    print(f"base posterior {base.loc[23, 'mean']:+.1f}, sd {base.loc[23, 'sd']:.1f}, P(D) {base.loc[23, 'p_d']:.2f}")
    rows = []
    dp2 = b["dp"].copy()
    dp2.loc[dp2.dist == 23, "partisan_flag"] = ""
    rows.append(("D internal taken at face value (no -4)", run(b, dp=dp2)["C"]))
    rows.append(("no poll at all", run(b, dp=b["dp"][b["dp"].dist != 23])["C"]))
    rem = pd.DataFrame([{"dist": 23, "pollster": "Remington Research", "field_end": "2026-07-28", "sample_size": "572",
                         "population": "LV", "r_pct": 47.0, "d_pct": 41.0, "partisan_flag": "R", "end": pd.Timestamp("2026-07-28")}])
    rows.append(("add Remington R+6 at the -4 internal haircut", run(b, dp=pd.concat([b["dp"], rem], ignore_index=True))["C"]))
    rem2 = rem.assign(partisan_flag="")
    rows.append(("add Remington R+6 at face value", run(b, dp=pd.concat([b["dp"], rem2], ignore_index=True))["C"]))
    c2 = C.copy()
    c2.loc[23, "rel_hat"] -= f["money"] * x.money
    rows.append(("no R money edge", run(b, C=c2)["C"]))
    c3 = C.copy()
    c3.loc[23, "rel_hat"] += 1.0
    rows.append(("firebrand stand-in, +1 margin (legislative value)", run(b, C=c3)["C"]))
    c4 = C.copy()
    c4.loc[23, "rel_hat"] += 3.0
    rows.append(("Herrera -3 candidate penalty", run(b, C=c4)["C"]))
    c5 = c4.copy()
    c5.loc[23, "rel_hat"] -= f["money"] * x.money
    rows.append(("-3 penalty, no money edge, D poll at face value", run(b, C=c5, dp=dp2)["C"]))
    print(f"{'change':<52}{'mean':>7}{'P(D)':>7}")
    for name, o in rows:
        print(f"{name:<52}{o.loc[23, 'mean']:>7.1f}{o.loc[23, 'p_d']:>7.2f}")

    print("\n=== Same seats, Split Ticket minus ours (D margin), by type ===")
    st = pd.read_csv(sys.argv[1] + "/Cy0fK.csv") if len(sys.argv) > 1 else None
    if st is not None:
        st = st[st.cd.astype(str).str.startswith("TX")].copy()
        st["dist"] = st.cd.str.extract(r"(\d+)").astype(int)
        st["m"] = st.prediction.map(lambda v: 0.0 if str(v).startswith("Even") else float(str(v)[2:]) * (1 if str(v)[0] == "D" else -1))
        j = st.set_index("dist").join(base[["mean", "p_d", "sd"]]).join(C[["ss", "inc_d", "inc_r", "pres24"]], rsuffix="_c")
        j["gap"] = j.m - j["mean"]
        j["open"] = (j.inc_d + j.inc_r) == 0
        cmp = j[j["mean"].abs() < 20]
        print(cmp.groupby(["open"]).gap.agg(["mean", "count"]).round(2).to_string())
        print(f"gap vs SSVR slope, seats within 20: {np.polyfit(cmp.ss, cmp.gap, 1)[0]:+.2f} per 10pt SSVR; corr {cmp.ss.corr(cmp.gap):+.2f}")
        print(cmp.sort_values("ss", ascending=False)[["pres24", "ss", "open", "mean", "m", "gap"]].head(8).round(1).to_string())
        print(f"our sd in polled Hispanic seats ~{base.loc[[15, 23, 34, 35], 'sd'].mean():.1f}; TX-23 needed for their 52% at D+0.2 is about a coin flip regardless of sd")


def french(b, k):
    R = b["R"]
    base = run(b)["R"]
    print("\n=== French term on the Railroad Commissioner race ===")
    print(f"poll avg {R.loc['railroad_commissioner', 'poll_avg']:+.1f} vs other down-ballot polls: "
          f"AG {R.loc['attorney_general', 'poll_avg']:+.1f}, Lt Gov {R.loc['lt_governor', 'poll_avg']:+.1f}, Comptroller {R.loc['comptroller', 'poll_avg']:+.1f}")
    print(f"base {base.loc['railroad_commissioner', 'mean']:+.2f}, P(D) {base.loc['railroad_commissioner', 'p_d']:.3f}")
    for pts in (1.0, 2.0, 3.0, abs(k)):
        r2 = R.copy()
        r2.loc["railroad_commissioner", "dev_hat"] += pts
        o = run(b, R=r2)
        print(f"  French -{pts:.1f} in the prior: RRC {o['R'].loc['railroad_commissioner', 'mean']:+.2f}, P(D) {o['R'].loc['railroad_commissioner', 'p_d']:.3f}; "
              f"Senate {o['R'].loc['us_senate', 'mean']:+.2f}, seats {o['seats']:.2f}")


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        b = sm_.build()
    k = scandal_test(b)
    french(b, k)
    tx23(b)
