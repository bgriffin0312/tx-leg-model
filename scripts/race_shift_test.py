"""
race_shift_test.py

Should race-specific polling move district means? Re-test of the shelved shift
term (docs/poll-integration-proposal.md §5-6) on the ADOPTED spec: open-race
baseline, clean-cycle refit, SHIP terms (scripts/refit_clean_cycles.py), with
leave-one-cycle-out scoring. The Sept-2026 version (shift_term_backtest_sept1.py)
scored against the pre-refit presidential-baseline coefficients.

Shift for district d in cycle y:
    s_d = sum_g (share_dg - w_TX_g) * delta_g
    delta_g = race g's D 2p in the polls - race g's D 2p in the baseline year
The open baseline is built from the prior presidential year's statewide races,
so the prior is that year's Catalist two-way presidential vote by race.
Centering on Texas weights leaves the uniform swing to the environment dial.

Variants:
  N        SHIP, no shift (the adopted model)
  SN x k   national poll delta (Sept-1 averages) imposed at weight k, no fitting
  SU x 1   same-pollster Texas delta (UT now - UT prior Oct), imposed
  SN fit   national shift as a fitted regressor
  ORACLE   2022 only: Catalist's ACTUAL 2022 vote by race minus 2020 - the best
           a perfect national race read could have done
2014 has no Catalist 2012 / 2014 race-poll inputs on file, so s = 0 there and
it is reported only as part of the training rows.

Then applies the same construction to 2026 with the current LV aggregate
(model_config.RACE_GENERIC_BALLOT_D_SHARE) minus Catalist 2024, through the real
Monte Carlo in src/model.py.

Usage:  python scripts/race_shift_test.py
"""
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import refit_clean_cycles as rc  # noqa: E402
import group_error_estimate as ge  # noqa: E402
import shift_term_backtest_sept1 as s1  # noqa: E402
from backtest_config import BACKTEST_CONFIGS  # noqa: E402

G = ["white", "black", "hispanic", "other"]
K = {"white": "white_nh", "black": "black_nh", "hispanic": "hispanic", "other": "other"}
CATALIST_PRES = {  # two-way D, national, What Happened 2024 (as in shift_term_backtest.py)
    2016: {"white": 0.41, "black": 0.93, "hispanic": 0.70, "other": 0.70},
    2020: {"white": 0.44, "black": 0.89, "hispanic": 0.63, "other": 0.65},
    2024: {"white": 0.42, "black": 0.85, "hispanic": 0.54, "other": 0.61},
}
CATALIST_2022 = {"white": 0.42, "black": 0.88, "hispanic": 0.62, "other": 0.59}  # backtest_config 2022


def deltas():
    out = {}
    for yr in (2018, 2022):
        src = s1.SOURCES[yr]
        fb = BACKTEST_CONFIGS[yr]["race_generic_ballot_d_share"]
        now = s1.avg_polls(src["national_polls"], fb)
        prior = CATALIST_PRES[src["prior_pres"]]
        nat = {g: now[K[g]] - prior[g] for g in G}
        ut = {g: (src["tx_now"][K[g]] - src["tx_prior"][K[g]])
              if g != "other" and src["tx_now"].get(K[g]) is not None and src["tx_prior"].get(K[g]) is not None
              else 0.0 for g in G}
        out[yr] = {"nat": nat, "ut": ut}
    out[2022]["oracle"] = {g: CATALIST_2022[g] - CATALIST_PRES[2020][g] for g in G}
    return out


def shift(d, w, dl):
    return sum((d[g] - w[g]) * dl.get(g, 0.0) for g in G)


def score(rows, resid):
    hisp = rows["hispanic"].values
    slope = np.polyfit(hisp, resid, 1)[0] if len(rows) > 5 else np.nan
    return f"rmse {np.sqrt((resid ** 2).mean()) * 100:5.2f}  mean {resid.mean() * 100:+5.2f}  resid~hisp {slope * 100:+6.1f}pp/unit"


def backtest():
    w = ge.texas_weights()
    d = ge.attach_shares(rc.load(rc.bb.PROC / "phase1_dataset.csv", "open"))
    d = d.dropna(subset=G + ["challenger_viability_flag"]).copy()
    ship = rc.BASE + rc.INC + ["challenger_viability_flag", "ie_c"]
    dl = deltas()
    for name in ("nat", "ut", "oracle"):
        d[f"s_{name}"] = 0.0
        for yr in (2018, 2022):
            if name in dl[yr]:
                m = d.year == yr
                d.loc[m, f"s_{name}"] = shift(d[m], w, dl[yr][name])

    print("Deltas vs prior presidential (pp):")
    for yr in (2018, 2022):
        for name, v in dl[yr].items():
            print(f"  {yr} {name:6s} " + "  ".join(f"{g} {v[g] * 100:+5.1f}" for g in G))
    print(f"\nRows: {len(d)} contested clean-vintage races ({d.groupby('year').size().to_dict()}), SHIP spec, LOO by cycle")

    _, base = rc.loo(d, ship)            # resid = pred - actual, indexed like d
    _, fit = rc.loo(d, ship + ["s_nat"])
    variants = {"N  (adopted)": base,
                "SN x 0.5": base + 0.5 * d["s_nat"],
                "SN x 1": base + d["s_nat"],
                "SU x 1": base + d["s_ut"],
                "SN fitted": fit,
                "ORACLE x 1 (2022)": base + d["s_oracle"]}
    for yr in (2018, 2022):
        rows = d[d.year == yr]
        print(f"\n  {yr} held out (n={len(rows)})   shift sd: nat {rows.s_nat.std() * 100:.2f}pp, ut {rows.s_ut.std() * 100:.2f}pp")
        for label, r in variants.items():
            if label.startswith("ORACLE") and yr != 2022:
                continue
            print(f"    {label:20s} {score(rows, r.loc[rows.index].values)}")
    both = d[d.year.isin([2018, 2022])]
    print("\n  2018+2022 pooled:")
    for label, r in variants.items():
        if not label.startswith("ORACLE"):
            print(f"    {label:20s} {score(both, r.loc[both.index].values)}")


def forward():
    import model
    import model_config as mc
    with contextlib.redirect_stdout(io.StringIO()):
        df = model.load_districts()
    now = mc.RACE_GENERIC_BALLOT_D_SHARE
    prior = CATALIST_PRES[2024]
    dl = {g: now[K[g]] - prior[g] for g in G}
    dl_shr = dict(dl, other=0.0)  # "other" is the topline residual: noise, shrunk to 0 as in §5
    centered = model.centered_group_shares(df, education=False)  # cols: white, black, hispanic, other
    s = centered @ np.array([dl_shr[g] for g in G])
    hisp = centered[:, 2] + 0  # centered share; recover raw below for display
    print("\n2026 forward: LV aggregate minus Catalist 2024 (pp): "
          + "  ".join(f"{g} {dl[g] * 100:+.1f}" for g in G) + "   (other shrunk to 0)")
    print(f"  shift across districts: mean {s.mean() * 100:+.2f}pp, sd {s.std() * 100:.2f}, "
          f"min {s.min() * 100:+.2f}, max {s.max() * 100:+.2f}")

    # Same-pollster Texas delta: latest UT LV Texas Legislature generic minus UT's Oct 2024 LV.
    t = pd.read_csv(ROOT / "data" / "raw" / "texas_crosstab_inputs.csv")
    t = t[t.pollster.str.startswith("UT/") & (t.ballot == "tx_legislature_generic") & (t.population == "LV")]
    t = t.sort_values("poll_date")
    prior_ut, now_ut = t[t.poll_date.str.startswith("2024-10")].iloc[-1], t.iloc[-1]
    dl_ut = {g: (now_ut[f"{g}_d_2p"] - prior_ut[f"{g}_d_2p"]) if g != "other" else 0.0 for g in G}
    s_ut = centered @ np.array([dl_ut[g] for g in G])
    print(f"  UT same-pollster delta ({now_ut.poll_date} vs {prior_ut.poll_date}, pp): "
          + "  ".join(f"{g} {dl_ut[g] * 100:+.1f}" for g in G[:3])
          + f"   shift sd {s_ut.std() * 100:.2f}, min {s_ut.min() * 100:+.2f}, max {s_ut.max() * 100:+.2f}")

    orig = model.build_linear_predictions
    res = {}
    for label, k, sv in (("N  (adopted)", 0.0, s), ("SN x 0.5", 0.5, s), ("SN x 1", 1.0, s), ("SU x 1", 1.0, s_ut)):
        model.build_linear_predictions = (lambda f, e, r, k=k, sv=sv: orig(f, e, r) + k * sv)
        model.RNG = np.random.default_rng(20261027)
        with contextlib.redirect_stdout(io.StringIO()):
            res[label] = model.run_monte_carlo(df, model.CURRENT_ENV, now)
    model.build_linear_predictions = orig
    print(f"  at D+{model.CURRENT_ENV}:")
    for label, r in res.items():
        print(f"    {label:14s} House {r['expected_house_seats']:5.1f}  P(maj) {r['house_control_prob'] * 100:4.1f}%   "
              f"Senate {r['expected_senate_seats']:4.1f}")
    out = df[["chamber_lower", "district", "incumbent", "incumbent_party"]].copy()
    out["shift_pp"] = s * 100
    for label, r in res.items():
        out[label] = r["district_win_probs"].values * 100
    out["ut_shift_pp"] = s_ut * 100
    out["chg_x1"] = out["SN x 1"] - out["N  (adopted)"]
    out["chg_ut"] = out["SU x 1"] - out["N  (adopted)"]
    live = out[out[["N  (adopted)", "SN x 1", "SU x 1"]].apply(lambda c: c.between(3, 97)).any(axis=1)]
    print("\n  Seats that move >= 3pp under either variant (D win %):")
    print(live[(live.chg_x1.abs() >= 3) | (live.chg_ut.abs() >= 3)].sort_values("chg_ut")
          [["chamber_lower", "district", "incumbent", "incumbent_party", "shift_pp", "ut_shift_pp",
            "N  (adopted)", "SN x 1", "SU x 1"]].round(1).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    backtest()
    forward()
