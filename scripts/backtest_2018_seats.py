"""
backtest_2018_seats.py

How would the ADOPTED model (open-race baseline, SHIP terms, sigma 0.044) have
called the 2018 Texas House, fit without 2018? Seat counts and the seats it
missed, under the actual national environment (D+8.6) and the Sept-1 polling
environment (D+7.5, shift_term_backtest_sept1.SOURCES).

Contested rows the model can score come from refit_clean_cycles.load(open);
every other House seat (uncontested, or missing a baseline/finance field) is
counted at its actual result and reported separately.

Usage:  python scripts/backtest_2018_seats.py [--year 2018]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402
import midterm_education_test as met  # noqa: E402  (sets --education for group_error_estimate)

SIGMA = 0.044  # model_config total sigma (adopted 2026-10-07)
SEPT1_ENV = {2018: 7.5, 2022: 0.5}
SHIP = rc.BASE + rc.INC + ["challenger_viability_flag", "ie_c", "c_white_col"]  # + education, as adopted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    yr = ap.parse_known_args()[0].year
    pd.set_option("display.width", 200)

    d = met.prep("open").dropna(subset=["challenger_viability_flag"])
    tr, te = d[d.year != yr], d[d.year == yr].copy()
    c, _, _ = rc.ols(tr, SHIP)
    print(f"Fit on {sorted(tr.year.unique())} (n={len(tr)}); env coef {c.get('national_env', np.nan):.4f}/pt, edu {c.get('c_white_col', np.nan):+.3f}, "
          f"baseline pass-through {c.get('x', np.nan):.3f}")

    allh = rc.bb.load()
    allh = allh[(allh.year == yr) & (allh.chamber_lower == "house")].copy()
    allh["dem_2p_share"] = pd.to_numeric(allh.dem_2p_share, errors="coerce")
    actual_total = int((allh.dem_2p_share > 0.5).sum())
    teh = te[te.chamber_lower == "house"]
    rest = allh[~allh.district.astype(int).isin(teh.district.astype(int))]
    rest_d = int((rest.dem_2p_share > 0.5).sum())
    print(f"\n{yr} Texas House: Democrats actually won {actual_total} of {len(allh)}.")
    print(f"  Scored by the model: {len(teh)} contested seats (actual D wins {int((teh.dem_2p_share > 0.5).sum())}); "
          f"other {len(rest)} seats counted at actual result (D {rest_d}).")

    real_env = float(te.national_env.iloc[0])
    for label, env in ((f"actual environment D+{real_env}", real_env), (f"Sept-1 polls D+{SEPT1_ENV.get(yr)}", SEPT1_ENV.get(yr))):
        x = teh.copy()
        x["national_env"] = env
        p = rc.predict(x, c)
        wp = norm.cdf(p, 0.5, SIGMA)
        sims = np.random.default_rng(1).normal(0, 0.0335, 10000)  # shared national error, model_config split
        idio = np.sqrt(SIGMA ** 2 - 0.0335 ** 2)
        seat_draws = (norm.cdf((p[None, :] + sims[:, None] - 0.5) / idio)).sum(axis=1) + rest_d
        print(f"\n  {label}: expected D seats {wp.sum() + rest_d:.1f} (actual {actual_total}); "
              f"10th-90th pct {np.percentile(seat_draws, 10):.0f}-{np.percentile(seat_draws, 90):.0f}; "
              f"mean vote miss {np.mean(p - x.dem_2p_share.values) * 100:+.2f}pp (positive = too Democratic)")
        if env == real_env:
            out = x[["district", "dem_incumbent", "rep_incumbent", "dem_2p_share"]].copy()
            out["pred"], out["p_win_d"] = p, wp
            out["miss_pp"] = (p - out.dem_2p_share) * 100
            flips = out[(out.rep_incumbent == 1) & (out.dem_2p_share > 0.5)]
            print(f"\n  R-incumbent seats Democrats won ({len(flips)}), model P(D) under the actual environment:")
            print((flips.assign(actual=flips.dem_2p_share * 100, pred=flips.pred * 100, p_win_d=flips.p_win_d * 100)
                   [["district", "actual", "pred", "p_win_d", "miss_pp"]].sort_values("p_win_d").round(1).to_string(index=False)))
            close = out[(out.dem_2p_share.between(0.45, 0.55)) & ~out.index.isin(flips.index)]
            print(f"\n  Other seats decided within 5 points ({len(close)}):")
            print((close.assign(actual=close.dem_2p_share * 100, pred=close.pred * 100, p_win_d=close.p_win_d * 100)
                   [["district", "dem_incumbent", "rep_incumbent", "actual", "pred", "p_win_d"]].sort_values("actual").round(1).to_string(index=False)))


if __name__ == "__main__":
    main()
