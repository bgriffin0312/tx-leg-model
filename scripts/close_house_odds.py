"""
close_house_odds.py

Probability of a narrowly divided Texas House (D seats in a band, e.g. 74-76 or
73-77 of 150) from the model's simulated seat distribution, at the current
environment and under the UT same-pollster race-shift sensitivity
(scripts/race_shift_test.py).

Usage:  python scripts/close_house_odds.py [--sims 50000]
"""
import argparse
import io
import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model  # noqa: E402
import model_config as mc  # noqa: E402

BANDS = [(74, 76), (73, 77), (72, 78), (76, 150)]


def ut_shift(df):
    t = pd.read_csv(ROOT / "data" / "raw" / "texas_crosstab_inputs.csv")
    t = t[t.pollster.str.startswith("UT/") & (t.ballot == "tx_legislature_generic") & (t.population == "LV")]
    t = t.sort_values("poll_date")
    prior, now = t[t.poll_date.str.startswith("2024-10")].iloc[-1], t.iloc[-1]
    dl = np.array([now[f"{g}_d_2p"] - prior[f"{g}_d_2p"] for g in ("white", "black", "hispanic")] + [0.0])
    return model.centered_group_shares(df, education=False) @ dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    n = ap.parse_known_args()[0].sims
    with contextlib.redirect_stdout(io.StringIO()):
        df = model.load_districts()
    orig = model.build_linear_predictions
    s_ut = ut_shift(df)
    runs = [("Current model", model.CURRENT_ENV, 0.0), ("UT race-shift sensitivity", model.CURRENT_ENV, 1.0),
            ("Texas runs 4 pts bluer (D+13.5)", 13.5, 0.0)]
    for label, env, k in runs:
        model.build_linear_predictions = (lambda f, e, r, k=k: orig(f, e, r) + k * s_ut)
        model.RNG = np.random.default_rng(20261027)
        with contextlib.redirect_stdout(io.StringIO()):
            r = model.run_monte_carlo(df, env, mc.RACE_GENERIC_BALLOT_D_SHARE, n_sims=n)
        seats = np.asarray(r["house_seat_dist"])
        parts = [f"D {lo}-{hi}: {np.mean((seats >= lo) & (seats <= hi)) * 100:4.1f}%" if hi < 150
                 else f"D {lo}+ (majority): {np.mean(seats >= lo) * 100:4.1f}%" for lo, hi in BANDS]
        print(f"{label:34s} mean {seats.mean():5.1f}  | " + "  ".join(parts))
    model.build_linear_predictions = orig


if __name__ == "__main__":
    main()
