"""
compare_2018_2026_terrain.py

Why does a ~D+9 environment flip ~12 Texas House seats in the 2018 backtest but
only ~4 now? Compare the terrain: Republican-held seats by baseline D share in
2018 (2011 lines, open-race baseline from 2016 statewide races) vs 2026 (2021
lines, open-race baseline from 2024), and the model's linear prediction at the
same environment.

Usage:  python scripts/compare_2018_2026_terrain.py
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

BANDS = [(0, .35), (.35, .40), (.40, .42), (.42, .44), (.44, .46), (.46, .48), (.48, .50), (.50, 1.01)]


def bands(x):
    return [int(((x >= lo) & (x < hi)).sum()) for lo, hi in BANDS]


def main():
    old = rc.bb.load()
    old = old[(old.year == 2018) & (old.chamber_lower == "house")].copy()
    old["base"] = pd.to_numeric(old.open_mean_d2p, errors="coerce")
    old["held_r"] = old.rep_incumbent.astype(float).eq(1) | (old.open_seat.astype(str).str.lower().isin(["1", "true"])
                                                           & (old.base < 0.5))
    # Party holding the seat going into 2018: R incumbent, or open seat (approximate by baseline for opens)
    import model
    with contextlib.redirect_stdout(io.StringIO()):
        new = model.load_districts()
    new = new[new.chamber_lower == "house"].copy()
    new["base"] = pd.to_numeric(new.dem_pres_2p_baseline, errors="coerce")  # open-race composite after _apply_baseline_source
    new["held_r"] = new.incumbent_party.eq("R")

    print("Statewide (unweighted mean of 150 House districts) baseline D share:")
    print(f"  2018 cycle (2016 races, 2011 lines): {old.base.mean() * 100:.1f}%   median {old.base.median() * 100:.1f}%")
    print(f"  2026 cycle (2024 races, 2021 lines): {new.base.mean() * 100:.1f}%   median {new.base.median() * 100:.1f}%")
    hdr = "  ".join(f"{int(lo * 100)}-{int(min(hi, 1) * 100)}" for lo, hi in BANDS)
    print(f"\nR-held House seats by baseline D share (%):\n  {'':10s}{hdr}")
    for lab, df in (("2018", old), ("2026", new)):
        r = df[df.held_r]
        print(f"  {lab} (n={len(r):2d}) " + "  ".join(f"{v:5d}" for v in bands(r.base)))
    print("\nAll 150 seats by baseline D share (%):")
    for lab, df in (("2018", old), ("2026", new)):
        print(f"  {lab}        " + "  ".join(f"{v:5d}" for v in bands(df.base)))

    # The model's own read: seats it puts within reach at the same environment.
    import model_config as mc
    for env in (8.6,):
        with contextlib.redirect_stdout(io.StringIO()):
            lin = model.build_linear_predictions(model.load_districts(), env, mc.RACE_GENERIC_BALLOT_D_SHARE)
        full = model.load_districts()
        h = (full.chamber_lower == "house").values
        rheld = (full.incumbent_party == "R").values & h
        p = lin.values
        print(f"\n2026 model at D+{env}: R-held seats predicted D share >=50%: {int((p[rheld] >= .5).sum())}, "
              f"47-50%: {int(((p[rheld] >= .47) & (p[rheld] < .5)).sum())}, 44-47%: {int(((p[rheld] >= .44) & (p[rheld] < .47)).sum())}")


if __name__ == "__main__":
    main()
