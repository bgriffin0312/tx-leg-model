"""
joint_combos.py

Joint odds across the 8 statewide races, the 38 U.S. House seats and the 10 county races,
from one simulation. County races take their shared error from the statewide run's Texas
environment draw plus their own formula error, so each race's own odds stay as published
in the county model. The legislative model is not in here.

  python scripts/joint_combos.py --build                   # rebuild draws and the pair table
  python scripts/joint_combos.py "us_senate & tarrant"     # P(both D wins)
  python scripts/joint_combos.py "tarrant" --given "us_senate"
  python scripts/joint_combos.py "seats>=12 & !tx23"

Names: us_senate, governor, lt_governor, attorney_general, comptroller, land_commissioner,
ag_commissioner, railroad_commissioner; tx1..tx38; tarrant, williamson, collin, denton,
fort_bend, harris, bexar, montgomery, el_paso, bexar_da. Every name means the Democrat wins;
! means the Democrat loses. For bexar_da that covers both Foster and Wolff winning.

Output: output/joint_outcomes_2026.npz, output/joint_pairs_2026.csv
"""
import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
NPZ = ROOT / "output" / "joint_outcomes_2026.npz"
PAIRS = ROOT / "output" / "joint_pairs_2026.csv"
KEY = ["us_senate", "governor", "railroad_commissioner", "attorney_general", "lt_governor", "tx15", "tx23", "tx28", "tx34", "tx35",
       "tarrant", "williamson", "collin", "denton", "fort_bend", "harris"]


def build():
    import statewide_pollster_history  # noqa: F401  (statewide_model reads what it writes)
    import statewide_model as sw
    import county_judge_model as cj
    with contextlib.redirect_stdout(io.StringIO()):
        b = sw.build()
        f = sw.forecast(*(b[k] for k in ("R", "C", "dp", "Tmu", "Tsd", "c_sd", "h_sd", "e_sd", "sds")), sw.POLL_B_MEAN, quiet=True)
        c = cj.main()
    T = f["T"] - f["T"].mean()
    rng = np.random.default_rng(4242)
    rest = np.sqrt(max((c["nat"] * 200) ** 2 - T.var(), 0.0))
    shared = (T + rng.normal(0, rest, len(T))) / 200
    keep = {}
    o = cj.run(c["r"], c["k_open"], c["k_held"], c["nat"], c["within"], shared_draws=shared, seed=2027, keep=keep)
    gap = (o.p_d - c["o"].p_d).abs().max()
    print(f"county odds vs the county model: largest gap {gap:.3f}; Texas sd {T.std():.2f}, other shared {rest:.2f}")
    R, C = f["R"], f["C"]
    race = list(R["race"]) if "race" in R else list(R.index)
    names = race + [f"tx{d}" for d in C.index]
    wins = [f["sw_y"] > 0, f["cd_y"] > 0]
    cnames = []
    for (county, office), w in keep.items():
        cnames.append("bexar_da" if office == "district_attorney" else county.replace(" ", "_"))
        wins.append(w[:, None])
    M = np.hstack(wins).astype(bool)
    names = names + cnames
    seats = (f["cd_y"] > 0).sum(1).astype(np.int8)
    np.savez_compressed(NPZ, wins=M, names=np.array(names), seats=seats)
    rows = []
    idx = {n: i for i, n in enumerate(names)}
    for i, a in enumerate(KEY):
        for b_ in KEY[i + 1:]:
            x, y = M[:, idx[a]], M[:, idx[b_]]
            rows.append({"a": a, "b": b_, "p_a": x.mean(), "p_b": y.mean(), "both_d": (x & y).mean(), "a_d_b_r": (x & ~y).mean(),
                         "a_r_b_d": (~x & y).mean(), "both_r": (~x & ~y).mean(), "p_b_if_a_d": y[x].mean(), "p_b_if_a_r": y[~x].mean(),
                         "corr": np.corrcoef(x, y)[0, 1]})
    pd.DataFrame(rows).round(4).to_csv(PAIRS, index=False)
    print(f"wrote {NPZ.name}, {PAIRS.name}: {M.shape[0]} draws, {M.shape[1]} races")


def mask(expr, M, names, seats):
    idx = {n: i for i, n in enumerate(names)}
    out = np.ones(len(M), bool)
    for tok in re.split(r"\s*&\s*", expr.strip()):
        s = re.fullmatch(r"seats\s*(>=|<=|==)\s*(\d+)", tok)
        if s:
            v = int(s.group(2))
            out &= {">=": seats >= v, "<=": seats <= v, "==": seats == v}[s.group(1)]
            continue
        neg = tok.startswith("!")
        name = tok.lstrip("!").strip()
        if name not in idx:
            sys.exit(f"unknown name {name}")
        out &= ~M[:, idx[name]] if neg else M[:, idx[name]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expr", nargs="*")
    ap.add_argument("--given")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if not a.expr:
        return
    d = np.load(NPZ)
    M, names, seats = d["wins"], [str(x) for x in d["names"]], d["seats"]
    g = mask(a.given, M, names, seats) if a.given else np.ones(len(M), bool)
    for e in a.expr:
        m = mask(e, M, names, seats)
        tail = f" given {a.given}" if a.given else ""
        print(f"P({e}{tail}) = {m[g].mean():.3f}")


if __name__ == "__main__":
    main()
