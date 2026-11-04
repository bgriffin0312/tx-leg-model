"""Election night: run the whole-election model at the T the early returns imply.

    python scripts/early_vote/election_night_T.py 2.4 0.4 4.0
Prints headline outcomes at T -6.0 (pre-election read) and each T given, plus seat means at the first T
for the seats the county early returns already show.
"""
import sys
import types
from pathlib import Path

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd


def load_patched(name, path, a, b_):
    s = Path(path).read_text(encoding="utf-8")
    assert s.count(a) == 1, a
    m = types.ModuleType(name)
    m.__file__ = str(path)
    sys.modules[name] = m
    exec(compile(s.replace(a, b_), str(path), "exec"), m.__dict__)
    return m


wm = load_patched("whole_election_model", ROOT / "src" / "whole_election_model.py", 'center = self.Tc if env == "statewide" else T_LEGISLATIVE', 'center = self.Tc if env == "statewide" else (T_LEGISLATIVE if env == "legislative" else float(env))')
wh = wm.Whole()
Ts = [-6.0] + [float(x) for x in sys.argv[1:]]
rows, runs = [], {}
for T in Ts:
    r = wh.run(T)
    runs[T] = r
    i = {k: n for n, k in enumerate(r["names"])}
    W, M = r["W"], r["M"]
    rows.append({"T": T, "TX House D": r["house"].mean(), "10-90": f"{np.percentile(r['house'], 10):.0f}-{np.percentile(r['house'], 90):.0f}", "P(76+)": (r["house"] >= 76).mean(),
                 "TX Senate D": r["senate"].mean(), "US House D": r["seats"].mean(),
                 "Talarico": W[:, i["us_senate"]].mean(), "Sen margin": M[:, i["us_senate"]].mean(),
                 "Hinojosa": W[:, i["governor"]].mean(), "Gov margin": M[:, i["governor"]].mean(),
                 "AG D": W[:, i["attorney_general"]].mean(), "LtGov D": W[:, i["lt_governor"]].mean(),
                 "RRC Rosenthal": W[:, i["railroad_commissioner"]].mean(), "RRC margin": M[:, i["railroad_commissioner"]].mean(),
                 "Tarrant CJ": W[:, i["tarrant"]].mean(), "Denton CJ": W[:, i["denton"]].mean(), "Collin CJ": W[:, i["collin"]].mean()})
pd.set_option("display.width", 250, "display.max_columns", 30)
print(pd.DataFrame(rows).round(3).to_string(index=False))

T0 = Ts[1] if len(Ts) > 1 else Ts[0]
r = runs[T0]
i = {k: n for n, k in enumerate(r["names"])}
seen = {"hd94": 6.7, "hd96": 8.1, "hd97": 5.1, "sd9": 5.2, "tx24": None, "hd108": -7.3, "hd112": 10.4, "hd106": 0.0, "hd57": -3.7, "hd63": 5.0, "hd65": -3.4}
print(f"\nSeats at T {T0:+.1f}: model mean margin and P(D), vs early count so far (county pieces only)")
for k, ev in seen.items():
    if k in i:
        mm = r["M"][:, i[k]].mean(); p = r["W"][:, i[k]].mean()
        m6 = runs[-6.0]["M"][:, i[k]].mean()
        print(f"  {k:6s} model at -6.0 {m6:+6.1f} | at {T0:+.1f} {mm:+6.1f} ({p:.0%}) | early so far {'' if ev is None else f'{ev:+.1f}'}")
# full list of House seats flipping in expectation at T0
hd = [k for k in r["names"] if k.startswith("hd")]
tab = pd.DataFrame({"race": hd, "m6": [runs[-6.0]["M"][:, i[k]].mean() for k in hd], "p6": [runs[-6.0]["W"][:, i[k]].mean() for k in hd], "mT": [r["M"][:, i[k]].mean() for k in hd], "pT": [r["W"][:, i[k]].mean() for k in hd]})
tab = tab[(tab.pT > 0.1) & (tab.pT < 0.97) | ((tab.p6 < 0.5) & (tab.pT > 0.5))].sort_values("pT", ascending=False)
print(f"\nHouse seats in play at T {T0:+.1f} (P(D) between 10% and 97%, or flipped to favored)")
print(tab.round(2).to_string(index=False))
