"""Read Clarity ENR county results (summary.json) on election night and compare with the model.

    python scripts/early_vote/clarity_read.py Tarrant/127297 Bexar/127291 ...
Saves each summary to data/raw/election_night/clarity_<county>_<ver>.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
OUT = ROOT / "data" / "raw" / "election_night"
OUT.mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
bench = pd.read_csv(ROOT / "output" / "ev_night_benchmarks_2026.csv", index_col=0)
rs = pd.read_csv(ROOT / "data" / "raw" / "early_vote" / "race_shift.csv").set_index("race")


def fmt(m):
    return "n/a" if m is None or np.isnan(m) else (f"D+{m:.1f}" if m >= 0 else f"R+{-m:.1f}")


def dr(c):
    d = r = 0
    for p, v in zip(c["P"], c["V"]):
        if p == "DEM":
            d += v
        elif p == "REP":
            r += v
    return d, r


def mm(d, r):
    return (d - r) / (d + r) * 100 if d + r else np.nan


for arg in sys.argv[1:]:
    county, eid = arg.split("/")
    base = f"https://results.enr.clarityelections.com/TX/{county}/{eid}"
    ver = requests.get(f"{base}/current_ver.txt", headers=HDR, timeout=30).text.strip()
    js = requests.get(f"{base}/{ver}/json/en/summary.json", headers=HDR, timeout=60).json()
    (OUT / f"clarity_{county}_{ver}.json").write_text(json.dumps(js))
    jd = jr = 0
    races = {}
    for c in js:
        name = c["C"].strip()
        d, r = dr(c)
        if d == 0 or r == 0:
            continue
        if re.search(r"Supreme Court|Criminal Appeals", name, re.I):
            jd += d; jr += r
        races[name] = (d, r, c.get("PR"), c.get("TP"))
    cu = county.upper().replace("FORTBEND", "FORT BEND")
    b = bench.loc[cu] if cu in bench.index else None
    jm = mm(jd, jr)
    print(f"\n== {county} (ver {ver}) judicial early {fmt(jm)} on {jd + jr:,} votes")
    if b is not None:
        base_c = b["lean24"] + b["cshift"] + (b["ev_T-6.0"] - b["final_T-6.0"])
        print(f"   benchmark early at T-6.0 {fmt(b['ev_T-6.0'])} (SW {fmt(b['ev_SW-4.8'])}, LEG {fmt(b['ev_LEG-7.2'])}); dev {jm - b['ev_T-6.0']:+.1f}; implied T {jm - base_c:+.1f}; 2024 judicial {fmt(b['j24'])}")
    for key in ("United States Senator", "Governor", "Lieutenant Governor", "Attorney General", "Comptroller of Public Accounts", "Railroad Commissioner", "County Judge"):
        for name, (d, r, pr, tp) in races.items():
            if name.lower().startswith(key.lower()):
                print(f"   {name}: {fmt(mm(d, r))}  (D {d:,} R {r:,}; precincts {pr}/{tp})")
                break
    for name, (d, r, pr, tp) in races.items():
        m = re.search(r"(State Representative|State Senator|United States Representative).*?(\d+)", name)
        if not m:
            continue
        key = {"State Representative": "hd", "State Senator": "sd", "United States Representative": "tx"}[m.group(1)] + m.group(2)
        mod = rs.loc[key] if key in rs.index else None
        tag = f"model final {fmt(mod.m1)} ({mod.p1:.0%})" if mod is not None else ""
        if mod is not None and 0.03 < mod.p1 < 0.97:
            print(f"   {key}: early {fmt(mm(d, r))}  {tag}")
