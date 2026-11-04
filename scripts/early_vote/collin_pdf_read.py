"""Parse Collin County's Electionware summary text (pdfplumber output) and compare with the model."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
txt = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/raw/election_night/collin/collin_ev_summary.txt").read_text(encoding="utf-8")
bench = pd.read_csv(ROOT / "output" / "ev_night_benchmarks_2026.csv", index_col=0)
rs = pd.read_csv(ROOT / "data" / "raw" / "early_vote" / "race_shift.csv").set_index("race")

contests, cur = {}, None
for line in txt.splitlines():
    s = line.strip()
    if s.startswith("Vote For"):
        continue
    m = re.match(r"^(REP|DEM|LIB|GRN|IND|W)\s+(.+?)\s+([\d,]+)\s+[\d.]+%", s)
    if m and cur:
        contests.setdefault(cur, {})[m.group(1)] = contests.get(cur, {}).get(m.group(1), 0) + int(m.group(3).replace(",", ""))
        continue
    if re.match(r"^(Total Votes Cast|Overvotes|Undervotes|Contest Totals|TOTAL VOTE|Mail Voting|Write-In|Summary Results|General and Special|November 3|Election Summary|Report generated|Collin County|Watermark|Statistics|Registered|Ballots Cast|Voter Turnout)", s):
        continue
    if re.search(r"[A-Za-z]", s) and not m:
        cur = s


def mm(d, r):
    return (d - r) / (d + r) * 100 if d + r else np.nan


def fmt(x):
    return f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}"


jd = jr = 0
for name, v in contests.items():
    if re.search(r"Supreme Court|Criminal Appeals", name, re.I) and "DEM" in v and "REP" in v:
        jd += v["DEM"]; jr += v["REP"]
        print(f"  {name}: {fmt(mm(v['DEM'], v['REP']))}")
jm = mm(jd, jr)
b = bench.loc["COLLIN"]
base_c = b["lean24"] + b["cshift"] + (b["ev_T-6.0"] - b["final_T-6.0"])
print(f"COLLIN judicial early {fmt(jm)} on {jd + jr:,} | bench {fmt(b['ev_T-6.0'])} dev {jm - b['ev_T-6.0']:+.1f} | implied T {jm - base_c:+.1f} | 2024 judicial {fmt(b['j24'])}")
for key in ("United States Senator", "Governor", "Lieutenant Governor", "Attorney General", "County Judge"):
    for name, v in contests.items():
        if name.lower().startswith(key.lower()) and "DEM" in v and "REP" in v:
            print(f"  {name}: {fmt(mm(v['DEM'], v['REP']))}")
            break
for name, v in contests.items():
    m = re.search(r"(State Representative|State Senator|United States Representative).*?(\d+)", name)
    if m and "DEM" in v and "REP" in v:
        k = {"State Representative": "hd", "State Senator": "sd", "United States Representative": "tx"}[m.group(1)] + m.group(2)
        mod = f"model -6.0 {fmt(rs.loc[k, 'm1'])} ({rs.loc[k, 'p1']:.0%})" if k in rs.index else ""
        print(f"  {k}: early {fmt(mm(v['DEM'], v['REP']))}  {mod}")
