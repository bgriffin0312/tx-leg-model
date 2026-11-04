"""Compare the two newest seat_report CSVs; print lead changes, first votes and first Election Day votes."""
import glob
import os
from pathlib import Path

import pandas as pd

EN = Path(r"C:\Users\Brennan\code\tx-leg-model\data\raw\election_night")
files = sorted(glob.glob(str(EN / "seat_report_*.csv")), key=os.path.getmtime)
if len(files) >= 2:
    prev, cur = pd.read_csv(files[-2]).set_index("seat"), pd.read_csv(files[-1]).set_index("seat")
    fmt = lambda x: "n/a" if pd.isna(x) else (f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}")
    for s, r in cur.iterrows():
        if s not in prev.index:
            continue
        p = prev.loc[s]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        ln = None if pd.isna(r.now) else ("D" if r.now > 0 else "R")
        lw = None if pd.isna(p.now) else ("D" if p.now > 0 else "R")
        nm = r.get("race", s)
        if lw is None and ln is not None:
            print(f"ALERT first votes {s} {nm}: {fmt(r.now)} [{r.pieces}]")
        elif ln is not None and ln != lw:
            print(f"ALERT LEAD CHANGE {s} {nm}: {fmt(p.now)} -> {fmt(r.now)} [E-day {r.eday_share:.0f}% of count]")
        if (pd.isna(p.eday_share) or p.eday_share == 0) and r.eday_share > 0:
            print(f"ALERT first Election Day votes {s} {nm}: E-day {fmt(r.eday)} vs early {fmt(r.early)}; now {fmt(r.now)}")
