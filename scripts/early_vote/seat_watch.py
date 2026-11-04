"""Loop the seat report every few minutes; print only changes worth an alert:
a flagged seat changes leader, gets its first Election Day votes, or first reports at all.
    python -u scripts/early_vote/seat_watch.py >> data/raw/election_night/poll.log
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
EN = ROOT / "data" / "raw" / "election_night"
EVERY = 180


def fmt(x):
    return "n/a" if pd.isna(x) else (f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}")


prev = None
print(f"SEATWATCH start {time.strftime('%H:%M:%S')}", flush=True)
while True:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "early_vote" / "seat_report.py")], cwd=ROOT, capture_output=True, timeout=900)
    files = sorted(glob.glob(str(EN / "seat_report_*.csv")), key=os.path.getmtime)
    if files:
        cur = pd.read_csv(files[-1]).set_index("seat")
        if prev is not None:
            msgs = []
            for s, r in cur.iterrows():
                p = prev.loc[s] if s in prev.index else None
                if p is None:
                    continue
                lead_now = None if pd.isna(r.now) else ("D" if r.now > 0 else "R")
                lead_was = None if pd.isna(p.now) else ("D" if p.now > 0 else "R")
                if lead_was is None and lead_now is not None:
                    msgs.append(f"{s} ({r.held}-held, model {fmt(r.model)} {r.p:.0%}) first votes: {fmt(r.now)} [{r.pieces}]")
                elif lead_now != lead_was and lead_now is not None:
                    msgs.append(f"{s} ({r.held}-held, model {r.p:.0%}) LEAD CHANGE {fmt(p.now)} -> {fmt(r.now)} [E-day {r.eday_share:.0f}% of count]")
                if (pd.isna(p.eday_share) or p.eday_share == 0) and r.eday_share > 0:
                    msgs.append(f"{s} first Election Day votes: E-day {fmt(r.eday)} vs early {fmt(r.early)}; now {fmt(r.now)} [{r.eday_share:.0f}% E-day]")
            if msgs:
                print(f"SEATWATCH {time.strftime('%H:%M:%S')}\n  " + "\n  ".join(msgs), flush=True)
        prev = cur
    time.sleep(EVERY)
