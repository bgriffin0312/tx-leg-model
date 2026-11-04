"""Election night: project the statewide races as projected T + each race's observed gap to the judicial vote.

Gap = pooled early-vote margin of the race minus pooled early-vote judicial margin, over every county with
both (state feed snapshot + saved Clarity county files + Harris + Collin). T from the latest projection
block in poll.log (center and the k range).
    python scripts/early_vote/statewide_races.py
"""
import glob
import gzip
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
EN = ROOT / "data" / "raw" / "election_night"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RACES = [("us_senate", r"SENATOR", "Talarico v Paxton", 6.9), ("governor", r"^GOVERNOR$", "Hinojosa v Abbott", 1.4), ("lt_governor", r"LIEUTENANT GOVERNOR", "Goodwin v Patrick", -0.1),
         ("attorney_general", r"ATTORNEY GENERAL", "Johnson v Middleton", 0.0), ("comptroller", r"COMPTROLLER", "D v R", -1.9), ("land", r"GENERAL LAND OFFICE|LAND COMMISSIONER", "D v R", -4.6),
         ("agriculture", r"AGRICULTURE", "Tucker v Sheets", -1.4), ("railroad", r"RAILROAD", "Rosenthal v French", 2.3)]
JUD = re.compile(r"SUPREME COURT|CRIMINAL APPEALS", re.I)


def norm(name):
    n = name.upper().replace("U. S.", "UNITED STATES").strip()
    if "SENATOR" in n and "STATE SENATOR" not in n:
        return "us_senate"
    for key, pat, *_ in RACES[1:]:
        if re.search(pat, n):
            return key
    return "judicial" if JUD.search(n) else None


county = {}   # county -> {race: [D_early, R_early]}


def jkey(name):
    """Each statewide judicial race on its own (added 12:30am), besides the pooled 'judicial' composite."""
    n = name.upper()
    if "15TH COURT OF APPEALS" in n:
        return "coa15-chief" if "CHIEF" in n else None
    if "SUPREME COURT" in n:
        if "CHIEF" in n:
            return "sc-chief"
        m = re.search(r"PLACE\D*(\d+)", n)
        return f"sc-{int(m.group(1))}" if m else None
    if "CRIMINAL APPEALS" in n:
        if "PRESIDING" in n:
            return "cca-presiding"
        m = re.search(r"PLACE\D*(\d+)", n)
        return f"cca-{int(m.group(1))}" if m else None
    return None


def skip_district(name):
    return "DISTRICT" in name.upper() and "15TH COURT OF APPEALS" not in name.upper()


def put(cn, race, d, r):
    if race:
        a = county.setdefault(cn, {}).setdefault(race, [0, 0])
        a[0] += d; a[1] += r


def putn(cn, name, d, r):
    put(cn, norm(name), d, r)
    put(cn, jkey(name), d, r)


JLAB = {"coa15-chief": "Zimmerer v Brister* (15th Court of Appeals, Chief)"}   # judicial key -> "Dem v Rep* (office)"


def office_label(jk, name):
    if jk == "sc-chief":
        return "Supreme Court, Chief"
    if jk.startswith("sc-"):
        return f"Supreme Court, Pl. {jk[3:]}" + (" unexpired" if "UNEXPIRED" in name.upper() else "")
    if jk.startswith("cca-"):
        return f"Criminal Appeals, Pl. {jk[4:]}"
    return "15th Court of Appeals, Chief"


# counties the projection calls fully counted from the state feed: use ALL their votes (added 5am Nov 4,
# after Election Day voters split tickets differently, e.g. the Senate gap +10.5 on Election Day vs +6.4 early)
_pj = sorted(glob.glob(str(EN / "projection_*.csv")), key=os.path.getmtime)
DONE = set()
if _pj:
    import pandas as _pd
    _p = _pd.read_csv(_pj[-1])
    DONE = set(_p[(_p.status == "complete") & (_p.src == "state")].county)
snaps = sorted(glob.glob(str(EN / "countyInfo_*.json.gz")), key=os.path.getmtime)
if snaps:
    for c in json.load(gzip.open(snaps[-1], "rt", encoding="utf-8")).values():
        cn = c["N"].upper()
        fld = "V" if cn.replace("LASALLE", "LA SALLE") in DONE else "EV"
        for r in c["Races"].values():
            if r.get("OT") != "SW" and "SENATOR" not in r["N"].upper():
                continue
            if "STATE SENATOR" in r["N"].upper():
                continue
            d = sum((x.get(fld) or 0) for x in r["C"].values() if x["P"] == "DEM")
            rr = sum((x.get(fld) or 0) for x in r["C"].values() if x["P"] == "REP")
            putn(cn, r["N"], d, rr)
            jk = jkey(r["N"])
            if jk and jk not in JLAB:
                nm = {x["P"]: (x.get("LN") or x.get("N", "")).title() + ("*" if "(I)" in x.get("N", "") else "") for x in r["C"].values() if x["P"] in ("DEM", "REP")}
                JLAB[jk] = f"{nm.get('DEM', '?')} v {nm.get('REP', '?')} ({office_label(jk, r['N'])})"
for cn in ("TARRANT", "DALLAS", "BEXAR", "DENTON", "EL_PASO", "WILLIAMSON", "FORT_BEND"):
    if cn.replace("_", " ") in DONE:     # fully counted in the state feed: keep its all-votes numbers
        continue
    files = sorted(glob.glob(str(EN / f"clarity_{cn}_*.json")), key=os.path.getmtime, reverse=True)
    for f in files:
        js = json.loads(Path(f).read_text())
        if isinstance(js, dict):
            county.pop(cn.replace("_", " "), None)
            for name, v in js.items():
                if "STATE SENATOR" in name.upper() or skip_district(name):
                    continue
                putn(cn.replace("_", " "), name, v["DEM"][0], v["REP"][0])
            break
r = requests.get(f"https://appfiles.harrisvotes.com/harrisvotes/prd/Data/1126/resultsbyballot.json?nocache={int(time.time() // 60 * 60000)}", headers=HDR, timeout=60)
if r.ok:
    county.pop("HARRIS", None)
    for c in r.json().get("contests", []):
        if skip_district(c["t"]):
            continue
        d = sum(c["ev"][i] + c["ab"][i] for i, n in enumerate(c["ca"]) if "(DEM)" in n)
        rr = sum(c["ev"][i] + c["ab"][i] for i, n in enumerate(c["ca"]) if "(REP)" in n)
        putn("HARRIS", c["t"], d, rr)
cp = sorted(glob.glob(str(EN / "collin" / "*ev*.txt")), key=os.path.getmtime)
if cp:
    county.pop("COLLIN", None)
    cur = None
    for line in Path(cp[-1]).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = re.match(r"^(REP|DEM)\s+(.+?)\s+([\d,]+)\s+[\d.]+%", s)
        if m and cur:
            putn("COLLIN", cur, int(m.group(3).replace(",", "")) if m.group(1) == "DEM" else 0, int(m.group(3).replace(",", "")) if m.group(1) == "REP" else 0)
        elif re.search(r"[A-Za-z]", s) and not m and not re.match(r"^(Vote For|TOTAL|Mail|Total Votes|Over|Under|Contest|Summary|General|November|Election Summary|Report|Collin|Watermark|Statistics|Registered|Ballots|Voter)", s):
            cur = None if skip_district(s) else s.replace(" Collin County Seal", "")

# latest projection T and k range from poll.log
log = (EN / "poll.log").read_text(encoding="utf-8", errors="replace")
blk = log[log.rfind("=== STATEWIDE PROJECTION"):]
T = float(re.search(r"PROJECTED T ([+-][\d.]+)", blk).group(1))
ks = {float(a): float(b) for a, b in re.findall(r"if k = ([\d.]+): T ([+-][\d.]+)", blk)}
lo, hi = ks.get(0.6, T - 3), ks.get(0.9, T + 2)
print(f"=== STATEWIDE RACES {time.strftime('%H:%M:%S')} === projected T {T:+.1f} (k 0.6 -> {lo:+.1f}, k 0.9 -> {hi:+.1f}); {len([c for c in county if sum(county[c].get('judicial', [0, 0])) > 0])} counties with votes; all votes in {len(DONE)} fully counted, early elsewhere")
fmt = lambda x: f"D+{x:.1f}" if x >= 0 else f"R+{-x:.1f}"
out_rows = []
for key, pat, label, model_gap in RACES:
    D = R = JD = JR = 0
    for cn, v in county.items():
        if key in v and "judicial" in v and sum(v[key]) > 0 and sum(v["judicial"]) > 0:
            D += v[key][0]; R += v[key][1]; JD += v["judicial"][0]; JR += v["judicial"][1]
    if D + R == 0:
        continue
    gap = (D - R) / (D + R) * 100 - (JD - JR) / (JD + JR) * 100
    print(f"  {label:20s} gap to judicial {gap:+5.1f} (model {model_gap:+.1f}) | projected {fmt(T + gap):>7s}  range {fmt(lo + gap)} to {fmt(hi + gap)}")
    out_rows.append({"key": key, "label": label, "gap": round(gap, 2), "model_gap": model_gap, "proj": round(T + gap, 2), "lo": round(lo + gap, 2), "hi": round(hi + gap, 2)})
# each statewide judicial race: its gap to the pooled judicial vote (the model treats them all as T)
jk_all = sorted({k for v in county.values() for k in v if re.match(r"^(sc|cca|coa15)-", k)})
for key in jk_all:
    D = R = JD = JR = 0
    for cn, v in county.items():
        if key in v and "judicial" in v and sum(v[key]) > 0 and sum(v["judicial"]) > 0:
            D += v[key][0]; R += v[key][1]; JD += v["judicial"][0]; JR += v["judicial"][1]
    if D + R == 0:
        continue
    gap = (D - R) / (D + R) * 100 - (JD - JR) / (JD + JR) * 100
    label = JLAB.get(key, key)
    print(f"  {label[:44]:44s} gap {gap:+5.1f} | projected {fmt(T + gap):>7s}  range {fmt(lo + gap)} to {fmt(hi + gap)}")
    out_rows.append({"key": key, "label": label, "gap": round(gap, 2), "model_gap": 0.0, "proj": round(T + gap, 2), "lo": round(lo + gap, 2), "hi": round(hi + gap, 2)})
import csv
with open(EN / "statewide_races_latest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["key"])
    w.writeheader()
    w.writerows(out_rows)
