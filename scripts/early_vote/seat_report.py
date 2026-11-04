"""Election night seat report: the legislative races the model flagged as possible changes.

Flagged = model P(D) between 4% and 96% at T -6.0 with county shifts (race_shift.csv), plus SD-9.
Current counts are summed over county pieces from the freshest source for each county:
Clarity county feeds (Tarrant, Dallas, Bexar, Denton, El Paso, Williamson), Harris JSON, the latest
Collin PDF parse, and the state feed for every other county. Early and Election Day are kept apart.
    python scripts/early_vote/seat_report.py
"""
import glob
import gzip
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
EN = ROOT / "data" / "raw" / "election_night"
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
rs = pd.read_csv(ROOT / "data" / "raw" / "early_vote" / "race_shift.csv").set_index("race")
FLAG = list(dict.fromkeys([r for r in rs.index if r.startswith(("hd", "sd")) and 0.01 <= rs.loc[r, "p1"] <= 0.99] + ["sd9"]))
if os.environ.get("SEATS") == "senate":       # every Senate seat on the ballot, whatever the model gave it
    FLAG = [r for r in rs.index if r.startswith("sd")]
if os.environ.get("SEATS") == "all":          # every House and Senate seat on the ballot (5am Nov 4, Brennan's ask)
    FLAG = [r for r in rs.index if r.startswith(("hd", "sd"))]
CLARITY = {"TRAVIS": "Travis/127492", "TARRANT": "Tarrant/127297", "DALLAS": "Dallas/127396", "BEXAR": "Bexar/127291", "DENTON": "Denton/127398", "EL PASO": "El_Paso/127348", "WILLIAMSON": "Williamson/127488", "FORT BEND": "Fort_Bend/127480"}
inc = pd.read_csv(ROOT / "data" / "processed" / "districts_2026.csv")
inc["key"] = np.where(inc.chamber.str.lower().str.startswith("h"), "hd", "sd") + inc.district.astype(str)
INC = inc.set_index("key").incumbent_party.fillna(inc.set_index("key").last_election_winner_party).to_dict()   # SD 4 vacant: last winner R


def seat_key(name):
    m = re.search(r"(STATE REPRESENTATIVE|STATE SENATOR)\D*(\d+)", name.upper())
    return ({"STATE REPRESENTATIVE": "hd", "STATE SENATOR": "sd"}[m.group(1)] + str(int(m.group(2)))) if m else None


def get(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HDR, timeout=90)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(2 + 2 * i)
    return None


pieces = {}   # (seat, county) -> dict(De, Re, Dd, Rd, src, reporting)


def add(seat, county, De, Re, Dd, Rd, src):
    if seat in FLAG:
        pieces[(seat, county)] = {"De": De, "Re": Re, "Dd": Dd, "Rd": Rd, "src": src}


# 1. state feed, latest snapshot (also the candidate names; "(I)" marks incumbents)
NAMES = {}
snaps = sorted(glob.glob(str(EN / "countyInfo_*.json.gz")), key=os.path.getmtime)
if snaps:
    ci = json.load(gzip.open(snaps[-1], "rt", encoding="utf-8"))
    for c in ci.values():
        cn = c["N"].upper()
        for r in c["Races"].values():
            k = seat_key(r["N"])
            if not k:
                continue
            for x in r["C"].values():
                if x.get("P") in ("DEM", "REP"):
                    ln = (x.get("LN") or x.get("N", "")).title()
                    NAMES.setdefault(k, {})[x["P"]] = ln + ("*" if "(I)" in x.get("N", "") else "")
            v = {"DEM": [0, 0], "REP": [0, 0]}
            for x in r["C"].values():
                if x["P"] in v:
                    v[x["P"]][0] += x.get("EV") or 0
                    v[x["P"]][1] += x.get("V") or 0
            add(k, cn, v["DEM"][0], v["REP"][0], v["DEM"][1] - v["DEM"][0], v["REP"][1] - v["REP"][0], "state")
# 2. Clarity counties, live
done = set()
LIVE_CLARITY = False   # 9pm: Clarity returning 502s; read county_poll.py's saved files instead
for county, path in (CLARITY.items() if LIVE_CLARITY else []):
    base = f"https://results.enr.clarityelections.com/TX/{path}"
    r = get(f"{base}/current_ver.txt")
    if r is None or not r.text.strip().isdigit():
        continue
    z = get(f"{base}/{r.text.strip()}/reports/detailxml.zip")
    if z is None:
        continue
    try:
        x = zipfile.ZipFile(io.BytesIO(z.content)).read("detail.xml").decode("utf-8", "replace")
    except zipfile.BadZipFile:
        continue
    ts = (re.findall(r"<Timestamp>([^<]*)</Timestamp>", x) or [""])[0]
    try:
        all_early = time.strptime(ts.replace(" CST", "").strip(), "%m/%d/%Y %I:%M:%S %p") < time.strptime("11/3/2026 8:00:00 PM", "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        all_early = False
    done.add(county)
    for c in re.finditer(r'<Contest key="\d+" text="([^"]+)"[^>]*>(.*?)</Contest>', x, re.S):
        k = seat_key(c.group(1))
        if not k:
            continue
        v = {"DEM": [0, 0], "REP": [0, 0]}
        for ch in re.finditer(r'<Choice key="\d+" text="[^"]*" party="([^"]*)" totalVotes="(\d+)">(.*?)</Choice>', c.group(2), re.S):
            p = ch.group(1)
            if p not in v:
                continue
            for vt, n in re.findall(r'<VoteType name="([^"]+)" votes="(\d+)"', ch.group(3)):
                if re.search(r"overvote|undervote", vt, re.I):
                    continue
                v[p][1 if (re.match(r"election", vt, re.I) and not all_early) else 0] += int(n)
        add(k, county, v["DEM"][0], v["REP"][0], v["DEM"][1], v["REP"][1], "county")
# 2b. Clarity counties the live fetch missed (server 502s): newest saved poller file
for county in set(CLARITY) - done:
    files = sorted(glob.glob(str(EN / f"clarity_{county.replace(' ', '_')}_*.json")), key=os.path.getmtime)
    for f in reversed(files):
        js = json.loads(Path(f).read_text())
        if isinstance(js, dict):          # county_poll format: {contest: {DEM: [early, eday, total], REP: [...]}}
            for name, v in js.items():
                k = seat_key(name)
                if k:
                    cur_p = pieces.get((k, county))
                    tot_saved = v["DEM"][0] + v["REP"][0] + v["DEM"][1] + v["REP"][1]
                    if cur_p and cur_p["De"] + cur_p["Re"] + cur_p["Dd"] + cur_p["Rd"] >= tot_saved:
                        continue          # the state feed is at least as far along
                    add(k, county, v["DEM"][0], v["REP"][0], v["DEM"][1], v["REP"][1], "saved")
            break
        if isinstance(js, list):          # raw summary.json: totals only, posted before Election Day votes
            for c in js:
                k = seat_key(c["C"])
                if k:
                    D = sum(v for p, v in zip(c["P"], c["V"]) if p == "DEM"); R = sum(v for p, v in zip(c["P"], c["V"]) if p == "REP")
                    add(k, county, D, R, 0, 0, "saved")
            break
# 3. Harris
r = get(f"https://appfiles.harrisvotes.com/harrisvotes/prd/Data/1126/resultsbyballot.json?nocache={int(time.time() // 60 * 60000)}")
if r is not None:
    for c in r.json().get("contests", []):
        k = seat_key(c["t"])
        if not k:
            continue
        v = {"DEM": [0, 0], "REP": [0, 0]}
        for i, nm in enumerate(c["ca"]):
            p = "DEM" if "(DEM)" in nm else "REP" if "(REP)" in nm else None
            if p:
                v[p][0] += c["ev"][i] + c["ab"][i]
                v[p][1] += c["ed"][i]
        add(k, "HARRIS", v["DEM"][0], v["REP"][0], v["DEM"][1], v["REP"][1], "county")
# 4. Collin, latest parsed PDF text (totals only; early report has no Election Day)
cp = sorted(glob.glob(str(EN / "collin" / "*.txt")), key=os.path.getmtime)
if cp:
    cur = None
    nm = Path(cp[-1]).name.lower()
    eday_report = not ("early" in nm or "_ev_" in nm)
    tot = {}

    def collin_parse(path, out):
        # 5am Nov 4 fix: a contest header is the line before "Vote For"; every header resets the current seat.
        # Before, only legislative headers did, so the contest after the last House race (HD 89) overwrote it.
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        cur = None
        for i, line in enumerate(lines):
            s = line.strip()
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("Vote For"):
                cur = seat_key(s) if re.search(r"(State Representative|State Senator)", s) else None
                continue
            m = re.match(r"^(REP|DEM)\s+(.+?)\s+([\d,]+)\s+[\d.]+%", s)
            if m and cur:
                out.setdefault(cur, {})[m.group(1)] = int(m.group(3).replace(",", ""))

    collin_parse(cp[-1], tot)
    early_tot = {}
    if eday_report:   # early-only numbers from Collin's early report; Election Day = latest total minus early
        evf = [f for f in cp if "_ev_" in Path(f).name.lower() or "early" in Path(f).name.lower()]
        if evf:
            collin_parse(evf[0], early_tot)
    for k, v in tot.items():
        if k:
            if eday_report:
                ev = early_tot.get(k, {})
                add(k, "COLLIN", ev.get("DEM", 0), ev.get("REP", 0), v.get("DEM", 0) - ev.get("DEM", 0), v.get("REP", 0) - ev.get("REP", 0), "county")
            else:
                add(k, "COLLIN", v.get("DEM", 0), v.get("REP", 0), 0, 0, "county")


def mg(d, r):
    return (d - r) / (d + r) * 100 if d + r else np.nan


def fmt(x):
    return "   n/a" if x is None or np.isnan(x) else (f"D+{x:4.1f}" if x >= 0 else f"R+{-x:4.1f}")


# counties fully counted (5am Nov 4): the projection's "complete" list, plus Collin when its newest PDF says
# every vote center reported. A seat whose every piece is in one of these is scored on its actual count.
DONE_C = set()
_pj = sorted(glob.glob(str(EN / "projection_*.csv")), key=os.path.getmtime)
if _pj:
    _p = pd.read_csv(_pj[-1])
    DONE_C = set(_p[_p.status == "complete"].county)
# Fort Bend: its Clarity count (12:45am) has HD 26 at 37 of 37 precincts, HD 28 at 53 of 54; the state feed never got
# its Election Day votes. Checked by hand at 5am Nov 4.
DONE_C.add("FORT BEND")
if cp:
    _vc = re.search(r"(\d+) of (\d+) Vote Centers Reporting", Path(cp[-1]).read_text(encoding="utf-8"))
    if _vc and _vc.group(1) == _vc.group(2) and eday_report:
        DONE_C.add("COLLIN")
rows = []
for seat in FLAG:
    ps = {c: v for (s, c), v in pieces.items() if s == seat}
    De = sum(v["De"] for v in ps.values()); Re = sum(v["Re"] for v in ps.values())
    Dd = sum(v["Dd"] for v in ps.values()); Rd = sum(v["Rd"] for v in ps.values())
    n = De + Re + Dd + Rd
    rep = sum(1 for v in ps.values() if v["De"] + v["Re"] + v["Dd"] + v["Rd"] > 0)
    nm = NAMES.get(seat, {})
    rows.append({"seat": seat, "race": f"{nm.get('REP', '?')} (R) v {nm.get('DEM', '?')} (D)", "held": INC.get(seat, "?"), "model": rs.loc[seat, "m1"], "p": rs.loc[seat, "p1"], "now": mg(De + Dd, Re + Rd), "early": mg(De, Re), "eday": mg(Dd, Rd),
                 "eday_share": (Dd + Rd) / n * 100 if n else np.nan, "votes": n, "pieces": f"{rep}/{len(ps)}",
                 "final": bool(ps) and rep == len(ps) and all(c.replace("LASALLE", "LA SALLE") in DONE_C for c in ps)})
t = pd.DataFrame(rows).sort_values("p", ascending=False)

# ---------- projection: same Election Day correction as the statewide projection ----------
from math import erf, sqrt
Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))
J24, BETA = -16.2, 0.95
logtxt = (EN / "poll.log").read_text(encoding="utf-8", errors="replace")
blk = logtxt[logtxt.rfind("=== STATEWIDE PROJECTION"):]
try:
    TP = float(re.search(r"PROJECTED T ([+-][\d.]+)", blk).group(1))
    KU = float(re.search(r"used ([\d.]+) \(prior", blk).group(1))
    KT = {float(a): float(b) for a, b in re.findall(r"if k = ([\d.]+): T ([+-][\d.]+)", blk)}
    km = re.search(r"metro factor for big counties[^:]*: ([\d.]+)", blk)
    if km:                       # nearly every close seat is metro or suburban: use the metro factor (10:22pm)
        KU = float(km.group(1))
except AttributeError:
    TP, KU, KT = -6.0, 0.6, {}


def proj(r, k):
    anchor = r.model - BETA * (-6.0 - J24)          # the model's race at 2024's statewide level
    if np.isnan(r.early):
        return np.nan, np.nan
    if r.get("final", False):      # fully counted: the count is the result. sd 0.25 pt (~220 votes in a 90k-vote
        return r.now, Phi(r.now / 0.25)  # district) covers provisionals, late mail, cures, canvass fixes; generous, not fitted
    pm = anchor + k * (r.early - anchor)
    sd = sqrt((0.12 * abs(r.early - anchor)) ** 2 + 2.5 ** 2)
    # the seat's own Election Day votes (10:30pm): direct = (1 - s) x early + s x E-day so far, s = 0.25 (typical
    # big-county E-day share), weighted by how much of the seat's expected E-day vote is in
    es = 0 if np.isnan(r.eday_share) else r.eday_share / 100
    if es > 0.005 and not np.isnan(r.eday) and r.votes * es >= 1500:
        s = 0.21   # 5am: fully counted metros ran ~0.8x their 2022/24 E-day share; Harris/Dallas history 0.26-0.28
        ev, ed = r.votes * (1 - es), r.votes * es
        w = min(1.0, ed / (s / (1 - s) * ev)) ** 0.5
        direct = (1 - s) * r.early + s * r.eday
        pm = w * direct + (1 - w) * pm
        sd = sqrt((1 - w) * sd ** 2 + w * 2.0 ** 2)
    return pm, Phi(pm / sd)


t["proj"], t["pproj"] = zip(*[proj(r, KU) for _, r in t.iterrows()])
# House total: seats with data use the projection; every other seat uses the model shifted to the projected T
allhd = [r for r in rs.index if r.startswith("hd")]
held_d = {k for k, v in INC.items() if v == "D" and k.startswith("hd")}


def house_total(k, Tk):
    ps = []
    for s in allhd:
        row = t[t.seat == s]
        if len(row) and not np.isnan(row.iloc[0].early):
            ps.append(proj(row.iloc[0], k)[1])
        else:
            ps.append(Phi((rs.loc[s, "m1"] + BETA * (Tk + 6.0)) / 5.9))
    ps = np.array(ps)
    return ps.sum(), (ps * (1 - ps)).sum()


scen = [(k, KT.get(k, TP)) for k in (0.6, 0.7, 0.8, 0.9)] if KT else [(KU, TP)]
wts = np.array([Phi(((k + 0.05) - KU) / 0.1) - Phi(((k - 0.05) - KU) / 0.1) for k, _ in scen])
wts = wts / wts.sum() if wts.sum() > 0 else np.ones(len(scen)) / len(scen)
tots = [house_total(k, Tk) for k, Tk in scen]
p76 = float(sum(w * (1 - Phi((75.5 - mu) / sqrt(max(v, 1e-6)))) for w, (mu, v) in zip(wts, tots)))
mean_seats = float(sum(w * mu for w, (mu, v) in zip(wts, tots)))
print(f"=== HOUSE PROJECTION {time.strftime('%H:%M:%S')} === expected D seats {mean_seats:.1f}, P(76+) {p76:.0%}  [by k: " + ", ".join(f"{k:.2f}->{mu:.1f}" for (k, _), (mu, v) in zip(scen, tots)) + f"]  (k in use {KU:.2f}, T {TP:+.1f})")
close = t[(t.pproj.between(0.05, 0.95)) | ((t.held == "R") & (t.pproj > 0.5)) | ((t.held == "D") & (t.pproj < 0.5))].sort_values("pproj", ascending=False)
if os.environ.get("SEATS") == "senate":
    close = t.sort_values("pproj", ascending=False)
for _, r in close.iterrows():
    tag = "FLIP" if (r.held == "R" and r.pproj > 0.5) or (r.held == "D" and r.pproj < 0.5) else ""
    a, b_ = (int(x) for x in str(r.pieces).split("/"))
    if a < b_:
        tag += f" (PARTIAL: {a} of {b_} counties)"
    if r.get("final", False):
        tag += " COUNTED"
    print(f"  SEAT {r.seat:6s} {r.race[:32]:32s} {r.held}-held | early {fmt(r.early).strip():>7s} -> proj {fmt(r.proj).strip():>7s} P(D) {r.pproj:4.0%} | model was {r.p:4.0%} | ED {0 if np.isnan(r.eday_share) else r.eday_share:3.0f}% {r.pieces} {tag}")
print(f"Flagged legislative races ({len(t)}), current count vs model at T -6.0. Early-only counts will move toward R as Election Day votes come in.")
print("* = incumbent; no * on either side = open seat")
print(f"{'seat':6s} {'race':34s} {'held':4s} {'model':>7s} {'P(D)':>5s} | {'now':>7s} {'early':>7s} {'E-day':>7s} {'ED%':>5s} {'votes':>8s} pieces")
for _, r in t.iterrows():
    flip = ""
    if not np.isnan(r.now) and ((r.held == "R" and r.now > 0) or (r.held == "D" and r.now < 0)):
        flip = "  <- leading flip"
    print(f"{r.seat:6s} {r.race[:34]:34s} {r.held:4s} {fmt(r.model)} {r.p:5.0%} | {fmt(r.now)} {fmt(r.early)} {fmt(r.eday)} {r.eday_share:5.1f} {int(r.votes):8,d} {r.pieces}{flip}")
t.to_csv(EN / f"seat_report_{time.strftime('%H%M%S')}.csv", index=False)
