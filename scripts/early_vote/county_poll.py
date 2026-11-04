"""Election-night county poller (2026-11-03): county result sites, which post before the state feed.

Clarity counties: reads detailxml.zip so early votes (EV in-person + mail) separate from Election Day.
Harris: its own JSON (ev + ab = early). For each county: early judicial margin (Supreme Court + CCA, all
races with both parties) vs the model's early-only benchmark, implied T, AG / Senate / Governor, and
the close seats. Prints a block when a county changes; composite implied T across counties read.
"""
import datetime as dt
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(r"C:\Users\Brennan\code\tx-leg-model")
OUT = ROOT / "data" / "raw" / "election_night"
OUT.mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
POLL = 60
bench = pd.read_csv(ROOT / "output" / "ev_night_benchmarks_2026.csv", index_col=0)
rs = pd.read_csv(ROOT / "data" / "raw" / "early_vote" / "race_shift.csv").set_index("race")
CLARITY = {"TARRANT": "Tarrant/127297", "DALLAS": "Dallas/127396", "BEXAR": "Bexar/127291"}
for a in sys.argv[1:]:
    k, v = a.split("=")
    CLARITY[k.upper().replace("_", " ")] = v
HARRIS = "https://appfiles.harrisvotes.com/harrisvotes/prd/Data/1126/resultsbyballot.json?nocache={}"
BIG9 = ["HARRIS", "DALLAS", "TARRANT", "BEXAR", "TRAVIS", "COLLIN", "DENTON", "FORT BEND", "WILLIAMSON"]


def log(s):
    print(s, flush=True)


def fmt(m):
    return "n/a" if m is None or np.isnan(m) else (f"D+{m:.1f}" if m >= 0 else f"R+{-m:.1f}")


def mm(d, r):
    return (d - r) / (d + r) * 100 if d + r else np.nan


def seat_key(name):
    m = re.search(r"(State Representative|State Senator|United States Representative).*?(\d+)", name, re.I)
    if not m:
        return None
    return {"state representative": "hd", "state senator": "sd", "united states representative": "tx"}[m.group(1).lower()] + m.group(2)


def clarity(path):
    base = f"https://results.enr.clarityelections.com/TX/{path}"
    ver = requests.get(f"{base}/current_ver.txt", headers=HDR, timeout=30).text.strip()
    if not ver.isdigit():
        raise TransientError("bad version text")
    return ver, base


class TransientError(Exception):
    pass


def parse_clarity(base, ver):
    z = requests.get(f"{base}/{ver}/reports/detailxml.zip", headers=HDR, timeout=120).content
    x = zipfile.ZipFile(io.BytesIO(z)).read("detail.xml").decode("utf-8", "replace")
    ts = (re.findall(r"<Timestamp>([^<]*)</Timestamp>", x) or [""])[0]
    # Bexar filed its early in-person votes under "Election Day" at 7:36pm; before 8pm nothing is Election Day
    try:
        all_early = dt.datetime.strptime(ts.replace(" CST", "").strip(), "%m/%d/%Y %I:%M:%S %p") < dt.datetime(2026, 11, 3, 20, 0)
    except ValueError:
        all_early = False
    out = {}
    for c in re.finditer(r'<Contest key="\d+" text="([^"]+)"[^>]*>(.*?)</Contest>', x, re.S):
        name = c.group(1)
        d = {"DEM": [0, 0, 0], "REP": [0, 0, 0]}   # early, eday, total
        for ch in re.finditer(r'<Choice key="\d+" text="[^"]*" party="([^"]*)" totalVotes="(\d+)">(.*?)</Choice>', c.group(2), re.S):
            p = ch.group(1)
            if p not in d:
                continue
            for vt, v in re.findall(r'<VoteType name="([^"]+)" votes="(\d+)"', ch.group(3)):
                if re.search(r"overvote|undervote", vt, re.I):
                    continue
                if re.match(r"election", vt, re.I) and not all_early:   # "Election Day" (Dallas, Tarrant) or "Election" (Denton)
                    d[p][1] += int(v)
                else:
                    d[p][0] += int(v)
            d[p][2] += int(ch.group(2))
        out[name] = d
    return ts, out


def parse_harris(js):
    out = {}
    for c in js.get("contests", []):
        d = {"DEM": [0, 0, 0], "REP": [0, 0, 0]}
        for i, nm in enumerate(c.get("ca", [])):
            p = "DEM" if "(DEM)" in nm else "REP" if "(REP)" in nm else None
            if p:
                d[p][0] += c["ev"][i] + c["ab"][i]
                d[p][1] += c["ed"][i]
                d[p][2] += c["tv"][i]
        out[c["t"]] = d
    return js.get("dt", ""), out


def read(county, ts, races):
    jd = jr = jde = jre = 0
    lines = []
    for name, d in races.items():
        if re.search(r"Supreme Court|Criminal Appeals", name, re.I) and d["DEM"][2] + d["DEM"][0] > 0 and d["REP"][2] + d["REP"][0] > 0:
            jd += d["DEM"][0]; jr += d["REP"][0]; jde += d["DEM"][1]; jre += d["REP"][1]
    jm = mm(jd, jr)
    res = {"county": county, "n": jd + jr, "jm": jm, "eday_votes": jde + jre}
    b = bench.loc[county] if county in bench.index else None
    head = f"  {county} [{ts}] judicial early {fmt(jm)} on {jd + jr:,}"
    if b is not None and not np.isnan(jm):
        base_c = b["lean24"] + b["cshift"] + (b["ev_T-6.0"] - b["final_T-6.0"])
        res["T"] = jm - base_c
        head += f" | bench {fmt(b['ev_T-6.0'])} dev {jm - b['ev_T-6.0']:+.1f} | implied T {res['T']:+.1f}"
    if jde + jre:
        head += f" | Election Day votes in: {jde + jre:,} (judicial ED margin {fmt(mm(jde, jre))})"
    lines.append(head)
    parts = []
    for lab, key in (("Sen", "united states senator"), ("Gov", "governor"), ("AG", "attorney general")):
        for name, d in races.items():
            if name.lower().strip() == key:
                parts.append(f"{lab} {fmt(mm(d['DEM'][0], d['REP'][0]))}")
    if parts:
        lines.append("    early: " + " | ".join(parts))
    seats = []
    for name, d in races.items():
        k = seat_key(name)
        if k and k in rs.index and 0.03 < rs.loc[k, "p1"] < 0.97 and d["DEM"][0] + d["REP"][0] > 0:
            seats.append(f"{k} {fmt(mm(d['DEM'][0], d['REP'][0]))} (model {fmt(rs.loc[k, 'm1'])}, {rs.loc[k, 'p1']:.0%})")
    if seats:
        lines.append("    seats early: " + "; ".join(sorted(seats)))
    return res, "\n".join(lines)


def main():
    last = {}
    reads = {}
    errs = {}
    log(f"county poller start {dt.datetime.now():%H:%M:%S}: {', '.join(CLARITY)} + HARRIS")
    while True:
        changed = []
        for county, path in CLARITY.items():
            try:
                ver, base = clarity(path)
                if last.get(county) == ver:
                    continue
                ts, races = parse_clarity(base, ver)
                res, txt = read(county, ts, races)
                (OUT / f"clarity_{county.replace(' ', '_')}_{ver}.json").write_text(json.dumps(races))
                last[county] = ver
                if res["n"] > 0:
                    reads[county] = res
                    changed.append(txt)
            except (TransientError, zipfile.BadZipFile, requests.RequestException) as ex:
                errs[county] = errs.get(county, 0) + 1
                if errs[county] % 15 == 0:
                    log(f"  {county}: {errs[county]} transient fetch failures so far (server 502s); still retrying")
            except Exception as ex:
                log(f"  ERROR {county}: {type(ex).__name__} {str(ex)[:150]}")
        try:
            js = requests.get(HARRIS.format(int(time.time() // 60 * 60000)), headers=HDR, timeout=30).json()
            if js.get("dt") != last.get("HARRIS"):
                ts, races = parse_harris(js)
                res, txt = read("HARRIS", ts, races)
                (OUT / f"harris_{dt.datetime.now():%H%M%S}.json").write_text(json.dumps(js))
                last["HARRIS"] = js.get("dt")
                if res["n"] > 0:
                    reads["HARRIS"] = res
                    changed.append(txt)
        except Exception as ex:
            log(f"  ERROR HARRIS: {type(ex).__name__} {str(ex)[:150]}")
        if changed:
            r = pd.DataFrame(reads.values()).set_index("county")
            r = r.dropna(subset=["T"]) if "T" in r else r.iloc[0:0]
            comp = f"implied T {np.average(r['T'], weights=r['n']):+.1f} over {', '.join(r.index)} ({int(r['n'].sum()):,} early judicial votes)" if len(r) else "n/a"
            log(f"=== COUNTY UPDATE {dt.datetime.now():%H:%M:%S} ===\n" + "\n".join(changed) + f"\n  COMPOSITE {comp}  (model center T -6.0; statewide version -4.8)")
            pd.DataFrame(reads.values()).assign(at=dt.datetime.now().strftime("%H:%M:%S")).to_csv(OUT / "county_poll_reads.csv", mode="a", header=not (OUT / "county_poll_reads.csv").exists(), index=False)
        time.sleep(POLL)


if __name__ == "__main__":
    main()
