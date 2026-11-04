"""Actual early-vote vs final margin gap in past generals, from Clarity's archived detail files.

    python scripts/early_vote/clarity_ev_gap_history.py Denton/92173:2018 Denton/115757:2022 Denton/122426:2024
Judicial = all Supreme Court + Court of Criminal Appeals contests with both parties. Early = every vote
type except Election Day. Gap = early margin minus final margin (D minus R), so + means the early vote
ran bluer than the final count.
"""
import io
import re
import sys
import time
import zipfile

import requests

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HDR, timeout=120)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(3 + 2 * i)
    raise RuntimeError(f"failed {url}")


def m(d, r):
    return (d - r) / (d + r) * 100 if d + r else float("nan")


for arg in sys.argv[1:]:
    path, year = arg.split(":")
    base = f"https://results.enr.clarityelections.com/TX/{path}"
    ver = fetch(f"{base}/current_ver.txt").text.strip()
    x = zipfile.ZipFile(io.BytesIO(fetch(f"{base}/{ver}/reports/detailxml.zip").content)).read("detail.xml").decode("utf-8", "replace")
    acc = {}
    for c in re.finditer(r'<Contest key="\d+" text="([^"]+)"[^>]*>(.*?)</Contest>', x, re.S):
        name = c.group(1)
        grp = "judicial" if re.search(r"Supreme Court|Criminal Appeals", name, re.I) else ("senate" if re.match(r"(U\.?\s?S\.?|United States) Senator", name, re.I) else ("governor" if name.strip().lower() == "governor" else ("president" if re.match(r"President", name, re.I) else None)))
        if not grp:
            continue
        d = {"DEM": [0, 0], "REP": [0, 0]}
        for ch in re.finditer(r'<Choice key="\d+" text="[^"]*" party="([^"]*)" totalVotes="(\d+)">(.*?)</Choice>', c.group(2), re.S):
            p = {"DEM": "DEM", "REP": "REP", "D": "DEM", "R": "REP"}.get(ch.group(1).upper())
            if not p:
                continue
            for vt, v in re.findall(r'<VoteType name="([^"]+)" votes="(\d+)"', ch.group(3)):
                if re.search(r"overvote|undervote", vt, re.I):
                    continue
                d[p][1 if re.match(r"election", vt, re.I) else 0] += int(v)
        if d["DEM"][0] + d["DEM"][1] == 0 or d["REP"][0] + d["REP"][1] == 0:
            continue
        a = acc.setdefault(grp, {"De": 0, "Re": 0, "Dd": 0, "Rd": 0})
        a["De"] += d["DEM"][0]; a["Re"] += d["REP"][0]; a["Dd"] += d["DEM"][1]; a["Rd"] += d["REP"][1]
    vts = sorted(set(re.findall(r'<VoteType name="([^"]+)"', x)))
    print(f"\n{path} {year} (vote types: {vts})")
    for g, a in acc.items():
        early = m(a["De"], a["Re"])
        eday = m(a["Dd"], a["Rd"])
        final = m(a["De"] + a["Dd"], a["Re"] + a["Rd"])
        share = (a["Dd"] + a["Rd"]) / (a["De"] + a["Re"] + a["Dd"] + a["Rd"]) * 100
        print(f"  {g:9s} early {early:+6.1f}  eday {eday:+6.1f}  final {final:+6.1f}  | early minus final {early - final:+5.1f} | ED share {share:4.1f}%")
