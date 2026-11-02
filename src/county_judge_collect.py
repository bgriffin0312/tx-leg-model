"""
county_judge_collect.py

Contested county-judge and district-attorney general elections, midterms 2014/2018/2022,
from the OpenElections Texas county precinct files (county races are not in the TLC or
SOS returns). Races the files lack, and incumbency, are filled in county_judge_model.py.

Output: data/raw/county_judge/local_races_raw.csv
  year, county, office, d_name, r_name, d_votes, r_votes, other_votes, source

Usage: python src/county_judge_collect.py
"""
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "county_judge"
OE = RAW / "oe"
API = "https://api.github.com/repos/openelections/openelections-data-tx/contents/{y}/counties"
BASE = "https://raw.githubusercontent.com/openelections/openelections-data-tx/master/{y}/counties/{f}"
UA = {"User-Agent": "tx-leg-model/1.0"}
CYCLES = (2014, 2018, 2022)
JUNK = r"under|over|registered|ballots cast|write|total"


def download() -> None:
    OE.mkdir(parents=True, exist_ok=True)
    for y in CYCLES:
        items = requests.get(API.format(y=y), headers=UA, timeout=60).json()
        names = [x["name"] for x in items if re.search(rf"^{y}\d{{4}}__tx__general__[a-z_]+__precinct\.csv$", x["name"])]
        for f in names:
            p = OE / f
            if p.exists() and p.stat().st_size:
                continue
            r = requests.get(BASE.format(y=y, f=f), headers=UA, timeout=120)
            if r.ok:
                p.write_bytes(r.content)
            time.sleep(0.05)
        print(y, len(names), "general county files")


def office_of(s: str) -> str | None:
    t = str(s).lower().strip()
    if re.search(r"court|probate|unexpired|precinct|constable|district judge|municipal|civil|family|justice", t):
        return None
    if re.search(r"\bcounty judge\b", t):
        return "county_judge"
    if re.search(r"^(criminal )?district attorney|\bdistrict attorney\b", t):
        return "district_attorney"
    return None


def party_of(p, name: str) -> str:
    p = str(p).upper().strip()
    if p in ("DEM", "D", "DEMOCRATIC"):
        return "D"
    if p in ("REP", "R", "REPUBLICAN"):
        return "R"
    return "O"


def tally() -> pd.DataFrame:
    rows = []
    for f in sorted(OE.glob("*__tx__general__*__precinct.csv")):
        year = int(f.name[:4])
        county = f.name.split("__")[3].replace("_", " ").title()
        try:
            d = pd.read_csv(f, low_memory=False, dtype=str, encoding_errors="replace")
        except Exception as e:
            print("unreadable", f.name, e)
            continue
        d.columns = [c.lower().strip() for c in d.columns]
        if not {"office", "candidate", "votes"} <= set(d.columns):
            continue
        d["off"] = d.office.astype(str).map(office_of)
        d = d[d.off.notna()].copy()
        if d.empty:
            continue
        if "party" not in d:
            d["party"] = ""
        d["votes"] = pd.to_numeric(d.votes.astype(str).str.replace(",", ""), errors="coerce").fillna(0)
        d = d[~d.candidate.astype(str).str.contains(JUNK, case=False, regex=True)]
        d["p"] = [party_of(a, b) for a, b in zip(d.party, d.candidate)]
        for off, g in d.groupby("off"):
            t = g.groupby(["p", "candidate"]).votes.sum().reset_index().sort_values("votes", ascending=False)
            dd, rr, oo = t[t.p == "D"], t[t.p == "R"], t[t.p == "O"]
            rows.append({"year": year, "county": county, "office": off,
                         "d_name": dd.candidate.iloc[0] if len(dd) else "",
                         "r_name": rr.candidate.iloc[0] if len(rr) else "",
                         "d_votes": dd.votes.iloc[0] if len(dd) else 0,
                         "r_votes": rr.votes.iloc[0] if len(rr) else 0,
                         "other_votes": oo.votes.sum(), "source": "openelections"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    download()
    t = tally()
    t.to_csv(RAW / "local_races_raw.csv", index=False)
    c = t[(t.d_votes > 0) & (t.r_votes > 0)]
    print(len(t), "contests,", len(c), "with D and R;")
    print(c.groupby(["year", "office"]).size().to_string())
