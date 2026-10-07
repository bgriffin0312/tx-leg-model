"""
collect_education_by_district.py

Educational attainment (population 25+) by race by Texas legislative district,
for the education splits in the correlated group-error layer and the staged
midterm education term: white non-Hispanic (C15002H), Hispanic (C15002I),
Black alone (C15002B; includes Hispanic Black, a small overlap).

Source: ACS 5-year TABLE-BASED SUMMARY FILE, tables C15002H/I/B (Sex by Educational
Attainment by race, 25+). Keyless bulk files on www2:
    https://www2.census.gov/programs-surveys/acs/summary_file/{year}/table-based-SF/
        data/5YRData/acsdt5y{year}-c15002{h,i,b}.dat
(The api.census.gov route now requires a key; this one does not. The format
starts with the 2021 release; earlier years 404.)

District vintages, verified 2026-10-07 by correlating each file's white 25+
counts with the CVAP white counts on each set of lines:
    2021 ACS (2017-21) -> pre-2022 lines (PlanH2100/S172 era)  r = 0.988
    2024 ACS (2020-24) -> current lines (PlanH2316/S2168)       r = 0.994
So 2021 covers races in 2014/2018/2020 and 2024 covers 2022/2024/2026.

Output: data/raw/tx_education_{chamber}_acs{year}.csv
  district, {group}_25p, {group}_ba_plus, {group}_college_rate for
  group in white_nh / hispanic / black (white_college_rate kept as the
  white_nh rate's name, which earlier code reads)

Usage:
  python src/collect_education_by_district.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = RAW / "_acs_cache"
URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/{y}/table-based-SF/"
       "data/5YRData/acsdt5y{y}-{t}.dat")
TABLES = {"white_nh": "c15002h", "hispanic": "c15002i", "black": "c15002b"}
YEARS = (2021, 2024)
# GEO_ID summary levels: 620 = SLD lower (House), 610 = SLD upper (Senate).
CHAMBERS = {"house": "620", "senate": "610"}


def fetch(year: int, table: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"acsdt5y{year}-{table}.dat"
    if not path.exists():
        r = requests.get(URL.format(y=year, t=table), timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return pd.read_csv(path, sep="|", dtype={"GEO_ID": str})


def main():
    for year in YEARS:
        tabs = {g: fetch(year, t) for g, t in TABLES.items()}
        for chamber, sumlev in CHAMBERS.items():
            out = None
            for g, t in TABLES.items():
                x = tabs[g]
                x = x[x.GEO_ID.str.match(rf"^{sumlev}[LU]\d00US48\d{{3}}$")].copy()
                p = t.upper()
                # E001 total; E006 male BA+; E011 female BA+ (C15002x layout).
                part = pd.DataFrame({
                    "district": x.GEO_ID.str[-3:].astype(int),
                    f"{g}_25p": x[f"{p}_E001"].astype(int),
                    f"{g}_ba_plus": (x[f"{p}_E006"] + x[f"{p}_E011"]).astype(int),
                })
                part[f"{g}_college_rate"] = (part[f"{g}_ba_plus"] / part[f"{g}_25p"]).round(4)
                out = part if out is None else out.merge(part, on="district")
            out = out.rename(columns={"white_nh_college_rate": "white_college_rate"})
            out = out.sort_values("district")
            dest = RAW / f"tx_education_{chamber}_acs{year}.csv"
            out.to_csv(dest, index=False)
            rates = ", ".join(f"{g} {out[f'{g}_ba_plus'].sum() / out[f'{g}_25p'].sum():.3f}" for g in TABLES)
            print(f"{year} {chamber}: {len(out)} districts, statewide BA+ rate {rates} -> {dest.name}")


if __name__ == "__main__":
    main()
