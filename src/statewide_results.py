"""
statewide_results.py

Build the results base for the statewide + congressional model from TLC VTD
returns (data/raw/_capitol_data_cache), which carry every general-election
contest 2002-2024 with party and incumbent flags:

  data/raw/statewide/results_statewide.csv   one row per statewide contest-year
  data/raw/statewide/results_us_house.csv    one row per TX U.S. House contest-year,
                                             with the same-VTD presidential and
                                             low-salience vote in that district
  data/raw/statewide/cd_baseline_2026.csv    PlanC2333 districts: 2024/2022 statewide
                                             contests overlaid area-weighted, plus
                                             TLC r202 demographics

A U.S. House contest's district is read from the returns themselves: a VTD
belongs to the district whose contest it voted in, so no historical plan files
are needed. (Only 6-16 of ~8,500 VTDs vote in two districts; those are shared
by each contest's own vote.)
"""

import io
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "_capitol_data_cache"
OUT = ROOT / "data" / "raw" / "statewide"
ZIPS = {y: "2016-general-vtds-election-data.zip" for y in (2002, 2004, 2006, 2008, 2010)}
ZIPS.update({y: "2024-general-vtds-election-data.zip" for y in range(2012, 2026, 2)})

OFFICE_MAP = [
    (r"^President", "president"), (r"^U\.S\. Sen", "us_senate"), (r"^Governor", "governor"),
    (r"^Lt\.? ?Gov", "lt_governor"), (r"^Attorney Gen", "attorney_general"),
    (r"^Comptroller", "comptroller"), (r"^Land Comm", "land_commissioner"),
    (r"^Ag(riculture)? Comm", "ag_commissioner"), (r"^RR Comm", "railroad_commissioner"),
    (r"^Sup Ct|^CCA", "judicial"),
]


def classify(office: str) -> str | None:
    for pat, name in OFFICE_MAP:
        if re.match(pat, office):
            return name
    return None


def load(year: int) -> pd.DataFrame:
    with zipfile.ZipFile(CACHE / ZIPS[year]) as z:
        raw = z.read(f"{year}_General_Election_Returns.csv")
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False,
                     dtype={"cntyvtd": str, "Office": str, "Name": str})
    df["Votes"] = pd.to_numeric(df.Votes, errors="coerce").fillna(0)
    df["Office"] = df.Office.astype(str).str.strip()
    df["cntyvtd"] = df.cntyvtd.astype(str).str.strip()
    return df


def statewide_rows(df: pd.DataFrame, year: int) -> list[dict]:
    rows = []
    for office, g in df.groupby("Office"):
        race = classify(office)
        if race is None:
            continue
        t = g.groupby(["Party", "Name", "Incumbent"], dropna=False).Votes.sum().reset_index()
        d, r = t[t.Party == "D"], t[t.Party == "R"]
        total = t.Votes.sum()
        if len(d) != 1 or len(r) != 1:
            # unopposed-by-a-major-party statewide race: kept out of the margin
            # work but recorded so the coverage audit can count it
            rows.append({"year": year, "office": office, "race": race, "contested": False,
                         "d_name": d.Name.iloc[0] if len(d) else "", "r_name": r.Name.iloc[0] if len(r) else "",
                         "d_votes": d.Votes.sum(), "r_votes": r.Votes.sum(), "total_votes": total,
                         "d_inc": False, "r_inc": False})
            continue
        rows.append({"year": year, "office": office, "race": race, "contested": True,
                     "d_name": d.Name.iloc[0], "r_name": r.Name.iloc[0],
                     "d_votes": d.Votes.iloc[0], "r_votes": r.Votes.iloc[0], "total_votes": total,
                     "d_inc": str(d.Incumbent.iloc[0]).upper().startswith("Y"),
                     "r_inc": str(r.Incumbent.iloc[0]).upper().startswith("Y")})
    return rows


def house_rows(df: pd.DataFrame, year: int, comps: dict) -> list[dict]:
    """Each TX U.S. House contest with the president and low-salience composite
    measured on the same VTDs. comps may come from earlier years: VTD keys are
    mostly stable, and the missed share is recorded per contest."""
    cd = df[df.Office.str.match(r"^U\.S\. Rep")].copy()
    cd["dist"] = cd.Office.str.extract(r"(\d+)").astype(int)
    share = cd.groupby(["cntyvtd", "dist"]).Votes.sum().reset_index()
    share["w"] = share.Votes / share.groupby("cntyvtd").Votes.transform("sum")
    rows = []
    for dist, g in cd.groupby("dist"):
        t = g.groupby(["Party", "Name", "Incumbent"], dropna=False).Votes.sum().reset_index()
        d, r = t[t.Party == "D"], t[t.Party == "R"]
        rec = {"year": year, "dist": int(dist), "total_votes": t.Votes.sum(),
               "d_name": d.Name.iloc[0] if len(d) else "", "r_name": r.Name.iloc[0] if len(r) else "",
               "d_votes": d.Votes.sum(), "r_votes": r.Votes.sum(), "n_d": len(d), "n_r": len(r),
               "d_inc": bool(d.Incumbent.astype(str).str.upper().str.startswith("Y").any()),
               "r_inc": bool(r.Incumbent.astype(str).str.upper().str.startswith("Y").any())}
        w = share[share.dist == dist].set_index("cntyvtd").w
        for key, vt in comps.items():
            hit = vt.reindex(w.index)
            m = hit.fillna(0).mul(w, axis=0).sum()
            tot = m.get("D", 0) + m.get("R", 0)
            rec[f"{key}_margin"] = (m.get("D", 0) - m.get("R", 0)) / tot * 100 if tot else np.nan
            rec[f"{key}_vtd_hit"] = float(hit.iloc[:, 0].notna().mean())
        rows.append(rec)
    return rows


def composites(df: pd.DataFrame, offices: list[str], label: str):
    """VTD D/R votes for one contest, or the equal-weight mean of several."""
    sel = df[df.Office.isin(offices) & df.Party.isin(["D", "R"])]
    if sel.empty:
        return None
    w = sel.groupby(["cntyvtd", "Party"]).Votes.sum().unstack(fill_value=0) / len(offices)
    return w


def vtd_assignment(df24: pd.DataFrame) -> pd.DataFrame:
    """Exact 2024-VTD -> PlanC2333 assignment from TLC Red-110 (map/r110.xls), with
    split VTDs shared by the VAP of each piece. An area overlay put TX-32 about 5
    points and TX-9 about 3 points off the published Trump margins, because split
    precincts in dense suburbs hold people unevenly."""
    import xlrd
    fips = df24.groupby("County").FIPS.first().to_dict()
    fips = {k.strip().lower(): int(v) for k, v in fips.items()}
    with open(os.devnull, "w") as null:
        sh = xlrd.open_workbook(OUT / "map" / "r110.xls", logfile=null).sheet_by_index(0)
    rows, dist, county = [], None, None
    for r in range(sh.nrows):
        v = [x for x in sh.row_values(r) if x != ""]
        if not v:
            continue
        first = str(v[0]).strip()
        m = re.match(r"^DISTRICT (\d+)$", first)
        if m:
            dist = int(m.group(1))
            continue
        m = re.match(r"^(.+?) \(\d+(\.\d+)? ?%\)$", first)
        if m and len(v) == 1:
            county = m.group(1).strip().lower()
            continue
        if "Total:" in v and first != "Total:" and county is not None and dist is not None:
            i = v.index("Total:")
            vap = sh.row_values(r + 1)
            vap = [x for x in vap if x != ""]
            rows.append({"cntyvtd": f"{fips[county]}{first}", "dist": dist,
                         "pop": float(v[i + 1]), "vap": float(vap[1]) if len(vap) > 1 else 0.0})
    a = pd.DataFrame(rows)
    a["tot"] = a.groupby("cntyvtd").vap.transform("sum")
    a["w"] = np.where(a.tot > 0, a.vap / a.tot.replace(0, np.nan),
                      a["pop"] / a.groupby("cntyvtd")["pop"].transform("sum").replace(0, np.nan))
    return a.dropna(subset=["w"])[lambda d: d.w > 0][["cntyvtd", "dist", "w"]]


def overlay_2026(df24: pd.DataFrame, df22: pd.DataFrame) -> pd.DataFrame:
    """2024 and 2022 statewide contests on PlanC2333, using the exact VTD assignment.
    2022 returns are on 2022 VTD keys; unmatched keys fall back to county shares and
    the share of vote affected is printed."""
    pieces = vtd_assignment(df24)
    print(f"  VTD pieces {len(pieces)}, split VTDs {(pieces.groupby('cntyvtd').dist.nunique() > 1).sum()}")
    out = pd.DataFrame({"dist": sorted(pieces.dist.unique())})
    for year, df in ((2024, df24), (2022, df22)):
        for office, g in df[df.Party.isin(["D", "R"])].groupby("Office"):
            if classify(office) is None or g.Party.nunique() < 2:
                continue
            v = g.groupby(["cntyvtd", "Party"]).Votes.sum().unstack(fill_value=0).reset_index()
            m = v.merge(pieces, on="cntyvtd", how="left")
            miss = m[m.dist.isna()].drop(columns=["dist", "w"])
            lost = miss[["D", "R"]].sum().sum() / v[["D", "R"]].sum().sum()
            if lost > 0.002:
                print(f"  {year} {office}: {lost:.2%} of vote matched by county share only")
            m = m.dropna(subset=["dist"])
            if len(miss):
                # renumbered precincts: spread over the county's districts in proportion
                # to that county's matched vote
                m["tv"] = (m.D + m.R) * m.w
                m["fips"] = m.cntyvtd.str[:-4]
                cs = m.groupby(["fips", "dist"]).tv.sum().reset_index()
                cs["w"] = cs.tv / cs.groupby("fips").tv.transform("sum")
                ex = (miss.assign(fips=miss.cntyvtd.str[:-4]).groupby("fips")[["D", "R"]].sum().reset_index()
                      .merge(cs[["fips", "dist", "w"]], on="fips", how="inner"))
                m = pd.concat([m, ex], ignore_index=True)
            m["dv"], m["rv"] = m.D * m.w, m.R * m.w
            a = m.groupby("dist")[["dv", "rv"]].sum()
            slug = re.sub(r"[^a-z0-9]+", "_", office.lower()).strip("_")
            out[f"m{year}_{slug}"] = out.dist.map((a.dv - a.rv) / (a.dv + a.rv) * 100)
            out[f"tv{year}_{slug}"] = out.dist.map(a.dv + a.rv)
    return out


def parse_r202() -> pd.DataFrame:
    import pdfplumber
    rows = []
    with pdfplumber.open(OUT / "map" / "r202b.pdf") as p:
        text = "\n".join(pg.extract_text() or "" for pg in p.pages)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        m = re.match(r"^(\d{1,2}) -?\d+ Total: ", ln)
        if not m or i + 1 >= len(lines):
            continue
        v = re.match(r"^-?\d+\.\d+% VAP: ([\d,]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", lines[i + 1])
        if v:
            rows.append({"dist": int(m.group(1)), "anglo_vap": float(v.group(2)), "asian_vap": float(v.group(4)),
                         "black_vap": float(v.group(5)), "hisp_vap": float(v.group(6))})
    return pd.DataFrame(rows)


def main():
    frames = {}
    sw, cd = [], []
    vt_all = {}
    for year in sorted(ZIPS):
        df = load(year)
        frames[year] = df
        sw += statewide_rows(df, year)
        comps = {}
        pres = composites(df, [o for o in df.Office.unique() if o.startswith("President")], "pres")
        # low-salience composite: every contested Supreme Court + CCA race that year
        jud = [o for o in df.Office.unique() if re.match(r"^(Sup Ct|CCA)", o)
               and df[df.Office == o].Party.isin(["D"]).any() and df[df.Office == o].Party.isin(["R"]).any()]
        vt_jud = composites(df, jud, "jud") if jud else None
        if pres is not None:
            comps["pres"] = pres
        if vt_jud is not None:
            comps["jud"] = vt_jud
        # the baseline a forecast could actually have used: the last presidential
        # year and the last cycle before this one
        if year - 4 in vt_all or year - 2 in vt_all:
            py = year - 2 if year % 4 == 2 else year - 4
            if py in vt_all:
                comps["prevpres"] = vt_all[py]["pres"]
                comps["prevjud"] = vt_all[py]["jud"]
        vt_all[year] = {"pres": pres, "jud": vt_jud}
        cd += house_rows(df, year, comps)
        print(f"{year}: {df.Office.nunique()} offices, judicial contests {len(jud)}")
    sw = pd.DataFrame(sw)
    sw["d_margin"] = (sw.d_votes - sw.r_votes) / (sw.d_votes + sw.r_votes) * 100
    sw.to_csv(OUT / "results_statewide.csv", index=False)
    cd = pd.DataFrame(cd)
    cd["contested"] = (cd.d_votes > 0) & (cd.r_votes > 0)
    cd["d_margin"] = (cd.d_votes - cd.r_votes) / (cd.d_votes + cd.r_votes) * 100
    cd.to_csv(OUT / "results_us_house.csv", index=False)
    print(f"statewide rows {len(sw)}; us house rows {len(cd)}, contested {cd.contested.sum()}")

    base = overlay_2026(frames[2024], frames[2022]).merge(parse_r202(), on="dist", how="left")
    base.to_csv(OUT / "cd_baseline_2026.csv", index=False)
    print(f"baseline: {len(base)} districts, {base.anglo_vap.notna().sum()} with demographics")


if __name__ == "__main__":
    main()
