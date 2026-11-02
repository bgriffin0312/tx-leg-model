"""
statewide_finance.py

Two-year receipts for statewide and congressional nominees.

  State offices: TEC cover sheets (data/raw/historical/_finance_cache/tec_cover.csv),
    latest filing per filer+period, 1 Jan of the prior year to the general.
  Federal: FEC candidate summary files (data/raw/statewide/fec/weball??.zip).

Writes data/raw/statewide/finance_statewide.csv and finance_us_house.csv.
Candidates are matched on surname + party against the result names (TLC gives
surnames only). Federal money runs through Sept 30 of the election year for 2026
(Q3 reports); historical FEC figures are full-cycle, so the 2026 ratio is the
measure that matters, not the dollar level.
"""

import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "statewide"
TEC = ROOT / "data" / "raw" / "historical"
SEEK = {"governor": {"GOVERNOR"}, "lt_governor": {"LTGOVERNOR"}, "attorney_general": {"ATTYGEN"},
        "comptroller": {"COMPTROLLER", "COMPTROLLR"}, "land_commissioner": {"LANDCOMM"},
        "ag_commissioner": {"AGRICULTUR", "AGRICOMM"}, "railroad_commissioner": {"RRCOMM"}}
FEC_COLS = ["cand_id", "name", "ici", "pty_cd", "party", "receipts", "from_auth", "disb", "to_auth", "coh_bop",
            "coh_cop", "cand_contrib", "cand_loans", "other_loans", "cand_repay", "other_repay", "debts",
            "indiv", "st", "dist", "spec", "prim", "run", "gen", "gen_pct", "pac", "pty_contrib", "cvg_end",
            "indiv_refund", "cand_refund"]
# 2026 nominees by TEC filer name (legal names differ from ballot names:
# Gina Hinojosa files as Regina; Mayes Middleton as David M. Middleton II)
NOM_2026 = {"governor": ("Abbott, Greg", r"Hinojosa, (Regina|Gina)"), "lt_governor": ("Patrick, Dan", "Goodwin, Vikki"),
            "attorney_general": ("Middleton II, David", "Johnson, Nathan"), "comptroller": ("Huffines, Donald", "Eckhardt, Sarah"),
            "railroad_commissioner": ("French, Bo", "Rosenthal, Jon"), "land_commissioner": ("Buckingham", "Flores, Benjamin"),
            "ag_commissioner": ("Sheets, Nathan", "Tucker, Clayton")}


def norm(s) -> str:
    s = re.sub(r"[^a-z' -]", "", str(s).lower().replace("ñ", "n"))
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return " ".join(s.split())


def load_tec() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["reportInfoIdent", "filerIdent", "filerName", "filedDt", "periodStartDt", "periodEndDt",
            "totalContribAmount", "filerSeekOfficeCd", "filerHoldOfficeCd", "filerNameLast", "filerTypeCd"]
    c = pd.read_csv(TEC / "_finance_cache" / "tec_cover.csv", usecols=cols, dtype=str, low_memory=False, encoding_errors="replace")
    c = c[c.filerTypeCd.isin(["COH", "SPAC"])]
    c = c.sort_values(["filedDt", "reportInfoIdent"]).drop_duplicates(["filerIdent", "periodStartDt", "periodEndDt"], keep="last")
    c["amt"] = pd.to_numeric(c.totalContribAmount, errors="coerce").fillna(0)
    c["end"] = pd.to_datetime(c.periodEndDt, format="%Y%m%d", errors="coerce")
    c["last"] = c.filerNameLast.map(norm)
    sp = pd.read_csv(TEC / "_finance_cache" / "tec_spacs.csv", dtype=str, encoding_errors="replace")
    sp = sp[sp.spacPositionCd.astype(str).str.upper().str.startswith("SUPPORT")]
    return c, sp[["spacFilerIdent", "candidateFilerIdent"]].dropna().drop_duplicates()


def raised(tec, spacs, ids: set[str], lo, hi) -> tuple[float, str]:
    """Candidate account plus every specific-purpose committee formed to support them.
    The big statewide money sits in the SPACs (Texans for Greg Abbott, Beto for Texas)."""
    ids = set(ids) | set(spacs[spacs.candidateFilerIdent.isin(ids)].spacFilerIdent)
    g = tec[tec.filerIdent.isin(ids) & (tec.end >= lo) & (tec.end <= hi)]
    return float(g.amt.sum()), "; ".join(sorted(g.filerName.dropna().str[:30].unique())[:3])


def statewide(tec: pd.DataFrame, spacs: pd.DataFrame) -> pd.DataFrame:
    sw = pd.read_csv(OUT / "results_statewide.csv")
    sw = sw[sw.contested & sw.race.isin(SEEK)]
    rows = []
    coh = tec[tec.filerTypeCd == "COH"]
    for r in sw.itertuples():
        lo, hi = pd.Timestamp(r.year - 1, 1, 1), pd.Timestamp(r.year, 11, 15)
        win = coh[(coh.end >= lo) & (coh.end <= hi)]
        win = win[win.filerSeekOfficeCd.isin(SEEK[r.race]) | win.filerHoldOfficeCd.isin(SEEK[r.race])]
        rec = {"year": r.year, "race": r.race, "office": r.office}
        for party, nm in (("d", r.d_name), ("r", r.r_name)):
            last = norm(nm).split()[-1]
            ids = set(win[win["last"].str.contains(rf"\b{re.escape(last)}\b", regex=True, na=False)].filerIdent)
            rec[f"{party}_raised"], rec[f"{party}_filer"] = raised(tec, spacs, ids, lo, hi)
        rows.append(rec)
    lo, hi = pd.Timestamp(2025, 1, 1), pd.Timestamp(2026, 11, 15)
    for race, (r_nm, d_nm) in NOM_2026.items():
        rec = {"year": 2026, "race": race, "office": race}
        for party, nm in (("d", d_nm), ("r", r_nm)):
            ids = set(coh[coh.filerName.str.contains(nm, regex=True, na=False, case=False)].filerIdent)
            rec[f"{party}_raised"], rec[f"{party}_filer"] = raised(tec, spacs, ids, lo, hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def load_fec(year: int) -> pd.DataFrame:
    yy = str(year)[2:]
    with zipfile.ZipFile(OUT / "fec" / f"weball{yy}.zip") as z:
        f = pd.read_csv(z.open(z.namelist()[0]), sep="|", header=None, names=FEC_COLS, dtype=str, encoding="latin-1", quoting=3)
    f = f[(f.st == "TX")].copy()
    f["receipts"] = pd.to_numeric(f.receipts, errors="coerce").fillna(0)
    f["indiv"] = pd.to_numeric(f.indiv, errors="coerce").fillna(0)
    f["last"] = f.name.str.split(",").str[0].map(norm)
    f["office"] = f.cand_id.str[0]
    f["dnum"] = pd.to_numeric(f.dist, errors="coerce")
    f["pty"] = f.party.str[0]
    return f


def us_house() -> pd.DataFrame:
    cd = pd.read_csv(OUT / "results_us_house.csv")
    cd = cd[cd.year >= 2012]
    rows = []
    for year, g in cd.groupby("year"):
        f = load_fec(int(year))
        h = f[f.office == "H"]
        for r in g.itertuples():
            rec = {"year": year, "dist": r.dist}
            for p, nm in (("D", r.d_name), ("R", r.r_name)):
                if not isinstance(nm, str) or not nm:
                    rec[f"{p.lower()}_raised"] = 0.0
                    continue
                last = norm(nm).split()[-1]
                m = h[(h.dnum == r.dist) & (h.pty == p) & h["last"].str.contains(rf"\b{re.escape(last)}\b", regex=True, na=False)]
                rec[f"{p.lower()}_raised"] = m.receipts.sum()
            rows.append(rec)
    races = pd.read_csv(OUT / "cd_races_2026.csv")
    f = load_fec(2026)
    h = f[f.office == "H"]
    for r in races.itertuples():
        rec = {"year": 2026, "dist": r.dist}
        for p, nm in (("D", r.d_name), ("R", r.r_name)):
            if not isinstance(nm, str) or not nm:
                rec[f"{p.lower()}_raised"] = 0.0
                continue
            last = norm(re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", "", nm)).split()[-1]
            m = h[(h.pty == p) & h["last"].str.contains(rf"\b{re.escape(last)}\b", regex=True, na=False)]
            # members in a renumbered seat filed under the old district: same-district first
            same = m[m.dnum == r.dist]
            rec[f"{p.lower()}_raised"] = (same if len(same) else m).receipts.max() if len(m) else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    tec, spacs = load_tec()
    s = statewide(tec, spacs)
    s.to_csv(OUT / "finance_statewide.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 90); pd.set_option("display.max_colwidth", 34)
    print(s[s.year >= 2018].round(0).to_string(index=False))
    u = us_house()
    u.to_csv(OUT / "finance_us_house.csv", index=False)
    print(u[u.year == 2026].round(0).to_string(index=False))
    f = load_fec(2026)
    print(f[f.office == "S"].sort_values("receipts", ascending=False)[["name", "party", "receipts", "indiv", "cvg_end"]].head(6).to_string(index=False))
    print("zero-money share by year:", u.assign(z=(u.d_raised == 0) | (u.r_raised == 0)).groupby("year").z.mean().round(2).to_dict())


if __name__ == "__main__":
    main()
