"""
analyze_early_ie_2026.py

Extended 2026 IE snapshot using all PACs with district-specific activity
in the April 6, 2026 TEC data.

Complements analyze_early_ie_signal.py (which uses only the 8 historical PACs).
This script adds PACs active only in 2026 with clear directional descriptions.
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
CACHE     = ROOT / "data" / "raw" / "historical" / "_finance_cache"
DATA_PROC = ROOT / "data" / "processed"

HD_PATS = [
    re.compile(r"\bH\.?D\.?\s*[-#]?\s*(\d{1,3})\b", re.I),
    re.compile(r"\bHOUSE[-\s]+DIST(?:RICT)?\s+(\d{1,3})\b", re.I),
]
SD_PATS = [
    re.compile(r"\bS\.?D\.?\s*[-#]?\s*(\d{1,2})\b", re.I),
    re.compile(r"\bSENATE[-\s]+DIST(?:RICT)?\s+(\d{1,2})\b", re.I),
]

OPPOSE_PAT  = re.compile(r"\boppos", re.I)
SUPPORT_PAT = re.compile(
    r"\b(support|in.?kind|inkind|direct\s+mail|contrib|donation|"
    r"block\s+walk|streaming|digital\s+ad|media\s+place|mail\b|polling)", re.I
)

# AFC Victory Fund uses explicit format: "IE - Mail - Oppose - Name - HD N"
AFC_IE_PAT = re.compile(
    r"IE\s*-\s*\S+\s*-\s*(Support|Oppose)\s*-\s*.+?(?:HD|SD)\s*(\d+)", re.I
)
# First Amendment Alliance: "OPPOSE OF NAME (INCUMBENT - HD N)"
FAA_PAT = re.compile(r"\b(SUPPORT|OPPOSE)\s+OF\s+.+?HD\s*(\d+)", re.I)

# Extended PAC registry for 2026
# dir: "R_favor"|"D_favor"|"parse"
# context: "general"|"R_primary"|"mixed"
PACS = {
    # Clear direction (partisan PACs)
    "00055005": {"name": "House Dem Campaign Cmte",        "dir": "D_favor",  "ctx": "general"},
    "00058081": {"name": "TX Rep Legislative Campaign",    "dir": "R_favor",  "ctx": "general"},
    "00084976": {"name": "RSLC Grassroots",                "dir": "R_favor",  "ctx": "general"},
    "00068897": {"name": "Battleground Texas",             "dir": "D_favor",  "ctx": "general"},
    "00088252": {"name": "Texans United/Conserv Majority", "dir": "R_favor",  "ctx": "R_primary"},
    "00090684": {"name": "Forge the Future (Meta)",        "dir": "R_favor",  "ctx": "general"},
    "00090254": {"name": "Strong Borders Action",          "dir": "R_favor",  "ctx": "R_primary"},
    "00090717": {"name": "Protecting Texas Children",      "dir": "R_favor",  "ctx": "R_primary"},
    "00086801": {"name": "Tarrant County Patriots",        "dir": "R_favor",  "ctx": "R_primary"},
    "00085365": {"name": "Protect and Serve Texas",        "dir": "R_favor",  "ctx": "R_primary"},
    "00086923": {"name": "Coalition for Working Families", "dir": "D_favor",  "ctx": "general"},
    "00031918": {"name": "Education Austin PAC",           "dir": "D_favor",  "ctx": "general"},
    "00016529": {"name": "Ironworkers State COPE",         "dir": "D_favor",  "ctx": "general"},
    "00016346": {"name": "TX State Teachers Assn PAC",     "dir": "D_favor",  "ctx": "general"},
    "00039023": {"name": "Travis County Republican Party", "dir": "R_favor",  "ctx": "general"},
    "00015617": {"name": "Austin Fire Fighters PAC",       "dir": "D_favor",  "ctx": "general"},
    "00070062": {"name": "Liberal Austin Democrats",       "dir": "D_favor",  "ctx": "general"},
    "00018807": {"name": "State COPE Fund",                "dir": "D_favor",  "ctx": "general"},
    "00085511": {"name": "Amarillo Firefighters PAC",      "dir": "D_favor",  "ctx": "general"},
    "00089532": {"name": "Texans for Property Rights",     "dir": "R_favor",  "ctx": "R_primary"},
    "00090781": {"name": "Citizens for a Secure Texas",    "dir": "R_favor",  "ctx": "general"},
    # Parsed from description
    "00088032": {"name": "AFC Victory Fund",               "dir": "parse",    "ctx": "general"},
    "00090668": {"name": "First Amendment Alliance",       "dir": "parse",    "ctx": "R_primary"},
    "00028135": {"name": "Texans for Lawsuit Reform",      "dir": "parse",    "ctx": "general"},
    "00015487": {"name": "TX REALTORS PAC",                "dir": "parse",    "ctx": "mixed"},
    "00015666": {"name": "TX Trial Lawyers PAC",           "dir": "parse",    "ctx": "mixed"},
    "00016623": {"name": "TX Farm Bureau AGFUND",          "dir": "parse",    "ctx": "general"},
    "00040966": {"name": "HillCo PAC",                     "dir": "parse",    "ctx": "general"},
    "00051076": {"name": "TX Alliance for Life",           "dir": "parse",    "ctx": "R_primary"},
    "00080619": {"name": "Charter Schools Now PAC",        "dir": "parse",    "ctx": "general"},
    "00053011": {"name": "Nucor Corp PAC TX",              "dir": "parse",    "ctx": "general"},
    "00088542": {"name": "Bexar County Conservative Coal", "dir": "R_favor",  "ctx": "R_primary"},
}

PAC_IDS = set(PACS.keys())

IE_FLAG = 25_000


def extract_dist(descr):
    for p in HD_PATS:
        m = p.search(descr)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 150:
                return "HD", n
    for p in SD_PATS:
        m = p.search(descr)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 31:
                return "SD", n
    return None, None


def get_direction(fid, descr):
    pac = PACS[fid]
    default = pac["dir"]
    if default in ("R_favor", "D_favor"):
        return default

    # AFC Victory Fund: explicit IE label
    m = AFC_IE_PAT.search(descr)
    if m:
        action = m.group(1).lower()
        # AFC spends on R-aligned races: support R = R_favor; oppose = D_favor
        # Since AFC targets R districts (HD-86 Holly Jeffreys R, HD-1 Spencer R,
        # HD-85 Kitzman R, HD-89 Noble R), treating support=R_favor, oppose=D_favor
        return "R_favor" if action == "support" else "D_favor"

    # First Amendment Alliance: explicit OPPOSE OF incumbent (always oppose R)
    m = FAA_PAT.search(descr)
    if m:
        action = m.group(1).lower()
        return "D_favor" if action == "oppose" else "R_favor"

    is_oppose  = bool(OPPOSE_PAT.search(descr))
    is_support = bool(SUPPORT_PAT.search(descr)) and not is_oppose
    if is_support:
        return "support_unknown"  # direction ambiguous without party
    return "unknown"


def load_meta():
    meta = {}
    p = DATA_PROC / "districts_2026.csv"
    if not p.exists():
        return meta
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ch = "HD" if row["chamber"].strip().lower() == "house" else "SD"
            d = row.get("district", "").strip()
            if d.isdigit():
                meta[(ch, int(d))] = row
    return meta


def main():
    # Load all 2026 rows from cached expend files
    rows = []
    for fname in ["tec_expend_12.csv", "tec_expend_13.csv"]:
        fpath = CACHE / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                dt = re.sub(r"\D", "", row.get("expendDt", ""))[:8]
                if not (dt.startswith("2026") and dt <= "20260406"):
                    continue
                fid = row.get("filerIdent", "")
                if fid not in PAC_IDS:
                    continue
                descr = row.get("expendDescr", "") or ""
                ch, d = extract_dist(descr)
                if not ch:
                    continue
                raw = (row.get("expendAmount", "") or "0").replace(",", "").replace("$", "").strip()
                try:
                    amt = float(raw)
                except ValueError:
                    continue
                if amt <= 0:
                    continue
                rows.append({
                    "fid": fid, "date": dt, "amount": amt,
                    "chamber": ch, "district": d, "descr": descr,
                })

    # Aggregate by district
    dist: dict[tuple, dict] = defaultdict(lambda: {
        "D_favor": 0.0, "R_favor": 0.0, "unknown": 0.0,
        "pacs": set(), "contexts": set(), "samples": [],
    })
    for r in rows:
        direction = get_direction(r["fid"], r["descr"])
        key = (r["chamber"], r["district"])
        if direction == "D_favor":
            dist[key]["D_favor"] += r["amount"]
        elif direction == "R_favor":
            dist[key]["R_favor"] += r["amount"]
        else:
            dist[key]["unknown"] += r["amount"]
        dist[key]["pacs"].add(r["fid"])
        dist[key]["contexts"].add(PACS[r["fid"]]["ctx"])
        if len(dist[key]["samples"]) < 2:
            dist[key]["samples"].append(
                f"  [{PACS[r['fid']]['name'][:22]}] {r['descr'][:55]}"
            )

    meta = load_meta()

    print("=" * 75)
    print("  2026 TX Legislative IE Snapshot — April 6, 2026")
    print("  (TEC data through April 2, 2026; all district-specific PAC spending)")
    print("=" * 75)

    total_d = sum(v["D_favor"] for v in dist.values())
    total_r = sum(v["R_favor"] for v in dist.values())
    total_unk = sum(v["unknown"] for v in dist.values())
    n_general = sum(1 for v in dist.values() if "general" in v["contexts"])
    n_primary = sum(1 for v in dist.values() if v["contexts"] == {"R_primary"})

    print(f"\n  Districts with activity: {len(dist)}  (flagged >{IE_FLAG/1e3:.0f}K: "
          f"{sum(1 for v in dist.values() if v['D_favor']+v['R_favor'] >= IE_FLAG)})")
    print(f"  D-favor total: ${total_d:>10,.0f}")
    print(f"  R-favor total: ${total_r:>10,.0f}")
    print(f"  Ambiguous dir: ${total_unk:>10,.0f}  (support without party marker)")
    print(f"  General-election races: {n_general}  |  R-primary-only: {n_primary}")

    # Split display: primary vs. general
    for ctx_label, ctx_filter in [
        ("GENERAL-ELECTION TARGETED", lambda v: "general" in v["contexts"]),
        ("R-PRIMARY BATTLEGROUNDS",   lambda v: v["contexts"] == {"R_primary"}),
        ("MIXED CONTEXT",             lambda v: "mixed" in v["contexts"] or
                                               (len(v["contexts"]) > 1 and "general" not in v["contexts"])),
    ]:
        subset = [(k, v) for k, v in dist.items() if ctx_filter(v)]
        if not subset:
            continue
        subset.sort(key=lambda x: -(x[1]["D_favor"] + x[1]["R_favor"] + x[1]["unknown"]))

        print(f"\n  --- {ctx_label} ---")
        print(f"  {'Dist':8s}  {'D-favor':>9s}  {'R-favor':>9s}  "
              f"{'Unknown':>9s}  {'Inc':3s}  {'Last D%':>7s}  {'PACs':4s}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*3}  {'-'*7}  {'-'*4}")

        for (ch, d), v in subset:
            tot = v["D_favor"] + v["R_favor"]
            flag = "★" if tot >= IE_FLAG else " "
            m = meta.get((ch, d), {})
            inc = m.get("incumbent_party", "?").strip()
            last_d = m.get("last_election_d_pct", "").strip()
            try:
                ld = float(last_d)
                ld_fmt = f"{ld:.1f}%"
            except (ValueError, TypeError):
                ld_fmt = "?"
            n_pacs = len(v["pacs"])
            print(f"  {flag}{ch}-{d:<4d}  ${v['D_favor']:>8,.0f}  ${v['R_favor']:>8,.0f}  "
                  f"${v['unknown']:>8,.0f}  {inc:3s}  {ld_fmt:>7s}  {n_pacs:4d}")
            for s in v["samples"]:
                print(f"  {s}")

    # PAC-level summary
    print(f"\n  --- Spending by PAC ---")
    pac_totals: dict[str, float] = defaultdict(float)
    pac_ndist:  dict[str, set]   = defaultdict(set)
    for (ch, d), v in dist.items():
        for fid in v["pacs"]:
            pac_totals[fid] += v["D_favor"] + v["R_favor"] + v["unknown"]
            pac_ndist[fid].add((ch, d))

    print(f"  {'PAC':43s}  {'Total $':>10s}  {'Dists':5s}  {'Dir':8s}  {'Ctx':10s}")
    print(f"  {'-'*43}  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*10}")
    for fid, total in sorted(pac_totals.items(), key=lambda x: -x[1]):
        if total < 500:
            continue
        pac  = PACS[fid]
        nd   = len(pac_ndist[fid])
        d_lbl = {"R_favor": "R-favor", "D_favor": "D-favor", "parse": "parsed"}.get(pac["dir"], pac["dir"])
        print(f"  {pac['name']:43s}  ${total:>9,.0f}  {nd:5d}  {d_lbl:8s}  {pac['ctx']:10s}")

    # Key interpretation
    print(f"\n  --- Interpretation ---")
    print(f"  Most April 2026 spending is in REPUBLICAN PRIMARIES (March 4 runoffs)")
    print(f"  and the SD-4 SPECIAL ELECTION (May 2). General-election targeting")
    print(f"  will ramp up after runoffs conclude (likely May-June) and accelerates")
    print(f"  sharply after July TEC filing deadline.")
    print()
    print(f"  Calibration from 2018/2022: April IE totals were 5-30% of final-cycle")
    print(f"  totals. Direction consistency was high (same districts stayed targeted).")
    print(f"  Early signals are most reliable for DISTRICT IDENTITY (which races will")
    print(f"  be competitive) rather than DOLLAR MAGNITUDE (how much will be spent).")


if __name__ == "__main__":
    main()
