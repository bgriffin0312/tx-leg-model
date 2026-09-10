"""
collect_downballot_spatial.py — statewide downballot races by legislative district

Companion to collect_presidential_spatial.py. For one general-election year and
one district plan, area-weights every statewide race of interest from the
year's precinct geometry onto the plan's districts and writes one row per
district with the Democratic two-party share for each race plus three
composites:

    rrc_d2p              Railroad Commissioner (mean if two seats were up, 2012)
    judicial_mean_d2p    mean of Supreme Court + Court of Criminal Appeals
                         contests that had both a D and an R
    downballot_mean_d2p  mean of the judicial contests and RRC
    pres_d2p             President (reference)
    sen_d2p              U.S. Senate when on the ballot (reference)

Why: docs/statewide-proxy-findings.md (2026-09-10). Same-year, the least
salient statewide races match contested House results best in every cycle
2016-2024 and the presidential vote matches worst; cross-cycle, the only pair
after straight-ticket voting ended (2020 -> 2022) favoured RRC over President
by 0.4pp residual sd and by 5pp in 70%+ Hispanic seats. These files let the
backtests and the live model use a downballot baseline instead of, or
alongside, `dem_pres_2p_baseline`.

The overlay is computed once per (year, plan) and reused for every office,
so a run is one spatial intersection, not one per race.

Usage:
    python src/collect_downballot_spatial.py --year 2024 --plan H2316
    python src/collect_downballot_spatial.py --year 2020 --plan S2168

Output: data/raw/historical/tx_downballot_{chamber}_{year}_plan{plan}.csv

Trap (recorded 2026-09-10): the 2024 comprehensive zip carries every general
2012-2024 re-keyed onto 2024 VTDs. Its 2020 file joins only 82% of the vote to
the 2020 precinct shapefile. Each year is read from its OWN zip here, as
collect_presidential_spatial.py does; 2012 comes from the FTP archive.
"""
import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_presidential_spatial as cps  # noqa: E402

HIST = cps.HIST
OFFICE_RE = re.compile(r"^(President|U\.S\. Sen|RR Comm \d|Sup Ct|CCA)")


def slug(office: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", office.lower()).strip("_")


def load_returns(year: int) -> pd.DataFrame:
    """D and R votes by precinct key for every statewide office of interest."""
    zpath = cps.CACHE / (cps.FTP_ARCHIVE if year in cps.FTP_YEARS
                         else cps.RETURNS_ZIP.format(year=year))
    if not zpath.exists():
        sys.exit(f"missing returns zip: {zpath}")
    with zipfile.ZipFile(zpath) as zf:
        member = next(f for f in zf.namelist()
                      if cps.RETURNS_CSV.format(year=year) in f)
        with zf.open(member) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()), encoding="utf-8-sig",
                             low_memory=False)
    df = df[df["Office"].astype(str).str.match(OFFICE_RE) & df["Party"].isin(["D", "R"])].copy()
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce").fillna(0)
    # Keys are strings; see collect_presidential_spatial.load_returns.
    df["PCTKEY"] = df["cntyvtd"].astype(str).str.strip().str.upper()
    # Incumbency per race, from the file's own flag (Brennan, 2026-09-10: a
    # statewide race is only a clean partisan baseline when nobody in it is an
    # incumbent). TLC flags appointees inconsistently (Blacklock 2018 N,
    # Bland 2020 Y); the flag is used as given.
    inc = (df.assign(_inc=df["Incumbent"].astype(str).str.upper().str.startswith("Y"))
             .groupby("Office")["_inc"].any())
    wide = (df.groupby(["PCTKEY", "Office", "Party"])["Votes"].sum()
              .unstack("Party", fill_value=0).reset_index())
    for c in ("D", "R"):
        if c not in wide.columns:
            wide[c] = 0.0
    totals = wide.groupby("Office")[["D", "R"]].sum()
    contested = totals[(totals["D"] > 0) & (totals["R"] > 0)].index
    dropped = sorted(set(totals.index) - set(contested))
    if dropped:
        print(f"  skipped (no D-v-R contest): {dropped}")
    wide = wide[wide["Office"].isin(contested)]
    wide.attrs["has_incumbent"] = {o: bool(inc.get(o, False)) for o in contested}
    return wide


def build_overlay(year: int, plan: str) -> tuple[pd.DataFrame, str, str, int, set[str]]:
    """Precinct-piece table with area shares, computed once per (year, plan)."""
    pg = cps.load_precinct_geometry(year)
    gpkg, dist_col, chamber, n_max = cps.PLAN_GEO[plan]
    if str(gpkg).lower().endswith(".zip"):
        with zipfile.ZipFile(gpkg) as zf:
            shp = next(n for n in zf.namelist() if n.lower().endswith(".shp"))
        dg = gpd.read_file(f"zip://{gpkg}!{shp}")
    else:
        dg = gpd.read_file(gpkg)
    dg = dg.to_crs(pg.crs)
    pg = pg.copy()
    pg["_pct_area"] = pg.geometry.area
    inter = gpd.overlay(pg, dg[[dist_col, "geometry"]], how="intersection",
                        keep_geom_type=True)
    inter["_share"] = inter.geometry.area / inter["_pct_area"]
    tot = inter.groupby("PCTKEY")["_share"].transform("sum")
    inter["_share"] = inter["_share"] / tot.where(tot > 0, 1.0)
    inter["district"] = inter[dist_col].astype(int)
    pieces = inter[["PCTKEY", "district", "_share"]].copy()
    print(f"  overlay: {len(pieces):,} pieces from {pg['PCTKEY'].nunique():,} precincts "
          f"onto {dg[dist_col].nunique()} districts ({chamber})")
    return pieces, chamber, dist_col, n_max, set(pg["PCTKEY"])


def allocate_office(votes: pd.DataFrame, pieces: pd.DataFrame, poly_keys: set[str],
                    office: str) -> tuple[pd.Series, float]:
    sub = votes[votes["Office"] == office][["PCTKEY", "D", "R"]].copy()
    sub["PCTKEY"] = sub["PCTKEY"].apply(
        lambda k: k if k in poly_keys else cps._strip_suffix(k, poly_keys))
    sub = sub.groupby("PCTKEY", as_index=False)[["D", "R"]].sum()
    in_file = (sub["D"] + sub["R"]).sum()
    m = pieces.merge(sub, on="PCTKEY", how="inner")
    m["D"] = m["D"] * m["_share"]
    m["R"] = m["R"] * m["_share"]
    agg = m.groupby("district")[["D", "R"]].sum()
    attached = (agg["D"] + agg["R"]).sum() / in_file if in_file else 0.0
    return agg["D"] / (agg["D"] + agg["R"]), attached


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, required=True, choices=sorted(cps.PRECINCT_GEO))
    ap.add_argument("--plan", required=True, choices=sorted(cps.PLAN_GEO))
    ap.add_argument("--min-attach", type=float, default=0.99,
                    help="refuse to write if any office attaches less than this share of its vote")
    args = ap.parse_args()

    print(f"=== {args.year} statewide downballot → PLAN{args.plan} ===")
    votes = load_returns(args.year)
    offices = sorted(votes["Office"].unique())
    print(f"  offices: {offices}")
    pieces, chamber, dist_col, n_max, poly_keys = build_overlay(args.year, args.plan)

    out = pd.DataFrame({"district": sorted(pieces["district"].unique())}).set_index("district")
    out = out[(out.index >= 1) & (out.index <= n_max)]
    for office in offices:
        share, attached = allocate_office(votes, pieces, poly_keys, office)
        print(f"  {office:<16s} attached {attached * 100:6.2f}%")
        if attached < args.min_attach:
            sys.exit(f"{office}: only {attached * 100:.2f}% of the vote attached — refusing to write")
        out[f"d2p__{slug(office)}"] = share.reindex(out.index).round(4)

    jud = [c for c in out.columns if re.match(r"d2p__(sup_ct|cca)", c)]
    rrc = [c for c in out.columns if c.startswith("d2p__rr_comm")]
    has_inc = votes.attrs["has_incumbent"]
    open_offices = [o for o in offices if re.match(r"^(RR Comm|Sup Ct|CCA)", o) and not has_inc[o]]
    opn = [f"d2p__{slug(o)}" for o in open_offices]
    print(f"  incumbents: {[o for o in offices if has_inc[o]]}")
    print(f"  open downballot races: {open_offices}")
    out["rrc_d2p"] = out[rrc].mean(axis=1).round(4) if rrc else float("nan")
    out["judicial_mean_d2p"] = out[jud].mean(axis=1).round(4) if jud else float("nan")
    out["downballot_mean_d2p"] = out[jud + rrc].mean(axis=1).round(4) if (jud or rrc) else float("nan")
    out["open_mean_d2p"] = out[opn].mean(axis=1).round(4) if opn else float("nan")
    out["pres_d2p"] = out["d2p__president"] if "d2p__president" in out else float("nan")
    out["sen_d2p"] = out["d2p__u_s_sen"] if "d2p__u_s_sen" in out else float("nan")
    out["n_judicial"] = len(jud)
    out["n_open"] = len(opn)
    out["open_offices"] = "; ".join(open_offices)
    out["chamber"] = chamber
    out["year"] = args.year
    out["plan"] = args.plan
    out["data_source"] = f"spatial_{args.year}_precincts_x_plan{args.plan}"
    out = out.reset_index()

    print(f"\n  districts: {len(out)}/{n_max}" + ("" if len(out) == n_max else "  ** MISSING **"))
    print(f"  statewide-ish means: pres {out['pres_d2p'].mean():.4f}  rrc {out['rrc_d2p'].mean():.4f}  "
          f"judicial {out['judicial_mean_d2p'].mean():.4f}  (n_judicial={len(jud)})")
    path = HIST / f"tx_downballot_{chamber}_{args.year}_plan{args.plan.lower()}.csv"
    lead = ["chamber", "district", "year", "plan", "rrc_d2p", "judicial_mean_d2p",
            "downballot_mean_d2p", "open_mean_d2p", "pres_d2p", "sen_d2p",
            "n_judicial", "n_open", "open_offices"]
    rest = [c for c in out.columns if c not in lead and c != "data_source"]
    out[lead + rest + ["data_source"]].to_csv(path, index=False)
    print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
