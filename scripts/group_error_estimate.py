"""
group_error_estimate.py

Estimate the correlated group-error layer's sigmas (docs/poll-integration-
proposal.md §3e) from past cycles, to set against the proposal's priors.

The layer adds, per simulation, one error per racial group and moves each
district by its CENTERED composition:
    group_err_d = sum_r (share_dr - w_TX[r]) * eps_r
Because the centered shares sum to zero, adding a constant to every eps_r
changes nothing (that part lives in the national layer), so only differences
between groups are identified. Here white is the reference.

Method. For each cycle, take the residuals (actual - predicted D share) from a
model with no demographic term, and regress them on the centered Black,
Hispanic and other shares. The coefficient b_r(cycle) estimates
eps_r - eps_white for that cycle. Across cycles,
    Var(b_r) - mean(se_r^2)  ~  sigma_r^2 + sigma_white^2
(the subtraction removes the part of the spread that is just sampling noise).

Cycles:
  2014/2018/2022 -- midterms, residuals vs the adopted +inc fit on the
                    open-race baseline (scripts/refit_clean_cycles.py rows)
  2020/2024      -- presidential years, residuals from data/processed/race_war.csv
                    (no-finance open-baseline prediction plus the cycle level)
Midterms are what 2026 is; the presidential years are reported separately.

Usage:
  python scripts/group_error_estimate.py
  python scripts/group_error_estimate.py --education   # split white by BA+
  python scripts/group_error_estimate.py --race-edu    # split white/Black/Hispanic by BA+
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import refit_clean_cycles as rc  # noqa: E402

RAW = ROOT / "data" / "raw"
GROUPS = ["white", "black", "hispanic", "other"]
# --education splits white into college (BA+) / non-college using the ACS
# white-NH 25+ BA rate (src/collect_education_by_district.py). Reference group
# becomes white non-college, so b_white_col is the college-vs-non-college gap.
EDU = "--education" in sys.argv
EDU_ACS = {2014: 2021, 2018: 2021, 2020: 2021, 2022: 2024, 2024: 2024}
if EDU:
    GROUPS = ["white_noncol", "white_col", "black", "hispanic", "other"]
# --race-edu splits white, Black and Hispanic each by BA+ (Catalist-style cells).
RACE_EDU = "--race-edu" in sys.argv
if RACE_EDU:
    EDU = True
    GROUPS = ["white_noncol", "white_col", "black_noncol", "black_col",
              "hispanic_noncol", "hispanic_col", "other"]
PRIOR_PP = {"white": 2.5, "black": 4.0, "hispanic": 4.5, "other": 6.0}
CVAP = {  # (year, chamber) -> CVAP file on the lines the race ran under
    (2014, "house"): "historical/tx_cvap_house_2018.csv",
    (2018, "house"): "historical/tx_cvap_house_2018.csv",
    (2020, "house"): "historical/tx_cvap_house_2018.csv",
    (2022, "house"): "tx_cvap_house.csv",
    (2024, "house"): "tx_cvap_house.csv",
    (2014, "senate"): "historical/tx_cvap_senate_2018.csv",
    (2018, "senate"): "historical/tx_cvap_senate_2018.csv",
    (2020, "senate"): "historical/tx_cvap_senate_2018.csv",
    (2022, "senate"): "tx_cvap_senate.csv",
    (2024, "senate"): "tx_cvap_senate.csv",
}


def shares_from_cvap(c: pd.DataFrame) -> pd.DataFrame:
    """Normalized group shares from CVAP COUNTS (other = Asian + AIAN + other)."""
    out = pd.DataFrame({
        "white": c["cvap_white_nh"], "black": c["cvap_black_nh"],
        "hispanic": c["cvap_hispanic"],
        "other": c["cvap_asian_nh"] + c["cvap_aian_nh"] + c["cvap_other"],
    }).astype(float)
    return out.div(out.sum(axis=1), axis=0)


def texas_weights(cvap_rel: str = "tx_cvap_house.csv") -> dict:
    c = pd.read_csv(RAW / cvap_rel)
    c = c[c["chamber"].str.lower() == "house"]
    tot = {"white": c.cvap_white_nh.sum(), "black": c.cvap_black_nh.sum(),
           "hispanic": c.cvap_hispanic.sum(),
           "other": (c.cvap_asian_nh + c.cvap_aian_nh + c.cvap_other).sum()}
    s = sum(tot.values())
    w = {g: v / s for g, v in tot.items()}
    if EDU:
        e = pd.read_csv(RAW / "tx_education_house_acs2024.csv")
        rate = e.white_nh_ba_plus.sum() / e.white_nh_25p.sum()
        w["white_col"], w["white_noncol"] = w["white"] * rate, w["white"] * (1 - rate)
        if RACE_EDU:
            for g in ("black", "hispanic"):
                r = e[f"{g}_ba_plus"].sum() / e[f"{g}_25p"].sum()
                w[f"{g}_col"], w[f"{g}_noncol"] = w[g] * r, w[g] * (1 - r)
    return w


def attach_shares(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (yr, ch), rel in CVAP.items():
        c = pd.read_csv(RAW / rel)
        c = c[c["chamber"].str.lower() == ch].reset_index(drop=True)
        sh = shares_from_cvap(c)
        sh["district"] = c["district"].astype(int).values
        if EDU:
            e = pd.read_csv(RAW / f"tx_education_{ch}_acs{EDU_ACS[yr]}.csv")
            sh = sh.merge(e[["district", "white_college_rate"]], on="district", how="left")
            sh["white_col"] = sh["white"] * sh["white_college_rate"]
            sh["white_noncol"] = sh["white"] - sh["white_col"]
            if RACE_EDU:
                sh = sh.merge(e[["district", "black_college_rate", "hispanic_college_rate"]],
                              on="district", how="left")
                for g in ("black", "hispanic"):
                    sh[f"{g}_col"] = sh[g] * sh[f"{g}_college_rate"]
                    sh[f"{g}_noncol"] = sh[g] - sh[f"{g}_col"]
                sh = sh.drop(columns=["black_college_rate", "hispanic_college_rate"])
            sh = sh.drop(columns=["white_college_rate"])
        sh["year"], sh["chamber_lower"] = yr, ch
        parts.append(sh)
    sh = pd.concat(parts, ignore_index=True)
    return df.merge(sh, on=["year", "chamber_lower", "district"], how="left")


def cycle_coefs(d: pd.DataFrame, w: dict) -> pd.DataFrame:
    rows = []
    for yr, g in d.groupby("year"):
        g = g.dropna(subset=GROUPS + ["resid"])
        X = np.column_stack([np.ones(len(g))] + [g[r].values - w[r] for r in GROUPS[1:]])
        y = g["resid"].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        s2 = e @ e / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        rows.append({"year": yr, "n": len(g),
                     **{f"b_{r}": b[i + 1] * 100 for i, r in enumerate(GROUPS[1:])},
                     **{f"se_{r}": se[i + 1] * 100 for i, r in enumerate(GROUPS[1:])}})
    return pd.DataFrame(rows)


def implied_sigma(t: pd.DataFrame, r: str) -> tuple[float, float]:
    raw = t[f"b_{r}"].var(ddof=1)
    noise = (t[f"se_{r}"] ** 2).mean()
    return np.sqrt(raw), np.sqrt(max(raw - noise, 0.0))


def main():
    pd.set_option("display.width", 200)
    w = texas_weights()
    print("Texas CVAP weights (2024 ACS, House districts summed): "
          + ", ".join(f"{g} {v:.3f}" for g, v in w.items()))

    # Midterms: residuals vs the +inc fit on the open-race baseline.
    mid = rc.load(rc.bb.PROC / "phase1_dataset.csv", "open")
    c, _, _ = rc.ols(mid, rc.BASE + rc.INC)
    mid["resid"] = mid["dem_2p_share"] - rc.predict(mid, c)
    mid = attach_shares(mid)

    # Presidential years: WAR's race table (already cycle-centred).
    rw = pd.read_csv(ROOT / "data" / "processed" / "race_war.csv")
    rw = rw[rw["year"].isin([2020, 2024])].drop_duplicates(["year", "chamber", "district"])
    rw["chamber_lower"] = rw["chamber"].str.lower()
    rw["resid"] = (rw["actual_dem_2p"] - rw["predicted_dem_2p"]) / 100
    pres = attach_shares(rw[["year", "chamber_lower", "district", "resid"]])

    for label, d in (("MIDTERMS 2014/2018/2022", mid), ("PRESIDENTIAL 2020/2024", pres),
                     ("ALL FIVE CYCLES", pd.concat([mid[["year", "resid"] + GROUPS],
                                                     pres[["year", "resid"] + GROUPS]]))):
        t = cycle_coefs(d, w)
        print(f"\n=== {label}: per-cycle group effect vs white (pp per unit centered share) ===")
        print(t.round(2).to_string(index=False))
        print("  implied sigma(group - white), pp:   raw spread  /  net of sampling noise   "
              "vs prior sqrt(s_r^2 + s_white^2)")
        for r in GROUPS[1:]:
            raw, net = implied_sigma(t, r)
            prior = np.hypot(PRIOR_PP.get(r, np.nan), PRIOR_PP["white"])
            print(f"    {r:<12} {raw:6.2f}  /  {net:6.2f}      prior {prior:5.2f}   "
                  f"mean {t[f'b_{r}'].mean():+6.2f}")


if __name__ == "__main__":
    main()
