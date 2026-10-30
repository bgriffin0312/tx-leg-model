"""2018-analog forecast: fit the shipped spec on 2018 alone and run 2026 through the model's MC.

Variants
  ship      published coefficients (pooled 2014/2018/2022)
  a2018     same terms fit on 2018 only; 2018's level carried to 2026 (+ env coef x 0.6)
  a2018+H   a2018 plus a centered Hispanic-share term fit on 2018
Also prints, for the pooled fit, each cycle's level miss and residual slope on Hispanic share.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.argv.append("--education")
import group_error_estimate as ge  # noqa: E402
import refit_clean_cycles as rc  # noqa: E402
import model as M  # noqa: E402

w = ge.texas_weights()
d = ge.attach_shares(rc.load(rc.bb.PROC / "phase1_dataset.csv", "open"))
for g in ("white_col", "hispanic"):
    d[f"c_{g}"] = d[g] - w[g]
d["challenger_viability_flag"] = d["challenger_viability_flag"].fillna(0)
d = d.dropna(subset=["c_white_col", "c_hispanic", "dem_incumbent", "rep_incumbent"])
print("rows:", d.groupby("year").size().to_dict())

BASE = ["x", "chamber_senate", "dem_incumbent", "rep_incumbent",
        "challenger_viability_flag", "ie_c", "c_white_col"]


def fit(rows, terms):
    X = np.column_stack([np.ones(len(rows))] + [rows[t] for t in terms])
    b, *_ = np.linalg.lstsq(X, rows["dem_2p_share"].values, rcond=None)
    return dict(zip(["const"] + terms, b)), rows["dem_2p_share"].values - X @ b


pooled, res = fit(d, BASE + ["national_env"])
d["res"] = res
print("\npooled fit, by cycle: level miss / slope on Hispanic share (pp per unit) / slope on white-college")
for y in rc.CYCLES:
    g = d[d.year == y]
    sh = np.polyfit(g["c_hispanic"], g["res"], 1)[0]
    sw = np.polyfit(g["c_white_col"], g["res"], 1)[0]
    print(f"  {y}: {100 * g.res.mean():+.2f}pp   Hisp {100 * sh:+.1f}   white-col {100 * sw:+.1f}")

d18 = d[d.year == 2018]
f18, _ = fit(d18, BASE)
f18h, _ = fit(d18, BASE + ["c_hispanic"])
print("\ncoefficients   pooled(ship-like)   2018-only   2018-only+H")
for k in ["const"] + BASE + ["c_hispanic", "national_env"]:
    print(f"  {k:28s} {pooled.get(k, np.nan):+.4f}   {f18.get(k, np.nan):+.4f}   {f18h.get(k, np.nan):+.4f}")

# ---- run 2026 -------------------------------------------------------------
df = M.load_districts()
rg = M.RACE_GENERIC_BALLOT_D_SHARE
ENV_COEF = M.COEFS["national_env"]
saved = dict(M.COEFS)
saved_ie = M.IE_COEFFICIENT if hasattr(M, "IE_COEFFICIENT") else None
orig_blp = M.build_linear_predictions
hisp_c = M.centered_group_shares(df)[:, M._GROUPS.index("hispanic")]


def apply(f, hisp=0.0):
    M.COEFS.update({
        "intercept": f["const"] - ENV_COEF * 8.6,   # reproduces 2018's level at env 8.6
        "dem_pres_2p_baseline": f["x"], "dem_incumbent": f["dem_incumbent"],
        "rep_incumbent": f["rep_incumbent"], "chamber_senate": f["chamber_senate"],
        "challenger_viability_flag": f["challenger_viability_flag"],
        "white_col_centered": f["c_white_col"],
    })
    if saved_ie is not None:
        M.IE_COEFFICIENT = f["ie_c"]
    M.build_linear_predictions = (lambda df_, e, r: orig_blp(df_, e, r) + hisp * hisp_c)


def restore():
    M.COEFS.clear(); M.COEFS.update(saved)
    if saved_ie is not None:
        M.IE_COEFFICIENT = saved_ie
    M.build_linear_predictions = orig_blp


ENVS = [9.2, 12, 15, 18]
for name, setup in [("ship", None), ("a2018", (f18, 0.0)), ("a2018+H", (f18h, f18h["c_hispanic"]))]:
    restore()
    if setup:
        apply(*setup)
    out = []
    for e in ENVS:
        M.RNG = np.random.default_rng(42)
        r = M.run_monte_carlo(df, e, rg, M.N_SIMULATIONS)
        out.append(f"D+{e}: {r['expected_house_seats']:.1f} seats / {100 * r['house_control_prob']:.1f}%")
    print(f"\n{name:8s} " + " | ".join(out))
    if name != "ship":
        r = M.run_monte_carlo(df, 9.2, rg, M.N_SIMULATIONS)
        p = r["district_win_probs"]
        restore(); M.RNG = np.random.default_rng(42)
        p0 = M.run_monte_carlo(df, 9.2, rg, M.N_SIMULATIONS)["district_win_probs"]
        delta = (p - p0)
        h = df["chamber_lower"].values == "house"
        idx = np.argsort(-np.abs(delta.values * h))[:12]
        print("   biggest House moves vs ship at D+9.2:",
              ", ".join(f"HD{int(df.district.iloc[i])} {100 * p0.iloc[i]:.0f}->{100 * p.iloc[i]:.0f}" for i in idx))
restore()
