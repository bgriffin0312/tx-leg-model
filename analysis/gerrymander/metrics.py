"""
Standard partisan-symmetry/efficiency metrics for redistricting analysis.

All functions take a district-level Democratic two-party vote share (0-1) and
return a metric where, BY CONVENTION IN THIS MODULE, **positive = pro-Republican
bias, negative = pro-Democratic bias**. This matches the sign convention used
by the Princeton Gerrymandering Project's Redistricting Report Card
(https://gerrymander.princeton.edu/redistricting-report-card-methodology/)
for partisan bias and mean-median difference, and is applied consistently here
to efficiency gap and declination as well so a results table reads the same
way across all four metrics.

Formulas and sources:
- Efficiency gap: wasted-vote definition (Stephanopoulos & McGhee 2015,
  "Partisan Gerrymandering and the Efficiency Gap"), generalized to unequal
  district turnout. The commonly cited turnout-equal shortcut
  EG = (seat_share - 0.5) - 2*(vote_share - 0.5) is included as a check.
- Mean-median difference: mean(dem_share) - median(dem_share); sign per
  Princeton methodology (positive favors Republicans).
- Partisan bias: uniform partisan swing to a hypothetical 50-50 statewide
  vote, then compare each party's seat share to 50%. Standard approach used
  by Princeton Gerrymandering Project / Wang and others.
- Declination: Warrington 2018 ("Quantifying Gerrymandering Using the Vote
  Distribution", arXiv:1705.09393) and Warrington 2018b ("Introduction to the
  declination function for gerrymanders", arXiv:1803.04799), formula verified
  against the arXiv text directly (not reconstructed from memory):
      f_R = |R_won| / n,  f_D = |D_won| / n
      theta = arctan( (1 - 2*mean(dem_share | R_won)) / f_R )
      gamma = arctan( (2*mean(dem_share | D_won) - 1) / f_D )
      declination = (2/pi) * (gamma - theta)
  Positive declination favors Republicans per both sources. Undefined
  (returns NaN) when either party wins zero seats. declination * n/2 is
  Warrington's rough heuristic for "seats affected."

IMPORTANT CAVEAT (apply when interpreting real Texas data): every function
here assumes every district was effectively contested, i.e. dem_share
reflects real relative party strength even in "uncontested" seats. Texas has
many uncontested legislative races. Callers MUST use an imputed/proxy dem
share (e.g. 2024 presidential two-party share) for uncontested districts
before calling these functions on actual legislative results — see
run_metrics.py for the imputation this project uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_array(dem_share) -> np.ndarray:
    arr = np.asarray(dem_share, dtype=float)
    if np.any(np.isnan(arr)):
        raise ValueError("dem_share contains NaN — impute uncontested/missing districts before calling metrics")
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError("dem_share must be a two-party share in [0, 1]")
    return arr


def efficiency_gap(dem_share, turnout=None) -> dict:
    """Wasted-vote efficiency gap. turnout: per-district total two-party votes;
    defaults to equal turnout (1 per district), which reproduces the standard
    turnout-equal shortcut formula as a cross-check.
    """
    dem_share = _as_array(dem_share)
    n = len(dem_share)
    if turnout is None:
        turnout = np.ones(n)
    else:
        turnout = np.asarray(turnout, dtype=float)

    dem_votes = dem_share * turnout
    rep_votes = turnout - dem_votes
    dem_wins = dem_share >= 0.5

    win_threshold = turnout / 2.0  # votes needed for a bare majority
    wasted_dem = np.where(dem_wins, dem_votes - win_threshold, dem_votes)
    wasted_rep = np.where(~dem_wins, rep_votes - win_threshold, rep_votes)

    total_votes = turnout.sum()
    eg = (wasted_dem.sum() - wasted_rep.sum()) / total_votes  # positive = pro-R (Dems wasted more)

    # Turnout-equal shortcut, as an internal cross-check when turnout is uniform.
    seat_share_dem = dem_wins.mean()
    vote_share_dem = dem_votes.sum() / total_votes
    eg_shortcut = (seat_share_dem - 0.5) - 2 * (vote_share_dem - 0.5)
    eg_shortcut = -eg_shortcut  # flip to positive = pro-R, matching the convention above

    return {
        "efficiency_gap": eg,
        "efficiency_gap_shortcut": eg_shortcut,
        "favors": "R" if eg > 0 else ("D" if eg < 0 else "even"),
        "wasted_dem_votes": wasted_dem.sum(),
        "wasted_rep_votes": wasted_rep.sum(),
    }


def mean_median(dem_share) -> dict:
    dem_share = _as_array(dem_share)
    mm = float(np.mean(dem_share) - np.median(dem_share))
    return {"mean_median": mm, "favors": "R" if mm > 0 else ("D" if mm < 0 else "even")}


def partisan_bias(dem_share, turnout=None, seat_threshold: float = 0.5) -> dict:
    """Uniform partisan swing to a hypothetical statewide 50-50 vote, then
    compare seat shares. Positive = pro-Republican (Republicans win >50% of
    seats when the statewide vote is tied)."""
    dem_share = _as_array(dem_share)
    n = len(dem_share)
    if turnout is None:
        turnout = np.ones(n)
    else:
        turnout = np.asarray(turnout, dtype=float)

    statewide_dem_share = (dem_share * turnout).sum() / turnout.sum()
    delta = 0.5 - statewide_dem_share
    swung = np.clip(dem_share + delta, 0.0, 1.0)

    dem_seat_share = (swung >= seat_threshold).mean()
    rep_seat_share = 1.0 - dem_seat_share
    bias = rep_seat_share - 0.5  # positive = pro-R

    return {
        "partisan_bias": bias,
        "favors": "R" if bias > 0 else ("D" if bias < 0 else "even"),
        "dem_seat_share_at_5050": dem_seat_share,
        "swing_applied": delta,
    }


def declination(dem_share) -> dict:
    """Warrington declination. NaN if either party wins zero seats."""
    dem_share = _as_array(dem_share)
    n = len(dem_share)
    r_won = dem_share[dem_share < 0.5]
    d_won = dem_share[dem_share >= 0.5]

    if len(r_won) == 0 or len(d_won) == 0:
        return {"declination": np.nan, "favors": "undefined (shutout)", "est_seats_affected": np.nan}

    f_r = len(r_won) / n
    f_d = len(d_won) / n
    theta = np.arctan((1 - 2 * r_won.mean()) / f_r)
    gamma = np.arctan((2 * d_won.mean() - 1) / f_d)
    decl = (2 / np.pi) * (gamma - theta)

    return {
        "declination": decl,
        "favors": "R" if decl > 0 else ("D" if decl < 0 else "even"),
        "est_seats_affected": decl * n / 2,
    }


def seats_votes_curve(dem_share, turnout=None, swing_range=None) -> pd.DataFrame:
    """Apply uniform swings across a range and record resulting Dem seat share.
    Used to plot the seats-votes curve and to estimate responsiveness (the
    curve's slope near the current statewide vote share)."""
    dem_share = _as_array(dem_share)
    n = len(dem_share)
    if turnout is None:
        turnout = np.ones(n)
    else:
        turnout = np.asarray(turnout, dtype=float)
    if swing_range is None:
        swing_range = np.linspace(-0.25, 0.25, 101)

    statewide_dem_share = (dem_share * turnout).sum() / turnout.sum()
    rows = []
    for s in swing_range:
        swung = np.clip(dem_share + s, 0.0, 1.0)
        rows.append({
            "swing": s,
            "statewide_dem_share": statewide_dem_share + s,
            "dem_seat_share": (swung >= 0.5).mean(),
        })
    return pd.DataFrame(rows)


def responsiveness(svc: pd.DataFrame, window: float = 0.02) -> float:
    """Local slope of the seats-votes curve near the current vote share
    (swing == 0), i.e. how many additional seat-share points per vote-share
    point near the status quo. Low responsiveness near 0 combined with a
    packed distribution is itself gerrymandering evidence."""
    near = svc[np.abs(svc["swing"]) <= window]
    if len(near) < 2:
        return np.nan
    return float(np.polyfit(near["statewide_dem_share"], near["dem_seat_share"], 1)[0])


def packing_diagnostics(dem_share) -> dict:
    dem_share = _as_array(dem_share)
    dem_margin = np.where(dem_share >= 0.5, dem_share - 0.5, np.nan)
    rep_margin = np.where(dem_share < 0.5, 0.5 - dem_share, np.nan)
    safe_threshold = 0.20  # >20-pt margin treated as "safe"

    return {
        "n_dem_seats": int((dem_share >= 0.5).sum()),
        "n_rep_seats": int((dem_share < 0.5).sum()),
        "median_dem_winning_margin": float(np.nanmedian(dem_margin)) if not np.all(np.isnan(dem_margin)) else np.nan,
        "median_rep_winning_margin": float(np.nanmedian(rep_margin)) if not np.all(np.isnan(rep_margin)) else np.nan,
        "n_safe_dem_seats": int(np.nansum(dem_margin > safe_threshold)),
        "n_safe_rep_seats": int(np.nansum(rep_margin > safe_threshold)),
        "n_competitive_seats": int(np.sum(np.abs(dem_share - 0.5) <= 0.05)),  # within 5 pts
    }


def run_all(dem_share, turnout=None) -> dict:
    """Convenience: run the full battery and return a flat dict."""
    out = {}
    out.update({f"eg_{k}": v for k, v in efficiency_gap(dem_share, turnout).items()})
    out.update({f"mm_{k}": v for k, v in mean_median(dem_share).items()})
    out.update({f"pb_{k}": v for k, v in partisan_bias(dem_share, turnout).items()})
    out.update({f"decl_{k}": v for k, v in declination(dem_share).items()})
    out.update({f"pack_{k}": v for k, v in packing_diagnostics(dem_share).items()})
    svc = seats_votes_curve(dem_share, turnout)
    out["responsiveness_at_status_quo"] = responsiveness(svc)
    return out


# ---------------------------------------------------------------------------
# Self-test: synthetic sanity checks run automatically so sign-convention
# bugs surface immediately rather than silently producing a misleading number.
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> None:
    rng = np.random.default_rng(0)

    # 1. Symmetric case: mirror-image dem_share around 0.5 -> all metrics ~0.
    half = np.linspace(0.05, 0.45, 20)
    symmetric = np.concatenate([half, 1 - half])  # 40 districts, perfectly symmetric
    r = run_all(symmetric)
    assert abs(r["eg_efficiency_gap"]) < 0.03, f"symmetric EG should be ~0, got {r['eg_efficiency_gap']}"
    assert abs(r["mm_mean_median"]) < 1e-9, f"symmetric mean-median should be exactly 0, got {r['mm_mean_median']}"
    assert abs(r["pb_partisan_bias"]) < 0.03, f"symmetric partisan bias should be ~0, got {r['pb_partisan_bias']}"
    assert abs(r["decl_declination"]) < 0.05, f"symmetric declination should be ~0, got {r['decl_declination']}"

    # 2. Deliberate pro-Republican gerrymander: pack Democrats into a few
    # very safe seats, crack the rest into modest Republican-won districts.
    # 10 districts: 2 packed D seats at 90%, 8 R seats at 55%.
    packed_pro_r = np.array([0.90, 0.90] + [0.45] * 8)
    r2 = run_all(packed_pro_r)
    assert r2["eg_efficiency_gap"] > 0, f"packed-pro-R case should show positive (pro-R) EG, got {r2['eg_efficiency_gap']}"
    assert r2["mm_mean_median"] > 0, f"packed-pro-R case should show positive (pro-R) mean-median, got {r2['mm_mean_median']}"
    assert r2["pb_partisan_bias"] > 0, f"packed-pro-R case should show positive (pro-R) partisan bias, got {r2['pb_partisan_bias']}"
    assert r2["decl_declination"] > 0, f"packed-pro-R case should show positive (pro-R) declination, got {r2['decl_declination']}"

    # 3. Mirror of #2 -> everything should flip sign (pro-Democratic).
    packed_pro_d = 1 - packed_pro_r
    r3 = run_all(packed_pro_d)
    assert r3["eg_efficiency_gap"] < 0
    assert r3["mm_mean_median"] < 0
    assert r3["pb_partisan_bias"] < 0
    assert r3["decl_declination"] < 0

    # 4. Declination undefined on a shutout.
    shutout = np.array([0.3, 0.35, 0.4, 0.45])
    d = declination(shutout)
    assert np.isnan(d["declination"])

    if verbose:
        print("metrics.py self_test: all sign-convention and shutout checks passed")


if __name__ == "__main__":
    self_test()
