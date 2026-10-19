"""
bullpen_tier_compare.py

Compare the Texas Bullpen's House GOP leadership worry tiers (October 2026) to
the model, and ask what "Tier 1 races within 5 points" would mean for the
majority: by conditioning the simulations on Tier 1 outcomes, and by solving
for the environment dial that puts the typical Tier 1 race at a given margin.
Findings: NEXT-STEPS.md, "Texas Bullpen worry list vs. the model".

Usage:
  python scripts/bullpen_tier_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model  # noqa: E402

TIER1 = [26, 28, 34, 37, 52, 61, 63, 80, 94, 96, 97, 108, 112, 121, 122, 127, 129, 132, 133, 138, 150]
TIER2 = [14, 20, 29, 54, 67, 126]
TIER3 = [55, 89]
TIER = {**{d: 1 for d in TIER1}, **{d: 2 for d in TIER2}, **{d: 3 for d in TIER3}}


def simulate(df, env, n=model.N_SIMULATIONS, seed=42):
    """Same draws as model.run_monte_carlo, but keep the vote-share matrix."""
    model.RNG = np.random.default_rng(seed)
    rng = model.RNG
    linear = model.build_linear_predictions(df, env, model.RACE_GENERIC_BALLOT_D_SHARE).values
    nat = rng.normal(0, model.SIGMA_NATIONAL, size=n)
    if model.GROUP_ERROR_ENABLED:
        groups = model._GROUPS_EDU if model.GROUP_ERROR_EDUCATION else model._GROUPS
        centered = model.centered_group_shares(df, model.GROUP_ERROR_EDUCATION)
        sig = np.array([model.GROUP_ERROR_SIGMA[g] for g in groups])
        geps = rng.normal(0, sig, size=(n, len(groups)))
        gerr = centered @ geps.T
        mgv = float(((centered ** 2) @ sig ** 2).mean())
        idio_sd = float(np.sqrt(max(model.SIGMA_IDIO ** 2 - mgv, 0.0)))
    else:
        gerr, idio_sd = 0.0, model.SIGMA_IDIO
    idio = rng.normal(0, idio_sd, size=(len(df), n))
    share = linear[:, None] + nat[None, :] + gerr + idio
    wins = share > 0.5
    forced = df["forced_d_win"].values
    wins[forced == 1.0, :] = True
    wins[forced == 0.0, :] = False
    return linear, share, wins, nat


def main():
    df = model.load_districts().reset_index(drop=True)
    cands = pd.read_csv(ROOT / "data/processed/candidates_2026.csv")
    cands["chamber_lower"] = cands.chamber.str.lower()
    df = df.merge(cands[["chamber_lower", "district", "r_candidate", "d_candidate"]],
                  on=["chamber_lower", "district"], how="left")
    house = (df.chamber_lower == "house").values
    env = model.CURRENT_ENV
    linear, share, wins, nat = simulate(df, env)
    hs = wins[house].sum(axis=0)
    print(f"Env D+{env:.1f}: House E={hs.mean():.1f}, P(>=76)={(hs >= 76).mean():.3f}\n")

    df["d_share"] = linear
    df["margin_d"] = (2 * linear - 1) * 100       # D minus R, points
    df["p_d"] = wins.mean(axis=1)
    df["tier"] = df.district.map(TIER).where(house)
    h = df[house].copy()

    # Rank every R-held seat by D win probability
    rh = h[h.incumbent_party == "R"].sort_values("p_d", ascending=False).reset_index(drop=True)
    rh["rank"] = rh.index + 1
    cols = ["rank", "district", "tier", "r_candidate", "d_candidate", "margin_d", "p_d"]
    print("R-held House seats, ranked by model D win prob (top 40):")
    print(rh[cols].head(40).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nBullpen seats not in model's top 40 R-held:")
    print(rh[rh.tier.notna() & (rh["rank"] > 40)][cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nBy tier: mean margin, sum p_d, n")
    print(rh.groupby("tier").agg(n=("district", "size"), mean_margin=("margin_d", "mean"),
                                 exp_flips=("p_d", "sum")).to_string())
    print(f"\nExpected flips from all R-held seats: {rh.p_d.sum():.1f}; "
          f"from Tier 1: {rh[rh.tier == 1].p_d.sum():.1f}")
    # D-held seats at risk (Bullpen only lists R seats)
    dh = h[h.incumbent_party == "D"].sort_values("p_d").head(10)
    print("\nMost vulnerable D-held seats:")
    print(dh[["district", "incumbent", "r_candidate", "margin_d", "p_d"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # --- Conditioning: Tier 1 "within 5 points"
    t1 = house & df.district.isin(TIER1).values & (df.incumbent_party == "R").values
    t1_margin = (2 * share[t1] - 1) * 100      # (n_t1, n_sims)
    med = np.median(t1_margin, axis=0)
    mean = t1_margin.mean(axis=0)
    print(f"\nTier 1 (n={t1.sum()}): model expected margin median {np.median(df.margin_d[t1]):.1f}, "
          f"mean {df.margin_d[t1].mean():.1f}")
    print("Distribution of the Tier 1 median margin across sims: "
          f"p10 {np.percentile(med, 10):.1f} p50 {np.percentile(med, 50):.1f} p90 {np.percentile(med, 90):.1f}")
    print(f"P(Tier 1 median within 5, i.e. >= R+5) = {(med >= -5).mean():.3f}")

    def report(mask, label):
        k = hs[mask]
        print(f"  {label:58s} share of sims {mask.mean():6.3f}  House E={k.mean():5.1f}  "
              f"P(maj)={(k >= 76).mean():.3f}  p10-p90 {np.percentile(k, 10):.0f}-{np.percentile(k, 90):.0f}")

    print("\nConditioned on the simulated Tier 1 outcomes:")
    report(np.ones_like(hs, dtype=bool), "unconditional")
    report(med >= -5, "Tier 1 median margin R+5 or closer")
    report((med >= -5) & (med <= 5), "Tier 1 median margin within +/-5")
    report((med >= -5) & (med <= 0), "Tier 1 median margin R+5 to even")
    report((med >= -2.5) & (med <= 2.5), "Tier 1 median margin within +/-2.5")
    report(mean >= -5, "Tier 1 mean margin R+5 or closer")
    frac_close = (np.abs(t1_margin) <= 5).mean(axis=0)
    report(frac_close >= 0.5, "at least half of Tier 1 seats within +/-5")
    report(frac_close >= 0.75, "at least 3/4 of Tier 1 seats within +/-5")

    # Env dial that makes the Tier 1 median expected margin R+5 / R+2.5 / even
    print("\nDial solve: environment at which Tier 1's median expected margin is X")
    for env_try in np.arange(9, 17.01, 0.5):
        lin = model.build_linear_predictions(df, env_try, model.RACE_GENERIC_BALLOT_D_SHARE).values
        m = (2 * lin[t1] - 1) * 100
        _, _, w, _ = simulate(df, env_try, n=20000)
        k = w[house].sum(axis=0)
        print(f"  env D+{env_try:4.1f}: Tier1 median {np.median(m):5.1f} mean {m.mean():5.1f} "
              f"n within 5 {int((np.abs(m) <= 5).sum()):2d}  House E={k.mean():.1f} P(maj)={(k >= 76).mean():.3f}")

    h.to_csv(ROOT / "output" / "bullpen_compare_house.csv", index=False)


if __name__ == "__main__":
    main()
