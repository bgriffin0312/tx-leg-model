"""Brennan's AG anchor (2026-09-10): use the Texas AG race banner as the Texas
equivalent of the national generic ballot by race, and correct the national
numbers where they are more Democratic for Hispanic voters than the AG race.

Prints, for the 2026 inputs on disk:
  1. national generic ballot by race, mean of the 2026 polls with a banner
  2. Texas banners: AG, Comptroller, TX-legislature generic (Jun, Aug 2026)
  3. Texas offset g[r] = Texas - national, per group
  4. shift since 2024 by group, national-implied vs Texas-anchored, against
     the reconciled 2024 Texas benchmark (proposal §2b)
  5. the centered shift term a district would receive (proposal §3a), at
     90% / 50% / 15% Hispanic CVAP
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
G = ["white_nh", "black_nh", "hispanic"]

# Reconciled 2024 Texas benchmark by race (docs/poll-integration-proposal.md §2b)
TX_2024 = {"white_nh": .314, "black_nh": .85, "hispanic": .506}
# CES validated 2024 Texas electorate (same doc); other (.064) carried at zero shift
W_TX = {"white_nh": .627, "black_nh": .115, "hispanic": .194}
# National 2024 by race, Catalist What Happened 2024 (proposal §5 CATALIST_PRES analogue)
NAT_2024 = {"white_nh": .43, "black_nh": .85, "hispanic": .54}


def two_party(df):
    return {g: df[c].mean() for g, c in zip(G, ["white_d_2p", "black_d_2p", "hispanic_d_2p"])}


def main():
    nat = pd.read_csv(RAW / "racial_crosstab_inputs.csv")
    nat = nat[nat["poll_date"] >= "2026-01-01"]
    nat_now = two_party(nat)
    print(f"1. National generic by race, mean of {len(nat)} 2026 polls "
          f"({', '.join(nat['pollster'])}):")
    print("   " + "  ".join(f"{g} {v * 100:.1f}" for g, v in nat_now.items()))

    tx = pd.read_csv(RAW / "texas_crosstab_inputs.csv")
    tx = tx[(tx["poll_date"] >= "2026-01-01") & (tx["pollster"].str.startswith("UT"))]
    print("\n2. UT/TPP Texas banners 2026 (RV), D two-party %:")
    rows = {}
    for (date, ballot), d in tx.groupby(["poll_date", "ballot"]):
        v = two_party(d)
        rows[(date, ballot)] = v
        print(f"   {date} {ballot:24s} " + "  ".join(f"{g} {x * 100:.1f}" for g, x in v.items()))
    ag_aug = rows[("2026-08-13", "attorney_general")]
    down = pd.DataFrame([rows[k] for k in rows if k[1] in ("attorney_general", "comptroller", "tx_legislature_generic")])
    down_mean = down.mean().to_dict()
    print("   mean of AG + Comptroller + lege generic, Jun+Aug:  "
          + "  ".join(f"{g} {x * 100:.1f}" for g, x in down_mean.items()))

    print("\n3. Texas offset g[r] = Texas - national (pp):")
    for label, txv in (("AG Aug 2026", ag_aug), ("downballot mean Jun+Aug", down_mean)):
        off = {g: (txv[g] - nat_now[g]) * 100 for g in G}
        rel = {g: off[g] - off["white_nh"] for g in G}
        print(f"   {label:26s} " + "  ".join(f"{g} {off[g]:+.1f}" for g in G)
              + "   | relative to white: " + "  ".join(f"{g} {rel[g]:+.1f}" for g in G))

    print("\n4. Shift since 2024 by group (pp):")
    nat_shift = {g: (nat_now[g] - NAT_2024[g]) * 100 for g in G}
    print("   national-implied (2026 national - 2024 national):   "
          + "  ".join(f"{g} {nat_shift[g]:+.1f}" for g in G))
    for label, txv in (("AG Aug 2026", ag_aug), ("downballot mean Jun+Aug", down_mean)):
        sh = {g: (txv[g] - TX_2024[g]) * 100 for g in G}
        mean = sum(W_TX[g] * sh[g] for g in G) / sum(W_TX.values())
        cen = {g: sh[g] - mean for g in G}
        print(f"   Texas-anchored, {label:26s} " + "  ".join(f"{g} {sh[g]:+.1f}" for g in G)
              + f"   | uniform part {mean:+.1f}, centered: " + "  ".join(f"{g} {cen[g]:+.1f}" for g in G))
        print("      district term at 90/50/15% Hispanic (rest white): "
              + "  ".join(f"{h:.0%}: {(h * cen['hispanic'] + (1 - h) * cen['white_nh']):+.1f}pp"
                          for h in (0.9, 0.5, 0.15)))
    print("\n   Caveats: UT banners are registered voters with ~24% of Hispanics undecided;"
          " the 2024 benchmark's Black entry is a national assumption; Hispanic cell n≈350."
          " June vs August AG Hispanic differ by 5pp, which is the noise band.")


if __name__ == "__main__":
    main()
