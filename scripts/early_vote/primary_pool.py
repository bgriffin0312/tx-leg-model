"""What the primary-voter count is worth.

Early counts by primary history, then Election Day's addition to each group from 2022 and 2024 on
one definition (theoelections, primaries 2020-2026, voters as of Oct 30). Then the margin non-primary
voters would need for a given statewide result. Loyalty a is the net share of the primary-voter gap
that carries to the vote; the 2022 and 2024 no-primary margins are backed out the same way.
Pool table: Transparency USA as quoted in search results, undated, about Oct 31; its page refuses
fetches, so treat as approximate. Its definition is not the same as theoelections'.
For the 2016-2026 panel on one definition, including 2018, see ryan_history.py.
"""
import pandas as pd

pd.set_option("display.width", 200)
H = {2022: {"all": dict(D=26.1, R=35.7, B=2.6, N=35.5, n=8084081, res=-11.0), "early": dict(D=30.0, R=39.0, B=3.1, N=27.9, n=5497479)},
     2024: {"all": dict(D=21.8, R=27.6, B=1.9, N=48.6, n=11320149, res=-13.8), "early": dict(D=24.0, R=31.0, B=2.2, N=42.8, n=9052401)}}
E26 = dict(D=28.3, R=29.9, B=2.5, N=39.4, n=7744549)
ED26 = (2.2e6, 2.6e6, 3.0e6)

print("=== Pool, Transparency USA ===")
R_v, R_n, D_v, D_n = 2151514, 1274352, 2.2e6, 977000
print(f"R only: {R_v / (R_v + R_n):.1%} voted, {R_n:,} left. D only: {D_v / (D_v + D_n):.1%} voted, {D_n:,.0f} left. Left gap R+{R_n - D_n:,.0f}")

print("\n=== Election Day's multiplier on each group, all voters / early voters ===")
mult = {}
for y, h in H.items():
    m = {k: (h["all"][k] * h["all"]["n"]) / (h["early"][k] * h["early"]["n"]) for k in "DRBN"}
    mult[y] = m
    print(y, {k: round(v, 3) for k, v in m.items()}, "early share of group voters", {k: round(1 / v, 3) for k, v in m.items()})

print("\n=== 2026 final, by year's multipliers (scaled to each total) ===")
rows = []
c26 = {k: E26[k] / 100 * E26["n"] for k in "DRBN"}
for y, m in mult.items():
    for ed in ED26:
        extra = {k: c26[k] * (m[k] - 1) for k in "DRBN"}
        s = (ed / sum(extra.values()))
        for scaled, f in (("group mix, ED size set", s), ("multipliers as they stand", 1.0)):
            fin = {k: c26[k] + extra[k] * f for k in "DRBN"}
            t = sum(fin.values())
            rows.append({"mult": y, "ED_target_m": ed / 1e6, "how": scaled, "total_m": t / 1e6, "R_minus_D": (fin["R"] - fin["D"]) / t * 100, "N_share": fin["N"] / t * 100, "D": fin["D"] / 1e6, "R": fin["R"] / 1e6})
t = pd.DataFrame(rows)
print(t.round(2).to_string(index=False))

print("\n=== Implied no-primary margin, past years ===")
for y, h in H.items():
    a_ = h["all"]
    print(y, {a: round((a_["res"] + a * (a_["R"] - a_["D"])) / (a_["N"] / 100), 1) for a in (0.85, 0.92, 1.0)})

print("\n=== No-primary margin needed in 2026, primary gap R+0.7 to R+2.4, no-primary share 45 to 47 ===")
rows = []
for gap in (0.7, 1.5, 2.4):
    for N in (45.0, 47.0):
        for a in (0.85, 0.92, 1.0):
            for lab, res in (("Senate D+0.9", 0.9), ("Governor R+4.6", -4.6), ("Senate D+2.1", 2.1), ("Governor R+5.8", -5.8)):
                rows.append({"gap": gap, "N": N, "a": a, "race": lab, "need": (res + a * gap) / (N / 100)})
x = pd.DataFrame(rows)
print(x.groupby(["race", "gap"]).need.agg(["min", "mean", "max"]).round(1).to_string())
