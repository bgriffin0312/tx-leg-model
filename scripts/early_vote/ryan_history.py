"""Primary history on one definition: Derek Ryan's voter-file counts, 2016 to 2026.

Reports are in data/raw/early_vote/ryan/. Final electorates by history come from his 2026 Day 12 report
(2018, 2022) and 2024 final report (2016, 2020, 2024); each election's labels were set on that cycle's
file, from the previous four cycles. R group = Rep (No D) + Mixed/R, D group = Dem (No R) + Mixed/D.
His "advantage" drops the mixed groups in the 2024 report and keeps them in 2026; this keeps them.

A. The panel. B. No-primary margin backed out of each result. C. 2026 final by Election Day conversion.
D. 2026 read at each past no-primary margin, on the labels as they stand and with the March relabeling
taken out. E. What the 2026 readings need. F. Top 15 counties at Day 5, the one same-point 2018 figure.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypdf

pd.set_option("display.width", 200, "display.max_columns", 30)
EV = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote"
COLS = ["Rep", "MixR", "Dem", "MixD", "Gen", "None"]
FINAL = {2016: (2932718, 142950, 1502154, 81959, 2682590, 1682865),
         2018: (2751482, 98216, 1583969, 103474, 2795595, 1054029),
         2020: (3041512, 77130, 2292239, 235560, 3563868, 2097816),
         2022: (2817822, 135624, 1915678, 162290, 2495980, 567780),
         2024: (3017809, 173698, 2063521, 84728, 4367919, 1610564)}
EARLY_26 = (2354516, 106695, 2354725, 132322, 2206553, 570741)
# D minus R. 2018, 2022, 2024 as used in docs/whole-election-model.md; 2016, 2020 judicial from county_panel.csv
TOP = {2016: -9.0, 2018: -2.6, 2020: -5.6, 2022: -11.04, 2024: -13.88}
J = {2016: -15.5, 2018: -7.3, 2020: -9.4, 2022: -13.96, 2024: -16.35}
GOV18 = -13.3
A_RANGE = (0.85, 0.92, 1.0)

print("=== A. Final electorates ===")
p = pd.DataFrame(FINAL, index=COLS).T
p["total"] = p[COLS].sum(axis=1)
p["R_group"], p["D_group"], p["N"] = p.Rep + p.MixR, p.Dem + p.MixD, p.Gen + p["None"]
p["adv"] = p.R_group - p.D_group
p["gap"], p["Nshare"] = p.adv / p.total * 100, p.N / p.total * 100
p["Rshare"], p["Dshare"] = p.R_group / p.total * 100, p.D_group / p.total * 100
p["top"], p["J"] = pd.Series(TOP), pd.Series(J)
print(p[["total", "adv", "gap", "Rshare", "Dshare", "Nshare", "top", "J"]].round(1).to_string())
print("corr gap, top", round(np.corrcoef(p.gap, p.top)[0, 1], 2), " gap, J", round(np.corrcoef(p.gap, p.J)[0, 1], 2), "(five points)")

print("\n=== B. No-primary margin implied, D minus R, by net carry of the primary gap ===")
N = {}
for name, res in (("top", p.top), ("J", p.J)):
    for a in A_RANGE:
        N[(name, a)] = (res + a * p.gap) / (p.Nshare / 100)
N = pd.DataFrame(N)
print(N.round(1).to_string())
print("2018 Governor, same voters:", {a: round((GOV18 + a * p.gap[2018]) / (p.Nshare[2018] / 100), 1) for a in A_RANGE})

print("\n=== C. 2026 final by Election Day conversion ===")
elig = {"1R": 1370180, "2R": 806738, "3R": 676127, "4R": 718655, "1D": 1711572, "2D": 722050, "3D": 450309, "4D": 421757}
vote = {"1R": 746936, "2R": 554803, "3R": 531061, "4R": 628411, "1D": 1134860, "2D": 572591, "3D": 387397, "4D": 392199}
early22 = {"1R": 46, "2R": 62, "3R": 72, "4R": 81, "1D": 43, "2D": 62, "3D": 75, "4D": 87}
final22 = {"1R": 68, "2R": 84, "3R": 91, "4R": 97, "1D": 62, "2D": 79, "3D": 88, "4D": 96}
final18 = {"1R": 75, "2R": 88, "3R": 92, "4R": 98, "1D": 76, "2D": 88, "3D": 90, "4D": 95}
gen = {"elig": 6735940, "vote": 2206553, "e22": 23, "f22": 40, "f18": 54}
non = {"elig": 5572635, "vote": 570741, "e22": 6, "f22": 11, "f18": 19}


def sides(fin):
    return sum(fin[k] for k in fin if k[1] == "R"), sum(fin[k] for k in fin if k[1] == "D")


rows = []
# 2022: of those who had not voted by Day 12, the share who voted on Election Day
conv = {k: (final22[k] - early22[k]) / (100 - early22[k]) for k in elig}
fin = {k: vote[k] + (elig[k] - vote[k]) * conv[k] for k in elig}
n = sum((g["elig"] - g["vote"]) * (g["f22"] - g["e22"]) / (100 - g["e22"]) + g["vote"] for g in (gen, non))
rows.append(("2022 conversion", *sides(fin), n))
# 2018 final rates, never below what has already voted
fin = {k: max(vote[k], elig[k] * final18[k] / 100) for k in elig}
for label, tgt in (("2018 final rates, no-primary 2022 conversion", n), ("2018 final rates, no-primary 2018 rates", sum(max(g["vote"], g["elig"] * g["f18"] / 100) for g in (gen, non)))):
    rows.append((label, *sides(fin), tgt))
# 2024's additions as multipliers on each group
e24 = (2696638, 157011, 1830886, 78145, 3254931, 1048215)
f24 = FINAL[2024]
m = [f / e for f, e in zip(f24, e24)]
v = [x * y for x, y in zip(EARLY_26, m)]
rows.append(("2024 multipliers", v[0] + v[1], v[2] + v[3], v[4] + v[5]))
c = pd.DataFrame(rows, columns=["how", "R", "D", "N"])
c["total_m"] = (c.R + c.D + c.N) / 1e6
c["adv"] = c.R - c.D
c["gap"] = c.adv / (c.R + c.D + c.N) * 100
c["Nshare"] = c.N / (c.R + c.D + c.N) * 100
print(c.round(1).to_string(index=False))

print("\n=== D. 2026 read at each past no-primary margin, net carry .92 ===")
# Primary voters with no prior primary, Ryan's post-election reports. 2018: 27.0% of D, 14.1% of R. 2026: 37%, 19%.
new = {2018: (0.270 * 1042914, 0.141 * 1545424), 2026: (0.37 * 2311826, 0.19 * 2165744)}
net = {y: d - r for y, (d, r) in new.items()}
excess = (net[2026] - net[2018]) * 0.9
print("net relabeled to D:", {y: round(v) for y, v in net.items()}, "excess voters in the electorate", round(excess))
a = 0.92
rows = []
for how, row in c.set_index("how").iterrows():
    t = row.R + row.D + row.N
    for lab, gap, sh in (("labels as they stand", row.gap, row.Nshare), ("March relabeling out", row.gap + excess / t * 100, row.Nshare + excess / t * 100)):
        rows.append({"how": how, "labels": lab, "gap": gap, "Nshare": sh, **{f"J at {y}": -a * gap + sh / 100 * N.loc[y, ("J", a)] for y in FINAL}, **{f"top at {y}": -a * gap + sh / 100 * N.loc[y, ("top", a)] for y in (2018, 2022)}})
d = pd.DataFrame(rows)
print(d.round(1).to_string(index=False))

print("\n=== E. No-primary margin the 2026 readings need, net carry .85 to 1.0 ===")
for lab, res in (("Senate D+0.9", 0.9), ("Governor R+4.6", -4.6), ("T -6.0", -6.0)):
    z = [(res + a_ * row.gap) / (row.Nshare / 100) for a_ in A_RANGE for _, row in d[d.labels == "labels as they stand"].iterrows()]
    z2 = [(res + a_ * row.gap) / (row.Nshare / 100) for a_ in A_RANGE for _, row in d[d.labels == "March relabeling out"].iterrows()]
    print(f"{lab:<16} as they stand {min(z):+.1f} to {max(z):+.1f}; relabeling out {min(z2):+.1f} to {max(z2):+.1f}")

print("\n=== F. Top 15 counties, Day 5 ===")
t = max(pd.read_html(io.StringIO((EV / "sos" / "2018_nov2.html").read_text(encoding="utf-8"))), key=lambda q: q.shape[0])
t = t[t.County.str.lower() != "total"].assign(c=lambda x: x.County.str.upper().str.strip())
top15 = list(t.sort_values("Reg Voters").tail(15).c)
print(top15)
row = re.compile(r"^([A-Z][A-Z .]+?) ([\d,]+) ([\d,]+) (\d+)% (\d+)% (\d+)% (\d+)% (\d+)% (\d+)% (\d+)%$")


def county5():
    rec = []
    for pg in pypdf.PdfReader(EV / "ryan" / "2026_county_day5.pdf").pages:
        for line in pg.extract_text().splitlines():
            m = row.match(line.strip())
            if m and m[1] != "TOTAL":
                v = int(m[3].replace(",", ""))
                rec.append((m[1], v, *[int(m[i]) / 100 * v for i in range(5, 11)]))
    return pd.DataFrame(rec, columns=["c", "votes", *COLS]).set_index("c")


def county12():
    # the Day 12 file carries counts, not only shares
    rec = []
    for pg in pypdf.PdfReader(EV / "ryan" / "2026_county_day12.pdf").pages:
        for line in pg.extract_text().splitlines():
            m = re.match(r"^([A-Z][A-Z .]+?) ?([\d,]+) ([\d,]+) \d+% ([\d,]+) ([\d,]+) ([\d,]+) ([\d,]+) ([\d,]+) ([\d,]+) ", line.strip())
            if m and m[1] != "TOTAL":
                rec.append((m[1], int(m[3].replace(",", "")), *[int(m[i].replace(",", "")) for i in range(4, 10)]))
    return pd.DataFrame(rec, columns=["c", "votes", *COLS]).set_index("c")


for lab, k in (("Day 5", county5()), ("Day 12", county12())):
    x = k.loc[top15]
    adv = (x.Rep + x.MixR - x.Dem - x.MixD).sum()
    all_ = (k.Rep + k.MixR - k.Dem - k.MixD).sum()
    print(f"{lab}: counties {len(k)}, state votes {k.votes.sum():,}, state R minus D {all_:,.0f}; top 15 votes {x.votes.sum():,}, R minus D {adv:,.0f} ({adv / x.votes.sum() * 100:+.1f}), no history {x['None'].sum() / x.votes.sum() * 100:.1f}%; rest {all_ - adv:,.0f}")
print("2018 top 15 at Day 5, Ryan via Tribune: R+90,000, no history 8.5%; four days, KXAN: R 35, D 32, general only about 24, none 8")
print("2018 top 15 final vote, Ryan: O'Rourke 57.3 to 41.9; rest of state Cruz 68.6 to 30.6")
