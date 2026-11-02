import numpy as np
import pandas as pd

OUT = "C:/Users/Brennan/code/tx-leg-model/output/"

# R-held seats on the 2026 ballot; SD 9, 13, 19, 21, 26 are the D-held on-ballot seats
D_HELD = {"sd9", "sd13", "sd19", "sd21", "sd26"}

for env in ("statewide", "legislative"):
    z = np.load(f"{OUT}whole_election_{env}_2026.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    W = z["wins"].astype(bool)
    sd = [n for n in names if n.startswith("sd")]
    idx = {n: names.index(n) for n in sd}
    r_held = [n for n in sd if n not in D_HELD]
    flips = sum(W[:, idx[n]].astype(int) for n in r_held)
    d_losses = sum((~W[:, idx[n]]).astype(int) for n in D_HELD)
    print(env, "draws", len(W), "seats", len(sd), "r_held", len(r_held))
    print("  P(any R-held seat flips D):", round((flips >= 1).mean(), 4))
    print("  mean R seats lost:", round(flips.mean(), 4))
    print("  P(2+ R seats lost):", round((flips >= 2).mean(), 4))
    print("  P(Rehmet holds SD 9):", round(W[:, idx["sd9"]].mean(), 4))
    print("  P(D lose SD 19):", round((~W[:, idx["sd19"]]).mean(), 4))
    print("  P(net D gain, seats>12):", round((z["senate"] > 12).mean(), 4))
    print("  senate seats dist:", dict(zip(*[x.tolist() for x in np.unique(z["senate"], return_counts=True)])))
    print("  mean senate:", round(z["senate"].mean(), 3))

df = pd.read_csv(OUT + "model_2026_scenarios.csv")
s = df[(df.chamber == "Senate") & (df.env_dial == 9.2)].sort_values("win_prob_d", ascending=False)
print(s[["district", "incumbent", "incumbent_party", "win_prob_d"]].to_string(index=False))
r = s[(s.incumbent_party != "D")]
print("independent-seat P(any R-held flips), published D+9.2:", round(1 - (1 - r.win_prob_d).prod(), 4))
print("expected R-held flips, published D+9.2:", round(r.win_prob_d.sum(), 4))
