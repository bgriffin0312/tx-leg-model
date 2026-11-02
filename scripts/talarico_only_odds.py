import numpy as np
import pandas as pd

OUT = "C:/Users/Brennan/code/tx-leg-model/output/"

rows = pd.read_csv(OUT + "whole_election_races_2026.csv")
print(rows.kind.value_counts().to_string())
statewide_names = rows[rows.kind == "statewide"].race.unique().tolist()
print("statewide races:", statewide_names)

for env in ("statewide", "legislative"):
    z = np.load(f"{OUT}whole_election_{env}_2026.npz", allow_pickle=True)
    names = [str(n) for n in z["names"]]
    W = z["wins"].astype(bool)
    print("\n==", env, "draws", len(W))
    r = rows[(rows.env == env) & (rows.kind == "statewide")].sort_values("p_d", ascending=False)
    print(r[["race", "mean", "sd", "p_d", "T"]].to_string(index=False))

    idx = {n: names.index(n) for n in statewide_names if n in names}
    missing = [n for n in statewide_names if n not in names]
    print("not in draws:", missing)
    others = [n for n in idx if n != "us_senate"]
    tal = W[:, idx["us_senate"]]
    any_other = W[:, [idx[n] for n in others]].any(axis=1)
    n_d = W[:, list(idx.values())].sum(axis=1)
    print("P(Talarico wins):", round(tal.mean(), 4))
    print("P(Talarico only D statewide winner):", round((tal & ~any_other).mean(), 4))
    print("P(Talarico + at least one other):", round((tal & any_other).mean(), 4))
    print("P(no D wins statewide):", round((n_d == 0).mean(), 4))
    print("P(a D wins statewide but Talarico loses):", round((~tal & any_other).mean(), 4))
    print("P(only | Talarico wins):", round((tal & ~any_other).sum() / tal.sum(), 4))
    print("distribution of D statewide wins:", dict(zip(*[x.tolist() for x in np.unique(n_d, return_counts=True)])))
    print("when Talarico wins, who else wins (conditional):")
    for n in others:
        print("  ", n, round(W[tal, idx[n]].mean(), 4))
