"""Model-implied partisan bias: seats D wins when the model's statewide House vote is 50/50.

Statewide vote = each district's predicted D 2p share weighted by 2024 presidential
two-party turnout (proxy for 2026 turnout). Unopposed seats get 1.0 / 0.0 like a real
vote count would, plus a variant that drops them.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model as M  # noqa: E402

df = M.load_districts()
race_generic = M.RACE_GENERIC_BALLOT_D_SHARE
house = (df["chamber_lower"] == "house").values

pres = pd.read_csv(ROOT / "data" / "raw" / "tx_presidential_house_2024.csv")
w_map = dict(zip(pres["district"], pres["trump_votes"] + pres["harris_votes"]))
w = df.loc[house, "district"].astype(int).map(w_map).values.astype(float)
forced = df.loc[house, "forced_d_win"].values

rows = []
for env in np.arange(0, 30.5, 0.5):
    lin = M.build_linear_predictions(df, env, race_generic).values[house]
    contested = np.isnan(forced)
    share_all = np.where(contested, lin, forced)
    sw_all = np.average(share_all, weights=w)
    sw_cont = np.average(lin[contested], weights=w[contested])
    det_seats = int(((lin > 0.5) & contested).sum() + (forced == 1).sum())
    sw_latent = np.average(lin, weights=w)  # every seat contested: model share everywhere
    rows.append((env, sw_all, sw_cont, det_seats, sw_latent))

t = pd.DataFrame(rows, columns=["env", "sw_d2p_unopp_as_votes", "sw_d2p_contested_only", "det_seats",
                                "sw_latent"])
print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("unopposed D:", int((forced == 1).sum()), " unopposed R:", int((forced == 0).sum()))

for col in ["sw_d2p_unopp_as_votes", "sw_d2p_contested_only", "sw_latent"]:
    e50 = float(np.interp(0.5, t[col], t["env"]))
    res = M.run_monte_carlo(df, e50, race_generic, M.N_SIMULATIONS)
    print(f"{col}: 50/50 statewide at env D+{e50:.1f} -> expected D seats "
          f"{res['expected_house_seats']:.1f}, P(maj) {res['house_control_prob']:.3f}")
cur = float(np.interp(M.CURRENT_ENV, t["env"], t["sw_d2p_unopp_as_votes"]))
print(f"current env D+{M.CURRENT_ENV}: implied TX statewide D2p {cur:.4f}")
