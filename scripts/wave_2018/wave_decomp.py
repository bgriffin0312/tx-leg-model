"""Where does the 2018 undershoot live? Every contested House seat, held-out formula, T at the
judicial vote that happened, so polls are out of it."""
import sys

sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\scripts")
sys.path.insert(0, r"C:\Users\Brennan\code\tx-leg-model\src")
import numpy as np
import pandas as pd

import whole_election_backtest as b

pd.set_option("display.width", 220, "display.max_rows", 300, "display.max_columns", 40)
sw, rc = b.sw, b.rc

for y in (2018, 2022, 2024):
    c = b.leg_fit(y)
    a = b.leg_frame(y)
    a = a[a.chamber_lower == "house"].copy()
    env = sw.NAT[y]
    a["lin"] = (rc.predict(a, c) + b.LEG_SLOPE * (env - b.ENV_BASE[y])) * 200 - 100
    a["act"] = a.dem_2p_share * 200 - 100
    m = a[a.contested & a.x.notna()].copy()
    m["res"] = m.act - m.lin
    print(f"\n===== {y}: env {env:+.1f}, contested {len(m)}, level miss (act-lin) {m.res.mean():+.2f}, sd {m.res.std():.2f}")
    m["band"] = pd.cut(m.lin, [-100, -30, -20, -12, -6, 0, 6, 12, 20, 100])
    print(m.groupby("band", observed=True).agg(n=("res", "size"), lin=("lin", "mean"), res=("res", "mean"), d_won=("act", lambda s: (s > 0).sum()), rinc=("rep_incumbent", "sum")).round(2).to_string())
    # OLS of residual on features
    import statsmodels.api as sm
    X = m[["c_white_col", "c_hispanic", "c_black", "dem_incumbent", "rep_incumbent", "lin"]].astype(float)
    print(sm.OLS(m.res, sm.add_constant(X)).fit().summary2().tables[1][["Coef.", "Std.Err.", "t"]].round(3).to_string())
    print("every seat where the sign of lin was wrong:")
    w = m[np.sign(m.lin) != np.sign(m.act)]
    print(w.sort_values("lin")[["district", "lin", "act", "res", "dem_incumbent", "rep_incumbent", "c_white_col", "c_hispanic", "challenger_viability_flag", "ie_c"]].round(2).to_string(index=False))
    print("linear seats among contested:", int((m.lin > 0).sum()), "actual:", int((m.act > 0).sum()))
    # how much is level, how much is shape: re-center lin by the level miss, count seats
    print("seats if lin shifted by level miss:", int(((m.lin + m.res.mean()) > 0).sum()))
    for k in (1.0, 1.2, 1.5, 2.0):
        pass
