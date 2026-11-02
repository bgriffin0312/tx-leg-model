"""Secretary of State archived final-day early-vote tables, top counties, 2012-2022."""
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

S = Path(__file__).resolve().parents[2] / "data" / "raw" / "early_vote" / "sos"
S.mkdir(exist_ok=True, parents=True)
TRY = {2012: ["nov2"], 2014: ["oct31"], 2016: ["nov4"], 2018: ["nov2"], 2020: ["oct30"], 2022: ["nov4"], 2024: ["nov1"]}
H = {"User-Agent": "Mozilla/5.0"}
for y, names in TRY.items():
    for n in names:
        for base in (f"https://www.sos.state.tx.us/elections/earlyvoting/{y}/{n}.shtml", f"https://www.sos.texas.gov/elections/earlyvoting/{y}/{n}.shtml"):
            f = S / f"{y}_{n}.html"
            try:
                r = requests.get(base, headers=H, timeout=40)
            except Exception as e:
                print(y, n, "err", e)
                continue
            print(y, n, r.status_code, len(r.text), base)
            if r.status_code == 200 and len(r.text) > 5000:
                f.write_text(r.text, encoding="utf-8")
                break
for f in sorted(S.glob("*.html")):
    try:
        ts = pd.read_html(io.StringIO(f.read_text(encoding="utf-8")))
    except Exception as e:
        print(f.name, "no tables", e)
        continue
    print("\n", f.name, [t.shape for t in ts])
    big = max(ts, key=lambda t: t.shape[0])
    print(big.head(6).to_string())
    print(big.tail(3).to_string())
