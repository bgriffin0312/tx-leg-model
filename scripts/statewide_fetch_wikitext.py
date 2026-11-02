"""
statewide_fetch_wikitext.py

Cache raw wikitext for the Texas statewide race articles (2010-2026) that feed
the statewide model. Results, nominees and poll tables are parsed from these
caches by src/statewide_collect.py, so a re-parse never needs the network.

Writes data/raw/statewide/wikitext/<title>.txt (gitignored with the rest of data/).
"""

import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "statewide" / "wikitext"
UA = {"User-Agent": "tx-leg-model research (personal project; contact via github.com/bgriffin0312)"}

TITLES = [f"{y} Texas elections" for y in range(2010, 2028, 2)]
TITLES += [
    "2026 Texas gubernatorial election",
    "2026 Texas lieutenant gubernatorial election",
    "2026 Texas Attorney General election",
    "2026 Texas Comptroller of Public Accounts election",
    "2026 United States Senate election in Texas",
    "2026 Texas Commissioner of Agriculture election",
    "2026 Texas Commissioner of the General Land Office election",
    "2026 Texas Railroad Commissioner election",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for title in TITLES:
        r = requests.get("https://en.wikipedia.org/w/index.php",
                         params={"title": title, "action": "raw"}, headers=UA, timeout=60)
        path = OUT / (title.replace(" ", "_") + ".txt")
        if r.status_code == 200 and r.text.strip():
            path.write_text(r.text, encoding="utf-8")
            print(f"{r.status_code} {len(r.text):>8,} {title}")
        else:
            print(f"{r.status_code} MISSING  {title}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
