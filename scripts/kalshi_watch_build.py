"""Build output/kalshi_watch.csv: every open Kalshi market that is purely about Texas politics.

Finds events by keyword, then keeps winner and event markets only. Dropped, with a count printed:
- anything that also depends on another state or the nation (2028 nominees, Maine-Texas combo, closer-race pairs)
- the 2028 Texas legislative district markets, which have no trading and settle after 2026
- bracket families (margin, vote percent, turnout, county), which are thin and would make up most of the list

Model probabilities come from output/statewide_forecast_2026.csv, output/congress_forecast_2026.csv
and output/county_judge_forecast_2026.csv.
Re-run after a model run so alerts quote the new numbers.
"""
import csv
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "kalshi_watch.csv"

KEYWORDS = re.compile(r"\btexas\b|\btx\b|TX-?\d|paxton|talarico|abbott|hinojosa|harris county|montgomery county"
                      r"|tarrant county|bexar county|williamson county|fort bend county|el paso county|denton county|collin county"
                      r"|dallas county|travis county|hidalgo county|cameron county|nueces county|galveston county|brazoria county|hays county", re.I)
OTHER_PLACES = re.compile(r"KXPRESNOM|KXVPRES|KXMETX|KXCLOSER|KXRAMAN|KXREDDEST|KXFOURSTATE|KXSENATEWITHIN|KXBERNIE|KXOBAMA"
                          r"|KXREDISTRICT|KXNGUARD|KXNEXTSTATE|KXACTBLUE|KXLAPRIMARY|KXHOUSERACE-(?!TX)|KXVOTEGENERAL-KXHOUSERACE-(?!TX)"
                          r"|KXSTATELEGTX-28")
BRACKETS = re.compile(r"KXMIDTERMMOV|KXGENERALMOV|KXVOTEGENERAL|KXVOTECOUNTY|KXMIDTERMVOTETURN|KXGENERALCOUNTY")

# Kalshi market -> statewide race in the model, with the side that Yes pays on
STATEWIDE = {"SENATETX-26": "us_senate", "GOVPARTYTX-26": "governor", "KXLTGOVTX-26": "lt_governor", "KXATTYGENTX-26": "attorney_general",
             "KXTXCOMPTROLLER-26": "comptroller", "KXTXGLO-26": "land_commissioner", "KXTXRAILROAD-26": "railroad_commissioner", "KXTXAGCOM-26": "ag_commissioner"}
DEM_MARKETS = {"SENATETX-26-D", "GOVPARTYTX-26-D", "KXLTGOVTX-26-DEM", "KXATTYGENTX-26-D", "KXTXCOMPTROLLER-26-SECK", "KXTXGLO-26-BFLO", "KXTXRAILROAD-26-JROS", "KXTXAGCOM-26-CTUC"}
LEGISLATIVE = {"KXSTATELEG-TXHOUS26-R": 0.851, "KXSTATELEG-TXSENA26-R": 0.997}  # same legislative figures as scripts/paper_bets.py
NOTES = {"KXTXSTATEWIDEDEMS-26NOV03-AL1": "You hold 59 Yes, hedged with 59 No on Any Dem.", "KXANYDEMWINTEXAS-26NOV03": "You hold 59 No, hedged with 59 Yes on At Least 1.",
         "KXWILCOJUDGE-26-SSNE": "You hold 24 No, average 60.5c with fees.", "KXWILCOJUDGE-26-HJEF": "You hold 24 No on Snell, the same side as Jefts Yes.",
         "KXTXRAILROAD-26-JROS": "You hold 25 No, average 42.7c with fees.", "KXTXRAILROAD-26-BFRE": "You hold 25 No on Rosenthal, the same side as French Yes."}
LABELS = {"KXTXSTATEWIDEDEMS-26NOV03-AL1": "At Least 1 statewide Dem", "KXANYDEMWINTEXAS-26NOV03": "Any Dem statewide"}


COUNTY_EVENTS = {"KXTARRANTCOUNTYJUDGE-26": "tarrant", "KXWILCOJUDGE-26": "williamson", "KXCOLLINCOUNTYJUDGE-26": "collin", "KXDENTONCOUNTYJUDGE-26": "denton",
                 "KXFORTBENDCOUNTYJUDGE-26": "fort bend", "KXHARRISCOUNTYJUDGE-26": "harris", "KXBEXARCOUNTYJUDGE-26": "bexar",
                 "KXMONTGOMERYCOUNTYJUDGE-26": "montgomery", "KXELPASOCOUNTYJUDGE-26": "el paso"}


def county_p(name, event, cj):
    """County judge races from output/county_judge_forecast_2026.csv. Bexar DA is left out: it has an independent, and only one number."""
    row = cj[(cj.county == COUNTY_EVENTS[event]) & (cj.office == "county_judge")].iloc[0]
    if row.d in name:
        return float(row.p_d)
    return 1 - float(row.p_d) if row.r in name else None


def get(path, **params):
    for attempt in range(8):
        r = requests.get(f"{BASE}{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"rate limited on {path}")


def open_events():
    events, cursor = [], None
    while True:
        params = {"status": "open", "limit": 200, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        j = get("/events", **params)
        events += j.get("events", [])
        cursor = j.get("cursor")
        if not cursor or not j.get("events"):
            return events


def model_p(ticker, event, sw, cg):
    if ticker in LEGISLATIVE:
        return LEGISLATIVE[ticker]
    if event in STATEWIDE and STATEWIDE[event] in sw.index:
        p_d = float(sw.p_d[STATEWIDE[event]])
        return p_d if ticker in DEM_MARKETS else 1 - p_d
    m = re.search(r"TX(\d+)-26-(D|R)$", ticker)
    if m and int(m[1]) in cg.index:
        p_d = float(cg.p_d[int(m[1])])
        return p_d if m[2] == "D" else 1 - p_d
    return None


def main():
    sw = pd.read_csv(ROOT / "output" / "statewide_forecast_2026.csv").set_index("race")
    cg = pd.read_csv(ROOT / "output" / "congress_forecast_2026.csv").set_index("dist")
    cj = pd.read_csv(ROOT / "output" / "county_judge_forecast_2026.csv")
    rows, dropped = [], {"other_places": 0, "brackets": 0}
    for e in open_events():
        ticker = e["event_ticker"]
        text = " ".join([e.get("title") or "", ticker] + [m.get("yes_sub_title") or "" for m in e.get("markets", [])[:6]])
        if e.get("category") not in ("Elections", "Politics") or not (KEYWORDS.search(text) or "TX" in ticker):
            continue
        if OTHER_PLACES.search(ticker):
            dropped["other_places"] += 1
            continue
        if BRACKETS.search(ticker):
            dropped["brackets"] += 1
            continue
        markets = e.get("markets", [])
        two_way = len(markets) == 2
        for i, m in enumerate(markets):
            name = m.get("yes_sub_title") or m.get("title") or m["ticker"]
            label = LABELS.get(m["ticker"], f"{e.get('title', '').rstrip('?')}: {name}" if not two_way else name)
            p = county_p(name, ticker, cj) if ticker in COUNTY_EVENTS else model_p(m["ticker"], ticker, sw, cg)
            rows.append({"label": label, "ticker": m["ticker"], "group": ticker, "flip": int(i == 1) if two_way else "",
                         "p_yes": "" if p is None else round(p, 4), "note": NOTES.get(m["ticker"], "")})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "ticker", "group", "flip", "p_yes", "note"])
        w.writeheader()
        w.writerows(rows)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{len(rows)} markets in {len({r['group'] for r in rows})} groups, {sum(r['p_yes'] != '' for r in rows)} with a model number; dropped {dropped}")
    print(sorted({r["label"] for r in rows if r["p_yes"] == ""}))


if __name__ == "__main__":
    main()
