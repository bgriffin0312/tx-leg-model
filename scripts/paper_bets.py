"""
paper_bets.py

Track hypothetical $50 Kalshi bets against live order books. Nothing here places an
order. Entries are filled by walking the book at record time, fees included, and
saved once to output/paper_bets_2026.csv (gitignored).

Usage: python scripts/paper_bets.py record     save entry fills; refuses to overwrite
       python scripts/paper_bets.py            mark to the bid, or to settlement once resolved
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "output" / "paper_bets_2026.csv"
BASE = "https://api.elections.kalshi.com/trade-api/v2"
FEE, STAKE = 0.07, 50.0

# label, ticker, model probability at entry (statewide model 15:39 and legislative
# model 2026-10-30), French at the 62% from the direct penalty rather than the model's 77%
BETS = [
    ("Patrick Yes", "KXLTGOVTX-26-GOP", 0.915),
    ("French Yes", "KXTXRAILROAD-26-BFRE", 0.62),
    ("Abbott Yes", "GOVPARTYTX-26-R", 0.858),
    ("TX-35 De La Cruz", "KXHOUSETX35-26-R", 0.443),
    ("TX-9 Mealer", "KXHOUSETX9-26-R", 0.981),
    ("TX-23 Herrera", "KXHOUSERACE-TX23-26-R", 0.724),
    ("Middleton Yes", "KXATTYGENTX-26-R", 0.911),
    ("TX House Republican", "KXSTATELEG-TXHOUS26-R", 0.851),
    ("TX Senate Republican", "KXSTATELEG-TXSENA26-R", 0.997),
    ("Talarico Yes", "SENATETX-26-D", 0.752),
]


def fee(px):
    return FEE * px * (1 - px)


def yes_book(ticker):
    ob = requests.get(f"{BASE}/markets/{ticker}/orderbook", params={"depth": 50}, timeout=30)
    ob.raise_for_status()
    ob = ob.json()["orderbook_fp"]
    bids = sorted(((float(p), float(n)) for p, n in (ob.get("yes_dollars") or [])), reverse=True)
    asks = sorted((round(1 - float(p), 4), float(n)) for p, n in (ob.get("no_dollars") or []))
    return asks, bids


def record():
    if LOG.exists():
        sys.exit(f"{LOG} already exists; entries are recorded once.")
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for label, ticker, p in BETS:
        asks, bids = yes_book(ticker)
        spent = n = 0.0
        for px, size in asks:
            unit = px + fee(px)
            take = min(size, (STAKE - spent) / unit)
            spent += take * unit
            n += take
            if STAKE - spent < 0.01:
                break
        mid = (asks[0][0] + bids[0][0]) / 2
        rows.append({"recorded": now, "label": label, "ticker": ticker, "model_p": p, "entry_mid": round(mid, 4),
                     "blended_p": round((p + mid) / 2, 4), "contracts": round(n, 2), "stake": round(spent, 2),
                     "avg_cost": round(spent / n, 4)})
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Recorded {len(rows)} paper bets to {LOG}")


def mark():
    with open(LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total_stake = total_value = 0.0
    print(f"{'bet':<24}{'cost':>7}{'n':>7}{'status':>11}{'value':>9}{'P/L':>9}")
    for r in rows:
        n, stake = float(r["contracts"]), float(r["stake"])
        m = requests.get(f"{BASE}/markets/{r['ticker']}", timeout=30)
        result = m.json().get("market", {}).get("result") if m.status_code == 200 else ""
        if result in ("yes", "no"):
            status, value = f"won" if result == "yes" else "lost", n if result == "yes" else 0.0
        else:
            _, bids = yes_book(r["ticker"])
            left, value = n, 0.0
            for px, size in bids:
                take = min(size, left)
                value += take * (px - fee(px))
                left -= take
                if left <= 0:
                    break
            status = "open"
        total_stake += stake
        total_value += value
        print(f"{r['label']:<24}{float(r['avg_cost']):>7.3f}{n:>7.1f}{status:>11}{value:>9.2f}{value - stake:>9.2f}")
    print(f"{'TOTAL':<24}{'':>7}{'':>7}{'':>11}{total_value:>9.2f}{total_value - total_stake:>9.2f}  on ${total_stake:.2f}")


if __name__ == "__main__":
    record() if sys.argv[1:] == ["record"] else mark()
