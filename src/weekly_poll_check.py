"""
Weekly checklist: which pollsters to look at, and what is new since last week.

WHY A LIST AND NOT DISCOVERY
----------------------------
An earlier attempt tried to discover polls automatically. It does not work well
enough to trust:

  - The obvious aggregators are bot-protected. NYT's polls hub returns 403.
    RealClearPolling answers with a Referer header but serves a DataDome
    challenge intermittently -- two identical requests returned different
    bodies -- so a scheduled job would silently start finding nothing. Silver
    Bulletin's table is paid-subscriber only.
  - Detecting "does this release break the generic ballot out by race?" cannot
    be generalised. The one detector in this repo that works keys on the string
    "genericcongressionalvote", which is a YouGov TABLE LABEL. Every pollster
    words and labels the question differently, and some publish the banner only
    in a linked workbook (Emerson) rather than the write-up.

So: keep a curated list of pollsters known or suspected to publish a national
generic ballot broken out by race, check it weekly, and let a session do the
reading. The registry lives in config/pollster_registry.csv as DATA, not code,
so it can be edited without touching this file.

Each registry row carries its own EVIDENCE and a crosstab_status:
    CONFIRMED  -- we have actually extracted a racial banner from this pollster
    UNVERIFIED -- plausible but never confirmed; worth one check, then promote or drop
    LIKELY NO  -- evidence suggests topline-only; kept as a cross-check

USAGE
    python src/weekly_poll_check.py              # the checklist
    python src/weekly_poll_check.py --scope texas
    python src/weekly_poll_check.py --fetch      # also probe each URL for new items
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# config/, not data/raw/ -- data/raw is gitignored, so a registry there
# would never be committed and would exist only on one machine.
REGISTRY = ROOT / "config" / "pollster_registry.csv"
STATE = ROOT / "data" / "raw" / "_weekly_poll_check_state.json"

ORDER = {"CONFIRMED": 0, "UNVERIFIED": 1, "LIKELY NO": 2}


def load_registry() -> list[dict]:
    with open(REGISTRY, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def probe(url: str) -> str:
    """Cheap liveness + item-count probe. Never tries to judge crosstabs."""
    import requests
    import urllib3
    urllib3.disable_warnings()
    ua = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
    try:
        r = requests.get(url, headers=ua, timeout=40, verify=False)
    except Exception as exc:
        return "UNREACHABLE (%s)" % type(exc).__name__
    if not r.ok:
        return "HTTP %s" % r.status_code
    dates = re.findall(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+2026",
        r.text)
    return "ok, %s dated mentions of 2026" % len(dates) if dates else "ok, no 2026 dates seen"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["all", "national", "texas"], default="all")
    ap.add_argument("--fetch", action="store_true",
                    help="probe each URL (slower; liveness only, no parsing)")
    args = ap.parse_args()

    rows = load_registry()
    if args.scope != "all":
        rows = [r for r in rows if r["scope"] in (args.scope, "both")]
    rows.sort(key=lambda r: (ORDER.get(r["crosstab_status"], 9), r["pollster"]))

    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    last = state.get("last_run", "never")

    print("=" * 78)
    print("  WEEKLY POLL CHECK — generic national ballot with racial crosstabs")
    print("  last run: %s" % last)
    print("=" * 78)
    print()
    print("Extraction is a SESSION job. Open the link, confirm the release")
    print("actually breaks the generic ballot out by race, and add a row to")
    print("data/raw/racial_crosstab_inputs.csv. A title is not evidence.")
    print()

    for status in ("CONFIRMED", "UNVERIFIED", "LIKELY NO"):
        group = [r for r in rows if r["crosstab_status"] == status]
        if not group:
            continue
        head = {"CONFIRMED": "CHECK THESE FIRST — known to publish a racial banner",
                "UNVERIFIED": "VERIFY ONCE — then promote to CONFIRMED or drop",
                "LIKELY NO":  "TOPLINE CROSS-CHECK ONLY"}[status]
        print("-" * 78)
        print("  %s  (%d)" % (head, len(group)))
        print("-" * 78)
        for r in group:
            print("  %-26s %-9s %s" % (r["pollster"], r["scope"], r["cadence"]))
            print("      %s" % r["check_url"])
            if r.get("notes"):
                print("      note: %s" % r["notes"][:150])
            if args.fetch:
                print("      probe: %s" % probe(r["check_url"]))
        print()

    n_conf = sum(1 for r in rows if r["crosstab_status"] == "CONFIRMED")
    print("%d pollsters listed, %d confirmed. Registry: config/pollster_registry.csv"
          % (len(rows), n_conf))
    print("Add or correct entries there — it is data, not code.")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"last_run": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
