"""
Weekly checklist: which pollsters to look at, and what to check on each.

CROSSTABS ARE A PROPERTY OF THE RELEASE, NOT THE POLLSTER
---------------------------------------------------------
This is the correction that matters. An earlier version of this registry carried
a per-pollster verdict -- CONFIRMED / LIKELY NO -- as though "does this outfit
publish a racial banner" were a stable fact. It is not.

Emerson is the proof. Their JULY 2026 release was checked three separate ways
(page links, sheet export, hidden-tab check) and had no generic ballot by race.
Their AUGUST 2026 release did, in a linked workbook tab. Same pollster, six
weeks apart, opposite answers. Quinnipiac was likewise written off as "topline
only, no racial crosstabs in the 2025-2026 format" on the strength of a single
December 2025 PDF -- a claim about one release, generalised into a permanent
property and then believed.

So the registry records HISTORY, not a verdict:
    last_confirmed_crosstab -- the most recent release we actually pulled a
                               banner from, or "none found" / "unknown"
    hit_rate                -- how many releases we have checked, and how many
                               had one. "1 of 1" is not a pattern.
Every wave gets checked. A pollster that had a banner last month may not this
month, and one that never had one may start.

WHY A LIST AND NOT AUTOMATED DISCOVERY
--------------------------------------
  - The aggregators are bot-protected. NYT's polls hub returns 403.
    RealClearPolling answers with a Referer header but serves a DataDome
    challenge intermittently -- two identical requests returned different
    bodies -- so a scheduled job would silently start finding nothing. Silver
    Bulletin's poll table is paid-subscriber only.
  - "Does this release break the ballot out by race?" does not generalise. The
    only detector in this repo that works keys on the literal string
    "genericcongressionalvote", a YouGov TABLE LABEL. Every pollster words it
    differently and Emerson hides it in a workbook.

USAGE
    python src/weekly_poll_check.py              # the checklist
    python src/weekly_poll_check.py --scope texas
    python src/weekly_poll_check.py --fetch      # also probe each URL is alive
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
# config/, not data/raw/ -- data/raw is gitignored, so a registry there would
# live on one machine only and be lost to anyone cloning.
REGISTRY = ROOT / "config" / "pollster_registry.csv"
STATE = ROOT / "data" / "raw" / "_weekly_poll_check_state.json"


def load_registry() -> list[dict]:
    with open(REGISTRY, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def sort_key(r: dict):
    """Most recently productive first — not by a verdict."""
    last = (r.get("last_confirmed_crosstab") or "").strip()
    has_date = bool(re.match(r"\d{4}-\d\d-\d\d", last))
    # recent hits first, then unknowns, then explicit misses
    bucket = 0 if has_date else (1 if last == "unknown" else 2)
    return (bucket, "" if not has_date else _invert(last), r["pollster"])


def _invert(d: str) -> str:
    """Sort dates descending inside an ascending sort."""
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in d)


def probe(url: str) -> str:
    """Liveness only. Deliberately does NOT try to judge crosstabs — see module
    docstring; that judgement is what cannot be automated."""
    import requests
    import urllib3
    urllib3.disable_warnings()
    ua = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
    try:
        r = requests.get(url, headers=ua, timeout=40, verify=False)
    except Exception as exc:
        return "UNREACHABLE (%s) — fix the URL in the registry" % type(exc).__name__
    if not r.ok:
        return "HTTP %s — fix the URL in the registry" % r.status_code
    dates = re.findall(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+2026",
        r.text)
    return "alive, %d dated 2026 mentions" % len(dates) if dates else "alive, no 2026 dates seen"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", choices=["all", "national", "texas"], default="all")
    ap.add_argument("--fetch", action="store_true",
                    help="probe each URL for liveness (slower; no parsing)")
    args = ap.parse_args()

    rows = load_registry()
    if args.scope != "all":
        rows = [r for r in rows if r["scope"] in (args.scope, "both")]
    rows.sort(key=sort_key)

    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    print("=" * 78)
    print("  WEEKLY POLL CHECK — generic ballot broken out by race")
    print("  last run: %s" % state.get("last_run", "never"))
    print("=" * 78)
    print()
    print("CHECK EVERY RELEASE. Whether a poll carries a racial banner is a")
    print("property of that release, not of the pollster: Emerson had none in")
    print("July 2026 and did in August. History below is a prior, not a promise.")
    print()
    print("For each: open the link, find the newest release, confirm it actually")
    print("breaks the generic ballot out by race, then add a row to")
    print("data/raw/racial_crosstab_inputs.csv. A headline is not evidence.")
    print()
    print("-" * 78)

    for r in rows:
        last = (r.get("last_confirmed_crosstab") or "unknown").strip()
        print("  %-26s %-8s %-11s last banner: %s"
              % (r["pollster"], r["scope"], r["cadence"], last))
        print("      %s" % r["check_url"])
        print("      checked so far: %s" % (r.get("hit_rate") or "?"))
        if r.get("notes"):
            print("      note: %s" % r["notes"][:160])
        if args.fetch:
            print("      probe: %s" % probe(r["check_url"]))
        print()

    ever = sum(1 for r in rows
               if re.match(r"\d{4}-\d\d-\d\d", (r.get("last_confirmed_crosstab") or "")))
    print("-" * 78)
    print("%d pollster%s listed; %d ha%s ever yielded a racial banner."
          % (len(rows), "" if len(rows) == 1 else "s", ever,
             "s" if ever == 1 else "ve"))
    print("Registry: config/pollster_registry.csv — data, not code. When you check")
    print("a release, update last_confirmed_crosstab and hit_rate, hit or miss.")
    print("Recording the misses is what keeps the hit rate honest.")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(
        {"last_run": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")},
        indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
