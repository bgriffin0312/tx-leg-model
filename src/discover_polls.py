"""
Find newly published polls worth looking at. Discovery only — no parsing.

WHY IT WORKS THIS WAY
---------------------
Two things are true about poll ingestion here:

  1. Discovery is mechanical and worth automating. "Has Marist put out anything
     since Tuesday?" is a question a cron job can answer.
  2. Extraction is NOT. Every pollster writes up results differently, question
     wording moves between waves, crosstab banners get renamed, and some
     releases bury the racial breakdown in a workbook tab. That needs a human or
     Claude reading the actual release. Automating it produces confident wrong
     numbers, which is worse than no numbers.

So this script answers (1) and deliberately refuses (2). It emits a short list
of new releases with links; a session then decides which are worth extracting
and adds rows to data/raw/racial_crosstab_inputs.csv.

WHY NOT AGGREGATORS
-------------------
RealClearPolling and the NYT polls hub are the natural discovery sources and
both are behind DataDome-style bot protection: NYT returns 403 outright, and RCP
returns 200 with a Referer header but serves a challenge page intermittently, so
a scheduled job would silently start finding nothing. Silver Bulletin's poll
table is paid-subscriber only. Wikipedia works but is a secondary transcription
and lags.

Going direct to the pollsters is more reliable AND better targeted: the binding
constraint on this model is not how many polls exist, it is the much smaller set
that publishes RACIAL CROSSTABS. Those sources are few and stable, so they can
just be listed.

USAGE
    python src/discover_polls.py                # new since last run
    python src/discover_polls.py --days 30      # anything in a window
    python src/discover_polls.py --scope texas  # texas sources only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "raw" / "_poll_discovery_state.json"

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}

# crosstabs: does this source USUALLY publish a racial breakdown we can use?
# Recorded honestly -- "sometimes" means check, do not assume.
SOURCES = [
    # name,              scope,      kind,   url,                                                  crosstabs
    ("Economist/YouGov", "national", "rss",  "https://today.yougov.com/rss",                        "yes"),
    # Marist's /feed/ is their BLOG ("Time Machine: ..."), not poll releases --
    # it parses fine and returns nothing useful, which is the worst kind of
    # source. The releases live on the HTML listing.
    ("Marist/NPR",       "national", "html", "https://maristpoll.marist.edu/polls/",                "yes"),
    ("Emerson College",  "both",     "rss",  "https://emersoncollegepolling.com/feed/",             "sometimes"),
    ("Quinnipiac",       "national", "html", "https://poll.qu.edu/poll-release",                    "sometimes"),
    ("UT Texas Politics","texas",    "html", "https://texaspolitics.utexas.edu/polling-data-archive", "yes"),
    ("UT Tyler",         "texas",    "html", "https://www.uttyler.edu/academics/colleges-schools/"
                                             "arts-sciences/departments/political-science/pollingcenter/", "sometimes"),
]

# Titles worth surfacing. Deliberately loose -- a false positive costs a glance,
# a false negative costs a poll.
KEEP = re.compile(r"poll|survey|ballot|voters|election|tracker", re.I)


def fetch(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=45, verify=False)
    r.raise_for_status()
    return r.text


def parse_rss(body: str) -> list[dict]:
    items = []
    for m in re.finditer(r"<item[ >].*?</item>|<entry[ >].*?</entry>", body, re.S | re.I):
        blk = m.group()

        def tag(name):
            t = re.search(rf"<{name}[^>]*>(.*?)</{name}>", blk, re.S | re.I)
            if not t:
                return ""
            v = re.sub(r"<!\[CDATA\[|\]\]>", "", t.group(1))
            return re.sub(r"<[^>]+>", "", v).strip()

        link = tag("link")
        if not link:
            lm = re.search(r'<link[^>]*href="([^"]+)"', blk, re.I)
            link = lm.group(1) if lm else ""
        items.append({"title": tag("title"), "link": link,
                      "date": tag("pubDate") or tag("updated") or tag("published")})
    return items


def parse_html_links(body: str, base: str) -> list[dict]:
    """Best-effort: anchor text that looks like a poll release."""
    out, seen = [], set()
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', body, re.S | re.I):
        href, text = m.group(1), re.sub(r"<[^>]+>", " ", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        if not text or len(text) < 12 or not KEEP.search(text):
            continue
        if href.startswith("/"):
            href = re.match(r"(https?://[^/]+)", base).group(1) + href
        if href in seen:
            continue
        seen.add(href)
        out.append({"title": text[:140], "link": href, "date": ""})
    return out


DATE_PATS = [
    ("%a, %d %b %Y %H:%M:%S %z", r"^\w{3}, \d{1,2} \w{3} \d{4} \d\d:\d\d:\d\d [+-]\d{4}"),
    ("%Y-%m-%dT%H:%M:%S%z", r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:?\d\d"),
]


def parse_date(s: str):
    s = (s or "").strip()
    # RFC-822 feeds are inconsistent about the zone: Marist emits "+0000" but
    # YouGov emits "GMT", which %z will not take. Left unhandled, every YouGov
    # item silently failed to parse and the whole source reported zero.
    s = re.sub(r"\b(GMT|UTC)\b\s*$", "+0000", s)
    for fmt, pat in DATE_PATS:
        if re.match(pat, s):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
    m = re.search(r"(\w{3,9})\.?\s+(\d{1,2}),?\s+(20\d\d)", s)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}",
                                         fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=None,
                    help="report anything this recent, ignoring last-run state")
    ap.add_argument("--scope", choices=["all", "national", "texas"], default="all")
    args = ap.parse_args()

    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    seen = set(state.get("seen_links") or [])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)) if args.days else None

    print("=== poll discovery ===")
    if cutoff:
        print("window: last %d days" % args.days)
    else:
        print("mode: new since last run (%s known links)" % len(seen))
    print()

    found, errors, broken = [], [], []
    for name, scope, kind, url, crosstabs in SOURCES:
        if args.scope != "all" and scope not in (args.scope, "both"):
            continue
        try:
            body = fetch(url)
            items = parse_rss(body) if kind == "rss" else parse_html_links(body, url)
        except Exception as exc:
            errors.append((name, type(exc).__name__))
            continue

        fresh = []
        for it in items:
            if not KEEP.search(it["title"]):
                continue
            d = parse_date(it["date"])
            if cutoff is not None:
                if d is None or d < cutoff:
                    continue
            elif it["link"] in seen:
                continue
            it["source"], it["crosstabs"], it["parsed_date"] = name, crosstabs, d
            fresh.append(it)

        # HTML listings have no reliable dates; cap them so they don't flood
        if kind == "html" and cutoff is None:
            fresh = fresh[:8]
        found.extend(fresh)

        # A zero that means "nothing new" and a zero that means "the parser
        # stopped working" must not look the same. That is exactly how the
        # presidential collector hid a 3.56% vote loss behind "0.0% unassigned".
        if not items:
            flag = "  <-- PARSER FOUND NOTHING; page layout probably changed"
            broken.append(name)
        elif kind == "html" and not any(parse_date(i["date"]) for i in items):
            flag = "  (no dates on this listing — items shown are unfiltered)"
        else:
            flag = ""
        print("  %-19s %-8s %-20s %3d new / %3d seen%s"
              % (name, scope, "crosstabs:" + crosstabs, len(fresh), len(items), flag))

    print()
    if errors:
        print("UNREACHABLE: %s" % ", ".join("%s (%s)" % e for e in errors))
    if broken:
        print("PARSER STALE: %s -- these returned a page but no recognisable"
              % ", ".join(broken))
        print("  items, so their zero above means 'not checked', not 'nothing new'.")
    if errors or broken:
        print()

    if not found:
        print("Nothing new.")
    else:
        print("%d release(s) to triage. Extraction is a session job — open the" % len(found))
        print("link, confirm it has a racial banner, and add a row to")
        print("data/raw/racial_crosstab_inputs.csv. Do NOT trust a title alone.\n")
        for it in sorted(found, key=lambda x: (x["source"], x["title"])):
            d = it["parsed_date"].strftime("%Y-%m-%d") if it["parsed_date"] else "     ?    "
            print("  [%s] %-19s %s" % (d, it["source"], it["title"][:88]))
            print("      %s" % it["link"])

    seen.update(it["link"] for it in found)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"last_run": datetime.now(timezone.utc).isoformat(),
                                 "seen_links": sorted(seen)[-2000:]}, indent=1),
                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
