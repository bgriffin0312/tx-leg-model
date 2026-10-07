"""
check_live_embeds.py

What is the Texas Proof post ACTUALLY showing? Reads the published post from
Substack's public API, lists every embed, follows each Datawrapper embed to the
version it currently serves, and compares that version's data with the local
CSVs the publish pipeline builds.

How the pieces fit (verified 2026-10-07):
  - The post embeds Datawrapper charts by a version-pinned URL
    (datawrapper.dwcdn.net/<id>/1/). Old version pages meta-refresh to the
    newest PUBLISHED version, so republishing a chart updates the post.
  - Uploading data without publishing (what sync_datawrapper.py does while its
    token lacks the publish scope) changes nothing readers see.
  - Substack also stores a static PNG thumbnail of each chart, taken when it
    was embedded; email and some app/social previews show that image, which
    never updates. Re-embedding the chart refreshes it.
  - The "Updated <date>" line is text inside each Datawrapper chart; it does
    not change on its own.

Usage:
  python scripts/check_live_embeds.py
  python scripts/check_live_embeds.py --post modeling-the-texas-state-elections
"""

import argparse
import html
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
PUB = "https://brennan339256.substack.com"
LOCAL = {  # Datawrapper chart id -> local CSV the pipeline uploads to it
    "l53Um": ROOT / "output" / "competitive_house.csv",
    "oz3IU": ROOT / "output" / "competitive_senate.csv",
    "L6hhb": ROOT / "output" / "war_top10_2026.csv",
}


def live_version(chart_id: str, start_url: str) -> tuple[str, int]:
    """Follow meta-refresh redirects to the version actually being served."""
    url = start_url
    for _ in range(10):
        page = requests.get(url, timeout=30).text
        m = re.search(r'http-equiv="REFRESH"\s+content="0;\s*url=([^"]+)"', page, re.I)
        if not m:
            break
        url = m.group(1)
    v = re.search(rf"/{chart_id}/(\d+)/", url)
    return url, int(v.group(1)) if v else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", help="post slug (default: every post in the archive)")
    args = ap.parse_args()
    slugs = [args.post] if args.post else [
        p["slug"] for p in requests.get(f"{PUB}/api/v1/archive?sort=new&limit=50", timeout=30).json()]

    for slug in slugs:
        post = requests.get(f"{PUB}/api/v1/posts/{slug}", timeout=30).json()
        body = post.get("body_html", "")
        print(f"\n=== {post.get('title')} ({slug}) — published {post.get('post_date', '')[:10]}, "
              f"last edited {post.get('updated_at', '')[:10]}")
        srcs = re.findall(r'<iframe[^>]+src="([^"]+)"', body)
        others = sorted({h for h in re.findall(r'https?://([a-z0-9.\-]+)/', " ".join(srcs))})
        print(f"  iframe hosts: {', '.join(others) or 'none'}")
        for attrs in re.findall(r'class="datawrapper-wrap[^"]*"\s+data-attrs="([^"]+)"', body):
            a = json.loads(html.unescape(attrs))
            cid = re.search(r"dwcdn\.net/([A-Za-z0-9]+)/", a["url"]).group(1)
            url, ver = live_version(cid, a["url"])
            live = pd.read_csv(io.StringIO(requests.get(url + "dataset.csv", timeout=30).text))
            note = re.search(r"Updated [0-9/]+", requests.get(url, timeout=30).text)
            print(f"\n  {a.get('title')} [{cid}]  embed {a['url']}  ->  serving v{ver}")
            print(f"    chart's own date note: {note.group(0) if note else '(none)'};  "
                  f"Substack thumbnail (static PNG, frozen at embed time): {a.get('thumbnail_url', '')[-40:]}")
            print(f"    live first rows: " + " | ".join(
                ", ".join(str(x) for x in r) for r in live.head(3).itertuples(index=False)))
            path = LOCAL.get(cid)
            if path and path.exists():
                loc = pd.read_csv(path)
                same = live.shape == loc.shape and live.astype(str).equals(loc.astype(str))
                print(f"    matches local {path.name}: {'YES' if same else 'NO — local is newer/different'}"
                      + ("" if same else f"  (local first row: {', '.join(str(x) for x in loc.iloc[0])})"))


if __name__ == "__main__":
    main()
