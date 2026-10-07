"""
Push the latest CSVs to Datawrapper, stamp each chart's "Updated <date>" line,
and republish. Republishing updates the Substack post automatically (its embeds
redirect to the newest published version; see scripts/check_live_embeds.py).

The token needs chart:read, chart:write, theme:read and visualization:read --
without the last two, publish fails with 403 "Insufficient scope".

The three charts pull their data from the GitHub Pages CSVs (upload-method
"external-data"), so the data PUT is overridden at publish; the script waits
until Pages serves the new CSV before publishing (wait_for_external_data).

Reads from .env:
  DATAWRAPPER_TOKEN   — API token (Datawrapper → Settings → API Tokens)
  DW_CHART_HOUSE      — chart ID for competitive House table
  DW_CHART_SENATE     — chart ID for competitive Senate table
  DW_CHART_WAR        — chart ID for top-10 WAR table

A chart ID is the short string in the chart's edit URL, e.g.
  https://app.datawrapper.de/chart/aBcDe/edit  →  aBcDe

Any chart whose env var is unset is silently skipped, so you can wire up
the three charts one at a time. Set the env var, rerun, repeat.

Run after a model rebuild:
  python src/model.py
  python scripts/build_competitive_csv.py
  python scripts/build_war_top10_csv.py
  python scripts/sync_datawrapper.py
"""

import os
import re
import time
import sys
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# The Windows console default (cp1252) can't print the arrow in the status
# line; that crash used to abort the remaining charts after the first publish.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OUTPUT = ROOT / "output"
API    = "https://api.datawrapper.de/v3"

CHARTS = [
    ("DW_CHART_HOUSE",  OUTPUT / "competitive_house.csv",  "Competitive House"),
    ("DW_CHART_SENATE", OUTPUT / "competitive_senate.csv", "Competitive Senate"),
    ("DW_CHART_WAR",    OUTPUT / "war_top10_2026.csv",     "WAR Top 10"),
]


UPDATED = re.compile(r"Updated \d{1,2}/\d{1,2}/\d{2,4}")


def stamp_updated_date(chart_id: str, csv_path: Path, label: str, headers: dict) -> None:
    """Rewrite the chart's "Updated M/D/YYYY" line to the date the CSV was built.

    The line lives in different fields on different charts (describe.intro on
    the House table, annotate.notes on the other two), so find it wherever it
    is and replace only the date. If no chart field has one, add it to the
    notes. Uses the CSV's modification date, not today's, so re-running the
    sync later without a rebuild doesn't claim fresher data than it has.
    """
    built = date.fromtimestamp(csv_path.stat().st_mtime)
    new_text = f"Updated {built.month}/{built.day}/{built.year}"
    got = requests.get(f"{API}/charts/{chart_id}", headers=headers, timeout=30)
    if got.status_code >= 300:
        print(f"  [{label}] could not read chart metadata ({got.status_code}); date not updated")
        return
    meta = got.json().get("metadata", {})
    patch = {}
    for section, field in (("describe", "intro"), ("annotate", "notes")):
        text = (meta.get(section) or {}).get(field) or ""
        if UPDATED.search(text):
            patch[section] = {field: UPDATED.sub(new_text, text)}
    if not patch:
        notes = (meta.get("annotate") or {}).get("notes") or ""
        patch["annotate"] = {"notes": (notes + " " if notes else "") + new_text}
    r = requests.patch(f"{API}/charts/{chart_id}", headers=headers,
                       json={"metadata": patch}, timeout=30)
    if r.status_code >= 300:
        print(f"  [{label}] date update failed: {r.status_code}  {r.text[:200]}")


def wait_for_external_data(chart_id: str, csv_text: str, label: str, headers: dict,
                           timeout_s: int = 720) -> bool:
    """These charts are set to upload-method "external-data": on publish,
    Datawrapper re-fetches the GitHub Pages copy of the CSV and ignores the
    data PUT above. Pages takes a minute to deploy and its CDN caches files
    for 10 minutes, so publishing right after publish_charts.ps1 pushed served
    the previous run (found 2026-10-07: House v6 went out with stale data).
    Wait until the external URL serves the local CSV; give up after timeout_s.
    """
    meta = requests.get(f"{API}/charts/{chart_id}", headers=headers, timeout=30).json()
    data_meta = (meta.get("metadata") or {}).get("data") or {}
    url = data_meta.get("external-data") if data_meta.get("upload-method") == "external-data" else None
    if not url:
        return True
    want = csv_text.replace("\r\n", "\n").strip()
    deadline = time.time() + timeout_s
    waited = False
    while True:
        got = requests.get(url, params={"nocache": time.time()}, timeout=30)
        if got.ok and got.content.decode("utf-8", "replace").replace("\r\n", "\n").strip() == want:
            if waited:
                print(f"  [{label}] external data source caught up")
            return True
        if time.time() > deadline:
            print(f"  [{label}] NOT published: {url} still serves older data after "
                  f"{timeout_s // 60} min (GitHub Pages not deployed/cached?). Re-run later.")
            return False
        if not waited:
            print(f"  [{label}] waiting for {url} to serve the new CSV (Pages deploy/cache)...")
            waited = True
        time.sleep(20)


def sync(chart_id: str, csv_path: Path, label: str, token: str) -> None:
    csv_text = csv_path.read_text(encoding="utf-8")
    headers  = {"Authorization": f"Bearer {token}"}

    put = requests.put(
        f"{API}/charts/{chart_id}/data",
        headers={**headers, "Content-Type": "text/csv"},
        data=csv_text.encode("utf-8"),
        timeout=30,
    )
    if put.status_code >= 300:
        print(f"  [{label}] data upload failed: {put.status_code}  {put.text[:200]}")
        return

    stamp_updated_date(chart_id, csv_path, label, headers)
    if not wait_for_external_data(chart_id, csv_text, label, headers):
        return

    pub = requests.post(
        f"{API}/charts/{chart_id}/publish",
        headers=headers,
        timeout=60,
    )
    if pub.status_code >= 300:
        print(f"  [{label}] publish failed: {pub.status_code}  {pub.text[:200]}")
        return

    public_url = pub.json().get("data", {}).get("publicUrl", "(no URL returned)")
    rows = csv_text.count("\n") - 1  # minus header
    print(f"  [{label}] {rows} rows  →  {public_url}")


def main() -> None:
    token = os.environ.get("DATAWRAPPER_TOKEN", "").strip()
    if not token:
        sys.exit("DATAWRAPPER_TOKEN not set in .env")

    any_done = False
    for env_var, csv_path, label in CHARTS:
        chart_id = os.environ.get(env_var, "").strip()
        if not chart_id:
            print(f"  [{label}] skipped — {env_var} not set in .env")
            continue
        if not csv_path.exists():
            print(f"  [{label}] skipped — {csv_path.name} not found (run the build script first)")
            continue
        sync(chart_id, csv_path, label, token)
        any_done = True

    if not any_done:
        print("\n  Nothing synced. Add chart IDs to .env and rerun.")


if __name__ == "__main__":
    main()
