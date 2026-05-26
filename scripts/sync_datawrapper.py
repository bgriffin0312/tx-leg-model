"""
Push the latest Flourish-ready CSVs to Datawrapper and republish each chart.

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
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

OUTPUT = ROOT / "output"
API    = "https://api.datawrapper.de/v3"

CHARTS = [
    ("DW_CHART_HOUSE",  OUTPUT / "competitive_house.csv",  "Competitive House"),
    ("DW_CHART_SENATE", OUTPUT / "competitive_senate.csv", "Competitive Senate"),
    ("DW_CHART_WAR",    OUTPUT / "war_top10_2026.csv",     "WAR Top 10"),
]


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
