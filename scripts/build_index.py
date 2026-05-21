"""
Build a simple index.html landing page for the GitHub Pages site,
listing the current rendered model artifacts with their last-updated
timestamps. Run after the model + chart builders, before publishing.
"""

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"

ARTIFACTS = [
    ("model_2026_map_house.html",             "TX House — interactive map",                "House districts colored by D win-prob at the selected national-environment scenario."),
    ("model_2026_map_senate.html",            "TX Senate — interactive map",               "16 on-ballot Senate districts shaded; holdover seats in khaki."),
    ("model_2026_map_house_competitive.html", "TX House — competitive districts only",     "Districts where D win-prob is 25–75%; safe seats greyed out."),
    ("model_2026_control_chart.html",         "Chamber control probability by scenario",   "P(D wins each chamber) across R+3 → D+9 generic-ballot scenarios."),
    ("model_2026_flip_table.html",            "Per-district flip table",                   "Sortable table of every on-ballot race with win-prob and candidate names."),
    ("model_2026_war_table.html",             "WAR rankings table",                        "Candidate Wins Above Replacement scores used in the quality term."),
]


def human_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> None:
    rows = []
    for fname, title, desc in ARTIFACTS:
        path = OUTPUT / fname
        if not path.exists():
            continue
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        size = human_size(path)
        rows.append(
            f'  <li><a href="{fname}"><strong>{title}</strong></a>'
            f'<br><span class="desc">{desc}</span>'
            f'<br><span class="meta">updated {ts} · {size}</span></li>'
        )

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Texas Legislature 2026 — Model Outputs</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 720px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.55; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .lede {{ color: #666; margin-top: 0; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ margin: 1.25rem 0; padding: 0.75rem 1rem; border-left: 3px solid #2563eb; background: #f7f9fc; }}
  a {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .desc {{ color: #444; font-size: 0.95rem; }}
  .meta {{ color: #888; font-size: 0.8rem; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
            color: #888; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>Texas Legislature 2026 — Model Outputs</h1>
<p class="lede">Interactive artifacts from the 2026 TX House + Senate Monte Carlo forecasting model.
Free to link to from Substack or anywhere else.</p>

<ul>
{chr(10).join(rows)}
</ul>

<footer>
  Index built {built_at}. Source: <a href="https://github.com/bgriffin0312/tx-leg-model">github.com/bgriffin0312/tx-leg-model</a>.
</footer>
</body>
</html>
"""
    out = OUTPUT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"  Wrote {out.relative_to(ROOT)} ({len(rows)} artifacts listed)")


if __name__ == "__main__":
    main()
