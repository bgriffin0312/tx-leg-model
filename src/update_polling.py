"""
update_polling.py

Update RACE_GENERIC_BALLOT_D_SHARE in model_config.py from free public polling
crosstabs, replacing the old single-poll DDHQ snapshot approach.

SOURCE HIERARCHY:
  PRIMARY: Economist/YouGov weekly crosstab PDFs (free, public, ~1,600 RV/week,
    consistent methodology). We average 4 most recent weeks for stability.
    Published at: https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_*.pdf
    Found via: https://yougov.com/en-us/topics/topic/The_Economist_YouGov_polls

  CROSS-CHECK: Civiqs daily dashboard (free, filterable by race, but single
    online panel with slight D lean — use for movement detection, not levels).
    Dashboard: civiqs.com/results

  DEPRECATED: DDHQ single-poll IDs (446/448/452) — replaced by YouGov rolling
    average for more stability. Individual poll crosstabs have large subgroup
    margins of error — a single poll's Hispanic n might be 150-300 respondents.
    Averaging across 4 weekly polls (~600-1,000 Hispanic respondents total)
    reduces this noise substantially.

  ASPIRATIONAL: FiftyPlusOne aggregated crosstab average — best available
    multi-poll aggregate with house-effect correction, but requires $150/mo
    Premium subscription. If budget allows, switch to this.

KNOWN LIMITATIONS:
  - YouGov is a single pollster (online panel), so you get their house effect
    rather than a multi-poll average. The 4-week rolling average reduces
    sampling noise but not systematic bias.
  - "Other/Asian" is solved from the topline constraint because pollster
    reporting of this group is inconsistent and sample sizes are small.
  - The improvement over DDHQ is consistency: YouGov gives a reliable weekly
    time series with known methodology, vs pulling from whichever single poll
    DDHQ happened to post.

USAGE:
  python src/update_polling.py                  # dry run: show comparison
  python src/update_polling.py --apply          # update model_config.py
  python src/update_polling.py --force-download # bypass PDF cache
"""

import argparse
import io
import re
import sys
from datetime import date
from pathlib import Path

import pdfplumber
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CACHE_DIR = DATA_RAW / "yougov_crosstabs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

# Known recent Economist/YouGov PDF URLs (most recent first).
# Update this list when new polls are published; the article pages at
# yougov.com/en-us/topics/topic/The_Economist_YouGov_polls link to them.
# URL pattern: https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_[ID].pdf
YOUGOV_PDFS = [
    ("2026-03-27_to_30", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_3wplfYX.pdf"),
    ("2026-03-20_to_23", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_o84FoNw.pdf"),
    ("2026-03-13_to_16", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_CwWXhS2.pdf"),
    ("2026-03-06_to_09", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_EcCnfRV.pdf"),
]

USER_AGENT = "TXLegislativeModel/1.0 (academic research)"


# ---------------------------------------------------------------------------
# Download and cache PDFs
# ---------------------------------------------------------------------------

def download_pdf(label: str, url: str, force: bool = False) -> bytes | None:
    cache_path = CACHE_DIR / f"{label}.pdf"
    if cache_path.exists() and not force:
        print(f"  Cache hit: {label}")
        return cache_path.read_bytes()

    print(f"  Downloading {label}...")
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        print(f"    {len(resp.content):,} bytes")
        return resp.content
    except requests.RequestException as e:
        print(f"    FAILED: {e}")
        return None


# ---------------------------------------------------------------------------
# Extract generic ballot by race from PDF
# ---------------------------------------------------------------------------

def extract_generic_ballot_race(pdf_bytes: bytes) -> dict | None:
    """
    Parse the GenericCongressionalVote question's race crosstab from a
    YouGov econTabReport PDF.

    The PDF page looks like:
      41. GenericCongressionalVote
      If the elections for U.S. Congress were being held today...
                            Sex           Race            Age      Education
      Total Male Female White Black Hispanic 18-29 30-44 45-64 65+ ...
      TheDemocraticPartycandidate  39%  33%  44%  34%  65%  41%  ...
      TheRepublicanPartycandidate  36%  43%  29%  43%   6%  34%  ...

    Returns dict with white/black/hispanic dem/rep percentages, or None.
    """
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))

    # Extract date range from first page
    date_range = ""
    first_text = pdf.pages[0].extract_text() or ""
    m = re.search(r"(\w+ \d+ [- ] \d+, \d{4})", first_text)
    if m:
        date_range = m.group(1)

    for page in pdf.pages:
        text = page.extract_text() or ""
        if "GenericCongressionalVote" not in text and "genericcongressionalvote" not in text.lower():
            continue

        lines = text.split("\n")

        # Find key rows — the column header row contains "Total Male Female White..."
        # and the data rows contain percentages. Be specific to avoid matching
        # the category label row ("Sex Race Age Education").
        header_line = None
        dem_line = None
        rep_line = None
        n_line = None

        for line in lines:
            # Header: must contain "Total" AND "White" AND "Black" (the column header,
            # not the category label "Sex Race Age Education")
            if "Total" in line and "White" in line and "Black" in line and "Hispanic" in line:
                header_line = line
            elif "DemocraticParty" in line.replace(" ", ""):
                # Take only the FIRST Dem line (race table, not party/ideology table)
                if dem_line is None:
                    dem_line = line
            elif "RepublicanParty" in line.replace(" ", ""):
                if rep_line is None:
                    rep_line = line
            elif "UnweightedN" in line.replace(" ", ""):
                if n_line is None:
                    n_line = line

        if not (header_line and dem_line and rep_line):
            continue

        # Extract percentages
        def extract_pcts(line: str) -> list[float]:
            return [float(p) for p in re.findall(r"(\d+)%", line)]

        dem_pcts = extract_pcts(dem_line)
        rep_pcts = extract_pcts(rep_line)

        # Parse header columns
        headers = re.split(r"\s+", header_line.strip())
        col_map = {h.lower(): i for i, h in enumerate(headers)}

        white_idx = col_map.get("white")
        black_idx = col_map.get("black")
        hisp_idx = col_map.get("hispanic")

        if white_idx is None or black_idx is None or hisp_idx is None:
            print(f"    Could not find race columns in header: {headers}")
            continue

        max_needed = max(white_idx, black_idx, hisp_idx)
        if len(dem_pcts) <= max_needed or len(rep_pcts) <= max_needed:
            print(f"    Not enough columns ({len(dem_pcts)} found, need {max_needed + 1})")
            continue

        result = {
            "white":    {"dem": dem_pcts[white_idx], "rep": rep_pcts[white_idx]},
            "black":    {"dem": dem_pcts[black_idx], "rep": rep_pcts[black_idx]},
            "hispanic": {"dem": dem_pcts[hisp_idx],  "rep": rep_pcts[hisp_idx]},
            "total":    {"dem": dem_pcts[0],          "rep": rep_pcts[0]},
            "date_range": date_range,
        }

        if n_line:
            n_match = re.search(r"\((\d[\d,]*)\)", n_line)
            if n_match:
                result["sample_size"] = int(n_match.group(1).replace(",", ""))

        pdf.close()
        return result

    pdf.close()
    return None


# ---------------------------------------------------------------------------
# Compute rolling average and compare to current config
# ---------------------------------------------------------------------------

def compute_d_2p(dem: float, rep: float) -> float:
    """D two-party share from raw D% and R% (ignoring undecided/other)."""
    return dem / (dem + rep) if (dem + rep) > 0 else 0.5


def solve_other_from_topline(white_d2p: float, black_d2p: float, hisp_d2p: float,
                              topline_d2p: float) -> float:
    """
    Back-solve for other/Asian D 2p share from the topline constraint.
    topline ≈ Σ(weight × group_d2p), so:
    other_d2p = (topline - w*white - b*black - h*hisp) / w_other

    FiftyPlusOne's known limitation: "other/Asian" can't be reliably estimated
    from individual polls due to small sample sizes. This topline-constraint
    solve is the correct workaround.
    """
    from model_config import NATIONAL_DEMO_WEIGHTS as w
    numerator = topline_d2p - (w["white_nh"] * white_d2p +
                                w["black_nh"] * black_d2p +
                                w["hispanic"] * hisp_d2p)
    if w["other"] > 0:
        return max(0.0, min(1.0, numerator / w["other"]))
    return 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Update racial generic ballot crosstabs from YouGov weekly polls")
    parser.add_argument("--apply", action="store_true",
                        help="Update model_config.py in place (default: dry run)")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download PDFs even if cached")
    parser.add_argument("--n-weeks", type=int, default=4,
                        help="Number of recent polls to average (default: 4)")
    args = parser.parse_args()

    print("=" * 65)
    print("  Update Racial Generic Ballot Crosstabs")
    print("  Source: Economist/YouGov Weekly Poll (4-Week Rolling Average)")
    print("=" * 65)

    # Download and parse PDFs
    print(f"\nFetching {args.n_weeks} most recent polls...")
    polls = []
    for label, url in YOUGOV_PDFS[:args.n_weeks]:
        pdf_bytes = download_pdf(label, url, force=args.force_download)
        if not pdf_bytes:
            continue
        result = extract_generic_ballot_race(pdf_bytes)
        if result:
            polls.append(result)
            w = compute_d_2p(result["white"]["dem"], result["white"]["rep"])
            b = compute_d_2p(result["black"]["dem"], result["black"]["rep"])
            h = compute_d_2p(result["hispanic"]["dem"], result["hispanic"]["rep"])
            n = result.get("sample_size", "?")
            print(f"    {result['date_range']:30s}  W={w:.1%}  B={b:.1%}  H={h:.1%}  n={n}")
        else:
            print(f"    {label}: could not extract generic ballot race data")

    if not polls:
        print("\nNo polls extracted. Cannot update.")
        return

    # Compute rolling average of D 2-party shares
    avg = {}
    for group in ["white", "black", "hispanic"]:
        dem_avg = sum(p[group]["dem"] for p in polls) / len(polls)
        rep_avg = sum(p[group]["rep"] for p in polls) / len(polls)
        avg[group] = compute_d_2p(dem_avg, rep_avg)

    # Topline for other/Asian solve
    topline_dem = sum(p["total"]["dem"] for p in polls) / len(polls)
    topline_rep = sum(p["total"]["rep"] for p in polls) / len(polls)
    topline_d2p = compute_d_2p(topline_dem, topline_rep)

    avg["other"] = solve_other_from_topline(
        avg["white"], avg["black"], avg["hispanic"], topline_d2p
    )

    # Load current config values
    from model_config import (
        RACE_GENERIC_BALLOT_D_SHARE as current,
        GENERIC_BALLOT_SOURCE,
        GENERIC_BALLOT_UPDATED,
        GENERIC_BALLOT_TOPLINE_D_2P,
        NATIONAL_DEMO_WEIGHTS,
    )

    # Map config keys to poll group names
    group_map = {
        "white_nh": "white",
        "black_nh": "black",
        "hispanic": "hispanic",
        "other":    "other",
    }

    # Compute implied topline from new values
    new_topline = sum(
        NATIONAL_DEMO_WEIGHTS[k] * avg[group_map[k]]
        for k in NATIONAL_DEMO_WEIGHTS
    )

    # Print comparison
    print(f"\n{'=' * 65}")
    print(f"  COMPARISON: Current vs YouGov {len(polls)}-Week Average")
    print(f"{'=' * 65}")
    print(f"  Polls: {len(polls)} weeks ({polls[-1]['date_range']} to {polls[0]['date_range']})")
    total_n = sum(p.get("sample_size", 0) for p in polls)
    if total_n:
        print(f"  Combined sample: ~{total_n:,} respondents")

    print(f"\n  {'Group':15s}  {'Current':>8s}  {'YouGov':>8s}  {'Delta':>8s}  {'Flag':>5s}")
    print(f"  {'-' * 50}")

    any_shift = False
    for config_key in ["white_nh", "black_nh", "hispanic", "other"]:
        group = group_map[config_key]
        old = current[config_key]
        new = avg[group]
        delta = new - old
        flag = " ***" if abs(delta) > 0.01 else ""
        if abs(delta) > 0.01:
            any_shift = True
        print(f"  {config_key:15s}  {old:8.1%}  {new:8.1%}  {delta:+8.1%}{flag}")

    print(f"\n  {'Topline D 2p':15s}  {GENERIC_BALLOT_TOPLINE_D_2P:8.1%}  {new_topline:8.1%}  "
          f"{new_topline - GENERIC_BALLOT_TOPLINE_D_2P:+8.1%}")
    print(f"\n  Current source: {GENERIC_BALLOT_SOURCE}")
    print(f"  Current date:   {GENERIC_BALLOT_UPDATED}")

    if any_shift:
        print(f"\n  *** One or more groups shifted >1pp — update recommended")
    else:
        print(f"\n  No group shifted >1pp — update not needed")

    # Apply
    if args.apply:
        config_path = Path(__file__).parent / "model_config.py"
        text = config_path.read_text(encoding="utf-8")

        for config_key in ["white_nh", "black_nh", "hispanic", "other"]:
            new_val = avg[group_map[config_key]]
            pattern = rf'("{config_key}":\s*)[\d.]+'
            text = re.sub(pattern, rf'\g<1>{new_val:.3f}', text)

        today_str = date.today().isoformat()
        date_range_str = f"{polls[-1]['date_range']} to {polls[0]['date_range']}"
        new_source = f"Economist/YouGov {len(polls)}-week avg ({date_range_str})"

        text = re.sub(
            r'(GENERIC_BALLOT_SOURCE\s*[:=]\s*(?:str\s*=\s*)?)(["\']).*?\2',
            rf'\1\2{new_source}\2',
            text,
        )
        text = re.sub(
            r'(GENERIC_BALLOT_UPDATED\s*[:=]\s*(?:str\s*=\s*)?)(["\']).*?\2',
            rf'\1\2{today_str}\2',
            text,
        )
        text = re.sub(
            r'(GENERIC_BALLOT_TOPLINE_D_2P\s*[:=]\s*(?:float\s*=\s*)?)[\d.]+',
            rf'\g<1>{new_topline:.4f}',
            text,
        )

        config_path.write_text(text, encoding="utf-8")
        print(f"\n  model_config.py updated:")
        print(f"    Source:  {new_source}")
        print(f"    Date:    {today_str}")
        print(f"    Topline: {new_topline:.4f}")
        print(f"\n  Re-run model: python src/model.py")
    elif any_shift:
        print(f"\n  Run with --apply to update model_config.py")


if __name__ == "__main__":
    main()
