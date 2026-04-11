"""
update_polling.py

Update RACE_GENERIC_BALLOT_D_SHARE in model_config.py from multiple free public
polling sources, replacing the old single-poll DDHQ snapshot approach.

SOURCES:
  1. Economist/YouGov weekly crosstab PDFs (free, ~1,600 RV/week, weekly cadence).
     Provides full racial crosstabs (White/Black/Hispanic).
  2. Marist/NPR monthly crosstab PDFs (free, ~1,400 adults, monthly cadence).
     Provides full racial crosstabs (White/Black/Latino).
  3. Quinnipiac monthly PDFs — topline only (no racial crosstabs in 2025-2026
     format). Used for topline cross-check, not racial aggregation.
  4. Manual CSV (data/raw/racial_crosstab_inputs.csv) — any poll you want to add
     by hand (e.g., from Pew, ANES, or paywalled aggregators).

AGGREGATION:
  - Filter to last 45 days (configurable)
  - Exclude Rasmussen racial subgroups (known R-lean outlier; topline kept)
  - Recency weighting: last 14 days = 2x, 15-45 days = 1x
  - Weighted average across all sources with racial crosstabs
  - Other/Asian solved from topline constraint

USAGE:
  python src/update_polling.py                         # dry run: show comparison
  python src/update_polling.py --apply                 # update model_config.py
  python src/update_polling.py --sources yougov,marist # specific sources only
  python src/update_polling.py --force-download        # bypass PDF cache
  python src/update_polling.py --verbose               # detailed per-poll output
"""

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pdfplumber
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
CACHE_DIR = DATA_RAW / "poll_crosstabs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

USER_AGENT = "TXLegislativeModel/1.0 (academic research)"

# Aggregation parameters
WINDOW_DAYS = 45          # only polls within this many days
RECENCY_BOOST_DAYS = 14   # polls within this many days get 2x weight
RASMUSSEN_EXCLUDE_RACIAL = True  # exclude Rasmussen racial subgroups

# Validation guardrails
GUARDRAILS = {
    "black_nh":  (0.80, 0.95),  # D 2p share range
    "hispanic":  (0.45, 0.70),
    "white_nh":  (0.35, 0.50),
}
MIN_POLLS_FOR_UPDATE = 2        # require at least this many polls
DIVERGENCE_WARN_PP = 0.10       # warn if any group diverges >10pp from current


# ---------------------------------------------------------------------------
# Common data structure for poll results
# ---------------------------------------------------------------------------

@dataclass
class PollResult:
    """A single poll's racial generic ballot crosstab."""
    source: str           # e.g. "yougov", "marist", "quinnipiac", "manual"
    pollster: str         # e.g. "Economist/YouGov", "Marist/NPR"
    poll_date: date       # midpoint or end date of fieldwork
    date_label: str       # human-readable date range
    sample_size: int = 0

    # D two-party shares (0-1 scale). None = not available from this source.
    white_d2p: Optional[float] = None
    black_d2p: Optional[float] = None
    hispanic_d2p: Optional[float] = None
    topline_d2p: Optional[float] = None

    # Whether this poll has full racial crosstabs (vs topline only)
    has_racial: bool = False

    def summary_line(self, verbose: bool = False) -> str:
        parts = [f"  {self.pollster:25s} {self.date_label:30s}"]
        if self.has_racial:
            parts.append(f"W={self.white_d2p:.1%}" if self.white_d2p else "W=n/a")
            parts.append(f"B={self.black_d2p:.1%}" if self.black_d2p else "B=n/a")
            parts.append(f"H={self.hispanic_d2p:.1%}" if self.hispanic_d2p else "H=n/a")
        if self.topline_d2p is not None:
            parts.append(f"Top={self.topline_d2p:.1%}")
        if self.sample_size:
            parts.append(f"n={self.sample_size:,}")
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compute_d_2p(dem: float, rep: float) -> float:
    """D two-party share from raw D% and R% (ignoring undecided/other)."""
    return dem / (dem + rep) if (dem + rep) > 0 else 0.5


def download_pdf(label: str, url: str, subdir: str = "", force: bool = False) -> bytes | None:
    """Download a PDF with caching."""
    cache_subdir = CACHE_DIR / subdir if subdir else CACHE_DIR
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_subdir / f"{label}.pdf"
    if cache_path.exists() and not force:
        return cache_path.read_bytes()

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return resp.content
    except requests.RequestException as e:
        print(f"    FAILED downloading {label}: {e}")
        return None


def parse_date_from_label(label: str) -> date:
    """Parse a date label like '2026-03-27_to_30' into a date (uses end date)."""
    # Try YYYY-MM-DD_to_DD format
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_to_(\d{2})", label)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(4)))
    # Try ISO date
    try:
        return date.fromisoformat(label)
    except ValueError:
        return date.today()


# ---------------------------------------------------------------------------
# Source 1: Economist/YouGov weekly PDFs
# ---------------------------------------------------------------------------

# Known recent PDF URLs (most recent first).
# Update this list when new polls are published.
YOUGOV_PDFS = [
    ("2026-04-03_to_06", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_SVRZJH8.pdf"),
    ("2026-03-27_to_30", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_3wplfYX.pdf"),
    ("2026-03-20_to_23", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_o84FoNw.pdf"),
    ("2026-03-13_to_16", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_CwWXhS2.pdf"),
    ("2026-03-06_to_09", "https://d3nkl3psvxxpe9.cloudfront.net/documents/econTabReport_EcCnfRV.pdf"),
]


def fetch_yougov(n_weeks: int = 4, force: bool = False, verbose: bool = False) -> list[PollResult]:
    """Download and parse YouGov weekly crosstab PDFs."""
    results = []
    for label, url in YOUGOV_PDFS[:n_weeks]:
        pdf_bytes = download_pdf(label, url, subdir="yougov", force=force)
        if not pdf_bytes:
            continue
        parsed = _parse_yougov_pdf(pdf_bytes)
        if not parsed:
            if verbose:
                print(f"    {label}: could not extract generic ballot race data")
            continue

        poll_date = parse_date_from_label(label)
        w = compute_d_2p(parsed["white"]["dem"], parsed["white"]["rep"])
        b = compute_d_2p(parsed["black"]["dem"], parsed["black"]["rep"])
        h = compute_d_2p(parsed["hispanic"]["dem"], parsed["hispanic"]["rep"])
        t = compute_d_2p(parsed["total"]["dem"], parsed["total"]["rep"])

        results.append(PollResult(
            source="yougov",
            pollster="Economist/YouGov",
            poll_date=poll_date,
            date_label=parsed.get("date_range", label),
            sample_size=parsed.get("sample_size", 0),
            white_d2p=w, black_d2p=b, hispanic_d2p=h, topline_d2p=t,
            has_racial=True,
        ))
    return results


def _parse_yougov_pdf(pdf_bytes: bytes) -> dict | None:
    """Parse GenericCongressionalVote race crosstab from YouGov PDF."""
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))

    date_range = ""
    first_text = pdf.pages[0].extract_text() or ""
    m = re.search(r"(\w+ \d+ [- ] \d+, \d{4})", first_text)
    if m:
        date_range = m.group(1)

    for page in pdf.pages:
        text = page.extract_text() or ""
        if "genericcongressionalvote" not in text.lower():
            continue

        lines = text.split("\n")
        header_line = dem_line = rep_line = n_line = None

        for line in lines:
            if "Total" in line and "White" in line and "Black" in line and "Hispanic" in line:
                header_line = line
            elif "DemocraticParty" in line.replace(" ", ""):
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

        def extract_pcts(line: str) -> list[float]:
            return [float(p) for p in re.findall(r"(\d+)%", line)]

        dem_pcts = extract_pcts(dem_line)
        rep_pcts = extract_pcts(rep_line)

        headers = re.split(r"\s+", header_line.strip())
        col_map = {h.lower(): i for i, h in enumerate(headers)}

        white_idx = col_map.get("white")
        black_idx = col_map.get("black")
        hisp_idx = col_map.get("hispanic")

        if white_idx is None or black_idx is None or hisp_idx is None:
            continue

        max_needed = max(white_idx, black_idx, hisp_idx)
        if len(dem_pcts) <= max_needed or len(rep_pcts) <= max_needed:
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
# Source 2: Marist/NPR monthly PDFs
# ---------------------------------------------------------------------------

# Known recent Marist/NPR PDF URLs (most recent first).
# Landing page: maristpoll.marist.edu/npr-pbs-newshour-marist-poll/
MARIST_PDFS = [
    ("2026-03-02_to_04", "https://maristpoll.marist.edu/wp-content/uploads/2026/03/NPR_PBS-News_Marist-Poll_USA-NOS-and-Tables_202603091024.pdf"),
]


def fetch_marist(force: bool = False, verbose: bool = False) -> list[PollResult]:
    """Download and parse Marist/NPR monthly crosstab PDFs."""
    results = []
    for label, url in MARIST_PDFS:
        pdf_bytes = download_pdf(label, url, subdir="marist", force=force)
        if not pdf_bytes:
            continue
        parsed = _parse_marist_pdf(pdf_bytes, verbose=verbose)
        if not parsed:
            if verbose:
                print(f"    Marist {label}: could not extract USCNGS01 race data")
            continue

        poll_date = parse_date_from_label(label)
        results.append(PollResult(
            source="marist",
            pollster="Marist/NPR",
            poll_date=poll_date,
            date_label=parsed.get("date_label", label),
            sample_size=parsed.get("sample_size", 0),
            white_d2p=parsed["white_d2p"],
            black_d2p=parsed["black_d2p"],
            hispanic_d2p=parsed["hispanic_d2p"],
            topline_d2p=parsed["topline_d2p"],
            has_racial=True,
        ))
    return results


def _parse_marist_pdf(pdf_bytes: bytes, verbose: bool = False) -> dict | None:
    """
    Parse USCNGS01 (generic congressional ballot) race crosstab from Marist PDF.

    Layout:
      - Page starts with "USCNGS01."
      - Topline row: "National Registered Voters  53%  44%  2%  1%"
      - Race section appears twice: first collapsed (White/Non-white),
        then expanded (White/Black/Latino).
      - Column order: Democrat%, Republican%, Other%, Unsure%
    """
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))

    for page in pdf.pages:
        text = page.extract_text() or ""
        # Must be the crosstab page, not the trend page
        if "USCNGS01" not in text:
            continue
        first_line = text.split("\n")[0]
        if "TRND" in first_line:
            continue

        lines = text.split("\n")

        # Extract date range from header
        date_label = ""
        date_m = re.search(r"(\w+ \d+\w* through \w+ \d+\w*, \d{4})", text)
        if date_m:
            date_label = date_m.group(1)

        # Parse topline: "National Registered Voters  53%  44%  ..."
        topline_d2p = None
        for line in lines:
            if "National Registered Voters" in line:
                pcts = re.findall(r"(\d+)%", line)
                if len(pcts) >= 2:
                    topline_d2p = compute_d_2p(float(pcts[0]), float(pcts[1]))
                break

        # Find the expanded Race/Ethnicity block (second occurrence of
        # "Race/Ethnicity White" line, followed by Black and Latino rows)
        race_white_indices = []
        for i, line in enumerate(lines):
            # Match lines starting with race label or continuation
            if re.match(r"Race/Ethnicity\s+White", line) or re.match(r"Race/Ethnicity White", line):
                race_white_indices.append(i)

        if len(race_white_indices) < 2:
            if verbose:
                print(f"    Marist: found {len(race_white_indices)} Race/Ethnicity White lines, need 2")
            pdf.close()
            return None

        # Use the second occurrence (expanded block)
        white_idx = race_white_indices[1]
        white_line = lines[white_idx]
        black_line = lines[white_idx + 1] if white_idx + 1 < len(lines) else ""
        latino_line = lines[white_idx + 2] if white_idx + 2 < len(lines) else ""

        if verbose:
            print(f"    Marist race lines:")
            print(f"      White:  {white_line}")
            print(f"      Black:  {black_line}")
            print(f"      Latino: {latino_line}")

        def parse_row_d2p(line: str) -> Optional[float]:
            pcts = re.findall(r"(\d+)%", line)
            if len(pcts) >= 2:
                return compute_d_2p(float(pcts[0]), float(pcts[1]))
            return None

        white_d2p = parse_row_d2p(white_line)
        black_d2p = parse_row_d2p(black_line)
        hispanic_d2p = parse_row_d2p(latino_line)

        # Validate that Black and Latino lines look right
        black_ok = black_line.strip().startswith("Black") or "Black" in black_line
        latino_ok = "Latino" in latino_line or "Hispanic" in latino_line
        if not (black_ok and latino_ok):
            if verbose:
                print(f"    Marist: unexpected race row labels (Black={black_ok}, Latino={latino_ok})")
            pdf.close()
            return None

        if white_d2p is None or black_d2p is None or hispanic_d2p is None:
            if verbose:
                print(f"    Marist: could not parse all race rows")
            pdf.close()
            return None

        # Sample size from methodology (approximate — Marist doesn't always put N on crosstab page)
        sample_size = 0
        n_m = re.search(r"(\d[,\d]+)\s+(?:National|Adults|Registered)", text)
        if n_m:
            sample_size = int(n_m.group(1).replace(",", ""))

        pdf.close()
        return {
            "white_d2p": white_d2p,
            "black_d2p": black_d2p,
            "hispanic_d2p": hispanic_d2p,
            "topline_d2p": topline_d2p,
            "date_label": date_label,
            "sample_size": sample_size,
        }

    pdf.close()
    return None


# ---------------------------------------------------------------------------
# Source 3: Quinnipiac (topline only — no racial crosstabs in 2025-2026 format)
# ---------------------------------------------------------------------------

# Known recent Quinnipiac PDFs with generic ballot question (most recent first).
# Not all Quinnipiac releases include generic ballot — only list those that do.
QUINNIPIAC_PDFS = [
    ("2025-12-17", "https://poll.qu.edu/images/polling/us/us12172025_ugli25.pdf"),
]


def fetch_quinnipiac(force: bool = False, verbose: bool = False) -> list[PollResult]:
    """
    Download and parse Quinnipiac PDFs for generic ballot topline.
    NOTE: Quinnipiac 2025-2026 PDFs do NOT include racial crosstabs.
    These results are topline-only, used for cross-checking the aggregate.
    """
    results = []
    for label, url in QUINNIPIAC_PDFS:
        pdf_bytes = download_pdf(label, url, subdir="quinnipiac", force=force)
        if not pdf_bytes:
            continue
        parsed = _parse_quinnipiac_pdf(pdf_bytes, verbose=verbose)
        if not parsed:
            if verbose:
                print(f"    Quinnipiac {label}: no generic ballot found")
            continue

        poll_date = parse_date_from_label(label)
        results.append(PollResult(
            source="quinnipiac",
            pollster="Quinnipiac",
            poll_date=poll_date,
            date_label=parsed.get("date_label", label),
            sample_size=parsed.get("sample_size", 0),
            topline_d2p=parsed["topline_d2p"],
            has_racial=False,  # no racial crosstabs available
        ))
    return results


def _parse_quinnipiac_pdf(pdf_bytes: bytes, verbose: bool = False) -> dict | None:
    """
    Parse generic congressional ballot topline from Quinnipiac PDF.

    Format:
      "If the election were today, would you want to see the Republican Party
       or the Democratic Party win control of the United States House..."
      REGISTERED VOTERS.....................
      Tot  Rep  Dem  Ind  Men  Wom
      Republican Party  43%  92%   2%  38%  49%  38%
      Democratic Party  47    5   95   46   40   53
    """
    pdf = pdfplumber.open(io.BytesIO(pdf_bytes))

    for page in pdf.pages:
        text = page.extract_text() or ""
        if "House of Representatives" not in text:
            continue
        if "Republican Party" not in text or "Democratic Party" not in text:
            continue

        lines = text.split("\n")
        rep_pct = dem_pct = None

        for line in lines:
            # First data line has % signs; subsequent lines may not
            if line.strip().startswith("Republican Party"):
                pcts = re.findall(r"(\d+)%?", line)
                if pcts:
                    rep_pct = float(pcts[0])
            elif line.strip().startswith("Democratic Party"):
                pcts = re.findall(r"(\d+)%?", line)
                if pcts:
                    dem_pct = float(pcts[0])

        if rep_pct is not None and dem_pct is not None:
            # Extract date from page text
            date_label = ""
            date_m = re.search(r"(\w+ \d+\s*[-–]\s*\d+,?\s*\d{4})", text)
            if date_m:
                date_label = date_m.group(1)

            pdf.close()
            return {
                "topline_d2p": compute_d_2p(dem_pct, rep_pct),
                "date_label": date_label or "Quinnipiac",
                "sample_size": 0,
            }

    pdf.close()
    return None


# ---------------------------------------------------------------------------
# Source 4: Manual CSV
# ---------------------------------------------------------------------------

MANUAL_CSV_PATH = DATA_RAW / "racial_crosstab_inputs.csv"


def fetch_manual(verbose: bool = False) -> list[PollResult]:
    """
    Read manually entered poll crosstabs from CSV.

    Expected columns:
      poll_date, pollster, white_d_2p, black_d_2p, hispanic_d_2p,
      topline_d_2p, sample_size, source_url, notes
    """
    if not MANUAL_CSV_PATH.exists():
        if verbose:
            print(f"    No manual CSV at {MANUAL_CSV_PATH}")
        return []

    results = []
    with open(MANUAL_CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                poll_date = date.fromisoformat(row["poll_date"].strip())
            except (ValueError, KeyError):
                if verbose:
                    print(f"    Skipping manual row with bad date: {row.get('poll_date')}")
                continue

            pollster = row.get("pollster", "Manual").strip()

            # Parse D 2p shares (already in 0-1 scale in CSV)
            def safe_float(val: str) -> Optional[float]:
                val = val.strip()
                if not val:
                    return None
                return float(val)

            white = safe_float(row.get("white_d_2p", ""))
            black = safe_float(row.get("black_d_2p", ""))
            hispanic = safe_float(row.get("hispanic_d_2p", ""))
            topline = safe_float(row.get("topline_d_2p", ""))

            has_racial = all(v is not None for v in [white, black, hispanic])

            results.append(PollResult(
                source="manual",
                pollster=pollster,
                poll_date=poll_date,
                date_label=row.get("notes", poll_date.isoformat()),
                sample_size=int(row.get("sample_size", "0") or "0"),
                white_d2p=white,
                black_d2p=black,
                hispanic_d2p=hispanic,
                topline_d2p=topline,
                has_racial=has_racial,
            ))

    if verbose and results:
        print(f"    Loaded {len(results)} manual entries")
    return results


# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

def aggregate_polls(polls: list[PollResult], window_days: int = WINDOW_DAYS,
                    verbose: bool = False) -> dict:
    """
    Aggregate multiple poll results into weighted racial crosstab averages.

    Returns dict with:
      white_d2p, black_d2p, hispanic_d2p, topline_d2p, n_polls, n_racial,
      topline_sources (list of topline-only polls for cross-check)
    """
    today = date.today()
    cutoff = today - timedelta(days=window_days)

    # Filter to window
    in_window = [p for p in polls if p.poll_date >= cutoff]
    out_of_window = [p for p in polls if p.poll_date < cutoff]

    if verbose and out_of_window:
        print(f"\n  Excluded {len(out_of_window)} polls outside {window_days}-day window:")
        for p in out_of_window:
            print(f"    {p.pollster} ({p.poll_date}) — {(today - p.poll_date).days} days old")

    # Exclude Rasmussen racial subgroups
    if RASMUSSEN_EXCLUDE_RACIAL:
        excluded_ras = [p for p in in_window if "rasmussen" in p.pollster.lower() and p.has_racial]
        if excluded_ras and verbose:
            print(f"\n  Excluded {len(excluded_ras)} Rasmussen racial subgroup entries (known R-lean)")
        # Keep Rasmussen topline, remove racial data
        for p in excluded_ras:
            p.has_racial = False
            p.white_d2p = p.black_d2p = p.hispanic_d2p = None

    # Separate racial and topline-only polls
    racial_polls = [p for p in in_window if p.has_racial]
    topline_polls = [p for p in in_window if not p.has_racial and p.topline_d2p is not None]

    # Compute recency weights
    def recency_weight(p: PollResult) -> float:
        age = (today - p.poll_date).days
        return 2.0 if age <= RECENCY_BOOST_DAYS else 1.0

    # Weighted average of racial crosstabs
    result = {
        "n_polls": len(in_window),
        "n_racial": len(racial_polls),
        "n_topline_only": len(topline_polls),
        "topline_sources": topline_polls,
    }

    if not racial_polls:
        result["white_d2p"] = result["black_d2p"] = result["hispanic_d2p"] = None
        result["topline_d2p"] = None
        return result

    total_weight = sum(recency_weight(p) for p in racial_polls)
    for group in ["white_d2p", "black_d2p", "hispanic_d2p"]:
        vals = [(getattr(p, group), recency_weight(p)) for p in racial_polls
                if getattr(p, group) is not None]
        if vals:
            result[group] = sum(v * w for v, w in vals) / sum(w for _, w in vals)
        else:
            result[group] = None

    # Topline: weighted average from ALL polls with topline (racial + topline-only)
    all_with_topline = [p for p in in_window if p.topline_d2p is not None]
    if all_with_topline:
        t_weight = sum(recency_weight(p) for p in all_with_topline)
        result["topline_d2p"] = sum(
            p.topline_d2p * recency_weight(p) for p in all_with_topline
        ) / t_weight
    else:
        result["topline_d2p"] = None

    return result


def solve_other_from_topline(white_d2p: float, black_d2p: float, hisp_d2p: float,
                              topline_d2p: float) -> float:
    """
    Back-solve for other/Asian D 2p share from the topline constraint.
    topline ≈ Σ(weight × group_d2p)
    """
    from model_config import NATIONAL_DEMO_WEIGHTS as w
    numerator = topline_d2p - (w["white_nh"] * white_d2p +
                                w["black_nh"] * black_d2p +
                                w["hispanic"] * hisp_d2p)
    if w["other"] > 0:
        return max(0.0, min(1.0, numerator / w["other"]))
    return 0.5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_results(agg: dict, current: dict, verbose: bool = False) -> list[str]:
    """Check aggregated results against guardrails. Returns list of warnings."""
    warnings = []

    group_map = {"white_nh": "white_d2p", "black_nh": "black_d2p", "hispanic": "hispanic_d2p"}

    for config_key, (lo, hi) in GUARDRAILS.items():
        agg_key = group_map[config_key]
        val = agg.get(agg_key)
        if val is None:
            continue
        if val < lo or val > hi:
            warnings.append(
                f"  WARNING: {config_key} = {val:.1%} outside expected range "
                f"[{lo:.0%}, {hi:.0%}]"
            )

    # Divergence from current values
    for config_key, agg_key in group_map.items():
        val = agg.get(agg_key)
        if val is None:
            continue
        old = current.get(config_key, 0.5)
        if abs(val - old) > DIVERGENCE_WARN_PP:
            warnings.append(
                f"  WARNING: {config_key} shifted {val - old:+.1%} from current — "
                f"verify source data"
            )

    if agg["n_racial"] < MIN_POLLS_FOR_UPDATE:
        warnings.append(
            f"  WARNING: Only {agg['n_racial']} polls with racial crosstabs "
            f"(minimum {MIN_POLLS_FOR_UPDATE})"
        )

    return warnings


# ---------------------------------------------------------------------------
# Config update
# ---------------------------------------------------------------------------

def apply_to_config(agg: dict, other_d2p: float):
    """Write aggregated values to model_config.py."""
    config_path = Path(__file__).parent / "model_config.py"
    text = config_path.read_text(encoding="utf-8")

    new_vals = {
        "white_nh": agg["white_d2p"],
        "black_nh": agg["black_d2p"],
        "hispanic": agg["hispanic_d2p"],
        "other": other_d2p,
    }

    # Only replace values in RACE_GENERIC_BALLOT_D_SHARE, not NATIONAL_DEMO_WEIGHTS.
    # Find the RACE_GENERIC_BALLOT block and replace only within it.
    block_start = text.index("RACE_GENERIC_BALLOT_D_SHARE")
    block_end = text.index("}", block_start) + 1
    block = text[block_start:block_end]
    for config_key, new_val in new_vals.items():
        pattern = rf'("{config_key}":\s*)[\d.]+'
        block = re.sub(pattern, rf'\g<1>{new_val:.4f}', block)
    text = text[:block_start] + block + text[block_end:]

    today_str = date.today().isoformat()
    new_topline = agg["topline_d2p"] or 0.5

    # Build source description
    sources_used = set()
    # We don't have the raw polls here, so describe from agg
    source_str = f"Multi-source {agg['n_racial']}-poll racial avg + {agg['n_topline_only']} topline-only"

    text = re.sub(
        r'(GENERIC_BALLOT_SOURCE\s*[:=]\s*(?:str\s*=\s*)?)(["\']).*?\2',
        rf'\1\2{source_str}\2',
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
    return today_str, source_str, new_topline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_SOURCES = ["yougov", "marist", "quinnipiac", "manual"]


def main():
    parser = argparse.ArgumentParser(
        description="Update racial generic ballot crosstabs from multiple polling sources")
    parser.add_argument("--apply", action="store_true",
                        help="Update model_config.py in place (default: dry run)")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download PDFs even if cached")
    parser.add_argument("--sources", type=str, default=None,
                        help=f"Comma-separated sources to use (default: all). "
                             f"Options: {', '.join(ALL_SOURCES)}")
    parser.add_argument("--n-weeks", type=int, default=4,
                        help="Number of recent YouGov polls to fetch (default: 4)")
    parser.add_argument("--window", type=int, default=WINDOW_DAYS,
                        help=f"Aggregation window in days (default: {WINDOW_DAYS})")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed per-poll output")
    args = parser.parse_args()

    # Parse source filter
    if args.sources:
        sources = [s.strip().lower() for s in args.sources.split(",")]
        invalid = [s for s in sources if s not in ALL_SOURCES]
        if invalid:
            print(f"Unknown sources: {', '.join(invalid)}")
            print(f"Valid sources: {', '.join(ALL_SOURCES)}")
            sys.exit(1)
    else:
        sources = ALL_SOURCES

    print("=" * 65)
    print("  Update Racial Generic Ballot Crosstabs")
    print(f"  Sources: {', '.join(sources)}")
    print(f"  Window: {args.window} days | Recency boost: last {RECENCY_BOOST_DAYS} days = 2×")
    print("=" * 65)

    # Fetch from each source
    all_polls: list[PollResult] = []

    if "yougov" in sources:
        print(f"\n[YouGov] Fetching {args.n_weeks} most recent weekly polls...")
        yg = fetch_yougov(n_weeks=args.n_weeks, force=args.force_download, verbose=args.verbose)
        all_polls.extend(yg)
        print(f"  → {len(yg)} polls with racial crosstabs")

    if "marist" in sources:
        print(f"\n[Marist/NPR] Fetching monthly polls...")
        mr = fetch_marist(force=args.force_download, verbose=args.verbose)
        all_polls.extend(mr)
        print(f"  → {len(mr)} polls with racial crosstabs")

    if "quinnipiac" in sources:
        print(f"\n[Quinnipiac] Fetching monthly polls (topline only — no racial crosstabs)...")
        qu = fetch_quinnipiac(force=args.force_download, verbose=args.verbose)
        all_polls.extend(qu)
        print(f"  → {len(qu)} polls (topline only)")

    if "manual" in sources:
        print(f"\n[Manual CSV] Reading {MANUAL_CSV_PATH.name}...")
        mn = fetch_manual(verbose=args.verbose)
        all_polls.extend(mn)
        if mn:
            racial_mn = sum(1 for p in mn if p.has_racial)
            print(f"  → {len(mn)} entries ({racial_mn} with racial crosstabs)")
        else:
            print(f"  → No manual CSV found (create {MANUAL_CSV_PATH.relative_to(ROOT)} to add)")

    if not all_polls:
        print("\nNo polls fetched from any source. Cannot update.")
        return

    # Show all polls
    print(f"\n{'─' * 65}")
    print(f"  All polls collected: {len(all_polls)}")
    print(f"{'─' * 65}")
    for p in sorted(all_polls, key=lambda x: x.poll_date, reverse=True):
        marker = "●" if p.has_racial else "○"
        age = (date.today() - p.poll_date).days
        weight = "2×" if age <= RECENCY_BOOST_DAYS else "1×"
        in_window = age <= args.window
        status = f"{weight}" if in_window else "excluded"
        print(f"  {marker} {p.summary_line():<75s}  [{age}d, {status}]")
    print(f"  ● = racial crosstabs | ○ = topline only")

    # Aggregate
    agg = aggregate_polls(all_polls, window_days=args.window, verbose=args.verbose)

    if agg["white_d2p"] is None:
        print("\nNo racial crosstab data available within window. Cannot update.")
        if agg["topline_sources"]:
            print("  (Topline-only polls found but racial breakdown is needed)")
        return

    # Solve other/Asian from topline
    if agg["topline_d2p"] is not None:
        other_d2p = solve_other_from_topline(
            agg["white_d2p"], agg["black_d2p"], agg["hispanic_d2p"], agg["topline_d2p"]
        )
    else:
        other_d2p = 0.5  # fallback

    # Load current config
    from model_config import (
        RACE_GENERIC_BALLOT_D_SHARE as current,
        GENERIC_BALLOT_SOURCE,
        GENERIC_BALLOT_UPDATED,
        GENERIC_BALLOT_TOPLINE_D_2P,
        NATIONAL_DEMO_WEIGHTS,
    )

    # Compute implied topline from new values
    new_vals = {
        "white_nh": agg["white_d2p"],
        "black_nh": agg["black_d2p"],
        "hispanic": agg["hispanic_d2p"],
        "other": other_d2p,
    }
    new_topline = sum(NATIONAL_DEMO_WEIGHTS[k] * new_vals[k] for k in NATIONAL_DEMO_WEIGHTS)

    # Validate
    warnings = validate_results(agg, current, verbose=args.verbose)

    # Print comparison
    print(f"\n{'=' * 65}")
    print(f"  COMPARISON: Current Config vs Multi-Source Aggregate")
    print(f"{'=' * 65}")
    print(f"  Polls in window: {agg['n_racial']} racial + {agg['n_topline_only']} topline-only")

    print(f"\n  {'Group':15s}  {'Current':>8s}  {'New':>8s}  {'Delta':>8s}  {'Flag':>5s}")
    print(f"  {'-' * 50}")

    any_shift = False
    for config_key in ["white_nh", "black_nh", "hispanic", "other"]:
        old = current[config_key]
        new = new_vals[config_key]
        delta = new - old
        flag = " ***" if abs(delta) > 0.01 else ""
        if abs(delta) > 0.01:
            any_shift = True
        print(f"  {config_key:15s}  {old:8.1%}  {new:8.1%}  {delta:+8.1%}{flag}")

    print(f"\n  {'Topline D 2p':15s}  {GENERIC_BALLOT_TOPLINE_D_2P:8.1%}  {new_topline:8.1%}  "
          f"{new_topline - GENERIC_BALLOT_TOPLINE_D_2P:+8.1%}")
    print(f"\n  Current source: {GENERIC_BALLOT_SOURCE}")
    print(f"  Current date:   {GENERIC_BALLOT_UPDATED}")

    # Topline cross-check from topline-only sources
    if agg["topline_sources"]:
        print(f"\n  Topline cross-check (non-racial sources):")
        for p in agg["topline_sources"]:
            diff = p.topline_d2p - new_topline if p.topline_d2p else 0
            print(f"    {p.pollster:20s}  {p.topline_d2p:.1%}  (vs aggregate: {diff:+.1%})")

    # Warnings
    if warnings:
        print(f"\n{'─' * 65}")
        for w in warnings:
            print(w)
        print(f"{'─' * 65}")

    if any_shift:
        print(f"\n  *** One or more groups shifted >1pp — update recommended")
    else:
        print(f"\n  No group shifted >1pp — update not needed")

    # Apply
    if args.apply:
        if agg["n_racial"] < MIN_POLLS_FOR_UPDATE:
            print(f"\n  BLOCKED: Need at least {MIN_POLLS_FOR_UPDATE} racial polls to apply "
                  f"(have {agg['n_racial']})")
            return

        today_str, source_str, topline_val = apply_to_config(agg, other_d2p)
        print(f"\n  model_config.py updated:")
        print(f"    Source:  {source_str}")
        print(f"    Date:    {today_str}")
        print(f"    Topline: {topline_val:.4f}")
        print(f"\n  Re-run model: python src/model.py")
    elif any_shift:
        print(f"\n  Run with --apply to update model_config.py")


if __name__ == "__main__":
    main()
