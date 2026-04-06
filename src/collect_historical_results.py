"""
collect_historical_results.py

Fetch and parse Texas House and Senate general election results for
historical off-year cycles: 2002, 2006, 2010, 2014, 2018, 2022.

Strategy:
  1. Check for a locally cached Wikipedia XML export in data/raw/.
     (The 2022 Senate XML already exists from prior work.)
  2. If not cached, download via Wikipedia Special:Export.
  3. Parse wikitext using an extended election-box parser that also
     captures the incumbent flag and handles older template variants.

Output: data/raw/historical/tx_{chamber}_results_{year}.csv
Columns:
  year, chamber, district, r_candidate, r_pct, d_candidate, d_pct,
  r_incumbent, d_incumbent, dem_2p_share, winner_party, contested,
  on_ballot, notes

Usage:
  python src/collect_historical_results.py
  python src/collect_historical_results.py --year 2018 --chamber house
"""

import argparse
import csv
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_HIST = DATA_RAW / "historical"
DATA_HIST.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Wikipedia page titles by (chamber, year)
# ---------------------------------------------------------------------------

WIKI_PAGES = {
    # All titles verified via Wikipedia API search (note: singular "election")
    ("house", 2022): "2022 Texas House of Representatives election",
    ("house", 2018): "2018 Texas House of Representatives election",
    ("house", 2014): "2014 Texas House of Representatives election",
    ("house", 2010): "2010 Texas House of Representatives election",
    ("house", 2006): "2006 Texas House of Representatives election",
    ("house", 2002): "2002 Texas House of Representatives election",
    ("senate", 2022): "2022 Texas Senate election",
    ("senate", 2018): "2018 Texas Senate election",
    ("senate", 2014): "2014 Texas Senate election",
    ("senate", 2010): "2010 Texas Senate election",
    ("senate", 2006): "2006 Texas Senate election",
    ("senate", 2002): "2002 Texas Senate election",
}

# Existing cached XMLs from prior project work (skip download if present)
EXISTING_CACHE = {
    ("senate", 2022): DATA_RAW / "_wiki_senate_2022_raw.xml",
}

MAX_DISTRICTS = {"house": 150, "senate": 31}

PARTY_MAP = {
    "republican party (united states)": "R",
    "republican party of texas": "R",
    "republican": "R",
    "democratic party (united states)": "D",
    "democratic party of texas": "D",
    "democratic": "D",
    "democrat": "D",
}

WIKI_EXPORT_URL = "https://en.wikipedia.org/w/index.php"
USER_AGENT = "TXLegislativeModel/1.0 (academic research; non-commercial)"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def xml_cache_path(chamber: str, year: int) -> Path:
    return DATA_RAW / f"_wiki_{chamber}_{year}_raw.xml"


def download_wikipedia_xml(chamber: str, year: int) -> Path:
    """
    Download a Wikipedia Special:Export XML for the given chamber/year page.
    Caches to data/raw/_wiki_{chamber}_{year}_raw.xml.
    Skips download if file already exists.
    """
    # Check for pre-existing cache (prior project convention)
    existing = EXISTING_CACHE.get((chamber, year))
    if existing and existing.exists():
        print(f"  Using existing cache: {existing.name}")
        return existing

    out = xml_cache_path(chamber, year)
    if out.exists():
        print(f"  Cache hit: {out.name}")
        return out

    title = WIKI_PAGES.get((chamber, year))
    if not title:
        raise ValueError(f"No Wikipedia page title configured for ({chamber}, {year})")

    print(f"  Downloading: {title}")
    params = {
        "title": "Special:Export",
        "pages": title,
        "action": "submit",
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(WIKI_EXPORT_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    out.write_bytes(resp.content)
    print(f"  Saved {len(resp.content):,} bytes -> {out.name}")
    time.sleep(1.0)  # be polite to Wikipedia
    return out


# ---------------------------------------------------------------------------
# XML → wikitext
# ---------------------------------------------------------------------------

def load_wikitext(xml_path: Path) -> str:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    text_el = root.find(".//mw:revision/mw:text", ns)
    if text_el is None:
        text_el = root.find(".//revision/text")
    return text_el.text if text_el is not None and text_el.text else ""


# ---------------------------------------------------------------------------
# Parser — Election box templates
# ---------------------------------------------------------------------------

def extract_incumbent_flag(raw: str) -> bool:
    """Return True if the raw candidate field contains '(incumbent)'."""
    return bool(re.search(r'\(incumbent\)', raw, re.IGNORECASE))


def clean_candidate_name(raw: str) -> str:
    """Strip wikilinks and '(incumbent)' annotation from a candidate field."""
    name = re.sub(r'\[\[[^\]]*\|([^\]]+)\]\]', r'\1', raw)   # [[A|B]] → B
    name = re.sub(r'\[\[([^\]]+)\]\]', r'\1', name)           # [[A]] → A
    name = re.sub(r'\[\[.*', '', name)                         # truncated wikilink
    name = re.sub(r'\s*\(incumbent\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'<[^>]+>', '', name)                        # strip HTML tags
    return name.strip()


def parse_election_box(block: str) -> list[dict]:
    """
    Parse candidates from one Election box begin...end block.
    Handles:
      - Multiline (2022/2024 style): each field on its own line
      - Inline (2022 style): fields separated by pipes
      - Variant template name: 'candidate with party link'
    Returns list of {party, candidate, votes, pct, incumbent}.
    """
    candidates = []

    # Split on each candidate entry template (winning or non-winning, with optional suffix).
    # Handles all template name variants:
    #   {{Election box winning candidate |...}}                   (2024 style)
    #   {{Election box candidate |...}}
    #   {{Election box winning candidate with party link no change\n  |...}}  (2022 style)
    # [^|{}]* matches any suffix (like " with party link no change\n  ") up to the first |
    parts = re.split(
        r'\{\{Election box (?:winning )?candidate[^|{}]*(?=\|)',
        block,
        flags=re.IGNORECASE,
    )

    for part in parts[1:]:
        part_norm = part.replace("|", "\n|")

        party_m = re.search(r'\|\s*party\s*=\s*([^\n|}\[]+)', part_norm)
        # candidate field may span a wikilink before hitting a pipe/newline
        cand_m = re.search(
            r'\|\s*candidate\s*=\s*([^\n|}]*(?:\[\[[^\]]*\][^\n|}]*)?[^\n|}]*)',
            part_norm,
        )
        votes_m = re.search(r'\|\s*votes\s*=\s*([\d,]+)', part_norm)
        # percentage may be spelled 'percentage' or 'percent'
        pct_m = re.search(r'\|\s*percent(?:age)?\s*=\s*([\d.]+)', part_norm)

        if not (party_m and pct_m):
            continue

        party_raw = party_m.group(1).strip().lower()
        party = PARTY_MAP.get(party_raw)
        if party is None:
            # Try partial match (some pages use abbreviated party names)
            for key, val in PARTY_MAP.items():
                if key in party_raw:
                    party = val
                    break
            if party is None:
                party = "?"

        cand_raw = cand_m.group(1) if cand_m else ""
        incumbent = extract_incumbent_flag(cand_raw)
        cand = clean_candidate_name(cand_raw)
        votes = int(votes_m.group(1).replace(",", "")) if votes_m else None
        pct = float(pct_m.group(1))

        candidates.append({
            "party": party,
            "candidate": cand,
            "votes": votes,
            "pct": pct,
            "incumbent": incumbent,
        })

    return candidates


# ---------------------------------------------------------------------------
# Parser — wikitable fallback (older articles, pre-2012)
# ---------------------------------------------------------------------------

def parse_wikitable_district(section_text: str) -> dict | None:
    """
    Attempt to extract R/D vote percentages from a wikitable row for a district.
    Older Wikipedia articles (2002-2010 era) may use tables instead of Election boxes.

    Looks for a row pattern like:
      | Candidate Name || N,NNN || NN.N || Candidate Name || N,NNN || NN.N

    Returns a partial row dict or None if not parseable.
    """
    # Find any wikitable in the section
    table_m = re.search(r'\{\|.*?\|\}', section_text, re.DOTALL)
    if not table_m:
        return None

    table = table_m.group()
    rows = re.split(r'^\|-', table, flags=re.MULTILINE)

    r_pct = d_pct = r_cand = d_cand = None
    r_inc = d_inc = False

    for row in rows[1:]:  # skip header row
        cells = [c.strip() for c in re.split(r'\|\||\n\|', row) if c.strip()]
        # Filter out header cells (contain !)
        cells = [c for c in cells if not c.startswith('!')]

        # Try to find percentage-like values (e.g., 65.2 or 65.23)
        pcts_found = []
        names_found = []
        for cell in cells:
            pct_match = re.fullmatch(r'[\d]{1,3}\.[\d]{1,2}', cell.strip('| '))
            if pct_match:
                pcts_found.append(float(cell.strip()))
            elif re.search(r'[A-Za-z]{3,}', cell):
                names_found.append(clean_candidate_name(cell))

        if len(pcts_found) >= 2 and len(names_found) >= 1:
            # Assume first pct is R, second is D (common table ordering)
            # This is a heuristic — may be wrong for some years
            r_pct = pcts_found[0]
            d_pct = pcts_found[1]
            if names_found:
                r_cand = names_found[0]
            if len(names_found) > 1:
                d_cand = names_found[1]
            break

    if r_pct is None:
        return None

    return {
        "r_candidate": r_cand or "",
        "r_pct": r_pct,
        "d_candidate": d_cand or "",
        "d_pct": d_pct,
        "r_incumbent": r_inc,
        "d_incumbent": d_inc,
        "notes": "wikitable_parse",
    }


# ---------------------------------------------------------------------------
# District-level parse
# ---------------------------------------------------------------------------

def find_district_section(wikitext: str, district: int, max_district: int) -> str:
    """Extract the wikitext section for a given district number."""
    section_pat = rf'===?\s*District\s*{district}\s*===?'
    next_pat = rf'===?\s*District\s*{district + 1}\s*===?' if district < max_district else None

    sec_m = re.search(section_pat, wikitext, re.IGNORECASE)
    if not sec_m:
        return ""

    start = sec_m.start()
    if next_pat:
        next_m = re.search(next_pat, wikitext[start:], re.IGNORECASE)
        return wikitext[start: start + next_m.start()] if next_m else wikitext[start:]
    return wikitext[start:]


def find_election_block(section_text: str, district: int) -> str | None:
    """
    Find the general/main election box within a district section.

    Two strategies:
    1. Title pattern (2024 format): look for blocks titled 'District N general election'.
    2. First non-primary block (2022 format): the 2022 articles use titles like
       '41st District' or a wikilink; pick the first block whose title does NOT
       contain 'primary', 'runoff', 'special', 'caucus'.
    """
    # Strategy 1: titled general election block (2024+ format)
    for pat in [
        rf'title\s*=\s*District\s*{district}\s*general election',
        rf'title\s*=\s*District\s*{district}\s*election',
        rf'title\s*=\s*(?:.*?)?{district}(?:th|st|nd|rd)?\s*(?:district)?\s*general',
    ]:
        m = re.search(
            pat + r'.*?\{\{Election box end\}\}',
            section_text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group()

    # Strategy 2: find ALL election boxes, return first non-primary
    # Uses .*?}} to match the full opening template (handles multiline "no change" variants)
    blocks = list(re.finditer(
        r'\{\{Election box begin.*?\}\}.*?\{\{Election box end\}\}',
        section_text,
        re.DOTALL | re.IGNORECASE,
    ))

    NON_GENERAL = ("primary", "runoff", "special", "caucus", "recall")
    for block_match in blocks:
        block_text = block_match.group()
        title_m = re.search(r'\|\s*title\s*=\s*([^\n|}{]+)', block_text, re.IGNORECASE)
        if title_m:
            title_text = title_m.group(1).lower()
            if not any(word in title_text for word in NON_GENERAL):
                return block_text
        else:
            # No title field — assume it's the general election
            return block_text

    # Last resort: first block regardless
    return blocks[0].group() if blocks else None


def parse_district(wikitext: str, district: int, max_district: int) -> dict:
    """
    Parse one district's general election result from wikitext.
    Returns a row dict with all result fields.
    """
    row = {
        "district": district,
        "r_candidate": "",
        "r_pct": None,
        "d_candidate": "",
        "d_pct": None,
        "r_incumbent": False,
        "d_incumbent": False,
        "dem_2p_share": None,
        "winner_party": "",
        "contested": False,
        "on_ballot": False,
        "notes": "",
    }

    section = find_district_section(wikitext, district, max_district)
    if not section:
        row["notes"] = "no_section"
        return row

    row["on_ballot"] = True  # section exists → district was on ballot

    # Try election box parser first
    block = find_election_block(section, district)
    parsed_via = "election_box"
    if block:
        candidates = parse_election_box(block)
    else:
        # Fall back to wikitable
        table_result = parse_wikitable_district(section)
        if table_result:
            # Synthesize a candidates list from table parse
            candidates = []
            if table_result.get("r_pct") is not None:
                candidates.append({
                    "party": "R",
                    "candidate": table_result["r_candidate"],
                    "votes": None,
                    "pct": table_result["r_pct"],
                    "incumbent": table_result["r_incumbent"],
                })
            if table_result.get("d_pct") is not None:
                candidates.append({
                    "party": "D",
                    "candidate": table_result["d_candidate"],
                    "votes": None,
                    "pct": table_result["d_pct"],
                    "incumbent": table_result["d_incumbent"],
                })
            parsed_via = "wikitable"
        else:
            row["notes"] = "no_block_or_table"
            return row

    for c in candidates:
        if c["party"] == "R" and row["r_pct"] is None:
            row["r_candidate"] = c["candidate"]
            row["r_pct"] = c["pct"]
            row["r_incumbent"] = c["incumbent"]
        elif c["party"] == "D" and row["d_pct"] is None:
            row["d_candidate"] = c["candidate"]
            row["d_pct"] = c["pct"]
            row["d_incumbent"] = c["incumbent"]

    r_has = row["r_pct"] is not None
    d_has = row["d_pct"] is not None

    if r_has and d_has:
        row["contested"] = True
        row["winner_party"] = "R" if row["r_pct"] > row["d_pct"] else "D"
        total = row["r_pct"] + row["d_pct"]
        row["dem_2p_share"] = round(row["d_pct"] / total, 6) if total > 0 else None
    elif r_has:
        row["winner_party"] = "R"
        row["dem_2p_share"] = 0.0
        row["notes"] = "uncontested_R"
    elif d_has:
        row["winner_party"] = "D"
        row["dem_2p_share"] = 1.0
        row["notes"] = "uncontested_D"
    else:
        row["notes"] = "no_candidates_found"

    if parsed_via == "wikitable" and not row["notes"]:
        row["notes"] = "wikitable_parse"

    return row


# ---------------------------------------------------------------------------
# Bulk "Results by district" wikitable parser (2002-2014 House articles)
# ---------------------------------------------------------------------------

def parse_results_table(wikitext: str) -> dict[int, dict]:
    """
    Parse the "Results by district" wikitable used in older TX legislative
    election Wikipedia articles (2002-2014).

    Table column order (0-indexed after district column):
      0: Dem votes  1: Dem %   2: Rep votes  3: Rep %
      4: Others votes  5: Others %  6: Total votes  7: Total %  8: Result

    Returns dict keyed by district int -> result dict.
    """
    # Strategy 1: find a dedicated "Results by district" section (must start with ==
    # to avoid matching infobox captions).
    table_text = None
    results_section = re.search(
        r'==+\s*(?:Summary of )?[Rr]esults by[^=\n]*district[^=\n]*==+.*?(?=\n==\s|\Z)',
        wikitext,
        re.DOTALL | re.IGNORECASE,
    )
    if results_section:
        table_m = re.search(r'\{\|.*?\|\}', results_section.group(), re.DOTALL)
        if table_m:
            table_text = table_m.group()

    # Strategy 2: directly find the characteristic district-results wikitable by its header.
    # Identified by having "District" + "Democratic" + "Republican" + "Result" columns.
    if not table_text:
        for table_m in re.finditer(r'\{\|.*?\|\}', wikitext, re.DOTALL):
            tbl = table_m.group()
            if (re.search(r'!\s*(?:rowspan[^|]*\|)?\s*District', tbl, re.I)
                    and re.search(r'!\s*(?:colspan[^|]*\|)?\s*Democratic', tbl, re.I)
                    and re.search(r'!\s*(?:colspan[^|]*\|)?\s*Republican', tbl, re.I)
                    and re.search(r'!\s*(?:rowspan[^|]*\|)?\s*Result', tbl, re.I)):
                table_text = tbl
                break

    if not table_text:
        return {}


    # Split into row blocks (each starting with |- possibly followed by shading template)
    row_blocks = re.split(r'\n\|-', table_text)

    results = {}
    for block in row_blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]

        # Find district number from first data cell (wikilink)
        district_line = None
        for line in lines:
            if re.search(r'District\s+\d+', line, re.IGNORECASE):
                district_line = line
                break
        if not district_line:
            continue

        dist_m = re.search(r'District\s+(\d+)', district_line, re.IGNORECASE)
        if not dist_m:
            continue
        district = int(dist_m.group(1))

        # Extract data cells: strip leading pipes and markup
        cells = []
        for line in lines[1:]:  # skip the district-name line
            # Remove leading pipe(s) and alignment markup
            cell = re.sub(r'^\|+\s*(?:align="[^"]*"\s*\|)?\s*', '', line)
            # Strip bold markup, % sign
            cell = re.sub(r"'''", '', cell)
            cell = cell.strip()
            cells.append(cell)

        def parse_pct(val: str) -> float | None:
            val = val.replace('%', '').replace(',', '').strip()
            if val in ('-', '', 'N/A'):
                return None
            try:
                return float(val)
            except ValueError:
                return None

        def parse_votes(val: str) -> int | None:
            val = val.replace(',', '').strip()
            if val in ('-', '', 'N/A'):
                return None
            try:
                return int(float(val))
            except ValueError:
                return None

        # Column positions: [0]=Dem votes, [1]=Dem %, [2]=Rep votes, [3]=Rep %,
        # [4]=Others votes, [5]=Others %, [6]=Total votes, [7]=Total %, [8]=Result
        if len(cells) < 4:
            continue

        d_pct = parse_pct(cells[1]) if len(cells) > 1 else None
        r_pct = parse_pct(cells[3]) if len(cells) > 3 else None
        result_text = cells[-1].lower() if cells else ""

        # Determine winner from result cell or from pcts
        if "republican" in result_text:
            winner = "R"
        elif "democratic" in result_text or "democrat" in result_text:
            winner = "D"
        elif r_pct is not None and d_pct is not None:
            winner = "R" if r_pct > d_pct else "D"
        elif r_pct is not None:
            winner = "R"
        elif d_pct is not None:
            winner = "D"
        else:
            continue

        r_has = r_pct is not None and r_pct > 0
        d_has = d_pct is not None and d_pct > 0
        contested = r_has and d_has

        if contested:
            total = r_pct + d_pct
            dem_2p = round(d_pct / total, 6) if total > 0 else None
            note = ""
        elif r_has:
            dem_2p = 0.0
            note = "uncontested_R"
        elif d_has:
            dem_2p = 1.0
            note = "uncontested_D"
        else:
            continue

        results[district] = {
            "district": district,
            "r_candidate": "",
            "r_pct": r_pct,
            "d_candidate": "",
            "d_pct": d_pct,
            "r_incumbent": False,
            "d_incumbent": False,
            "dem_2p_share": dem_2p,
            "winner_party": winner,
            "contested": contested,
            "on_ballot": True,
            "notes": note or "district_table",
        }

    return results


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------

def collect(chamber: str, year: int) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"Collecting: TX {chamber.title()} {year}")
    print(f"{'='*60}")

    xml_path = download_wikipedia_xml(chamber, year)
    wikitext = load_wikitext(xml_path)
    if not wikitext:
        print(f"  ERROR: Empty wikitext — check XML at {xml_path}")
        return []

    print(f"  Wikitext: {len(wikitext):,} chars")

    max_d = MAX_DISTRICTS[chamber]
    rows = []
    for district in range(1, max_d + 1):
        row = parse_district(wikitext, district, max_d)
        row["year"] = year
        row["chamber"] = chamber.title()
        rows.append(row)

    # If section-based parser found fewer than 50% of districts, fall back to
    # the "Results by district" wikitable (used in 2002-2014 articles)
    on_ballot = sum(1 for r in rows if r["on_ballot"])
    if on_ballot < max_d * 0.5:
        table_results = parse_results_table(wikitext)
        if table_results:
            filled = 0
            for row in rows:
                d = row["district"]
                if not row["on_ballot"] and d in table_results:
                    for k, v in table_results[d].items():
                        if k not in ("year", "chamber"):
                            row[k] = v
                    filled += 1
            print(f"  District table fallback: filled {filled} districts from results table")

    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"  No rows to write for {path.name}")
        return
    fields = ["year", "chamber", "district", "r_candidate", "r_pct",
              "d_candidate", "d_pct", "r_incumbent", "d_incumbent",
              "dem_2p_share", "winner_party", "contested", "on_ballot", "notes"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows → {path.name}")


def summarize(rows: list[dict], label: str):
    on_ballot = [r for r in rows if r["on_ballot"]]
    contested = [r for r in rows if r["contested"]]
    unc_r = [r for r in rows if r["notes"] == "uncontested_R"]
    unc_d = [r for r in rows if r["notes"] == "uncontested_D"]
    no_block = [r for r in rows if "no_block" in r.get("notes", "") or r["notes"] == "no_candidates_found"]
    no_section = [r for r in rows if r["notes"] == "no_section"]
    r_incumbents = sum(1 for r in rows if r.get("r_incumbent"))
    d_incumbents = sum(1 for r in rows if r.get("d_incumbent"))

    print(f"\n{label} summary:")
    print(f"  On ballot:   {len(on_ballot)}")
    print(f"  Contested:   {len(contested)}")
    print(f"  Uncontested R: {len(unc_r)} | Uncontested D: {len(unc_d)}")
    print(f"  Incumbents parsed: R={r_incumbents}, D={d_incumbents}")
    if no_block:
        dists = [r["district"] for r in no_block]
        print(f"  Parse failures ({len(no_block)}): districts {dists}")
    if no_section:
        dists = [r["district"] for r in no_section]
        print(f"  No section ({len(no_section)}): districts {dists}")

    # Incumbent flag coverage warning
    if contested and (r_incumbents + d_incumbents) == 0:
        print(f"  WARNING: No incumbents flagged — older wikitext may not use '(incumbent)' tag.")
        print(f"           Incumbency will need to be inferred in build_phase1_dataset.py.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
CHAMBERS = ["house", "senate"]


def main(years=None, chambers=None):
    years = years or YEARS
    chambers = chambers or CHAMBERS

    all_rows = []
    for year in years:
        for chamber in chambers:
            try:
                rows = collect(chamber, year)
                if rows:
                    out = DATA_HIST / f"tx_{chamber}_results_{year}.csv"
                    write_csv(rows, out)
                    summarize(rows, f"TX {chamber.title()} {year}")
                    all_rows.extend(rows)
            except Exception as exc:
                print(f"  ERROR collecting {chamber} {year}: {exc}")

    print(f"\n{'='*60}")
    print(f"Total rows collected: {len(all_rows)}")
    print(f"Output directory: {DATA_HIST}")

    # Cross-year coverage summary
    print("\nContested race counts by year:")
    for year in years:
        for chamber in chambers:
            year_rows = [r for r in all_rows if r["year"] == year and r["chamber"].lower() == chamber]
            contested = sum(1 for r in year_rows if r["contested"])
            on_ballot = sum(1 for r in year_rows if r["on_ballot"])
            print(f"  {year} {chamber.title():6s}: {on_ballot:3d} on ballot, {contested:3d} contested")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect historical TX legislative election results")
    parser.add_argument("--year", type=int, choices=YEARS, help="Collect only this year")
    parser.add_argument("--chamber", choices=CHAMBERS, help="Collect only this chamber")
    args = parser.parse_args()

    years = [args.year] if args.year else None
    chambers = [args.chamber] if args.chamber else None
    main(years=years, chambers=chambers)
