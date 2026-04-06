"""
Parse 2024 Texas legislative election results from locally cached Wikipedia XML exports.

XML files are fetched once via Special:Export and stored in data/raw/.
This script parses them locally — no API calls after the initial download.

Output:
  data/raw/tx_house_results_2024.csv
  data/raw/tx_senate_results_2024.csv

Usage:
  python src/collect_2024_results.py
"""

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"

HOUSE_XML = DATA_RAW / "_wiki_house_2024_raw.xml"
SENATE_XML = DATA_RAW / "_wiki_senate_2024_raw.xml"

PARTY_MAP = {
    "republican party (united states)": "R",
    "republican party of texas": "R",
    "democratic party (united states)": "D",
    "democratic party of texas": "D",
}


# ---------------------------------------------------------------------------
# XML → wikitext extraction
# ---------------------------------------------------------------------------

def load_wikitext(xml_path: Path) -> str:
    """Extract wikitext content from a Wikipedia Special:Export XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}
    text_el = root.find(".//mw:revision/mw:text", ns)
    if text_el is None:
        # Try without namespace
        text_el = root.find(".//revision/text")
    return text_el.text if text_el is not None else ""


# ---------------------------------------------------------------------------
# Wikitext parser
# ---------------------------------------------------------------------------

def clean_candidate_name(raw: str) -> str:
    """Strip wikilinks and '(incumbent)' from a candidate name field."""
    # [[Article|Display]] -> Display
    name = re.sub(r'\[\[[^\]]*\|([^\]]+)\]\]', r'\1', raw)
    # [[Article]] -> Article
    name = re.sub(r'\[\[([^\]]+)\]\]', r'\1', name)
    # Truncated wikilink (no closing]])
    name = re.sub(r'\[\[.*', '', name)
    # Remove (incumbent)
    name = re.sub(r'\s*\(incumbent\)', '', name, flags=re.IGNORECASE)
    return name.strip()


def parse_election_box(block: str) -> list[dict]:
    """
    Parse candidates from one {{Election box begin}}...{{Election box end}} block.
    Handles both multiline (2024 style) and inline (2022 style) wikitext.
    Returns list of {party, candidate, votes, pct}.
    """
    candidates = []
    # Split on each candidate entry template (winning or non-winning)
    parts = re.split(r'\{\{Election box (?:winning )?candidate[^\}]*?(?=\||\}\})', block)
    for part in parts[1:]:
        # Normalize: inline pipes become newlines for uniform parsing
        part_norm = part.replace("|", "\n|")
        party_m = re.search(r'\|\s*party\s*=\s*([^\n|]+)', part_norm)
        cand_m = re.search(r'\|\s*candidate\s*=\s*([^\n|}\]]+(?:\[\[[^\]]*\][^\n|}]*)?[^\n|}]*)', part_norm)
        votes_m = re.search(r'\|\s*votes\s*=\s*([\d,]+)', part_norm)
        pct_m = re.search(r'\|\s*percentage\s*=\s*([\d.]+)', part_norm)

        if not (party_m and pct_m):
            continue

        party_raw = party_m.group(1).strip().lower()
        party = PARTY_MAP.get(party_raw, "?")

        cand = clean_candidate_name(cand_m.group(1)) if cand_m else ""
        votes = int(votes_m.group(1).replace(",", "")) if votes_m else None
        pct = float(pct_m.group(1))

        candidates.append({"party": party, "candidate": cand, "votes": votes, "pct": pct})

    return candidates


def find_election_block(wikitext: str, district: int) -> str | None:
    """
    Find the general/main election box for a given district number.
    Tries 'District N general election', then 'District N election', then last box.
    """
    patterns = [
        rf'title\s*=\s*District\s*{district}\s*general election',
        rf'title\s*=\s*District\s*{district}\s*election',
    ]
    for pat in patterns:
        m = re.search(pat + r'.*?\{\{Election box end\}\}', wikitext, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group()

    # Fallback: last election box before the next district section or end
    # Narrow wikitext to just this district's section
    section_pat = rf'===?\s*District\s*{district}\s*===?'
    next_section_pat = rf'===?\s*District\s*{district + 1}\s*===?'
    sec_m = re.search(section_pat, wikitext, re.IGNORECASE)
    if not sec_m:
        return None
    start = sec_m.start()
    next_m = re.search(next_section_pat, wikitext[start:], re.IGNORECASE)
    section_text = wikitext[start: start + next_m.start()] if next_m else wikitext[start:]

    blocks = list(re.finditer(
        r'\{\{Election box begin[^\}]*\}.*?\{\{Election box end\}\}',
        section_text, re.DOTALL | re.IGNORECASE,
    ))
    return blocks[-1].group() if blocks else None


def parse_district(wikitext: str, district: int) -> dict:
    row = {
        "district": district,
        "r_candidate": "",
        "r_votes": None,
        "r_pct": None,
        "d_candidate": "",
        "d_votes": None,
        "d_pct": None,
        "winner_party": "",
        "contested": False,
        "notes": "",
    }

    block = find_election_block(wikitext, district)
    if not block:
        row["notes"] = "no_block"
        return row

    candidates = parse_election_box(block)
    for c in candidates:
        if c["party"] == "R" and not row["r_candidate"] and row["r_votes"] is None:
            row["r_candidate"] = c["candidate"]
            row["r_votes"] = c["votes"]
            row["r_pct"] = c["pct"]
        elif c["party"] == "D" and not row["d_candidate"] and row["d_votes"] is None:
            row["d_candidate"] = c["candidate"]
            row["d_votes"] = c["votes"]
            row["d_pct"] = c["pct"]

    r_has = row["r_pct"] is not None
    d_has = row["d_pct"] is not None
    if r_has and d_has:
        row["contested"] = True
        # Prefer votes for winner; fall back to pct
        if row["r_votes"] is not None and row["d_votes"] is not None:
            row["winner_party"] = "R" if row["r_votes"] > row["d_votes"] else "D"
        else:
            row["winner_party"] = "R" if row["r_pct"] > row["d_pct"] else "D"
    elif r_has:
        row["winner_party"] = "R"
        row["notes"] = "uncontested_R"
    elif d_has:
        row["winner_party"] = "D"
        row["notes"] = "uncontested_D"
    else:
        row["notes"] = "no_candidates_found"

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect(xml_path: Path, max_district: int, label: str) -> list[dict]:
    print(f"Loading {xml_path.name}...")
    wikitext = load_wikitext(xml_path)
    print(f"  Wikitext length: {len(wikitext):,} chars")

    results = []
    for district in range(1, max_district + 1):
        row = parse_district(wikitext, district)
        results.append(row)

    return results


def write_csv(rows: list[dict], path: Path):
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path.name}")


def summarize(rows: list[dict], label: str):
    contested = sum(1 for r in rows if r["contested"])
    unc_r = sum(1 for r in rows if r["notes"] == "uncontested_R")
    unc_d = sum(1 for r in rows if r["notes"] == "uncontested_D")
    problems = [r for r in rows if r["notes"] in ("no_block", "no_candidates_found")]
    blank_r = [r for r in rows if r["contested"] and not r["r_candidate"]]
    blank_d = [r for r in rows if r["contested"] and not r["d_candidate"]]
    print(f"\n{label}: {len(rows)} districts | {contested} contested | "
          f"{unc_r} uncontested-R | {unc_d} uncontested-D")
    if problems:
        print(f"  Problems ({len(problems)}): districts {[r['district'] for r in problems]}")
    if blank_r:
        print(f"  Blank R names ({len(blank_r)}): {[r['district'] for r in blank_r]}")
    if blank_d:
        print(f"  Blank D names ({len(blank_d)}): {[r['district'] for r in blank_d]}")


SENATE_2022_XML = DATA_RAW / "_wiki_senate_2022_raw.xml"


if __name__ == "__main__":
    house = collect(HOUSE_XML, max_district=150, label="House")
    write_csv(house, DATA_RAW / "tx_house_results_2024.csv")
    summarize(house, "House")

    senate_24 = collect(SENATE_XML, max_district=31, label="Senate 2024")
    write_csv(senate_24, DATA_RAW / "tx_senate_results_2024.csv")
    summarize(senate_24, "Senate 2024")

    if SENATE_2022_XML.exists():
        senate_22 = collect(SENATE_2022_XML, max_district=31, label="Senate 2022")
        write_csv(senate_22, DATA_RAW / "tx_senate_2022_results.csv")
        summarize(senate_22, "Senate 2022")
