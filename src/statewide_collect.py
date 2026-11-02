"""
statewide_collect.py

Parse 2026 general-election polls for Texas statewide races out of the cached
wikitext (scripts/statewide_fetch_wikitext.py) into
data/raw/statewide/polls_2026.csv, in the same shape as
data/raw/historical/tx_statewide_polls_2014_2024.csv.

Only head-to-heads between the actual nominees count: a table is kept when its
header names both nominees. Hypothetical match-ups (Cornyn v. Talarico) drop out.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "data" / "raw" / "statewide" / "wikitext"
OUT = ROOT / "data" / "raw" / "statewide" / "polls_2026.csv"

RACES = {
    "governor": ("2026_Texas_gubernatorial_election.txt", "Abbott", "Hinojosa"),
    "lt_governor": ("2026_Texas_lieutenant_gubernatorial_election.txt", "Patrick", "Goodwin"),
    "attorney_general": ("2026_Texas_Attorney_General_election.txt", "Middleton", "Johnson"),
    "comptroller": ("2026_Texas_Comptroller_of_Public_Accounts_election.txt", "Huffines", "Eckhardt"),
    "us_senate": ("2026_United_States_Senate_election_in_Texas.txt", "Paxton", "Talarico"),
    "ag_commissioner": ("2026_Texas_Commissioner_of_Agriculture_election.txt", "Sheets", "Tucker"),
    "land_commissioner": ("2026_Texas_Land_Commissioner_election.txt", "Buckingham", "Flores"),
    "railroad_commissioner": ("2026_Texas_Railroad_Commissioner_election.txt", "French", "Rosenthal"),
}
MONTHS = "January February March April May June July August September October November December".split()


def strip_markup(s: str) -> str:
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"\{\{efn[^{}]*\}\}", "", s, flags=re.I)
    s = re.sub(r"\{\{nbsp\}\}", " ", s)
    s = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", s)
    s = s.replace("'''", "").replace("''", "")
    s = re.sub(r"\{\{(party shading|Party shading)[^{}]*\}\}\s*\|?", "", s)
    s = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def split_cells(block: str) -> list[str]:
    """Cells of one row: one per line starting with |, or || on one line.
    Cell attributes (style="...") sit before a single | and are dropped."""
    block = re.sub(r"<ref.*?</ref>", "", block, flags=re.S)
    block = re.sub(r"<ref[^>]*/>", "", block)
    block = re.sub(r"<!--.*?-->", "", block, flags=re.S)
    cells = []
    for line in block.split("\n"):
        line = line.strip()
        if not line.startswith("|") or line.startswith("|-") or line.startswith("|}"):
            continue
        for part in re.split(r"\|\|", line[1:]):
            part = re.sub(r"^\s*\{\{[Pp]arty shading/[^{}]*\}\}\s*\|", "", part)
            part = re.sub(r"^\s*(style|class|rowspan|colspan|data-sort-value|align)\s*=\s*\"?[^|]*\|", "", part)
            cells.append(strip_markup(part))
    return cells


def tables(text: str):
    # wikitables do not nest on these pages
    for m in re.finditer(r"\{\|.*?\n\|\}", text, flags=re.S):
        yield m.group(0)


def pct(s: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    return float(m.group(1)) if m else None


def parse_dates(s: str):
    """'September 25 – October 5, 2026' / 'October 5–8, 2026' -> (start, end)."""
    s = s.replace("&ndash;", "–").replace("—", "–").replace("-", "–")
    y = re.search(r"(20\d\d)", s)
    year = y.group(1) if y else "2026"
    months = re.findall("|".join(MONTHS), s)
    days = re.findall(r"(?<![\d])(\d{1,2})(?![\d])", s)
    if not months or not days:
        return None, None
    m1, m2 = months[0], months[-1]
    d1, d2 = days[0], days[-1] if len(days) > 1 else days[0]
    fmt = "%B %d %Y"
    return (pd.to_datetime(f"{m1} {d1} {year}", format=fmt).date(),
            pd.to_datetime(f"{m2} {d2} {year}", format=fmt).date())


def parse_race(race: str, fname: str, r_name: str, d_name: str | None) -> list[dict]:
    text = (WIKI / fname).read_text(encoding="utf-8")
    out = []
    for tbl in tables(text):
        head = tbl.split("|-")[0] + (tbl.split("|-")[1] if tbl.count("|-") else "")
        header_lines = [strip_markup(h) for h in re.findall(r"^\s*!(.*)$", tbl, flags=re.M)]
        cols = []
        for h in header_lines:
            for piece in h.split("!!"):
                piece = re.sub(r"^[^|]*=\"[^|]*\"\s*\|", "", piece)
                cols.append(strip_markup(piece))
        if not any("Poll source" in c or "Source of poll" in c for c in cols):
            continue
        if any("aggregat" in c.lower() for c in cols):
            continue
        ri = next((i for i, c in enumerate(cols) if r_name in c), None)
        di = next((i for i, c in enumerate(cols) if d_name and d_name in c), None)
        if ri is None or di is None:
            continue
        # a table with a third named major candidate is a different match-up
        rows = re.split(r"\n\|-[^\n]*", tbl)
        for row in rows[1:]:
            c = split_cells(row)
            if len(c) < max(ri, di) + 1 or not re.search(r"\d", c[1] if len(c) > 1 else ""):
                continue
            start, end = parse_dates(c[1])
            samp = c[2] if len(c) > 2 else ""
            n = re.search(r"([\d,]{3,})", samp)
            pop = re.search(r"\b(LV|RV|A|V)\b", samp)
            rp, dp = pct(c[ri]), pct(c[di])
            if rp is None or dp is None:
                continue
            und = next((pct(c[i]) for i, h in enumerate(cols) if "Undecided" in h and i < len(c)), None)
            source = c[0]
            flag = "R" if re.search(r"\(R\)", source) else "D" if re.search(r"\(D\)", source) else ""
            out.append({"cycle": 2026, "race": race, "pollster": re.sub(r"\s*\((R|D)\)\s*$", "", source).strip(),
                        "sponsor": "", "field_start": start, "field_end": end,
                        "sample_size": float(n.group(1).replace(",", "")) if n else None,
                        "population": pop.group(1) if pop else "",
                        "r_pct": rp, "d_pct": dp,
                        "other_pct": round(100 - rp - dp - (und or 0), 1), "undecided_pct": und,
                        "partisan_flag": flag, "source_url": f"wikipedia:{fname}", "notes": ""})
    return out


def main():
    rows = []
    for race, (fname, r_name, d_name) in RACES.items():
        got = parse_race(race, fname, r_name, d_name)
        print(f"{race:<24} {len(got)} polls")
        rows += got
    df = pd.DataFrame(rows).drop_duplicates(["race", "pollster", "field_end", "population", "r_pct", "d_pct"])
    df = df.sort_values(["race", "field_end"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} -> {OUT}")


if __name__ == "__main__":
    main()
