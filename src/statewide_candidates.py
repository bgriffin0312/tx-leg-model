"""
statewide_candidates.py

Parse the 38 congressional races on PlanC2333 from the cached wikitext into
data/raw/statewide/cd_races_2026.csv: nominees, who is an incumbent, current
ratings, and any general-election district polls
(data/raw/statewide/cd_polls_2026.csv).

Ratings are kept for comparison only; the model does not read them.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from statewide_collect import parse_dates, pct, split_cells, strip_markup  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
WIKI = ROOT / "data" / "raw" / "statewide" / "wikitext"
OUT = ROOT / "data" / "raw" / "statewide"
PAGE = WIKI / "2026_United_States_House_of_Representatives_elections_in_Texas.txt"


def surname(name: str) -> str:
    name = re.sub(r"\s*\((incumbent|politician|American politician)[^)]*\)", "", name)
    name = re.sub(r",?\s+(Jr\.?|Sr\.?|II|III|IV)$", "", name.strip())
    return name.split()[-1] if name.split() else ""


def district_blocks(text: str):
    parts = re.split(r"^==District (\d+)==\s*$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        yield int(parts[i]), parts[i + 1]


def general_section(block: str) -> str:
    m = re.search(r"^===\s*General election\s*===\s*$", block, flags=re.M)
    return block[m.end():] if m else ""


def candidates(gen: str) -> list[tuple[str, str]]:
    box = re.search(r"\{\{Election box begin.*?\{\{Election box end\}\}", gen, flags=re.S)
    if not box:
        return []
    out = []
    for c in re.finditer(r"\{\{Election box candidate(.*?)\}\}\s*(?=\{\{)", box.group(0), flags=re.S):
        # one-line and multi-line templates both occur; protect wikilink pipes first
        body = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"[[\2]]", c.group(1))
        cand = re.search(r"\|\s*candidate\s*=\s*([^|\n]*)", body)
        party = re.search(r"\|\s*party\s*=\s*([^|\n]*)", body)
        if cand and party:
            raw = cand.group(1)
            inc = "incumbent" in raw.lower()
            out.append((strip_markup(raw).replace("(incumbent)", "").strip(), party.group(1).strip(), inc))
    return out


def ratings(gen: str) -> dict:
    sec = re.search(r"====\s*Predictions\s*====(.*?)(?=^====)", gen, flags=re.S | re.M)
    if not sec:
        return {}
    rows = re.split(r"\n\|-[^\n]*", sec.group(1))
    out = {}
    for row in rows:
        src = re.search(r"\[\[([^\]|]+)", row)
        rt = re.search(r"\{\{(?:USRaceRating|US political race rating)\|([^|}]+)(?:\|([^|}]+))?", row)
        if src and rt:
            key = {"The Cook Political Report": "cook", "Inside Elections": "ie", "Sabato's Crystal Ball": "sabato",
                   "Decision Desk HQ": "ddhq", "Silver Bulletin": "silver"}.get(src.group(1))
            if key:
                out[key] = f"{rt.group(1).strip()} {(rt.group(2) or '').strip()}".strip()
    return out


def polls(dist: int, gen: str, r_name: str, d_name: str) -> list[dict]:
    out = []
    for tbl in re.findall(r"\{\|.*?\n\|\}", gen, flags=re.S):
        if "Poll source" not in tbl:
            continue
        cols = []
        for h in re.findall(r"^\s*!(.*)$", tbl, flags=re.M):
            cols += [re.sub(r"^[^|]*=\"[^|]*\"\s*\|", "", x) for x in h.split("!!")]
        cols = [strip_markup(c) for c in cols]
        ri = next((i for i, c in enumerate(cols) if r_name and r_name in c), None)
        di = next((i for i, c in enumerate(cols) if d_name and d_name in c), None)
        if ri is None or di is None:
            continue
        for row in re.split(r"\n\|-[^\n]*", tbl)[1:]:
            c = split_cells(row)
            if len(c) <= max(ri, di) or not re.search(r"\d", c[1]):
                continue
            start, end = parse_dates(c[1])
            rp, dp = pct(c[ri]), pct(c[di])
            if rp is None or dp is None:
                continue
            n = re.search(r"([\d,]{3,})", c[2])
            pop = re.search(r"\b(LV|RV|A|V)\b", c[2])
            src = c[0]
            flag = "R" if re.search(r"\(R\)|NRCC", src) else "D" if re.search(r"\(D\)|DCCC", src) else ""
            out.append({"dist": dist, "pollster": src, "field_end": end, "sample_size": n.group(1) if n else "",
                        "population": pop.group(1) if pop else "", "r_pct": rp, "d_pct": dp, "partisan_flag": flag})
    return out


def main():
    text = PAGE.read_text(encoding="utf-8")
    rows, plist = [], []
    for dist, block in district_blocks(text):
        gen = general_section(block)
        cands = candidates(gen)
        rc = [c for c in cands if "Republican" in c[1]]
        dc = [c for c in cands if "Democratic" in c[1]]
        other = [c[0] for c in cands if "Republican" not in c[1] and "Democratic" not in c[1]]
        rec = {"dist": dist, "r_name": rc[0][0] if rc else "", "d_name": dc[0][0] if dc else "",
               "r_inc": bool(rc and rc[0][2]), "d_inc": bool(dc and dc[0][2]),
               "n_r": len(rc), "n_d": len(dc), "others": "; ".join(other)}
        rec.update(ratings(gen))
        rows.append(rec)
        plist += polls(dist, gen, surname(rec["r_name"]), surname(rec["d_name"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cd_races_2026.csv", index=False)
    pd.DataFrame(plist).to_csv(OUT / "cd_polls_2026.csv", index=False)
    print(f"{len(df)} districts; {df.r_inc.sum()} R inc, {df.d_inc.sum()} D inc; {len(plist)} district polls")
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 60); pd.set_option("display.max_colwidth", 24)
    print(df[["dist", "r_name", "r_inc", "d_name", "d_inc", "n_r", "n_d", "others", "cook", "silver"]].to_string(index=False))
    print(pd.DataFrame(plist).to_string(index=False))


if __name__ == "__main__":
    main()
