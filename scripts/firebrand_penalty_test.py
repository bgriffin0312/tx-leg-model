"""
firebrand_penalty_test.py

Do legislators seen as firebrands (R) or rabble-rousers (D) pay a general-
election penalty beyond their district's partisanship? Brennan's question,
2026-10-07: measured by visible, high-profile signals, not roll-call votes.

Outcome: per-race WAR (data/processed/race_war.csv) -- the candidate's vote
share minus what the model expects for a replacement-level candidate of that
party in that district and cycle, signed toward the candidate's party, in pp.
Contested general elections 2018-2024 with an incumbent on the ballot.

Tags (built 2026-10-07 from public sources, see the CSVs' source_url column):
  data/raw/historical/tx_texas_monthly_lists.csv   Texas Monthly Best/Worst lists
  data/raw/historical/tx_firebrand_markers.csv     Freedom Caucus, censures,
      stripped posts, speaker revolts, insurgent PAC backing, quorum-break
      leaders, scandals, viral confrontations
A race is tagged if the incumbent carried the marker in a session BEFORE that
election (session year < election year), i.e. voters could have seen it.

Usage:
  python scripts/firebrand_penalty_test.py
  python scripts/firebrand_penalty_test.py --merge <dir with the agents' CSVs>
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "data" / "raw" / "historical"
TM = HIST / "tx_texas_monthly_lists.csv"
MK = HIST / "tx_firebrand_markers.csv"

RECENCY_YEARS = 4  # last two sessions before the election
FIREBRAND_R = {"freedom_caucus", "censure", "stripped_post", "speaker_revolt",
               "insurgent_backed", "scandal_expelled", "viral_confrontation", "other"}
ROUSER_D = {"quorum_break_leader", "censure", "stripped_post", "viral_confrontation",
            "scandal_expelled", "other"}


def norm(name: str) -> str:
    """Last name + first initial, accent- and suffix-free (matches across sources)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)|\b(jr|sr|ii|iii|iv|dr|the honorable|mr|mrs|ms)\b\.?", " ", s)
    s = re.sub(r"[^a-z ,\-]", " ", s)
    if "," in s:                       # "Last, First"
        last, first = s.split(",", 1)
    else:
        parts = s.split()
        if not parts:
            return ""
        first, last = parts[0], parts[-1]
    last = last.strip().split()[-1] if last.strip() else ""
    first = first.strip()[:1]
    return f"{last}_{first}"


def load_tags() -> pd.DataFrame:
    rows = []
    if TM.exists():
        t = pd.read_csv(TM)
        t = t[t.chamber.astype(str).str.lower().isin(["house", "senate"])]
        for r in t.itertuples():
            cat = str(r.category_norm).lower()
            verb = str(r.category_verbatim).lower()
            # Dishonorable Mention and Cockroach are Worst-family labels (the
            # 2019 Cockroach replaced a third straight Worst for Stickland).
            if "dishonorable" in verb or "cockroach" in verb:
                cat = "worst"
            if cat in ("worst", "furniture", "best", "honorable"):
                rows.append({"key": norm(r.name), "party": str(r.party).upper()[:1],
                             "year": int(r.session_year), "tag": f"tm_{cat}"})
    if MK.exists():
        m = clean_markers(pd.read_csv(MK))
        for r in m.itertuples():
            rows.append({"key": norm(r.name), "party": str(r.party).upper()[:1],
                         "year": int(r.event_year), "tag": str(r.marker_type).lower()})
    return pd.DataFrame(rows)


def clean_markers(m: pd.DataFrame) -> pd.DataFrame:
    """Keep only markers that signal an insurgent firebrand (R) or a visible
    rabble-rouser (D). Coded 2026-10-07 after reading every pre-2024 row:
      - censures: all four pre-2024 censures hit ESTABLISHMENT members (Straus,
        Byron Cook, Murr, Kuempel) -- the opposite signal -- so dropped;
      - speaker_revolt: drop the 2019 calls for Bonnen to resign (caucus-wide,
        not ideological) and the 2025 Cook-vs-Burrows vote (52 mainstream
        members; after the WAR window anyway);
      - scandal_expelled, R "other" caucus-chair resignation, and Seliger's lost
        chairmanship (a moderate clashing with Patrick) are not firebrand signals;
      - D "other": drop the 22-plaintiff group lawsuit rows (collective, not
        individual visibility) unless the row records something individual;
      - weakly sourced rows flagged by the collector: Gutierrez 2023 "viral",
        Cortez 2017 (secondary source only).
    """
    desc = m.description.fillna("").str.lower()
    mt = m.marker_type.str.lower()
    party = m.party.str.upper().str[:1]
    drop = (
        (mt == "censure")
        | ((mt == "speaker_revolt") & (desc.str.contains("bonnen") | (m.event_year >= 2025)))
        | (mt == "scandal_expelled")
        | ((mt == "stripped_post") & (party == "R"))
        | ((mt == "other") & (party == "R") & desc.str.contains("caucus chair"))
        | ((mt == "other") & (party == "D") & desc.str.startswith("named plaintiff")
           & ~desc.str.contains("arrested"))
        | ((m.name == "Roland Gutierrez") & (m.event_year == 2023))
        | ((m.name == "Philip Cortez") & (m.event_year == 2017))
    )
    return m[~drop]


def summarize(d: pd.DataFrame, col: str, label: str):
    print(f"\n--- {label} ---")
    for party in ("R", "D"):
        g = d[d.party == party]
        a, b = g[g[col]], g[~g[col]]
        if len(a) < 2:
            print(f"  {party}: only {len(a)} tagged races")
            continue
        diff = a.war_race.mean() - b.war_race.mean()
        se = np.sqrt(a.war_race.var(ddof=1) / len(a) + b.war_race.var(ddof=1) / len(b))
        print(f"  {party}: tagged {len(a):>3} races ({a.candidate_norm.nunique()} people) WAR {a.war_race.mean():+.2f} | "
              f"untagged {len(b):>3} WAR {b.war_race.mean():+.2f} | difference {diff:+.2f}pp (se {se:.2f}, t {diff / se:+.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", type=Path)
    args = ap.parse_args()
    if args.merge:
        pd.read_csv(args.merge / "texas_monthly_lists.csv").to_csv(TM, index=False)
        pd.read_csv(args.merge / "markers.csv").to_csv(MK, index=False)
        print(f"merged -> {TM.name}, {MK.name}")

    tags = load_tags()
    races = pd.read_csv(ROOT / "data" / "processed" / "race_war.csv")
    races = races[races.is_incumbent == True].copy()
    races["key"] = races.candidate.map(norm)

    def tagged(row, which):
        # Visible before the election, and recent: markers from the last two
        # sessions only (a 2015 vote shapes the 2016 and 2018 races, not 2024).
        t = tags[(tags.key == row.key) & (tags.party == row.party)
                 & (tags.year < row.year) & (tags.year >= row.year - RECENCY_YEARS)]
        return t.tag.isin(which).any()

    groups = {
        "firebrand (any visible marker, R) / rabble-rouser (D)": FIREBRAND_R | ROUSER_D | {"tm_worst"},
        "Texas Monthly WORST list": {"tm_worst"},
        "Texas Freedom Caucus (R)": {"freedom_caucus"},
        "censure / stripped post / speaker revolt": {"censure", "stripped_post", "speaker_revolt"},
        "quorum-break leader (D)": {"quorum_break_leader"},
        "Texas Monthly BEST list (comparison: does visibility for good work pay?)": {"tm_best"},
        "Texas Monthly FURNITURE (low-visibility comparison)": {"tm_furniture"},
    }
    print(f"Incumbent contested races 2018-2024: {len(races)} ({races.candidate_norm.nunique()} people); "
          f"tag rows loaded: {len(tags)}")
    unmatched = sorted(set(tags.key) - set(races.key))
    print(f"Tagged people with no contested general-election race in the WAR window: {len(unmatched)}")
    for label, which in groups.items():
        col = re.sub(r"\W+", "_", label)[:30]
        races[col] = races.apply(lambda r: tagged(r, which), axis=1)
        summarize(races, col, label)
        comp = races[races.competitive_race == True]
        if comp[col].sum() >= 2:
            summarize(comp, col, label + " -- competitive races only (30-70%)")

    # Regression: WAR ~ firebrand tag + year fixed effects, by party; SEs
    # clustered by person (repeat races of the same member are not independent).
    col = re.sub(r"\W+", "_", list(groups)[0])[:30]
    print("\n--- regression, any-marker tag, year FE, person-clustered SE ---")
    for party in ("R", "D"):
        g = races[races.party == party].reset_index(drop=True)
        X = pd.get_dummies(g.year.astype(str), drop_first=True).astype(float)
        X.insert(0, "tag", g[col].astype(float))
        X.insert(0, "const", 1.0)
        y = g.war_race.values
        Xv = X.values
        beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
        e = y - Xv @ beta
        bread = np.linalg.inv(Xv.T @ Xv)
        meat = sum(np.outer(Xv[idx].T @ e[idx], Xv[idx].T @ e[idx])
                   for idx in g.groupby("candidate_norm").indices.values())
        se = np.sqrt(np.diag(bread @ meat @ bread))
        print(f"  {party}: tag effect {beta[1]:+.2f}pp (clustered se {se[1]:.2f}, t {beta[1] / se[1]:+.1f}), "
              f"n={len(g)}, tagged={int(X.tag.sum())}")


if __name__ == "__main__":
    main()
