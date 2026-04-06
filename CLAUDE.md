# Claude Code — Project Notes

## Purpose
Race-by-race model of Texas legislative elections (TX House + TX Senate) in 2026. The core mechanic is a configurable national environment dial (e.g. D+8, R+3) that shifts all races by a calibrated amount, layered on top of district-level baselines.

## Model design goals
- Each race has a baseline: partisan lean, incumbent status, open seat, fundraising, candidate quality
- A single "national environment" input (generic ballot equivalent) shifts all races
- Upballot effects from statewide TX races (U.S. Senate, Governor if applicable) can be toggled
- Output: per-race win probability, expected seat totals, chamber control probability under each scenario

## Key conventions
- Raw source data → `data/raw/` (gitignored)
- Cleaned/model-ready data → `data/processed/` (gitignored)
- Model run outputs → `output/` (gitignored)
- Reusable logic → `src/`
- Exploratory notebooks → `notebooks/`

---

## Current state (as of April 2026)

### Data collected — all in `data/raw/`

| File | Contents | Source |
|------|----------|--------|
| `tx_house_members_89th.csv` | All 150 TX House members: district, name, party | Wikipedia (89th Leg) |
| `tx_senate_members_89th.csv` | All 31 TX Senate members: district, name, party, election year, notes | Wikipedia (89th Leg) |
| `tx_house_results_2024.csv` | 2024 general election results for all 150 House districts: R/D candidate, votes, pct, contested flag | Wikipedia (parsed from XML export) |
| `tx_senate_results_2024.csv` | 2024 results for the 15 Senate seats that were on the ballot in 2024 (the 2028-cycle seats) | Wikipedia |
| `tx_senate_2022_results.csv` | 2022 results for all 31 Senate seats (all were on ballot after redistricting) | Wikipedia |
| `_wiki_house_2024_raw.xml` | Cached Wikipedia XML export — source for House results | Special:Export |
| `_wiki_senate_2024_raw.xml` | Cached Wikipedia XML export — source for 2024 Senate results | Special:Export |
| `_wiki_senate_2022_raw.xml` | Cached Wikipedia XML export — source for 2022 Senate results | Special:Export |

### Master district table — `data/processed/districts_2026.csv`
181 rows (150 House + 31 Senate). Columns:
- `chamber`, `district`, `incumbent`, `incumbent_party`
- `last_election_year` — which election the results below come from
- `last_election_r_candidate`, `last_election_r_pct`
- `last_election_d_candidate`, `last_election_d_pct`
- `last_election_contested`, `last_election_winner_party`, `last_election_notes`
- `up_in_2026` — True for all 150 House + 16 Senate seats
- `open_seat` — blank (to be filled as candidates declare)
- `notes_2026`

**Important:** For the 16 Senate seats up in 2026, `last_election_year = 2022` (the 2022 results are used as their baseline, since those are the most recent results for those specific seats). The other 15 Senate seats use 2024 results but are not up until 2028.

### Current composition (89th Legislature)
- **TX House:** 88R, 62D (Republicans gained 2 seats in 2024, picking up Rio Grande Valley seats)
- **TX Senate:** 20R, 11D (SD9 flipped D via special election in 2025; SD4 is vacant)

### Senate 2026 seats (16 districts)
Districts: 1, 2, 3, 4, 5, 9, 11, 13, 18, 19, 21, 22, 24, 26, 28, 31
- SD4 is **vacant** (special election May 2, 2026 + general Nov 2026)
- SD9: Taylor Rehmet (D) won special election in 2025, replacing Republican Kelly Hancock

### Data quality notes
- 18 House districts have blank R candidate names (wikilink formatting issue) — pcts are correct; names can be backfilled from `tx_house_members_89th.csv`
- 5 House districts have blank D candidate names — same issue
- SD13 (2022) has no result parsed — needs manual lookup
- SD1, SD3, SD28 (2022) have blank incumbent names but correct pcts

### Source scripts — `src/`
| Script | Purpose |
|--------|---------|
| `collect_2024_results.py` | Parses Wikipedia XML exports → writes House/Senate results CSVs. Run this to refresh raw data after re-downloading XMLs. No API calls needed (reads local files). |
| `build_district_table.py` | Merges members + results → writes `data/processed/districts_2026.csv`. Run after `collect_2024_results.py`. |

---

## What's next

### Immediate data gaps
1. **Presidential 2024 results by legislative district** — the most important missing piece. Trump won 96/150 House districts and 21/31 Senate districts in 2024. District-level pcts would give a clean partisan baseline independent of candidate quality. Best source: Texas Legislative Council WRM district reports (`wrm.capitol.texas.gov/fyiwebdocs/PDF/house/dist[N]/r8.pdf`) — PDFs, require `pdfplumber` to parse. Or find a pre-compiled table.
2. **SD13 2022 result** — one missing Senate data point, can be looked up manually (Borris Miles D ran uncontested or near-uncontested in 2022).

### Model to build
The core model in `src/model.py` (not yet started) should:
1. Load `data/processed/districts_2026.csv`
2. Compute a **partisan lean** for each district (using presidential pct or last legislative result as proxy)
3. Accept a **national environment** parameter (e.g. `env = -3` for R+3, `env = +8` for D+8)
4. Apply a **uniform swing** from national environment to each district's baseline
5. Add optional **incumbency advantage** offset
6. Convert to **win probability** using a logistic/normal CDF
7. Output: per-race win prob, seat totals, chamber control probability
8. Run scenarios: e.g. R+3, even, D+3, D+6, D+8

### Key modeling questions to resolve
- What's the right baseline metric: last legislative result, presidential result, or a blend?
- How much does incumbency advantage shift the baseline? (Typical estimate: +5 to +7 pts)
- What's the right swing sensitivity? (How many points does a D+1 national environment shift a district?)
- How to handle uncontested districts in the model (they won't be competitive regardless of environment)
- Senate: 2022 results are 4 years old — how to weight/discount them vs. presidential lean?
