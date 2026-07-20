"""
compute_war.py

Wins Above Replacement (WAR) for TX legislative candidates.

Methodology (Split Ticket-style):
  WAR_race = actual_dem_2p_share − predicted_dem_2p_share

  predicted_dem_2p_share uses the Phase 1 regression model
  (REGRESSION_COEFFICIENTS in model_config.py), controlling for:
    - District presidential baseline  (dem_pres_2p_baseline)
    - National environment            (generic ballot env_dial)
    - Incumbency                      (dem_incumbent / rep_incumbent)
    - Chamber                         (senate adjustment)
    - Challenger viability / finance  (when available)

  The residual is the candidate-specific contribution — how much
  better or worse they performed than a "replacement-level" generic
  party candidate would in that seat under those conditions.

  WAR is converted to the candidate's perspective:
    D candidate: WAR =  (actual − predicted)   [positive = overperformed]
    R candidate: WAR = −(actual − predicted)   [positive = outperformed R]

  Career WAR uses exponential time-decay:
    2024 (most recent): weight = 1.00
    2022:               weight = 0.60
    2018:               weight = 0.36
  Career WAR = Σ(weight × WAR_race), not averaged — accumulates value.

  Cross-chamber tracking: candidates matched by normalized name
  (lowercase, no punctuation/suffixes). Senators who previously
  served in the House carry their House WAR into career totals.

Data coverage: 2018, 2020, 2022, 2024.
  2020 and 2024 use partial model (no finance data for presidential years).
  National env for 2024: R+3.2; for 2020: D+4.6.

Output:
  data/processed/candidate_war.csv        — per-candidate career WAR
  data/processed/race_war.csv             — per-race WAR (all cycles)
  output/model_2026_war_table.html        — sortable HTML table

USAGE:
  python src/compute_war.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT   = Path(__file__).parent.parent
RAW    = ROOT / "data" / "raw"
HIST   = RAW / "historical"
PROC   = ROOT / "data" / "processed"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from model_config import REGRESSION_COEFFICIENTS, IE_COEFFICIENT, IE_MIN_THRESHOLD

COEFS = REGRESSION_COEFFICIENTS
COEFS["ie_coefficient"] = IE_COEFFICIENT

# Time-decay weights per cycle (geometric: 0.6 per cycle back from most recent)
# 2024 = current (1.00), 2022 = 1 cycle back (0.60), 2020 = 2 back (0.36),
# 2018 = 3 back (0.216). Adding 2020 shifts 2018 from 0.36 to 0.216 — more
# honest decay now that more cycles are available.
CYCLE_WEIGHTS = {2024: 1.00, 2022: 0.60, 2020: 0.36, 2018: 0.216}

# National env (D-R generic ballot margin) per cycle
# 2018: D wave, D+8.6 in the national House popular vote
# 2020: presidential year, D+4.6 (Biden 52.3% 2p national)
# 2022: slight R environment, R+2.8 national
# 2024: presidential year, R+3.2 (Harris 48.4% 2p national)
NATIONAL_ENV = {2018: 8.6, 2020: 4.6, 2022: -2.8, 2024: -3.2}

# Normalise a candidate name for matching across chambers/cycles
_STRIP = re.compile(r"[^a-z ]")
_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|the honorable|hon)\b\.?", re.I)


# Canonical identities for candidates whose ballot name changed across cycles
# (marriage, preferred-name change). Keys and values are normalize_name()
# output (post-hyphen/suffix stripping); applied inside normalize_name so a
# candidate's races aggregate under one identity — the one that matches their
# current name in districts_2026.csv.
NAME_CANONICAL = {
    "christian hayes": "christian manuel",        # HD22 — ran 2022 as Hayes
    "caroline harris": "caroline harris davila",  # HD52 — ran 2022 as Harris
}


def normalize_name(name) -> str:
    if not isinstance(name, str) or not name.strip() or name.strip().lower() in ("nan", "none", "n/a", "—"):
        return ""
    n = name.lower()
    n = n.replace("-", " ")  # hyphenated surnames match their spaced form
    n = _SUFFIXES.sub("", n)
    n = _STRIP.sub("", n)
    n = " ".join(n.split())
    return NAME_CANONICAL.get(n, n)


# ---------------------------------------------------------------------------
# Load raw results for a given cycle
# ---------------------------------------------------------------------------

def load_results(year: int, chamber: str) -> pd.DataFrame:
    """Load candidate-level results for one cycle/chamber. Returns all rows."""
    if year == 2024:
        path = RAW / f"tx_{chamber}_results_2024.csv"
    else:
        path = HIST / f"tx_{chamber}_results_{year}.csv"

    df = pd.read_csv(path)
    df["year"]    = year
    df["chamber"] = chamber.capitalize()

    # Normalise column names across years
    if "r_pct" not in df.columns and "r_votes" in df.columns:
        total = df["r_votes"].fillna(0) + df["d_votes"].fillna(0)
        df["r_pct"] = df["r_votes"] / total * 100
        df["d_pct"] = df["d_votes"] / total * 100

    # dem_2p_share
    if "dem_2p_share" not in df.columns:
        d = df["d_pct"].fillna(0) / 100
        r = df["r_pct"].fillna(0) / 100
        total = d + r
        df["dem_2p_share"] = np.where(total > 0, d / total, np.nan)

    # contested flag
    if "contested" not in df.columns:
        df["contested"] = df["r_pct"].notna() & df["d_pct"].notna()

    return df


# ---------------------------------------------------------------------------
# Build incumbency for each cycle from prior cycle results
# ---------------------------------------------------------------------------

def build_incumbency_2024() -> pd.DataFrame:
    """
    2024 incumbency = 2022 winners (plus known special-election changes).
    Returns DataFrame with columns: chamber, district, dem_incumbent, rep_incumbent.
    """
    return _build_incumbency_from_prior(prior_year=2022)


def build_incumbency_2020() -> pd.DataFrame:
    """
    2020 incumbency = 2018 winners (House: exact; Senate: imperfect because
    half the seats were last contested in 2016, which we don't have data for).
    """
    return _build_incumbency_from_prior(prior_year=2018)


def _build_incumbency_from_prior(prior_year: int) -> pd.DataFrame:
    rows = []
    for ch in ["house", "senate"]:
        df = load_results(prior_year, ch)
        for _, r in df.iterrows():
            if not r.get("on_ballot", r.get("contested", False)):
                continue
            wp = str(r.get("winner_party", "")).strip().upper()
            rows.append({
                "chamber": ch.capitalize(),
                "district": int(r["district"]),
                "dem_incumbent": wp == "D",
                "rep_incumbent": wp == "R",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Merge presidential baseline
# ---------------------------------------------------------------------------

def load_pres_baseline(chamber: str, election_year: int) -> pd.DataFrame:
    """
    dem_pres_2p_baseline per district, using the most appropriate presidential cycle:
      2024 election → 2024 presidential results (exact match)
      2022 election → 2020 presidential results (contemporaneous)
      2018 election → 2020 presidential results (closest available; 2016 not on hand)
    """
    if election_year == 2024:
        path = RAW / f"tx_presidential_{chamber}_2024.csv"
    else:
        path = HIST / f"tx_presidential_{chamber}_2020.csv"
    df = pd.read_csv(path)[["district", "dem_pres_2p_baseline"]].copy()
    df["district"] = df["district"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Load finance viability data per cycle
# ---------------------------------------------------------------------------

def load_ie_data(year: int, chamber: str) -> pd.DataFrame | None:
    """
    Full-cycle IE data (ie_dem_share, ie_total) per district.
    Uses combined PAC+SPAC file when available.
    For 2024, no historical IE file exists — returns None.
    """
    path = HIST / f"tx_ies_combined_{year}.csv"
    if not path.exists():
        path = HIST / f"tx_ies_{year}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["district"] = df["district"].astype(int)
    if "chamber" in df.columns:
        df = df[df["chamber"].str.lower() == chamber.lower()]
    keep = ["district", "ie_dem_share", "ie_total"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# ---------------------------------------------------------------------------
# Build the full race dataset (2018, 2022, 2024)
# ---------------------------------------------------------------------------

def build_race_dataset() -> pd.DataFrame:
    all_races = []

    inc_2024 = build_incumbency_2024()
    inc_2020 = build_incumbency_2020()

    # Phase1 dataset has correct incumbency for 2018 and 2022
    phase1 = pd.read_csv(PROC / "phase1_dataset.csv")[
        ["year", "chamber", "district", "dem_incumbent", "rep_incumbent"]
    ].copy()
    phase1["district"] = phase1["district"].astype(int)
    phase1["chamber"]  = phase1["chamber"].str.capitalize()

    for year in [2018, 2020, 2022, 2024]:
        for ch in ["house", "senate"]:
            results = load_results(year, ch)
            pres    = load_pres_baseline(ch, year)
            results["district"] = results["district"].astype(int)
            results["chamber"]  = ch.capitalize()   # ensure consistent

            # Incumbency
            if year == 2024:
                inc_src = inc_2024[inc_2024["chamber"] == ch.capitalize()][
                    ["district", "dem_incumbent", "rep_incumbent"]
                ]
                results = results.merge(inc_src, on="district", how="left")
            elif year == 2020:
                inc_src = inc_2020[inc_2020["chamber"] == ch.capitalize()][
                    ["district", "dem_incumbent", "rep_incumbent"]
                ]
                results = results.drop(columns=["r_incumbent", "d_incumbent"], errors="ignore")
                results = results.merge(inc_src, on="district", how="left")
            else:
                # Use phase1_dataset incumbency (more reliable than raw results flags for 2018)
                ph_inc = phase1[
                    (phase1["year"] == year) & (phase1["chamber"] == ch.capitalize())
                ][["district", "dem_incumbent", "rep_incumbent"]]
                # Drop any existing incumbent columns from raw results before merging
                results = results.drop(columns=["r_incumbent", "d_incumbent"], errors="ignore")
                results = results.drop(columns=["dem_incumbent", "rep_incumbent"], errors="ignore")
                results = results.merge(ph_inc, on="district", how="left")

            # Presidential baseline
            results = results.merge(pres, on="district", how="left")

            # Full-cycle IE data (used in baseline; exogenous targeting decision, not candidate quality)
            ie = load_ie_data(year, ch)
            if ie is not None:
                results = results.merge(ie, on="district", how="left")
            else:
                results["ie_dem_share"] = np.nan
                results["ie_total"]     = np.nan

            # National env
            results["national_env"]    = NATIONAL_ENV[year]
            results["chamber_senate"]  = 1 if ch == "senate" else 0

            all_races.append(results)

    combined = pd.concat(all_races, ignore_index=True)

    # Keep only contested on-ballot races with a valid 2p share
    combined = combined[
        combined["contested"].fillna(False) &
        combined["dem_2p_share"].notna() &
        (combined["dem_2p_share"] > 0) &
        (combined["dem_2p_share"] < 1)
    ].copy()

    return combined


# ---------------------------------------------------------------------------
# Predict dem_2p_share from regression model
# ---------------------------------------------------------------------------

def predict_dem_share(df: pd.DataFrame) -> pd.Series:
    """
    Baseline prediction using district fundamentals, environment, and incumbency only.
    Finance (challenger_viability_flag, dem_fundraising_share) is intentionally excluded:
    candidate fundraising capacity is partly a quality signal and would create double-counting.
    IEs are included as they represent exogenous targeting decisions by outside groups,
    not candidate-driven effects.
    """
    predicted = pd.Series(COEFS["intercept"], index=df.index, dtype=float)

    predicted += COEFS["dem_pres_2p_baseline"] * df["dem_pres_2p_baseline"].fillna(df["dem_pres_2p_baseline"].mean())
    predicted += COEFS["dem_incumbent"]         * df["dem_incumbent"].fillna(False).astype(float)
    predicted += COEFS["rep_incumbent"]         * df["rep_incumbent"].fillna(False).astype(float)
    predicted += COEFS["chamber_senate"]        * df["chamber_senate"].fillna(0)
    predicted += COEFS["national_env"]          * df["national_env"]

    # IE adjustment — full-cycle weight (1.0) for historical data
    ie_total     = df["ie_total"].fillna(0)
    ie_dem_share = df["ie_dem_share"].fillna(0.5)
    ie_active    = ie_total >= IE_MIN_THRESHOLD
    predicted   += COEFS["ie_coefficient"] * 1.0 * (ie_dem_share - 0.5) * ie_active.astype(float)

    return predicted


# ---------------------------------------------------------------------------
# Compute per-race WAR
# ---------------------------------------------------------------------------

SIGMA_FULL_MODEL = COEFS["sigma"]   # full model residual SE = 0.0785
# Will be replaced with actual no-finance residual SE after computing WAR
SIGMA_WAR_BASELINE = None


def _warp(actual: float, predicted: float, sigma: float) -> float:
    """
    Wins Above Replacement in Probability.
    How much did this candidate shift P(win) vs. a replacement-level candidate?
    Positive = increased party's win probability.
    Uses the WAR baseline σ (no-finance model), not the full-model σ.
    """
    return float(norm.cdf((actual - 0.5) / sigma) - norm.cdf((predicted - 0.5) / sigma))


def compute_race_war(df: pd.DataFrame) -> pd.DataFrame:
    global SIGMA_WAR_BASELINE
    df = df.copy()
    df["predicted_dem_share"] = predict_dem_share(df)
    df["raw_war"]             = df["dem_2p_share"] - df["predicted_dem_share"]

    # Compute actual residual SE from the no-finance baseline
    # This is larger than SIGMA_FULL_MODEL because the WAR baseline
    # excludes finance terms (challenger_viability_flag, dem_fundraising_share)
    SIGMA_WAR_BASELINE = float(np.sqrt((df["raw_war"]**2).mean()))
    print(f"  Full model sigma={SIGMA_FULL_MODEL}, WAR baseline sigma={SIGMA_WAR_BASELINE:.4f}")

    # Build long-form table: one row per candidate per race
    rows = []
    for _, r in df.iterrows():
        year    = int(r["year"])
        chamber = str(r["chamber"]).strip()
        dist    = int(r["district"])
        raw     = float(r["raw_war"])
        pred    = float(r["predicted_dem_share"])
        actual  = float(r["dem_2p_share"])

        r_name = str(r.get("r_candidate") or "").strip()
        d_name = str(r.get("d_candidate") or "").strip()

        d_norm = normalize_name(d_name)
        r_norm = normalize_name(r_name)

        # Race is "competitive" if both predicted AND actual D share are 30-70%.
        is_comp = (0.30 <= pred <= 0.70) and (0.30 <= actual <= 0.70)

        # WARP: from D perspective (positive = D shifted P(win) upward)
        warp_d = _warp(actual, pred, SIGMA_WAR_BASELINE)
        warp_r = -warp_d   # R candidate gets opposite sign

        if d_name and d_norm:
            rows.append({
                "year":             year,
                "chamber":          chamber,
                "district":         dist,
                "candidate":        d_name,
                "party":            "D",
                "candidate_norm":   d_norm,
                "is_incumbent":     bool(r.get("dem_incumbent", False)),
                "war_race":         round(raw * 100, 2),
                "warp_race":        round(warp_d * 100, 2),  # in pp-probability
                "actual_dem_2p":    round(actual * 100, 2),
                "predicted_dem_2p": round(pred * 100, 2),
                "competitive_race": is_comp,
                "cycle_weight":     CYCLE_WEIGHTS.get(year, 1.0),
            })
        if r_name and r_norm:
            rows.append({
                "year":             year,
                "chamber":          chamber,
                "district":         dist,
                "candidate":        r_name,
                "party":            "R",
                "candidate_norm":   r_norm,
                "is_incumbent":     bool(r.get("rep_incumbent", False)),
                "war_race":         round(-raw * 100, 2),
                "warp_race":        round(warp_r * 100, 2),
                "actual_dem_2p":    round(actual * 100, 2),
                "predicted_dem_2p": round(pred * 100, 2),
                "competitive_race": is_comp,
                "cycle_weight":     CYCLE_WEIGHTS.get(year, 1.0),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Career WAR (weighted sum across cycles)
# ---------------------------------------------------------------------------

def compute_career_war(race_war: pd.DataFrame) -> pd.DataFrame:
    """
    Group by normalized name + party, compute career WAR and summary stats.
    Cross-chamber races are included under the same candidate.
    """
    grouped = []
    for (norm_name, party), grp in race_war.groupby(["candidate_norm", "party"]):
        grp = grp.sort_values("year")
        career_war  = (grp["war_race"]  * grp["cycle_weight"]).sum()
        career_warp = (grp["warp_race"] * grp["cycle_weight"]).sum()
        # Weighted average WAR: time-decayed, normalized by sum of weights so
        # 1-cycle candidates aren't penalized vs. multi-cycle vets. Recency
        # still matters because recent cycles have higher weight in both the
        # numerator and (cycle's own) denominator term.
        weight_sum = grp["cycle_weight"].sum()
        weighted_avg_war = (grp["war_race"] * grp["cycle_weight"]).sum() / weight_sum if weight_sum else None
        n_races     = len(grp)
        avg_war     = grp["war_race"].mean()

        # Competitive-only career WAR (both predicted and actual 30-70%)
        comp_grp = grp[grp["competitive_race"] == True]
        career_war_comp = (comp_grp["war_race"] * comp_grp["cycle_weight"]).sum() if len(comp_grp) else None
        n_comp = len(comp_grp)

        # Most recent appearance
        latest = grp.iloc[-1]
        # All display names (pick most common / longest)
        display_name = grp["candidate"].value_counts().index[0]

        # Per-cycle WAR for display
        cycle_war = {int(row["year"]): round(row["war_race"], 1) for _, row in grp.iterrows()}

        # Chambers and districts served
        chambers = grp[["year", "chamber", "district"]].to_dict("records")

        grouped.append({
            "candidate":        display_name,
            "candidate_norm":   norm_name,
            "party":            party,
            "career_war":       round(career_war, 2),
            "weighted_avg_war": round(weighted_avg_war, 2) if weighted_avg_war is not None else None,
            "career_warp":      round(career_warp, 2),
            "career_war_comp":  round(career_war_comp, 2) if career_war_comp is not None else None,
            "n_comp_races":     n_comp,
            "avg_war":          round(avg_war, 2),
            "n_races":          n_races,
            "latest_year":      int(latest["year"]),
            "latest_chamber":   str(latest["chamber"]),
            "latest_district":  int(latest["district"]),
            "is_incumbent_latest": bool(latest["is_incumbent"]),
            "war_2018":         cycle_war.get(2018),
            "war_2020":         cycle_war.get(2020),
            "war_2022":         cycle_war.get(2022),
            "war_2024":         cycle_war.get(2024),
            "chambers_served":  chambers,
        })

    return pd.DataFrame(grouped).sort_values("weighted_avg_war", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# HTML table
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>2026 TX Legislature — Candidate WAR</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; margin: 20px; background: #fff; }}
  h2   {{ font-size: 17px; margin-bottom: 4px; }}
  .subtitle {{ color: #555; font-size: 12px; margin-bottom: 14px; line-height: 1.6; }}
  .controls {{ margin-bottom: 12px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  select, input[type=text] {{ font-size: 13px; padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }}
  label {{ font-weight: bold; }}
  input[type=text] {{ width: 220px; }}

  table  {{ border-collapse: collapse; width: 100%; max-width: 1050px; }}
  th, td {{ border: 1px solid #ddd; padding: 5px 9px; text-align: left; white-space: nowrap; }}
  th     {{ background: #f0f0f0; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  th:hover {{ background: #e0e0e0; }}
  th.sorted-asc::after  {{ content: " ▲"; font-size: 10px; }}
  th.sorted-desc::after {{ content: " ▼"; font-size: 10px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  tr:hover {{ background: #f0f5ff; }}

  .party-r {{ color: #8B0000; font-weight: bold; }}
  .party-d {{ color: #1A5BA8; font-weight: bold; }}

  .war-pos-hi  {{ color: #1A5BA8; font-weight: bold; }}
  .war-pos-md  {{ color: #4488BB; }}
  .war-neg-hi  {{ color: #8B0000; font-weight: bold; }}
  .war-neg-md  {{ color: #BB4444; }}
  .war-neutral {{ color: #888; }}

  .bar-wrap {{ display: flex; align-items: center; gap: 4px; }}
  .bar-pos  {{ height: 9px; border-radius: 2px; background: #1A5BA8; min-width: 2px; }}
  .bar-neg  {{ height: 9px; border-radius: 2px; background: #8B0000; min-width: 2px; }}
</style>
</head>
<body>
<h2>2026 TX Legislature — Candidate WAR (Wins Above Replacement)</h2>
<div class="subtitle">
  <b>WAR</b> measures how much a candidate outperformed their district's fundamentals
  (partisan lean, national environment, incumbency, finance).<br>
  <b>+3.0</b> = won 3pp more than a generic party candidate would in that seat.<br>
  Covers 2018, 2022, and 2024 elections (only cycles with candidate name data).<br>
  Career WAR = time-decay weighted sum (2024 × 1.0, 2022 × 0.6, 2018 × 0.36).
</div>

<div class="controls">
  <div>
    <label>Party:</label>&nbsp;
    <select id="f-party" onchange="applyFilters()">
      <option value="">All</option>
      <option value="R">R</option>
      <option value="D">D</option>
    </select>
  </div>
  <div>
    <label>Chamber:</label>&nbsp;
    <select id="f-chamber" onchange="applyFilters()">
      <option value="">All</option>
      <option value="House">House</option>
      <option value="Senate">Senate</option>
    </select>
  </div>
  <div>
    <label>Cycles:</label>&nbsp;
    <select id="f-cycles" onchange="applyFilters()">
      <option value="0">Any</option>
      <option value="2">2+</option>
      <option value="3">3 (all)</option>
    </select>
  </div>
  <div>
    <input type="text" id="f-search" placeholder="Search candidate…" oninput="applyFilters()">
  </div>
</div>

<table>
<thead>
<tr>
  <th onclick="sortTable(0)">Candidate</th>
  <th onclick="sortTable(1)">Party</th>
  <th onclick="sortTable(2)">Latest Seat</th>
  <th onclick="sortTable(3)" class="sorted-desc">Career WAR</th>
  <th onclick="sortTable(4)" title="Career WARP: how much this candidate shifted their party's win probability (more meaningful than WAR in safe seats)">Career WARP</th>
  <th onclick="sortTable(5)" title="Career WAR in races where both predicted and actual D% were 30-70%">Comp. WAR</th>
  <th onclick="sortTable(6)">2018</th>
  <th onclick="sortTable(7)">2020</th>
  <th onclick="sortTable(8)">2022</th>
  <th onclick="sortTable(9)">2024</th>
  <th onclick="sortTable(10)">Races</th>
  <th onclick="sortTable(11)">Avg WAR</th>
</tr>
</thead>
<tbody id="tbody"></tbody>
</table>

<script>
const DATA = {data_json};

let sortCol = 3, sortAsc = false;

function warClass(v) {{
  if (v === null || v === undefined) return "war-neutral";
  if (v >= 4)  return "war-pos-hi";
  if (v >= 1)  return "war-pos-md";
  if (v <= -4) return "war-neg-hi";
  if (v <= -1) return "war-neg-md";
  return "war-neutral";
}}

function warBar(v) {{
  if (v === null || v === undefined) return "—";
  const cls  = v >= 0 ? "bar-pos" : "bar-neg";
  const width = Math.min(60, Math.round(Math.abs(v) * 8));
  const sign  = v >= 0 ? "+" : "";
  return `<div class="bar-wrap">
    <span class="${{warClass(v)}}">${{sign}}${{v.toFixed(1)}}</span>
    <div class="${{cls}}" style="width:${{width}}px"></div>
  </div>`;
}}

function fmt(v) {{
  if (v === null || v === undefined) return "<span style='color:#ccc'>—</span>";
  const sign = v >= 0 ? "+" : "";
  return `<span class="${{warClass(v)}}">${{sign}}${{v.toFixed(1)}}</span>`;
}}

function seat(r) {{
  const pfx = r.latest_chamber === "House" ? "HD" : "SD";
  return `${{pfx}}-${{String(r.latest_district).padStart(3,"0")}}`;
}}

function renderRow(r) {{
  const pc = r.party === "R" ? "party-r" : "party-d";
  const compWar = r.career_war_comp !== null && r.career_war_comp !== undefined
    ? fmt(r.career_war_comp) + `<span style='color:#aaa;font-size:10px'> (${{r.n_comp_races}})</span>`
    : "<span style='color:#ccc'>—</span>";
  return `<tr data-party="${{r.party}}" data-chamber="${{r.latest_chamber}}"
              data-races="${{r.n_races}}" data-search="${{r.candidate.toLowerCase()}}">
    <td>${{r.candidate}}</td>
    <td class="${{pc}}">${{r.party}}</td>
    <td>${{seat(r)}}</td>
    <td data-val="${{r.career_war}}">${{warBar(r.career_war)}}</td>
    <td data-val="${{r.career_warp}}">${{fmt(r.career_warp)}}</td>
    <td data-val="${{r.career_war_comp ?? -999}}">${{compWar}}</td>
    <td data-val="${{r.war_2018 ?? -999}}">${{fmt(r.war_2018)}}</td>
    <td data-val="${{r.war_2020 ?? -999}}">${{fmt(r.war_2020)}}</td>
    <td data-val="${{r.war_2022 ?? -999}}">${{fmt(r.war_2022)}}</td>
    <td data-val="${{r.war_2024 ?? -999}}">${{fmt(r.war_2024)}}</td>
    <td>${{r.n_races}}</td>
    <td data-val="${{r.avg_war}}">${{fmt(r.avg_war)}}</td>
  </tr>`;
}}

function getColVal(row, col) {{
  const cells = row.querySelectorAll("td");
  const el = cells[col];
  if (!el) return "";
  const dv = el.dataset.val;
  if (dv !== undefined) return parseFloat(dv);
  return el.textContent.trim().toLowerCase();
}}

function applyFilters() {{
  const party   = document.getElementById("f-party").value;
  const chamber = document.getElementById("f-chamber").value;
  const cycles  = parseInt(document.getElementById("f-cycles").value);
  const search  = document.getElementById("f-search").value.toLowerCase();

  let filtered = DATA.filter(r => {{
    if (party   && r.party          !== party)   return false;
    if (chamber && r.latest_chamber !== chamber) return false;
    if (cycles  && r.n_races        <  cycles)   return false;
    if (search  && !r.candidate.toLowerCase().includes(search)) return false;
    return true;
  }});

  filtered.sort((a, b) => {{
    const fields = ["candidate","party","latest_chamber","career_war","career_warp",
                    "career_war_comp","war_2018","war_2020","war_2022","war_2024","n_races","avg_war"];
    let va = a[fields[sortCol]] ?? -999;
    let vb = b[fields[sortCol]] ?? -999;
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ?  1 : -1;
    return 0;
  }});

  document.getElementById("tbody").innerHTML = filtered.map(renderRow).join("");
}}

function sortTable(col) {{
  sortAsc = (sortCol === col) ? !sortAsc : (col < 3);
  sortCol = col;
  document.querySelectorAll("th").forEach((th, i) => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (i === col) th.classList.add(sortAsc ? "sorted-asc" : "sorted-desc");
  }});
  applyFilters();
}}

applyFilters();
</script>
</body>
</html>
"""


def main():
    print("Building WAR dataset...")
    races_raw = build_race_dataset()
    print(f"  Contested races loaded: {len(races_raw)}  "
          f"({races_raw.groupby('year').size().to_dict()})")

    race_war = compute_race_war(races_raw)
    print(f"  Candidate-race rows: {len(race_war)}")

    career = compute_career_war(race_war)
    print(f"  Unique candidates: {len(career)}")

    # Save CSVs
    race_csv = PROC / "race_war.csv"
    career_csv = PROC / "candidate_war.csv"
    race_war.to_csv(race_csv, index=False)
    career_drop = career.drop(columns=["chambers_served"])
    career_drop.to_csv(career_csv, index=False)
    print(f"  Saved → {race_csv.name}, {career_csv.name}")

    # Print top/bottom by career WAR
    print("\n  Top 10 by Career WAR:")
    for _, r in career.head(10).iterrows():
        seat = f"{'HD' if r['latest_chamber']=='House' else 'SD'}-{r['latest_district']:03d}"
        print(f"    {r['candidate']:<28s} ({r['party']}) {seat}  "
              f"Wgt-Avg: {r['weighted_avg_war']:+.2f}  Career: {r['career_war']:+.1f}  "
              f"[{r['war_2018'] or '—':>5}  {r['war_2020'] or '—':>5}  {r['war_2022'] or '—':>5}  {r['war_2024'] or '—':>5}]")

    print("\n  Bottom 10 by Career WAR:")
    for _, r in career.tail(10).iloc[::-1].iterrows():
        seat = f"{'HD' if r['latest_chamber']=='House' else 'SD'}-{r['latest_district']:03d}"
        print(f"    {r['candidate']:<28s} ({r['party']}) {seat}  "
              f"Wgt-Avg: {r['weighted_avg_war']:+.2f}  Career: {r['career_war']:+.1f}  "
              f"[{r['war_2018'] or '—':>5}  {r['war_2020'] or '—':>5}  {r['war_2022'] or '—':>5}  {r['war_2024'] or '—':>5}]")

    # Build HTML
    records = career.to_dict("records")
    # Remove non-serializable sets
    for r in records:
        r.pop("chambers_served", None)
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None

    html = HTML_TEMPLATE.format(data_json=json.dumps(records))
    out_path = OUTPUT / "model_2026_war_table.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  Written → {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
