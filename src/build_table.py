"""
build_table.py

Sortable HTML table of TX legislative districts showing candidates and
the probability that each seat flips from its current party.

Flip probability:
  R-held seat → P(D wins)   = win_prob_d
  D-held seat → P(R wins)   = 1 − win_prob_d

Only districts on the 2026 ballot are shown (all 150 House + 16 Senate).
Defaults to the current-environment scenario; use the dropdown to switch.

Output: output/model_2026_flip_table.html

USAGE:
  python src/build_table.py
  python src/model.py && python src/build_table.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
DATA   = ROOT / "data" / "processed"
OUTPUT.mkdir(exist_ok=True)


def load_data():
    scen_path = OUTPUT / "model_2026_scenarios.csv"
    dist_path = DATA   / "districts_2026.csv"
    if not scen_path.exists():
        raise FileNotFoundError(f"{scen_path}\nRun: python src/model.py")
    if not dist_path.exists():
        raise FileNotFoundError(f"{dist_path}")

    scenarios = pd.read_csv(scen_path)
    districts = pd.read_csv(dist_path)

    sys.path.insert(0, str(Path(__file__).parent))
    from model_config import GENERIC_BALLOT_TOPLINE_D_2P
    current_env = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)

    # Load scenario summary for seat-count header (optional)
    summary_lookup: dict = {}
    summary_path = OUTPUT / "model_2026_summary.csv"
    if summary_path.exists():
        sdf = pd.read_csv(summary_path)
        for _, row in sdf.iterrows():
            summary_lookup[row["scenario"]] = {
                "expected_house_seats":  round(float(row["expected_house_seats"]), 1),
                "expected_senate_seats": round(float(row["expected_senate_seats"]), 1),
                "house_control_prob":    round(float(row["house_control_prob"]), 4),
                "senate_control_prob":   round(float(row["senate_control_prob"]), 4),
            }

    return scenarios, districts, current_env, summary_lookup


def rating(flip_prob: float) -> str:
    if flip_prob >= 0.75: return "Likely flip"
    if flip_prob >= 0.55: return "Lean flip"
    if flip_prob >= 0.40: return "Competitive"
    if flip_prob >= 0.20: return "Likely holds"
    return "Safe"


def build_rows(scenarios: pd.DataFrame, districts: pd.DataFrame) -> dict:
    """
    Returns {scenario_label: [row_dict, ...]} for all on-ballot districts.
    """
    # Only 2026 ballot districts
    on_ballot = districts[districts["up_in_2026"] == True].copy()

    # Scenario list sorted by env_dial
    scen_meta = (
        scenarios[["env_dial", "scenario"]]
        .drop_duplicates()
        .sort_values("env_dial")
    )
    scenario_labels = scen_meta["scenario"].tolist()

    result = {}
    # Load 2026 candidate overrides (primary winners, runoff status)
    cand_path = DATA / "candidates_2026.csv"
    cand_lookup: dict[tuple, dict] = {}
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path)
        for _, cr in cand_df.iterrows():
            key = (str(cr["chamber"]).strip(), int(cr["district"]))
            r_cand = cr.get("r_candidate")
            d_cand = cr.get("d_candidate")
            cand_lookup[key] = {
                "r": str(r_cand).strip() if pd.notna(r_cand) and str(r_cand).strip() else "",
                "d": str(d_cand).strip() if pd.notna(d_cand) and str(d_cand).strip() else "",
                "r_status": str(cr.get("r_status") or "").strip(),
                "d_status": str(cr.get("d_status") or "").strip(),
            }

    # Load May 26 runoff pairs so we can show both names instead of "Not Yet Determined"
    runoff_path = ROOT / "data" / "raw" / "tx_primary_2026_runoffs.csv"
    runoff_lookup: dict[tuple, str] = {}  # (chamber, district, party) -> "Name1 / Name2"
    if runoff_path.exists():
        ro = pd.read_csv(runoff_path)
        for _, rr in ro.iterrows():
            k = (str(rr["chamber"]).strip(), int(rr["district"]), str(rr["party"]).strip().upper())
            runoff_lookup[k] = f"{str(rr['candidate_1']).strip()} / {str(rr['candidate_2']).strip()}"

    for _, sm in scen_meta.iterrows():
        scen = sm["scenario"]
        scen_rows = scenarios[scenarios["scenario"] == scen].set_index(
            ["chamber", "district"]
        )

        rows = []
        for _, dist in on_ballot.iterrows():
            ch  = dist["chamber"]   # "House" or "Senate"
            num = int(dist["district"])
            inc_party = str(dist.get("incumbent_party") or "").strip().upper()
            inc_name  = str(dist.get("incumbent") or "").strip()
            is_open   = str(dist.get("open_seat") or "").strip().lower() in ("true", "1", "yes")
            _lr = dist.get("last_election_r_candidate")
            _ld = dist.get("last_election_d_candidate")
            last_r = str(_lr).strip() if pd.notna(_lr) and str(_lr).strip().lower() != "nan" else ""
            last_d = str(_ld).strip() if pd.notna(_ld) and str(_ld).strip().lower() != "nan" else ""

            # Look up win prob
            key = (ch, num)
            wp_d = None
            if key in scen_rows.index:
                wp_d = scen_rows.loc[key, "win_prob_d"]
                if hasattr(wp_d, "iloc"):
                    wp_d = wp_d.iloc[0]

            if wp_d is None or pd.isna(wp_d):
                flip_prob = None
            elif inc_party == "R":
                flip_prob = float(wp_d)
            elif inc_party == "D":
                flip_prob = float(1 - wp_d)
            else:
                flip_prob = None

            # Candidate display + uncontested detection
            cand = cand_lookup.get((ch, num))
            uncontested = False
            if cand:
                r_no_opponent = cand["r_status"] == "none_filed" and not cand["r"]
                d_no_opponent = cand["d_status"] == "none_filed" and not cand["d"]
                if r_no_opponent or d_no_opponent:
                    uncontested = True
                # For runoff statuses, look up the May 26 runoff pair and
                # display both names instead of "Not Yet Determined".
                r_runoff = runoff_lookup.get((ch, num, "R"))
                d_runoff = runoff_lookup.get((ch, num, "D"))
                r_name = cand["r"] if cand["r"] else (
                    (f"{r_runoff} (runoff)" if r_runoff else "Not Yet Determined")
                    if cand["r_status"] == "runoff"
                    else "No opponent" if cand["r_status"] == "none_filed"
                    else "—"
                )
                d_name = cand["d"] if cand["d"] else (
                    (f"{d_runoff} (runoff)" if d_runoff else "Not Yet Determined")
                    if cand["d_status"] == "runoff"
                    else "No opponent" if cand["d_status"] == "none_filed"
                    else "—"
                )
                # Mark incumbents with ★ so the table shows incumbency at a glance
                if cand["r_status"] == "incumbent" and cand["r"]:
                    r_name = f"{r_name}★"
                if cand["d_status"] == "incumbent" and cand["d"]:
                    d_name = f"{d_name}★"
            elif is_open:
                r_name = "—"
                d_name = "—"
            elif inc_party == "R":
                r_name = f"{inc_name}★" if inc_name else "(R incumbent)"
                d_name = last_d if last_d else "—"
            elif inc_party == "D":
                r_name = last_r if last_r else "—"
                d_name = f"{inc_name}★" if inc_name else "(D incumbent)"
            else:
                r_name = last_r if last_r else "—"
                d_name = last_d if last_d else "—"

            # Force flip probability to 0 for uncontested races
            if uncontested:
                flip_prob = 0.0

            # Last election margin
            last_contested = dist.get("last_election_contested")
            is_last_contested = str(last_contested).strip().lower() in ("true", "1", "yes")
            if not is_last_contested:
                last_margin = "No challenger"
            else:
                try:
                    r_pct_f = float(dist.get("last_election_r_pct") or "")
                    d_pct_f = float(dist.get("last_election_d_pct") or "")
                    mg = r_pct_f - d_pct_f
                    last_margin = f"R+{mg:.1f}" if mg > 0 else f"D+{-mg:.1f}"
                except (TypeError, ValueError):
                    last_margin = "—"

            flip_pct = round(flip_prob * 100, 1) if flip_prob is not None else None
            rows.append({
                "chamber":     ch,
                "district":    num,
                "dist_label":  f"{'HD' if ch=='House' else 'SD'}-{num:03d}",
                "cur_party":   inc_party,
                "r_candidate": r_name,
                "d_candidate": d_name,
                "flip_pct":    flip_pct,
                "rating":      rating(flip_prob) if flip_prob is not None else "—",
                "last_margin": last_margin,
            })

        # Sort default: flip_pct descending (most competitive first), then district
        rows.sort(key=lambda r: (-(r["flip_pct"] or 0), r["chamber"], r["district"]))
        result[scen] = rows

    return result, scenario_labels


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>2026 TX Legislature — District Flip Probabilities</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 14px;
    color: #1a1a1a;
    background: #f2f4f7;
    padding: 24px 20px 40px;
  }}

  /* ── Page header ─────────────────────────────────────── */
  .page-header {{ max-width: 1400px; margin-bottom: 18px; }}
  h1 {{ font-size: 21px; font-weight: 700; color: #111; margin-bottom: 5px; }}
  .subtitle {{ font-size: 12px; color: #666; line-height: 1.6; }}

  .scenario-bar {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    padding: 8px 14px;
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 7px;
    font-size: 13px;
    color: #333;
    max-width: 1400px;
    flex-wrap: wrap;
  }}
  .scenario-bar .sep {{ color: #ccc; }}
  .scenario-bar b {{ color: #111; }}
  .scenario-bar .label {{ font-weight: 600; color: #555; }}

  /* ── Controls ────────────────────────────────────────── */
  .controls {{
    max-width: 1400px;
    margin-bottom: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: flex-end;
  }}
  .ctrl-group label {{
    display: block;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #999;
    margin-bottom: 4px;
  }}
  select {{
    font-size: 13px;
    padding: 5px 8px;
    border: 1px solid #d0d0d0;
    border-radius: 5px;
    background: #fff;
    color: #1a1a1a;
    cursor: pointer;
  }}
  select:focus {{ outline: 2px solid #1666CB; outline-offset: 1px; border-color: transparent; }}

  .pill-row {{ display: flex; gap: 4px; }}
  .pill {{
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #d0d0d0;
    border-radius: 20px;
    background: #fff;
    color: #555;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.12s, color 0.12s, border-color 0.12s;
  }}
  .pill.active {{ background: #1a1a1a; color: #fff; border-color: #1a1a1a; }}
  .pill:hover:not(.active) {{ background: #f0f0f0; border-color: #bbb; }}

  input[type=text] {{
    font-size: 13px;
    padding: 5px 8px;
    width: 200px;
    border: 1px solid #d0d0d0;
    border-radius: 5px;
    background: #fff;
    color: #1a1a1a;
  }}
  input[type=text]:focus {{ outline: 2px solid #1666CB; outline-offset: 1px; border-color: transparent; }}

  /* ── Table wrapper ───────────────────────────────────── */
  .table-wrap {{
    max-width: 1400px;
    overflow-x: auto;
    border-radius: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.09);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
  }}

  /* ── Header ──────────────────────────────────────────── */
  thead th {{
    background: #1a1a1a;
    color: #d8d8d8;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.055em;
    padding: 10px 13px;
    text-align: left;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 10;
    border-right: 1px solid #2e2e2e;
  }}
  thead th:last-child {{ border-right: none; }}
  thead th:hover {{ background: #2e2e2e; color: #fff; }}
  thead th.sorted-asc::after  {{ content: " ↑"; opacity: 0.8; }}
  thead th.sorted-desc::after {{ content: " ↓"; opacity: 0.8; }}

  /* ── Rows ────────────────────────────────────────────── */
  tbody td {{
    padding: 9px 13px;
    border-bottom: 1px solid #f0f0f0;
    vertical-align: middle;
    white-space: nowrap;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover td {{ background: #f0f5ff; }}
  tbody tr.section-header td {{ background: none; border-bottom: none; }}
  tbody tr.section-header:hover td {{ background: none; }}
  .section-hdr {{ font-size: 15px; font-weight: 700; color: #1a1a1a; padding: 18px 8px 6px 8px !important; letter-spacing: 0.02em; border-bottom: 2px solid #333 !important; }}

  /* ── District label ─────────────────────────────────── */
  .dist-lbl {{ font-weight: 700; font-size: 13px; color: #111; letter-spacing: 0.01em; }}

  /* ── Chamber badge ───────────────────────────────────── */
  .ch-badge {{
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    background: #e8e8e8;
    color: #444;
  }}

  /* ── Party badge ─────────────────────────────────────── */
  .pty {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
  }}
  .pty-r {{ background: #fce8e8; color: #800000; }}
  .pty-d {{ background: #e4eeff; color: #002B84; }}

  /* ── Candidate names ─────────────────────────────────── */
  .cand {{ font-size: 13px; color: #333; }}
  .inc-mark {{ color: #888; font-size: 10px; vertical-align: super; margin-left: 1px; }}

  /* ── Flip probability bar ────────────────────────────── */
  .flip-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .flip-num {{
    font-variant-numeric: tabular-nums;
    font-size: 13px;
    font-weight: 600;
    min-width: 44px;
    text-align: right;
    color: #111;
  }}
  .bar-track {{
    width: 80px;
    height: 8px;
    background: #e4e4e4;
    border-radius: 4px;
    overflow: hidden;
    flex-shrink: 0;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
  }}
  .bar-d {{ background: #1666CB; }}
  .bar-r {{ background: #D40000; }}

  /* ── Rating badge ────────────────────────────────────── */
  .badge {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    color: #fff;
  }}
  .bg-flip4  {{ background: #c0392b; }}
  .bg-flip3  {{ background: #e67e22; }}
  .bg-flip2  {{ background: #d4a017; color: #333; }}
  .bg-flip1  {{ background: #95a5a6; }}
  .bg-flip0  {{ background: #d8dde1; color: #666; }}

  /* ── Last margin ─────────────────────────────────────── */
  .margin {{
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    font-weight: 500;
  }}
  .mg-r {{ color: #800000; }}
  .mg-d {{ color: #002B84; }}
  .mg-n {{ color: #aaa; }}

  /* ── Row count ───────────────────────────────────────── */
  .row-count {{ font-size: 12px; color: #999; margin-top: 8px; max-width: 1400px; }}
</style>
</head>
<body>

<div class="page-header">
  <h1>2026 TX Legislature — District Flip Probabilities</h1>
  <p class="subtitle">
    Flip probability = chance the seat changes party from its current holder. &nbsp;
    Candidates marked ★ are incumbents; &ldquo;(Open)&rdquo; = incumbent not seeking re-election. Opponent names from last contested election.
  </p>
</div>

<div class="scenario-bar" id="scenario-bar">
  <span class="label">Scenario:</span>
  <span id="scen-label">—</span>
  <span class="sep">│</span>
  <span id="scen-house">—</span>
  <span class="sep">│</span>
  <span id="scen-senate">—</span>
</div>

<div class="controls" style="margin-top:14px">
  <div class="ctrl-group">
    <label>Scenario</label>
    <select id="scen-select" onchange="switchScenario(this.value)">
      {scenario_options}
    </select>
  </div>
  <div class="ctrl-group">
    <label>Chamber</label>
    <select id="filter-chamber" onchange="applyFilters()">
      <option value="">Both</option>
      <option value="House">House</option>
      <option value="Senate">Senate</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Holds</label>
    <select id="filter-party" onchange="applyFilters()">
      <option value="">All parties</option>
      <option value="R">R-held</option>
      <option value="D">D-held</option>
    </select>
  </div>
  <div class="ctrl-group">
    <label>Quick filter</label>
    <div class="pill-row">
      <button class="pill active" onclick="setQuick('all',this)">All</button>
      <button class="pill" onclick="setQuick('comp',this)">Competitive</button>
      <button class="pill" onclick="setQuick('flip',this)">Likely flip</button>
    </div>
  </div>
  <div class="ctrl-group">
    <label>Search</label>
    <input type="text" id="filter-search" placeholder="District or candidate…" oninput="applyFilters()">
  </div>
</div>

<div class="table-wrap">
<table id="flip-table">
<thead>
<tr>
  <th onclick="sortTable(0)">District</th>
  <th onclick="sortTable(1)">Ch.</th>
  <th onclick="sortTable(2)">Holds</th>
  <th onclick="sortTable(3)">R Candidate</th>
  <th onclick="sortTable(4)">D Candidate</th>
  <th onclick="sortTable(5)" class="sorted-desc">Flip Prob</th>
  <th onclick="sortTable(6)">Rating</th>
  <th onclick="sortTable(7)">Last Margin</th>
</tr>
</thead>
<tbody id="table-body"></tbody>
</table>
</div>
<div class="row-count" id="row-count"></div>

<script>
const ALL_DATA        = {all_data_json};
const SCENARIO_LABELS = {scenario_labels_json};
const DEFAULT_SCENARIO= {default_scenario_json};
const SUMMARY         = {summary_json};

let currentScenario = DEFAULT_SCENARIO;
let sortCol = 5;
let sortAsc = false;
let quickMode = 'all';

function badgeClass(rating) {{
  if (rating === "Likely flip")  return "bg-flip4";
  if (rating === "Lean flip")    return "bg-flip3";
  if (rating === "Competitive")  return "bg-flip2";
  if (rating === "Likely holds") return "bg-flip1";
  return "bg-flip0";
}}

function candHtml(name) {{
  if (!name || name === "—") return '<span style="color:#bbb">—</span>';
  if (name.endsWith("*")) {{
    return '<span class="cand">' + name.slice(0,-1) + '<sup class="inc-mark">★</sup></span>';
  }}
  return '<span class="cand">' + name + '</span>';
}}

function flipBarHtml(pct, party) {{
  if (pct === null) return '<span style="color:#bbb">—</span>';
  // bar color = direction of flip (R-held→D wins=blue; D-held→R wins=red)
  const barCls = party === "R" ? "bar-d" : "bar-r";
  const w = Math.min(100, Math.round(pct));
  return '<div class="flip-wrap">' +
    '<span class="flip-num">' + pct.toFixed(1) + '%</span>' +
    '<div class="bar-track"><div class="bar-fill ' + barCls + '" style="width:' + w + '%"></div></div>' +
    '</div>';
}}

function marginHtml(mg) {{
  if (!mg || mg === "—") return '<span class="margin mg-n">—</span>';
  const cls = mg.startsWith("R") ? "mg-r" : mg.startsWith("D") ? "mg-d" : "mg-n";
  return '<span class="margin ' + cls + '">' + mg + '</span>';
}}

function updateSummaryBar() {{
  const s = SUMMARY[currentScenario];
  document.getElementById("scen-label").innerHTML = '<b>' + currentScenario + '</b>';
  if (s) {{
    const hCtrl = (s.house_control_prob * 100).toFixed(0);
    const sCtrl = (s.senate_control_prob * 100).toFixed(0);
    document.getElementById("scen-house").innerHTML =
      'House: <b>' + s.expected_house_seats.toFixed(1) + 'D</b> expected &nbsp;(' + hCtrl + '% majority)';
    document.getElementById("scen-senate").innerHTML =
      'Senate: <b>' + s.expected_senate_seats.toFixed(1) + 'D</b> on ballot &nbsp;(' + sCtrl + '% majority)';
  }} else {{
    document.getElementById("scen-house").textContent = '';
    document.getElementById("scen-senate").textContent = '';
  }}
}}

function getSorted() {{
  let rows = [...ALL_DATA[currentScenario]];
  rows.sort((a, b) => {{
    let va, vb;
    switch(sortCol) {{
      case 0: va = a.dist_label;    vb = b.dist_label;    break;
      case 1: va = a.chamber;       vb = b.chamber;       break;
      case 2: va = a.cur_party;     vb = b.cur_party;     break;
      case 3: va = a.r_candidate;   vb = b.r_candidate;   break;
      case 4: va = a.d_candidate;   vb = b.d_candidate;   break;
      case 5: va = a.flip_pct ?? -1; vb = b.flip_pct ?? -1; break;
      case 6: va = a.rating;        vb = b.rating;        break;
      default: va = a.last_margin || ""; vb = b.last_margin || ""; break;
    }}
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  }});
  return rows;
}}

function applyFilters() {{
  const chamberF = document.getElementById("filter-chamber").value;
  const partyF   = document.getElementById("filter-party").value;
  const search   = document.getElementById("filter-search").value.toLowerCase();

  let rows = getSorted().filter(r => {{
    if (chamberF && r.chamber !== chamberF) return false;
    if (partyF   && r.cur_party !== partyF) return false;
    if (search && !(r.dist_label + " " + r.r_candidate + " " + r.d_candidate).toLowerCase().includes(search)) return false;
    if (quickMode === 'comp' && (r.flip_pct === null || r.flip_pct < 20)) return false;
    if (quickMode === 'flip' && (r.flip_pct === null || r.flip_pct < 55)) return false;
    return true;
  }});

  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";

  // Group by chamber, render with section headers
  const chamberFilter = document.getElementById("filter-chamber").value;
  const chambers = chamberFilter ? [chamberFilter] : ["House", "Senate"];
  chambers.forEach(ch => {{
    const chRows = rows.filter(r => r.chamber === ch);
    if (chRows.length === 0) return;
    // Section header row
    const hdr = document.createElement("tr");
    hdr.className = "section-header";
    hdr.innerHTML = '<td colspan="8" class="section-hdr">' +
      (ch === "House" ? "Texas House" : "Texas Senate") +
      ' <span style="font-weight:400;color:#888;">(' + chRows.length + ' districts)</span></td>';
    tbody.appendChild(hdr);
    chRows.forEach(r => {{
      const bc = badgeClass(r.rating);
      const tr = document.createElement("tr");
      tr.innerHTML =
        '<td><span class="dist-lbl">' + r.dist_label + '</span></td>' +
        '<td><span class="ch-badge">' + (r.chamber === "House" ? "H" : "S") + '</span></td>' +
        '<td><span class="pty pty-' + r.cur_party.toLowerCase() + '">' + r.cur_party + '</span></td>' +
        '<td>' + candHtml(r.r_candidate) + '</td>' +
        '<td>' + candHtml(r.d_candidate) + '</td>' +
        '<td>' + flipBarHtml(r.flip_pct, r.cur_party) + '</td>' +
        '<td><span class="badge ' + bc + '">' + r.rating + '</span></td>' +
        '<td>' + marginHtml(r.last_margin) + '</td>';
      tbody.appendChild(tr);
    }});
  }});

  document.getElementById("row-count").textContent =
    "Showing " + rows.length + " of " + ALL_DATA[currentScenario].length + " districts";
}}

function sortTable(col) {{
  sortAsc = (sortCol === col) ? !sortAsc : (col !== 5);
  sortCol = col;
  document.querySelectorAll("thead th").forEach((th, i) => {{
    th.classList.remove("sorted-asc", "sorted-desc");
    if (i === col) th.classList.add(sortAsc ? "sorted-asc" : "sorted-desc");
  }});
  applyFilters();
}}

function switchScenario(scen) {{
  currentScenario = scen;
  updateSummaryBar();
  applyFilters();
}}

function setQuick(mode, btn) {{
  quickMode = mode;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  applyFilters();
}}

updateSummaryBar();
applyFilters();
</script>
</body>
</html>
"""


def main():
    scenarios, districts, current_env, summary_lookup = load_data()

    env_str = f"D+{current_env:.1f}" if current_env >= 0 else f"R+{-current_env:.1f}"
    print(f"Building flip table  (current env: {env_str})")

    row_data, scenario_labels = build_rows(scenarios, districts)

    # Find default scenario (closest to current env)
    scen_meta = (
        scenarios[["env_dial", "scenario"]]
        .drop_duplicates()
        .sort_values("env_dial")
    )
    default_scen = min(
        scen_meta.itertuples(),
        key=lambda r: abs(r.env_dial - current_env)
    ).scenario

    # Build scenario dropdown options
    options_html = "\n      ".join(
        f'<option value="{s}"{" selected" if s == default_scen else ""}>'
        f'{s}{"  ← current" if s == default_scen else ""}'
        f'</option>'
        for s in scenario_labels
    )

    html = HTML_TEMPLATE.format(
        scenario_options=options_html,
        all_data_json=json.dumps(row_data),
        scenario_labels_json=json.dumps(scenario_labels),
        default_scenario_json=json.dumps(default_scen),
        summary_json=json.dumps(summary_lookup),
    )

    out_path = OUTPUT / "model_2026_flip_table.html"
    out_path.write_text(html, encoding="utf-8")

    n = sum(len(v) for v in row_data.values()) // len(row_data)
    print(f"  Districts: {n}  |  Scenarios: {len(scenario_labels)}")
    print(f"  Written → {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
