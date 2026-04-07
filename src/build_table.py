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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], float]:
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

    return scenarios, districts, current_env


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
            last_r    = str(dist.get("last_election_r_candidate") or "").strip()
            last_d    = str(dist.get("last_election_d_candidate") or "").strip()

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

            # Candidate display: incumbent gets *, other side from last election
            if inc_party == "R":
                r_name = f"{inc_name}*" if inc_name else "(R incumbent)"
                d_name = last_d if last_d else "—"
            elif inc_party == "D":
                r_name = last_r if last_r else "—"
                d_name = f"{inc_name}*" if inc_name else "(D incumbent)"
            else:
                r_name = last_r if last_r else "—"
                d_name = last_d if last_d else "—"

            flip_pct = round(flip_prob * 100, 1) if flip_prob is not None else None
            rows.append({
                "chamber":    ch,
                "district":   num,
                "dist_label": f"{'HD' if ch=='House' else 'SD'}-{num:03d}",
                "cur_party":  inc_party,
                "r_candidate": r_name,
                "d_candidate": d_name,
                "flip_pct":   flip_pct,
                "rating":     rating(flip_prob) if flip_prob is not None else "—",
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
<title>2026 TX Legislature — Flip Probability by District</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; margin: 20px; background: #fff; }}
  h2   {{ font-size: 17px; margin-bottom: 4px; }}
  .subtitle {{ color: #555; font-size: 12px; margin-bottom: 14px; }}
  .controls {{ margin-bottom: 12px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  select {{ font-size: 13px; padding: 3px 6px; }}
  label  {{ font-weight: bold; }}
  input[type=text] {{ font-size: 13px; padding: 3px 6px; width: 200px; border: 1px solid #ccc; border-radius: 3px; }}

  table  {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; white-space: nowrap; }}
  th     {{ background: #f0f0f0; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  th:hover {{ background: #e0e0e0; }}
  th.sorted-asc::after  {{ content: " ▲"; font-size: 10px; }}
  th.sorted-desc::after {{ content: " ▼"; font-size: 10px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  tr:hover {{ background: #f5f5f5; }}

  .party-r {{ color: #8B0000; font-weight: bold; }}
  .party-d {{ color: #1A5BA8; font-weight: bold; }}
  .flip-bar-wrap {{ display: flex; align-items: center; gap: 6px; }}
  .flip-bar {{ height: 10px; border-radius: 3px; min-width: 2px; }}
  .flip-pct  {{ font-weight: bold; min-width: 38px; display: inline-block; text-align: right; }}

  .rating-likely  {{ color: #8B0000; font-weight: bold; }}
  .rating-lean    {{ color: #CC5500; }}
  .rating-comp    {{ color: #888800; }}
  .rating-holds   {{ color: #555; }}
  .rating-safe    {{ color: #999; }}

  .filter-row td {{ background: #f8f8f8; }}
</style>
</head>
<body>
<h2>2026 TX Legislature — Flip Probability by District</h2>
<div class="subtitle">
  Flip probability = chance the seat changes party from current incumbent.<br>
  Candidates marked <b>*</b> are incumbents. Opponent names from last contested election.
</div>

<div class="controls">
  <div>
    <label for="scen-select">Scenario:</label>&nbsp;
    <select id="scen-select" onchange="switchScenario(this.value)">
      {scenario_options}
    </select>
  </div>
  <div>
    <label for="filter-chamber">Chamber:</label>&nbsp;
    <select id="filter-chamber" onchange="applyFilters()">
      <option value="">Both</option>
      <option value="House">House</option>
      <option value="Senate">Senate</option>
    </select>
  </div>
  <div>
    <label for="filter-party">Holds:</label>&nbsp;
    <select id="filter-party" onchange="applyFilters()">
      <option value="">All</option>
      <option value="R">R-held</option>
      <option value="D">D-held</option>
    </select>
  </div>
  <div>
    <input type="text" id="filter-search" placeholder="Search candidate or district…" oninput="applyFilters()">
  </div>
</div>

<table id="flip-table">
<thead>
<tr>
  <th onclick="sortTable(0)" data-col="0">District</th>
  <th onclick="sortTable(1)" data-col="1">Chamber</th>
  <th onclick="sortTable(2)" data-col="2">Holds</th>
  <th onclick="sortTable(3)" data-col="3">R Candidate</th>
  <th onclick="sortTable(4)" data-col="4">D Candidate</th>
  <th onclick="sortTable(5)" data-col="5" class="sorted-desc">Flip Prob</th>
  <th onclick="sortTable(6)" data-col="6">Rating</th>
</tr>
</thead>
<tbody id="table-body">
</tbody>
</table>

<script>
const ALL_DATA = {all_data_json};
const SCENARIO_LABELS = {scenario_labels_json};
const DEFAULT_SCENARIO = {default_scenario_json};

let currentScenario = DEFAULT_SCENARIO;
let sortCol = 5;
let sortAsc = false;

function ratingClass(rating) {{
  if (rating === "Likely flip")  return "rating-likely";
  if (rating === "Lean flip")    return "rating-lean";
  if (rating === "Competitive")  return "rating-comp";
  if (rating === "Likely holds") return "rating-holds";
  return "rating-safe";
}}

function flipBar(pct, party) {{
  if (pct === null) return "—";
  const color = party === "R" ? "#1A5BA8" : "#8B0000";
  const width = Math.round(pct * 1.2);
  return `<div class="flip-bar-wrap">
    <span class="flip-pct">${{pct.toFixed(1)}}%</span>
    <div class="flip-bar" style="width:${{width}}px;background:${{color}}"></div>
  </div>`;
}}

function renderRows(rows) {{
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";
  rows.forEach(r => {{
    const partyClass = r.cur_party === "R" ? "party-r" : "party-d";
    const rc = ratingClass(r.rating);
    const tr = document.createElement("tr");
    tr.dataset.chamber = r.chamber;
    tr.dataset.party   = r.cur_party;
    tr.dataset.search  = (r.dist_label + " " + r.r_candidate + " " + r.d_candidate).toLowerCase();
    tr.innerHTML = `
      <td>${{r.dist_label}}</td>
      <td>${{r.chamber}}</td>
      <td class="${{partyClass}}">${{r.cur_party}}</td>
      <td>${{r.r_candidate}}</td>
      <td>${{r.d_candidate}}</td>
      <td data-val="${{r.flip_pct !== null ? r.flip_pct : -1}}">${{flipBar(r.flip_pct, r.cur_party)}}</td>
      <td class="${{rc}}">${{r.rating}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function getFilteredSorted() {{
  let rows = [...ALL_DATA[currentScenario]];

  // Sort
  rows.sort((a, b) => {{
    let va, vb;
    if (sortCol === 5) {{
      va = a.flip_pct !== null ? a.flip_pct : -1;
      vb = b.flip_pct !== null ? b.flip_pct : -1;
    }} else if (sortCol === 0) {{
      va = a.dist_label; vb = b.dist_label;
    }} else if (sortCol === 1) {{
      va = a.chamber; vb = b.chamber;
    }} else if (sortCol === 2) {{
      va = a.cur_party; vb = b.cur_party;
    }} else if (sortCol === 3) {{
      va = a.r_candidate; vb = b.r_candidate;
    }} else if (sortCol === 4) {{
      va = a.d_candidate; vb = b.d_candidate;
    }} else {{
      va = a.rating; vb = b.rating;
    }}
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  }});

  return rows;
}}

function applyFilters() {{
  const chamberFilter = document.getElementById("filter-chamber").value;
  const partyFilter   = document.getElementById("filter-party").value;
  const search        = document.getElementById("filter-search").value.toLowerCase();
  const rows = getFilteredSorted();

  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";

  rows.forEach(r => {{
    if (chamberFilter && r.chamber !== chamberFilter) return;
    if (partyFilter   && r.cur_party !== partyFilter)  return;
    if (search && !(r.dist_label + " " + r.r_candidate + " " + r.d_candidate).toLowerCase().includes(search)) return;

    const partyClass = r.cur_party === "R" ? "party-r" : "party-d";
    const rc = ratingClass(r.rating);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{r.dist_label}}</td>
      <td>${{r.chamber}}</td>
      <td class="${{partyClass}}">${{r.cur_party}}</td>
      <td>${{r.r_candidate}}</td>
      <td>${{r.d_candidate}}</td>
      <td data-val="${{r.flip_pct !== null ? r.flip_pct : -1}}">${{flipBar(r.flip_pct, r.cur_party)}}</td>
      <td class="${{rc}}">${{r.rating}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function sortTable(col) {{
  if (sortCol === col) {{ sortAsc = !sortAsc; }}
  else {{ sortCol = col; sortAsc = col !== 5; }}  // flip prob defaults desc

  // Update header classes
  document.querySelectorAll("th").forEach((th, i) => {{
    th.classList.remove("sorted-asc", "sorted-desc");
    if (i === col) th.classList.add(sortAsc ? "sorted-asc" : "sorted-desc");
  }});

  applyFilters();
}}

function switchScenario(scen) {{
  currentScenario = scen;
  applyFilters();
}}

// Initial render
applyFilters();
</script>
</body>
</html>
"""


def main():
    scenarios, districts, current_env = load_data()

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
    )

    out_path = OUTPUT / "model_2026_flip_table.html"
    out_path.write_text(html, encoding="utf-8")

    n = sum(len(v) for v in row_data.values()) // len(row_data)
    print(f"  Districts: {n}  |  Scenarios: {len(scenario_labels)}")
    print(f"  Written → {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
