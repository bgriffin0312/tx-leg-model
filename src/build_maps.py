"""
build_maps.py

Build interactive HTML choropleth maps of TX legislative district win probabilities
for all model scenarios. Scenario toggle (R+3 → D+8) defaults to current environment.

Outputs:
  output/model_2026_map_house.html    — TX House (150 districts)
  output/model_2026_map_senate.html   — TX Senate (16 on ballot in 2026)

  For flip analysis, see output/model_2026_flip_table.html (build_table.py)

Data sources:
  output/model_2026_scenarios.csv     — from model.py --no-save (or default run)
  Census TIGER Cartographic Boundary  — downloaded and cached in data/raw/geo/

USAGE:
  python src/build_maps.py                          # build both chambers + flip map
  python src/build_maps.py --chamber house          # house only
  python src/build_maps.py --chamber senate         # senate only
  python src/build_maps.py --no-flip               # skip flip map
  python src/model.py && python src/build_maps.py  # full pipeline
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).parent.parent
DATA_GEO = ROOT / "data" / "raw" / "geo"
OUTPUT   = ROOT / "output"
DATA_GEO.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(exist_ok=True)

# Census Cartographic Boundary Files — 500k scale, TX (FIPS 48)
TIGER_URLS = {
    "house":  "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_48_sldl_500k.zip",
    "senate": "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_48_sldu_500k.zip",
}

# Field in shapefile that holds the district number (zero-padded string)
DIST_FIELD = {
    "house":  "SLDLST",
    "senate": "SLDUST",
}

# Color scale: Wikipedia election convention — red (R) → gray (tossup) → blue (D)
# Category boundaries: Safe=0/10%, Likely=10/25%, Lean=25/40%, Tossup=40/60%
COLORSCALE = [
    [0.00, "#800000"],   # Safe R — dark maroon
    [0.10, "#AA0000"],   # Likely R — medium red
    [0.25, "#D40000"],   # Lean R — bright red
    [0.40, "#F0A09A"],   # near tossup R — pale salmon
    [0.50, "#E8E8E8"],   # Tossup — neutral gray
    [0.60, "#9ABBE0"],   # near tossup D — pale blue
    [0.75, "#1666CB"],   # Lean D — medium blue
    [0.90, "#0645B4"],   # Likely D — bright blue
    [1.00, "#002B84"],   # Safe D — dark navy
]

# Rating bands for hover text
def win_prob_label(p: float) -> str:
    if p >= 0.95: return "Safe D"
    if p >= 0.75: return "Likely D"
    if p >= 0.55: return "Lean D"
    if p >= 0.45: return "Toss-up"
    if p >= 0.25: return "Lean R"
    if p >= 0.05: return "Likely R"
    return "Safe R"


# ---------------------------------------------------------------------------
# Load scenario seat-count summary
# ---------------------------------------------------------------------------

def load_summary_lookup() -> dict:
    """Return {scenario_label: {expected_house_seats, ...}} from model_2026_summary.csv."""
    path = OUTPUT / "model_2026_summary.csv"
    if not path.exists():
        return {}
    import pandas as _pd
    df = _pd.read_csv(path)
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["scenario"]] = {
            "expected_house_seats":  float(row["expected_house_seats"]),
            "expected_senate_seats": float(row["expected_senate_seats"]),
            "house_control_prob":    float(row["house_control_prob"]),
            "senate_control_prob":   float(row["senate_control_prob"]),
        }
    return lookup


# ---------------------------------------------------------------------------
# Download and cache shapefiles
# ---------------------------------------------------------------------------

def get_shapefile(chamber: str) -> gpd.GeoDataFrame:
    """Download Census TIGER shapefile for TX legislative districts, cache locally."""
    cache_path = DATA_GEO / f"tx_{chamber}_districts.gpkg"

    if cache_path.exists():
        print(f"  Shapefile cache hit: {cache_path.name}")
        return gpd.read_file(cache_path)

    url = TIGER_URLS[chamber]
    print(f"  Downloading Census TIGER {chamber} boundaries from:\n    {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Read shapefile from in-memory ZIP
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        shp_names = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_names:
            raise RuntimeError(f"No .shp file found in {url}")
        print(f"  Shapefile: {shp_names[0]}")

        # Extract all relevant files to a temp dir and read
        extract_dir = DATA_GEO / f"_tmp_{chamber}"
        extract_dir.mkdir(exist_ok=True)
        zf.extractall(extract_dir)

    shp_path = next(extract_dir.glob("*.shp"))
    gdf = gpd.read_file(shp_path)

    # Project to WGS84 (lat/lon) for Plotly
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Cache as GeoPackage
    gdf.to_file(cache_path, driver="GPKG")
    print(f"  Cached → {cache_path.name}  ({len(gdf)} features)")
    return gdf


# ---------------------------------------------------------------------------
# Load model scenario outputs
# ---------------------------------------------------------------------------

def load_scenarios(chamber: str) -> pd.DataFrame:
    """
    Load win probabilities per district per scenario.
    Returns wide-format DataFrame: rows = districts, cols = one per scenario.
    """
    path = OUTPUT / "model_2026_scenarios.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\nRun: python src/model.py"
        )
    df = pd.read_csv(path)
    df = df[df["chamber"].str.lower() == chamber].copy()
    if df.empty:
        raise ValueError(f"No {chamber} rows in model_2026_scenarios.csv")

    # Build (env_dial → scenario_label) mapping, sorted by env_dial
    scenarios = (
        df[["env_dial", "scenario"]]
        .drop_duplicates()
        .sort_values("env_dial")
    )
    scenario_list = list(zip(scenarios["env_dial"], scenarios["scenario"]))

    # Pivot: district → wide (use district only as index to avoid NaN key issues)
    # Then re-attach metadata from the first scenario
    meta = (
        df[df["scenario"] == scenario_list[0][1]]
        [["district", "incumbent", "incumbent_party", "dem_pres_2p_baseline"]]
        .drop_duplicates("district")
    )
    wide = df.pivot_table(
        index="district",
        columns="scenario",
        values="win_prob_d",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.merge(meta, on="district", how="left")

    # Keep scenario columns in sorted order
    scenario_cols = [s for _, s in scenario_list if s in wide.columns]

    # Override win_prob_d for uncontested races (no opponent filed)
    cand_path = ROOT / "data" / "processed" / "candidates_2026.csv"
    if cand_path.exists():
        cand_df = pd.read_csv(cand_path)
        ch_label = "House" if chamber == "house" else "Senate"
        cand_ch = cand_df[cand_df["chamber"] == ch_label]
        for _, cr in cand_ch.iterrows():
            dist = int(cr["district"])
            r_status = str(cr.get("r_status") or "").strip()
            d_status = str(cr.get("d_status") or "").strip()
            r_cand = str(cr.get("r_candidate") or "").strip() if pd.notna(cr.get("r_candidate")) else ""
            d_cand = str(cr.get("d_candidate") or "").strip() if pd.notna(cr.get("d_candidate")) else ""
            if r_status == "none_filed" and not r_cand:
                # D runs unopposed → win_prob_d = 1.0
                mask = wide["district"] == dist
                for sc in scenario_cols:
                    wide.loc[mask, sc] = 1.0
            elif d_status == "none_filed" and not d_cand:
                # R runs unopposed → win_prob_d = 0.0
                mask = wide["district"] == dist
                for sc in scenario_cols:
                    wide.loc[mask, sc] = 0.0

    return wide, scenario_list, scenario_cols


# ---------------------------------------------------------------------------
# Build Plotly choropleth figure
# ---------------------------------------------------------------------------

def build_figure(
    gdf: gpd.GeoDataFrame,
    wide: pd.DataFrame,
    scenario_list: list[tuple],
    scenario_cols: list[str],
    chamber: str,
    current_env: float,
    summary_lookup: dict | None = None,
) -> go.Figure:
    """
    Build a Plotly choropleth figure with one trace per scenario and
    dropdown buttons to toggle between them.
    """
    dist_field = DIST_FIELD[chamber]

    # Parse district numbers from shapefile (zero-padded strings → int)
    gdf = gdf.copy()
    gdf["district_int"] = gdf[dist_field].astype(int)

    # Merge model data onto GDF
    merge_cols = ["district", "incumbent", "incumbent_party", "dem_pres_2p_baseline"] + scenario_cols
    gdf = gdf.merge(
        wide[merge_cols],
        left_on="district_int",
        right_on="district",
        how="left",
    )

    # Simplify geometry slightly to reduce file size (tolerance in degrees ≈ 0.01° ≈ 1km)
    gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)

    # Build GeoJSON
    geojson = json.loads(gdf.geometry.to_json())
    # Attach district ID to each feature for featureidkey
    for i, feat in enumerate(geojson["features"]):
        feat["id"] = str(gdf.iloc[i]["district_int"])

    # Determine default scenario (closest to current environment)
    default_env  = min(scenario_list, key=lambda x: abs(x[0] - current_env))
    default_scen = default_env[1]
    default_idx  = scenario_cols.index(default_scen) if default_scen in scenario_cols else 0

    # Hover text template (shared across scenarios) — accepts a gdf subset so
    # active-district traces can pass their filtered rows.
    def make_hover(col: str, hover_gdf: gpd.GeoDataFrame | None = None) -> list[str]:
        rows_gdf = hover_gdf if hover_gdf is not None else gdf
        texts = []
        for _, row in rows_gdf.iterrows():
            d   = row.get("district_int", "?")
            inc = row.get("incumbent", "?") or "?"
            inc_party = row.get("incumbent_party", "?") or "?"
            wp  = row.get(col)
            pres = row.get("dem_pres_2p_baseline")
            if pd.isna(wp):
                label = "No data"
                wp_str = "—"
            else:
                label  = win_prob_label(wp)
                wp_str = f"{wp*100:.1f}%"
            pres_str = f"{float(pres)*100:.1f}%" if pd.notna(pres) else "?"
            ch_prefix = "HD" if chamber == "house" else "SD"
            texts.append(
                f"<b>{ch_prefix}-{d:03d}</b><br>"
                f"Incumbent: {inc} ({inc_party})<br>"
                f"Pres baseline: {pres_str}<br>"
                f"D win prob: <b>{wp_str}</b> — {label}"
            )
        return texts

    # Identify districts with no model data (Senate seats not up in 2026).
    # They get a distinct fill so they don't blend with the gray-toss-up color.
    has_data_mask = ~gdf[scenario_cols[0]].isna() if scenario_cols else None

    # Build a background "not on ballot" trace once — visible in every scenario.
    # Uses a tan/beige outside the political colorscale so it reads as "inactive."
    not_on_ballot_trace = None
    if has_data_mask is not None and (~has_data_mask).any():
        nb = gdf[~has_data_mask]
        not_on_ballot_trace = go.Choropleth(
            geojson=geojson,
            locations=[str(d) for d in nb["district_int"]],
            z=[0.5] * len(nb),  # constant
            featureidkey="id",
            colorscale=[[0.0, "#C8B89A"], [1.0, "#C8B89A"]],  # solid khaki/tan
            showscale=False,
            zmin=0,
            zmax=1,
            marker_line_color="#888888",
            marker_line_width=0.8,
            text=[f"<b>SD-{int(r.district_int):03d}</b><br>Not on the 2026 ballot"
                  for _, r in nb.iterrows()],
            hovertemplate="%{text}<extra></extra>",
            visible=True,
            name="Not on ballot",
        )

    # Build one trace per scenario (active districts only)
    traces = []
    if not_on_ballot_trace is not None:
        traces.append(not_on_ballot_trace)
    for i, (env_dial, scen_label) in enumerate(scenario_list):
        if scen_label not in scenario_cols:
            continue
        # Only include active districts in the political colorscale trace —
        # not-on-ballot districts are drawn by the background trace above.
        active_gdf = gdf[has_data_mask] if has_data_mask is not None else gdf
        z_vals = active_gdf[scen_label].fillna(0.5).tolist()

        trace = go.Choropleth(
            geojson=geojson,
            locations=[str(d) for d in active_gdf["district_int"]],
            z=z_vals,
            featureidkey="id",
            colorscale=COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="D Win<br>Prob", side="right", font=dict(size=12)),
                # 7 discrete category labels matching Wikipedia convention thresholds
                tickvals=[0.05, 0.175, 0.325, 0.50, 0.675, 0.825, 0.95],
                ticktext=["Safe R", "Likely R", "Lean R", "Toss-up",
                          "Lean D", "Likely D", "Safe D"],
                tickfont=dict(size=11),
                len=0.70,
                thickness=16,
                x=1.01,
                outlinewidth=0,
                bgcolor="rgba(255,255,255,0.85)",
            ),
            text=make_hover(scen_label, active_gdf),
            hovertemplate="%{text}<extra></extra>",
            marker_line_color="white",
            marker_line_width=0.5,
            visible=(scen_label == default_scen),
            name=scen_label,
        )
        traces.append(trace)

    # Build scenario dropdown buttons
    # Each button shows the background "not on ballot" trace (always on) plus
    # exactly one scenario trace.
    has_bg = not_on_ballot_trace is not None
    buttons = []
    for i, (env_dial, scen_label) in enumerate(scenario_list):
        if scen_label not in scenario_cols:
            continue
        scen_visible = [sc == scen_label for _, sc in scenario_list if sc in scenario_cols]
        visible = ([True] + scen_visible) if has_bg else scen_visible
        abs_margin = abs(env_dial)
        party = "D" if env_dial >= 0 else "R"
        env_str = f"D+{env_dial:.0f}" if env_dial >= 0 else f"R+{abs_margin:.0f}"
        is_current = abs(env_dial - current_env) < 0.5
        btn_label = f"{env_str}{'  ← current' if is_current else ''}"

        buttons.append(dict(
            label=btn_label,
            method="update",
            args=[
                {"visible": visible},
                {"title": _make_title(chamber, scen_label, env_dial, current_env,
                                      summary_lookup)},
            ],
        ))

    current_scen_label = default_env[1]
    title = _make_title(chamber, current_scen_label, default_env[0], current_env,
                        summary_lookup)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=16, family="system-ui, -apple-system, Arial, sans-serif")),
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#ECECEC",
            showlakes=True,
            lakecolor="#C8DDF0",
            showcoastlines=True,
            coastlinecolor="#BBBBBB",
            showsubunits=True,
            subunitcolor="#BBBBBB",
            fitbounds="locations",
            visible=True,
            bgcolor="#F8F9FA",
            projection_type="albers usa",
        ),
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            x=0.01,
            xanchor="left",
            y=1.02,
            yanchor="bottom",
            bgcolor="white",
            bordercolor="#CCCCCC",
            font=dict(size=13),
            active=default_idx,
            buttons=buttons,
            pad=dict(t=5, b=5),
        )],
        annotations=[dict(
            text="Scenario:",
            x=0.01,
            xanchor="left",
            y=1.065,
            yanchor="bottom",
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13),
        )],
        margin=dict(t=90, b=10, l=10, r=20),
        height=700,
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        font=dict(family="system-ui, -apple-system, Arial, sans-serif"),
    )

    return fig


def _make_title(chamber: str, scen_label: str, env_dial: float, current_env: float,
                summary_lookup: dict | None = None) -> str:
    ch = "TX House" if chamber == "house" else "TX Senate"
    is_current = abs(env_dial - current_env) < 0.5
    env_note = "  (current)" if is_current else ""

    seat_note = ""
    if summary_lookup and scen_label in summary_lookup:
        s = summary_lookup[scen_label]
        if chamber == "house":
            exp  = s["expected_house_seats"]
            ctrl = s["house_control_prob"] * 100
            seat_note = f"  ·  Expected: {exp:.1f}D  ·  Majority: {ctrl:.0f}%"
        else:
            exp  = s["expected_senate_seats"]
            ctrl = s["senate_control_prob"] * 100
            seat_note = f"  ·  Expected: {exp:.1f}D on ballot  ·  Majority: {ctrl:.0f}%"

    return (f"2026 {ch} — D Win Probability by District"
            f"<br><sup>{scen_label}{env_note}{seat_note}</sup>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_chamber_map(chamber: str, current_env: float):
    print(f"\n{'='*60}")
    print(f"  Building {chamber.upper()} map...")
    print(f"{'='*60}")

    gdf = get_shapefile(chamber)
    print(f"  Shapefile: {len(gdf)} districts")

    wide, scenario_list, scenario_cols = load_scenarios(chamber)
    print(f"  Model scenarios: {[s for _, s in scenario_list]}")
    print(f"  Districts with model data: {len(wide)}")
    print(f"  Default scenario: closest to D+{current_env:.1f}")

    summary_lookup = load_summary_lookup()
    fig = build_figure(gdf, wide, scenario_list, scenario_cols, chamber, current_env,
                       summary_lookup)

    out_path = OUTPUT / f"model_2026_map_{chamber}.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",      # link CDN — keeps file small
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"tx_legislature_2026_{chamber}",
                "height": 700,
                "width": 1100,
                "scale": 2,
            },
        },
    )
    print(f"  Written → {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build interactive TX legislative district maps")
    parser.add_argument("--chamber", choices=["house", "senate"],
                        help="Build map for one chamber only (default: both)")
    args = parser.parse_args()

    # Read current environment from model config
    sys.path.insert(0, str(Path(__file__).parent))
    from model_config import GENERIC_BALLOT_TOPLINE_D_2P
    current_env = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)
    print(f"Current environment: D+{current_env:.1f}")

    # Ensure scenario CSV exists
    scenarios_path = OUTPUT / "model_2026_scenarios.csv"
    if not scenarios_path.exists():
        print("model_2026_scenarios.csv not found. Running model.py first...")
        import subprocess
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / "model.py")],
            check=True
        )

    chambers = [args.chamber] if args.chamber else ["house", "senate"]
    out_paths = []
    for ch in chambers:
        p = build_chamber_map(ch, current_env)
        out_paths.append(p)

    print(f"\n{'='*60}")
    print("  Maps ready:")
    for p in out_paths:
        print(f"    {p}")
    print()
    print("  Open in browser. Use the 'Scenario' dropdown to toggle")
    print("  between environment levels. Hover over districts for details.")
  

if __name__ == "__main__":
    main()
