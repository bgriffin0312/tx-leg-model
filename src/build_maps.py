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

# Color scale: red (R) → white (tossup) → blue (D)
# Plotly colorscale format: [[position, color], ...]
COLORSCALE = [
    [0.00, "#B22222"],   # firebrick — Safe R
    [0.15, "#D9534F"],   # soft red — Likely R
    [0.35, "#F4A58A"],   # salmon — Lean R
    [0.45, "#FDE8DC"],   # very light — near tossup R
    [0.50, "#F5F5F5"],   # near white — tossup
    [0.55, "#D8E8F8"],   # very light blue — near tossup D
    [0.65, "#6AACD4"],   # medium blue — Lean D
    [0.85, "#3170A7"],   # blue — Likely D
    [1.00, "#08306B"],   # dark navy — Safe D
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

    # Hover text template (shared across scenarios)
    def make_hover(col: str) -> list[str]:
        texts = []
        for _, row in gdf.iterrows():
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

    # Build one trace per scenario
    traces = []
    for i, (env_dial, scen_label) in enumerate(scenario_list):
        if scen_label not in scenario_cols:
            continue
        z_vals = gdf[scen_label].fillna(0.5).tolist()

        trace = go.Choropleth(
            geojson=geojson,
            locations=[str(d) for d in gdf["district_int"]],
            z=z_vals,
            featureidkey="id",
            colorscale=COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="D Win<br>Prob",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["Safe R", "Likely R", "Toss-up", "Likely D", "Safe D"],
                len=0.6,
                thickness=14,
                x=1.01,
            ),
            text=make_hover(scen_label),
            hovertemplate="%{text}<extra></extra>",
            marker_line_color="white",
            marker_line_width=0.5,
            visible=(scen_label == default_scen),
            name=scen_label,
        )
        traces.append(trace)

    # Build scenario dropdown buttons
    # Each button shows exactly one trace (the one for that scenario)
    buttons = []
    for i, (env_dial, scen_label) in enumerate(scenario_list):
        if scen_label not in scenario_cols:
            continue
        visible = [sc == scen_label for _, sc in scenario_list if sc in scenario_cols]
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
                {"title": _make_title(chamber, scen_label, env_dial, current_env)},
            ],
        ))

    chamber_title = "TX House" if chamber == "house" else "TX Senate"
    current_scen_label = default_env[1]
    title = _make_title(chamber, current_scen_label, default_env[0], current_env)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#F0F0F0",
            showlakes=True,
            lakecolor="#D0E8F8",
            showcoastlines=True,
            coastlinecolor="#AAAAAA",
            showsubunits=True,
            subunitcolor="#888888",
            fitbounds="locations",
            visible=True,
            bgcolor="#FAFAFA",
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
        margin=dict(t=80, b=20, l=20, r=20),
        height=680,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def _make_title(chamber: str, scen_label: str, env_dial: float, current_env: float) -> str:
    ch = "TX House" if chamber == "house" else "TX Senate"
    is_current = abs(env_dial - current_env) < 0.5
    suffix = "  (current environment)" if is_current else ""
    return f"2026 {ch} — D Win Probability by District<br><sup>{scen_label}{suffix}</sup>"


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

    fig = build_figure(gdf, wide, scenario_list, scenario_cols, chamber, current_env)

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
