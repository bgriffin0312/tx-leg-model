"""
build_competitive_map.py

Focused TX House map highlighting only competitive districts at the current
environment. Competitive = D win prob in [25%, 75%]. Each competitive district
is colored on a red-to-blue gradient by P(D wins); non-competitive districts
are washed out so the eye lands on the action.

Output: output/model_2026_map_house_competitive.html

USAGE:
  python src/build_competitive_map.py
  python src/build_competitive_map.py --low 0.20 --high 0.80   # widen band
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from build_maps import (
    COLORSCALE,
    DIST_FIELD,
    get_shapefile,
    load_scenarios,
    load_summary_lookup,
    win_prob_label,
)

OUTPUT = ROOT / "output"


def load_candidate_lookup() -> dict[tuple[str, int], dict]:
    """
    Build {(chamber, district): {r, d, r_status, d_status}} from
    candidates_2026.csv, with runoff pairs filled in from tx_primary_2026_runoffs.csv.
    Incumbents are marked with a trailing ★.
    """
    cand_path = ROOT / "data" / "processed" / "candidates_2026.csv"
    runoff_path = ROOT / "data" / "raw" / "tx_primary_2026_runoffs.csv"

    runoff_lookup: dict[tuple, str] = {}
    if runoff_path.exists():
        ro = pd.read_csv(runoff_path)
        for _, rr in ro.iterrows():
            key = (str(rr["chamber"]).strip(), int(rr["district"]), str(rr["party"]).strip())
            runoff_lookup[key] = f"{str(rr['candidate_1']).strip()} / {str(rr['candidate_2']).strip()}"

    out: dict[tuple[str, int], dict] = {}
    if not cand_path.exists():
        return out
    cdf = pd.read_csv(cand_path)
    for _, row in cdf.iterrows():
        ch = str(row["chamber"]).strip()
        dist = int(row["district"])
        r_status = str(row.get("r_status") or "").strip()
        d_status = str(row.get("d_status") or "").strip()
        r_name = str(row.get("r_candidate") or "").strip()
        d_name = str(row.get("d_candidate") or "").strip()

        if r_status == "runoff":
            r_pair = runoff_lookup.get((ch, dist, "R"))
            r_name = f"{r_pair} (runoff)" if r_pair else "TBD (runoff)"
        if d_status == "runoff":
            d_pair = runoff_lookup.get((ch, dist, "D"))
            d_name = f"{d_pair} (runoff)" if d_pair else "TBD (runoff)"

        if r_status == "incumbent" and r_name:
            r_name = f"{r_name}★"
        if d_status == "incumbent" and d_name:
            d_name = f"{d_name}★"

        out[(ch, dist)] = {"r": r_name, "d": d_name,
                           "r_status": r_status, "d_status": d_status}
    return out


def build_competitive_figure(
    gdf: gpd.GeoDataFrame,
    wide: pd.DataFrame,
    scenario_list: list[tuple],
    scenario_cols: list[str],
    current_env: float,
    low: float,
    high: float,
    summary_lookup: dict | None = None,
) -> tuple[go.Figure, list[dict]]:
    """Build a single-scenario House map highlighting the competitive band."""

    # Pick the scenario closest to current environment
    default_env, default_scen = min(scenario_list, key=lambda x: abs(x[0] - current_env))

    # Merge model data onto the shapefile
    gdf = gdf.copy()
    gdf["district_int"] = gdf[DIST_FIELD["house"]].astype(int)
    merge_cols = ["district", "incumbent", "incumbent_party",
                  "dem_pres_2p_baseline", default_scen]
    gdf = gdf.merge(wide[merge_cols], left_on="district_int", right_on="district",
                    how="left")
    gdf["geometry"] = gdf["geometry"].simplify(0.005, preserve_topology=True)

    win_prob = gdf[default_scen]
    competitive_mask = (win_prob >= low) & (win_prob <= high)
    bg_mask = ~competitive_mask

    geojson = json.loads(gdf.geometry.to_json())
    for i, feat in enumerate(geojson["features"]):
        feat["id"] = str(gdf.iloc[i]["district_int"])

    cand_lookup = load_candidate_lookup()

    def hover_text(rows: pd.DataFrame) -> list[str]:
        out = []
        for _, r in rows.iterrows():
            d = int(r["district_int"])
            wp = r.get(default_scen)
            pres = r.get("dem_pres_2p_baseline")
            if pd.isna(wp):
                label, wp_str = "No data", "—"
            else:
                label = win_prob_label(wp)
                wp_str = f"{wp*100:.1f}%"
            pres_str = f"{float(pres)*100:.1f}%" if pd.notna(pres) else "?"

            cand = cand_lookup.get(("House", d), {})
            r_name = cand.get("r") or "(none filed)"
            d_name = cand.get("d") or "(none filed)"
            out.append(
                f"<b>HD-{d:03d}</b><br>"
                f"R: {r_name}<br>"
                f"D: {d_name}<br>"
                f"Pres baseline: {pres_str}<br>"
                f"D win prob: <b>{wp_str}</b> — {label}"
            )
        return out

    traces = []

    # Background: non-competitive districts in muted gray
    bg = gdf[bg_mask]
    if len(bg):
        traces.append(go.Choropleth(
            geojson=geojson,
            locations=[str(d) for d in bg["district_int"]],
            z=[0.5] * len(bg),
            featureidkey="id",
            colorscale=[[0.0, "#EAEAEA"], [1.0, "#EAEAEA"]],
            showscale=False,
            zmin=0,
            zmax=1,
            marker_line_color="#BBBBBB",
            marker_line_width=0.3,
            text=hover_text(bg),
            hovertemplate="%{text}<extra></extra>",
            name="Not competitive",
        ))

    # Foreground: competitive districts on the full red→blue scale
    comp = gdf[competitive_mask].sort_values(default_scen)
    if len(comp):
        traces.append(go.Choropleth(
            geojson=geojson,
            locations=[str(d) for d in comp["district_int"]],
            z=comp[default_scen].tolist(),
            featureidkey="id",
            colorscale=COLORSCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title=dict(text="D Win<br>Prob", side="right", font=dict(size=12)),
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
            marker_line_color="white",
            marker_line_width=0.8,
            text=hover_text(comp),
            hovertemplate="%{text}<extra></extra>",
            name="Competitive",
        ))

    # Title
    env_str = f"D+{default_env:.1f}" if default_env >= 0 else f"R+{abs(default_env):.1f}"
    seat_note = ""
    if summary_lookup and default_scen in summary_lookup:
        s = summary_lookup[default_scen]
        seat_note = (f"  ·  Expected: {s['expected_house_seats']:.1f}D"
                     f"  ·  Majority: {s['house_control_prob']*100:.0f}%")
    title = (f"2026 TX House — Competitive Districts "
             f"(D win prob {int(low*100)}–{int(high*100)}%)"
             f"<br><sup>{env_str} (current){seat_note}  ·  "
             f"{len(comp)} competitive of 150</sup>")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=16, family="system-ui, -apple-system, Arial, sans-serif")),
        geo=dict(
            scope="usa",
            showland=True,
            landcolor="#F5F5F5",
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
        margin=dict(t=90, b=10, l=10, r=20),
        height=720,
        paper_bgcolor="#F8F9FA",
        plot_bgcolor="#F8F9FA",
        font=dict(family="system-ui, -apple-system, Arial, sans-serif"),
    )

    comp_rows = []
    for _, r in comp.sort_values(default_scen, ascending=False).iterrows():
        d = int(r["district_int"])
        cand = cand_lookup.get(("House", d), {})
        comp_rows.append({
            "district": d,
            "r_candidate": cand.get("r") or "(none filed)",
            "d_candidate": cand.get("d") or "(none filed)",
            "win_prob_d": float(r[default_scen]),
        })
    return fig, comp_rows


def main():
    p = argparse.ArgumentParser(description="Competitive-only TX House map")
    p.add_argument("--low", type=float, default=0.25,
                   help="Lower D-win-prob bound for competitive band (default 0.25)")
    p.add_argument("--high", type=float, default=0.75,
                   help="Upper D-win-prob bound for competitive band (default 0.75)")
    args = p.parse_args()

    from model_config import GENERIC_BALLOT_TOPLINE_D_2P
    current_env = round((GENERIC_BALLOT_TOPLINE_D_2P - 0.5) * 200, 1)
    print(f"Current environment: D+{current_env:.1f}")
    print(f"Competitive band: [{args.low:.0%}, {args.high:.0%}]")

    scenarios_path = OUTPUT / "model_2026_scenarios.csv"
    if not scenarios_path.exists():
        print("model_2026_scenarios.csv not found — run python src/model.py first.")
        sys.exit(1)

    gdf = get_shapefile("house")
    wide, scenario_list, scenario_cols = load_scenarios("house")
    summary_lookup = load_summary_lookup()

    fig, comp_rows = build_competitive_figure(
        gdf, wide, scenario_list, scenario_cols,
        current_env, args.low, args.high, summary_lookup,
    )

    out_path = OUTPUT / "model_2026_map_house_competitive.html"
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "tx_house_2026_competitive",
                "height": 720,
                "width": 1100,
                "scale": 2,
            },
        },
    )

    print(f"\nCompetitive districts ({len(comp_rows)}):  ★ = incumbent")
    print(f"  {'HD':>4s}  {'D win':>6s}  {'R candidate':36s}  {'D candidate':36s}")
    for r in comp_rows:
        print(f"  {r['district']:>4d}  {r['win_prob_d']*100:>5.1f}%  "
              f"{r['r_candidate']:36s}  {r['d_candidate']:36s}")

    print(f"\nWritten → {out_path.name}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
