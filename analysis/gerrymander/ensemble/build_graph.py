"""
Phase C, step 1: build the VTD adjacency graph GerryChain needs, with
population and partisan data attached to every node, and the enacted House
and Senate district assignments attached so the enacted map can be scored
against the ensemble on the same graph.

MUST be run with the dedicated Python 3.12 venv (see requirements-ensemble.txt
for why): .venv-ensemble\\Scripts\\python.exe build_graph.py

Data sources:
  - VTD geometry + 2020 census population + 2020/2024 presidential votes +
    CVAP, from the ALARM/Dave's Redistricting App VTD dataset (CC BY 4.0):
    data/raw/geo/vtd_2020/extracted/TX_2020_VD_tabblock.vtd.datasets.geojson
  - Pre-built VTD adjacency (curated by DRA specifically for MCMC tools like
    GerryChain -- used here instead of recomputing adjacency from geometry,
    which is slower and prone to the island/gap topology errors this file
    already resolves): TX_2020_graph.json
  - Enacted TX House/Senate district boundaries:
    data/raw/geo/tx_house_districts.gpkg, tx_senate_districts.gpkg

IMPORTANT decision, documented here because it deviates from the original
plan: the project's own precinct->district crosswalk
(data/raw/_capitol_data_cache/precincts24g_districts.xlsx, keyed by PCTKEY =
county FIPS + county-specific precinct number) uses the Texas Legislative
Council's precinct numbering, which does NOT correspond to the DRA dataset's
GEOID20 VTD codes (Census P.L. redistricting VTD IDs) -- these are two
different, non-interchangeable numbering schemes with no shared key. Rather
than attempt a fragile name-based reconciliation across ~9,000 precincts,
the enacted district assignment is derived by GEOMETRY: each VTD is assigned
to whichever enacted district contains the largest share of its area
(maup.assign), which is standard practice in the redistricting-analysis
literature when no attribute crosswalk exists between two precinct schemes.

Output: ensemble_graph.pkl (pickled dict with the networkx graph + both
chamber assignments), cached so run_chain.py doesn't repeat this work.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import geopandas as gpd
import maup
import networkx as nx
from gerrychain import Graph

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEO = PROJECT_ROOT / "data" / "raw" / "geo"
VTD_DIR = GEO / "vtd_2020" / "extracted"
OUT = PROJECT_ROOT / "output" / "gerrymander"
OUT.mkdir(parents=True, exist_ok=True)

GEOJSON_PATH = VTD_DIR / "TX_2020_VD_tabblock.vtd.datasets.geojson"
ADJACENCY_PATH = VTD_DIR / "TX_2020_graph.json"
HOUSE_DISTRICTS = GEO / "tx_house_districts.gpkg"
SENATE_DISTRICTS = GEO / "tx_senate_districts.gpkg"

CACHE_PATH = OUT / "ensemble_graph.pkl"


def load_vtds() -> gpd.GeoDataFrame:
    print(f"Loading VTD geometry + data: {GEOJSON_PATH.name} ...")
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf.rename(columns={"id": "GEOID20"})
    gdf = gdf.set_index("GEOID20", drop=False)

    # The OGR GeoJSON driver stringifies nested JSON properties (the
    # 'datasets' object) rather than preserving them as dicts -- parse back.
    if len(gdf) and isinstance(gdf["datasets"].iloc[0], str):
        gdf["datasets"] = gdf["datasets"].apply(json.loads)

    def get(ds, key, field):
        d = ds.get(key)
        return d.get(field) if d else None

    gdf["POP20"] = gdf["datasets"].apply(lambda d: get(d, "T_20_CENS", "Total"))
    gdf["PRES20_DEM"] = gdf["datasets"].apply(lambda d: get(d, "E_20_PRES", "Dem"))
    gdf["PRES20_REP"] = gdf["datasets"].apply(lambda d: get(d, "E_20_PRES", "Rep"))
    gdf["PRES20_TOTAL"] = gdf["datasets"].apply(lambda d: get(d, "E_20_PRES", "Total"))
    gdf["PRES24_DEM"] = gdf["datasets"].apply(lambda d: get(d, "E_24_PRES", "Dem"))
    gdf["PRES24_REP"] = gdf["datasets"].apply(lambda d: get(d, "E_24_PRES", "Rep"))
    gdf["PRES24_TOTAL"] = gdf["datasets"].apply(lambda d: get(d, "E_24_PRES", "Total"))
    gdf["CVAP_HISPANIC"] = gdf["datasets"].apply(lambda d: get(d, "V_23_CVAP", "Hispanic"))
    gdf["CVAP_BLACK"] = gdf["datasets"].apply(lambda d: get(d, "V_23_CVAP", "Black"))
    gdf["CVAP_TOTAL"] = gdf["datasets"].apply(lambda d: get(d, "V_23_CVAP", "Total"))

    n_missing_pop = gdf["POP20"].isna().sum()
    n_missing_pres24 = gdf["PRES24_TOTAL"].isna().sum()
    print(f"  {len(gdf)} VTDs loaded. Missing POP20: {n_missing_pop}. Missing 2024 pres data: {n_missing_pres24}.")
    return gdf


def build_networkx_graph(vtds: gpd.GeoDataFrame) -> nx.Graph:
    print(f"Loading pre-built adjacency: {ADJACENCY_PATH.name} ...")
    with open(ADJACENCY_PATH) as f:
        adjacency = json.load(f)

    g = nx.Graph()
    valid_ids = set(vtds.index) - {"OUT_OF_STATE"}
    g.add_nodes_from(valid_ids)

    n_edges = 0
    for node_id, neighbors in adjacency.items():
        if node_id not in valid_ids:
            continue
        for nb in neighbors:
            if nb in valid_ids and nb != node_id:
                g.add_edge(node_id, nb)
                n_edges += 1
    print(f"  Built graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges (from {n_edges} directed entries).")

    n_isolates = list(nx.isolates(g))
    if n_isolates:
        print(f"  WARNING: {len(n_isolates)} isolated VTD(s) with no adjacency listed: {n_isolates[:10]}{'...' if len(n_isolates) > 10 else ''}")

    for node_id in g.nodes:
        row = vtds.loc[node_id]
        g.nodes[node_id]["POP20"] = int(row["POP20"]) if row["POP20"] is not None else 0
        g.nodes[node_id]["PRES20_DEM"] = float(row["PRES20_DEM"] or 0)
        g.nodes[node_id]["PRES20_REP"] = float(row["PRES20_REP"] or 0)
        g.nodes[node_id]["PRES24_DEM"] = float(row["PRES24_DEM"] or 0)
        g.nodes[node_id]["PRES24_REP"] = float(row["PRES24_REP"] or 0)
        g.nodes[node_id]["CVAP_HISPANIC"] = float(row["CVAP_HISPANIC"] or 0)
        g.nodes[node_id]["CVAP_BLACK"] = float(row["CVAP_BLACK"] or 0)
        g.nodes[node_id]["CVAP_TOTAL"] = float(row["CVAP_TOTAL"] or 0)
        g.nodes[node_id]["geometry"] = row["geometry"]

    return g


def assign_enacted_districts(vtds: gpd.GeoDataFrame, chamber_path: Path, label: str) -> dict:
    print(f"Assigning VTDs to enacted {label} districts by max-area overlap (maup.assign) ...")
    districts = gpd.read_file(chamber_path)
    # Texas Centric Albers Equal Area (EPSG:3083) -- project before comparing
    # areas so maup's overlap ranking isn't computed in degrees.
    vtds = vtds.to_crs("EPSG:3083")
    districts = districts.to_crs("EPSG:3083")

    district_id_col = None
    for candidate in ["district", "DISTRICT", "SLDLST", "SLDUST", "NAME"]:
        if candidate in districts.columns:
            district_id_col = candidate
            break
    if district_id_col is None:
        raise ValueError(f"Could not find a district id column in {chamber_path.name}; columns: {list(districts.columns)}")

    assignment = maup.assign(vtds.geometry, districts.geometry)
    # maup.assign returns a Series indexed like vtds, valued as the positional
    # index into `districts`; map that back to the actual district id/number.
    district_labels = districts[district_id_col].astype(str)
    assigned = assignment.map(district_labels)

    n_unassigned = assigned.isna().sum()
    if n_unassigned:
        print(f"  WARNING: {n_unassigned} VTD(s) could not be assigned to a {label} district (likely edge/topology slivers)")

    print(f"  Assigned {assigned.notna().sum()}/{len(assigned)} VTDs across {districts[district_id_col].nunique()} {label} districts.")
    return assigned.to_dict()


def main() -> None:
    vtds = load_vtds()
    g = build_networkx_graph(vtds)

    house_assignment = assign_enacted_districts(vtds, HOUSE_DISTRICTS, "House")
    senate_assignment = assign_enacted_districts(vtds, SENATE_DISTRICTS, "Senate")

    for node_id in g.nodes:
        g.nodes[node_id]["enacted_house_district"] = house_assignment.get(node_id)
        g.nodes[node_id]["enacted_senate_district"] = senate_assignment.get(node_id)

    total_pop = sum(d["POP20"] for _, d in g.nodes(data=True))
    print(f"Total 2020 population across graph: {total_pop:,} (TX 2020 census population was 29,145,505 -- should be close)")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"graph": g}, f)
    print(f"Wrote cached graph: {CACHE_PATH}")


if __name__ == "__main__":
    main()
