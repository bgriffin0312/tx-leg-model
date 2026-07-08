"""
Phase C, step 2: run a ReCom Markov chain ensemble over the TX House and
Senate VTD graph, scoring every sampled plan on the same partisan-metric
battery used in Phases A/B, so the enacted map can be placed in the
distribution of neutrally-drawn alternatives.

MUST be run with the dedicated Python 3.12 venv:
  .venv-ensemble\\Scripts\\python.exe run_chain.py --chamber house --steps 1000
  .venv-ensemble\\Scripts\\python.exe run_chain.py --chamber senate --steps 1000

Requires ensemble_graph.pkl from build_graph.py to already exist.

Every sampled plan is scored on BOTH 2020 and 2024 presidential lean (the
graph carries both), which lets analyze.py show whether the enacted map is
an outlier under either year's electorate, not just the most recent one.

`--steps` default is intentionally modest (see plan verification section:
run a small diagnostic chain first to confirm contiguity/population balance
hold and the enacted-plan seed scores correctly before committing to a large
run). Re-run with a larger --steps for a stronger statistical claim once
timing is known; the script is safe to re-run standalone each time (it does
not require the previous run's output).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import pandas as pd
from gerrychain import Graph, MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.constraints import contiguous, within_percent_of_ideal_population
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import Tally, cut_edges

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for metrics.py
import metrics as gm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT_ROOT / "output" / "gerrymander"
CACHE_PATH = OUT / "ensemble_graph.pkl"

N_DISTRICTS = {"house": 150, "senate": 31}
DEFAULT_POP_EPSILON = 0.02  # +/-2% of ideal district population, standard ReCom tolerance


def load_graph() -> Graph:
    with open(CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    return Graph(cached["graph"])


def build_updaters() -> dict:
    return {
        "population": Tally("POP20", alias="population"),
        "PRES24_DEM": Tally("PRES24_DEM", alias="PRES24_DEM"),
        "PRES24_REP": Tally("PRES24_REP", alias="PRES24_REP"),
        "PRES20_DEM": Tally("PRES20_DEM", alias="PRES20_DEM"),
        "PRES20_REP": Tally("PRES20_REP", alias="PRES20_REP"),
        "cut_edges": cut_edges,
    }


def partition_metrics(partition: Partition) -> dict:
    """Recompute the standard battery for both election years on one sampled plan."""
    dem24, rep24, dem20, rep20 = [], [], [], []
    for part in partition.parts:
        d24, r24 = partition["PRES24_DEM"][part], partition["PRES24_REP"][part]
        d20, r20 = partition["PRES20_DEM"][part], partition["PRES20_REP"][part]
        dem24.append(d24 / (d24 + r24) if (d24 + r24) > 0 else 0.5)
        dem20.append(d20 / (d20 + r20) if (d20 + r20) > 0 else 0.5)

    row = {}
    row.update({f"y24_{k}": v for k, v in gm.run_all(dem24).items()})
    row.update({f"y20_{k}": v for k, v in gm.run_all(dem20).items()})
    return row


def print_phase_a_cross_check(chamber: str, n_districts: int) -> None:
    """Compare the VTD-area enacted reconstruction against Phase A's numbers,
    which come from run_metrics.py's district_detail_2024.csv -- built from
    Texas Legislative Council's own official precinct-to-district crosswalk
    (dem_share_pres_2024), a more precise source than the geometry
    approximation used to seed this ensemble's node attributes."""
    detail_path = OUT / "district_detail_2024.csv"
    if not detail_path.exists():
        print("  [cross-check] Phase A district_detail_2024.csv not found -- run run_metrics.py first for a corroborating check")
        return
    detail = pd.read_csv(detail_path)
    sub = detail[detail["chamber"] == chamber]["dem_share_pres_2024"].dropna().to_numpy()
    if len(sub) == 0:
        print(f"  [cross-check] no Phase A rows found for chamber={chamber}")
        return
    official = gm.run_all(sub)
    print(f"  [cross-check vs Phase A official crosswalk] EG(2024)={official['eg_efficiency_gap']:.4f}, "
          f"Dem seats(2024)={official['pack_n_dem_seats']}/{len(sub)} districts "
          f"(compare to the VTD-area reconstruction above -- similar sign/magnitude corroborates the approximation)")


def run(chamber: str, steps: int, node_repeats: int, pop_epsilon: float) -> pd.DataFrame:
    assignment_col = f"enacted_{chamber}_district"
    n_districts = N_DISTRICTS[chamber]

    print(f"Loading graph for {chamber} ensemble run ({steps} steps) ...")
    graph = load_graph()
    updaters = build_updaters()

    enacted_partition = Partition(graph, assignment=assignment_col, updaters=updaters)
    n_parts = len(enacted_partition.parts)
    if n_parts != n_districts:
        print(f"  WARNING: enacted assignment has {n_parts} parts, expected {n_districts} -- check build_graph.py assignment step")

    total_pop = sum(enacted_partition["population"].values())
    ideal_pop = total_pop / n_parts
    print(f"  Total population {total_pop:,}, ideal district population {ideal_pop:,.0f} across {n_parts} districts")

    # NOTE: enacted_partition is built from the VTD-area-max approximation of
    # the true enacted districts (see build_graph.py docstring) -- VTDs that
    # straddle a real district line are wholly assigned to one side, which
    # both unbalances district population by up to ~10% and creates a few
    # non-contiguous districts. It is therefore NOT used as the chain's
    # initial_state (it would fail gerrychain's is_valid check). It IS still
    # scored here as our best reconstruction of the enacted map's partisan
    # metrics, for placement in the ensemble distribution -- cross-check
    # against run_metrics.py's district_detail_2024.csv (built from Texas
    # Legislative Council's own official precinct-to-district crosswalk, a
    # more precise source for the enacted numbers) is printed below.
    enacted_row = {"step": 0, "is_enacted": True}
    enacted_row.update(partition_metrics(enacted_partition))
    print(f"  Enacted plan (VTD-area reconstruction, approximate): EG(2024)={enacted_row['y24_eg_efficiency_gap']:.4f}, "
          f"Dem seats(2024)={enacted_row['y24_pack_n_dem_seats']}, "
          f"EG(2020)={enacted_row['y20_eg_efficiency_gap']:.4f}, Dem seats(2020)={enacted_row['y20_pack_n_dem_seats']}")
    print_phase_a_cross_check(chamber, n_districts)

    print(f"  Generating a valid contiguous/balanced starting partition (recursive_tree_part, epsilon={pop_epsilon}) for the chain seed ...")
    t_seed = time.time()
    seed_assignment = recursive_tree_part(
        graph, parts=range(n_districts), pop_target=ideal_pop, pop_col="POP20",
        epsilon=pop_epsilon, node_repeats=node_repeats,
    )
    seed_partition = Partition(graph, assignment=seed_assignment, updaters=updaters)
    print(f"  Seed partition generated in {(time.time() - t_seed)/60:.1f} min")

    proposal = partial(
        recom,
        pop_col="POP20",
        pop_target=ideal_pop,
        epsilon=pop_epsilon,
        node_repeats=node_repeats,
    )

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            contiguous,
            within_percent_of_ideal_population(seed_partition, pop_epsilon),
        ],
        accept=always_accept,
        initial_state=seed_partition,
        total_steps=steps,
    )

    rows = [enacted_row]
    t0 = time.time()
    log_every = max(1, steps // 20)
    for i, partition in enumerate(chain, start=1):
        row = {"step": i, "is_enacted": False}
        row.update(partition_metrics(partition))
        rows.append(row)
        if i % log_every == 0 or i == steps:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (steps - i)
            print(f"  step {i}/{steps}  ({rate:.2f}s/step, ~{eta/60:.1f}min remaining)  "
                  f"EG(2024)={row['y24_eg_efficiency_gap']:.4f}  Dem seats(2024)={row['y24_pack_n_dem_seats']}")

    elapsed = time.time() - t0
    print(f"Chain complete: {steps} steps in {elapsed/60:.1f} minutes ({elapsed/steps:.2f}s/step average)")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chamber", choices=["house", "senate"], required=True)
    parser.add_argument("--steps", type=int, default=100, help="Chain length (default 100 for the diagnostic pass; scale up once timing is known)")
    parser.add_argument("--node-repeats", type=int, default=2, help="ReCom node_repeats parameter (standard default)")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_POP_EPSILON,
                         help="Population balance tolerance (default 0.02 = +/-2%%). Loosen for chambers with very "
                              "large districts (e.g. TX Senate) where a tight tolerance makes the balanced-cut "
                              "search in recursive_tree_part/ReCom impractically slow.")
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        raise SystemExit(f"Missing {CACHE_PATH} -- run build_graph.py first")

    df = run(args.chamber, args.steps, args.node_repeats, args.epsilon)
    eps_suffix = "" if args.epsilon == DEFAULT_POP_EPSILON else f"_eps{args.epsilon}"
    out_path = OUT / f"ensemble_{args.chamber}_steps{args.steps}{eps_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
