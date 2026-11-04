"""
Phase C, county-line version (2026-11-04): a neutral ReCom ensemble for the Texas House that follows the
county-line rule (Tex. Const. art. III, sec. 26), scored on 2024 presidential lean.

How the rule is imposed:
  1. County groups held fixed at the enacted plan's (PlanH2316). Two counties are in one group when an enacted district
     touches both; the groups come from TLC's own precinct -> district crosswalk (precincts24g_districts.xlsx), not from
     geometry. No sampled district may cross a group boundary, and each group keeps its enacted number of districts.
  2. Counties the enacted plan keeps whole stay whole: each is collapsed to a single node. Counties the enacted plan
     splits (big counties and their spillover partners) stay at VTD level.
  3. Inside a group, ReCom prefers cuts on county lines (region_surcharge on county), so a big county's surplus tends
     to spill across its line once rather than many times. Splits are counted every SPLIT_EVERY steps and compared
     with the enacted plan's.
  4. Population within +/-5% of the statewide ideal (Texas House practice: total deviation under 10%).

What this does and does not test: line-drawing inside the enacted county groups. The choice of groups is the
legislature's and is held fixed, so a partisan choice of groups would not show up here.

Run with the ensemble venv:
  .venv-ensemble\\Scripts\\python.exe ensemble\\county_line_chain.py --steps 200      (timing / diagnostic)
  .venv-ensemble\\Scripts\\python.exe ensemble\\county_line_chain.py --steps 20000
"""
from __future__ import annotations

import argparse
import collections
import pickle
import random
import sys
import time
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from gerrychain import Graph, MarkovChain, Partition
from gerrychain.accept import always_accept
from gerrychain.constraints import contiguous, within_percent_of_ideal_population
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part, bipartition_tree
from gerrychain.updaters import Tally, cut_edges

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import metrics as gm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "gerrymander"
XW = ROOT / "data" / "raw" / "_capitol_data_cache" / "precincts24g_districts.xlsx"
EPS = 0.05
SPLIT_EVERY = 25
ATTRS = ["POP20", "PRES24_DEM", "PRES24_REP", "PRES20_DEM", "PRES20_REP"]


def county_groups():
    x = pd.read_excel(XW)
    c2d = x.groupby("FIPS").PlanH2316.apply(lambda s: set(s.astype(int)))
    parent = {c: c for c in c2d.index}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d2c = collections.defaultdict(set)
    for c, ds in c2d.items():
        for d in ds:
            d2c[d].add(c)
    for cs in d2c.values():
        cs = list(cs)
        for c in cs[1:]:
            parent[find(c)] = find(cs[0])
    group = {c: find(c) for c in c2d.index}
    gk = collections.Counter()
    for d, cs in d2c.items():
        gk[group[next(iter(cs))]] += 1
    split = {c for c, ds in c2d.items() if len(ds) > 1}
    # enacted county-crossing pieces: (county, district) where the district also has territory outside that county
    crossing = sum(1 for d, cs in d2c.items() if len(cs) > 1 for c in cs if c in split)
    return group, gk, split, crossing, c2d


def build(graph_nx, group, split):
    """Collapse whole counties to one node; drop edges between county groups. Each node also carries the enacted
    district that holds most of its population (VTD-area reconstruction), used only as a fallback seed."""
    rep = {}
    for n in graph_nx.nodes:
        f = int(n[2:5])
        rep[n] = n if f in split else f"C{f:03d}"
    G = nx.Graph()
    enacted_pop = collections.defaultdict(collections.Counter)
    for n, d in graph_nx.nodes(data=True):
        r = rep[n]
        f = int(n[2:5])
        if r not in G:
            G.add_node(r, county=f, group=group[f], **{a: 0.0 for a in ATTRS})
        for a in ATTRS:
            G.nodes[r][a] += float(d.get(a, 0) or 0)
        enacted_pop[r][d.get("enacted_house_district")] += float(d.get("POP20", 0) or 0) + 1e-6
    for r, cnt in enacted_pop.items():
        G.nodes[r]["enacted"] = cnt.most_common(1)[0][0]
    for u, v in graph_nx.edges:
        ru, rv = rep[u], rep[v]
        if ru != rv and G.nodes[ru]["group"] == G.nodes[rv]["group"]:
            G.add_edge(ru, rv)
    for n in G.nodes:
        G.nodes[n]["POP20"] = int(G.nodes[n]["POP20"])
    return G


def seed(G, gk, ideal, rng_seed):
    random.seed(rng_seed)
    assign, label = {}, 0
    by_group = collections.defaultdict(list)
    for n, d in G.nodes(data=True):
        by_group[d["group"]].append(n)
    bad = []
    for g, nodes in sorted(by_group.items()):
        k = gk[g]
        sub = G.subgraph(nodes)
        if not nx.is_connected(sub):
            bad.append((g, "not connected", nx.number_connected_components(sub)))
        cpop = sum(G.nodes[n]["POP20"] for n in nodes)
        dev = cpop / k / ideal - 1
        if k == 1:
            for n in nodes:
                assign[n] = label
            label += 1
            if abs(dev) > EPS:
                bad.append((g, f"single-district group off ideal by {dev:+.1%}"))
            continue
        e = EPS - abs(dev) - 0.0005
        part = None
        if e > 0:
            for attempt in range(3):
                try:
                    part = recursive_tree_part(Graph.from_networkx(sub), parts=range(label, label + k), pop_target=cpop / k,
                                               pop_col="POP20", epsilon=e, node_repeats=2,
                                               method=partial(bipartition_tree, max_attempts=2000))
                    break
                except Exception as ex:  # noqa: BLE001
                    err = ex
        if part is None:
            # fallback: the enacted plan's own lines for this group (VTD-area reconstruction), if they are balanced
            en = {n: G.nodes[n]["enacted"] for n in nodes}
            labs = sorted(set(en.values()), key=str)
            m = {l: label + i for i, l in enumerate(labs)}
            ppop = collections.Counter()
            for n in nodes:
                ppop[en[n]] += G.nodes[n]["POP20"]
            worst = max(abs(p / ideal - 1) for p in ppop.values())
            conn = all(nx.is_connected(sub.subgraph([n for n in nodes if en[n] == l])) for l in labs)
            bad.append((g, f"k={k}, average off ideal {dev:+.2%}: random seed failed, enacted lines used "
                           f"({len(labs)} parts, worst {worst:+.2%}, contiguous {conn})"))
            if len(labs) != k or worst > EPS or not conn:
                raise RuntimeError(f"group {g}: no valid seed (enacted reconstruction {len(labs)} parts, worst {worst:+.2%}, contiguous {conn})")
            part = {n: m[en[n]] for n in nodes}
        assign.update(part)
        label += k
    return assign, bad


def splits(partition, G, split):
    """County-crossing pieces: (split county, district) pairs where the district also has territory outside it."""
    c2d = collections.defaultdict(set)
    d2c = collections.defaultdict(set)
    for n, d in partition.assignment.items():
        c = G.nodes[n]["county"]
        c2d[c].add(d)
        d2c[d].add(c)
    return sum(1 for d, cs in d2c.items() if len(cs) > 1 for c in cs if c in split)


def score(partition):
    sh = []
    for p in sorted(partition.parts):
        dv, rv = partition["PRES24_DEM"][p], partition["PRES24_REP"][p]
        sh.append(dv / (dv + rv) if dv + rv else 0.5)
    r = gm.run_all(sh)
    return {k: r[k] for k in ("eg_efficiency_gap", "mm_mean_median", "pb_partisan_bias", "decl_declination", "pack_n_dem_seats")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--surcharge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plain", action="store_true", help="no county rule: one statewide group, every county splittable, "
                                                          "no surcharge (same chain settings, for an apples-to-apples baseline)")
    a = ap.parse_args()
    if sys.platform == "win32":                      # below-normal CPU priority: yields the moment Brennan needs the machine
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    random.seed(a.seed)

    group, gk, split, enacted_crossing, c2d = county_groups()
    if a.plain:
        group = {c: 0 for c in group}
        gk = collections.Counter({0: 150})
        split = set(group)
        a.surcharge = 0.0
    multi = {g: k for g, k in gk.items() if k > 1}
    print(f"county groups {len(gk)}: {sum(1 for k in gk.values() if k == 1)} single-district, {len(multi)} multi-district "
          f"holding {sum(multi.values())} districts; enacted splits {len(split)} counties, {enacted_crossing} county-crossing pieces")

    g0 = pickle.load(open(OUT / "ensemble_graph.pkl", "rb"))["graph"]
    G = build(g0, group, split)
    total = sum(d["POP20"] for _, d in G.nodes(data=True))
    ideal = total / 150
    print(f"contracted graph {G.number_of_nodes()} nodes, {G.number_of_edges()} edges; population {total:,}, ideal {ideal:,.0f}")

    assign, bad = seed(G, gk, ideal, a.seed)
    for b in bad:
        print("  GROUP PROBLEM", b)
    graph = Graph.from_networkx(G)
    up = {"population": Tally("POP20", alias="population"), "PRES24_DEM": Tally("PRES24_DEM", alias="PRES24_DEM"),
          "PRES24_REP": Tally("PRES24_REP", alias="PRES24_REP"), "cut_edges": cut_edges}
    p0 = Partition(graph, assignment=assign, updaters=up)
    pops = np.array(list(p0["population"].values()))
    print(f"seed: {len(p0.parts)} districts, population {pops.min() / ideal - 1:+.2%} to {pops.max() / ideal - 1:+.2%} of ideal, "
          f"county-crossing pieces {splits(p0, G, split)}")

    # enacted, from Phase A's official crosswalk numbers (2024 presidential lean)
    det = pd.read_csv(OUT / "district_detail_2024.csv")
    en = gm.run_all(det[det.chamber == "house"].dem_share_pres_2024.dropna().to_numpy())
    print(f"enacted (TLC crosswalk): EG {en['eg_efficiency_gap']:+.4f}, Dem seats {en['pack_n_dem_seats']}, "
          f"mean-median {en['mm_mean_median']:+.4f}, bias {en['pb_partisan_bias']:+.4f}, declination {en['decl_declination']:+.4f}")

    # a pair that cannot be rebalanced within 200 spanning trees is skipped for that step and another pair is drawn
    # (tight rural groups whose average sits near -5% would otherwise stall the chain)
    proposal = partial(recom, pop_col="POP20", pop_target=ideal, epsilon=EPS, node_repeats=2,
                       region_surcharge={"county": a.surcharge} if a.surcharge else None,
                       method=partial(bipartition_tree, max_attempts=200, allow_pair_reselection=True))
    chain = MarkovChain(proposal=proposal, constraints=[contiguous, within_percent_of_ideal_population(p0, EPS)],
                        accept=always_accept, initial_state=p0, total_steps=a.steps)
    rows, t0 = [], time.time()
    moved = collections.Counter()
    prev = None
    for i, p in enumerate(chain, 1):
        row = {"step": i}
        row.update(score(p))
        if i % SPLIT_EVERY == 0:
            row["county_crossing"] = splits(p, G, split)
        if prev is not None and p.flips:
            moved[G.nodes[next(iter(p.flips))]["group"]] += 1
        prev = p
        rows.append(row)
        if i % max(1, a.steps // 20) == 0:
            el = time.time() - t0
            print(f"  step {i}/{a.steps} ({el / i:.3f}s/step, ~{el / i * (a.steps - i) / 60:.1f} min left) "
                  f"EG {row['eg_efficiency_gap']:+.4f} Dem seats {row['pack_n_dem_seats']}", flush=True)
    df = pd.DataFrame(rows)
    out = OUT / f"ensemble_house_{'plain_eps05' if a.plain else 'countyline'}_steps{a.steps}_seed{a.seed}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}; {(time.time() - t0) / 60:.1f} min")

    # placement of the enacted map
    print("\nENACTED vs COUNTY-LINE ENSEMBLE (2024 presidential lean; positive = pro-R)")
    for k, lab in (("eg_efficiency_gap", "efficiency gap"), ("mm_mean_median", "mean-median"), ("pb_partisan_bias", "partisan bias"),
                   ("decl_declination", "declination"), ("pack_n_dem_seats", "Dem seats")):
        s = df[k].dropna()
        e = en[k]
        if k == "pack_n_dem_seats":
            pct = (s > e).mean() * 100 + (s == e).mean() * 50   # share of plans giving Democrats more seats (ties half)
            print(f"  {lab:15} enacted {e:>7}  ensemble mean {s.mean():7.2f} (5-95%: {s.quantile(.05):.0f} to {s.quantile(.95):.0f})  "
                  f"plans giving D more seats {pct:5.1f}%")
        else:
            pct = (s < e).mean() * 100 + (s == e).mean() * 50
            print(f"  {lab:15} enacted {e:+.4f}  ensemble mean {s.mean():+.4f} (5-95%: {s.quantile(.05):+.4f} to {s.quantile(.95):+.4f})  "
                  f"plans less pro-R {pct:5.1f}%")
    cc = df.county_crossing.dropna()
    print(f"  county-crossing pieces: enacted {enacted_crossing}, ensemble mean {cc.mean():.1f} (5-95%: {cc.quantile(.05):.0f} to {cc.quantile(.95):.0f})")
    print(f"  groups that moved (steps): {len(moved)} of {len(multi)} multi-district groups; fewest {min(moved.values()) if moved else 0}, "
          f"never moved: {len(set(multi) - set(moved))}")


if __name__ == "__main__":
    main()
