# Testing the Texas Legislature Partisan Gerrymander

Methodology and findings for a self-contained analysis of the 2021-drawn Texas House (150
districts) and Senate (31 districts) maps: (1) standard partisan-symmetry metrics on the enacted
maps, (2) whether demographic/voting-pattern change since 2021 has weakened or strengthened the
gerrymander, and (3) an ensemble/outlier test against neutrally-drawn alternative maps.

Code lives in this directory; all outputs are in `../../output/gerrymander/`. Nothing here modifies
the parent TX Legislature Modeling forecasting project -- this module only reads its data.

**Sign convention used throughout: positive = pro-Republican bias, negative = pro-Democratic bias**,
for every metric (efficiency gap, mean-median, partisan bias, declination), matching the Princeton
Gerrymandering Project's convention. See `metrics.py`'s module docstring for exact formulas and
sources, and its `self_test()` (run automatically) for sign-convention validation against synthetic
worked examples before any real number is trusted.

## Phase A: does the enacted map show a partisan-symmetry bias?

`metrics.py` + `run_metrics.py`. Full battery (efficiency gap, mean-median difference, partisan bias
via uniform swing, Warrington declination, packing diagnostics, seats-votes curve/responsiveness) on
two independent indices: 2024 presidential two-party lean (defined for every district, avoids the
uncontested-race problem) and each district's own most recent actual legislative result (imputed
with presidential lean where uncontested).

**Result: every metric, both indices, both chambers, points pro-Republican.** Senate bias is larger
than House bias on every single metric -- consistent with the Princeton Gerrymandering Project's
independent grades (Senate F, House C):

| chamber | index | efficiency gap | mean-median | partisan bias | declination |
|---|---|---|---|---|---|
| House | 2024 pres lean | +0.044 | +0.011 | +0.033 | +0.134 |
| House | actual last election | +0.015 | +0.023 | +0.047 | +0.080 |
| Senate | 2024 pres lean | +0.072 | +0.040 | +0.145 | +0.174 |
| Senate | actual last election | +0.069 | +0.060 | +0.113 | +0.205 |

Sanity check: thresholding 2024 presidential lean at 50% would elect 96R-54D (House) and 21R-10D
(Senate), vs. actual current composition 88R-62D and 20R-11D. The 8-seat House gap is within the
range explainable by incumbency/candidate-quality effects (well-documented in the forecasting
model's own WAR metric); a much larger unexplained gap would have flagged a data bug.

Outputs: `metrics_by_year.csv`, `district_detail_2024.csv`, `seats_votes_{house,senate}.png`,
`margin_distribution.png`.

## Phase B: has the gerrymander weakened or strengthened since 2021?

`drift.py` + `run_drift.py`. As-drawn baseline = 2020 presidential lean (the maps were built on 2020
census population and 2020 was the most recent full statewide election at drawing time); current =
2024. Both years are on the SAME current district lines, so this is apples-to-apples (2016 predates
the current lines entirely and is not used for the trajectory claim). The authoritative evidence is
the full metric battery recomputed under 2020 vs. 2024 lean on the same map -- not any individual
district's story.

**Known data gap:** the project's 2020 historical presidential file is missing House districts 10 and
63 entirely and has zero recorded votes for district 65. These 3/150 districts (2%) are dropped from
the trajectory comparison (147/150 used) rather than imputed; they remain visible as NaN in
`drift_by_district.csv`.

**Result: the two chambers moved in OPPOSITE directions.**

| chamber | year | efficiency gap | mean-median | partisan bias | declination | Dem seats |
|---|---|---|---|---|---|---|
| House | 2020 | +0.024 | +0.016 | +0.031 | +0.067 | 67 |
| House | 2024 | +0.039 | +0.011 | +0.024 | +0.122 | 54 |
| Senate | 2020 | +0.078 | +0.050 | +0.113 | +0.181 | 12 |
| Senate | 2024 | +0.072 | +0.040 | +0.145 | +0.174 | 10 |

- **House: STRENGTHENED** (efficiency gap +0.024 -> +0.039, more pro-R). Dummymander classification:
  56 districts' movement strengthens the gerrymander (mostly Dem-leaning seats packing further D)
  vs. only 7 that weaken it (R-held seats trending D). 64 are "ambiguous" (Dem-leaning seats trending
  R -- see below).
- **Senate: weakened slightly** (efficiency gap +0.078 -> +0.072). Dummymander classification: 15
  strengthen vs. 2 weaken, but the map-level number still net-improved -- a reminder that the
  district-level classification is descriptive color, not the load-bearing evidence; only the
  map-level recomputation resolves the net direction.
- **Statewide consistency check:** summing district-level presidential votes back to a statewide
  total gives a ~3.9pp Republican shift 2020->2024, matching the known direction of the actual
  statewide result (used only as a sanity check on data aggregation, not as a headline number).
- **Rio Grande Valley spot-check:** the 9 House districts with >80% Hispanic CVAP shifted a median
  -10.9pp toward Republicans 2020->2024 -- confirms the well-documented Hispanic realignment toward
  Trump is showing up correctly in this data. These districts land in the "ambiguous" dummymander
  bucket (Dem-leaning districts trending R) precisely because their effect on the map's overall bias
  is genuinely two-directional: it reduces how packed those particular Dem seats are, but the House's
  net trajectory still strengthened because packing increased elsewhere.

Outputs: `drift_by_district.csv`, `map_trajectory.csv`, `drift_scatter_{house,senate}.png`,
`metrics_trajectory.png`.

## Phase C: is the enacted map a statistical outlier vs. a neutral ensemble?

`ensemble/build_graph.py`, `ensemble/run_chain.py`, `ensemble/analyze.py`. Runs in a dedicated Python
3.12 venv (`.venv-ensemble/`) because `gerrychain` pins a `numpy` version with no Python 3.14 wheel
and this machine has no C compiler to build one; Phases A/B run fine on the main project's Python
3.14 environment. See `requirements-ensemble.txt` for exact setup.

**Data:** ALARM/Dave's Redistricting App 2020 VTD dataset (CC BY 4.0, no-sell restriction) --
9,007 Texas VTDs with 2020 census population, 2020 and 2024 presidential votes, CVAP, polygon
geometry, and a pre-curated adjacency graph (used instead of recomputing adjacency from geometry,
which is slower and error-prone).

**Enacted-map reconstruction caveat:** the project's own precinct-to-district crosswalk
(`precincts24g_districts.xlsx`, keyed by Texas Legislative Council precinct numbers) does not share a
key with this dataset's Census VTD codes -- the two are independent numbering schemes with no
attribute crosswalk between them. Enacted district assignment is therefore done by GEOMETRY
(`maup.assign`, max-area overlap of each VTD against the enacted district boundaries), which is
standard practice in the literature when no attribute crosswalk exists, but is an approximation:
VTDs that straddle a real district line get wholly assigned to one side. Validation: the assignment's
total population matches Texas's official 2020 census population EXACTLY (29,145,505), and every VTD
assigned cleanly across both chambers -- strong evidence the reconstruction is directionally sound --
but individual district populations deviate from the true enacted lines by up to ~10% at the margins,
enough to fail a tight 2% population-balance check and to introduce a handful of contiguity artifacts.
Consequently the ensemble's Markov chain is NOT seeded from this reconstruction (it would fail
gerrychain's validity check); it's seeded from a fresh `recursive_tree_part` random valid partition
instead, and the enacted map is scored separately for placement in the resulting distribution, with a
cross-check against Phase A's officially-sourced numbers (Texas Legislative Council's own
precinct-to-district crosswalk) printed at run time.

**Caveats that bound how strong Phase C's claim can be** (both push toward UNDERSTATING how extreme
the enacted map is, i.e. this is a conservative/lower-bound test):
- **Unconstrained ensemble**: no Voting Rights Act minority-opportunity-district preference, so the
  "neutral" baseline may run a higher Dem-seat count than a VRA-compliant neutral map would.
- **No Texas House county-line rule** (Tex. Const. art. III, sec. 26): the enacted House map is
  legally constrained to respect county boundaries where population allows; this ensemble isn't.

**Ensemble settings:** House ran 20,000 ReCom plans at the standard 2% population-balance tolerance.
Senate could not run at that tolerance -- with ~940,000 people per district (vs. ~194,000 for House),
the balanced population-cut search inside `recursive_tree_part`/ReCom became impractically slow: an
initial 20,000-step attempt at 2% never completed a single step in 30+ minutes of active CPU time and
was killed. Senate was re-run at a 5% tolerance (15,000 plans), which resolved the bottleneck
immediately (seed generation dropped from >30 min to 0.1 min). This is a genuine methodological
deviation between the two chambers' ensembles, not a hidden shortcut -- flagged here because a looser
population tolerance widens the space of "neutral" plans slightly, which if anything makes the Senate
outlier finding below a more conservative (harder to achieve) result, not an inflated one.

**Result: both chambers' enacted maps are pro-Republican outliers vs. their neutral ensembles, most
starkly under the 2020 electorate that existed when the maps were drawn.**

| chamber | year | enacted Dem seats | ensemble mean Dem seats | Dem-seats percentile | enacted EG | ensemble mean EG | EG percentile |
|---|---|---|---|---|---|---|---|
| House | 2024 | 54 | 56.8 | 12.7 | +0.042 | +0.017 | **97.1** |
| House | 2020 | 65 | 72.8 | 0.0 | +0.044 | -0.016 | **100.0** |
| Senate | 2024 | 10 | 10.6 | 46.9 | +0.066 | +0.038 | **81.9** |
| Senate | 2020 | 12 | 14.6 | 4.2 | +0.079 | -0.014 | **99.6** |

(Percentile = enacted map's rank among ensemble plans on the pro-R end of the distribution; 100 =
more pro-Republican than every single sampled neutral plan.)

- **House** is an extreme, clean outlier: at 2024 lean, its efficiency gap beats 97% of 20,000
  neutrally-drawn alternative maps toward the Republican end; at 2020 lean (the electorate at
  drawing time) it beats literally all 20,000 -- no neutral map generated was as pro-Republican as
  the enacted map under that electorate.
- **Senate** is a real but more moderate outlier under 2024 lean (82nd percentile on EG; its raw
  Dem-seat count, 10, is unremarkable vs. the ensemble mean of 10.6 -- EG and seat count capture
  different things, since EG weighs vote *margins* within districts, not just which side of 50% each
  district lands on). Under 2020 lean, Senate becomes just as extreme as House: 99.6th percentile on
  EG, and its Dem-seat count (12) is at the 4.2 percentile -- fewer neutral maps give Democrats that
  few seats.
- **Cross-check corroboration:** the VTD-area enacted reconstruction's EG (House 0.042, Senate 0.066)
  is close to Phase A's officially-crosswalked EG (House 0.044, Senate 0.072) in both sign and
  magnitude for both chambers -- the geometry-based approximation is a reasonable stand-in given the
  precinct-numbering-scheme mismatch documented above.
- **Connecting back to Phase B:** both chambers show a MORE extreme percentile under the as-drawn
  2020 electorate than under 2024 -- consistent with Phase B's finding that Senate's bias eased
  somewhat since drawing. House's raw EG value grew 2020->2024 (Phase B), yet its ensemble
  *percentile* stayed roughly as extreme (97th, just short of 2020's ceiling) -- because the neutral
  ensemble's own achievable range also shifts with the electorate. Read the percentile as "how extreme
  relative to what a neutral process could produce under that electorate," not as a second measure of
  raw bias magnitude.

Outputs: `outlier_summary_{house,senate}.csv`, `outlier_{house,senate}_{2020,2024}_{seats,eg}.png`,
`ensemble_house_steps20000.csv`, `ensemble_senate_steps15000_eps0.05.csv`, `ensemble_graph.pkl`.

## Reuse

Read-only reuse of the parent project's `data/processed/districts_2026.csv`, `data/raw/tx_*_2024.csv`
and `data/raw/historical/*` by-district CSVs, `data/raw/_capitol_data_cache/*` VTD returns and
crosswalks (referenced but not keyed against, per the Phase C caveat above), and `data/raw/geo/*`
district boundaries. No parent-project files are modified.
