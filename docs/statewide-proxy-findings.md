# Which statewide race tracks Texas House results? (2026-09-10)

Question from Brennan: Texas polls with race crosstabs measure idiosyncratic
statewide races (a weak R Senate candidate, well-funded incumbent Governor and
Lt Gov). Which statewide contest is the right instrument for the *legislative*
environment, and is a low-salience race like the 2026 AG contest the best look?

Two tests, both on Capitol Data Portal VTD returns. Scripts:
`scripts/statewide_proxy_same_year.py` and `scripts/statewide_proxy_cross_cycle.py`.
Contested House seats only (both a D and an R on the ballot).

## Test 1 — same year: which race's district pattern matches the House vote?

Residual sd (pp) after regressing contested House D two-party share on the
office's D two-party share in the same districts, same election. Lower = the
office's geography looks more like the legislative geography.

| year | best proxies (resid sd) | marquee races (resid sd) |
|---|---|---|
| 2016 | CCA 5 1.80, Sup Ct 9 1.85, RRC 1.97 | **President 2.98** |
| 2018 | Sup Ct 6 1.51, CCA 7 1.55, Comptroller 1.57 | Gov 1.96, AG 2.00, LtGov 2.08, **Sen 2.18** |
| 2020 | Sup Ct 8 1.71, Chief 1.80, RRC 1.88 | Sen 2.26, **President 3.14** |
| 2022 | Sup Ct 9 1.63, Ag Comm 1.73, Comptroller 1.78 | AG 2.17, Gov 2.32, **LtGov 2.68** |
| 2024 | RRC 1.62, CCA 8 1.64, Sup Ct 4 1.66 | Sen 2.10, **President 2.40** |

Every cycle: the least salient statewide races (Supreme Court, Court of
Criminal Appeals, Railroad Commissioner, Comptroller) track the House best;
the top-of-ticket races track it worst. The presidential vote is the *worst*
of the nine 2024 statewide contests as a House proxy.

Where the marquee races miss is the Hispanic-heavy seats. Mean gap (House
minus office) in districts over 70% Hispanic CVAP:

| year | President / Governor | judicial or RRC |
|---|---|---|
| 2018 | Governor +7.4 | Sup Ct 6 +1.1 |
| 2020 | President +8.4, Senate +8.1 | Sup Ct 8 +4.7 |
| 2022 | Governor +4.4 | Sup Ct 9 +1.5 |
| 2024 | President +3.9 | Sup Ct 6 +1.0, RRC +2.0 |

AG specifically: mid-pack in 2018 and 2022. Both were Paxton races, not
low-salience. The 2026 pairing (Middleton v. Johnson) is a different kind of
race and should behave more like the judicial column, but that is an
expectation, not a measurement.

## Test 2 — cross cycle: which presidential-year race best predicts the next midterm's House?

This is the model's actual use of a baseline. Presidential-year precinct
votes area-weighted onto the midterm's House plan (reusing
`collect_presidential_spatial.py`; 99.96% of vote attached), then OLS of the
midterm contested House D share on each office.

| pair | best (resid sd) | President (resid sd) | 70%+ Hispanic residual: President vs best judicial |
|---|---|---|---|
| 2016 → 2018 | **President 2.35** | 2.35 | −1.8 vs −5.3 (CCA 5) |
| 2020 → 2022 | **RRC 2.47** | 2.84 (worst of 10) | +5.7 vs +0.7 (RRC) |

Split decision, and the split has a structural explanation: Texas abolished
straight-ticket voting starting with the 2020 general. Before that, downballot
judicial races in South Texas carried inflated Democratic margins from
straight-ticket Democratic voters, which is why the 2016 judicial baselines
overshoot the 2018 House in Hispanic districts by 5pp. From 2020 on, the
downballot races reflect candidate-level partisanship, and the 2020 → 2022 pair
is the only one that resembles 2024 → 2026. In that pair the presidential
baseline misses the 70%+ Hispanic seats by +5.7pp (House Democrats ran far
ahead of Biden's 2020 number) and the RRC baseline misses by +0.7pp.

Empirical pass-through: the district-level slope of midterm House on
presidential-year statewide is 1.00 (2016→2018) and 1.08–1.12 (2020→2022) for
every office. This is direct evidence for the refit branch's 0.98 over
master's 0.60, *provided the baseline is a downballot race*. The refit's
South Texas problem (HD 34/41/74 predicted at their Trump-era presidential
number) is a baseline-choice problem, not a pass-through problem.

## What the 2026 Texas banners say (UT/TPP, registered voters, D two-party %)

| | White | Black | Hispanic |
|---|---|---|---|
| Aug: TX Lege generic | 35.5 | 82.6 | 53.3 |
| Aug: AG (Middleton/Johnson) | 37.7 | 81.7 | 51.4 |
| Aug: Comptroller (Huffines/Eckhardt) | 35.9 | 83.1 | 47.9 |
| Aug: Governor | 37.9 | 82.4 | 50.6 |
| Aug: Senate | 41.0 | 85.2 | 57.3 |
| Jun: TX Lege generic | 35.4 | 86.6 | 55.6 |
| Jun: AG | 34.2 | 83.8 | 56.8 |
| Jun: Comptroller | 35.1 | 83.8 | 54.9 |
| Jun: Senate | 38.4 | 83.8 | 58.3 |

Senate runs 3–5pp more Democratic among white voters than everything else —
the Talarico effect. AG, Comptroller, Governor and the legislative generic
cluster within about 2pp of each other. 23–25% of Hispanic respondents are
undecided on AG and Comptroller, so those cells are noisy.

## Recommendations

1. **Baseline:** build 2024 RRC (Craddick v. Warford) and a 2024 judicial-mean
   D share by district on PlanH2316 from the cached 2024 VTD file, and test it
   against the presidential baseline in the 2022 backtest harness before the
   refit decision. Data is cached; this is one script.
2. **Texas crosstab instrument:** read the group shift from the mean of the
   TX-legislature generic, AG and Comptroller banners in the same release.
   Exclude Senate. Governor at most half weight.
3. **Shift form stands** (proposal §3a): Texas-now-by-race minus a Texas
   benchmark by race, centered so the statewide average lives in the
   environment dial. Two cycles of null backtests are not a refutation; the
   2018 failure was an RV-vs-LV mismatch in the instrument. Ship it shrunk
   toward zero with the correlated group-error layer carrying the uncertainty,
   and re-test on UT's October LV wave.
