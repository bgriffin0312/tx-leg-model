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

## Test 3 — the baseline backtest (2026-09-10, later the same day)

`src/collect_downballot_spatial.py` built RRC, judicial-mean and presidential
D shares by district for 2012→H358/S172, 2016→H2100/S172, 2020→H2316/S2168
and 2024→H2316/S2168 (99.9% of vote attached; the spatial presidential column
matches the repo's key-join file to 0.03pp). `scripts/baseline_backtest.py`
then fits the structural regression (baseline + incumbency + chamber + env)
on the contested races of two midterms and scores the third, for each
baseline on identical rows, σ = 0.0742 throughout.

Pooled fit, 259 contested races 2014+2018+2022:

| baseline | pass-through | sigma |
|---|---|---|
| presidential | 0.986 | 2.57pp |
| RRC | 1.067 | 2.95pp |
| judicial mean | 1.076 | 2.89pp |
| blend (½ pres + ½ RRC) | 1.036 | 2.52pp |

Held out, one cycle at a time:

| held-out | metric | presidential | RRC | judicial | blend |
|---|---|---|---|---|---|
| 2014 | resid sd / MAE (pp) | **2.71** / 2.29 | 2.98 / 2.39 | 2.78 / 2.65 | 2.74 / **2.26** |
| 2014 | House seat error | **+1.5** | +1.8 | +3.4 | +1.6 |
| 2018 | resid sd / MAE (pp) | **2.70** / 5.42 | 3.43 / 3.62 | 3.28 / 5.60 | 2.75 / 4.38 |
| 2018 | House seat error | −6.2 | **−4.1** | +11.4 | −5.4 |
| 2018 | resid, 70%+ Hispanic | **+0.0** | +4.1 | +11.7 | +2.0 |
| 2022 | resid sd / MAE (pp) | 3.08 / 2.39 | 2.60 / 1.96 | 2.69 / 2.42 | **2.56** / **1.95** |
| 2022 | House seat error | +2.2 | **+1.0** | −1.5 | +1.6 |
| 2022 | resid ~ Hispanic slope | −0.085 | **−0.002** | −0.012 | −0.046 |
| 2022 | resid, 70%+ Hispanic (n=10) | −3.8 | **−0.6** | −2.5 | −2.3 |
| avg | resid sd / Brier | 2.83 / .0375 | 3.00 / **.0355** | 2.92 / .0462 | **2.69** / .0354 |
| avg | abs House seat error | 3.3 | **2.3** | 5.5 | 2.9 |

Reading:

- **Judicial mean is out.** Worst or near-worst in every cycle. Two causes:
  Democrats fielded candidates in only two 2012 judicial races, and in the
  straight-ticket era the judicial D share in South Texas was inflated
  (2018 residual +11.7pp in 70%+ Hispanic seats).
- **RRC wins the only post-straight-ticket cycle outright** — 2022 residual
  sd 2.60 vs 3.08, seat error 1.0 vs 2.2, and the Hispanic-correlated
  residual that the −0.05 constant was invented to cancel goes from −0.085 to
  −0.002. It loses on dispersion in 2014 and 2018 (straight-ticket era) while
  still beating the presidential baseline on seat error in 2018.
- **The blend is the best average** on residual sd and Brier and is never
  worse than second. It halves the 2022 Hispanic slope rather than removing it.
- Pass-through is ~1.0 for every baseline on clean cycles. The 0.596 in
  master is the mismatched-geography artefact the refit branch already
  identified.

## Test 3b — open races only (Brennan's rule, same day)

"Statewide races only make sense as baselines if they have no incumbents."
The Capitol Data returns flag incumbents per candidate, so the collector now
tags every race and writes `open_mean_d2p`: the mean of the RRC / Supreme
Court / CCA contests with no incumbent on either side.

| year | open downballot races | note |
|---|---|---|
| 2012 | RRC 1 (Craddick v Henry) | |
| 2016 | RRC 1 (Christian v Yarbrough), CCA 5 (Walker v Johnson) | |
| 2020 | RRC 1 (Wright v Castañeda) | the race that won the 2022 backtest |
| 2024 | CCA Presiding Judge, CCA 7, CCA 8 | all three incumbents lost their primaries; **RRC 2024 had Craddick, so it is out** |

TLC flags appointees inconsistently (Blacklock 2018 no, Bland 2020 yes); the
flag is used as given.

Backtest with the open composite added (same harness, same rows):

| held-out | metric | presidential | RRC | open | ½ pres + ½ open |
|---|---|---|---|---|---|
| 2018 | MAE (pp) / seat err | 5.42 / −6.2 | 3.62 / −4.1 | **3.43 / −3.8** | 4.30 / −5.3 |
| 2022 | resid sd / seat err | 3.08 / +2.2 | 2.60 / +1.0 | 2.59 / **+0.9** | **2.56** / +1.6 |
| 2022 | resid ~ Hispanic slope | −0.085 | −0.002 | −0.002 | −0.046 |
| avg | resid sd / Brier | 2.83 / .0375 | 3.00 / .0355 | 2.95 / .0351 | **2.66 / .0352** |
| avg | abs seat err / MAE | 3.3 / 3.36 | 2.3 / 2.66 | **2.2 / 2.59** | 2.9 / 2.83 |

The open composite is at least as good as RRC in every cycle and slightly
better on average, and the pres+open blend has the best average residual sd
and Brier of everything tried. The rule costs nothing in the backtest and
removes an incumbent (Craddick) from the 2026 instrument, so it is adopted:
**the 2024 baseline candidates are `open` (three CCA races) and `blend_open`.**

## What it does to 2026

`scripts/baseline_project_2026.py` (structural model, each baseline with its
own pooled coefficients, D+9.1, no WAR/finance/IE, no Monte Carlo): 59 House
seats predicted D on the presidential baseline, 65 on RRC or open, 62 on
either blend. Six seats cross 50% between presidential and open: HD 34, 35,
37, 41 and 118 move to D, HD 112 moves to R; the pres+open blend moves only
HD 34, 35 and 41. The RGV seats move +5 to +6pp; the Anglo suburban seats
(HD 70, 108, 138) move −0.7 to −1.7pp. Part of that spread is the open fit's
larger environment coefficient at D+9.1, not only the baseline.

## The AG race as the Texas generic ballot by race (Brennan's original question)

`scripts/ag_anchor_2026.py`. The national generic ballot by race, mean of
the five 2026 polls with a banner, is white 46.8 / Black 85.5 / Hispanic
62.5. The UT August AG banner (Middleton v Johnson, RV) is 37.7 / 81.7 /
51.3. Texas offset, Texas minus national: white −9.1, Black −3.8, Hispanic
−11.2. **Relative to white voters the Hispanic offset is only −2.1pp**, and
+1.9 on the Jun+Aug mean of AG, Comptroller and legislative generic. So the
national-to-Texas gap is mostly uniform (Texas is redder), which belongs in
the environment dial and intercept, not in a race term. The −0.05 Hispanic
constant assumed a Hispanic-specific gap; the 2026 banners do not show one.

Where the AG anchor bites is the *trend*. National banners imply Hispanic
voters have moved +8.5pp toward Democrats since 2024 (Catalist 54 → 62.5);
the Texas AG banner puts Texas Hispanics at 51.3 against a 2024 Texas
benchmark of 50.6, a move of +0.7. Texas whites moved +6.3 (31.4 → 37.7).
Centered on the validated 2024 Texas electorate, the shift term is white
+2.3 / Black −7.3 / Hispanic −3.2, which gives a 90% Hispanic district
−2.6pp relative to uniform swing and a 15% Hispanic district +1.5pp. On the
smoothed Jun+Aug downballot mean the same term is −0.4 / +0.8: close to
nothing. Either way the answer to "correct the national Hispanic lean with
the AG race" is: do not apply the national Hispanic rebound to Texas. The
Texas banners say Texas Hispanics are where they were in 2024.

Caveats: RV banners with ~24% of Hispanics undecided; Hispanic cell n≈350;
June and August AG Hispanic readings differ by 5pp, which is the noise band;
the 2024 Black benchmark is a national assumption. Use the mean of AG,
Comptroller and legislative generic across the last two releases, not one
race from one wave, and shrink toward zero.

Running the *full* model with `TXLEG_BASELINE=rrc` under master's current
coefficients goes the other way (D+9.1: 66.5 → 65.1 expected House seats,
P(majority) 11.9% → 7.8%), because a 0.596 pass-through damps the South
Texas gain while the statewide −1.3pp shift of the downballot vote hits every
seat. **The baseline switch and the coefficient refit are one decision.**
`model_config.BASELINE_SOURCE` (pres | rrc | blend; env override
`TXLEG_BASELINE`) is wired and defaults to pres.

## Recommendations

1. **Baseline:** take the decision with the refit. On the refit branch, run
   `TXLEG_BASELINE=rrc` and `=blend` and compare the competitive list to
   the presidential run. Preference from the evidence: blend if you weight
   all three cycles, RRC if you weight the post-straight-ticket regime, which
   is the one 2026 will be run under. Judicial mean is not a candidate.
2. **Texas crosstab instrument:** read the group shift from the mean of the
   TX-legislature generic, AG and Comptroller banners in the same release.
   Exclude Senate. Governor at most half weight.
3. **Shift form stands** (proposal §3a): Texas-now-by-race minus a Texas
   benchmark by race, centered so the statewide average lives in the
   environment dial. Two cycles of null backtests are not a refutation; the
   2018 failure was an RV-vs-LV mismatch in the instrument. Ship it shrunk
   toward zero with the correlated group-error layer carrying the uncertainty,
   and re-test on UT's October LV wave.
