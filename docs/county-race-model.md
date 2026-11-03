# Texas county race model (built 2026-10-30)

SNAPSHOT as of 2026-10-30 — forecast numbers below are not current after that date.

Nine county judge races and the Bexar County DA. Each county is priced as if it were a
legislative district, because almost none of these races has a public poll.

```
python src/county_judge_collect.py   # OpenElections county files, 2014/2018/2022
python src/county_judge_model.py     # backtest, forecast, sensitivity
```

`data/raw/county_judge/` is gitignored. Output: `output/county_judge_forecast_2026.csv`.

## Structure

The shipped formula from `model_config.py`, unchanged: open-race composite of the prior
presidential year × 1.05, differenced environment, white-college term, incumbency. Then:

- **Seat and money factors**, from the nine forecast counties, replacing the legislative
  incumbency values. See "Nine-county factors" below.
- **Candidate record:** residual in earlier county judge races in that county × n/(n+2.93).
  Same candidate, next race: slope 0.52 (se 0.13, n 15), which supports the legislative form.
- **Polls** only where a public head-to-head exists: Harris (UH D+7) and Fort Bend (TSU D+9),
  at the 6-point noise used for U.S. House district polls. Campaign polls get the −4 haircut;
  the mid-August HIT poll for Plummer is outside the window and is not used.
- **Bexar DA** uses Chapa vs Foster plus an assumed share for independent Jason Wolff
  (mean 15%, 55–90% of it from the Republican). No data sets those two numbers.

Market prices are never an input. They were compared only after the 18:40 CDT freeze.

## Backtest

39 contested county judge races in 18 large counties, on the three midterms
the formula was fit on. 33 outside the border counties: mean miss −0.5, sd 5.4 margin.
Cycle means +1.5, +0.4, −2.4.

| Group | n | Mean miss | sd |
|---|---|---:|---:|
| Outside the border | 33 | −0.5 | 5.4 |
| Border (Cameron, Hidalgo, Nueces) | 6 | −6.2 | 8.8 |
| R officeholder | 17 | −2.8 | 4.5 |
| D officeholder | 12 | −4.4 | 7.1 |
| Open | 10 | +4.8 | 2.8 |

Leave-one-cycle-out rmse in share points, outside the border: formula 2.67, +constant 2.82,
+R officeholder 2.88, +D and R officeholder 2.70, **+open seat 2.31**. Only the open-seat
term helps. All eight open cases outside the border run D of the formula, on both R-held and
D-held seats (of the two border cases, Nueces does and Hidalgo does not). Ten DA races, not used to find it, repeat it: open +2.3, held −6.3, rmse 6.6 → 5.5.
Five specifications were compared on 33 rows, so the size is less certain than the sign.

Border counties are not adjusted. El Paso is the only one forecast and it is safe.

## Nine-county factors (added 2026-10-30, Brennan)

Fit on Tarrant, Williamson, Collin, Denton, Fort Bend, Harris, Bexar and Montgomery. El Paso
has no D-vs-R history. The legislative incumbency values are removed from the formula first.

**Incumbency.** n=18: an R-held seat runs 5.6 more Republican than other seats (se 2.0,
permutation p 0.03 against open seats). Out-of-sample rmse, by year and by county: constant
4.89 / 5.48, R seat 4.43 / 4.79, held vs open 4.71 / 5.28, separate D and R 5.59 / 5.75. The
"no effect for D officeholders" reading is Nelson Wolff: without Bexar, D seats average −0.7
(n 2), like R seats at −0.4. So a D-seat term is not separately established. D and open seats
share a level, which only matters for El Paso, and it is safe.

**Fundraising.** `data/raw/county_judge/finance.csv`, 54 candidate-years: 33 full, 5 partial,
3 news, 13 none. Line 2 of the cover sheet, July 1 of the prior year through the 8-day report,
loans excluded, log(D+25k / R+25k) clipped at ±3.5 as in the statewide and House layers.
Ten races have both sides. Fit with the seat term: **+1.98 margin per log unit (se 1.25),
permutation p 0.13**, R seat −4.1 (se 4.0), other seats +4.6. It sits near the U.S. House
(+1.08) and statewide (+1.54) values, but it does not pass in these counties: out-of-sample
rmse rises (R seat 7.72 / 7.06, R seat + money 7.60 / 10.85), and without Bexar the
coefficient is −3.3. It is applied because Brennan asked for it and the other race types
agree on sign and size, not because this sample proves it. It may also partly reflect who
donors expect to win.

Calls: Wong's $1.505M in loans is counted, because excluding it scores a 12-to-1 gap. Keough
v Graf and the Bexar DA have no usable pair and get no money term. Brown's $0 and Pappas's
$2,325 are partial. Gravell 2022 and Snell 2026 are floors or unreconciled. The Williamson 2018
row is news figures for part of the window. Missing pairs that would add most: Tarrant 2018,
Fort Bend 2014 and 2018, Collin 2018 and 2022, Denton 2018, Montgomery 2018, Williamson 2014.

## Error sizes

County 5.1 margin, from the formula's within-cycle spread. With the open term that spread
is 4.1; the larger value is kept because the term rests on eight cases and D+9.2 is outside
the formula's training range. Shared 3.4: cycle miss 1.7, statewide posterior on Texas 2.7,
half the 2.5-point gap between the two models 1.25. The backtest knew each year's
environment, so its error alone would be too small. The shipped legislative national sd
(6.7) is a sensitivity case.

## Frozen forecast, 18:40 CDT 2026-10-30

Margin is D−R. "Statewide env" adds 2.5 for the statewide model's environment.

| Race | Matchup (D v R) | Margin | P(D) | Statewide env |
|---|---|---:|---:|---:|
| Tarrant | Simmons v O'Hare (inc) | −1.4 | 40% | 58% |
| Williamson | Jefts v Snell (appointed) | +4.0 | 76% | 87% |
| Collin | Brown v Hill (inc) | −5.3 | 18% | 31% |
| Denton | Pappas v Eads (inc) | −7.0 | 11% | 21% |
| Fort Bend | McCoy v Wong (acting) | +6.6 | 94% | 97% |
| Harris | Plummer v Sanchez (open) | +9.5 | 98% | 99% |
| Bexar judge | Nirenberg v Von Dohlen (open) | +19.7 | >99% | >99% |
| Bexar DA | Chapa v Foster v Wolff (I) | +19.7 | >99% | >99% |
| El Paso | Samaniego (inc) v Torres Shelton | +22.7 | >99% | >99% |
| Montgomery | Graf v Keough (inc) | −43.6 | <1% | <1% |

## Current forecast, with the nine-county factors

Replaces the frozen table. It was built after the market table had been seen, so it is not
a blind forecast. "No money" uses the n=18 seat term alone.

| Race | Margin | P(D) | No money | Statewide env |
|---|---:|---:|---:|---:|
| Tarrant | −0.5 | 47% | 49% | 64% |
| Williamson | +5.4 | 83% | 77% | 91% |
| Collin | −6.5 | 13% | 28% | 24% |
| Denton | −8.8 | 7% | 15% | 14% |
| Fort Bend | +6.7 | 95% | 95% | 97% |
| Harris | +9.0 | 98% | 98% | 99% |
| Bexar judge | +21.9 | >99% | >99% | >99% |
| Bexar DA | +19.0 | >99% | >99% | >99% |
| El Paso | +26.2 | >99% | >99% | >99% |
| Montgomery | −42.2 | <1% | <1% | <1% |

Snell and Wong are scored as held seats. Scored as open, Williamson is +9.5 / 95% and Fort
Bend +8.7 / 98%. The one appointed case in the data, Faith Johnson (Dallas DA 2018),
behaved like a held seat at −6.6.

Two of the three midterms behind the formula had straight-ticket voting, which Texas ended
after 2018, so only 2022 looks like 2026. Candidate effects could be larger than the error
sizes here assume.

## Joint odds

`docs/whole-election-model.md` replaces this section's run for joint odds. This one paired
statewide races centered on R+4.8 with county races centered on R+7.2, so its Talarico and
Simmons odds were a mix of two environments.

`python scripts/joint_combos.py --build` writes `output/joint_outcomes_2026.npz` and
`output/joint_pairs_2026.csv`: 200,000 draws across 8 statewide races, 38 U.S. House seats
and these 10 races. County races take the statewide run's Texas environment draw plus 2.1
points of their own shared error, so each race's own odds match the county model to 0.1
points. Query with `python scripts/joint_combos.py "us_senate & !tarrant"`, or add
`--given "us_senate"`.

The correlations are model output and have not been backtested. County races do not carry
the Hispanic swing used in the House seats. The legislative model is not in this run, and no
ballot-level split-ticket or roll-off behavior is modeled. Rebuild after any forecast change.

## Coverage of the history

Found by hand, not from the files: Harris 2022, Hays 2018 and 2022 (the OE file is partial),
Denton 2022 (D derived from 59.4% R), Bexar 2014, Travis 2022 and Galveston 2022
(percentages only). No contested result located: Hidalgo 2022, Collin 2014, Montgomery 2022,
Galveston 2014. No D-vs-R contest: Tarrant 2014, Harris 2014, Denton 2014, Montgomery 2014,
Travis 2018, Galveston 2018, El Paso 2014, 2018 and 2022.

## Williamson Precinct 4 commissioner (2026-11-02, Brennan)

`scripts/county_commissioner_wilco.py`. Boles (R, inc) v Rhea (D), write-in Flores, the only
contested commissioner race: Precinct 2 is McDonald (R) unopposed and Precincts 1 and 3 are not up.
It is read off the county race, not simulated in the whole-election model. Precinct 4 is 25.5% of the
county and ran 3.6 right of it on the judicial vote in 2022, 2.8 in 2024, on 40 precincts. Before the
2021 lines it ran left of the county, so older offsets do not carry. Six Precinct 2 and 4 races, 2014 to
2022, ran 1.2 right of the judicial vote on their own precincts, spread 1.4; the error here is floored
at 3 because six races is thin. No finance is on file for either candidate.

Three means at T -6.0: the county race with the offset +3.1, Boles's 2022 margin plus the county's
swing +1.0, and the judicial vote directly +0.2. The direct one is the most grounded, since the county
judge mean sits about 2 points left of Williamson's judicial reading. Rhea's chance on the average of
the three is 52% at T -7.15, 61% at T -6.0, 70% at T -4.8, 98% at T +3.1, and 42%, 52%, 61%, 97% on the
direct one alone.

Court control turns on this race. Terry Cook (D, Precinct 1) and Valerie Covey (R, Precinct 3) are not up,
and McDonald (R) takes Precinct 2 unopposed, so Republicans hold two seats and Democrats one before the
two contested votes. Jefts and Rhea together make a 3-2 Democratic majority, from 4-1 Republican today:
51.5% at T -7.15, 60% at T -6.0, 69% at T -4.8, 98% at T +3.1, and 42%, 51%, 61%, 97% on the direct read.
Rhea rarely wins without Jefts, so those run within a point of Rhea's own chance.

## After Nov 3

Score all ten against results, with and without the money term, and test the seat term
again. Add the 2026 races to the pooled fit: eight more pairs would nearly double the money
sample.

Re-run the early-vote test with 2026 as a second transition, with `scripts/early_vote/` and the
ColdSpark snapshots through Oct 30 in `data/raw/early_vote/`. Results are in
`docs/whole-election-model.md`. Early vote is not a model input. Tarrant is the one forecast county
ahead of the state on all three baselines.
