# Texas whole-election model (built 2026-10-31)

SNAPSHOT as of 2026-10-31 — forecast numbers below are not current after that date.

One electorate per simulation across 8 statewide races, 38 U.S. House seats, 150 Texas House
seats, 16 Texas Senate seats and 10 county races. It replaces the separate error structures of
the statewide, legislative and county models. Each race's mean still comes from its own layer.

```
python scripts/whole_election_estimates.py      # roll-off, splitting, swing tests from TLC VTDs
python src/whole_election_model.py              # both versions, sensitivity
python src/whole_election_model.py --query "us_senate & !tarrant" --env statewide
python src/whole_election_model.py --query "tarrant" --given "us_senate"
python scripts/whole_election_backtest.py      # 2018, 2022, 2024; --env poll|538|rcp|actual; --probe
```

Output: `output/whole_election_<env>_2026.csv` and `.npz`, `whole_election_races_2026.csv`,
`whole_election_summary_2026.csv`, `whole_election_estimates.json`,
`whole_election_rolloff.csv`, `whole_election_swing.csv`.

## Structure

    margin = T + place + incomplete ballots + splitters

- **T:** one Texas party vote per run, the statewide posterior draws, so poll information stays.
- **Place:** one normal draw per county and per House district. Every race reads them through
  its 2024 presidential vote weights, so overlapping races share a local electorate. The place
  share of a race's own error is 0.60: 0.16 county-wide, 0.44 local. Race error sizes are kept.
- **Incomplete ballots:** a mean by office from 2022, plus one cycle draw of the D-R drop-off gap.
- **Splitters:** what remains of the race's gap to T, including candidate effects.
- **Shared by office type:** U.S. House offset c as fitted, Texas House and Senate 1.65, county 1.7.
- **Hispanic slope:** one draw. Legislative and county races take its deviation from the mean.

A race's gap to T is fixed, so the versions differ only in where T centers: statewide R+4.8,
legislative R+7.2. The statewide version moves legislative and county races by 0.95 per point of
T. The legislative version moves statewide and U.S. House races by 1.0 per point, against their
polls, so it says those polls miss by 2.4 more than the statewide model expects.

## Measured

All from TLC VTD returns for the same precincts. Total votes in a race cannot change through
splitting, so drop-off separates from splitting on aggregate data.

**One electorate.** Same seat, consecutive cycles, against the judicial vote on its own VTDs:
slope 0.95 (se 0.02, n 344), House 0.94, U.S. House 0.99; between cycles 0.96 (se 0.06, n 5);
within cycle 0.90 (se 0.04). County judges against the county judicial vote: 0.96 (se 0.09, n 20),
within cycle 0.83 (se 0.24). The legislative formula's 0.0034 is 0.68 margin per national point.

**Overlapping offices.** U.S. House and Texas House changes in the same county and cycle
correlate 0.45 (n 56). After each race's own precinct party vote, 0.10. So place carries across
offices and what remains looks race-specific.

**Incomplete ballots, 2022.** Roll-off 0.9–1.9% statewide, 1.4–2.1% district. It falls on D
top-of-ticket voters (Railroad Commissioner 2.3% vs 0.5% R). Margin effect about −0.8 against the
top of the ticket, nearly the same at every office below it, so about 1 point at most between any two races.
2024 ran about double. Before 2022 straight-ticket voting makes rates a poor guide.

**Competitive seats.** House seats within 20 ran −1.1, +2.4, +4.1, +1.8, 0.0 against the state
in five cycle pairs: level sd 1.65, on top of T.

## Forecast

| | Statewide env | Legislative env |
|---|---:|---:|
| T | R+4.8 | R+7.2 |
| U.S. Senate | D+2.1, 75% | R+0.3, 47% |
| Governor | R+3.4, 14% | R+5.8, 3% |
| Railroad Commissioner | R+2.4, 23% | R+4.9, 8% |
| Tarrant | D+1.8, 63% | R+0.5, 46% |
| Williamson | D+7.7, 92% | D+5.4, 84% |
| Collin | R+4.2, 22% | R+6.5, 12% |
| Denton | R+6.5, 12% | R+8.8, 6% |
| Fort Bend | D+7.8, 98% | D+6.7, 95% |
| Harris | D+10.2, 99% | D+9.0, 99% |
| Texas House D seats | 68.7 | 66.0 |
| P(76+) | 8.2% | 2.0% |
| Texas Senate D seats, P(16+) | 11.1, 0% | 11.0, 0% |
| U.S. House D seats | 11.8 | 10.9 |
| Talarico + Simmons | 51.7% | 27.0% |
| Talarico, Simmons loses | 23.6% | 19.6% |
| Paxton, Simmons wins | 11.4% | 19.3% |
| Both lose | 13.3% | 34.2% |
| Talarico + Abbott | 61.1% | 43.2% |
| Talarico + Hinojosa | 14.1% | 3.3% |
| Paxton + Hinojosa | 0.1% | 0.1% |

Senate minus Governor is 5.5 in both versions, a net 2.8% of voters. Both races sit at the top
of the ballot, so that gap is all splitters. Only net crossing is identified; gross is larger.

With the legislative shared error as shipped (6.7), the legislative version returns 67.2 seats
and 15.1%, the published 67.1 and 14.9%. The statewide version then gives 70.0 and 24.2%.
So the chamber odds depend mostly on the shared error: 1.65 on top of T is measured on five
cycle pairs, 6.7 is the shipped 58% variance split.

## Limits

- Aggregate returns, not ballots. Roll-off by party is ecological.
- County drop-off rates are Harris 2022 alone.
- White, Black and other group shocks reach legislative seats only.
- Race gaps are fixed within a version, so a Senate poll miss does not move Paxton's gap.
- The backtest below covers three elections, none near a D House majority, so it does not test P(76+) directly.
- 2012–2020 pairs had straight-ticket voting. Only 2022–2024 did not.
- `T_LEGISLATIVE` is -7.2; the same reading computed directly is -7.54. The 0.3 is left in.

## Backtest (2026-10-31)

`scripts/whole_election_backtest.py`. 2018, 2022 and 2024, each left out of the statewide,
U.S. House, T prior, poll miss, legislative, county seat, slope and place fits. Environment is
the final generic ballot: FiveThirtyEight (+8.6, -1.2, +0.7, Internet Archive copy of its daily
averages) and RealClearPolitics (+7.3, -2.5, -0.3), averaged. Saved in
`data/raw/statewide/generic_ballot_finals.csv`. Using either alone, or the national House vote,
moves no House mean by more than a seat. Not held out: legislative error sizes (2024 is clean).
No district polls exist for these years. 2024 is a midterm formula on a presidential year, with
no viability or IE data. County races skip the record and money terms. The formula has only 8
2014 rows, so its held-out 2018 fit is close to a 2022-only fit and the reverse. The 0.0034 slope
is not held out; it comes from those same two cycles, 2018 included.

| D seats, mean (10–90) | 2018 | 2022 | 2024 |
|---|---:|---:|---:|
| Texas House, actual | 67 | 64 | 62 |
| Statewide version | 62.0 (57–67) | 63.3 (60–66) | 64.3 (61–67) |
| Legislative version | 63.9 (59–70) | 63.5 (60–66) | 64.4 (62–67) |
| Texas Senate seats up, actual | 5 | 12 | 7 |
| Statewide version | 3.7 (3–5) | 11.4 (10–12) | 7.9 (7–8) |
| U.S. House, actual | 13 | 13 | 13 |
| Statewide version | 14.4 (12–17) | 12.6 (11–13) | 12.6 (11–14) |

All 18 chamber bands held the result. Seat means ran low in the 2018 wave and high in 2024.
The two versions were close each year: the legislative center ran 2.1, 0.3 and 0.4 to the left
of the statewide T, so only 2018 separates them, and 2026, at 2.4 to the right, is the first
time it is the redder one. The backtest cannot pick between them.

Statewide races: mean miss -5.4, +0.9, +1.8 in the statewide version. 2018 is the polls.
Abbott and Cruz fell outside the 80% band on the blue side (15 of 17 inside overall). T against
the judicial vote: -2.0, +1.5, +6.0, with a stated sd of 3.0.

Races with P(D) from 10% to 90% (56): mean P 48%, D won 55%, Brier 0.20, 75% inside the 80% band.
Legislative version: 60 races, 47%, 50%, 0.16, 80%. Competitive-race wins fell inside the band
each year: 17 of 28 (expected 11.2), 10 of 17 (8.9), 4 of 11 (6.5).

Joint checks. How often pairs of competitive races split: real 0.50, 0.52, 0.51; model 0.43,
0.46, 0.43; each is inside the simulated range. Pairs barely distinguish the joint model from
independence (Brier differences under 0.03, and they go both ways). Overlap between Texas House
and U.S. House residuals sharing voters runs below the model all three years (0.13, 0.02, 0.11
against 0.23, 0.26, 0.26; 2022 at the 1st percentile), so the place share of 0.6 looks too high
by perhaps three times. Cutting it to 0.3 of its size moves no 2026 seat mean or headline
combination by more than 0.3 points, so it stays.

Shared legislative error. Mean log score of the Texas House count, statewide version: -2.44 with
T + 1.65, -2.67 with the shipped 6.7. Legislative version: -2.32, -2.64. Over three elections
that is about 3:1 for the smaller error: weak. It does not settle 8% against 24%.

Two bugs found on the way. `electorate()` read House districts from race rows, and an uncontested
race has none, so 41 of the 133 contested 2026 seats had no place draw. It now reads TLC's
precinct file, which changed the forecast by 0.1 seat. An early backtest pass carried the same
gap into its weights and showed a false 4–8 point gap between the legislative center and the
judicial vote. With complete weights that gap is 2.9, 0.7, 2.7 and 4.5 (2014–2024).

## Why 2018 ran low (2026-11-01)

`scripts/wave_2018/`. With T fixed at the 2018 judicial vote the model gives 63.7 seats against
67, so polls are not the whole miss. It missed six Republican incumbents it had 0.2 to 5.2 points
behind (HD 132, 52, 65, 135, 45, 47) and wrongly took HD 134. The all-seat level was right within
0.4. Republican-held seats within 30 points ran 3.7 left of the model, at the 95th percentile of
its range, while safe D seats ran 5 red. Those R seats ran 0.7 behind the local judicial vote, and
that vote ran 4.0 left of the formula, so this was a local partisan swing, not candidates.

| Idea | Test | Result |
|---|---|---|
| Swing on a logit scale | Refit, held out | R-held close miss 3.17 to 3.24: no help |
| Coattails | House minus J on top of ticket minus J | Slope -0.16 (se 0.20) 2018, 0.07 2022, 0.03 2024 |
| Wave lands on college whites | J swing by district, six cycle pairs | Fixed midterm effect: +29, +27, +20 in midterms of both directions, 0 or negative in presidential years. Already in the formula |
| Elasticity by lean | Same pairs | Slope -0.0075 (se 0.0062), and the wrong shape: the hump sits near the state mean |
| Split slope, D incumbents 0.43, others 0.81 | Backtest | 2018 legislative 63.9 to 66.1, but 2024 no gain, and same-seat swings outside 2018 give equal slopes (1.78, 1.80): the R-minus-D gap is a constant 2 to 7 in all five pairs. Not adopted |
| Higher uniform slope | Backtest, 0.80 and 0.94 | Legislative center passes J in 2018 (-3.3 against -7.3 at 0.94); statewide scores fall. Not adopted |

Straight-ticket voting: races followed the local judicial swing at 0.91 in 2016-18 and 0.97 in
2022-24, with scatter 3.9 and 3.5. One pair after it ended, so thin, but no sign the wave is
weaker without it. In 2018 the judicial vote itself contains straight tickets, so turnout and
persuasion cannot be separated.

Replay of 2018's band pattern on 2026 R-held and open seats, center unchanged, as a conditional
rather than a forecast: statewide version 73.2 seats and 31% for 76+, legislative 69.4 and 12%.
Half of it: 70.7 and 17%, 67.5 and 5%. Terrain: 5 such seats sit 0-6 points behind against 12 in
2018, and 31 sit 6-20 behind against 16.

The model is left as is. A mean-zero error on the 51 R-held and open seats within 30 points, sd
2.5, moves P(76+) to 12.5% and 4.4% with seat means up 0.3. Its seat log score is level with no
such error (-2.45 and -2.44 statewide, -2.36 and -2.32 legislative), against -2.67 and -2.65 for
the uniform 6.7, so it is the better way to state wave risk, but the backtest does not ask for it.

## What the early vote says (2026-11-01)

`scripts/early_vote/`, with data in `data/raw/early_vote/`. Early voting ended Oct 30 with about
7.75 million votes by the tracker and Tribune counts; ColdSpark's feed holds 7.42 million. None of
this changes a model file. Sources are ColdSpark (modeled party, 2022, 2024, 2026, county; precinct
for 2026 only), theoelections (primary history, 2020 to 2026), Derek Ryan's voter-file reports
(primary history, 2016 to 2026, `data/raw/early_vote/ryan/`), and the Secretary of State's
top-county tables for 2012 to 2018. Its 2020 and later pages refuse fetches, as do theoelections'
data files. Ryan's county reports give primary history by county for 2022, 2024 and 2026, and have
not been tested against county swing.

**Statewide.** Early net (R minus D, all early votes): 9.7 in 2022, 10.5 in 2024, 3.2 in 2026. The
judicial vote ran 4.3 and 5.9 left of that net's mirror image. Primary history gap (R only minus D
only): 5.6 in 2020, 9.0 in 2022, 7.0 in 2024, 1.6 in 2026. Readings of the 2026 judicial vote:

| Reading | 2022 offset | Other years |
|---|---|---|
| Modeled party, as published | -7.5 | -9.1 (2024) |
| Modeled party, turnout pattern only | -6.3 to -8.4 | none |
| Primary history | -6.6 | -5.6 (2020), -11.0 (2024) |
| Primary history, Ryan's file, no-primary margin held | -6.5 to -10.1 | 2.7 to -0.1 (2018), -2.4 to -5.7 (2016), -6.0 to -9.5 (2020), -8.3 to -12.0 (2024) |

The turnout-pattern row applies 2022's turnout rates by party and history to the 2026 file. That
gives R+10.9, so labels moved the state only 1.1 (border 14.6, elsewhere -0.1), and the 2026 pattern
took 7.7 off it. The lower figure, 5.6, treats the excess low-history D votes (about 155,000) as a
label effect, since a 2026 primary vote can create the label.

The 2022 and 2024 offsets in the other rows assume those years' persuasion. Only 2022 and 2024 have
one definition for modeled party, and the two differ by 1.6 to 4.4 points. Ryan's file adds 2018
and shows why that matters: the two midterms read about 9 points apart. Read the early vote as a
range of about -10 to +3, not as a confirmation of -7.

**Primary voters.** `ryan_history.py`, on Ryan's definition (R group minus D group, labels from the
last four cycles on each year's file; primary voters split by most recent primary):

| | 2016 | 2018 | 2020 | 2022 | 2024 | 2026 early |
|---|---|---|---|---|---|---|
| Votes (millions) | 9.03 | 8.39 | 11.31 | 8.10 | 11.32 | 7.73 |
| R minus D (votes) | +1,491,555 | +1,162,255 | +590,843 | +875,478 | +1,043,258 | -25,836 |
| Gap | R+16.5 | R+13.9 | R+5.2 | R+10.8 | R+9.2 | D+0.3 |
| No-primary share | 48.4 | 45.9 | 50.1 | 37.8 | 52.8 | 36.0 |
| Top of ticket | -9.0 | -2.6 | -5.6 | -11.0 | -13.9 | |
| Judicial | -15.5 | -7.3 | -9.4 | -14.0 | -16.4 | |
| No-primary margin, top | +12.8 | +22.1 | -1.6 | -2.9 | -10.2 | |
| No-primary margin, judicial | -0.6 | +11.9 | -9.2 | -10.6 | -14.9 | |

No-primary margins carry 0.92 of the primary gap to the vote; 0.85 to 1.0 moves them about 2 to 3
points. The gap does not predict the result across these five (correlation 0.1 with the top of the
ticket, -0.18 with the judicial vote), and 2018 had a larger R gap than 2022 with a result 8.4 points
bluer at the top. On the same 2018 ballot no-primary voters ran about D+22 for Senate and even for
Governor (Ryan's own calculation gives Abbott about 50%), so the margin is a candidate result as
much as a group trait.

Ryan's 2018 file is final only: he tracked 20 counties that cycle, so no statewide same-point 2018
figure exists. Top 15 counties: R+90,000 through Day 5 of 2018 (Tribune), against D+416,675 at Day 5
of 2026 and D+663,068 at Day 12. O'Rourke won those 15 by 15.4. The 2018 early totals on his old site
are gone (ryandata.com returns 404 on every page as of 11/2); a search excerpt gives R 1.2 million,
D 1.0 million early, universe not stated, which fits the top 15. NBC's "400,000 at the same point in
the last midterm" is 2022 (R+385,396 at Day 7).

2026 final, by Election Day conversion: R+2.4 at 2022's conversion by primary tier, R+3.1 to R+3.3 at
2018's final participation, and Ryan's own guess is about R+300,000. 2024's multipliers give D+0.5
but ignore the pools left, so that figure is discarded. theoelections' definition gives R+0.7 to
R+2.4. No-primary voters are about 41% of the electorate on Ryan's window (45 to 47% on
theoelections'), and more if their 2018-style turnout returns (45%).

The Democratic number is partly relabeling. In March, 37% of 2.31 million Democratic primary voters
and 19% of 2.17 million Republican ones had no prior primary, a net 444,000 moved to the D label,
against 64,000 in 2018 (27.0% and 14.1%). Ryan puts 770,000 of the Democratic group in the general-only
pool before March and says most were likely already voting for Democrats. Taking out the excess
(about 342,000 voters at 90% turnout) makes the gap R+5.9 at 2022's conversion. That is an upper
bound, since some of it is new participation: 546,000 Democratic primary voters skipped November
2022, against 262,000 Republican.

Holding the no-primary margin at a past year, gap R+2.4 and labels as they stand: 2022's gives -3.4
at the top and -6.5 judicial, 2018's gives +6.9 and +2.7. With the relabeling out: -6.7 and -10.1,
+4.4 and -0.1. The bars for the other reads, labels as they stand then relabeling out: Senate D+0.9
needs no-primary voters at D+7 to D+10 or D+13 to D+17, Governor R+4.6 at R+3 to R+6 or D+1 to D+5,
and T -6.0 at R+7 to R+10 or R+2 to D+2. Past values: top R+3 in 2022, D+22 for Senate in 2018;
judicial R+11 in 2022, D+12 in 2018.

Participation by tier now has past-year series. At the end of early voting 87% of 4R and 93% of 4D had
voted, and 79% of 3R and 86% of 3D. Final 2022: 97, 96, 91, 88. Final 2018: 98, 95, 92, 90. Not voted:
1.11 million R group (90,244 4R), 818,641 D group (29,558 4D). Ryan's likely-voter pool left is R+289,717.

`primary_pool.py` keeps the theoelections arithmetic and the Transparency USA pool (a search excerpt,
63% of R-only and 69% of D-only voted).

Election Day cannot be read off the early vote. Early and Election Day nets matched in 2022 (9.7,
9.9) and split in 2024 (10.5, 0.5). 2022 Election Day voters who have not voted yet run R+14; 2024's
run R+4. With 2.6 million on the day, the final electorate is about R+4 to R+6; ColdSpark projects
R+6.8 (3.9 to 7.4). Its projection is of party mix, not of the vote.

**County.** 2022 to 2024, leave one county out, judicial swing: error 2.70 to 1.66 in the top 60
without the border (slope -0.48), 4.23 to 2.86 in all counties, 4.90 to 4.34 outside the top 60.
It survives controls for the prior swing. One transition. A levels version that needs no prior
early vote, the only kind precinct data would allow, gains almost nothing, so there is no
precinct-level signal.

Turnout needs no labels. Relative early growth moved with the judicial swing in all four pairs
(11, 15, 13 and 25 margin points per log unit, 2012-16, 2014-18, 2018-22, 2022-24; pooled 12.2,
se 7.7 for the first three). It is weak but on the same side each time.

**Seats.** The model's own House residuals with T known: 2024 early net slope 0.55 (county-clustered
se 0.22), error down 5% without the border, nothing for R-held close seats. Turnout slope 17.9, 16.7
and 33.4 in 2018, 2022 and 2024, error down 2% or less. Early turnout would not have flagged the
2018 run in R-held seats (slope 3.8, se 11.0).

**The border.** Label drift is largest there (Hidalgo +18.8 from labels alone), so modeled party is
not usable. Early growth relative to the state is highest there (Hidalgo +.16, El Paso +.15, Webb
+.13, Cameron +.09), which leans D if the turnout pattern holds, and matches the UT poll's Valley
read, but the border slope is noisy (17.6, se 11.6).

**2026, whole-election model, early-vote read.** T center -6.0, the poll center and an early reading
near -7 weighted by precision, plus county shifts at 0.5 per point, averaging the 2024-based and
label-corrected signals, in the 54 largest non-border counties:

| Read | Senate | Governor | TX House | P(76+) | TX Senate | U.S. House |
|---|---|---|---|---|---|---|
| Statewide version | D+2.1, 75% | R+3.4, 14% | 68.7 | 8.2% | 11.1 | 11.8 |
| Legislative version | R+0.3, 47% | R+5.8, 3% | 66.0 | 2.0% | 11.0 | 10.9 |
| T -6.0 | D+0.9, 62% | R+4.6, 7% | 67.3 | 4.2% | 11.1 | 11.3 |
| T -6.0 with county shifts | same | same | 67.8 | 5.4% | 11.1 | 11.3 |

Adding the 2.5 mean-zero error on 51 R-held and open seats takes the last row to 68.1 and 9.3%.
T -8 gives Senate 37%, House 65.2; T -3.5 gives Senate 86%, House 70.4. Only Tarrant agrees across
all three county baselines (+1.5 after centering): county judge 55% to 65%, HD 94 25% to 33%, HD 97
19% to 27%, HD 96 13% to 19%. Harris moves +1.3 but the baselines split (0.2 and 2.4), so HD 144, 133
and 138 are weaker.

## A swing bigger than 2018, focused by the polls (2026-11-02)

`scripts/early_vote/focus_2026.py`. A conditional, not a forecast. The judicial vote swung 8.2 from
2016 to 2018, so one point more on 2024 is T -7.15, right of the -6.0 read: the model already assumes
a swing about 2 points bigger than 2018's. Midterm to midterm, 2014 to 2018 was 16.1, so one point more
on 2022 is T +3.1. The focus is a zero-sum race pattern on legislative and county races. Statewide races
do not move by construction, and U.S. House races are left alone because they condition on their own
polls. UT is the same-pollster delta from Oct 2024 (white +0.7, Black -3.4, Hispanic +5.1). All polls is
every LV crosstab since September against UT Oct 2024 (white -0.2, Black +4.4, Hispanic +11.6). 200,000
draws.

| T | Senate | Governor | Where | TX House | P(76+) | TX Senate | U.S. House |
|---|---|---|---|---|---|---|---|
| -7.15 | R+0.2, 47% | R+5.7, 3.5% | even | 66.0 | 2.1% | 11.0 | 10.9 |
| | | | UT | 66.8 | 1.5% | 11.0 | 10.9 |
| | | | all polls | 66.6 | 0.5% | 11.0 | 10.9 |
| -6.0 | D+0.9, 62% | R+4.6, 7% | even | 67.3 | 4.2% | 11.1 | 11.3 |
| | | | UT | 67.8 | 3.1% | 11.1 | 11.3 |
| | | | all polls | 67.3 | 1.1% | 11.0 | 11.3 |
| +3.1 | D+10.0, 99.9% | D+4.5, 92% | even | 80.8 | 77% | 12.1 | 16.0 |
| | | | UT | 79.5 | 71% | 11.9 | 16.0 |
| | | | all polls | 76.8 | 54% | 11.6 | 16.0 |

Focus adds at most a seat and lowers P(76+) in every row, because the Hispanic pattern takes from the
suburban seats the majority runs through. At T -6.3, even to UT: HD 37 30% to 59%, HD 34 66% to 86%, HD 118
67% to 82%, HD 112 57% to 46%, HD 108 41% to 34%, HD 63 31% to 24%, Tarrant county judge 53% to 43%.

The polls disagree on the place. Eight polled congressional districts swung 11 to 22 from the 2024 judicial
vote with no Hispanic slope (-0.18 per unit, partisan polls discounted 2). UT, same pollster, by place:
urban -5.1, suburban +5.0, rural -0.2. By education: college about +3.8, non-college about -1.7, on
subgroups of 100 to 200. The formula's midterm white-college term is 0.16 two-party, about 32 margin per
unit of share, against about 11 for UT's gap, so the model already holds more than that crosstab shows, and
an education pattern on top counts some of it twice. That run did not finish, so education is not in the
table.

## After Nov 3

Score both versions and the early-vote read, and add 2026 as a third point for each statewide
offset. Re-run the estimates with 2026 as the second cycle pair without straight
tickets and the second midterm for roll-off.
