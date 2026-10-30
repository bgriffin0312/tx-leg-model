# Next steps — tx-leg-model

*Written 2026-09-09 at the end of a full bug-and-code audit. Everything below is
either measured or explicitly flagged as unverified.*

*Published 2026-10-07: GitHub Pages (`19a0073`) and the Substack's Datawrapper
embeds (House v5, Senate v4, WAR v4, verified with `scripts/check_live_embeds.py`).
Publishing needed token scopes chart:read/write + theme:read + visualization:read.
Routine: `publish_charts.ps1` → `sync_datawrapper.py` → `check_live_embeds.py`.*

*2026-10-27: TEC 8-day reports and IEs are in (IE dollars corrected, see below),
and polls refreshed LV-only (3 releases, D+9.2). **66.2 House seats / 13.1%
majority, Senate 11.1 / 0.1%** — published 2026-10-27 (Pages `2304675`;
Datawrapper House v9, Senate v8, WAR v8, verified with check_live_embeds.py).
See "October 8-day refresh" below; the refit and model spec are in the
2026-10-07 sections.*

*2026-10-28: no model change. "Where the majority line sits" below records
how far right of the nation the model puts Texas, the 2018-analog fit, and one
open publishing decision.*

---

## Where the majority line sits, and the 2018 analog (2026-10-28)

Analysis only; the published model is unchanged. Scripts in `scripts/`, all
reproduced 2026-10-30.

**Threshold.** A 50% House majority needs about **D+18** nationally (D+12 →
69.1 seats / 20.8%, D+15 → 73.0 / 35.3%, D+18 → 77.3 / 50.4%, D+20 → 80.3 /
61.1%). Senate is 4.5% at D+18. `majority_threshold_bias.py`: with every seat
scored at the model's share (`sw_latent`), Texas reaches a 50/50 House vote at
D+18.4, where Democrats get 77.7 seats / 52% — so at 50/50 the map is close to
neutral in the model. The 13-point Texas lean is what keeps the majority far
away, not seat bias. (Earlier in the session I quoted "70 seats at 50/50 = 5
seats of bias"; that used the district-average convention, which counts turnout
differences, and was wrong as a bias measure.) The standard metrics in
`analysis/gerrymander/` still say the House is modestly R-tilted (EG +1.5 to
+4.4%) and the Senate twice that.

**How far right of the nation Texas runs.** `tx_house_lean_vs_nation.py`
(uncontested seats imputed from presidential + the contested-race gap): TX House
vote vs the national House vote 2012–2024 = 15.1, 16.9, 16.5, 16.1, 14.1, **8.8**
(2022), 12.9 points right. The model puts 2026 at about 17 right: the 2024 open
composite is R+16.7 (14.0 right of the nation's R+2.7) plus ~3 of midterm
adjustment. `statewide_lean_reproduction.py`: the fitted model reproduces each
training midterm's lean (2014 15.9 vs 16.9 measured, 2018 15.0 vs 16.1, 2022 8.4
vs 8.8), so 2022's small gap came from its R+9.5 2020 baseline, not from a
midterm Texas effect; anchoring 2022 on 2018 would have missed by 7.
`env_base_level_test.py`: absolute vs differenced environment fit the training
cycles equally (cycle-level RMSE 0.20 vs 0.16pp); the differenced spec would add
~+1.6 margin for 2026 (≈13% → 17%). Not identifiable from three cycles.

**The friend's D+12 argument.** Checked: 14 R-held seats are bluer than Trump's
+13.7 statewide (HD 34, 112, 118, 121, 108, 133, 52, 138, 94, 97, 63, 96, 37,
122) + 61 D-held = 75; HD 74 is 76th. The gap between D+12 and D+18 is ≈1 pt
generic vs presidential, 2–3 down-ballot vs Trump, ≈3 the model's 2024 base-year
treatment. Presidential→midterm moves (2012→14, 2016→18, 2020→22) put Texas at
50/50 around D+12–13 if you believe them, but 2016's House vote is mostly
imputed (52 of 150 contested); against fully observed composites 2018 ran ~2
points further from the nation than 2016, which is what the model assumes.

**2018 analog (Brennan: changing circumstances, not a cycle).**
`analog_2018_forecast.py` fits the shipped spec (with education) on 2018 alone,
same MC and seed:

| Env | Shipped | 2018 analog | 2018 + Hispanic term |
|---|---|---|---|
| D+9.2 | 66.0 / 12.0% | 66.0 / 14.3% | 65.8 / 14.0% |
| D+12 | 69.1 / 20.8% | 69.5 / 24.2% | 69.3 / 23.7% |
| D+15 | 72.9 / 34.3% | 73.7 / 38.7% | 73.4 / 38.1% |
| D+18 | 77.1 / 50.3% | 78.1 / 54.9% | 77.9 / 54.3% |

2018 changes which seats, not how many: bigger white-college suburban swing and
smaller incumbent cushions (dem_incumbent 0.0066 vs 0.0237 pooled) lift HD 108
37→54%, 94 23→36, 112, 118, 121 and drop HD 74 68→54, 144 61→47, 35 80→69.
With education in the fit, 2018's Hispanic residual slope is ≈ −3pp/unit; the
+9.2 seen in `race_shift_test.py` without it was white-college share, so 2018
gives no evidence of a Hispanic snapback.
`hispanic_downballot_trend.py`: judicial D 2p in 70%+ Hispanic districts 69.6
(2012), 69.1 (2016), 62.3 (2020), 52.8 (2024) — the shift reached low-salience
races and started before inflation. Polls are the only read on 2026 Hispanic
movement and disagree: UT Oct LV +5 vs 2024 (→ 10.3% if imposed); NYT/Siena 9/30
and ReconMR 9/11 ≈66% D.

**Open (Brennan):** publish 13% as is, add a stated range (≈13–35%, low end the
model, high end the D+12–13 reading of Texas's lean), or widen σ_national.
Separately: apply a poll-based Hispanic shift (which poll) or leave it as a
sensitivity.

---

## October 8-day refresh (2026-10-27)

**Finance.** TEC 8-day reports (due 10/26) pulled with `--force-download`: ~286
legislative 8-day reports received 10/22–27; every district's raised total rose,
none fell, no party-assignment changes. Five new viability flags: HD 41 (R
Groves), 55, 61, 63, 132. Four live-seat nominees had no 8-day report yet:
Morgan (HD 26), DeBurr (HD 65), Wheatley (HD 67), Bean (HD 94) — re-pull once
late filings land. `collect_finance_2026.py` now falls back to the cached
`cover.csv` when TEC's CloudFront 403s and the run is not `--force-download`.

**IE bug, fixed.** In `cand.csv` a multi-candidate expenditure is one row per
beneficiary, each carrying the FULL `expendAmount` (all 8,169 such
expenditures). `collect_ies_2026.py` counted each in full, so Texas Now PAC's
single $365,330 buy counted in nine districts; 2026 legislative IE read $30.9M
classified instead of **$9.7M ($7.6M R / $2.1M D)**. Now split evenly across
beneficiaries. The Oct 7 run had the same inflation. Training IE is unaffected
(built from the expenditure files, one row per expenditure). Also changed:
direction is now by **beneficiary first** (nominee's party, flipped when the
purpose text opposes that candidate — AFC Victory Fund files attack ads under the
target); a named non-nominee is excluded ($87K: Legacy 44 for Haddad, AFT for
Lawrence Allen, Tovo, …); filer party only when no candidate is named. Filer-first
had signed Texas REALTORS' $150 gifts to ~59 D nominees as R. Typo/wrong-district
beneficiaries mapped in `config/dce_beneficiary_overrides.csv` (Zack Dunn HD 121
$47K from Northside AFT, Bean, Richardson, Patterson, Tepper, Perry).

**Effect: zero net seats** (−0.01). Moves ≥2pp all trace to IE crossing the $50K
threshold or the IE share moving off zero (HD 122 −4.2, 20 −3.4, 52 +3.7, 74 +4.4,
67 +3.1) or new flags (HD 41 −3.2, 63 −2.2) — the 0.0089 coefficient doing what
the refit intended.

**Houston (the Bullpen question).** No Democratic money surge outside HD 133
($341K raised in October) and HD 138 ($228K); R incumbents there raised $1.0–1.1M
each in the same window, and HD 138 carries $521K of R IE. HD 129's Democrat has
raised $55K all cycle. Nothing here argues for a Houston-specific shift.

**Polls — LV only (Brennan, 2026-10-27).** NYT's list (`/interactive/polls/
congressional-vote-2026.html`; data at `/newsgraphics/polls/house.csv`) had 50
national releases since 9/23; the LV average of one release per pollster since
10/13 is ~D+9.4. Race banners on an LV base: NYT/Siena 10/12–19, Morning
Consult 10/19–25 (same 2026/09 URL, updated in place), CBS/YouGov 9/30–10/2.
Aggregate D+9.2 (unchanged); W 47.5, B 86.7, H 65.0 (+1.8). Checked and
excluded as RV/adult: Echelon 10/15–19 (RV), Pew 9/28–10/4 (RV), Focaldata
10/16–20 (adults), Economist/YouGov (race on RV base only — none wired now).
No B/H split: Marist 10/19–20 (White/Non-white). Paywalled: Strength In
Numbers/Verasight. Not located: CBS 10/21–23 PDF. `update_polling.py` now keeps
one release per pollster in the window. Race shares no longer move the mean
(demographic term deleted 10/7) — only the topline and the group-error layer.
Wedgewood, the fake-poll outfit exposed 10/21, was never in our inputs.

**Race-shift term re-tested on the adopted spec (2026-10-27,
`scripts/race_shift_test.py`).** The September test scored against pre-refit
coefficients; this one uses the open baseline + SHIP terms, leave-one-cycle-out.
Shift = Σ (district share − TX share) × (race D 2p now − prior presidential).
RMSE, held-out cycle (adopted → variant):
- National crosstab delta, imposed: 2018 3.59→3.65, 2022 2.80→2.88 (worse both).
  Half weight: pooled 3.24→3.17. Oracle (Catalist's actual 2022 race vote): 2.89,
  worse — national race swings don't transfer to Texas.
- UT same-pollster delta, imposed: **2018 3.59→3.17**, fixing the Hispanic-
  correlated miss (resid slope +9.2→−2.4pp/unit); 2022 2.80→2.87 (UT delta ≈0
  that year, so no signal either way). Pooled 3.24→3.03. One informative cycle.
- 2026 forward, same seed: UT Oct LV (new, `ut_tpp_oct2026_crosstabs.pdf`, Q27B
  LV p.136: W 35/57, B 70/18, H 48/41) minus UT Oct 2024 LV = W +0.7, B −3.4,
  H +5.1. House 66.0→66.7 seats but majority 12.0%→10.3%: Valley seats up
  (HD 37 29→49%, 34 55→72, 118 53→65, 41 64→82, 74 68→84, 80 6→15), DFW/Houston
  suburban targets down (HD 112 47→39, 108 37→31, 63 29→25, 61/66/67 −3 to −4).
  The national-crosstab version is the same shape, larger (66.5 / 8.9%).
- Published practice (research 2026-10-27): FiftyPlusOne matches our structure
  (uniform shift + zero-centered demographic error); Silver's error layer is the
  same, and he moves means only indirectly (CANTOR: topline polls in similar
  places); Split Ticket 2026 shifts by demographic group using its own large
  polls through an MRP-style electorate. Nobody feeds public race crosstabs into
  means. **Undecided (Brennan's call):** keep, ship the UT shift (full/half
  weight), or publish it as a sensitivity.

**2018 seat backtest and the "pros expect a bigger swing" question (2026-10-27).**
`scripts/backtest_2018_seats.py`: adopted spec incl. education, fit on 2014+2022
only → 2018 House **64.4 D seats vs 67 actual** (10–90: 56–74) at the real D+8.6;
63.4 at the Sept-1 D+7.5. Under by ~3, all on the R side: the 12 R-incumbent
seats Democrats took were predicted 3–5pp too Republican (model gave most of
them 23–65%). `scripts/compare_2018_2026_terrain.py` (open-race baseline, D
seats if every district moves D by +0/+2/+4/+6/+8/+10):
2016 races on 2013 map 56/58/60/69/74/82; 2020 races on 2013 map
68/74/79/85/91/96; **2020 races on 2021 map 66/66/67/69/76/92**; 2024 races on
2021 map 60/62/63/68/71/81. Same election, new map: −8 to −16 seats at +2 to
+6 — a wall of R seats just out of reach until ~+8, then it collapses. The
2020→2024 electorate shift costs another 1–6. In absolute seats the 2026 terrain
≈ 2018's; the gain looks small because Democrats start at 62 (above their 2024
baseline's 60), not 55. D+9.2 moves R-held seats +3.4pp vs baseline; D+13.5
moves them +5.4.

---

## The refit decision — DECIDED 2026-10-07

Adopted: the open-race baseline, the clean-cycle refit (pass-through 1.058,
env 0.0046, sigma 0.044), deletion of the demographic level term and the TX
Hispanic constant, incumbency and the signed viability flag after out-of-sample
tests (fundraising share dropped), IE re-estimated at 0.0089, WAR rebuilt on
the open baseline with a cycle fixed effect. Full record, test results and the
district-level consequences: `docs/baseline-refit-decision.md`. Branch
`refit-clean-cycles` is superseded (its WAR re-centering is folded in).

Still open from the September recommendation:
- ~~**Correlated group-error layer**~~ — **built 2026-10-07** (`model.run_monte_carlo`,
  `model_config.GROUP_ERROR_*`). σ white/Black/Hispanic/other 2.5/4/4.5/8.6pp:
  priors kept where larger than the five-cycle estimate
  (`scripts/group_error_estimate.py`), "other" from the estimate (2018 Asian-suburb
  swing). Variance carved out of σ_idio; check √(0.0335²+0.0139²+0.0249²)=0.0440.
  Effect at D+9.2 (`scripts/group_error_compare.py`): House 66.9→66.7 seats,
  12.7→12.2% majority, sd 7.80→7.94; Valley eight-seat block sd 1.81→2.01 and
  P(D loses all eight) 2.6%→5.4%.
- **Education — ADOPTED 2026-10-07 (Brennan: both on).** District white BA+ data
  now exists (`src/collect_education_by_district.py`, keyless ACS table-based
  summary files; 2021 ACS = pre-2022 lines, 2024 ACS = current). Both on by default (`TXLEG_EDU=0` / `TXLEG_EDU_ERR=0` turn them off for one run):
  the error layer splits white by education (σ non-college
  3.5 / college 2.5, priors); and a fixed midterm term is added
  (`REGRESSION_COEFFICIENTS_EDU`, +15.5pp per unit centered white-BA+ share, t 10).
  The term passes out of sample (`scripts/midterm_education_test.py`: RMSE
  3.37→3.13 / 3.20→2.72pp) because it fixes a 3.4pp pro-R miss in the most
  college-educated fifth of seats; it is zero on the presidential baseline (an
  open-race-composite blind spot). Side effect: it moves Valley seats R
  (HD 118 74→53%, HD 41 83→67%, HD 34 66→50%); held out, the Valley miss flips
  +1.0→−0.8pp, i.e. neutral within noise (n≈27, RMSE ~4pp), and a Hispanic term
  does not help. At D+9.2: off 66.7/12.5% · error split 66.8/13.2% · term 66.0/12.6%
  · both 66.1/13.1%. Finer race × education cells (Hispanic and Black split by
  BA+ too) were tested and do worse out of sample at every step — RMSE 3.13 →
  3.52 → 3.85 (`midterm_education_test.py --race-edu`); district results can't
  identify them. White education only is the right granularity for this data.
- **Midterm composition pattern** found while estimating it: every clean midterm
  runs R of the open baseline in proportion to Black and Hispanic share
  (turnout drop-off from a presidential-year baseline, most likely). A fixed term
  for it was tested and rejected — worse out of sample on all 259 races
  (`scripts/midterm_composition_test.py`). Re-test when 2026 results exist.
- **Sigma**: 0.044 kept by choice; the open-baseline leave-one-cycle-out on three
  midterms says ~0.031. Revisit if more clean cycles become available.
- **Wave-year level**: held out, 2018 (D+8.6) is under-predicted for Democrats by
  ~1.2pp under the shipped spec. 2026 sits at D+9.2, so the same miss is plausible.
- **Presidential-year cycles** miss by −2.3 (2020) / +2.4pp (2024) — absorbed in
  WAR by the cycle fixed effect, but a sign the env coefficient is midterm-only.

---

## Texas Bullpen worry list vs. the model (2026-10-19)

The Texas Bullpen published House GOP leadership's worry tiers (Tier 1: HD 26,
28, 34, 37, 52, 61, 63, 80, 94, 96, 97, 108, 112, 121, 122, 127, 129, 132, 133,
138, 150; Tier 2: 14, 20, 29, 54, 67, 126; Tier 3: 55, 89).
`scripts/bullpen_tier_compare.py`, run at D+9.2 (66.0 seats / 12.3%).

- **Broad agreement on membership.** All 29 listed seats are in the model's top
  40 R-held seats by flip probability; 15 of the model's top 16 are listed.
  Tier 1: median R+8.9, 4.4 expected flips, 6 of 21 within 5 points.
- **HD 118 is not listed at all** — the model's #2 flip (52%, open, D outraising
  R $756K to $660K).
- **Houston runs much deeper on their list than in the model.** Eight Tier 1
  seats are Harris/Fort Bend (26, 28, 127, 129, 132, 133, 138, 150), averaging
  R+13 in the model, vs. R+6 for their DFW Tier 1 seats. HD 129 (Bowen) is the
  widest gap: R+19 / 1% now; 46% in the May post-runoff table and 35%→9% with
  the old IE term. The model has no regional error term (group error is by race
  and education), so a Houston-specific move would not be represented. Watch
  the 8-day reports (Oct 26) for Houston money.
- Tier 2 includes two seats the model calls safe: HD 126 (R+21), HD 29 (R+17.5).
  Unlisted but about as live as their Tier 2/3 seats: HD 66, 106, 65, 57 (~R+11, 7–12%).
- **"Tier 1 within 5 points."** Conditioning the simulations on the median Tier 1
  margin: R+5 to even → 71.0 seats / 8% majority; within ±5 → 73.9 / 34%;
  within ±2.5 → 75.6 / 47%. Dial-equivalent: Tier 1 median at R+5 ≈ D+13.5
  (71.1 / 28%), at R+2.5 ≈ D+16 (74.5 / 40%). As a poll reading it implies
  ~4pp bluer than the model in those seats; as a final result where Republicans
  narrowly hold them, it barely helps (D needs ~14 net flips of 21).
- **No model change.** A worry list is a resource/donor document, not polling,
  and Texas polls have historically missed toward Democrats.

---

## Firebrand / rabble-rouser penalty (2026-10-07)

Brennan asked whether visibly extra-partisan members pay a general-election
penalty, measured by high-profile signals rather than votes.
`scripts/firebrand_penalty_test.py`; tags in `data/raw/historical/
tx_texas_monthly_lists.csv` (Texas Monthly Best/Worst 2013–2025, 203 rows) and
`tx_firebrand_markers.csv` (Freedom Caucus by session, insurgent PAC backing,
speaker revolts, viral confrontations, quorum-break leaders; 295 rows, every row
sourced; establishment-target censures and other wrong-way rows recoded out).
Outcome: race WAR, contested incumbent races 2018–2024; a tag counts for the two
elections after it.

- **Raw: yes.** Republican firebrands run −0.9pp (t −2.8, person-clustered;
  60 races / 39 people); Texas Monthly Worst Rs −1.1; Freedom Caucus −0.8;
  speaker-revolt Rs −2.1 (14 races). Texas Monthly Best Rs +1.5. Democratic
  rabble-rousers +1.0 (13 races) — no penalty, if anything a bonus.
- **Beyond the member's own track record: no.** Controlling for prior WAR the
  R tag is −0.2 (t −0.5); adding a penalty on top of the model's WAR term makes
  out-of-sample prediction worse. The penalty is real but already carried by
  WAR persistence. No model change.
- **WAR persistence re-estimated (done, 2026-10-07):** on the rebuilt WAR,
  carry-forward rises with races on record (1 race 0.16, 2 races 0.62);
  adopted beta(n) = n/(n+2.93) — 0.25/0.41/0.51/0.58 for 1–4 races
  (`scripts/war_persistence_estimate.py`, `model_config.WAR_PERSISTENCE_K`),
  replacing the flat 0.46. Beats a single beta in every held-out year.
- **Non-incumbent stand-in (Brennan, 2026-10-07):** flagged Republican
  non-incumbents get +0.5pp toward the Democrat (half the raw incumbent effect;
  `FIREBRAND_STANDIN_PP`). 22 R non-incumbents in live seats screened
  (`config/firebrand_nonincumbents_2026.csv`, sourced): flagged Wambsganss (SD 9:
  Patriot Mobile school-board takeovers, $600K+ Dunn/Wilks network) and Bean
  (HD 94: $250K TUCM); Bowen (HD 129) cleared the $40K line but set to 0 as an
  establishment consensus pick. No D flag (no D penalty found). Effect: HD 94
  19→22%, SD 9 3.6→4.7%. House 66.2 seats / 13.3%.

## Texas-poll calibration — historical piece (2026-10-07)

Brennan asked how to use Texas polls against the generic-ballot dial. Built
`scripts/texas_poll_error_history.py` on 333 Texas statewide general-election
polls, 2014–2024 (Wikipedia, revision-pinned; `data/raw/historical/
tx_statewide_polls_2014_2024.csv` + results) and 538/RCP national generic
averages as of Oct 1 (Wayback-archived; 538's 2014 series is a 2020 back-cast).

- Texas polls as of Oct 1 ran **+2.8 points too Democratic** on average,
  too D in 5 of 6 cycles (2014 +6.4, 2016 +1.5, 2018 −3.2, 2020 +3.1,
  2022 +2.7, 2024 +5.8); cycle sd ~3.5. In 2014 the Oct-1 average was 7 too D
  and the final month dead on — Texas polls have drifted R late.
- **They carry no independent information against the model.** In all three
  midterms the Texas polls missed in the same direction as the national polls
  that drive the dial, only larger: model real-time miss +4.7/−2.1/+1.2 vs
  Texas polls +6.4/−3.2/+2.7, correlation +0.99. Blending would have made every
  midterm worse (weight 0.25: rms 3.0→3.4; 0.5: →3.7); the error-minimizing
  weight is negative. Same with partisan-sponsored polls excluded.
- Caveats: three midterms; most historical Texas polls are candidate races
  (governor/Senate), not a generic ballot — there were essentially no Texas
  generic polls before 2020.
- **Implication: do not shift the forecast level toward Texas polls.** The
  2026 gap (Texas generic polls ~R+1, R+4 after the historical bias, vs the
  model's R+8.3) is large, but history says Texas polls amplify the national
  polling error rather than correct it. Open option (Brennan's call): widen
  σ_national when Texas polls and the model disagree by more than history,
  without moving the mean. Re-test once 2026 results exist.

## Polling refresh (2026-10-07)

11 national releases fielded Sep 7–Oct 7 carried a W/B/H generic banner
(Economist/YouGov, Emerson, Quantus, Big Data, NYT/Siena, Echelon, Zogby,
Rainey, McLaughlin, Morning Consult, Marquette); recorded with the misses in
`config/pollster_registry.csv`. Aggregate topline 54.6% (D+9.2) vs. Silver
Bulletin D+8.9 raw / D+9.6 LV — the August-15 values had drifted back into line,
so the dial barely moved. YouGov's table was retitled "2026 Congressional
Vote" in October, which silently broke the extractor; fixed. Texas: no new
Legislature generic since UT August; NYT/Siena (9/21–30) and ReconMR (9/8–11)
US-House generics put Texas Hispanic D2p at ~66% vs. UT's 53% in August —
added to `texas_crosstab_inputs.csv` with TSU AG/Comptroller banners.

## The IE term overstates what IE money means (2026-10-07)

**Decided 2026-10-07 (Brennan): `IE_COEFFICIENT` set to 0.026**, then re-estimated
the same day on the adopted refit's terms at **0.0089** (neutral out of sample).
The findings below still describe what IE money signals.

With the October 30-day reports in, every competitive R-held House seat now
carries $65K–$520K of one-sided Republican IE money (dollar figures inflated:
multi-candidate expenditures were counted in full per candidate, fixed
2026-10-27; the one-sidedness stands). The model reads that as
−3.7pp for the Democrat in each seat (HD 129 35%→9%, HD 96 51%→30%, HD 94
67%→52%). **The whole −0.9 House-seat drop from the October money comes from
this term**; with it zeroed, the new finance data leaves the forecast at 66.5.

`scripts/ie_meaning_backtest.py` tests what IE money signalled on the training
rows (268 contested midterm races, 2002–2022, 45 with ≥$50K of IE):

- **It is not a weakness signal.** Favourites whose side got ≥75% of the IE
  money ran +0.15pp vs. the no-IE model (se 0.9) and were upset *less* often
  than expected (7% vs. 16%). In base-model toss-ups, defended favourites ran
  +1.9pp (se 1.3, n=16).
- **But it is not a 3.7pp push either.** Fit on all 268 rows with the
  master terms, the IE coefficient is **+0.026 (t 1.1)**. The 0.074 in config
  came from the `full_ie` regression on a 102-row subsample with a different
  specification (pres pass-through 0.36), bolted onto coefficients fit
  elsewhere.
- **Context matters a bit:** the slope is concentrated in races the base model
  calls within 10pp (interaction +0.040, t 1.4); nothing in safe seats.
- **Thin evidence on the 2026 pattern:** 41 of the 45 training IE races are
  defence, 4 offence, 0 mixed. Training IE was built by `collect_ies_pac.py`
  (PAC registry + expenditure descriptions), 2026 by `collect_ies_2026.py`
  (cand.csv DCE) — not the same measurement.

At D+9.1: coef 0.074 → 65.6 seats / 9.0%; 0.026 → 66.1 / 10.6%; 0 → 66.5 / 11.7%.
(Superseded by the refit: 0.0089.)

## Corrected TEC reports were double-counted (2026-10-07)

When a filer corrects a report, TEC keeps the original and the correction as
separate cover rows, and every collector summed both. Excess dollars:
2014 18.6%, 2018 13.1%, 2022 9.4%, 2026 7.8% (e.g. Rehmet's July report filed
7/15 and 7/17, $380,152 each). **Fixed for 2026** (`latest_reports_only` in
`collect_finance_2026.py`; superseded-report filter in `collect_ies_2026.py`,
which dropped $69K incl. $59,683 of RPT spending in SD 4) — moved HD 41, 61,
63 off the viability flag. **Training pipeline fixed the same day**:
`latest_reports_only` / `superseded_report_ids` now live in `collect_finance.py`
and every collector uses them; historical IE fell sharply in places (2010
R-favouring $16.7M→$9.6M, 2014 $67.7M→$54.0M). The historical IE rebuild ran
from the May-16 TEC ZIP because TEC was 403ing (2002–2022 expenditures do not
change); those expend files were removed from the cache afterwards.

## October finance era (2026-10-07)

`VIABILITY_THRESHOLD_OCT30` = $100K House / $220K Senate, from
`scripts/calibrate_viability_threshold.py` — the in-cycle threshold that best
reproduces the full-cycle training flag. The same criterion reproduces the
SEMIJUL values ($75–80K / $140K), and the answer is unchanged with or without
the dedup. Senate n is small. `IE_WEIGHT` set to 1.0 per its own schedule.

---

## Poll integration (the Nate Silver thread)

**What Silver actually does**, from the FLIPR methodology:
- The fundamentals prior "essentially fades out by Election Day"; drift zeroes out.
- Recency decay gets *more aggressive* as the election nears.
- LV versions are used outright; RV/adult shifted toward whichever party wins the
  head-to-head comparisons. He **discarded** the old prior that LV favours
  Republicans, and the adjustment is worth ~1.5pp on the 2026 generic ballot.
- A **correlated demographic error** layer lets subgroups miss together
  ("Democrats might do better than expected among Hispanic voters").
- Uncertainty *widens* when fundamentals and polls disagree.

**Texas poll supply — measured, not assumed.** Texas Politics Project Senate
tracker, June–August 2026: nine general-election polls, six likely-voter,
samples 619–1,200. That is ~3/month. Historical monthly shape (2022 Gov, 2024
Sen) shows October running 2–3× September and September ~2× August, projecting
**5–7 Texas polls in September and 10–15 in October**.

*The binding constraint is not volume — it is crosstabs.* But **do not read that
as a fixed roster of pollsters who "have crosstabs".** Whether a release breaks
the generic ballot out by race is a property of **the release, not the
pollster**: Emerson's July 2026 release had none (checked three ways) and its
August release did, in a linked workbook tab. **Check every wave.**

Rough expectation only: perhaps half of Texas releases carry a usable banner, so
on the order of **2 Texas crosstab polls per month now, more in October.** A
prior session reached the same conclusion from the other direction: the
NYT/Silver roster lists toplines, so it tells you who fielded, not what you can
consume.

The maintained list is `config/pollster_registry.csv`, which records history —
last release we actually pulled a banner from, and how many we have checked —
rather than a verdict. `src/weekly_poll_check.py` prints it; scheduled task
`poll-check-weekly` runs Thursdays 08:00.

**The proposal and its two rounds of backtests are in
`docs/poll-integration-proposal.md` (2026-09-09; §5 April inputs, §6
as-of-September-1 inputs).** Verdict: the demo term is a *level* that
double-counts the presidential baseline (+3.7pp mean, r = +0.70) and
manufactures the Hispanic-correlated residual the −0.05 constant was fit to
cancel; with no demographic term that residual is zero in both cycles under
both input sets. No shift-form replacement — national-only or same-pollster
Texas (UT) — beat the null in either cycle. **Ship: no demographic term,
delete the constant, add the correlated group-error layer, refit the σ
split.** Shelved: the shift term (re-test each new Texas banner with
`scripts/shift_term_backtest_sept1.py`). New open item: a Texas-specific
environment offset from Texas generic *toplines* — 2022's null run is +2.3pp
too Democratic across the board with the right national dial, and UT's RV
generic read the 2022 Texas swing in the wrong direction. Texas banners
2016–2026 captured in `data/raw/texas_crosstab_inputs.csv` (gitignored;
mirrored).

**Which statewide race is the right instrument — measured 2026-09-10, see
`docs/statewide-proxy-findings.md`.** Same-year, five cycles: the low-salience
races (Supreme Court, CCA, Railroad Commissioner, Comptroller) track the House
best and the top of the ticket worst; the 2024 presidential vote is the worst
of the nine 2024 statewide contests as a House proxy (resid sd 2.4pp vs 1.6 for
RRC) and misses 70%+ Hispanic seats by +3.9pp vs +1–2 for judicial races.
Cross-cycle is split: 2016→2018 favoured President, 2020→2022 (the only
post-straight-ticket pair) favoured RRC, where President missed South Texas
by +5.7pp and RRC by +0.7pp. Empirical district-level pass-through is 1.0–1.1
in both pairs. **Open item, ahead of the refit decision:** build a 2024
RRC/judicial-mean baseline on PlanH2316 and run it through the 2022 backtest
against the presidential baseline. For the Texas banners: use the mean of the
TX-legislature generic, AG and Comptroller; exclude Senate (Talarico runs 3–5pp
ahead among white voters in every UT wave).

**Caveat to keep stating:** these are *statewide* polls. They give the Texas
Hispanic vote share, not HD 74's. Better than borrowing the national number, but
it is not district polling.

---

## Small, self-contained, worth doing

- **`backtest.py` / `backtest_config.py` still run the old design** (presidential
  baseline, master-era coefficients). The demographic term was removed there too
  on 2026-10-07, but its coefficients and baseline were not moved to the refit;
  `scripts/refit_clean_cycles.py` is the current out-of-sample check. Port it
  before trusting a `backtest.py` number again.

- ~~Fix the `update_polling.py` docstring~~ — **done 2026-09-09.** It named four
  sources and called Quinnipiac topline-only while the manual CSV already held
  banners from Quantus, The Argument/Verasight, Big Data Poll and Emerson, and
  it claimed a 45-day window the code never used. Rewritten to point at the
  registry and state the per-release rule. **The general lesson stands: any
  table in this repo asserting which pollsters "have crosstabs" is the wrong
  shape and will rot — record per-release history instead.**
- ~~Wire the Fox Texas crosstab PDF~~ — **parsed 2026-09-09** into
  `data/raw/texas_crosstab_inputs.csv` along with UT/TPP (Jun/Oct 2024, Jun/Aug
  2026) and Emerson Texas. `update_polling.py` still has no Texas loader; that
  is step 1 of `docs/poll-integration-proposal.md` §3f.
- **`backtest_config.py` 2018 `env_dial` is 0.** Its own comment says the
  April-2018 generic ballot was ~D+8; 2022 uses the actual −2.8. Setting +8.6
  moves the 2018 house error by ~3 seats and mean residual by ~2pp. Fix when
  the backtests are next touched.
- **`build_district_table.py` guard.** It silently destroys `open_seat` (28
  rows), `notes_2026` (30) and the whole finance/IE block; nothing repopulates
  the first two. Blast radius is the six R-held seats atop the competitive list,
  which would each silently gain an 8pp incumbency term.
- ~~Phantom R seats~~ — **fixed 2026-09-10.** `model.py` now reads
  `candidates_2026.csv` (`_attach_unopposed`) and holds any seat with one side
  `none_filed` at 1.0/0.0 inside `run_monte_carlo` before seats are summed;
  the display-only override in `build_maps.py` is gone. Effect on master at
  D+9.1: +0.3 expected House seats (66.2 → 66.5), +0.6 at R+3; P(majority)
  unchanged to four decimals because the leak was idiosyncratic noise on HD
  36/39/75/78/105, not national swing. Only the 19 flagged districts moved.
- **Fix the validation harness** (`~/.claude/skills/election-model-validation/`).
  It reported 4 blocking FAILs, **all four wrong**, and missed the one real
  name-match failure (HD 22 Hayes/Manuel). Row count compared to 166 without
  filtering `up_in_2026`; duplicate check ignores `chamber`; B2 describes a fix
  `model.py:382-400` already implements; B3 models the join as `.lower().strip()`
  when the model runs a fuzzy matcher. Its σ reference is stale too (0.0785).
- **No test suite** anywhere in 17k lines, and the **methodology doc is not in
  git** (`output/*` is gitignored bar HTML and a few CSVs) — the one file where
  drift is the thing being audited.

---

## Known-unreachable (don't re-litigate)

- **2002/2006/2010 training cycles.** Need 2000/2004/2008 presidential by
  district. Returns exist (`ftp_election_data_12g.zip` carries 1996–2014) but
  TLC precinct geometry *and* RED-365 crosswalks both stop at the 2010 general,
  so those precincts cannot be districted at all.
- **Extending WAR before 2018.** Candidate *names* only appear from 2018 on;
  earlier articles are summary tables with votes by party. The Texas SoS Race
  Summary Report does carry names and is reachable, but tested against value:
  only **11 of 166** on-ballot incumbents hold all four available cycles, three
  of those were first elected in 2018, and of the remaining eight only **Angie
  Chen Button (HD 112, 22.7%)** sits in a seat competitive enough to matter.
  Not worth a new scraper.
- **Presidential years in the midterm regression.** Tested: adding 2012/2016
  moves full-sample σ 0.0255 → 0.0287 and worsens out-of-sample MAE on two of
  three midterms. They stay in the dataset (WAR uses them; they give the env
  term five identifying values) but the midterm coefficients stay midterm-fit.
