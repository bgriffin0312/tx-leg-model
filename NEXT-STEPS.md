# Next steps — tx-leg-model

*Written 2026-09-09 at the end of a full bug-and-code audit. Everything below is
either measured or explicitly flagged as unverified.*

*Published: GitHub Pages republished 2026-10-07 (`19a0073`, the D+9.2 run below).
The Substack's Datawrapper embeds (l53Um / oz3IU / L6hhb) have the new data
uploaded but NOT published — `sync_datawrapper.py` still gets 403 "Insufficient
scope" on publish; the token needs the publish scope, or click Publish by hand.*

*2026-10-07: polls (11-pollster national race aggregate, D+9.2), October 30-day
TEC reports and IEs are in, and the September refit decision was taken (open-race
baseline, clean-cycle refit, demographic term deleted, training data deduplicated).
With the group-error layer and the education term + error split: **66.1 House
seats / 13.1% majority, Senate 11.1 / 0.2%** at D+9.2. See the 2026-10-07 sections below.*

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
carries $65K–$520K of one-sided Republican IE money. The model reads that as
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
