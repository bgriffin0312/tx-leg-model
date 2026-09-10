# Next steps — tx-leg-model

*Written 2026-09-09 at the end of a full bug-and-code audit. Everything below is
either measured or explicitly flagged as unverified. Published artifacts in
`output/` still reflect the **2026-08-18** model run; nothing from the audit has
been republished.*

---

## The one decision blocking everything else

**Adopt, damp, or reject the clean-cycle coefficient refit** (branch
`refit-clean-cycles`, commit message has the full reasoning).

It is statistically cleaner and it makes a contestable bet:

| | current (master) | refit (branch) |
|---|---|---|
| pass-through | 0.5962 | 0.9759 |
| intercept | 0.1781 | −0.0322 |
| sigma | 0.0742 | 0.0440 |
| House D seats @ D+9.1 | 67.4 | 64.0 |
| P(House majority) | 16.0% | 7.0% |

At pass-through 0.98 every district is predicted at its **2024 presidential**
number. In South Texas that number carries the Trump-era Latino swing, so the
competitive list inverts — HD 34/41/74/36/37/42 take it over while HD 118 falls
from 84.8% to 35.9% and the DFW seats drop off. But the 2024 *legislative*
results show Democratic incumbents there still running well ahead of the top of
the ticket (HD 74 Morales +9.1pp, HD 41 Guerra +4.3pp). **The refit bets that gap
closes by 2026.** That is a claim about Texas politics, not statistics.

Two parts of the branch are *not* contested and can be adopted independently:
- **WAR re-centering.** Mean baseline residual +2.464 → +0.279pp. Straightforwardly correct.
- **The σ decomposition** (national 0.0339 / idio 0.0280, total 0.0440), which
  also validated model.py's existing 58% shared-variance split at 59%.

---

## Highest-value single item outstanding

**The generic-ballot dial is ~1.5 points too Democratic.**

`GENERIC_BALLOT_TOPLINE_D_2P = 0.5455` (D+9.1). Every external check disagrees:

| source | value |
|---|---|
| your own unapplied `update_polling.py` refresh | **D+7.4** |
| Silver Bulletin, LV-adjusted (2026-09-09) | **D+7.6** |
| Silver Bulletin raw | D+6.6 |
| FiftyPlusOne / RCP / VoteHub / Race to the WH | D+5.6 – D+6.5 |

This dial shifts **all 166 races**. It is a bigger effect than most of what the
audit fixed, and the correction is already computed and one command away. Do not
decide it separately from the refit above — they push in opposite directions.

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
