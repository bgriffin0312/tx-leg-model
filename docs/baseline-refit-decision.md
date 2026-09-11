# DECISION PENDING — coefficient refit and baseline source (written 2026-09-10)

Brennan intends to take this decision over the weekend of 2026-09-12/13.
Nothing here is adopted; the model still publishes master coefficients on the
presidential baseline with the demographic term in place.

**Start here next session.** This file is the step-by-step review. The
evidence behind it is in `docs/statewide-proxy-findings.md` (tests 1–3b and
the AG anchor) and `docs/poll-integration-proposal.md` (§5–6, the demographic
term backtests). Reproduce any district with:

    python scripts/decision_walkthrough.py --district 41
    python scripts/decision_walkthrough.py --district 112 --chamber house
    python scripts/baseline_backtest.py          # the three-cycle comparison
    python scripts/baseline_project_2026.py      # structural-model seat counts
    TXLEG_BASELINE=open python src/model.py --no-save   # full model, any baseline

---

## The prediction line

Every district's predicted D two-party share is:

    intercept
    + pass_through × baseline_2024
    + env_coef × generic_ballot_dial          (D+9.1 today)
    + dem_incumbent / rep_incumbent / senate
    + demographic level term + Hispanic constant
    + WAR persistence, finance, IE terms

Win probability = Φ((prediction − 0.5) / σ).

## Choice 1 — which coefficient set

Master's set was fit on a dataset that joined outcomes to the wrong district
geography for every cycle before 2022 (see `build_phase1_dataset.py` header);
that measurement error attenuates the pass-through. Branch
`refit-clean-cycles` fit the same regression on clean cycles. The refit is
**four changes, not one**:

| coefficient | master | refit |
|---|---|---|
| pass-through on baseline | 0.596 | 0.976 |
| intercept | +0.178 | −0.032 |
| D incumbent / R incumbent | +6.8 / −8.0pp | +3.9 / +0.8pp |
| finance (viability flag, D fundraising share) | 4.5 / 7.3pp | 0.2 / 0.6pp |
| environment coefficient (per pp of dial) | 0.0025 | 0.0036 |
| sigma | 0.074 | 0.044 |

Well supported: pass-through and sigma. Every cross-cycle test on 2026-09-10
put the district-level slope of midterm House on presidential-year statewide
between 1.00 and 1.12 (`scripts/statewide_proxy_cross_cycle.py`,
`scripts/baseline_backtest.py` pooled fits), and the clean-cycle residual is
genuinely tighter.

Not separately tested: the incumbency and finance changes. They are what the
same clean-cycle fit produces (the pooled fit in `baseline_backtest.py` gives
dem_inc +3.5 / rep_inc +0.6 as well), but they drive HD 34 and HD 112 more
than the baseline does. **They can be adopted or held back independently of
the pass-through.**

## Choice 2 — which 2024 race is the baseline

| option | what it is | `BASELINE_SOURCE` |
|---|---|---|
| presidential | Harris two-party share by district (current) | `pres` |
| open | mean of the three 2024 CCA races, the only 2024 statewide downballot contests with no incumbent on either side (Brennan's rule) | `open` |
| blend | half presidential, half open | `blend_open` |

Evidence summary (`docs/statewide-proxy-findings.md`):

- Same year, five cycles: the least salient statewide races match contested
  House results best; the presidential vote is the worst of nine 2024
  statewide contests as a House proxy and misses 70%+ Hispanic seats by +3.9pp.
- Leave-one-cycle-out, 2014/2018/2022: open wins 2022 outright (the only
  post-straight-ticket cycle; residual sd 2.59 vs 3.08, Hispanic residual
  slope −0.002 vs −0.085), loses on dispersion in the straight-ticket era,
  best seat error on average. The blend has the best average residual sd
  (2.66pp) and Brier of everything tried.
- 2024 pattern: the open races run ~2pp more R than Harris in Anglo suburban
  seats and ~2.4pp more D in 70%+ Hispanic seats.

Judicial races with incumbents are out (worst in every cycle). RRC 2024 is
out under the no-incumbent rule (Craddick).

**Interaction:** under master's 0.596 the open baseline *lowers* expected D
seats (66.5 → ~65 at D+9.1) because the damped pass-through mutes the South
Texas gain while the downballot vote's statewide R lean hits every seat.
Under refit coefficients the structural model gains six seats. The two
choices are one decision.

## Rider — delete the demographic level term and Hispanic constant

Backtested twice (April inputs, September-1 inputs): the level term
double-counts composition already in the baseline (+3.7pp mean, r = +0.70
with the baseline) and manufactures the Hispanic-correlated residual the
−0.05 constant was fit to cancel. The 2026-09-10 AG-banner check found the
Texas-minus-national gap is mostly uniform (−9 white / −4 Black / −11
Hispanic; relative to white the Hispanic offset is −2 to +2pp), so there is
no Hispanic-specific gap for the constant to represent. Replace with the
correlated group-error layer in the Monte Carlo (proposal §3e).

---

## Worked example 1 — HD 41 (open seat, Groves v. Salinas, 81% Hispanic)

Fixed inputs: CVAP white 14.3 / Black 0.8 / Hispanic 81.4 / other 3.0.
Baselines: presidential 0.4918, open 0.5049, blend 0.4984. Demographic term
nets +1.4pp (level +5.5, constant −4.1). No incumbency. Dial D+9.1.

| coefs / baseline | pass × base | env | demo | WAR+fin+IE | pred | P(D) | without demo |
|---|---|---|---|---|---|---|---|
| master / pres **(published)** | 0.293 | +2.3pp | +1.4 | +6.0 | 56.8% | 82% | 55.4% → 77% |
| master / open | 0.301 | +2.3 | +1.4 | +6.0 | 57.6% | 85% | 56.2% → 80% |
| refit / pres | 0.480 | +3.3 | +1.4 | +0.7 | 50.2% | 52% | 48.8% → 39% |
| refit / open | 0.493 | +3.3 | +1.4 | +0.7 | 51.5% | 63% | 50.1% → 51% |
| refit / blend | 0.486 | +3.3 | +1.4 | +0.7 | 50.8% | 57% | 49.4% → 45% |

Reading: master = 0.178 + 0.596 × 0.492 = 0.471, then the finance terms add
six points because the D candidate has a fundraising edge and master weights
that heavily — that is the published 82%. Under the refit the baseline term
nearly doubles in weight, the intercept drops to cancel that for an average
district, and finance collapses to under a point: a coin flip. The open
baseline then adds 1.3pp at pass-through 0.98, worth ~11 points of win
probability at σ 0.044. Removing the demographic term takes 1.4pp back.

## Worked example 2 — HD 112 (Angie Chen Button, R incumbent, 14% Hispanic)

Baselines: presidential 0.4826, open 0.4547 (Harris ran 2.8pp ahead of the
downballot ticket here). Demographic term nets −1.4pp.

| coefs / baseline | pred | P(D) |
|---|---|---|
| master / pres **(published)** | 40.6% | 10% |
| master / open | 39.0% | 7% |
| refit / pres | 47.7% | 30% |
| refit / open | 45.0% | 13% |
| refit / blend | 46.4% | 21% |

The refit raises her risk mostly because master gives R incumbents eight
points and the refit gives them under one. The open baseline lowers it. The
two choices pull opposite ways and roughly cancel at the blend.

## Worked example 3 — HD 34 (Villalobos, R incumbent, 71% Hispanic)

Baselines: presidential 0.4929, open 0.4939 — a wash.

| coefs (any baseline) | pred | P(D) |
|---|---|---|
| master **(published)** | 43.3% | 19% |
| refit | 50.8% | 58% |
| refit, without demo term | 49.1% | 42% |

The entire move is the incumbency coefficient and the pass-through, not the
baseline. This is the district that shows the incumbency change is a
decision of its own.

---

## Recommendation on the table (Claude, 2026-09-10)

1. Adopt the refit pass-through (0.976) and σ (0.044).
2. Adopt the open-race baseline, or `blend_open` to hedge a one-cycle result.
3. Delete the demographic level term and the Hispanic constant; add the
   correlated group-error layer.
4. Treat the incumbency and finance coefficient changes as a separate
   question and backtest them in isolation before adopting.

Under refit + open + no demographic term: HD 41 ≈ 51%, HD 112 ≈ 21%,
HD 34 ≈ 43%.

## What else touches this

- The generic-ballot dial (D+9.1) is ~1.5pp too Democratic against every
  external read (NEXT-STEPS). It pushes the opposite way from the refit and
  should be set in the same pass.
- The Texas group-shift term (proposal §3a) is shelved pending UT's October
  LV wave; the AG anchor says the national Hispanic rebound should not be
  applied to Texas.
- Published artifacts in `output/` still reflect the 2026-08-18 run. Nothing
  from the audit, the unopposed-seat fix, or this work is republished.
