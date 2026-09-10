# Proposal: Texas-anchored racial polling term (replaces the level-form demo term and the Hispanic constant)

Written 2026-09-09. Status: **proposal — no model code changed.** Data captured
alongside it: `data/raw/texas_crosstab_inputs.csv` (three Texas polls with race
banners), `data/raw/ut_tpp_aug2026_crosstabs.pdf`, `data/raw/emerson_tx_aug2026.xlsx`.
Numbers below were computed in-session from those files, the repo's
`racial_crosstab_inputs.csv`, and CES 2024 Common Content (Harvard Dataverse
doi:10.7910/DVN/X11EP6, CC0).

## 1. What is wrong today

`model.py` adds `demo_deviation = Σ_r CVAP_d[r]·D_nat[r] − Σ_r w_nat[r]·D_nat[r]`
to a prediction that already contains `pass_through × dem_pres_2p_baseline`.
That term is a **level**, and the presidential baseline already carries the
level. Across the 166 on-ballot districts it averages **+3.69pp** and correlates
**+0.70** with the presidential baseline. `run_phase1_regression.py` contains no
race or demo regressor, so the term enters entirely outside the fit — the
intercept never absorbed it. `TX_HISPANIC_ADJUSTMENT = −0.05` trims the mean
to +2.02pp (r = +0.67); it is a patch on the formula, not a fact about Texas
Hispanics. Both of its calibrations (the 2022 backtest and the 2018 validation)
were run against **national** crosstabs — Harvard-Harris and Pew, per
`backtest_config.py:139-144` — so they measured the double-count.

The methodology doc says the term captures "shifts since 2024". The formula
does not compute a shift. This proposal makes the formula match the prose.

## 2. Inputs that now exist

### 2a. Texas polls with a race banner (all figures 2-party D share)

| Poll | Ballot | White | Black | Hispanic | Topline |
|---|---|---|---|---|---|
| **UT/TPP** Aug 5–13 2026, RV 1,200 (YouGov) | **Texas Legislature generic** | .355 | .826 | .533 | .467 |
| UT/TPP | U.S. House generic | .376 | .833 | .522 | .478 |
| UT/TPP | Senate / Governor | .410 / .379 | .852 / .824 | .573 / .506 | .519 / .471 |
| Fox (Beacon/Shaw) Jul 23–27, RV 1,006, Hispanic oversample n=556 | Senate / Governor | .424 / .394 | .870 / .859 | .590 / .580 | .515 / .495 |
| Emerson TX Aug 9–10, LV 1,000 | Senate / Governor | .378 / .361 | .877 / .867 | .607 / .595 | .495 / .479 |
| *National generic, 4-poll avg Jul 7–Aug 17 (repo)* | — | .475 | .846 | .618 | .545 |

Within the same UT sample, Hispanics are 4pp more Democratic on the Senate
ballot than on the legislative generic; Fox/Emerson Senate numbers run 6–7pp
above UT's legislative generic. That is a Talarico effect, not a lean. **Only a
generic-legislative banner defines the Texas offset.** Senate/Governor banners
are discounted evidence (see §3b).

### 2b. 2024 Texas benchmark by race — validated voters, not exit polls

CES 2024 Common Content, Texas respondents with a TargetSmart-validated 2024
general-election vote and a presidential choice (n = 2,317), weighted with
`vvweight_post`:

| Group | D 2-party | n | Note |
|---|---|---|---|
| All | **.423** | 2,317 | actual Texas result .431 — within 0.8pp |
| White | .321 | 1,601 | |
| Hispanic (race) | .513 | 346 | .492 using the any-race Hispanic flag (n=423) |
| Black | .752 | 238 | implausible; small n |
| Asian | .635 | 23 | unusable |

Same file, 2024 **state House** vote (`CC24_415d`): all .419, white .315,
Black .735, Hispanic .517 — the legislative vote by race tracks the
presidential vote by race to within a point, which is the pass-through
assumption in survey form.

Catalist (*What Happened 2024*, p.28) independently puts Texas Latino support
"below 50%" in 2024, −8 from 2020 and −15 from 2016; state-level levels are not
public.

**Reconciled benchmark** (hold Black at Catalist's national .85 and other at
.60; keep CES's white–Hispanic gap; force the CES validated electorate —
white .627 / Black .115 / Hispanic .194 / other .064 — to sum to the actual .431):

    D_TX_2024 = { white_nh: .314, black_nh: .85, hispanic: .506, other: .60 }

The Black and other entries are **assumptions anchored on national data**, not
Texas measurements. Say so in the methodology doc.

### 2c. What the shift form produces

Texas shifts since 2024 from UT's legislative generic minus the benchmark:
**white +4.1, Hispanic +2.7, Black −2.4** (other −10.0, from UT's tiny Asian
cell — noise; see shrinkage). The national crosstabs alone imply Hispanic
**+7.8**, white +1.5.

Centered shift term across the 166 districts:

| Source of Δ | mean | sd | min | max | r with pres baseline |
|---|---|---|---|---|---|
| Texas (UT lege-generic − benchmark) | −0.16pp | 1.08 | −4.35 | +1.47 | −0.66 |
| National only (4-poll − Catalist 2024) | +0.87pp | 1.51 | −0.92 | **+5.06** | +0.18 |
| *Current level form, for comparison* | *+3.69pp* | — | — | — | *+0.70* |

The +5pp South Texas bump under national-only Δ is exactly the thing the
Hispanic constant was invented to suppress. Texas data removes it directly: the
2026 movement is among Anglo Texans, not a Hispanic swing-back. The −0.66
correlation on the Texas row is driven by the Black −2.4 and other −10 cells,
both small-n — hence §3c.

## 3. The proposal

### 3a. Replace the level with a centered shift

    Δ_TX[r]      = D_TX_now[r] − D_TX_2024[r]
    demo_term_d  = Σ_r CVAP_d[r]·Δ_TX[r]  −  Σ_r w_TX[r]·Δ_TX[r]

`w_TX` is the validated 2024 Texas electorate (§2b), not the national weights.
Centering makes the term a pure composition deviation (mean ≈ 0), so the
uniform part of the swing stays in the environment dial. Set
`TX_HISPANIC_ADJUSTMENT = 0` and delete its provenance block; its job no
longer exists. Rewrite the methodology section to describe what is computed.

### 3b. Texas offset applied to the frequent national polls

Maintain a per-group offset from the most recent Texas **generic-legislative**
banner against the contemporaneous national average:

    g[r] = D_TX_lege[r] − D_nat_30day[r]        # today: white −12.0, Black −2.0, Hispanic −8.5

Between Texas releases, `D_TX_now[r] = D_nat_30day[r] + g[r]`. A new Texas
legislative banner re-anchors `g`. Senate/Governor banners contribute at half
weight and only as **within-poll relative structure** —
`(D_TX_poll[r] − D_TX_poll_topline)` compared with the same quantity in the
national average — so candidate effects cancel.

### 3c. Shrinkage — small cells drive the sign

Weight each Texas subgroup cell by effective n with a 400-respondent
pseudo-count toward the prior value of `g[r]` (prior = last accepted Texas
offset; before any exists, 0). Fox's 556-Hispanic oversample earns near-full
weight; UT's Asian cell (n≈40) earns almost none, which is what stops the
spurious −10 "other" shift. Black and other benchmarks stay on national priors
until a Texas source with n > 400 exists.

### 3d. Time and likely-voter weighting (Silver)

Recency window 30 days now, tightening to 14 days after Oct 15. LV releases
weighted 1.5× RV. After Oct 1 — when Texas supply historically runs 2–3×
September's — the national-derived path (`D_nat + g`) decays to half weight
and Texas legislative banners carry the term outright. Tie the switch to
**observed Texas banner count in the window**, not the calendar: if fewer than
two Texas legislative banners exist in-window, stay on the national-derived
path.

### 3e. Correlated group error in the Monte Carlo (required)

`run_monte_carlo` draws two layers: a national scalar per simulation and an
idiosyncratic draw per district. Add a third, per simulation:

    ε ~ N(0, diag(σ_white², σ_black², σ_hisp², σ_other²))       # one 4-vector per sim
    group_err_d = Σ_r (CVAP_d[r] − w_TX[r]) · ε_r                # per district
    predicted_d = linear_d + national_err + group_err_d + idio_err_d

Centering on `w_TX` keeps the uniform part of any polling miss in the national
layer; what remains is the composition error. A Hispanic miss of +5pp then moves
an 80%-Hispanic district by (0.80 − 0.19)·5 ≈ +3pp and a 10%-Hispanic suburb by
−0.5pp, **in the same simulation** — the "we lost all of the Valley at once"
outcome that independent idio draws cannot produce.

Priors: σ_white 2.5, σ_black 4, σ_hispanic 4.5, σ_other 6 pp. (Texas Hispanic
polling missed by ~8–10pp in 2020 and ~1–2pp in 2024; 4.5 is a middling
one-sigma.) **Estimate rather than assume:** for each clean cycle
(2018/2022/2024) regress that cycle's residuals on the centered CVAP shares;
the cycle-to-cycle spread of those coefficients is σ_g. Three cycles is thin,
so report the estimate next to the prior and use the larger.

**Carve out, don't stack.** High-Hispanic districts' residual variance
currently sits in σ_idio. Refit the national/idio split with the group layer
included, or the total σ overshoots the regression residual — the 0.0801 bug in
a new costume. The validation-skill check is
`sqrt(σ_nat² + E[Var(group_err_d)] + σ_idio²) ≈ regression σ`.

### 3e-bis. What does NOT fade — fundamentals keep full weight (Brennan, 2026-09-09)

Silver's model deprecates fundamentals over the cycle and approaches straight
polling for many races. **This model does not**, and the earlier NEXT-STEPS
item proposing a λ(t) fundamentals weight decaying to ~0 is withdrawn. Reasons:

- There is no district-level polling. Statewide banners give the Texas
  Hispanic share, not HD 74's; a poll of a ~190,000-person House district would
  carry error bars wider than the effects being estimated even if one existed.
- Finance is not a time series to decay. After the July semiannual there are
  two scheduled refreshes — the TEC 30-day and 8-day pre-election reports —
  plus daily late-contribution reports in the final nine days. Two refreshes
  are re-runs of the same term, not a signal that strengthens over time.

So the presidential baseline, incumbency, WAR and finance terms hold their
coefficients through Election Day. The only time-varying element is §3d: which
*source* feeds the racial shift term (national-derived vs Texas banners) and
the environment dial. Polls change the two statewide inputs; they never replace
the district structure.

### 3f. Sequencing

Do this on `refit-clean-cycles`, not master. The refit's intercept (−0.032)
moved by roughly the bias this removes; merging one without the other will look
wrong in both directions. Order:

1. `data/raw/texas_crosstab_inputs.csv` schema + loader (done alongside this doc).
2. §3a shift term with `Δ` from national-only inputs; backtest 2018/2022 (no
   Texas banners exist for those years) and confirm `house_err` falls without
   the constant. If it does not, stop and report — the double-count theory is
   wrong and the constant stays.
3. §3b/§3c Texas offset + shrinkage; re-run 2026.
4. §3e group-error layer; refit σ split; validation harness check.
5. §3d source weighting last — it changes nothing until October. Re-run the
   finance term on the TEC 30-day and 8-day reports as they land; no other
   change to fundamentals.
6. Methodology doc rewrite (it is gitignored under `output/`; move the
   methodology to `docs/` so it is versioned).

## 4. Caveats to keep stating

- Statewide banners give the Texas Hispanic share, not HD 74's.
- The 2024 Black and other benchmarks are national priors, not Texas
  measurements. CES's Texas Black cell (n=238, .752) is not credible.
- Whether a Texas release carries a race banner is a property of the release,
  not the pollster (`config/pollster_registry.csv`). UT's crosstab PDF did this
  time; check every wave.
- `D_TX_now` is 2-party with undecideds dropped; UT's legislative generic had 6%
  "haven't thought about it" among all voters and 8% among Hispanics. Dropping
  them assumes undecideds split like decideds within group.
