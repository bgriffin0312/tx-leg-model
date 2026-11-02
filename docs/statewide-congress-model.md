# Texas statewide + congressional model (built 2026-10-30)

SNAPSHOT as of 2026-10-30 — forecast numbers below are not current after that date.

One simulated Texas environment prices all eight statewide races and all 38 U.S.
House seats on PlanC2333, so their errors move together. Separate from the
legislative model; it reads none of that model's coefficients.

**Run order** (all in the repo root):

```
python scripts/statewide_fetch_wikitext.py      # + curl the House and Land pages, see below
python src/statewide_collect.py                 # 2026 statewide polls
python src/statewide_results.py                 # TLC results 2002-2024, PlanC2333 baseline
python src/statewide_candidates.py              # 38 races, ratings, district polls
python src/statewide_finance.py                 # TEC (with SPACs) + FEC
python src/statewide_war.py                     # district WAR and transfer tests
python scripts/statewide_pollster_history.py    # pollster record
python src/statewide_model.py                   # fit, backtest, forecast, sensitivity
python scripts/statewide_compare_outside.py DIR # outside check, after freezing
```

`data/raw/statewide/` is gitignored. Map files come from data.capitol.texas.gov
(planc2333): `r110.xls` (exact VTD assignment), `r202b.pdf` (demographics),
`r119_24.pdf` (SSVR).

## Structure

T is the Texas party margin with candidates removed, measured as the Supreme
Court + CCA composite. It runs near the imputed Texas House vote but not on it: J minus
House −0.9 (2014), +0.2 (2018), −2.4 (2022), −0.7 (2024).

- Statewide race: `T + dev_r`
- Congressional seat: `T + c + L_d + money + WAR + h*SSVR + e_d`
- Any poll: `true margin + b + noise`, where b is the cycle's shared miss

Linear-Gaussian, so conditioning on polls is exact and the weight on polls follows
from the measured error sizes. District residuals are drawn with t(5) tails.

## Estimates, all from Texas races of the same kind

| Effect (margin pts) | Statewide (n=52–55, 2002–24) | U.S. House (n=210, 2012–24) |
|---|---|---|
| Incumbency, total | +2.0 (se 1.4); no out-of-sample gain | R +2.7 (se 1.1), D +0.7 (se 1.4) |
| Incumbency after money + WAR | same | ≈0: absorbed |
| Money, per log-unit of D/R receipts | +1.54 (se 0.51); LOCO 6.07→5.39 | +1.08 (se 0.26); LOCO 5.70→5.54 |
| Prior record | 0.80 × shrunk prior statewide dev (t 1.5) | 0.43 × prior WAR (t 5.3); LOCO →5.31 |
| Residual sd | low 4.2, mid 5.0, top 5.0 | 4.4 within cycle; 5.0 in races within 15 |

Hispanic slope varies by cycle (sd 1.31 per 10 points SSVR) and is carried as a shared
shock, not a mean term; a fixed term failed out of sample (5.31→5.36).

T prior: J change on national change, slope 0.95 (se 0.10), LOO 2.74. A midterm step
was tested and dropped (−0.85, se 0.84; LOO 2.79; joint backtest 3.07 vs 2.88).

## First-time statewide candidates

Overperformance against the judicial vote on the same VTDs, net of cycle, incumbency,
lean and SSVR, 1,280 contested district races 2002–2024. TLC returns carry surnames
back to 2002.

| Transfer | Slope |
|---|---|
| Same seat, next race | 0.52 (se 0.03, n 849) |
| Change of office type, surname match | 0.18 (se 0.12, n 72; includes false matches) |
| Same, held the first office | −0.18 (se 0.25, n 28) |
| Into a first statewide run | 0.37 (se 0.33, n 11; permutation p 0.39) |

Applied at 0.2 × n/(n+2.93). No race moves more than 0.8. The profile adjustment did
not beat raw overperformance at n=11 (r 0.29 vs 0.43), which that n cannot settle.
Eckhardt and Sheets have no usable record; Middleton's SD 11 and Eckhardt's SD 14 had
no two-party general.

## Pollster record

120 polls in the last 35 days, 2014–2024. House effects are not separable from noise
(mean |effect| 1.23, permutation p 0.17). Walk-forward rms: plain average 4.21, house
correction 4.29, record weighting 4.39. **So the 2026 average uses no pollster
ratings**, only recency and sample size, excluding party-stake sponsors. The shared
cycle miss is what matters: sd 3.69 across cycles against 1.44 between races in a
cycle. Midterms −0.55, presidential years +3.50. Undecided share does not predict a
direction (−0.08 per point). The poll-miss mean is set at +0.5, between the midterm
and all-cycle means; that is a judgment, and the sensitivity table prices it.

## Backtest

Leave-one-cycle-out, 13 race-cycles with polls: polls 3.27, structure 4.09, joint 2.88.
Stated 80% band covered 77%. Error sizes were measured on the same cycles, so this
is mildly optimistic.

## Frozen forecast (15:16 CDT 2026-10-30), then the outside check

Statewide, D margin (P D win): Senate +2.1 (74%), Governor −3.5 (14%), RRC −3.0 (19%),
AG −4.8 (9%), Lt Gov −4.9 (8%), Ag −6.1 (6%), Comptroller −6.7 (3%), Land −9.4 (1%).
U.S. House: 11.8 D seats (10–13); TX-15 +4.5 (87%), TX-34 +4.5 (86%), TX-35 +0.5
(55%), TX-23 −2.6 (27%), TX-28 +13.0 (>99%), TX-9 −8.3 (2%).

Split Ticket/The Argument, tables dated 10/27 (House) and 10/29 (Senate), opened after
the freeze: Senate D+2.7 (74%), 12.0 expected seats. All 38 seats corr 0.998, mean gap
−1.7 (they run bluer, mostly in safe seats); competitive 18: gap −1.0, sd 1.6. Main
differences: TX-23 (they 52%, us 27%), TX-15 (they 67%, us 87%).

**Trap:** the article's embeds are pinned to June 17 versions. Read the unversioned
`datawrapper.dwcdn.net/<id>/` URL to get the current version number first.

## Changes after the freeze

**French controversy term, 2026-10-30 (Brennan).** +2.0 D margin in the RRC prior
(`SCANDAL_2026`), set after Split Ticket was opened; the frozen numbers above stay as the
record of the holdout. RRC moves R+3.0 / 19% → R+2.4 / 23%; no other race or seat moves more
than about 0.1. `scripts/scandal_tx23_tests.py`: six statewide cases, four people, mean −3.6 per person
(p 0.06), −2.8 without Cruz. Adding Cuellar as a backtest case (no forecast effect) breaks
it: after the 2022 raid and 2024 indictment he ran +10.6 and +5.3 against the U.S. House
layer, about his earlier +7.6 average, so the pooled mean is −1.3 (p 0.27), or −0.1
without Cruz. Scored against his own earlier level (+0.3) it is −2.8. Miller also ran
behind before his scandal (−3.7 in 2014, −2.4 in 2018), so the two cases with a pre-scandal
record both show no drop. No penalty is established, and the 2 is a judgment. Other weaknesses: cases chosen with outcomes roughly
known, Paxton is half of the statewide cases, a race residual belongs to both candidates,
and 24% of unflagged races run as far behind (Patrick 2018 and 2022, with no flag). Paxton
stays off: three runs at −2.6, −3.4, −4.4, carried by `prior_own`. The cases do not separate a
new scandal from a persistent candidate penalty, because each Paxton run had fresh news.

**TX-23.** Ours R+2.6 / 27%, Split Ticket D+0.2 / 52%. The prior is R+4.8; without its
only windowed poll the South Texas shift alone gives R+0.9. That poll (Normington Petts
for HMP, tied) reads R+4 after the −4 internal haircut, which costs 1.7 of the 2.8 gap.
Remington R+6 (July 25–28, n=572, sponsor not stated, first reported by Breitbart) is not
on Wikipedia and falls outside the 75-day window, so it is not in the model; adding it
gives 15–23%. Their inputs are not public, but they run 1.3 points bluer than us in open
seats than in incumbent seats, and this model has no incumbency loss for an open R seat
and no candidate flag for Herrera.

## Known gaps

- Texas district polls have no back file; the ±4 internal adjustment and 6-pt noise
  are conventions, and South Texas leans on TSU and D internals. Without district polls
  the seat mean is 10.4 rather than 11.8.
- No early-vote data, no Trump-approval term, no candidate-quality flag for
  non-incumbents (Herrera in TX-23 is the live case).
- Statewide T sits about 2.5 points left of the legislative model's roughly R+7.2 after
  that model's 2026-10-30 differenced refit (it was R+8.3). The gap predates polls here
  (prior R+5.1 → posterior R+4.7). What remains is mainly the midterm step tested here
  and dropped (T would be R+6.3), the legislative slope of 0.0034 against 0.95 margin per
  point here, and the 0.4 starting baseline (J −16.3 vs open composite −16.7). The two
  quantities are also not identical (J vs a House vote).
