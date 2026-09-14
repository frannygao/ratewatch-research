# RateWatch

## Next BoC meeting

2026-10-28. Saved snapshot preview. Cutoff: 2026-09-13T07:15:33.381966+00:00.

| Decision | Probability |
| --- | ---: |
| Cut | 5.7% |
| Hold | 87.0% |
| Hike | 7.3% |

## Historical evaluation

73 scheduled meetings, 2017-09-06 to 2026-09-02.

| Method | Accuracy | Balanced accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| model | 74.0% | 51.3% | 0.698 | 0.407 |
| frequency | 67.1% | 33.3% | 0.944 | 0.531 |
| transition | 71.2% | 52.8% | 0.704 | 0.399 |
| always_hold | 67.1% | 33.3% | 9.084 | 0.658 |
| persistence | 76.7% | 72.0% | 6.435 | 0.466 |

Deterministic baselines are chiefly accuracy comparisons. Lower log loss and Brier score are better.

## Interpretation

The model assigns the highest probability to a hold. These estimates use six macro indicators; policy-context text does not affect them.
Historical results test whether the macro model adds value; they do not establish a forecasting advantage.

## Prospective record

Deadline: 0 issued, 0 scored.
Early: 1 issued, 0 scored.

Only explicitly issued audit records count as prospective forecasts. Early and deadline forecasts are scored separately.

## Limitations

- Historical inputs use current numerical vintages and may contain revisions.
- Historical results are reconstructions. Same-day releases are excluded.
- Only scheduled decisions are targets; six features omit other policy drivers.
- No timestamped meeting-specific market benchmark is available.
- Local hashes detect edits but cannot prove issue time or prevent wholesale rewriting.
- 2 meetings lacked usable inputs; details are in evaluation.json.
