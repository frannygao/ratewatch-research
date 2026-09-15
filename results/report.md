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

Model accuracy did not exceed the repeat-previous-decision baseline on this sample.
Lower log loss and Brier score are better; deterministic baselines are mainly accuracy comparisons.

Historical values may be revised. Same-day releases are excluded. Only scheduled decisions are tested, and six indicators omit other policy drivers.
2 meetings lacked usable inputs.
