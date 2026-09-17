# RateWatch

Estimates the probability of a Bank of Canada rate cut, hold or hike.

Uses six inflation, unemployment and policy-rate indicators in a logistic regression. Historical forecasts use data released before each meeting and are tested against simple baselines.

## Files

- `ratewatch/config.py`: paths, data sources and constants.
- `ratewatch/sources.py`: downloads and loads official data.
- `ratewatch/features.py`: builds features from data published before each meeting.
- `ratewatch/model.py`: fits, backtests and forecasts.
- `ratewatch/metrics.py`: scores forecasts against the baselines.
- `ratewatch/record.py`: records forecasts for later evaluation.
- `ratewatch/viz.py`: draws the figures.
- `ratewatch/report.py`: writes the report.
- `data/`: saved inputs and forecast records.
- `results/`: forecast, evaluation, figures and short report.
- `tests/`: checks timing, calculations and forecast records.
- `docs/method.md`: formulas and metric definitions.

## Run

Python 3.10+ on macOS or Linux. In a Python environment:

```sh
python -m pip install -r requirements.txt
python -m ratewatch
```
