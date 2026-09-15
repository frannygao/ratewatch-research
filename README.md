# RateWatch

Estimates the probability of a Bank of Canada rate cut, hold or hike.

Uses six inflation, unemployment and policy-rate indicators in a logistic regression. Historical forecasts use data released before each meeting and are tested against simple baselines.

Across 73 meetings, model accuracy was **74.0%**, versus **76.7%** for repeating the previous decision. The model has not shown an accuracy advantage. Historical data may contain revisions; same-day releases are excluded.

## Files

- `run.py`: runs the project.
- `ratewatch/sources.py`: downloads and loads official data.
- `ratewatch/forecast.py`: builds features, predicts and evaluates.
- `ratewatch/audit.py`: records forecasts for later evaluation.
- `data/`: saved inputs and forecast records.
- `results/`: forecast, evaluation and short report.
- `tests/`: checks timing, calculations and forecast records.

## Run

Python 3.10+ on macOS or Linux. In a Python environment:

```sh
python -m pip install -r requirements.txt
python run.py
```

Read `results/report.md`. Use `python run.py --help` for optional commands.

License: MIT. Official data retain their original terms.
