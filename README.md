# RateWatch

Bank of Canada rate decisions: **cut / hold / hike** probabilities for the next
scheduled meeting, from official data published before the decision.

## What it does

1. **Collects** the policy rate and CPI from the Bank of Canada Valet API,
   unemployment from Statistics Canada, and scrapes every scheduled meeting date
   and decision from the Bank's press releases.
2. **Gates** each meeting on Statistics Canada's release calendar, so a meeting
   only sees the figures that were actually public before its 16:00 cutoff.
3. **Replays** history walk-forward with scikit-learn, refitting at every
   meeting on earlier meetings only. No random train/test split.
4. **Measures** the result against four baselines, split by whether the Bank
   changed course, with a significance test and bootstrap confidence intervals.
5. **Draws** four Matplotlib figures and writes `results/report.md`.

## Run

```bash
pip install -r requirements.txt
python -m ratewatch
```

```bash
python -m ratewatch fetch     # refresh data/ from the official sources
python -m ratewatch record    # log the current forecast, to grade later
```

`fetch` is the only command that needs the network. Everything else reads
`data/` from disk, so a run is reproducible.

## What the record shows

The model scores **74.0%** overall but has **never once called a change of
course**. On the 8 meetings where the Bank started or resumed moving the rate
it forecast hold every time, at 79% to 96% confidence. Its overall gap to simply
repeating the previous decision (76.7%) is two meetings, p = 0.79.

It is a hold detector. Its probabilities are well calibrated (**Brier skill
+0.13**); it just never gets confident enough to call a turn.

![macro trends](results/figures/macro_trends.png)

## Layout

```
ratewatch/
  config.py      paths, sources, every tunable constant
  sources.py     download and load the official data
  features.py    cutoffs, release gating, the six indicators
  model.py       fit, walk-forward backtest, forecast
  metrics.py     scores, splits, significance, intervals
  viz.py         the four figures
  record.py      dated log of forecasts made before the fact
  report.py      render results/report.md
```

Method, formulas and metric definitions: [docs/method.md](docs/method.md).
Tests: `python -m unittest discover -s tests` (40 checks).

License: MIT. Official data retain their original terms.
