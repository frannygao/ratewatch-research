# RateWatch

Estimates the probability of a Bank of Canada rate cut, hold or hike at the next scheduled meeting.

Official macro data -> release-aware features -> cut/hold/hike probabilities -> walk-forward evaluation.

Six features describe the inflation gap and three-month trend, unemployment change and gap against its 36-month average, real policy rate, and three-month policy momentum. A StandardScaler and multinomial logistic regression (`C=0.5`) train on earlier meetings, starting with 40 observations and all three outcomes. Inflation uses the Bank's 2% target.

The saved reconstruction tests 73 meetings: model accuracy **74.0%**, persistence **76.7%**, model log loss **0.698**, and Brier score **0.407**. Frequency, transition and always-hold baselines are also reported. These results do not establish a forecasting advantage.

Historical cutoffs are 16:00 Toronto time on the preceding banking business day. Reference periods, publication dates, cutoffs and meeting dates remain separate. Same-day releases are excluded. Numerical data use current vintages and may contain revisions; this is a reconstruction, not an untouched live test. Only scheduled decisions are targets, and six features cannot represent every policy driver.

## Files

- `run.py`: load data, evaluate, forecast and save results.
- `ratewatch/sources.py`: official series, meetings, release dates and cached provenance.
- `ratewatch/forecast.py`: six features, model, baselines, evaluation and report.
- `ratewatch/context.py`: optional Bank of Canada document collection and topic matching.
- `ratewatch/audit.py`: prospective records, duplicate checks, hashes and resolved scoring.
- `data/`: saved inputs, source pages, protocol and original audit records.
- `results/`: `forecast.json`, `evaluation.json`, `report.md` and optional `context.md`.
- `tests/`: timing, features, chronological training, probabilities and audit checks.
- `requirements.txt`, `.gitignore`, `LICENSE`: dependencies, ignored files and MIT terms.

## Run

Python 3.10+ on macOS or Linux. The audit writer uses POSIX file locking.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --offline
python -m unittest discover -s tests
```

`--offline` reproduces the saved September 13, 2026 data and cutoff, not a live forecast. Run `python run.py` to refresh official data and preview a forecast. Results contain named cut/hold/hike probabilities, inputs, diagnostics and historical predictions.

`python run.py --issue` downloads fresh data and records a forecast under `data/audit/`. One record is allowed per meeting, protocol and stage. The deadline stage is the last hour before the cutoff; earlier issues are scored separately. Records retain issue time, inputs and fitted parameters. Local hashes detect edits but cannot prove issue time or prevent wholesale rewriting.

Optional context: `python -m ratewatch.context --offline --as-of 2026-09-13`. Omit `--offline` to collect official HTML documents. Policy-context text does not affect the numerical forecast probabilities. PDF attachments are not parsed.

License: MIT. Source data and Bank of Canada documents retain their original terms.
