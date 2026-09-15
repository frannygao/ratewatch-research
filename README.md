# RateWatch

Estimates the probability of a Bank of Canada rate cut, hold or hike.

## How it predicts

Each meeting is one observation, labelled **cut**, **hold** or **hike**. The model uses official CPI, unemployment and policy-rate data available before the decision.

Annual inflation is calculated from the CPI index:

$$
\pi_m = 100\left(\frac{\mathrm{CPI}_m}{\mathrm{CPI}_{m-12}}-1\right)
$$

The six inputs are:

| Input | Calculation |
| --- | --- |
| Inflation gap | $\pi_m - 2$ |
| Inflation trend | $\pi_m - \pi_{m-3}$ |
| Unemployment change | $u_\ell - u_{\ell-3}$ |
| Unemployment gap | $u_\ell - \frac{1}{36}\sum_{j=0}^{35}u_{\ell-j}$ |
| Real policy rate | $r_d - \pi_m$ |
| Policy momentum | $r_d - r_{d-3\text{ months}}$ |

Here, $m$ and $\ell$ are the latest released inflation and unemployment months; $d$ is the day before the cutoff. Policy rates use the last observation on or before each comparison date. Rates are in percent; changes are in percentage points.

Inputs are standardized using the training data's mean and standard deviation: $z_j=(x_j-\mu_j)/\sigma_j$. Multinomial logistic regression learns a coefficient vector $\beta_k$ and intercept $b_k$ for each decision, then converts their scores into probabilities:

$$
P(y=k\mid z)=\frac{\exp(b_k+\beta_k^\top z)}{\sum_{c\in\{\mathrm{cut},\mathrm{hold},\mathrm{hike}\}}\exp(b_c+\beta_c^\top z)}
$$

The coefficients are fitted by minimizing classification log loss with an L2 penalty (`C=0.5`) to discourage large coefficients. The three probabilities sum to one; the largest determines the predicted decision. This predicts direction, not the size of a rate change.

## Does it work?

An expanding-window test starts with at least 40 earlier meetings and all three outcomes. At each subsequent meeting, the scaler and model are refitted using only earlier decisions. There is no random train/test split.

Across 73 tested meetings, model accuracy was **74.0%**, versus **76.7%** for repeating the previous decision. The model has not shown an accuracy advantage. Frequency, transition and always-hold baselines are also evaluated.

Probability quality is measured by log loss and the multiclass Brier score:

$$
\mathrm{LogLoss}=-\frac{1}{N}\sum_{i=1}^{N}\log\max(p_{i,y_i},10^{-12}),\qquad
\mathrm{Brier}=\frac{1}{N}\sum_{i=1}^{N}\sum_k\left(p_{i,k}-\mathbf{1}[y_i=k]\right)^2
$$

Here, $p_{i,k}$ is the forecast probability for decision $k$, and $y_i$ is the actual decision. Lower is better. The model scored **0.698** log loss and **0.407** Brier score. Balanced accuracy, the average recall across observed classes, is also reported.

Historical cutoffs are 16:00 Toronto time on the preceding banking business day. Same-day releases are excluded. Numerical data may contain revisions, so historical results are reconstructions. Saved prospective forecasts allow later evaluation on new decisions.

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
