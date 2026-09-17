# Method

## Sources

| What | Source | Series |
| --- | --- | --- |
| Policy rate | Bank of Canada Valet API | `V39079`, daily |
| CPI | Bank of Canada Valet API | `V41690973`, monthly index |
| Unemployment | Statistics Canada WDS | vector `2062815`, monthly |
| Meeting dates | Bank of Canada schedule releases | scraped, 8 per year |
| Decisions | Bank of Canada announcements | read off each headline |
| Release dates | Statistics Canada release calendar | when each figure was published |

The release calendar is what makes the backtest honest. Knowing *when* a figure
was published lets each meeting see only the numbers that were already public,
instead of the whole series.

## The cutoff

Each meeting's cutoff is 16:00 Toronto on the previous weekday. A figure counts
as available only from the day after it was published, because the release
calendar records dates but not times, so a same-day release is excluded rather
than guessed at.

A full Bank of Canada holiday calendar was tried and moved this date on none of
the 120 meetings in the record, so the simple previous-weekday rule is kept.

## The six indicators

Annual inflation comes from the CPI index:

$$\pi_m = 100\left(\frac{\mathrm{CPI}_m}{\mathrm{CPI}_{m-12}}-1\right)$$

| Indicator | Definition |
| --- | --- |
| Inflation gap | $\pi_m - 2$ |
| Inflation trend | $\pi_m - \pi_{m-3}$ |
| Unemployment change | $u_\ell - u_{\ell-3}$ |
| Unemployment gap | $u_\ell - \frac{1}{36}\sum_{j=0}^{35}u_{\ell-j}$ |
| Real policy rate | $r_d - \pi_m$ |
| Policy momentum | $r_d - r_{d-3\text{ months}}$ |

$m$ and $\ell$ are the latest published inflation and unemployment months, $d$
the day before the cutoff. Rates are in percent, changes in percentage points.

## The model

Standardise the six indicators, then multinomial logistic regression:

$$P(y=k\mid z)=\frac{\exp(b_k+\beta_k^\top z)}{\sum_{c}\exp(b_c+\beta_c^\top z)}$$

Fitted by minimising log loss with an L2 penalty (`C=0.5`). The three
probabilities sum to one; the largest is the predicted decision. This forecasts
direction, not the size of a move.

The specification is fixed in `data/protocol.json` and is never tuned against
the backtest, because a model picked for its score on these meetings would have
a meaningless score.

## The backtest

A walk-forward replay. At each meeting the scaler and the regression are refitted
from scratch on earlier meetings only, so nothing from the future leaks
backwards. A meeting is scored once 40 earlier meetings exist and all three
decisions have been seen, which is why 116 usable meetings yield 73 scored ones.

## Scores

- **Accuracy**: share of meetings where the highest probability was right
- **Balanced accuracy**: mean of the per-class recalls, so rare cuts and hikes
  count as much as common holds
- **Log loss**: mean of $-\log$(probability given to what happened)
- **Brier score**: summed squared error across the three probabilities, 0 to 2

$$\mathrm{LogLoss}=-\frac{1}{N}\sum_{i}\log p_{i,y_i},\qquad
\mathrm{Brier}=\frac{1}{N}\sum_{i}\sum_k\left(p_{i,k}-\mathbf{1}[y_i=k]\right)^2$$

### And three things a single accuracy number cannot tell you

**A split by meeting type.** Roughly three quarters of meetings repeat the
previous decision, so overall accuracy mostly measures how often nothing
happened. Meetings are classified as *new move* (the Bank changed course and
moved), *returned to hold*, or *unchanged*.

**A significance test.** McNemar's test on the meetings where two methods
disagreed, using an exact two-sided binomial test against a fair coin. A
p-value near 1 means the record cannot tell the two methods apart.

**Bootstrap confidence intervals.** Meetings are resampled with replacement
20,000 times and the score recomputed, giving the middle 95%.

Plus a **Brier skill score** against the persistence baseline: the share of its
squared error the model removes. Zero means no better, negative means worse.

### Why log loss is blank for some baselines

A method forecasting only 0% or 100% takes an unbounded penalty when it is
wrong, so its log loss is decided by the floor used to keep the logarithm finite
rather than by the forecast. With a floor of `1e-12`, persistence's log loss is
exactly $27.631 \times$ its error rate and nothing else. Reporting that number
would invite a comparison that means nothing, so it is left as `n/a`.

Note also that the Brier score of a 0/100 forecast is just twice its error rate,
so it carries no information beyond accuracy either.

## Baselines

| Baseline | What it forecasts |
| --- | --- |
| `frequency` | how often each decision has happened, lightly smoothed |
| `transition` | what followed this same decision in the past |
| `always_hold` | always hold |
| `persistence` | repeat the previous decision, the one to beat |

## Prospective record

The backtest is a reconstruction: release *dates* are point-in-time but the
*values* are not, because it uses today's revised figures. The only clean score
is a forecast written down before the decision, which is what
`python -m ratewatch record` does. Forecasts are grouped by protocol version and
scored separately, so a change to the model cannot borrow the old one's record.

## Limits

- Historical figures carry later revisions.
- Only six indicators, all from CPI, unemployment and the policy rate. The Bank
  discusses core inflation measures (CPI-trim, CPI-median), the output gap, oil
  prices and US policy at least as often, and none of those are inputs here.
- The panel starts in 2012, as far back as the release calendar reaches, leaving
  roughly 27 cuts and hikes to learn from against 21 fitted parameters.
- Only scheduled decisions are tested; inter-meeting moves are out of scope.
- A lexicon over previous announcement text was tested and did not help.
