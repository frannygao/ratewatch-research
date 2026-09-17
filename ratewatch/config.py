"""Paths, data sources, and every tunable constant in one place.

Nothing here does any work. If you want to know what the model reads, how far
back the history goes, or which series a number came from, it is all below.
"""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

# --- where things live ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FIGURES = RESULTS / 'figures'
MEETINGS = DATA / 'meetings'
FORECAST_LOG = DATA / 'audit'
PROTOCOL = DATA / 'protocol.json'

TORONTO = ZoneInfo('America/Toronto')
LABELS = ['cut', 'hold', 'hike']

# --- what the model reads ------------------------------------------------------

FEATURES = [
    'inflation_gap',          # annual CPI inflation, less the 2% target
    'inflation_trend',        # 3-month change in that inflation rate
    'unemployment_change',    # 3-month change in the unemployment rate
    'unemployment_gap',       # unemployment, less its own 36-month average
    'real_rate',              # policy rate, less inflation
    'policy_momentum',        # policy rate, less the rate 3 months ago
]

INFLATION_TARGET = 2.0        # the Bank's stated target, in percent
TREND_MONTHS = 3              # window for the "change over recent months" features
NORMAL_MONTHS = 36            # window defining a normal level of unemployment
STALE_DAYS = 65               # refuse to forecast on figures older than this

# --- the model, deliberately frozen; see data/protocol.json --------------------

PENALTY = 0.5                 # sklearn's C: smaller means stronger shrinkage
MAX_ITER = 2000
MIN_TRAIN = 40                # earlier meetings required before a forecast is scored

BASELINES = ['frequency', 'transition', 'always_hold', 'persistence']
METHODS = ['model'] + BASELINES
REFERENCE = 'persistence'     # the baseline everything is compared against

# --- measurement ---------------------------------------------------------------

LOG_FLOOR = 1e-12             # keeps log loss finite on a zero probability
BOOTSTRAP_DRAWS = 20000
CONFIDENCE = 0.95

MEETING_TYPES = {
    'new move': 'The Bank changed course and moved the rate.',
    'returned to hold': 'The Bank stopped moving and held.',
    'unchanged': 'The Bank repeated its previous decision.',
}

# --- where the data comes from -------------------------------------------------

USER_AGENT = 'ratewatch-research/2.0 (public economic research)'
TIMEOUT = 40
CRAWL_PAUSE = 0.15            # seconds between page requests

VALET = 'https://www.bankofcanada.ca/valet/observations/'
VALET_SERIES = {'policy': 'V39079', 'cpi': 'V41690973'}

WDS = 'https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods'
WDS_UNEMPLOYMENT = 2062815    # Labour Force Survey unemployment rate, Canada, SA
WDS_PERIODS = 500

BOC = 'https://www.bankofcanada.ca'
RELEASE_CALENDAR = ('https://www150.statcan.gc.ca/dai-quo/ssi/homepage/'
                    'schedule-previous_releases-eng.json')
STATCAN_DAILY = 'https://www150.statcan.gc.ca/n1/'

MONTH_NAMES = ('January|February|March|April|May|June|'
               'July|August|September|October|November|December')
MEETINGS_PER_YEAR = 8
FIRST_YEAR = 2012             # as far back as the release calendar reaches

# --- figures -------------------------------------------------------------------

COLOURS = {'cut': '#c0392b', 'hold': '#7f8c8d', 'hike': '#27ae60',
           'model': '#2c6fbb', 'reference': '#c9a227', 'rule': '#95a5a6'}
FIGURE_DPI = 140
