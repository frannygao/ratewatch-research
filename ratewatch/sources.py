"""Download and load the official data the forecast is built from.

Everything comes from a primary source:

    policy rate   Bank of Canada Valet API, series V39079 (daily)
    CPI           Bank of Canada Valet API, series V41690973 (monthly index)
    unemployment  Statistics Canada WDS, vector 2062815 (monthly rate)
    meetings      Bank of Canada press releases: the scheduled announcement
                  dates, and the decision each announcement made
    releases      Statistics Canada release calendar: the day each CPI and
                  unemployment figure was actually published

That last one is what makes the backtest honest. Knowing when a figure was
published lets every meeting see only the numbers that were already public
before it, instead of the whole series.

`fetch()` needs the network and writes into data/. Nothing else does: every
other module reads the saved files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import (BOC, CRAWL_PAUSE, DATA, FIRST_YEAR, MEETINGS_PER_YEAR, MONTH_NAMES,
                     RELEASE_CALENDAR, STATCAN_DAILY, TIMEOUT, USER_AGENT, VALET,
                     VALET_SERIES, WDS, WDS_PERIODS, WDS_UNEMPLOYMENT)

__all__ = ['fetch', 'load_meetings', 'load_series']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Cache:
    """URL to page body, read once at the start of a fetch and written once at the end.

    The pages are kept so a re-fetch does not hammer the Bank's site, and so the
    scraped labels stay reproducible.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.records = {}
        if self.path.exists():
            with gzip.open(self.path, 'rt') as saved:
                self.records = json.load(saved)
        self.session = requests.Session()
        self.session.headers['User-Agent'] = USER_AGENT

    def get(self, url: str, refresh: bool = False) -> dict[str, str]:
        key = hashlib.sha256(url.encode()).hexdigest()
        if key in self.records and not refresh:
            return self.records[key]
        time.sleep(CRAWL_PAUSE)
        response = self.session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        self.records[key] = {'url': url, 'retrieved_at': now(), 'body': response.text}
        return self.records[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.path, 'wt') as saved:
            json.dump(self.records, saved)


# ---------------------------------------------------------------- numeric series

def fetch_series(folder: Path) -> None:
    """Save the three numeric series and a manifest recording where they came from."""
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    sources = {}

    for name, series in VALET_SERIES.items():
        url = f'{VALET}{series}/json?start_date=1995-01-01'
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        (folder / f'{name}.json').write_text(json.dumps(response.json()))
        sources[name] = {'url': url, 'series': series}

    response = session.post(WDS, json=[{'vectorId': WDS_UNEMPLOYMENT, 'latestN': WDS_PERIODS}], timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if payload[0]['status'] != 'SUCCESS':
        raise ValueError('Statistics Canada download failed.')
    (folder / 'unemployment.json').write_text(json.dumps(payload))
    sources['unemployment'] = {'url': WDS, 'vector': WDS_UNEMPLOYMENT}

    (folder / 'manifest.json').write_text(
        json.dumps({'retrieved_at': now(), 'sources': sources}, indent=2))


def to_series(rows: list[tuple[str, Any]]) -> pd.Series:
    frame = pd.DataFrame(rows, columns=['date', 'value'])
    frame['date'] = pd.to_datetime(frame['date'])
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame = frame.dropna().sort_values('date').drop_duplicates('date', keep='last')
    if frame.empty or not np.isfinite(frame['value']).all():
        raise ValueError('Empty or invalid source series.')
    return frame.set_index('date')['value']


def load_series(folder: Path | str = DATA) -> dict[str, pd.Series]:
    """The three numeric series, as pandas Series indexed by date."""
    folder = Path(folder)
    series = {}
    for name, code in VALET_SERIES.items():
        payload = json.loads((folder / f'{name}.json').read_text())
        series[name] = to_series([(r['d'], r[code]['v']) for r in payload['observations'] if code in r])
    payload = json.loads((folder / 'unemployment.json').read_text())[0]['object']
    series['unemployment'] = to_series(
        [(r['refPer'], r['value']) for r in payload['vectorDataPoint'] if r.get('value') is not None])
    return series


# ------------------------------------------------------------- meetings & labels

def parse_schedule(html: str, year: int) -> list[str]:
    """Pull the year's announcement dates out of the Bank's schedule release.

    Refuses anything that is not exactly eight dates, so a layout change fails
    loudly instead of silently producing a short calendar.
    """
    soup = BeautifulSoup(html, 'html.parser')
    pattern = rf'(?:(\d{{1,2}})\s+({MONTH_NAMES})|({MONTH_NAMES})\s+(\d{{1,2}}))(?:\s+(20\d{{2}}))?'
    for block in soup.select('main ul, main table, main p'):
        text = ' '.join(block.get_text(' ', strip=True).split())
        if 'January' not in text:
            continue
        dates = set()
        for day_first, month_a, month_b, day_second, found_year in re.findall(pattern, text):
            if found_year and int(found_year) != year:
                continue
            stamp = datetime.strptime(f'{day_first or day_second} {month_a or month_b} {year}', '%d %B %Y')
            dates.add(stamp.date().isoformat())
        if len(dates) == MEETINGS_PER_YEAR:
            return sorted(dates)
    raise ValueError(f'Could not verify {MEETINGS_PER_YEAR} scheduled meetings for {year}.')


def read_label(html: str) -> tuple[str, str]:
    """Read cut / hold / hike off an announcement headline."""
    title = BeautifulSoup(html, 'html.parser').select_one('main h1')
    if title is None:
        raise ValueError('Missing announcement headline.')
    text = title.get_text(' ', strip=True)
    for label, pattern in [('hold', r'\b(maintains|maintain|holds|hold|leaves)\b'),
                           ('cut', r'\b(cuts|lowers|reduces)\b'),
                           ('hike', r'\b(raises|increases)\b')]:
        if re.search(pattern, text, re.I):
            return label, text
    raise ValueError('Announcement direction is not unambiguous: ' + text)


def press_releases(cache: Cache, year: int, refresh: bool,
                   pages: int = 3) -> dict[str, str]:
    """Every press-release link the Bank filed under a given year."""
    links = {}
    for page in range(1, pages + 1):
        query = urlencode({'mtf_search': 'schedule', 'mtf_date_after': f'{year}-01-01',
                           'mtf_date_before': f'{year}-12-31', 'mt_page': page})
        record = cache.get(f'{BOC}/press/press-releases/?{query}', refresh=refresh)
        soup = BeautifulSoup(record['body'], 'html.parser')
        for anchor in soup.select('main article h3 a'):
            links[anchor['href']] = anchor.get_text(' ', strip=True)
        if not any('Next' in a.get_text() for a in soup.select('main a')):
            break
    return links


def fetch_meetings(folder: Path, first_year: int = FIRST_YEAR) -> None:
    """Scrape the meeting calendar and the decision each past meeting made.

    A year's schedule is announced in the preceding year, so the archive is read
    from first_year - 1. The current year is always re-fetched, because its pages
    are still changing.
    """
    folder.mkdir(parents=True, exist_ok=True)
    cache = Cache(folder / 'pages.json.gz')
    last_year = date.today().year
    links = {}
    for year in range(first_year - 1, last_year + 1):
        print(f'  reading {year} press releases', flush=True)
        links[year] = press_releases(cache, year, refresh=year == last_year)

    meetings = []
    for year in range(first_year, last_year + 1):
        schedules = [url for url, title in links[year - 1].items()
                     if str(year) in title and 'schedule' in title.lower()]
        if not schedules:
            raise ValueError(f'No official schedule found for {year}.')
        schedule_url = schedules[0]
        calendar = cache.get(schedule_url, refresh=year == last_year)
        for day in parse_schedule(calendar['body'], year):
            if date.fromisoformat(day) > date.today():
                meetings.append({'date': day, 'label': None, 'schedule_url': schedule_url})
                continue
            found = [url for url in links[year] if day in url and 'fad-press-release' in url]
            url = found[0] if found else f'{BOC}/{year}/{day[5:7]}/fad-press-release-{day}/'
            announcement = cache.get(url)
            label, headline = read_label(announcement['body'])
            meetings.append({'date': day, 'label': label, 'headline': headline,
                             'schedule_url': schedule_url, 'announcement_url': url,
                             'retrieved_at': announcement['retrieved_at']})

    calendar = cache.get(RELEASE_CALENDAR, refresh=True)
    releases = parse_releases(json.loads(calendar['body']))
    cache.save()

    (folder / 'meetings.json').write_text(json.dumps(meetings, indent=2))
    (folder / 'releases.json').write_text(json.dumps(releases, indent=2))
    (folder / 'manifest.json').write_text(json.dumps(
        {'retrieved_at': now(), 'release_calendar': RELEASE_CALENDAR, 'first_year': first_year,
         'meetings': len(meetings), 'releases': len(releases)}, indent=2))


def parse_releases(payload: list[dict[str, Any]]) -> list[dict[str, str]]:
    """When each CPI and unemployment figure was published.

    `available_after` is the day after publication. The calendar records dates
    but not times, so a figure published on the day of a cutoff is treated as
    not yet available rather than guessed at.
    """
    wanted = {'consumer price index': 'cpi', 'labour force survey': 'unemployment'}
    today = date.today()
    releases = []
    for row in payload:
        series = wanted.get(row['title'].casefold())
        if not series or not row.get('url'):
            continue
        try:
            reference = datetime.strptime(row['description'].strip(), '%B %Y').date()
        except ValueError:
            continue
        published = date.fromisoformat(row['date'][:10])
        if published > today:
            continue
        releases.append({'series': series, 'reference': reference.isoformat(),
                         'published': published.isoformat(),
                         'available_after': (published + timedelta(days=1)).isoformat(),
                         'url': urljoin(STATCAN_DAILY, row['url'].lstrip('/'))})
    return sorted(releases, key=lambda r: (r['series'], r['reference'], r['published']))


def load_meetings(folder: Path | str = DATA) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    path = Path(folder) / 'meetings'
    meetings = sorted(json.loads((path / 'meetings.json').read_text()), key=lambda r: r['date'])
    releases = json.loads((path / 'releases.json').read_text())
    return meetings, releases


def fetch(folder: Path | str = DATA) -> None:
    """Refresh everything under data/. The only function here that needs the network."""
    folder = Path(folder)
    print('Fetching numeric series')
    fetch_series(folder)
    print('Fetching meeting calendar and decisions')
    fetch_meetings(folder / 'meetings')
    print(f'Saved to {folder}')
