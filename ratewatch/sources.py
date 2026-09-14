import gzip
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


series_ids = {'policy': 'V39079', 'cpi': 'V41690973'}
valet = 'https://www.bankofcanada.ca/valet/observations/'
wds = 'https://www150.statcan.gc.ca/t1/wds/rest/'
base = 'https://www.bankofcanada.ca'
release_calendar_url = 'https://www150.statcan.gc.ca/dai-quo/ssi/homepage/schedule-previous_releases-eng.json'
months = 'January|February|March|April|May|June|July|August|September|October|November|December'
root = Path(__file__).resolve().parents[1]


def fetch_data(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'ratewatch-research/1.0'
    sources = {}
    for name, series in series_ids.items():
        url = f'{valet}{series}/json?start_date=1995-01-01'
        response = session.get(url, timeout=40)
        response.raise_for_status()
        payload = response.json()
        (folder / f'{name}.json').write_text(json.dumps(payload))
        sources[name] = {'url': url, 'metadata': payload['seriesDetail']}
    response = session.post(wds + 'getDataFromVectorsAndLatestNPeriods',
                            json=[{'vectorId': 2062815, 'latestN': 500}], timeout=40)
    response.raise_for_status()
    payload = response.json()
    if payload[0]['status'] != 'SUCCESS':
        raise ValueError('Statistics Canada download failed.')
    (folder / 'unemployment.json').write_text(json.dumps(payload))
    sources['unemployment'] = {'url': wds + 'getDataFromVectorsAndLatestNPeriods', 'vector': 2062815}
    manifest = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'sources': sources}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def load_series(folder):
    folder = Path(folder)
    result = {}
    for name, series in series_ids.items():
        payload = json.loads((folder / f'{name}.json').read_text())
        rows = [(r['d'], r[series]['v']) for r in payload['observations'] if series in r]
        result[name] = make_series(rows)
    payload = json.loads((folder / 'unemployment.json').read_text())[0]['object']
    rows = [(r['refPer'], r['value']) for r in payload['vectorDataPoint'] if r.get('value') is not None]
    result['unemployment'] = make_series(rows)
    return result


def make_series(rows):
    frame = pd.DataFrame(rows, columns=['date', 'value'])
    frame['date'] = pd.to_datetime(frame['date'])
    frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
    frame = frame.dropna().sort_values('date').drop_duplicates('date', keep='last')
    if frame.empty or not np.isfinite(frame['value']).all():
        raise ValueError('Empty or invalid source series.')
    return frame.set_index('date')['value']


class Download:
    def __init__(self, folder, refresh=False):
        self.path = Path(folder).with_suffix('.json.gz')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.refresh = refresh
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'RateWatch/2.0 (public economic research)'

    def get(self, url):
        key = hashlib.sha256(url.encode()).hexdigest()
        records = {}
        if self.path.exists():
            with gzip.open(self.path, 'rt') as saved:
                records = json.load(saved)
        if key in records and not self.refresh:
            return records[key]
        time.sleep(0.15)
        response = self.session.get(url, timeout=40)
        response.raise_for_status()
        record = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'body': response.text}
        records[key] = record
        with gzip.open(self.path, 'wt') as saved:
            json.dump(records, saved)
        return record


def parse_schedule(html, year):
    soup = BeautifulSoup(html, 'html.parser')
    for block in soup.select('main ul, main table, main p'):
        text = ' '.join(block.get_text(' ', strip=True).split())
        if 'January' not in text:
            continue
        pattern = rf'(?:(\d{{1,2}})\s+({months})|({months})\s+(\d{{1,2}}))(?:\s+(20\d{{2}}))?'
        dates = []
        for a, b, c, d, y in re.findall(pattern, text):
            if y and int(y) != year:
                continue
            stamp = datetime.strptime(f'{a or d} {b or c} {year}', '%d %B %Y').date()
            dates.append(stamp.isoformat())
        if len(set(dates)) == 8:
            return sorted(set(dates))
    raise ValueError(f'Could not verify eight scheduled meetings for {year}.')


def archive_links(download, year, pages=3):
    result = {}
    for page in range(1, pages + 1):
        query = urlencode({'mtf_search': 'schedule', 'mtf_date_after': f'{year}-01-01',
                           'mtf_date_before': f'{year}-12-31', 'mt_page': page})
        record = download.get(base + '/press/press-releases/?' + query)
        soup = BeautifulSoup(record['body'], 'html.parser')
        links = soup.select('main article h3 a')
        for a in links:
            result[a['href']] = a.get_text(' ', strip=True)
        if not any('Next' in a.get_text() for a in soup.select('main a')):
            break
    return result


def announcement_label(html):
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.select_one('main h1')
    if title is None:
        raise ValueError('Missing announcement headline.')
    text = title.get_text(' ', strip=True)
    for label, pattern in [('hold', r'\b(maintains|maintain|holds|hold|leaves)\b'),
                           ('cut', r'\b(cuts|lowers|reduces)\b'),
                           ('hike', r'\b(raises|increases)\b')]:
        if re.search(pattern, text, re.I):
            return label, text
    raise ValueError('Announcement direction is not unambiguous: ' + text)


def parse_releases(payload, today):
    result = []
    kinds = {'consumer price index': 'cpi', 'labour force survey': 'unemployment'}
    for row in payload:
        kind = kinds.get(row['title'].casefold())
        if not kind or not row.get('url'):
            continue
        try:
            reference = datetime.strptime(row['description'].strip(), '%B %Y').date()
        except ValueError:
            continue
        released = date.fromisoformat(row['date'][:10])
        if released > today:
            continue
        result.append({'series': kind, 'reference': reference.isoformat(),
                       'published_date': released.isoformat(),
                       'available_after': (released + timedelta(days=1)).isoformat(),
                       'precision': 'day; usable next day',
                       'url': urljoin('https://www150.statcan.gc.ca/n1/', row['url'].lstrip('/'))})
    return sorted(result, key=lambda r: (r['series'], r['reference'], r['published_date']))


def refresh_sources(folder, start=2012, end=None):
    folder = Path(folder)
    end = end or date.today().year
    download = Download(folder / 'pages')
    meetings, links = [], {}
    for year in range(start - 1, end + 1):
        print('Official archive:', year, flush=True)
        links[year] = archive_links(Download(folder / 'pages', refresh=True) if year == end else download, year)
    for year in range(start, end + 1):
        schedules = [(u, t) for u, t in links[year - 1].items() if str(year) in t and 'schedule' in t.lower()]
        if not schedules:
            raise ValueError(f'No official schedule found for {year}.')
        url, _ = schedules[0]
        calendar = (Download(folder / 'pages', refresh=True) if year == end else download).get(url)
        dates = parse_schedule(calendar['body'], year)
        for day in dates:
            if date.fromisoformat(day) > date.today():
                meetings.append({'date': day, 'label': None, 'schedule_url': url, 'announcement_url': None})
                continue
            candidates = [u for u in links[year] if day in u and 'fad-press-release' in u]
            if not candidates:
                candidates = [base + f'/{year}/{day[5:7]}/fad-press-release-{day}/']
            announcement = download.get(candidates[0])
            label, title = announcement_label(announcement['body'])
            meetings.append({'date': day, 'label': label, 'headline': title,
                             'schedule_url': url, 'announcement_url': candidates[0],
                             'retrieved_at': announcement['retrieved_at']})
    release_download = Download(folder / 'release_pages', refresh=True)
    calendar = release_download.get(release_calendar_url)
    releases = parse_releases(json.loads(calendar['body']), date.today())
    (folder / 'meetings.json').write_text(json.dumps(meetings, indent=2))
    (folder / 'releases.json').write_text(json.dumps(releases, indent=2))
    (folder / 'manifest.json').write_text(json.dumps({'retrieved_at': datetime.now(timezone.utc).isoformat(),
                                                   'release_calendar': release_calendar_url, 'start_year': start,
                                                   'end_year': end, 'meeting_count': len(meetings),
                                                   'release_count': len(releases)}, indent=2))
    return meetings, releases


def download_data(folder=None):
    folder = Path(folder) if folder else root / 'data'
    fetch_data(folder)
    refresh_sources(folder / 'meetings')


def load_meetings(folder):
    return sorted(json.loads((Path(folder) / 'meetings.json').read_text()), key=lambda row: row['date'])


def load_releases(folder):
    return json.loads((Path(folder) / 'releases.json').read_text())


def load_data(offline=False, folder=None):
    folder = Path(folder) if folder else root / 'data'
    if not offline:
        download_data(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    now = datetime.fromisoformat(manifest['retrieved_at']) if offline else datetime.now(timezone.utc)
    return {'raw': load_series(folder), 'meetings': load_meetings(folder / 'meetings'),
            'releases': load_releases(folder / 'meetings'), 'cutoff': now,
            'offline': offline, 'folder': folder}
