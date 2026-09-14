import argparse
import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup


factors = [
    {'id': 'inflation', 'name': 'Inflation and persistence',
     'terms': ['inflation', 'cpi', 'price pressures', 'core inflation'],
     'features': ['inflation_gap', 'inflation_trend'],
     'missing': 'core inflation, breadth, services inflation, short-run seasonally adjusted momentum',
     'reason': 'Persistent price pressure affects the inflation outlook; temporary shocks can reverse.'},
    {'id': 'expectations', 'name': 'Inflation expectations and credibility',
     'terms': ['inflation expectations', 'expectations of inflation', 'pricing behaviour', 'price-setting', 'anchored'],
     'features': [], 'missing': 'dated household and business expectation series',
     'reason': 'Expectations can affect wage setting, pricing and the persistence of inflation.'},
    {'id': 'labour', 'name': 'Employment and labour supply',
     'terms': ['unemployment', 'employment', 'labour market', 'labor market', 'labour shortages', 'participation'],
     'features': ['unemployment_change', 'unemployment_gap'],
     'missing': 'employment growth, hours, vacancies, participation and underemployment',
     'reason': 'Labour conditions help assess demand and supply; unemployment alone cannot distinguish them.'},
    {'id': 'wages', 'name': 'Wages and unit labour costs',
     'terms': ['wage', 'wages', 'unit labour costs', 'compensation'],
     'features': [], 'missing': 'wage growth adjusted for composition and productivity',
     'reason': 'Wages relative to productivity can affect costs, household income and service prices.'},
    {'id': 'demand', 'name': 'GDP, consumption and investment',
     'terms': ['gdp', 'economic growth', 'consumption', 'consumer spending', 'business investment', 'domestic demand'],
     'features': [], 'missing': 'GDP growth, GDP per capita, retail consumption and capital spending',
     'reason': 'Demand growth relative to productive capacity changes inflation pressure.'},
    {'id': 'capacity', 'name': 'Output gap and productive capacity',
     'terms': ['output gap', 'excess supply', 'excess demand', 'capacity', 'productivity', 'potential output'],
     'features': [], 'missing': 'real-time output-gap estimates, capacity use and productivity',
     'reason': 'Demand above capacity can create pressure; potential output is uncertain and revised.'},
    {'id': 'housing', 'name': 'Housing and shelter costs',
     'terms': ['housing', 'shelter', 'mortgage', 'mortgages', 'rents', 'rent inflation'],
     'features': [], 'missing': 'housing activity, rent inflation, renewals and debt-service costs',
     'reason': 'Housing affects demand and CPI; interest-rate increases can initially raise measured mortgage costs.'},
    {'id': 'credit', 'name': 'Credit conditions and financial stress',
     'terms': ['credit', 'financial conditions', 'financial stability', 'financial stress', 'liquidity', 'banking'],
     'features': [], 'missing': 'credit spreads, lending standards, defaults and funding stress',
     'reason': 'Credit tightening can weaken demand; stability tools and policy rates serve distinct purposes.'},
    {'id': 'markets', 'name': 'Market rate expectations',
     'terms': ['bond yields', 'government bond', 'yield curve', 'market expectations', 'overnight index'],
     'features': [], 'missing': 'meeting-specific OIS pricing and term-premium adjustment',
     'reason': 'Yields contain expectations and risk premiums; they are not a direct instruction to the Bank.'},
    {'id': 'currency', 'name': 'Canadian dollar and import prices',
     'terms': ['canadian dollar', 'exchange rate', 'exchange rates', 'import prices', 'currency depreciation'],
     'features': [], 'missing': 'effective exchange rate and import-price changes',
     'reason': 'Currency changes affect imported inflation and trade; Canada does not target a fixed exchange rate.'},
    {'id': 'global', 'name': 'Global demand and foreign policy',
     'terms': ['united states', 'us economy', 'u.s. economy', 'federal reserve', 'global economy', 'china', 'euro area'],
     'features': [], 'missing': 'US and global growth, foreign policy rates and external demand',
     'reason': 'Foreign demand and financial conditions transmit to Canada without mechanically determining its rate.'},
    {'id': 'trade', 'name': 'Tariffs and trade disruption',
     'terms': ['tariff', 'tariffs', 'trade policy', 'trade uncertainty', 'exports', 'trade restrictions', 'trade talks'],
     'features': [], 'missing': 'dated policy announcements, implementation dates and sector exposure',
     'reason': 'Trade restrictions can raise costs and reduce demand at the same time.'},
    {'id': 'supply', 'name': 'Energy and supply shocks',
     'terms': ['oil prices', 'energy prices', 'gasoline', 'supply chain', 'supply shocks', 'commodity', 'food prices'],
     'features': [], 'missing': 'energy and food prices, shipping and supply-chain indicators',
     'reason': 'Supply shocks can move inflation and activity in opposite directions; persistence matters.'},
    {'id': 'fiscal', 'name': 'Government spending, taxes and regulation',
     'terms': ['fiscal', 'government spending', 'government investment', 'tax', 'taxes', 'taxation', 'regulation'],
     'features': [], 'missing': 'budget measures, implementation dates, transfers and tax effects',
     'reason': 'Fiscal measures alter demand, capacity and sometimes measured prices directly.'},
    {'id': 'demographics', 'name': 'Population and structural change',
     'terms': ['population', 'immigration', 'demographic', 'aging', 'ageing', 'technology', 'artificial intelligence'],
     'features': [], 'missing': 'population flows, age structure, capital deepening and productivity shifts',
     'reason': 'Structural changes affect both labour supply and demand, housing and longer-run capacity.'},
    {'id': 'policy', 'name': 'Policy stance and transmission',
     'terms': ['policy rate', 'monetary policy', 'restrictive', 'neutral rate', 'quantitative', 'balance sheet', 'transmission'],
     'features': ['real_rate', 'policy_momentum'],
     'missing': 'expected inflation, neutral-rate estimates, balance-sheet policy and mortgage-renewal exposure',
     'reason': 'Previous decisions work with lags; current inflation alone does not measure the remaining effect.'},
    {'id': 'risks', 'name': 'Geopolitics, uncertainty and risk management',
     'terms': ['geopolitical', 'uncertainty', 'conflict', 'war', 'pandemic', 'climate', 'wildfire', 'upside risks', 'downside risks'],
     'features': [], 'missing': 'verified event timing, exposure, scenarios and forecast-error distributions',
     'reason': 'The Bank weighs alternative outcomes and risks, not just a single central forecast.'},
]


base = 'https://www.bankofcanada.ca'


sources = {
    'speech': base + '/press/speeches/',
    'statement': base + '/press/press-releases/',
    'deliberations': base + '/search/?content_type%5B%5D=summary-of-deliberations',
    'outlook': base + '/publications/mpr/',
    'business_survey': base + '/publications/bos/',
    'consumer_survey': base + '/publications/canadian-survey-of-consumer-expectations/',
}


root = Path(__file__).resolve().parents[1]


def canonical(url):
    parts = urlsplit(url)
    if parts.scheme != 'https' or parts.hostname != 'www.bankofcanada.ca':
        raise ValueError('Expected an official Bank of Canada HTTPS page.')
    return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))


def discover(html, base, kind):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    for link in soup.select('main article h3 a, main article h2 a, main .media-heading a'):
        url = urljoin(base, link.get('href', ''))
        if not re.match(r'https://www\.bankofcanada\.ca/(?:20\d{2}/\d{2}/|publications/mpr/mpr-20\d{2}-)', url):
            continue
        title = link.get_text(' ', strip=True)
        if kind == 'speech':
            article = link.find_parent('article')
            subjects = article.select_one('.subject') if article else None
            if subjects is not None and not re.search(r'monetary policy|inflation|economy|financial stability', subjects.get_text(' ', strip=True), re.I):
                continue
        if kind == 'statement' and not re.search(r'policy rate|interest rate|policy interest', title, re.I):
            continue
        url = canonical(url)
        article = link.find_parent('article')
        stamp = article.select_one('.media-date') if article else None
        hint = ' '.join(stamp.get_text(' ', strip=True).split()) if stamp else None
        if not any(item['url'] == url for item in found):
            found.append({'url': url, 'publication_hint': hint})
    return found


def extract(html, url, kind, retrieved_at, publication_hint=None):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.select_one('main .post-content')
    title = soup.select_one('main h1')
    stamp = soup.select_one('main .post-date')
    if body is None or len(body.get_text(' ', strip=True).split()) < 80:
        body = soup.select_one('main')
    if body is None or title is None or (stamp is None and not publication_hint):
        raise ValueError('Article body, title or publication date was not found.')
    date_text = stamp.get_text(' ', strip=True) if stamp else publication_hint
    date_text = ' '.join(date_text.split())
    match = re.search(r'[A-Z][a-z]+ \d{1,2}, \d{4}', date_text)
    if not match:
        raise ValueError('Publication date format was not recognized.')
    published = datetime.strptime(match.group(), '%B %d, %Y').date()
    author = soup.select_one('main .post-authors')
    author_text = author.get_text(' ', strip=True) if author else None
    for node in body.select('script, style, nav, aside, .bocss-share, .post-meta, form, article.media, .related-posts'):
        node.decompose()
    paragraphs = [p.get_text(' ', strip=True) for p in body.select('p')]
    text = '\n'.join(p for p in paragraphs if p)
    if len(text.split()) < 80:
        raise ValueError('Insufficient article text; page may be a video or PDF landing page.')
    normalized = canonical(url)
    identifier = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {'id': identifier, 'url': normalized, 'title': title.get_text(' ', strip=True),
            'kind': kind, 'author': author_text,
            'published_date': published.isoformat(), 'date_precision': 'day',
            'date_source': 'article' if stamp is not None else 'listing',
            'available_after': (published + timedelta(days=1)).isoformat(),
            'retrieved_at': retrieved_at, 'first_seen_at': retrieved_at,
            'text_sha256': digest, 'text': text, 'word_count': len(text.split())}


def topic_matches(text):
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    matches = []
    for factor in factors:
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in factor['terms']) + r')\b', re.I)
        evidence = next((s for s in sentences if pattern.search(s)), None)
        if evidence:
            words = evidence.split()
            position = pattern.search(evidence).start()
            start = max(0, len(evidence[:position].split()) - 8)
            snippet = ('… ' if start else '') + ' '.join(words[start:start + 22])
            if len(words) > start + 22:
                snippet += ' …'
            matches.append({'factor': factor['id'], 'terms': sorted({m.group().lower() for m in pattern.finditer(text)}),
                            'snippet': snippet if len(matches) < 2 else None})
    return matches


def eligible(document, cutoff):
    return document['available_after'] <= cutoff.isoformat()


def collect(folder, cutoff=None, lookback_days=180, pages=2, per_source=4):
    cutoff = cutoff or date.today()
    folder = Path(folder)
    archive = folder / 'html'
    archive.mkdir(parents=True, exist_ok=True)
    path = folder / 'documents.json'
    previous = json.loads(path.read_text()) if path.exists() else []
    versions = {(d['url'], d['text_sha256']): d for d in previous}
    session = requests.Session()
    session.headers['User-Agent'] = 'RateWatch/1.0 (economic research; bounded public-page collection)'
    errors, discovery_log = [], []
    earliest = cutoff - timedelta(days=lookback_days)

    def get(url):
        time.sleep(0.35)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        canonical(response.url)
        return response.text

    for kind, index in sources.items():
        print('Collecting:', kind, flush=True)
        candidates = []
        for page in range(1, pages + 1):
            parts = urlsplit(index)
            query = dict(parse_qsl(parts.query))
            query['mt_page'] = str(page)
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ''))
            try:
                candidates.extend(discover(get(url), url, kind))
            except (requests.RequestException, ValueError) as error:
                errors.append({'url': url, 'stage': 'discovery', 'error': str(error)})
        candidates = list({item['url']: item for item in candidates}.values())
        discovery_log.append({'kind': kind, 'index': index, 'candidates': len(candidates)})
        accepted = 0
        for candidate in candidates:
            url = candidate['url']
            if accepted >= per_source:
                break
            try:
                html = get(url)
                stamp = datetime.now(timezone.utc).isoformat()
                document = extract(html, url, kind, stamp, candidate['publication_hint'])
                published = date.fromisoformat(document['published_date'])
                if not earliest <= published <= cutoff:
                    continue
                if 'schedule' in document['title'].lower():
                    document['kind'] = 'publication_calendar'
                document['topics'] = topic_matches(document['text'])
                if not document['topics']:
                    continue
                key = (document['url'], document['text_sha256'])
                if key in versions:
                    document['first_seen_at'] = versions[key]['first_seen_at']
                document['archive'] = f"html/{document['id']}-{document['text_sha256'][:12]}.html"
                (folder / document['archive']).write_text(html)
                versions[key] = document
                accepted += 1
            except (requests.RequestException, ValueError) as error:
                errors.append({'url': url, 'stage': 'article', 'error': str(error)})
        discovery_log[-1]['accepted'] = accepted
        print(f'  accepted {accepted} documents', flush=True)
    documents = list(versions.values())
    documents.sort(key=lambda d: (d['published_date'], d['url']))
    path.write_text(json.dumps(documents, indent=2, ensure_ascii=False))
    status = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'cutoff': cutoff.isoformat(),
              'lookback_days': lookback_days, 'pages_per_source': pages, 'limit_per_source': per_source,
              'sources': discovery_log, 'errors': errors,
              'stored_versions': len(documents), 'new_run_accepted': sum(item['accepted'] for item in discovery_log)}
    (folder / 'collection.json').write_text(json.dumps(status, indent=2))
    return documents, status


def write_digest(documents, status, output, cutoff=None, origin=None):
    """policy-context text does not affect numerical forecast probabilities"""
    cutoff = cutoff or date.today()
    latest = {}
    for document in sorted(documents, key=lambda row: row['retrieved_at']):
        if eligible(document, cutoff):
            latest[document['url']] = document
    earliest = cutoff - timedelta(days=status.get('lookback_days', 180))
    recent = sorted((d for d in latest.values() if date.fromisoformat(d['published_date']) >= earliest),
                    key=lambda row: row['published_date'], reverse=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lines = ['# Policy context', '', f'Context cutoff: {cutoff}.', '',
             'Policy-context text does not affect the numerical forecast probabilities.',
             'Topic matches describe coverage, not sentiment. Same-day documents are excluded.',
             'Archived text may have been edited before collection. HTML only; PDF attachments are not parsed.', '']
    if origin:
        lines += [f'Forecast cutoff: {origin}. Later documents are context only.', '']
    for document in recent:
        topics = ', '.join(match['factor'] for match in document['topics'])
        title = document['title'].replace(chr(8212), '-')
        lines += [f"- {document['published_date']}: [{title}]({document['url']}). Topics: {topics}."]
    lines += ['', f"{len(recent)} documents; {len(status['errors'])} collection errors. Details: data/communications/collection.json."]
    (output / 'context.md').write_text('\n'.join(lines) + '\n')
    return len(recent)


def main():
    parser = argparse.ArgumentParser(description='Optional Bank of Canada policy context.')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--as-of', type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    folder = root / 'data' / 'communications'
    if args.offline:
        documents = json.loads((folder / 'documents.json').read_text())
        status = json.loads((folder / 'collection.json').read_text())
    else:
        documents, status = collect(folder, args.as_of)
    forecast_path = root / 'results' / 'forecast.json'
    forecast = json.loads(forecast_path.read_text()) if forecast_path.exists() else {}
    count = write_digest(documents, status, root / 'results', args.as_of, forecast.get('information_cutoff'))
    print(f'Wrote results/context.md with {count} documents.')


if __name__ == '__main__':
    main()
