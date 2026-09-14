import unittest
from datetime import date

from ratewatch.context import canonical, discover, eligible, extract, topic_matches

url = 'https://www.bankofcanada.ca/2026/07/example/'
stamp = '2026-09-12T12:00:00+00:00'
paragraph = 'Inflation is not accelerating and unemployment remains elevated. ' * 12


def article(body=None):
    return '<nav><p>Tariffs in unrelated navigation</p></nav><main><div class="post-body">' + \
        '<h1>Deliberations for the June 10 decision</h1><div class="post-meta">' + \
        '<div class="post-authors">Governing Council</div><div class="post-date">July 2, 2026</div></div>' + \
        '<div class="post-content"><p>' + (body or paragraph) + '</p></div></div></main>'


class CommunicationTests(unittest.TestCase):
    def test_extracts_article_without_navigation(self):
        result = extract(article(), url, 'deliberations', stamp)
        self.assertNotIn('unrelated navigation', result['text'])
        self.assertEqual(result['author'], 'Governing Council')
        self.assertEqual(result['published_date'], '2026-07-02')
        self.assertEqual(result['available_after'], '2026-07-03')

    def test_uses_publication_not_event_date(self):
        result = extract(article(), url, 'deliberations', stamp)
        self.assertFalse(eligible(result, date(2026, 6, 10)))
        self.assertFalse(eligible(result, date(2026, 7, 2)))
        self.assertTrue(eligible(result, date(2026, 7, 3)))

    def test_modular_page(self):
        html = '<main><h1>Speech</h1><div class="post-date">July 2, 2026</div>' + \
            '<div class="post-content"></div><div class="cfct-mod-content"><p>' + paragraph + '</p></div></main>'
        result = extract(html, url, 'speech', stamp)
        self.assertIn('Inflation is not accelerating', result['text'])

    def test_survey_listing_date(self):
        html = '<main><h1>Survey</h1><p>' + paragraph + '</p></main>'
        result = extract(html, url, 'business_survey', stamp, 'July 6, 2026')
        self.assertEqual(result['published_date'], '2026-07-06')
        self.assertEqual(result['date_source'], 'listing')

    def test_missing_date_rejected(self):
        html = '<main><h1>Survey</h1><p>' + paragraph + '</p></main>'
        with self.assertRaises(ValueError):
            extract(html, url, 'business_survey', stamp)

    def test_topic_match_is_not_sentiment(self):
        matches = topic_matches('Inflation is not accelerating. Unemployment remains elevated.')
        self.assertEqual(matches[0]['factor'], 'inflation')
        self.assertIn('not accelerating', matches[0]['snippet'])
        self.assertNotIn('direction', matches[0])

    def test_discovery_scope_and_deduplication(self):
        link = '<article><span class="media-date">July 6, 2026</span><h3><a href="' + url + '">Bank holds policy rate</a></h3></article>'
        html = '<nav>' + link + '</nav><main>' + link * 2 + '</main>'
        self.assertEqual(len(discover(html, url, 'statement')), 1)
        self.assertEqual(discover(html, url, 'statement')[0]['publication_hint'], 'July 6, 2026')

    def test_external_host_rejected(self):
        with self.assertRaises(ValueError):
            canonical('https://example.com/2026/07/page/')

    def test_short_page_rejected(self):
        with self.assertRaises(ValueError):
            extract(article('Watch the video.'), url, 'speech', stamp)

    def test_nonpolicy_speech_filtered(self):
        html = '<main><article><span class="subject">Currency, Bank notes</span><h3><a href="' + url + '">New bank note</a></h3></article></main>'
        self.assertEqual(discover(html, url, 'speech'), [])


if __name__ == '__main__':
    unittest.main()
