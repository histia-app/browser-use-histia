from __future__ import annotations

from datetime import date

from examples.histia.betalist_extractor import (
	_build_report_from_payloads,
	_day_label_to_date,
	_duplicate_id_to_date,
	_parse_startup_payload,
)

SOURCE_URL = 'https://betalist.com/'

SAMPLE_CARD_HTML = """
<div class="block" id="startup-138993">
  <a class="block min-w-[128px] rounded-sm overflow-hidden relative aspect-4/3" style="background-color: #3b3b55" href="/startups/simcardo">
    <img src="https://img.test/simcardo.jpg" srcset="https://img.test/simcardo@2x.jpg 2x" alt="Simcardo logo" />
  </a>
  <div class="block">
    <div class="mt-3 text-base">
      <div class="flex items-start gap-2">
        <a class="block whitespace-nowrap text-ellipsis overflow-hidden font-medium text-gray-900" href="/startups/simcardo">Simcardo</a>
      </div>
      <a class="block text-gray-500" href="/startups/simcardo">Instant mobile data worldwide</a>
    </div>
    <div class="block">
      <a class="pill" href="/topics/travel">Travel</a>
      <a class="" href="/@czteam1"><div class="sr-only">czteam1</div></a>
      <a class="cta" href="https://simcardo.com" target="_blank">Visit website</a>
    </div>
  </div>
</div>
"""


def test_duplicate_id_to_date_returns_epoch_and_date() -> None:
	epoch, parsed = _duplicate_id_to_date('day_1763596800')
	assert epoch == 1763596800
	assert parsed == date(2025, 11, 20)


def test_day_label_to_date_parses_month_text() -> None:
	assert _day_label_to_date('Today November 20th', reference_year=2025) == date(2025, 11, 20)


def test_parse_startup_payload_extracts_expected_fields() -> None:
	payload = {
		'id': 'startup-138993',
		'html': SAMPLE_CARD_HTML,
		'dayDuplicateId': 'day_1763596800',
		'dayLabel': 'Today November 20th',
		'topics': [{'label': 'Travel', 'href': '/topics/travel'}],
		'founders': [{'label': 'czteam1', 'href': '/@czteam1'}],
	}
	startup = _parse_startup_payload(payload, SOURCE_URL)
	assert startup is not None
	assert startup.startup_id == 138993
	assert startup.slug == 'simcardo'
	assert startup.startup_url == 'https://betalist.com/startups/simcardo'
	assert startup.tagline == 'Instant mobile data worldwide'
	assert startup.image_url == 'https://img.test/simcardo.jpg'
	assert startup.image_srcset == 'https://img.test/simcardo@2x.jpg 2x'
	assert startup.image_alt == 'Simcardo logo'
	assert startup.background_color == '#3b3b55'
	assert startup.cta_url == 'https://simcardo.com'
	assert startup.cta_label == 'Visit website'
	assert startup.topics == ['Travel']
	assert startup.topic_urls == ['https://betalist.com/topics/travel']
	assert startup.founders and startup.founders[0].handle == 'czteam1'
	assert startup.published_at and startup.published_at.date() == date(2025, 11, 20)


def test_build_report_filters_old_entries_and_limits_max() -> None:
	recent_payload = {
		'id': 'startup-138993',
		'html': SAMPLE_CARD_HTML,
		'dayDuplicateId': 'day_1763596800',
		'dayLabel': 'Today November 20th',
	}
	old_payload = {
		'id': 'startup-1',
		'html': SAMPLE_CARD_HTML.replace('Simcardo', 'OldCo').replace('/simcardo', '/oldco'),
		'dayDuplicateId': 'day_1700000000',
		'dayLabel': 'January 1st',
	}
	report = _build_report_from_payloads(
		[recent_payload, old_payload],
		SOURCE_URL,
		cutoff_date=date(2025, 11, 19),
		max_startups=1,
	)
	assert report is not None
	assert len(report.startups) == 1
	assert report.startups[0].name == 'Simcardo'

