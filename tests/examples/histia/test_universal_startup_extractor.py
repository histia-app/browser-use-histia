from __future__ import annotations

from examples.histia.universal_startup_extractor import (
	Startup,
	_extract_startups_from_contents,
)

BASE_URL = 'https://21st.centralesupelec.com/nos-startups'

SAMPLE_TABLE = """
| Startup Name | Description | URL | Tags |
| --- | --- | --- | --- |
| .omics | Deeptech seeds | /nos-startups/omics | Deeptech, AI |
| Auditoo | Copilot IA | https://21st.centralesupelec.com/nos-startups/auditoo | Productivity |
"""


def test_extract_startups_from_contents_parses_markdown_tables() -> None:
	startups = _extract_startups_from_contents([SAMPLE_TABLE], base_url=BASE_URL)

	assert len(startups) == 2
	first = startups[0]
	assert isinstance(first, Startup)
	assert first.name == '.omics'
	assert first.startup_url == 'https://21st.centralesupelec.com/nos-startups/omics'
	assert first.description == 'Deeptech seeds'
	assert first.tags == ['Deeptech', 'AI']


def test_extract_startups_from_contents_deduplicates_entries() -> None:
	duplicate_content = """
| Startup Name | Description | URL |
| --- | --- | --- |
| .omics | Deeptech seeds | /nos-startups/omics |
"""
	startups = _extract_startups_from_contents([SAMPLE_TABLE, duplicate_content], base_url=BASE_URL)
	assert len(startups) == 2

