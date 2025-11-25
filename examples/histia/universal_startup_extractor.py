"""
Universal Startup Extractor - Generalist Agent for Exhaustive Startup Extraction

This agent is designed to extract ALL startups from ANY website, regardless of structure.
It uses intelligent LLM-guided strategies to find and extract startups exhaustively.

Usage:
    python examples/histia/universal_startup_extractor.py --url "https://example.com/startups"
    python examples/histia/universal_startup_extractor.py --url "https://example.com" --output startups.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer

# Load environment variables immediately
load_dotenv()

# Configure timeouts for heavy pages
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class UniversalStartupExtractorInput(BaseModel):
    """User-provided parameters for the universal startup extraction task."""

    url: str = Field(
        ...,
        description='URL of the website to extract startups from',
    )
    max_startup: int = Field(
        100000,
        ge=1,
        le=1000000,
        description='Nombre maximum de startups à extraire avant arrêt immédiat (ex: 50)',
    )
    output_path: Path = Field(
        default=Path('extracted_startups.json'),
        description='Destination for the JSON list of startups',
    )

    @property
    def target_url(self) -> AnyHttpUrl:
        """Build the target URL."""
        return AnyHttpUrl(self.url)


class Startup(BaseModel):
    """Structured information for each startup entry."""

    name: str = Field(..., description='Startup/company name exactly as written on the page')
    startup_url: str | None = Field(
        None,
        description='Complete URL to the startup detail page if available',
    )
    description: str | None = Field(
        None,
        description='Startup description/tagline/business model if available',
    )
    website: str | None = Field(
        None,
        description='Startup website URL if available',
    )
    sector: str | None = Field(
        None,
        description='Startup sector/industry/category if available',
    )
    location: str | None = Field(
        None,
        description='Startup location (city, country, region) if available',
    )
    founded_year: int | None = Field(
        None,
        description='Year the startup was founded if available',
    )
    employees: str | None = Field(
        None,
        description='Number of employees or employee range if available',
    )
    funding_stage: str | None = Field(
        None,
        description='Funding stage (seed, series A, etc.) if available',
    )
    tags: list[str] = Field(
        default_factory=list,
        description='Startup tags/categories/technologies visible on the card',
    )
    logo_url: str | None = Field(
        None,
        description='URL to startup logo if available',
    )
    additional_info: dict[str, Any] = Field(
        default_factory=dict,
        description='Any additional information found (social links, metrics, etc.)',
    )


def _clean_str(value: str | None) -> str | None:
    """Return a stripped string or None if empty."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_tags(value: str | None) -> list[str]:
    """Parse comma/semicolon separated tags."""
    cleaned = _clean_str(value)
    if not cleaned:
        return []
    parts = re.split(r'[;,]', cleaned)
    return [part.strip() for part in parts if part.strip()]


def _resolve_url(raw_url: str | None, base_url: str) -> str | None:
    """Convert relative URLs to absolute and normalize blanks."""
    cleaned = _clean_str(raw_url)
    if not cleaned:
        return None
    return urljoin(base_url, cleaned)


def _looks_like_table_line(line: str) -> bool:
    """Return True when the line appears to be part of a markdown table."""
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2


def _iter_markdown_tables(text: str) -> Iterable[list[str]]:
    """Yield markdown table blocks (including header & separator)."""
    current: list[str] = []
    for line in text.splitlines():
        if _looks_like_table_line(line):
            current.append(line.strip())
        else:
            if current:
                yield current
                current = []
    if current:
        yield current


HEADER_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    'name': ('name', 'startup name', 'company'),
    'description': ('description', 'tagline', 'summary', 'pitch'),
    'startup_url': ('url', 'link', 'detail url', 'startup url'),
    'website': ('website', 'site', 'homepage'),
    'sector': ('sector', 'category', 'industry', 'domain'),
    'location': ('location', 'city', 'country', 'region'),
    'tags': ('tags', 'labels', 'technologies'),
}


def _normalize_header(header: str) -> str | None:
    """Map a markdown header label to a canonical column name."""
    cleaned = re.sub(r'[^a-z0-9 ]+', ' ', header.lower()).strip()
    for canonical, aliases in HEADER_ALIAS_MAP.items():
        if any(alias in cleaned for alias in aliases):
            if canonical == 'startup_url' and 'website' in cleaned:
                continue
            if canonical == 'website' and 'url' in cleaned and 'website' not in cleaned:
                continue
            return canonical
    return None


def _parse_table(table_lines: Sequence[str]) -> tuple[list[str], list[list[str]]] | None:
    """Convert raw markdown table lines into headers + row values."""
    if len(table_lines) < 2:
        return None
    header = [cell.strip() for cell in table_lines[0].strip('|').split('|')]
    separator_cells = [cell.strip() for cell in table_lines[1].strip('|').split('|')]
    if not separator_cells or any(not cell or not set(cell) <= {'-', ':'} for cell in separator_cells):
        return None
    rows: list[list[str]] = []
    for line in table_lines[2:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('>'):
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        if len(cells) != len(header):
            continue
        rows.append(cells)
    if not rows:
        return None
    return header, rows


def _startups_from_tables(table_lines: Sequence[str], base_url: str) -> list[Startup]:
    """Convert parsed markdown tables into Startup objects."""
    parsed = _parse_table(table_lines)
    if not parsed:
        return []

    header, rows = parsed
    mapped_indices: dict[str, int] = {}
    for idx, label in enumerate(header):
        normalized = _normalize_header(label)
        if normalized and normalized not in mapped_indices:
            mapped_indices[normalized] = idx

    if 'name' not in mapped_indices:
        return []

    startups: list[Startup] = []
    for row in rows:
        name = _clean_str(row[mapped_indices['name']])
        if not name:
            continue
        description = _clean_str(row[mapped_indices['description']]) if 'description' in mapped_indices else None
        startup_url = (
            _resolve_url(row[mapped_indices['startup_url']], base_url) if 'startup_url' in mapped_indices else None
        )
        website = _resolve_url(row[mapped_indices['website']], base_url) if 'website' in mapped_indices else None
        sector = _clean_str(row[mapped_indices['sector']]) if 'sector' in mapped_indices else None
        location = _clean_str(row[mapped_indices['location']]) if 'location' in mapped_indices else None
        tags = _parse_tags(row[mapped_indices['tags']]) if 'tags' in mapped_indices else []

        startup = Startup(
            name=name,
            startup_url=startup_url or website,
            description=description,
            website=website if website and website != startup_url else None,
            sector=sector,
            location=location,
            founded_year=None,
            employees=None,
            funding_stage=None,
            logo_url=None,
            tags=tags,
            additional_info={},
        )
        startups.append(startup)
    return startups


def _deduplicate_startups(startups: Sequence[Startup]) -> list[Startup]:
    """Remove duplicates while preserving order."""
    unique: dict[tuple[str, str], Startup] = {}
    for startup in startups:
        key = (startup.name.lower(), (startup.startup_url or '').lower())
        if key not in unique:
            unique[key] = startup
    return list(unique.values())


def _extract_startups_from_contents(
    contents: Sequence[str],
    base_url: str,
) -> list[Startup]:
    """Parse markdown tables emitted by the agent into structured startups."""
    collected: list[Startup] = []
    for content in contents:
        for table in _iter_markdown_tables(content):
            collected.extend(_startups_from_tables(table, base_url))
    return _deduplicate_startups(collected)


def _unique_pages(urls: Sequence[str | None]) -> list[str]:
    """Return URLs with duplicates removed, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


class StartupExtractionReport(BaseModel):
    """Complete response returned by the agent."""

    source_url: AnyHttpUrl = Field(..., description='Source URL that was analysed')
    startups: list[Startup] = Field(
        ...,
        min_length=0,
        description='Startup entries extracted from all pages',
    )
    pages_visited: list[str] = Field(
        default_factory=list,
        description='List of all URLs visited during extraction',
    )
    extraction_notes: str | None = Field(
        None,
        description='Notes about the extraction process (structure found, challenges, etc.)',
    )

    @field_serializer('source_url')
    def serialize_source_url(self, value: AnyHttpUrl, _info) -> str:
        """Convert AnyHttpUrl to string for JSON serialization."""
        return str(value)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to ensure AnyHttpUrl is converted to string."""
        result = super().model_dump(**kwargs)
        if 'source_url' in result:
            source_url = result['source_url']
            if hasattr(source_url, '__str__') and not isinstance(source_url, str):
                result['source_url'] = str(source_url)
        return result


def _fallback_report(source_url: str, reason: str) -> StartupExtractionReport:
    """Return a minimal report when the agent cannot finish properly."""

    reason = reason.strip() or "Could not extract startups from the page."
    return StartupExtractionReport(
        source_url=AnyHttpUrl(source_url),
        startups=[],
        pages_visited=[],
        extraction_notes=reason,
    )


def build_comprehensive_task(task_input: UniversalStartupExtractorInput) -> str:
    """Create extremely detailed natural-language instructions for exhaustive startup extraction."""

    target_url = str(task_input.target_url)
    max_startup = task_input.max_startup

    return dedent(
        f"""
        You are an ULTRA-INTELLIGENT agent specialized in EXHAUSTIVE startup extraction from ANY website.
        Your mission: Find and extract EVERY SINGLE startup/company from the target website, regardless of its structure.

        ═══════════════════════════════════════════════════════════════════════════════
        🎯 CORE MISSION: EXHAUSTIVE EXTRACTION (BUT STOP IMMEDIATELY IF YOU HAVE FOUND {max_startup} STARTUPS)
        ═══════════════════════════════════════════════════════════════════════════════

        CRITICAL RULES:
        1. Extract ALL startups visible on listing/gridded pages - DO NOT MISS a card that is currently discoverable via pagination or lazy-loading.
        2. Stay on the high-level listing views whenever possible. DO NOT open every startup/product detail page just to gather richer information. We are only building a list.
        3. Required fields per startup: name (exact label on the page), short description/tagline, and the link exposed on the card (category/product link or external website). If website links are not shown on the card, keep the card URL itself. Extra info (tags, scores, etc.) is optional.
        4. Navigate to additional listing pages ONLY when the current list is exhausted or pagination is clearly required. Avoid deep navigation into profile/detail pages unless the listing card is missing both the description and the link.
        5. Use multiple extraction strategies (scroll, pagination, section toggles) but keep the footprint minimal—do not re-open pages that already provided enough data.
        6. Stop exploration immediately once you have extracted {max_startup} startups; do not exceed this number and move to finalization rapidly.
        7. Before finishing, re-check the current listing page to ensure no visible cards were skipped, but do not traverse past pages that you have already validated.
        7. Before finishing, re-check the current listing page to ensure no visible cards were skipped, but do not traverse past pages that you have already validated.

        ...(rest of task prompt unchanged; you should insert the instruction to stop after max_startup startups where it is relevant in the global rules and summary at the top/Final Report note, as above)...

        """).strip()


async def run_universal_startup_extraction(
    task_input: UniversalStartupExtractorInput,
) -> StartupExtractionReport | None:
    """Execute the agent and return the structured list of startups."""

    print("🔧 Configuring LLM...")
    if os.getenv('BROWSER_USE_API_KEY'):
        llm = ChatBrowserUse()
        page_extraction_llm = ChatBrowserUse()
        print("✅ Using ChatBrowserUse")
    else:
        model_name = os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-preview-09-2025')
        if 'gemini' not in model_name.lower():
            model_name = 'gemini-2.5-flash-lite-preview-09-2025'
            print(f"⚠️  Non-Gemini model detected, using {model_name} instead")
        is_gemini = 'gemini' in model_name.lower()
        llm = ChatOpenAI(
            model=model_name,
            timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
            max_retries=3,
            max_completion_tokens=15000,
            add_schema_to_system_prompt=is_gemini,
            dont_force_structured_output=is_gemini,
        )
        extraction_model = os.getenv('PAGE_EXTRACTION_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
        page_extraction_llm = ChatOpenAI(
            model=extraction_model,
            timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=20.0),
            max_retries=2,
            max_completion_tokens=15000,
            add_schema_to_system_prompt=True,
            dont_force_structured_output=True,
        )
        print(f"✅ Using ChatOpenAI with model: {model_name}")

    print("🌐 Creating browser...")
    browser = Browser(headless=False, keep_alive=True)
    await browser.start()

    try:
        source_url_model = task_input.target_url
        target_url = str(source_url_model)
        print(f"📍 Navigating to: {target_url}")
        navigate_event = NavigateToUrlEvent(url=target_url, new_tab=True)
        await browser.event_bus.dispatch(navigate_event)
        await navigate_event

        # Get the current page after navigation
        page = await browser.get_current_page()
        if not page:
            page = await browser.new_page(target_url)

        await asyncio.sleep(5)  # Wait for initial page load

        # Use LLM-based agent for intelligent extraction
        print("🤖 Starting intelligent LLM-based extraction agent...")
        agent = Agent(
            task=build_comprehensive_task(task_input),
            llm=llm,
            page_extraction_llm=page_extraction_llm,
            browser=browser,
            output_model_schema=StartupExtractionReport,
            use_vision='auto',
            vision_detail_level='auto',
            step_timeout=300,
            llm_timeout=180,
            max_failures=5,
            max_history_items=30,  # Keep more history for multi-page navigation
            max_steps=1000,  # Allow many steps for exhaustive extraction
            directly_open_url=False,
        )
        print("✅ Agent created")

        print("▶️  Starting agent execution...")
        history = await agent.run()
        print("✅ Execution completed")
        print_llm_usage_summary(history)

        parsed_startups = _extract_startups_from_contents(history.extracted_content(), base_url=target_url)
        pages_visited = _unique_pages(history.urls())

        def _finalize_report(report: StartupExtractionReport, rebuilt_note: bool = False) -> StartupExtractionReport:
            if parsed_startups and len(parsed_startups) > len(report.startups):
                report.startups = parsed_startups
                note = (
                    "Startup list rebuilt from extraction tables because the agent response was truncated."
                )
                if report.extraction_notes:
                    report.extraction_notes = f'{report.extraction_notes}\n{note}'
                else:
                    report.extraction_notes = note
            if rebuilt_note and parsed_startups and not report.extraction_notes:
                report.extraction_notes = (
                    "Startup list built from extraction tables because the agent response did not include structured data."
                )
            if not report.pages_visited:
                report.pages_visited = pages_visited
            if hasattr(task_input, 'max_startup'):
                report.startups = report.startups[:task_input.max_startup]
            return report

        # Try to get structured output
        if history.structured_output:
            report = history.structured_output  # type: ignore[arg-type]
            return _finalize_report(report)

        # Try to extract from final result
        final_result = history.final_result()
        if final_result:
            try:
                report = StartupExtractionReport.model_validate_json(final_result)
                return _finalize_report(report)
            except ValidationError:
                match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
                if match:
                    try:
                        report = StartupExtractionReport.model_validate_json(match.group(1))
                        return _finalize_report(report)
                    except ValidationError:
                        pass

        if parsed_startups:
            return _finalize_report(
                StartupExtractionReport(
                    source_url=source_url_model,
                    startups=parsed_startups,
                    pages_visited=pages_visited,
                    extraction_notes=None,
                ),
                rebuilt_note=True,
            )

        return _fallback_report(str(source_url_model), "Extraction failed: no startups found.")

    finally:
        # Close browser
        print("🧹 Closing browser...")
        try:
            await browser.kill()
            print("✅ Browser closed")
        except Exception as e:
            print(f"⚠️  Error closing browser: {e}")


def parse_arguments() -> UniversalStartupExtractorInput:
    """Validate CLI arguments via Pydantic before launching the agent."""

    parser = argparse.ArgumentParser(
        description='Extract ALL startups from ANY website exhaustively'
    )
    parser.add_argument(
        '--url',
        required=True,
        help='URL of the website to extract startups from',
    )
    parser.add_argument(
        '--max-startup',
        type=int,
        default=100000,
        help='Nombre maximum de startups à extraire avant arrêt immédiat',
    )
    parser.add_argument(
        '--output',
        default='extracted_startups.json',
        help='Output JSON file path (default: ./extracted_startups.json)',
    )
    args = parser.parse_args()

    return UniversalStartupExtractorInput(
        url=args.url,
        max_startup=args.max_startup,
        output_path=Path(args.output),
    )


async def main() -> None:
    """CLI entry point."""

    try:
        task_input = parse_arguments()
        target_url = str(task_input.target_url)
        print(f"🚀 Starting Universal Startup Extractor")
        print(f"📍 Target URL: {target_url}")
        print(f"📊 Max startups: {task_input.max_startup}")
        print(f"💾 Output file: {task_input.output_path}")

        result = await run_universal_startup_extraction(task_input)

        if result is None:
            print("❌ Agent did not return structured data.")
            return

        # Truncate output if needed for ultimate insurance
        if hasattr(task_input, 'max_startup'):
            result.startups = result.startups[:task_input.max_startup]

        output_json = result.model_dump_json(indent=2, ensure_ascii=False)
        output_path = task_input.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding='utf-8')

        print(f'\n✅ {len(result.startups)} startups extracted and saved to: {output_path.resolve()}')
        print(f'📄 Pages visited: {len(result.pages_visited)}')
        if result.extraction_notes:
            print(f'📝 Notes: {result.extraction_notes}')

    except KeyboardInterrupt:
        print("\n⚠️  User interruption detected.")
        raise
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    asyncio.run(main())
