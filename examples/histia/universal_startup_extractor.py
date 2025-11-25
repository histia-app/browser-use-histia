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
from typing import Any

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
        1. Extract ALL startups - DO NOT MISS A SINGLE ONE
        2. Work on ANY website structure - adapt your strategy intelligently
        3. Explore ALL pages, sections, tabs, and navigation paths
        4. Use multiple extraction strategies to ensure completeness
        5. Stop your exploration and extraction as soon as you have extracted {max_startup} startups (do not extract more than this number, stop exactly at this count and end the extraction rapidly)
        6. Verify you haven't missed anything from pages explored so far before finishing

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
            max_completion_tokens=90960,
            add_schema_to_system_prompt=is_gemini,
            dont_force_structured_output=is_gemini,
        )
        extraction_model = os.getenv('PAGE_EXTRACTION_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
        page_extraction_llm = ChatOpenAI(
            model=extraction_model,
            timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=20.0),
            max_retries=2,
            max_completion_tokens=90960,
            add_schema_to_system_prompt=True,
            dont_force_structured_output=True,
        )
        print(f"✅ Using ChatOpenAI with model: {model_name}")

    print("🌐 Creating browser...")
    browser = Browser(headless=False, keep_alive=True)
    await browser.start()

    try:
        target_url = str(task_input.target_url)
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

        # Try to get structured output
        if history.structured_output:
            report = history.structured_output  # type: ignore[arg-type]
            # Truncate startups to max_startup if more were returned
            if hasattr(task_input, 'max_startup'):
                report.startups = report.startups[:task_input.max_startup]
            return report

        # Try to extract from final result
        final_result = history.final_result()
        if final_result:
            try:
                report = StartupExtractionReport.model_validate_json(final_result)
                if hasattr(task_input, 'max_startup'):
                    report.startups = report.startups[:task_input.max_startup]
                return report
            except ValidationError:
                match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
                if match:
                    try:
                        report = StartupExtractionReport.model_validate_json(match.group(1))
                        if hasattr(task_input, 'max_startup'):
                            report.startups = report.startups[:task_input.max_startup]
                        return report
                    except ValidationError:
                        pass

        return _fallback_report(str(task_input.target_url), "Extraction failed: no startups found.")

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
