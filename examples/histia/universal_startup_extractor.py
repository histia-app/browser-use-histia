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


class UniversalStartupExtractorInput(BaseModel):
	"""User-provided parameters for the universal startup extraction task."""

	url: str = Field(
		...,
		description='URL of the website to extract startups from',
	)
	max_startups: int = Field(
		100000,
		ge=1,
		le=1000000,
		description='Maximum number of startups to capture (use a high number like 100000 to extract all)',
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
	max_startups = task_input.max_startups

	return dedent(
		f"""
		You are an ULTRA-INTELLIGENT agent specialized in EXHAUSTIVE startup extraction from ANY website.
		Your mission: Find and extract EVERY SINGLE startup/company from the target website, regardless of its structure.

		═══════════════════════════════════════════════════════════════════════════════
		🎯 CORE MISSION: EXHAUSTIVE EXTRACTION
		═══════════════════════════════════════════════════════════════════════════════

		CRITICAL RULES:
		1. Extract ALL startups - DO NOT MISS A SINGLE ONE
		2. Work on ANY website structure - adapt your strategy intelligently
		3. Explore ALL pages, sections, tabs, and navigation paths
		4. Use multiple extraction strategies to ensure completeness
		5. Verify you haven't missed anything before finishing

		═══════════════════════════════════════════════════════════════════════════════
		📋 PHASE 1: INITIAL EXPLORATION & SITE ANALYSIS
		═══════════════════════════════════════════════════════════════════════════════

		STEP 1.1: Navigate to Target URL
		- Use `navigate` action to go to: {target_url}
	- Wait 3-5 seconds for page load using `wait` action
		- Accept cookies if prompted (look for "Accept", "Agree", "OK" buttons)
		- Dismiss any popups or modals that might block content

		STEP 1.2: Understand the Website Structure
		This is CRITICAL - you must understand WHERE startups are located before extracting.

		Use `extract` action with this EXACT query:
		"Analyze this website structure to find where startups/companies are located. Look for:
		1. Navigation menus (header, sidebar, footer) - find links to 'Startups', 'Companies', 'Directory', 'Portfolio', 'Showcase', 'Marketplace', 'Listings'
		2. Main content areas - are startups displayed directly on this page?
		3. Search or filter options - can I search for startups?
		4. Categories or sections - are startups organized by category?
		5. Pagination indicators - are there page numbers, 'Next' buttons, or infinite scroll?
		6. Tabs or sections - are there different views (all, featured, new, etc.)?
		
		Provide a detailed analysis of:
		- Where startups are located on this site
		- How to navigate to startup listings
		- What navigation elements I should click
		- Whether there are multiple pages/sections to explore"

		STEP 1.3: Identify Navigation Paths to Startups
		Based on the analysis, identify ALL possible paths to startup listings:

		STRATEGY A: Direct Listing Page
		- If startups are visible on current page, proceed to extraction
		- But FIRST, check if there are filters, tabs, or views to explore

		STRATEGY B: Navigation Required
		- Look for navigation elements containing keywords:
		  * "Startup", "Start-up", "Startups", "Companies", "Company"
		  * "Directory", "Portfolio", "Showcase", "Marketplace"
		  * "Listings", "Browse", "Explore", "Discover"
		  * "Innovation", "Tech", "Ventures", "Accelerator"
		- Use `extract` to find these elements:
		  "Find all clickable elements (buttons, links, menu items) in the navigation that might lead to startup/company listings. List each element with its text and where it's located (header, menu, sidebar, footer)."

		STEP 1.4: Click Navigation to Startup Section
		- Once you identify the correct navigation element:
		  1. Use `find_text` to locate it, OR
		  2. Use `click` with the element index
		- Wait 3-5 seconds after clicking for page load
		- Verify you're now on a page with startup listings

		═══════════════════════════════════════════════════════════════════════════════
		📋 PHASE 2: COMPREHENSIVE PAGE DISCOVERY
		═══════════════════════════════════════════════════════════════════════════════

		STEP 2.1: Discover ALL Pages with Startups
		You must find EVERY page that contains startups. Use multiple strategies:

		STRATEGY 1: Pagination Discovery
		Use `extract` with:
		"Find ALL pagination elements on this page:
		- Page numbers (1, 2, 3, ...)
		- 'Next' or 'Previous' buttons
		- 'Load More' or 'Show More' buttons
		- Infinite scroll indicators
		- URL patterns that suggest pagination (e.g., ?page=, #page=, /page/)
		List ALL page numbers and navigation URLs you can find."

		STRATEGY 2: Category/Section Discovery
		Use `extract` with:
		"Find ALL category links, filter options, or section tabs that might show different sets of startups:
		- Category links (e.g., 'AI', 'Fintech', 'Healthcare')
		- Filter buttons (e.g., 'All', 'Featured', 'New', 'Popular')
		- Sector tabs or industry filters
		- Location filters
		List ALL such navigation elements and their URLs."

		STRATEGY 3: Search/Filter Discovery
		Use `extract` with:
		"Are there search boxes, filters, or advanced search options that can show different startup listings?
		If yes, describe how to use them to find all startups."

		STEP 2.2: Build Complete Page List
		- Create a mental list of ALL pages you need to visit:
		  * Main listing page
		  * All paginated pages (page 1, 2, 3, ...)
		  * All category pages
		  * All filtered views
		- Keep track of visited URLs to avoid duplicates

		═══════════════════════════════════════════════════════════════════════════════
		📋 PHASE 3: EXHAUSTIVE EXTRACTION FROM EACH PAGE
		═══════════════════════════════════════════════════════════════════════════════

		STEP 3.1: Prepare Page for Extraction
		For EACH page you visit:
		1. Navigate to the page using `navigate`
		2. Wait 3-5 seconds for load
		3. Scroll to top using `scroll` with `up: true, pages: 10`
		4. Handle infinite scroll or lazy loading:
		   - Scroll down progressively
		   - Wait 2-3 seconds between scrolls
		   - Continue until no new content loads
		   - Use `extract` to check: "How many startup cards/elements are currently visible on this page?"

		STEP 3.2: Identify Startup Elements
		Before extracting, understand HOW startups are displayed:

		Use `extract` with:
		"Analyze how startups/companies are displayed on this page:
		- Are they in cards, list items, table rows, or other containers?
		- What HTML structure do they use? (div, article, li, etc.)
		- What classes or attributes identify them?
		- Are they clickable links to detail pages?
		- How many startup elements are visible?
		Describe the structure in detail."

		STEP 3.3: Extract Startups Using LLM Intelligence
		This is the CORE extraction step. Use `extract` with a VERY DETAILED query:

		"Extract ALL startups/companies from this page. A startup/company is defined as:
		- A business entity with a name
		- Typically has additional information: description, website, sector, location, funding, etc.
		- May be displayed as a card, list item, or in a table
		
		EXCLUDE:
		- Navigation elements (menus, buttons, footers)
		- Section headers or category titles
		- Advertisements or sponsored content
		- UI elements (search boxes, filters, pagination)
		- Generic placeholders or empty cards
		
		For EACH startup found, extract:
		1. name (REQUIRED) - the exact company/startup name
		2. startup_url (if available) - link to the startup's detail page
		3. description (if available) - what the startup does
		4. website (if available) - the startup's main website URL
		5. sector (if available) - industry/category
		6. location (if available) - city, country, or region
		7. founded_year (if available) - year founded
		8. employees (if available) - team size
		9. funding_stage (if available) - seed, series A, etc.
		10. tags (if available) - technologies, categories, labels
		11. logo_url (if available) - logo image URL
		12. additional_info (if available) - any other relevant data
		
		IMPORTANT: Extract EVERY startup visible on this page. Do not skip any. If the page has many startups, list them ALL.
		If content is truncated, note that you can use start_from_char parameter to continue extraction."

		STEP 3.4: Handle Large Pages (Iterative Extraction)
		If a page has many startups and extraction is truncated:
		1. Note the `next_start_char` from the extraction result
		2. Use `extract` again with `start_from_char` parameter set to that value
		3. Continue until all startups are extracted
		4. Deduplicate based on name + URL

		STEP 3.5: Verify Completeness
		After extraction, verify you got everything:
		- Use `extract`: "How many startup cards/elements are visible on this page? Did I extract all of them?"
		- If you missed some, extract again with a more specific query
		- Scroll through the entire page to ensure nothing is hidden

		═══════════════════════════════════════════════════════════════════════════════
		📋 PHASE 4: SYSTEMATIC PAGE NAVIGATION
		═══════════════════════════════════════════════════════════════════════════════

		STEP 4.1: Visit All Paginated Pages
		- For each page number you discovered:
		  1. Navigate to that page (update URL or click page number)
		  2. Wait for load
		  3. Extract all startups (follow Phase 3)
		  4. Move to next page
		- Continue until you reach the last page (no more "Next" button or no new content)

		STEP 4.2: Visit All Category Pages
		- For each category/section you discovered:
		  1. Click or navigate to that category
		  2. Extract all startups from that category
		  3. Check if category has pagination (repeat Step 4.1)
		  4. Return to main page and go to next category

		STEP 4.3: Visit All Filtered Views
		- Apply different filters (if available):
		  * All startups
		  * Featured startups
		  * New startups
		  * By location
		  * By sector
		- Extract from each filtered view

		STEP 4.4: Check for Hidden Sections
		- Look for "Show All", "View More", "See All" buttons
		- Check for expandable sections or accordions
		- Look for tabs that might contain more startups
		- Use `extract`: "Are there any hidden sections, expandable areas, or tabs that might contain additional startups I haven't explored yet?"

		═══════════════════════════════════════════════════════════════════════════════
		📋 PHASE 5: FINAL VERIFICATION & COMPLETION
		═══════════════════════════════════════════════════════════════════════════════

		STEP 5.1: Final Verification
		Before finishing, verify exhaustiveness:
		1. Use `extract`: "Have I visited all pages with startups? Are there any navigation paths, sections, or pages I haven't explored yet?"
		2. Check if there are any "See All" or "Browse All" links you haven't clicked
		3. Verify you've handled all pagination
		4. Check if there are any search results or filtered views you haven't explored

		STEP 5.2: Deduplication
		- Remove duplicate startups (same name + same URL = duplicate)
		- If same startup appears on multiple pages, keep the entry with most complete information
		- Use name as primary key for deduplication

		STEP 5.3: Build Final Report
		Construct a `StartupExtractionReport` object with:
		- `source_url`: "{target_url}"
		- `startups`: Array of ALL unique startups extracted (limit to {max_startups} if needed)
		- `pages_visited`: List of all URLs you visited
		- `extraction_notes`: Brief notes about the site structure and extraction process

		STEP 5.4: Complete the Task
		Use `done` action with the complete `StartupExtractionReport` in the `data` field.
		Format as valid JSON matching the StartupExtractionReport schema.

		═══════════════════════════════════════════════════════════════════════════════
		🎯 CRITICAL SUCCESS FACTORS
		═══════════════════════════════════════════════════════════════════════════════

		✅ DO:
		- Explore EVERY navigation path that might lead to startups
		- Visit EVERY page (pagination, categories, filters)
		- Extract EVERY startup from each page
		- Use `extract` action intelligently with detailed queries
		- Scroll through entire pages to load all content
		- Verify completeness before finishing
		- Keep track of visited URLs
		- Deduplicate startups properly

		❌ DON'T:
		- Don't stop after first page
		- Don't miss pagination
		- Don't skip categories or sections
		- Don't extract navigation elements as startups
		- Don't assume all startups are on one page
		- Don't finish without verifying you've explored everything

		🔧 AVAILABLE TOOLS:
		- `navigate`: Navigate to URLs
		- `extract`: Use LLM to intelligently extract startups (PRIMARY TOOL)
		- `scroll`: Scroll pages to load content
		- `wait`: Wait for page loads
		- `click`: Click buttons/links
		- `find_text`: Find text on page
		- `evaluate`: Execute JavaScript if needed
		- `done`: Complete with final report

		📝 EXTRACTION STRATEGY SUMMARY:
		1. Understand site structure (where are startups?)
		2. Find all navigation paths (pages, categories, filters)
		3. Visit each path systematically
		4. Extract all startups from each page using intelligent LLM queries
		5. Verify completeness
		6. Deduplicate and compile final report

		🎯 REMEMBER: Your goal is EXHAUSTIVE extraction. Leave no startup behind!
		"""
	).strip()


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

		# Try to get structured output
		if history.structured_output:
			report = history.structured_output  # type: ignore[arg-type]
			return report

		# Try to extract from final result
		final_result = history.final_result()
		if final_result:
			try:
				report = StartupExtractionReport.model_validate_json(final_result)
				return report
			except ValidationError:
				match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if match:
					try:
						report = StartupExtractionReport.model_validate_json(match.group(1))
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
		'--max-startups',
		type=int,
		default=100000,
		help='Maximum number of startups to extract (default: 100000 to extract all)',
	)
	parser.add_argument(
		'--output',
		default='extracted_startups.json',
		help='Output JSON file path (default: ./extracted_startups.json)',
	)
	args = parser.parse_args()

	return UniversalStartupExtractorInput(
		url=args.url,
		max_startups=args.max_startups,
		output_path=Path(args.output),
	)


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		target_url = str(task_input.target_url)
		print(f"🚀 Starting Universal Startup Extractor")
		print(f"📍 Target URL: {target_url}")
		print(f"📊 Max startups: {task_input.max_startups}")
		print(f"💾 Output file: {task_input.output_path}")

		result = await run_universal_startup_extraction(task_input)

		if result is None:
			print("❌ Agent did not return structured data.")
			return

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


