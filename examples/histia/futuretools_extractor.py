"""
Agent designed to extract companies/tools from FutureTools newly-added page.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer

# Load environment variables immediately so the agent can access API keys.
load_dotenv()

# Configure timeouts BEFORE importing browser_use to ensure they're applied
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class FutureToolsInput(BaseModel):
	"""User-provided parameters for the FutureTools extraction task."""

	url: AnyHttpUrl = Field(
		default=AnyHttpUrl('https://www.futuretools.io/newly-added'),
		description='URL of the FutureTools newly-added page',
	)
	max_tools: int = Field(
		1000,
		ge=1,
		le=10000,
		description='Maximum number of tools to capture (use a high number like 1000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('futuretools_tools.json'),
		description='Destination for the JSON list of tools',
	)


class FutureToolsTool(BaseModel):
	"""Structured information for each FutureTools tool entry."""

	name: str = Field(..., description='Tool name exactly as written on the page')
	tool_url: str | None = Field(
		None,
		description='Complete URL to the tool page (if available)',
	)
	category: str | None = Field(
		None,
		description='Tool category/tag visible on the card (e.g., "Automation & Agents", "Productivity")',
	)
	description: str | None = Field(
		None,
		description='Tool description if available on the card',
	)


class FutureToolsReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='FutureTools URL that was analysed')
	tools: list[FutureToolsTool] = Field(
		...,
		min_length=1,
		description='Tool entries ordered as they appear on the page',
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

	def model_dump_json(self, **kwargs) -> str:
		"""Override model_dump_json to ensure AnyHttpUrl is serialized correctly."""
		return super().model_dump_json(**kwargs)


def _fallback_report(source_url: str, reason: str) -> FutureToolsReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	return FutureToolsReport(
		source_url=AnyHttpUrl(source_url),
		tools=[
			FutureToolsTool(
				name='Informations indisponibles',
				tool_url=source_url,
				category=None,
				description=None,
			)
		],
	)


def _normalize_url(url: str | None, base_url: str) -> str | None:
	"""Convert relative URLs to absolute URLs."""
	if not url:
		return None

	url = url.strip()
	if not url:
		return None

	# If it's already an absolute URL, return as is
	if url.startswith(('http://', 'https://')):
		return url

	# If it starts with /, make it relative to the base domain
	if url.startswith('/'):
		parsed_base = urlparse(base_url)
		return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"

	# Otherwise, try to resolve relative to base URL
	try:
		return urljoin(base_url, url)
	except Exception:
		return None


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> FutureToolsReport | None:
	"""Parse HTML sections from evaluate action to build FutureToolsReport."""

	tools: list[FutureToolsTool] = []
	seen_names: set[str] = set()

	# Handle both string (JSON array) and list
	if isinstance(html_sections, str):
		try:
			html_sections = json.loads(html_sections)
		except json.JSONDecodeError:
			# If it's not JSON, try to split by some delimiter or treat as single HTML
			html_sections = [html_sections]

	if not isinstance(html_sections, list):
		return None

	for idx, html_section in enumerate(html_sections):
		if not html_section or not isinstance(html_section, str):
			continue

		try:
			soup = BeautifulSoup(html_section, 'html.parser')

			# FutureTools structure: tools can be in <li> or <div> elements
			# Strategy 1: Look for containers (li, div) that contain tool links
			tool_elements = []
			
			# First, try to find containers with tool links
			all_links = soup.find_all('a', href=True)
			for link in all_links:
				href = link.get('href', '')
				if isinstance(href, str) and '/tools/' in href and '?tags' not in href:
					# Get parent container - try to find li or div parent
					container = None
					parent = link.parent
					while parent and parent.name != 'body' and parent.name != 'html':
						if parent.name == 'li':
							container = parent
							break
						elif parent.name == 'div' and not container:
							container = parent
						parent = parent.parent
					
					if container:
						# Check if we already added this container
						if container not in tool_elements:
							tool_elements.append(container)
					else:
						# Fallback: use direct parent
						parent = link.parent
						if parent and parent.name in ['div', 'article', 'li', 'section']:
							if parent not in tool_elements:
								tool_elements.append(parent)

			# Strategy 2: If still nothing, try to extract from the entire section
			if not tool_elements:
				tool_elements = [soup]

			for element in tool_elements:
				# FutureTools structure: each <li> contains:
				# - A link with href="/tools/..." (the tool link)
				# - A link with the tool name (also href="/tools/...")
				# - A generic div/span with the description text
				# - Potentially category tags/links
				
				# Extract name - look for link with tool name (href contains /tools/)
				name = None
				tool_link = element.select_one('a[href*="/tools/"]')
				
				# Check if this is a category link (href contains ?tags=) rather than a tool
				is_category_link = False
				if tool_link:
					href = tool_link.get('href', '')
					if isinstance(href, str) and ('?tags' in href or 'tags=' in href):
						is_category_link = True
				
				# Skip category links - they're not tools
				if is_category_link:
					continue
				
				if tool_link:
					name_text = tool_link.get_text(strip=True)
					# Filter out common non-tool names and empty text
					if name_text and len(name_text) > 1 and len(name_text) < 200:
						# Skip navigation/common text
						skip_words = ['home', 'about', 'contact', 'privacy', 'terms', 'login', 'sign up', 'menu', 'search', 'showing']
						if not any(skip in name_text.lower() for skip in skip_words):
							name = name_text

				# If no name found, try alternative selectors
				if not name:
					name_selectors = [
						'a[href*="/tools/"]',
						'h1', 'h2', 'h3', 'h4', 'h5',
						'strong', 'b',
					]
					for selector in name_selectors:
						name_elem = element.select_one(selector)
						if name_elem:
							name_text = name_elem.get_text(strip=True)
							if name_text and len(name_text) > 1 and len(name_text) < 200:
								skip_words = ['home', 'about', 'contact', 'privacy', 'terms', 'login', 'sign up', 'menu', 'search', 'showing']
								if not any(skip in name_text.lower() for skip in skip_words):
									name = name_text
									break

				# If no name found, skip this element
				if not name:
					continue

				# Skip duplicates
				name_lower = name.lower()
				if name_lower in seen_names:
					continue
				seen_names.add(name_lower)

				# Extract URL - look for tool link (href contains /tools/)
				tool_url = None
				if tool_link:
					href = tool_link.get('href', '')
					if isinstance(href, str) and href:
						tool_url = _normalize_url(href, source_url)
				else:
					# Fallback: look for any link
					link_elem = element.select_one('a[href]')
					if link_elem:
						href = link_elem.get('href', '')
						if isinstance(href, str) and href and not href.startswith('#') and '/tools/' in href:
							tool_url = _normalize_url(href, source_url)

				# Extract description - FutureTools structure: description is in a generic div/span
				# It's usually the text that comes after the tool name link
				description = None
				
				# Strategy 1: Look for divs/spans that are siblings or children of the tool link
				# The description is typically in a div that's a sibling of the link container
				if tool_link:
					# Get the parent container of the link
					link_parent = tool_link.parent
					if link_parent:
						# Look for siblings of the link parent that contain description text
						parent_siblings = []
						if link_parent.parent:
							parent_siblings = link_parent.parent.find_all(['div', 'span', 'p'], recursive=False)
						
						# Also check children of the parent
						parent_children = link_parent.find_all(['div', 'span', 'p'], recursive=True)
						
						# Combine siblings and children
						all_candidates = parent_siblings + parent_children
						
						for desc_elem in all_candidates:
							desc_text = desc_elem.get_text(strip=True)
							# Description should be longer than name and start with "A tool" or similar
							if desc_text and len(desc_text) > len(name) + 10 and len(desc_text) < 500:
								# Skip if it's just the name or very short
								desc_lower = desc_text.lower()
								if desc_lower != name_lower and name_lower not in desc_lower[:len(name_lower)+5]:
									# Check if it looks like a description (starts with "A tool", "A", etc.)
									if desc_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
										description = desc_text
										break
				
				# Strategy 2: Look for text that comes after the tool name in the DOM structure
				if not description and tool_link:
					# Get all text nodes and elements after the tool link
					all_text = element.get_text(separator='|', strip=True)
					# Split by the tool name to find what comes after
					parts = all_text.split(name, 1)
					if len(parts) > 1:
						text_after_name = parts[1].split('|')[0].strip() if '|' in parts[1] else parts[1].strip()
						# Clean up the text (remove extra separators)
						text_after_name = text_after_name.replace('|', ' ').strip()
						if text_after_name and len(text_after_name) > 10 and len(text_after_name) < 500:
							text_lower = text_after_name.lower()
							if text_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
								description = text_after_name
				
				# Strategy 3: Look for all div/span elements and find the one with description-like text
				if not description:
					all_divs_spans = element.find_all(['div', 'span', 'p'])
					for desc_elem in all_divs_spans:
						desc_text = desc_elem.get_text(strip=True)
						# Description should be longer than name and start with "A tool" or similar
						if desc_text and len(desc_text) > len(name) + 10 and len(desc_text) < 500:
							# Skip if it's just the name or very short
							desc_lower = desc_text.lower()
							if desc_lower != name_lower and name_lower not in desc_lower[:len(name_lower)+5]:
								# Check if it looks like a description (starts with "A tool", "A", etc.)
								if desc_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
									description = desc_text
									break
				
				# Strategy 2: Look for all text in the element and find the longest meaningful text
				# that comes after the tool name
				if not description:
					all_text = element.get_text(separator=' ', strip=True)
					# Find the tool name in the text and get what comes after
					name_index = all_text.lower().find(name_lower)
					if name_index >= 0:
						# Get text after the name
						text_after_name = all_text[name_index + len(name):].strip()
						# Look for description pattern
						if text_after_name and len(text_after_name) > 10 and len(text_after_name) < 500:
							text_lower = text_after_name.lower()
							if text_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
								description = text_after_name
					
					# If still not found, try to find any long text that looks like a description
					if not description:
						# Split text by common separators and find the longest meaningful part
						parts = all_text.split(name)
						for part in parts:
							part = part.strip()
							if part and len(part) > len(name) + 10 and len(part) < 500:
								part_lower = part.lower()
								if part_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
									description = part
									break

				# Strategy 3: Look for specific description selectors (generic divs/spans without specific classes)
				if not description:
					# Find divs/spans that don't have name/title classes and contain description-like text
					generic_elements = element.find_all(['div', 'span'], class_=False)
					for desc_elem in generic_elements:
						desc_text = desc_elem.get_text(strip=True)
						if desc_text and len(desc_text) > len(name) + 10 and len(desc_text) < 500:
							desc_lower = desc_text.lower()
							if desc_lower != name_lower and desc_lower.startswith(('a tool', 'a ', 'an ', 'the ', 'tool')):
								description = desc_text
								break

				# Extract category - look for category tags/links
				# Categories are in links with href containing "?tags=" or "?tags-n5zn="
				# They're usually in a list (<ul> or <ol>) within the tool element
				category = None
				
				# Strategy 1: Look for links with tags in href within the same element
				# These are the category filter links - extract their text as the category
				category_links = element.select('a[href*="tags"]')
				if category_links:
					# Get the first category link's text (usually there's one category per tool)
					for cat_link in category_links:
						cat_text = cat_link.get_text(strip=True)
						# Filter reasonable category length
						if cat_text and 2 < len(cat_text) < 100:
							# Skip navigation/common text
							skip_cats = ['home', 'about', 'contact', 'read more', 'learn more', 'view', 'click', 'showing']
							cat_text_lower = cat_text.lower()
							if not any(skip in cat_text_lower for skip in skip_cats):
								category = cat_text
								break
				
				# Strategy 2: Look for category text in list items (categories are often in <li> within <ul>/<ol>)
				if not category:
					# Look for lists that might contain categories
					lists_in_element = element.find_all(['ul', 'ol'])
					for list_elem in lists_in_element:
						list_items = list_elem.find_all('li')
						for li in list_items:
							# Check if this list item contains a category link
							all_links_in_li = li.find_all('a', href=True)
							for link in all_links_in_li:
								href = link.get('href', '')
								if isinstance(href, str) and ('tags' in href or 'tags-n5zn' in href):
									cat_text = link.get_text(strip=True)
									if cat_text and 2 < len(cat_text) < 100:
										skip_cats = ['home', 'about', 'contact', 'read more', 'learn more']
										if not any(skip in cat_text.lower() for skip in skip_cats):
											category = cat_text
											break
							if category:
								break
						if category:
							break
				
				# Strategy 3: Look for category text in elements with tag-related classes
				if not category:
					category_selectors = [
						'[class*="category"]', '[class*="tag"]', '[class*="badge"]',
						'span[class*="category"]', 'span[class*="tag"]',
						'div[class*="category"]', 'div[class*="tag"]',
					]
					for selector in category_selectors:
						category_elems = element.select(selector)
						for category_elem in category_elems:
							category_text = category_elem.get_text(strip=True)
							if category_text and 2 < len(category_text) < 100:
								skip_cats = ['home', 'about', 'contact', 'read more', 'learn more', 'view', 'click', 'showing']
								category_lower = category_text.lower()
								if not any(skip in category_lower for skip in skip_cats):
									if category_text[0].isupper() or len(category_text.split()) <= 3:
										category = category_text
										break
						if category:
							break

				# Only add tool if we have at least a name
				if name:
					tools.append(FutureToolsTool(
						name=name,
						tool_url=tool_url,
						category=category,
						description=description,
					))

		except Exception as e:
			# Skip malformed sections
			continue

	if tools:
		return FutureToolsReport(
			source_url=AnyHttpUrl(source_url),
			tools=tools,
		)
	return None


async def _scroll_to_bottom_naive(browser, max_scrolls: int = 20) -> None:
	"""Scroll progressivement jusqu'en bas de la page de manière naïve (sans LLM) en utilisant JavaScript direct."""
	print("📜 Démarrage du scroll progressif jusqu'en bas...")

	# Get current page
	page = await browser.get_current_page()
	if not page:
		raise RuntimeError("No current page available")

	# Get viewport height using JavaScript
	try:
		viewport_height = await page.evaluate("() => window.innerHeight || document.documentElement.clientHeight")
		viewport_height = int(viewport_height) if viewport_height else 1000
	except Exception:
		viewport_height = 1000  # Fallback

	print(f"   📏 Hauteur du viewport: {viewport_height}px")

	# Wait for initial page load
	print("   ⏳ Attente initiale de 3 secondes pour le chargement de la page...")
	await asyncio.sleep(3)

	# Scroll progressivement jusqu'en bas
	scroll_count = 0
	last_scroll_position = -1
	no_change_count = 0

	for i in range(max_scrolls):
		# Get current scroll position
		try:
			current_position = await page.evaluate("() => window.pageYOffset || document.documentElement.scrollTop")
			current_position = int(current_position) if current_position else 0
		except Exception:
			current_position = 0

		# Check if we're at the bottom
		if scroll_count >= 5 and current_position == last_scroll_position:
			no_change_count += 1
			if no_change_count >= 3:  # No change for 3 consecutive scrolls
				print(f"   ✅ Arrivé en bas après {scroll_count} scrolls")
				break
		else:
			no_change_count = 0

		last_scroll_position = current_position

		# Scroll down by one page using JavaScript
		try:
			await page.evaluate(f"() => window.scrollBy(0, {viewport_height})")
		except Exception as e:
			print(f"   ⚠️ Erreur lors du scroll: {e}")
			break

		scroll_count += 1
		await asyncio.sleep(2)  # Wait for content to load

		if scroll_count % 5 == 0:
			print(f"   📜 {scroll_count} scrolls effectués...")

	# Final wait to ensure everything is loaded
	print("   ⏳ Attente finale de 3 secondes pour le chargement complet...")
	await asyncio.sleep(3)

	print(f"✅ Scroll terminé: {scroll_count} scrolls effectués au total")


async def run_futuretools_extraction(task_input: FutureToolsInput) -> FutureToolsReport | None:
	"""Execute the agent and return the structured list of tools."""

	print("🔧 Configuration du LLM...")
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
		print("✅ Utilisation de ChatBrowserUse")
	else:
		model_name = os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		if 'gemini' not in model_name.lower():
			model_name = 'gemini-2.5-flash-lite-preview-09-2025'
			print(f"⚠️  Modèle non-Gemini détecté, utilisation de {model_name} à la place")
		is_gemini = 'gemini' in model_name.lower()
		llm = ChatOpenAI(
			model=model_name,
			timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
			max_retries=3,
			max_completion_tokens=90960,
			add_schema_to_system_prompt=is_gemini,
			dont_force_structured_output=is_gemini,
		)
		print(f"✅ Utilisation de ChatOpenAI avec le modèle: {model_name}")

	print("🌐 Création du navigateur...")
	browser = Browser(headless=False)
	await browser.start()

	try:
		# Navigate to the FutureTools URL
		futuretools_url = str(task_input.url)
		print(f"📍 Navigation vers: {futuretools_url}")
		navigate_event = NavigateToUrlEvent(url=futuretools_url, new_tab=False)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event

		# Get the current page after navigation
		page = await browser.get_current_page()
		if not page:
			page = await browser.new_page(futuretools_url)

		await asyncio.sleep(5)  # Wait for initial page load

		# Scroll to bottom naively (without LLM)
		await _scroll_to_bottom_naive(browser, max_scrolls=20)

		# Extract HTML directly using JavaScript
		print("🔍 Extraction directe du HTML des outils...")
		current_page = await browser.get_current_page()
		if not current_page:
			raise RuntimeError("No current page available after scroll")

		# Try multiple selectors to find tool elements
		# FutureTools structure: tools are in <li> elements with links to /tools/
		# Use arrow function format for page.evaluate() - NO COMMENTS allowed
		# First, try to debug what's on the page
		debug_code = """() => {
			const allLis = document.querySelectorAll('li');
			const allLinks = document.querySelectorAll('a[href]');
			const toolLinks = document.querySelectorAll('a[href*="/tools/"]');
			return JSON.stringify({
				totalLis: allLis.length,
				totalLinks: allLinks.length,
				toolLinks: toolLinks.length,
				sampleHrefs: Array.from(toolLinks).slice(0, 5).map(a => a.getAttribute('href'))
			});
		}"""
		
		# Try to debug first
		try:
			debug_result = await current_page.evaluate(debug_code)
			print(f"   🔍 Debug - Résultat: {debug_result}")
			if debug_result:
				debug_data = json.loads(debug_result)
				print(f"   📊 Total <li>: {debug_data.get('totalLis', 0)}")
				print(f"   📊 Total liens: {debug_data.get('totalLinks', 0)}")
				print(f"   📊 Liens /tools/: {debug_data.get('toolLinks', 0)}")
				print(f"   📊 Exemples hrefs: {debug_data.get('sampleHrefs', [])}")
		except Exception as debug_err:
			print(f"   ⚠️ Erreur debug: {debug_err}")
		
		# Now try extraction with multiple strategies
		# Get parent containers to include description and category context
		# IMPORTANT: Include the full container that has both the tool link AND the category list
		extraction_code = """() => {
			let result = [];
			const directToolLinks = Array.from(document.querySelectorAll('a[href*="/tools/"]'));
			if (directToolLinks.length > 0) {
				result = directToolLinks.map(link => {
					let container = link.closest('li');
					if (!container) {
						container = link.closest('div[class*="tool"]') || link.closest('div[class*="item"]');
					}
					if (!container) {
						let parent = link.parentElement;
						let bestContainer = parent;
						while (parent && parent.tagName !== 'BODY' && parent.tagName !== 'HTML') {
							if (parent.classList && parent.classList.length > 0) {
								const hasCategoryList = parent.querySelector('ul, ol') && parent.querySelector('a[href*="tags"]');
								if (hasCategoryList) {
									bestContainer = parent;
									break;
								}
								if (parent.classList.contains('tool') || parent.classList.contains('item') || parent.classList.contains('collection-item')) {
									bestContainer = parent;
								}
							}
							parent = parent.parentElement;
						}
						container = bestContainer;
					}
					if (!container) {
						container = link.parentElement;
					}
					return container ? container.outerHTML : link.outerHTML;
				});
			}
			return JSON.stringify(result);
		}"""

		try:
			html_sections_json = await current_page.evaluate(extraction_code)
			
			# Debug: print what we got
			if html_sections_json:
				print(f"   🔍 Données reçues (premiers 200 caractères): {html_sections_json[:200]}")
			else:
				print(f"   ⚠️ Aucune donnée retournée par evaluate")

			if html_sections_json and html_sections_json != '[]' and html_sections_json.strip():
				# Parse the JSON string
				try:
					html_sections = json.loads(html_sections_json)
					print(f"   ✅ {len(html_sections)} éléments HTML extraits")
				except json.JSONDecodeError as json_err:
					print(f"   ⚠️ Erreur de parsing JSON: {json_err}")
					print(f"   🔍 Contenu reçu: {html_sections_json[:500]}")
					html_sections = []

				if html_sections and len(html_sections) > 0:
					# Debug: check if categories are in the HTML
					if html_sections and isinstance(html_sections[0], str):
						first_html = html_sections[0]
						has_tags_links = 'tags' in first_html.lower() or 'tags-n5zn' in first_html.lower()
						has_lists = '<ul' in first_html or '<ol' in first_html
						print(f"   🔍 Debug HTML: contient liens tags={has_tags_links}, contient listes={has_lists}")
						if has_tags_links:
							# Try to find category links in the HTML
							from bs4 import BeautifulSoup
							test_soup = BeautifulSoup(first_html, 'html.parser')
							test_cat_links = test_soup.select('a[href*="tags"]')
							if test_cat_links:
								print(f"   🔍 Debug: {len(test_cat_links)} liens de catégorie trouvés dans le premier élément")
								for i, link in enumerate(test_cat_links[:3]):
									print(f"      - Lien {i+1}: {link.get_text(strip=True)} (href: {link.get('href', '')})")
					
					# Parse HTML sections directly
					html_report = _parse_html_sections(html_sections, str(task_input.url))
					if html_report and html_report.tools and len(html_report.tools) > 0:
						print(f"   ✅ {len(html_report.tools)} outils parsés depuis le HTML")
						
						# Debug: check categories in parsed tools
						tools_with_categories = [t for t in html_report.tools if t.category]
						print(f"   🔍 Debug: {len(tools_with_categories)}/{len(html_report.tools)} outils ont une catégorie")

						# Remove duplicates based on name
						seen_names = set()
						unique_tools = []
						for tool in html_report.tools:
							if tool.name.lower() not in seen_names:
								seen_names.add(tool.name.lower())
								unique_tools.append(tool)

						html_report.tools = unique_tools

						# Limit to max_tools if needed
						if task_input.max_tools < 10000 and len(html_report.tools) > task_input.max_tools:
							html_report.tools = html_report.tools[:task_input.max_tools]

						return html_report
					else:
						print(f"   ⚠️ Aucun outil parsé depuis le HTML (report: {html_report})")
						if html_sections:
							print(f"   🔍 Premier élément HTML (premiers 500 caractères): {html_sections[0][:500] if isinstance(html_sections[0], str) else str(html_sections[0])[:500]}")
				else:
					print(f"   ⚠️ Tableau HTML vide ou invalide")
			else:
				print(f"   ⚠️ Aucune donnée extraite (html_sections_json vide ou invalide)")
		except json.JSONDecodeError as json_err:
			print(f"   ⚠️ Erreur de parsing JSON dans le bloc principal: {json_err}")
			print(f"   🔍 Contenu reçu: {html_sections_json[:500] if 'html_sections_json' in locals() else 'N/A'}")
		except Exception as e:
			print(f"   ⚠️ Erreur lors de l'extraction directe: {e}")
			import traceback
			traceback.print_exc()

		# Fallback: use agent if direct extraction failed
		print("🤖 Utilisation de l'agent comme fallback...")
		task = dedent(
			f"""
			Tu es un agent spécialisé dans l'extraction d'outils depuis la page FutureTools.

			Objectif CRITIQUE:
			- Tu es déjà sur la page: {futuretools_url}
			- IMPORTANT: Utilise l'action `evaluate` pour extraire le HTML des outils depuis le DOM
			- Identifie et extrait TOUS les outils présents sur cette page, SANS AUCUNE EXCEPTION.
			- Pour chaque outil, capture:
			  • `name`: nom exact de l'outil tel qu'affiché
			  • `tool_url`: URL complète vers la page de l'outil (si disponible)
			  • `category`: catégorie/tag visible (ex: "Automation & Agents", "Productivity")
			  • `description`: description de l'outil si disponible

			Processus:
			1. PREMIÈRE TENTATIVE: Utilise l'action `evaluate` avec le code JavaScript suivant pour extraire le HTML:
			   (function(){{const toolItems = Array.from(document.querySelectorAll('li')); const toolListItems = toolItems.filter(li => {{const toolLink = li.querySelector('a[href*="/tools/"]'); return toolLink !== null;}}); const result = toolListItems.map(li => li.outerHTML); return JSON.stringify(result);}})()
			   Note: L'action `evaluate` utilise le format IIFE (function(){{...}})() avec le paramètre `code`
			   IMPORTANT: Ne pas utiliser de commentaires dans le code JavaScript
			
			2. SI `evaluate` RETOURNE UN TABLEAU VIDE []:
			   - Utilise l'action `extract` avec la requête suivante:
			   {{"extract": {{"query": "Extract all tools listed on this page. For each tool, extract: the tool name (text of the link), the tool URL (href attribute of the link), the category/tag if visible, and the description text that follows the tool name", "extract_links": true}}}}
			   - Parse le résultat markdown pour extraire les informations
			
			3. Pour chaque outil trouvé, extrais:
			   - `name`: nom exact de l'outil (texte du lien)
			   - `tool_url`: URL complète (attribut href du lien, normalisé en URL absolue)
			   - `category`: catégorie/tag visible si présent
			   - `description`: texte de description qui suit le nom
			
			4. Construis un objet FutureToolsReport avec tous les outils extraits
			5. Utilise l'action `done` avec le champ `data` contenant l'objet FutureToolsReport complet

			Règles importantes:
			- ⚠️ NE PAS naviguer - tu es déjà sur la bonne page!
			- ⚠️ NE PAS scroller - tout le contenu a déjà été chargé!
			- ⚠️ Utilise UNIQUEMENT `evaluate` pour extraire le HTML depuis le DOM
			- Format `evaluate`: {{"evaluate": {{"code": "TON_CODE_JAVASCRIPT"}}}}
			- Ne visite JAMAIS les pages individuelles des outils - extrais toutes les informations depuis la page principale
			"""
		).strip()

		agent = Agent(
			task=task,
			llm=llm,
			browser=browser,
			output_model_schema=FutureToolsReport,
			use_vision='auto',
			vision_detail_level='auto',
			step_timeout=300,
			llm_timeout=180,
			max_failures=5,
			max_history_items=10,
			directly_open_url=False,
		)
		print("✅ Agent créé")

	print("▶️  Démarrage de l'exécution de l'agent...")
	history = await agent.run()
	print("✅ Exécution terminée")
	print_llm_usage_summary(history)

		# Try to get structured output first
		if history.structured_output:
			return history.structured_output  # type: ignore[arg-type]

		# Try to extract from final result
		final_result = history.final_result()
		if final_result:
			try:
				report = FutureToolsReport.model_validate_json(final_result)
				return report
			except ValidationError:
				import re
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if json_match:
					try:
						report = FutureToolsReport.model_validate_json(json_match.group(1))
						return report
					except ValidationError:
						pass

		return _fallback_report(str(task_input.url), "L'agent n'a pas pu extraire les données.")

	finally:
		# Close browser to free resources
		print("🧹 Fermeture du navigateur...")
		try:
			await browser.kill()
			print("✅ Navigateur fermé")
		except Exception as e:
			print(f"⚠️  Erreur lors de la fermeture du navigateur: {e}")


def parse_arguments() -> FutureToolsInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Extrait les outils depuis la page FutureTools newly-added')
	parser.add_argument(
		'--url',
		default='https://www.futuretools.io/newly-added',
		help='URL de la page FutureTools (par défaut: https://www.futuretools.io/newly-added)',
	)
	parser.add_argument(
		'--max-tools',
		type=int,
		default=1000,
		help='Nombre maximal d\'outils à extraire (par défaut: 1000)',
	)
	parser.add_argument(
		'--output',
		default='futuretools_tools.json',
		help='Chemin du fichier JSON résultat (par défaut: ./futuretools_tools.json)',
	)
	args = parser.parse_args()
	return FutureToolsInput(url=AnyHttpUrl(args.url), max_tools=args.max_tools, output_path=Path(args.output))


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		futuretools_url = str(task_input.url)
		print(f"🚀 Démarrage de l'extraction depuis: {futuretools_url}")
		print(f"📊 Nombre max d'outils: {task_input.max_tools}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")

		result = await run_futuretools_extraction(task_input)

		if result is None:
			print("❌ L'agent n'a retourné aucune donnée structurée.")
			return

		output_json = result.model_dump_json(indent=2, ensure_ascii=False)
		output_path = task_input.output_path
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(output_json, encoding='utf-8')

		print(output_json)
		print(f'\n✅ Listing sauvegardé dans: {output_path.resolve()}')
	except KeyboardInterrupt:
		print("\n⚠️  Interruption utilisateur détectée.")
		raise
	except Exception as e:
		print(f"❌ Erreur lors de l'exécution: {e}")
		import traceback
		traceback.print_exc()
		raise


if __name__ == '__main__':
	asyncio.run(main())

