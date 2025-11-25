"""
Agent designed to extract ALL startups from Zone Secure website.

This agent is designed to extract startups exhaustively without missing any.
It handles pagination, infinite scroll, and uses multiple extraction strategies.

Usage:
    python examples/histia/zone_secure_startups_agent.py

    # With custom URL
    python examples/histia/zone_secure_startups_agent.py --url "https://fr.zone-secure.net/20412/2540033/#page=1"

    # With custom output path
    python examples/histia/zone_secure_startups_agent.py --output startups.json
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
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer

# Load environment variables immediately
load_dotenv()

# Configure timeouts for heavy pages
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI, Tools
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class ZoneSecureStartupsInput(BaseModel):
	"""User-provided parameters for the Zone Secure startups extraction task."""

	url: str = Field(
		default='https://fr.zone-secure.net/20412/2540033/#page=1',
		description='URL of the Zone Secure startups page',
	)
	max_startups: int = Field(
		10000,
		ge=1,
		le=50000,
		description='Maximum number of startups to capture (use a high number like 10000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('zone_secure_startups.json'),
		description='Destination for the JSON list of startups',
	)

	@property
	def startups_url(self) -> AnyHttpUrl:
		"""Build the Zone Secure startups URL."""
		return AnyHttpUrl(self.url)


class ZoneSecureStartup(BaseModel):
	"""Structured information for each Zone Secure startup entry."""

	name: str = Field(..., description='Startup name exactly as written on the page')
	startup_url: str | None = Field(
		None,
		description='Complete URL to the startup detail page if available',
	)
	description: str | None = Field(
		None,
		description='Startup description/tagline if available',
	)
	website: str | None = Field(
		None,
		description='Startup website URL if available',
	)
	sector: str | None = Field(
		None,
		description='Startup sector/industry if available',
	)
	location: str | None = Field(
		None,
		description='Startup location if available',
	)
	founded_year: int | None = Field(
		None,
		description='Year the startup was founded if available',
	)
	employees: str | None = Field(
		None,
		description='Number of employees or employee range if available',
	)
	tags: list[str] = Field(
		default_factory=list,
		description='Startup tags/categories visible on the card',
	)
	logo_url: str | None = Field(
		None,
		description='URL to startup logo if available',
	)
	additional_info: dict[str, Any] = Field(
		default_factory=dict,
		description='Any additional information found on the card',
	)


class ZoneSecureStartupsReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='Startups page URL that was analysed')
	startups: list[ZoneSecureStartup] = Field(
		...,
		min_length=0,
		description='Startup entries ordered as they appear on the page',
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


def _fallback_report(source_url: str, reason: str) -> ZoneSecureStartupsReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	return ZoneSecureStartupsReport(
		source_url=AnyHttpUrl(source_url),
		startups=[],
	)


def _filter_navigation_elements(startups: list[ZoneSecureStartup]) -> list[ZoneSecureStartup]:
	"""Filter out navigation elements and section titles that are not real startups."""
	
	# List of known navigation elements and section titles to exclude
	excluded_names = {
		'forum', 'remerciements', 'plan', 'sommaire',
		'rechercher', 'partager', 'télécharger', 'plein écran',
		'onglets', 'retour au document', 'toutes les pages',
		'conseil audit', 'construction & transport', 'energie environnement',
		'finance banque assurance', 'formation', 'it digital', 'public',
		'production supply chain', 'santé biotech', 'start-up', 'startup',
	}
	
	filtered: list[ZoneSecureStartup] = []
	
	for startup in startups:
		if not startup.name:
			continue
		
		name_lower = startup.name.lower().strip()
		
		# Skip if it's a known navigation element
		if name_lower in excluded_names:
			continue
		
		# Skip if it contains multiple navigation words (like "OngletsRetour au document...")
		if any(excluded in name_lower for excluded in excluded_names):
			continue
		
		# Skip if it's just a section title without any other information
		# A real startup should have at least a description, website, or other info
		has_additional_info = (
			startup.description or
			startup.website or
			startup.sector or
			startup.location or
			startup.founded_year or
			startup.employees or
			startup.tags or
			startup.logo_url or
			startup.startup_url
		)
		
		# If it's a short name without additional info, it's likely navigation
		if not has_additional_info and len(name_lower) < 15:
			# But allow it if it looks like a company name (capitalized, multiple words, etc.)
			words = name_lower.split()
			if len(words) == 1 and name_lower.islower():
				continue  # Single lowercase word without info = likely navigation
		
		filtered.append(startup)
	
	return filtered


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
		from urllib.parse import urlparse, urljoin
		parsed_base = urlparse(base_url)
		return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"

	# Otherwise, try to resolve relative to base URL
	try:
		from urllib.parse import urljoin
		return urljoin(base_url, url)
	except Exception:
		return None


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> ZoneSecureStartupsReport | None:
	"""Parse HTML sections from evaluate action to build ZoneSecureStartupsReport."""
	
	startups: list[ZoneSecureStartup] = []
	
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
			
			# Try multiple selectors for startup cards
			# Common patterns: cards, items, articles, divs with specific classes
			startup_element = (
				soup.find('article') or
				soup.find('div', class_=re.compile(r'startup|company|card|item|listing', re.I)) or
				soup.find('a', href=re.compile(r'startup|company', re.I)) or
				soup.find('li', class_=re.compile(r'startup|company|card|item', re.I)) or
				soup
			)
			
			# Extract name - try various selectors
			name = None
			name_selectors = [
				'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
				'[class*="name"]', '[class*="title"]', '[class*="heading"]',
				'a[href*="startup"]', 'a[href*="company"]',
				'[data-name]', '[data-title]',
			]
			for selector in name_selectors:
				name_elem = startup_element.select_one(selector)
				if name_elem:
					name = name_elem.get_text(strip=True)
					if name and len(name) > 1:
						break
			
			# If no name found, try getting text from the element itself
			if not name:
				name = startup_element.get_text(strip=True)
				# Take first line or first 100 chars
				if '\n' in name:
					name = name.split('\n')[0].strip()
				if len(name) > 100:
					name = name[:100].strip()
			
			# Extract startup URL
			startup_url = None
			link = (
				startup_element.find('a', href=True) or
				startup_element.select_one('a[href]') or
				(startup_element if startup_element.name == 'a' and startup_element.get('href') else None)
			)
			if link:
				href = link.get('href', '') if hasattr(link, 'get') else (link if isinstance(link, str) else None)
				if href:
					href_str = str(href).strip()
					if href_str:
						startup_url = _normalize_url(href_str, source_url)
			
			# Extract description
			description = None
			desc_selectors = [
				'[class*="description"]', '[class*="tagline"]', '[class*="summary"]',
				'[class*="bio"]', '[class*="about"]',
				'p', '[class*="text"]',
			]
			for selector in desc_selectors:
				desc_elem = startup_element.select_one(selector)
				if desc_elem:
					desc_text = desc_elem.get_text(strip=True)
					# Skip if it's the same as name
					if desc_text and desc_text != name and len(desc_text) > 3:
						description = desc_text
						break
			
			# Extract website
			website = None
			website_links = startup_element.select('a[href^="http"]:not([href*="zone-secure.net"])')
			for website_link in website_links:
				href = website_link.get('href', '')
				if href:
					href_str = str(href).lower()
					if not any(domain in href_str for domain in ['zone-secure.net', 'facebook.com', 'twitter.com', 'linkedin.com']):
						website = str(href).strip()
						break
			
			# Extract sector/stage/location from various patterns
			sector = None
			location = None
			founded_year = None
			employees = None
			
			# Look for metadata in spans, divs, or list items
			metadata_elements = startup_element.select('span, div, li, p')
			for elem in metadata_elements:
				text = elem.get_text(strip=True)
				if not text or text == name:
					continue
				
				# Sector detection
				if not sector and any(keyword in text.lower() for keyword in ['sector', 'industry', 'category', 'secteur', 'industrie']):
					sector = text.split(':', 1)[-1].strip() if ':' in text else text
				
				# Location detection
				if not location and any(keyword in text.lower() for keyword in ['location', 'city', 'country', 'ville', 'pays', 'paris', 'france']):
					location = text.split(':', 1)[-1].strip() if ':' in text else text
				
				# Founded year detection
				if not founded_year:
					year_match = re.search(r'\b(19|20)\d{2}\b', text)
					if year_match:
						try:
							founded_year = int(year_match.group(0))
						except ValueError:
							pass
				
				# Employees detection
				if not employees and any(keyword in text.lower() for keyword in ['employees', 'team', 'people', 'employés', 'équipe']):
					employees = text
			
			# Extract tags
			tags = []
			tag_elements = startup_element.select('[class*="tag"], [class*="badge"], [class*="category"], [class*="label"]')
			for tag_elem in tag_elements:
				tag_text = tag_elem.get_text(strip=True)
				if tag_text and tag_text not in tags:
					tags.append(tag_text)
			
			# Extract logo
			logo_url = None
			logo_img = startup_element.select_one('img[src], img[data-src], img[data-lazy-src]')
			if logo_img:
				logo_src = logo_img.get('src') or logo_img.get('data-src') or logo_img.get('data-lazy-src')
				if logo_src:
					logo_src_str = str(logo_src).strip()
					if logo_src_str:
						logo_url = _normalize_url(logo_src_str, source_url)
			
			# Extract additional info
			additional_info: dict[str, Any] = {}
			
			# Only add startup if we have at least a name
			if name and len(name.strip()) > 0:
				# Build startup with only fields that have values
				startup_dict: dict[str, Any] = {'name': name.strip()}
				
				# Only add fields that have values
				if startup_url:
					startup_dict['startup_url'] = startup_url
				if description:
					startup_dict['description'] = description
				if website:
					startup_dict['website'] = website
				if sector:
					startup_dict['sector'] = sector
				if location:
					startup_dict['location'] = location
				if founded_year is not None:
					startup_dict['founded_year'] = founded_year
				if employees:
					startup_dict['employees'] = employees
				if tags:
					startup_dict['tags'] = tags
				if logo_url:
					startup_dict['logo_url'] = logo_url
				if additional_info:
					startup_dict['additional_info'] = additional_info
				
				startups.append(ZoneSecureStartup.model_validate(startup_dict))
		except Exception as e:
			# Skip malformed sections
			continue
	
	if startups:
		return ZoneSecureStartupsReport(
			source_url=AnyHttpUrl(source_url),
			startups=startups,
		)
	return None


async def _scroll_to_bottom_exhaustive(browser, max_scrolls: int = 100) -> None:
	"""Scroll exhaustively until no more content loads."""
	print("📜 Démarrage du scroll exhaustif jusqu'en bas...")
	
	page = await browser.get_current_page()
	if not page:
		raise RuntimeError("No current page available")
	
	# Get viewport height
	try:
		viewport_height = await page.evaluate("() => window.innerHeight || document.documentElement.clientHeight")
		viewport_height = int(viewport_height) if viewport_height else 1000
	except Exception:
		viewport_height = 1000
	
	print(f"   📏 Hauteur du viewport: {viewport_height}px")
	
	# Wait for initial page load
	print("   ⏳ Attente initiale de 3 secondes pour le chargement de la page...")
	await asyncio.sleep(3)
	
	# Check initial content
	try:
		# Try various selectors to count startups
		initial_count_script = """
		() => {
			const selectors = [
				'article', '[class*="startup"]', '[class*="company"]', 
				'[class*="card"]', '[class*="item"]', '[class*="listing"]'
			];
			for (const selector of selectors) {
				const count = document.querySelectorAll(selector).length;
				if (count > 0) return { selector, count };
			}
			return { selector: 'none', count: 0 };
		}
		"""
		initial_info = await page.evaluate(initial_count_script)
		initial_count = initial_info.get('count', 0) if isinstance(initial_info, dict) else 0
		print(f"   📊 Éléments initiaux trouvés: {initial_count}")
	except Exception:
		initial_count = 0
	
	# Scroll progressively
	scroll_count = 0
	last_scroll_position = -1
	last_count = initial_count
	no_change_count = 0
	
	for i in range(max_scrolls):
		# Get current scroll position and content count
		try:
			scroll_info = await page.evaluate("""
				() => {
					const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
					const scrollHeight = document.documentElement.scrollHeight;
					const clientHeight = window.innerHeight || document.documentElement.clientHeight;
					const isAtBottom = (scrollTop + clientHeight >= scrollHeight - 10);
					
					// Count startups
					const selectors = [
						'article', '[class*="startup"]', '[class*="company"]', 
						'[class*="card"]', '[class*="item"]', '[class*="listing"]'
					];
					let count = 0;
					for (const selector of selectors) {
						const elements = document.querySelectorAll(selector);
						if (elements.length > count) count = elements.length;
					}
					
					return {
						scrollTop: scrollTop,
						scrollHeight: scrollHeight,
						clientHeight: clientHeight,
						isAtBottom: isAtBottom,
						count: count,
						distanceFromBottom: scrollHeight - (scrollTop + clientHeight)
					};
				}
			""")
			
			if isinstance(scroll_info, str):
				try:
					scroll_info = json.loads(scroll_info)
				except json.JSONDecodeError:
					scroll_info = {}
			
			current_position = scroll_info.get('scrollTop', 0) if isinstance(scroll_info, dict) else 0
			current_count = scroll_info.get('count', 0) if isinstance(scroll_info, dict) else 0
			is_at_bottom = scroll_info.get('isAtBottom', False) if isinstance(scroll_info, dict) else False
		except Exception:
			current_position = 0
			current_count = 0
			is_at_bottom = False
		
		# Check if we're at the bottom and no new content
		if scroll_count >= 5:
			if is_at_bottom and current_position == last_scroll_position and current_count == last_count:
				no_change_count += 1
				if no_change_count >= 3:
					print(f"   ✅ Arrivé en bas après {scroll_count} scrolls ({current_count} éléments au total)")
					break
			else:
				no_change_count = 0
		
		if current_count > last_count:
			print(f"   📊 {current_count} éléments trouvés (nouveau contenu chargé)")
		
		last_scroll_position = current_position
		last_count = current_count
		
		# Scroll down
		try:
			await page.evaluate(f"() => window.scrollBy(0, {viewport_height})")
			await page.evaluate(f"() => document.documentElement.scrollTop += {viewport_height}")
		except Exception as e:
			print(f"   ⚠️ Erreur lors du scroll: {e}")
			try:
				await page.evaluate(f"() => window.scrollTo(0, window.pageYOffset + {viewport_height})")
			except Exception:
				break
		
		scroll_count += 1
		
		# Wait for content to load
		await asyncio.sleep(2)
		
		if scroll_count % 10 == 0:
			print(f"   📜 {scroll_count} scrolls effectués, {current_count} éléments trouvés...")
	
	# Final wait
	print("   ⏳ Attente finale de 3 secondes pour le chargement complet...")
	await asyncio.sleep(3)
	
	# Final check
	try:
		final_info = await page.evaluate(initial_count_script)
		final_count = final_info.get('count', 0) if isinstance(final_info, dict) else 0
		print(f"   📊 Éléments finaux trouvés: {final_count}")
	except Exception:
		final_count = 0
	
	print(f"✅ Scroll terminé: {scroll_count} scrolls effectués au total, {final_count} éléments trouvés")


async def _extract_startups_with_scroll(page, source_url: str, max_startups: int) -> list[ZoneSecureStartup]:
	"""Extract startups using iterative scroll and extraction strategy."""
	
	all_startups: list[ZoneSecureStartup] = []
	seen_startups: set[tuple[str, str]] = set()  # (name, url) tuple for deduplication
	iteration = 0
	no_new_startups_count = 0
	
	while iteration < 50:  # Max 50 iterations
		iteration += 1
		print(f"   🔄 Itération {iteration} d'extraction...")
		
		# Extract HTML of visible startup cards
		extract_script = """
		() => {
			const selectors = [
				'article', '[class*="startup"]', '[class*="company"]', 
				'[class*="card"]', '[class*="item"]', '[class*="listing"]'
			];
			
			let allElements = [];
			for (const selector of selectors) {
				const elements = Array.from(document.querySelectorAll(selector));
				if (elements.length > 0) {
					allElements = elements;
					break;
				}
			}
			
			// If no specific selector worked, try to find any clickable card-like elements
			if (allElements.length === 0) {
				allElements = Array.from(document.querySelectorAll('div[class*="card"], div[class*="item"], a[href*="startup"], a[href*="company"]'));
			}
			
			return JSON.stringify(allElements.map(el => el.outerHTML));
		}
		"""
		
		try:
			html_items_json = await page.evaluate(extract_script)
		except Exception as e:
			print(f"   ⚠️ Erreur lors de l'extraction: {e}")
			no_new_startups_count += 1
			if no_new_startups_count >= 5:
				break
			await asyncio.sleep(1)
			continue
		
		if not html_items_json or html_items_json == '[]':
			print(f"   ⚠️ Aucun élément HTML extrait à l'itération {iteration}")
			no_new_startups_count += 1
			if no_new_startups_count >= 5:
				break
			await asyncio.sleep(1)
			continue
		
		# Parse HTML items
		try:
			html_items = json.loads(html_items_json)
			print(f"   📦 {len(html_items)} items HTML extraits à l'itération {iteration}")
		except json.JSONDecodeError as e:
			print(f"   ⚠️ Erreur de parsing JSON: {e}")
			no_new_startups_count += 1
			if no_new_startups_count >= 5:
				break
			await asyncio.sleep(1)
			continue
		
		current_batch = _parse_html_sections(html_items, source_url)
		
		if not current_batch or not current_batch.startups:
			print(f"   ⚠️ Aucune startup parsée à l'itération {iteration}")
			no_new_startups_count += 1
			if no_new_startups_count >= 5:
				break
			await asyncio.sleep(1)
			continue
		
		# Count new startups using composite key (name + URL)
		new_startups_count = 0
		for startup in current_batch.startups:
			startup_key = (startup.name or '', startup.startup_url or '')
			if startup.name and startup_key not in seen_startups:
				all_startups.append(startup)
				seen_startups.add(startup_key)
				new_startups_count += 1
		
		print(f"   📊 Itération {iteration}: {len(current_batch.startups)} startups trouvées, {new_startups_count} nouvelles (total: {len(all_startups)})")
		
		# Check if we've reached max_startups
		if max_startups < 50000 and len(all_startups) >= max_startups:
			print(f"   ✅ Limite de {max_startups} startups atteinte")
			break
		
		# If no new startups, increment counter
		if new_startups_count == 0:
			no_new_startups_count += 1
			if no_new_startups_count >= 5:
				print("   ✅ Plus de startups à extraire (5 itérations sans nouvelles startups)")
				break
		else:
			no_new_startups_count = 0
		
		# Scroll down to load more
		try:
			viewport_height = await page.evaluate("() => window.innerHeight || document.documentElement.clientHeight")
			viewport_height = int(viewport_height) if viewport_height else 1000
			await page.evaluate(f"() => window.scrollBy(0, {viewport_height})")
			await asyncio.sleep(2)  # Wait for content to load
		except Exception:
			break
	
	return all_startups


def build_task(task_input: ZoneSecureStartupsInput) -> str:
	"""Create the natural-language instructions fed to the agent."""
	
	extract_all = task_input.max_startups >= 10000
	startups_url = str(task_input.startups_url)
	
	return dedent(
		f"""
		Tu es un agent spécialisé dans l'extraction EXHAUSTIVE de startups depuis Zone Secure.
		Ton objectif est d'identifier TOUTES les pages contenant des startups, naviguer vers chacune, et extraire TOUTES les startups de chaque page.
		
		═══════════════════════════════════════════════════════════════════════════════
		ÉTAPE 1: EXPLORATION ET IDENTIFICATION DES PAGES DE STARTUPS
		═══════════════════════════════════════════════════════════════════════════════
		
		1.1. NAVIGATION INITIALE:
		   - Navigue vers l'URL: {startups_url}
		   - Utilise l'action `navigate` avec l'URL complète
		   - Attends que la page se charge complètement (utilise `wait` avec `seconds: 3`)
		
		1.2. ACCEPTATION DES COOKIES:
		   - Si une bannière de cookies apparaît, trouve et clique sur le bouton "Accepter" ou "Accept All"
		   - Utilise `click` pour accepter les cookies
		   - Attends 2 secondes après l'acceptation
		
		1.3. ⚠️ ÉTAPE CRITIQUE - IDENTIFICATION ET CLIC SUR LE BOUTON "STARTUP":
		   Cette étape est OBLIGATOIRE et doit être faite en PREMIER avant toute autre action.
		   
		   a. EXAMEN DES ÉLÉMENTS EN HAUT DE LA PAGE:
		      - Regarde attentivement la partie supérieure de la page (header, barre de navigation, menu)
		      - Utilise `extract` avec cette requête PRÉCISE:
		        "Look at the TOP of the page (header, navigation bar, menu). Find all clickable elements (buttons, links, tabs) that are visible in the top section. List each element with its text/label. Specifically look for any element that contains the word 'startup', 'start-up', 'Start-up', 'Startup', or similar variations in its text or label."
		      - Le LLM va analyser les éléments de navigation en haut de la page
		   
		   b. IDENTIFICATION DU BOUTON/LIEN "STARTUP":
		      - Parmi les éléments trouvés, identifie celui qui contient explicitement "startup" ou "start-up"
		      - Il peut s'agir d'un bouton, d'un lien, d'un onglet, ou d'un élément de menu
		      - Les variations possibles: "Start-up", "Startup", "startup", "start-up", "START-UP", etc.
		      - Si plusieurs éléments contiennent "startup", choisis celui qui semble être le lien principal vers la section des startups
		   
		   c. CLIC SUR LE BOUTON "STARTUP":
		      - Une fois le bon élément identifié, utilise l'action `click` pour cliquer dessus
		      - Utilise l'index de l'élément ou son texte pour le sélectionner
		      - Exemple: Si tu vois un bouton avec le texte "Start-up" à l'index 5, utilise `click` avec `index: 5`
		      - OU utilise `find_text` pour trouver le texte "Start-up" puis `click` sur l'élément trouvé
		   
		   d. ATTENTE DU CHARGEMENT:
		      - Après avoir cliqué, attends que la nouvelle page/section se charge
		      - Utilise `wait` avec `seconds: 3` pour laisser le temps au contenu de se charger
		      - Vérifie que tu es maintenant sur la page/section des startups
		   
		   e. VÉRIFICATION:
		      - Utilise `extract` pour vérifier que tu es bien sur la page des startups:
		        "Confirm that I am now on the startups page/section. Are there startup listings visible on this page? If yes, describe what you see. If no, I may need to click on the startup button again or look for it in a different location."
		      - Si tu n'es pas sur la bonne page, réessaie de trouver et cliquer sur le bouton "startup"
		      - Si le bouton n'est pas visible en haut, cherche-le dans d'autres parties de la page (menu latéral, footer, etc.)
		
		1.4. ANALYSE DE LA STRUCTURE DE LA PAGE DES STARTUPS:
		   - Maintenant que tu es sur la page des startups, examine attentivement sa structure
		   - Utilise `extract` avec cette requête: "Analyze this startups page structure. Identify: (1) Are there links to other pages with startups? (2) Is there pagination (page numbers, next/previous buttons)? (3) How are the startups displayed (list, grid, cards)? (4) Are startups listed directly on this page? List all clickable links that might lead to more startup pages."
		   - Note tous les liens, boutons de pagination, et sections qui pourraient contenir plus de startups
		
		1.5. IDENTIFICATION DES PAGES DE STARTUPS:
		   - Maintenant que tu es sur la page des startups, cherche les liens vers d'autres pages de startups
		   - Les indices typiques:
		     • Liens de pagination (page 1, page 2, etc.) si l'URL contient "#page="
		     • Boutons "Suivant", "Next", ou numéros de page
		     • Liens vers d'autres sections de startups
		   - Utilise `extract` avec: "Find all links and buttons on this startups page that lead to other pages containing startups. Include pagination links (page numbers, next/previous), and any navigation elements that might show more startup listings."
		   - Liste toutes les URLs uniques que tu dois visiter
		
		═══════════════════════════════════════════════════════════════════════════════
		ÉTAPE 2: NAVIGATION SYSTÉMATIQUE VERS CHAQUE PAGE
		═══════════════════════════════════════════════════════════════════════════════
		
		2.1. STRATÉGIE DE NAVIGATION:
		   - Si tu trouves plusieurs pages (pagination), visite-les TOUTES dans l'ordre
		   - Si tu trouves des sections différentes, visite chaque section
		   - Si la page actuelle contient déjà des startups, commence par extraire celles-ci
		
		2.2. POUR CHAQUE PAGE À VISITER:
		   a. Utilise `navigate` pour aller à l'URL de la page
		   b. Attends le chargement (utilise `wait` avec `seconds: 3`)
		   c. Scrolle jusqu'en bas de la page pour charger tout le contenu
		     - Utilise `scroll` avec `down: true` et `pages: 2` plusieurs fois
		     - Continue jusqu'à ce que tu sois vraiment en bas (vérifie avec plusieurs scrolls)
		   d. Passe à l'étape 3 pour extraire les startups de cette page
		   e. Une fois l'extraction terminée, passe à la page suivante
		
		2.3. GESTION DE LA PAGINATION:
		   - Si l'URL contient "#page=1", essaie "#page=2", "#page=3", etc.
		   - Si tu vois des boutons "Suivant" ou "Next", clique dessus
		   - Continue jusqu'à ce qu'il n'y ait plus de nouvelles pages
		   - Pour chaque page, extrais toutes les startups avant de passer à la suivante
		
		═══════════════════════════════════════════════════════════════════════════════
		ÉTAPE 3: EXTRACTION DES STARTUPS SUR CHAQUE PAGE
		═══════════════════════════════════════════════════════════════════════════════
		
		3.1. IDENTIFICATION DES VRAIES STARTUPS:
		   ⚠️ CRITIQUE - Une STARTUP est une entreprise/compagnie avec:
		   • Un nom d'entreprise (ex: "TechCorp", "InnovateLab", "DataFlow")
		   • Généralement une description de ce qu'elle fait
		   • Potentiellement: secteur, localisation, site web, logo, etc.
		
		   ⚠️ NE PAS extraire:
		   • "Forum", "Remerciements", "Plan", "Sommaire"
		   • "Rechercher", "Partager", "Télécharger", "Plein écran"
		   • "Onglets", "Retour au document", "Toutes les pages"
		   • Titres de sections: "Conseil Audit", "Construction & Transport", "Energie Environnement", etc.
		   • Boutons d'interface utilisateur
		
		3.2. EXTRACTION AVEC LE LLM:
		   - Utilise l'action `extract` avec cette requête PRÉCISE:
		     "Extract ONLY real startups/companies from this page. A startup must have a company name and typically includes additional information like description, sector, location, website, or other business details. EXCLUDE navigation elements like 'Forum', 'Remerciements', 'Plan', 'Sommaire', 'Rechercher', 'Partager', 'Télécharger', 'Plein écran', 'Onglets', 'Retour au document', 'Toutes les pages'. EXCLUDE section titles like 'Conseil Audit', 'Construction & Transport', 'Energie Environnement', 'Finance Banque Assurance', 'Formation', 'IT Digital', 'Public', 'Production Supply Chain', 'Santé Biotech', 'Start-up'. For each startup found, extract: name (required), startup_url (if available), description, website, sector, location, founded_year, employees, tags, logo_url."
		   - L'action `extract` utilise le LLM pour analyser intelligemment la page
		   - Parse le résultat markdown pour extraire les informations de chaque startup
		
		3.3. EXTRACTION ITÉRATIVE (si la page est longue):
		   - Si la page est très longue avec beaucoup de contenu:
		   a. Scrolle jusqu'en haut de la page
		   b. Utilise `extract` pour extraire les startups visibles actuellement
		   c. Scrolle vers le bas d'une page d'écran (utilise `scroll` avec `pages: 1`)
		   d. Répète l'extraction pour les startups maintenant visibles
		   e. Continue jusqu'à avoir extrait toutes les startups de la page
		   f. Assure-toi de ne pas compter deux fois la même startup (utilise le nom comme identifiant unique)
		
		3.4. INFORMATIONS À CAPTURER POUR CHAQUE STARTUP:
		   • `name`: nom exact de la startup (OBLIGATOIRE)
		   • `startup_url`: URL complète vers la page de la startup si disponible
		   • `description`: description/tagline de la startup si disponible
		   • `website`: URL du site web de la startup si disponible
		   • `sector`: secteur/industrie si disponible
		   • `location`: localisation si disponible
		   • `founded_year`: année de fondation si disponible
		   • `employees`: nombre d'employés ou fourchette si disponible
		   • `tags`: tags/catégories visibles
		   • `logo_url`: URL du logo si disponible
		
		═══════════════════════════════════════════════════════════════════════════════
		ÉTAPE 4: AGGREGATION ET FINALISATION
		═══════════════════════════════════════════════════════════════════════════════
		
		4.1. COLLECTE DE TOUTES LES STARTUPS:
		   - Garde une liste de toutes les startups extraites de toutes les pages visitées
		   - Évite les doublons (même nom = même startup)
		   - Si tu trouves la même startup sur plusieurs pages, garde-la une seule fois avec les informations les plus complètes
		
		4.2. VÉRIFICATION FINALE:
		   - Assure-toi d'avoir visité TOUTES les pages contenant des startups
		   - Vérifie qu'il n'y a pas d'autres liens ou sections que tu n'as pas encore explorés
		   - Si tu n'es pas sûr, utilise `extract` pour chercher: "Are there any other links, buttons, or navigation elements that might lead to more startup pages that I haven't visited yet?"
		
		4.3. CONSTRUCTION DU RAPPORT FINAL:
		   - Construis un objet `ZoneSecureStartupsReport` avec:
		     • `source_url`: "{startups_url}" (chaîne de caractères, pas un objet URL)
		     • `startups`: tableau de TOUTES les startups extraites de TOUTES les pages visitées
		   - Trie les startups par ordre d'apparition si possible
		   - Limite à {task_input.max_startups} startups si nécessaire (mais essaie d'extraire toutes)
		
		4.4. FINALISATION:
		   - Utilise l'action `done` avec le champ `data` contenant l'objet `ZoneSecureStartupsReport` complet
		   - Format: {{"done": {{"success": true, "data": {{"source_url": "{startups_url}", "startups": [{{"name": "...", ...}}, ...]}}}}}}
		
		═══════════════════════════════════════════════════════════════════════════════
		RÈGLES CRITIQUES ET BONNES PRATIQUES
		═══════════════════════════════════════════════════════════════════════════════
		
		✅ À FAIRE:
		- Visiter TOUTES les pages contenant des startups
		- Extraire TOUTES les startups de chaque page
		- Utiliser `extract` avec le LLM pour identifier intelligemment les vraies startups
		- Scroller jusqu'en bas de chaque page pour charger tout le contenu
		- Être exhaustif et ne rien manquer
		- Filtrer les éléments de navigation et les titres de sections
		- Garder une trace de toutes les pages visitées pour éviter les doublons
		
		❌ À NE PAS FAIRE:
		- Ne pas visiter les pages individuelles des startups (reste sur les pages de listing)
		- Ne pas extraire les éléments de navigation
		- Ne pas compter deux fois la même startup
		- Ne pas arrêter l'extraction avant d'avoir visité toutes les pages
		- Ne pas inventer des informations manquantes (laisse null si absent)
		
		🔧 OUTILS DISPONIBLES:
		- `navigate`: pour aller à une URL
		- `extract`: pour utiliser le LLM et extraire les startups intelligemment
		- `scroll`: pour scroller la page
		- `wait`: pour attendre le chargement
		- `click`: pour cliquer sur des liens/boutons
		- `done`: pour finaliser avec le rapport complet
		
		📝 NOTES IMPORTANTES:
		- Utilise la vision pour mieux comprendre la structure de la page
		- Sois patient avec le chargement des pages
		- Si une page ne charge pas, attends plus longtemps ou réessaie
		- Si tu n'es pas sûr qu'un élément est une startup, utilise `extract` pour l'analyser
		- Le but est d'être EXHAUSTIF: extraire TOUTES les startups sans en manquer une seule
		"""
	).strip()


async def run_zone_secure_startups(task_input: ZoneSecureStartupsInput) -> ZoneSecureStartupsReport | None:
	"""Execute the agent and return the structured list of startups."""

	print("🔧 Configuration du LLM...")
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
		page_extraction_llm = ChatBrowserUse()
		print("✅ Utilisation de ChatBrowserUse")
	else:
		model_name = os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-preview-09-2025')
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
		extraction_model = os.getenv('PAGE_EXTRACTION_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		page_extraction_llm = ChatOpenAI(
			model=extraction_model,
			timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=20.0),
			max_retries=2,
			max_completion_tokens=90960,
			add_schema_to_system_prompt=True,
			dont_force_structured_output=True,
		)
		print(f"✅ Utilisation de ChatOpenAI avec le modèle: {model_name}")

	print("🌐 Création du navigateur...")
	browser = Browser(headless=False, keep_alive=True)
	await browser.start()
	
	try:
		# Navigate to the startups URL
		startups_url = str(task_input.startups_url)
		print(f"📍 Navigation vers: {startups_url}")
		navigate_event = NavigateToUrlEvent(url=startups_url, new_tab=True)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event
		
		# Get the current page after navigation
		page = await browser.get_current_page()
		if not page:
			page = await browser.new_page(startups_url)
		
		await asyncio.sleep(5)  # Wait for initial page load
		
		# Use LLM-based agent for intelligent extraction and multi-page navigation
		# The agent will first click on the "startup" button, then navigate and extract
		print("🤖 Utilisation de l'agent LLM pour identifier et naviguer vers toutes les pages de startups...")
		agent = Agent(
			task=build_task(task_input),
			llm=llm,
			page_extraction_llm=page_extraction_llm,
			browser=browser,
			output_model_schema=ZoneSecureStartupsReport,
			use_vision='auto',
			vision_detail_level='auto',
			step_timeout=300,
			llm_timeout=180,
			max_failures=5,
			max_history_items=20,  # Keep more history for multi-page navigation
			max_steps=500,  # Allow many steps for exhaustive multi-page extraction
			directly_open_url=False,
		)
		print("✅ Agent créé")
		
		print("▶️  Démarrage de l'exécution de l'agent...")
		history = await agent.run()
		print("✅ Exécution terminée")
		print_llm_usage_summary(history)
		
		# Try to get structured output
		if history.structured_output:
			report = history.structured_output  # type: ignore[arg-type]
			# Filter out navigation elements
			filtered_startups = _filter_navigation_elements(report.startups)
			report.startups = filtered_startups
			return report
		
		# Try to extract from final result
		final_result = history.final_result()
		if final_result:
			try:
				report = ZoneSecureStartupsReport.model_validate_json(final_result)
				# Filter out navigation elements
				filtered_startups = _filter_navigation_elements(report.startups)
				report.startups = filtered_startups
				return report
			except ValidationError:
				match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if match:
					try:
						report = ZoneSecureStartupsReport.model_validate_json(match.group(1))
						# Filter out navigation elements
						filtered_startups = _filter_navigation_elements(report.startups)
						report.startups = filtered_startups
						return report
					except ValidationError:
						pass
		
		return _fallback_report(str(task_input.startups_url), "Extraction échouée: aucune startup trouvée.")
	
	finally:
		# Close browser
		print("🧹 Fermeture du navigateur...")
		try:
			await browser.kill()
			print("✅ Navigateur fermé")
		except Exception as e:
			print(f"⚠️  Erreur lors de la fermeture du navigateur: {e}")


def parse_arguments() -> ZoneSecureStartupsInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Extrait toutes les startups depuis Zone Secure')
	parser.add_argument(
		'--url',
		default='https://fr.zone-secure.net/20412/2540033/#page=1',
		help='URL de la page des startups Zone Secure',
	)
	parser.add_argument(
		'--max-startups',
		type=int,
		default=10000,
		help='Nombre maximal de startups à extraire (par défaut: 10000 pour extraire toutes)',
	)
	parser.add_argument(
		'--output',
		default='zone_secure_startups.json',
		help='Chemin du fichier JSON résultat (par défaut: ./zone_secure_startups.json)',
	)
	args = parser.parse_args()
	
	return ZoneSecureStartupsInput(
		url=args.url,
		max_startups=args.max_startups,
		output_path=Path(args.output),
	)


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		startups_url = str(task_input.startups_url)
		print(f"🚀 Démarrage de l'agent pour Zone Secure")
		print(f"📍 URL des startups: {startups_url}")
		print(f"📊 Nombre max de startups: {task_input.max_startups}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")

		result = await run_zone_secure_startups(task_input)

		if result is None:
			print("❌ L'agent n'a retourné aucune donnée structurée.")
			return

		output_json = result.model_dump_json(indent=2, ensure_ascii=False)
		output_path = task_input.output_path
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(output_json, encoding='utf-8')

		print(f'\n✅ {len(result.startups)} startups extraites et sauvegardées dans: {output_path.resolve()}')
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

