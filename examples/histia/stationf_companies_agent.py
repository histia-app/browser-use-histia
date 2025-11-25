"""
Agent designed to extract companies from Station F HAL companies page.

Usage:
    # Without authentication (if page is accessible)
    python examples/histia/stationf_companies_agent.py

    # With authentication via command line
    python examples/histia/stationf_companies_agent.py --email votre@email.com --password votre_mot_de_passe

    # With authentication via .env file (recommended)
    # Add to your .env file:
    # STATIONF_EMAIL=votre@email.com
    # STATIONF_PASSWORD=votre_mot_de_passe
    # Then run:
    python examples/histia/stationf_companies_agent.py

Environment Variables:
    STATIONF_EMAIL: Email for Station F authentication (optional)
    STATIONF_PASSWORD: Password for Station F authentication (optional)
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer
from bs4 import BeautifulSoup

# Load environment variables immediately so the agent can access API keys.
load_dotenv()

# Configure timeouts BEFORE importing browser_use to ensure they're applied
# Use longer timeouts for heavy pages with lots of content
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')  # Increase scroll timeout for slow-loading pages

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI, Tools
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class StationFCompaniesInput(BaseModel):
	"""User-provided parameters for the Station F companies extraction task."""

	url: str = Field(
		default='https://hal2.stationf.co/companies',
		description='URL of the Station F companies page',
	)
	max_companies: int = Field(
		1000,
		ge=1,
		le=10000,
		description='Maximum number of companies to capture (use a high number like 1000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('stationf_companies.json'),
		description='Destination for the JSON list of companies',
	)
	email: str | None = Field(
		default=None,
		description='Email for authentication (optional, will try to login if provided)',
	)
	password: str | None = Field(
		default=None,
		description='Password for authentication (optional, will try to login if provided)',
	)

	@property
	def companies_url(self) -> AnyHttpUrl:
		"""Build the Station F companies URL."""
		return AnyHttpUrl(self.url)


class StationFCompany(BaseModel):
	"""Structured information for each Station F company entry."""

	name: str = Field(..., description='Company name exactly as written on the page')
	stationf_url: str | None = Field(
		None,
		description='Complete URL to the Station F company page (format: https://hal2.stationf.co/companies/...)',
	)
	description: str | None = Field(
		None,
		description='Company description/tagline if available',
	)
	website: str | None = Field(
		None,
		description='Company website URL if available',
	)
	sector: str | None = Field(
		None,
		description='Company sector/industry if available',
	)
	stage: str | None = Field(
		None,
		description='Company stage (e.g., Seed, Series A) if available',
	)
	location: str | None = Field(
		None,
		description='Company location if available',
	)
	founded_year: int | None = Field(
		None,
		description='Year the company was founded if available',
	)
	employees: str | None = Field(
		None,
		description='Number of employees or employee range if available',
	)
	tags: list[str] = Field(
		default_factory=list,
		description='Company tags/categories visible on the card',
	)
	logo_url: str | None = Field(
		None,
		description='URL to company logo if available',
	)


class StationFCompaniesReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='Companies page URL that was analysed')
	companies: list[StationFCompany] = Field(
		...,
		min_length=1,
		description='Company entries ordered as they appear on the page',
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


def _fallback_report(source_url: str, reason: str) -> StationFCompaniesReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	return StationFCompaniesReport(
		source_url=AnyHttpUrl(source_url),
		companies=[
			StationFCompany(
				name='Informations indisponibles',
				stationf_url=source_url,
				description=None,
				website=None,
				sector=None,
				stage=None,
				location=None,
				founded_year=None,
				employees=None,
				tags=[],
				logo_url=None,
			)
		],
	)


def _normalize_stationf_url(url: str | None, base_url: str) -> str | None:
	"""Convert relative URLs to absolute Station F URLs."""
	if not url:
		return None

	url = url.strip()
	if not url:
		return None

	# If it's already an absolute URL, return as is
	if url.startswith(('http://', 'https://')):
		# Ensure it's a Station F URL
		if 'stationf.co' in url.lower():
			return url
		return None

	# If it starts with /, make it relative to the base domain
	if url.startswith('/'):
		parsed_base = urlparse(base_url)
		return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"

	# Otherwise, try to resolve relative to base URL
	try:
		resolved = urljoin(base_url, url)
		if 'stationf.co' in resolved.lower():
			return resolved
		return None
	except Exception:
		return None


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> StationFCompaniesReport | None:
	"""Parse HTML sections from evaluate action to build StationFCompaniesReport."""
	
	companies: list[StationFCompany] = []
	
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
			
			# Try multiple selectors for company cards
			# Priority: data-slot="drawer-trigger" (actual structure), then fallback to common patterns
			company_element = (
				soup.find(attrs={'data-slot': 'drawer-trigger'}) or
				soup.find('article') or
				soup.find('div', class_=re.compile(r'company|card|item', re.I)) or
				soup.find('a', href=re.compile(r'/companies/', re.I)) or
				soup
			)
			
			# Extract name - try data-slot="item-title" first (actual structure), then fallback
			name = None
			name_selectors = [
				'[data-slot="item-title"] h5',  # Actual structure from Station F
				'[data-slot="item-title"]',     # Fallback if no h5
				'h1', 'h2', 'h3', 'h4', 'h5',
				'a[href*="/companies/"]',
				'[class*="name"]',
				'[class*="title"]',
			]
			for selector in name_selectors:
				name_elem = company_element.select_one(selector)
				if name_elem:
					name = name_elem.get_text(strip=True)
					if name:
						break
			
			# Extract Station F URL - look for links in the drawer-trigger or its children
			stationf_url = None
			# First try to find a link in the drawer-trigger itself or its children
			link = (
				company_element.find('a', href=re.compile(r'/companies/', re.I)) or
				company_element.select_one('a[href*="/companies/"]') or
				# Also check if the drawer-trigger itself is a link
				(company_element if company_element.name == 'a' and company_element.get('href') and '/companies/' in str(company_element.get('href', '')) else None)
			)
			if link:
				href = link.get('href', '') if hasattr(link, 'get') else (link if isinstance(link, str) else None)
				if href:
					href_str = str(href).strip()
					if href_str:
						stationf_url = _normalize_stationf_url(href_str, source_url)
			
			# Extract description - try data-slot="item-description" first (actual structure)
			description = None
			desc_selectors = [
				'[data-slot="item-description"]',  # Actual structure from Station F
				'[class*="description"]',
				'[class*="tagline"]',
				'[class*="summary"]',
				'p',
			]
			for selector in desc_selectors:
				desc_elem = company_element.select_one(selector)
				if desc_elem:
					description = desc_elem.get_text(strip=True)
					if description and len(description) > 3:  # Reduced threshold for short descriptions like "Landing"
						break
			
			# Extract website
			website = None
			website_link = company_element.select_one('a[href^="http"]:not([href*="stationf.co"])')
			if website_link:
				href = website_link.get('href', '')
				if href:
					website = str(href).strip()
			
			# Extract sector/stage/location from various patterns
			sector = None
			stage = None
			location = None
			founded_year = None
			employees = None
			
			# Look for metadata in spans, divs, or list items
			metadata_elements = company_element.select('span, div, li')
			for elem in metadata_elements:
				text = elem.get_text(strip=True)
				if not text:
					continue
				
				# Sector detection
				if not sector and any(keyword in text.lower() for keyword in ['sector', 'industry', 'category']):
					sector = text.split(':', 1)[-1].strip() if ':' in text else text
				
				# Stage detection
				if not stage and any(keyword in text.lower() for keyword in ['stage', 'round', 'seed', 'series']):
					stage = text
				
				# Location detection
				if not location and any(keyword in text.lower() for keyword in ['location', 'city', 'country', 'paris', 'france']):
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
				if not employees and any(keyword in text.lower() for keyword in ['employees', 'team', 'people']):
					employees = text
			
			# Extract tags
			tags = []
			tag_elements = company_element.select('[class*="tag"], [class*="badge"], [class*="category"]')
			for tag_elem in tag_elements:
				tag_text = tag_elem.get_text(strip=True)
				if tag_text:
					tags.append(tag_text)
			
			# Extract logo - check data-slot="item-media" first (actual structure)
			logo_url = None
			item_media = company_element.select_one('[data-slot="item-media"]')
			if item_media:
				logo_img = item_media.select_one('img[src], img[data-src]')
				if logo_img:
					logo_src = logo_img.get('src') or logo_img.get('data-src')
					if logo_src:
						logo_src_str = str(logo_src).strip()
						if logo_src_str:
							logo_url = _normalize_stationf_url(logo_src_str, source_url)
			# Fallback to general img search
			if not logo_url:
				logo_img = company_element.select_one('img[src], img[data-src]')
				if logo_img:
					logo_src = logo_img.get('src') or logo_img.get('data-src')
					if logo_src:
						logo_src_str = str(logo_src).strip()
						if logo_src_str:
							logo_url = _normalize_stationf_url(logo_src_str, source_url)
			
			# Only add company if we have at least a name
			# Only include fields that have actual values (not None/empty)
			if name:
				# Build company with only fields that have values
				# Use model_validate to ensure proper type handling
				company_dict: dict[str, Any] = {'name': name}
				
				# Only add fields that have values
				if stationf_url:
					company_dict['stationf_url'] = stationf_url
				if description:
					company_dict['description'] = description
				if website:
					company_dict['website'] = website
				if sector:
					company_dict['sector'] = sector
				if stage:
					company_dict['stage'] = stage
				if location:
					company_dict['location'] = location
				if founded_year is not None:  # founded_year is int | None
					company_dict['founded_year'] = founded_year
				if employees:
					company_dict['employees'] = employees
				if tags:  # tags is list[str]
					company_dict['tags'] = tags
				if logo_url:
					company_dict['logo_url'] = logo_url
				
				companies.append(StationFCompany.model_validate(company_dict))
		except Exception as e:
			# Skip malformed sections
			continue
	
	if companies:
		return StationFCompaniesReport(
			source_url=AnyHttpUrl(source_url),
			companies=companies,
		)
	return None


def _parse_extracted_markdown(content: str, source_url: str) -> StationFCompaniesReport | None:
	"""Parse markdown content from extract action to build StationFCompaniesReport."""

	companies: list[StationFCompany] = []
	current_company: dict[str, Any] | None = None

	# First, try to parse markdown tables
	table_pattern = re.compile(
		r'\|\s*Name\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*\n'
		r'\|[-\s|:]+\|\s*\n'
		r'((?:\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*\n?)+)',
		re.IGNORECASE | re.MULTILINE
	)

	table_match = table_pattern.search(content)
	if table_match:
		table_rows = table_match.group(1).strip().split('\n')
		for row in table_rows:
			if not row.strip() or not row.strip().startswith('|'):
				continue
			parts = [p.strip() for p in row.split('|')]
			if len(parts) >= 3:
				try:
					name = parts[1].strip() if len(parts) > 1 else ''
					description = parts[2].strip() if len(parts) > 2 else ''
					website = parts[3].strip() if len(parts) > 3 else ''

					# Skip header rows
					if name.lower() in ['name', 'company', 'title']:
						continue

					if name:
						companies.append(StationFCompany(
							name=name,
							stationf_url=None,
							description=description if description and description.lower() not in ['null', 'n/a', ''] else None,
							website=website if website and website.lower() not in ['null', 'n/a', ''] else None,
							sector=None,
							stage=None,
							location=None,
							founded_year=None,
							employees=None,
							tags=[],
							logo_url=None,
						))
				except (IndexError, ValueError):
					continue

		if companies:
			return StationFCompaniesReport(
				source_url=AnyHttpUrl(source_url),
				companies=companies,
			)

	# Fallback to line-by-line parsing
	lines = content.split('\n')

	for i, line in enumerate(lines):
		line_stripped = line.strip()

		# Skip empty lines
		if not line_stripped:
			continue

		# Detect new company entry
		name_match = re.match(r'^\d+\.\s*\*\*Name\*\*:\s*(.+)$', line_stripped)
		if name_match:
			# Save previous company if exists
			if current_company and current_company.get('name'):
				try:
					companies.append(StationFCompany.model_validate(current_company))
				except ValidationError:
					pass

			# Start new company
			name = name_match.group(1).strip()
			current_company = {
				'name': name,
				'stationf_url': None,
				'description': None,
				'website': None,
				'sector': None,
				'stage': None,
				'location': None,
				'founded_year': None,
				'employees': None,
				'tags': [],
				'logo_url': None,
			}
			continue

		# Detect standalone company name
		if line_stripped.startswith('**') and line_stripped.endswith('**') and len(line_stripped) > 4:
			if 'company' in line_stripped.lower() and ('list' in line_stripped.lower() or 'page' in line_stripped.lower()):
				continue

			if current_company and current_company.get('name'):
				try:
					companies.append(StationFCompany.model_validate(current_company))
				except ValidationError:
					pass

			name = line_stripped.strip('*').strip()
			current_company = {
				'name': name,
				'stationf_url': None,
				'description': None,
				'website': None,
				'sector': None,
				'stage': None,
				'location': None,
				'founded_year': None,
				'employees': None,
				'tags': [],
				'logo_url': None,
			}
			continue

		# Parse fields if we have a current company
		if current_company and current_company.get('name'):
			if '**Name:**' in line_stripped or '**Company:**' in line_stripped:
				name_match = re.search(r'\*\*(?:Name|Company):\*\*\s*(.+)', line_stripped)
				if name_match:
					current_company['name'] = name_match.group(1).strip()
				continue

			if '**Station F URL:**' in line_stripped or '**URL:**' in line_stripped:
				url_match = re.search(r'https?://[^\s*]+|/companies/[^\s*]+', line_stripped)
				if url_match:
					url = url_match.group(0)
					current_company['stationf_url'] = _normalize_stationf_url(url, source_url)
				continue

			if '**Description:**' in line_stripped:
				desc_match = re.search(r'\*\*Description:\*\*\s*(.+)', line_stripped)
				if desc_match:
					desc = desc_match.group(1).strip()
					if desc.lower() not in ['null', '(no description provided)', '(information not available)']:
						current_company['description'] = desc
				continue

			if '**Website:**' in line_stripped:
				website_match = re.search(r'https?://[^\s*]+', line_stripped)
				if website_match:
					current_company['website'] = website_match.group(0)
				continue

			if '**Sector:**' in line_stripped or '**Industry:**' in line_stripped:
				sector_match = re.search(r'\*\*(?:Sector|Industry):\*\*\s*(.+)', line_stripped)
				if sector_match:
					sector = sector_match.group(1).strip()
					if sector.lower() not in ['null', 'n/a', '']:
						current_company['sector'] = sector
				continue

			if '**Stage:**' in line_stripped:
				stage_match = re.search(r'\*\*Stage:\*\*\s*(.+)', line_stripped)
				if stage_match:
					current_company['stage'] = stage_match.group(1).strip()
				continue

			if '**Location:**' in line_stripped:
				location_match = re.search(r'\*\*Location:\*\*\s*(.+)', line_stripped)
				if location_match:
					current_company['location'] = location_match.group(1).strip()
				continue

			if '**Founded:**' in line_stripped or '**Year:**' in line_stripped:
				year_match = re.search(r'\b(19|20)\d{2}\b', line_stripped)
				if year_match:
					try:
						current_company['founded_year'] = int(year_match.group(0))
					except ValueError:
						pass
				continue

			if '**Employees:**' in line_stripped:
				employees_match = re.search(r'\*\*Employees:\*\*\s*(.+)', line_stripped)
				if employees_match:
					current_company['employees'] = employees_match.group(1).strip()
				continue

			if '**Tags:**' in line_stripped:
				tags_match = re.search(r'\*\*Tags:\*\*\s*(.+)', line_stripped)
				if tags_match:
					tags_str = tags_match.group(1).strip()
					if tags_str.lower() not in ['null', 'n/a', '']:
						tag_list = [t.strip() for t in re.split(r'[,;]', tags_str) if t.strip()]
						current_company['tags'] = tag_list
				continue

	# Don't forget the last company
	if current_company and current_company.get('name'):
		try:
			companies.append(StationFCompany.model_validate(current_company))
		except ValidationError:
			pass

	if companies:
		return StationFCompaniesReport(
			source_url=AnyHttpUrl(source_url),
			companies=companies,
		)
	return None


def _sanitize_report(report: StationFCompaniesReport) -> StationFCompaniesReport:
	"""Apply basic clean-up rules on top of the structured output."""

	base_url = str(report.source_url)
	for company in report.companies:
		# Normalize Station F URL
		if company.stationf_url:
			stationf_url_str = str(company.stationf_url)
			if not stationf_url_str.startswith(('http://', 'https://')):
				company.stationf_url = _normalize_stationf_url(stationf_url_str, base_url)
			elif 'stationf.co' not in stationf_url_str.lower():
				company.stationf_url = None

		# Clean up tags
		if company.tags:
			company.tags = [tag.strip() for tag in company.tags if tag.strip()]

	return report


def build_task(task_input: StationFCompaniesInput) -> str:
	"""Create the natural-language instructions fed to the agent, specialized for Station F companies."""

	extract_all = task_input.max_companies >= 1000
	companies_url = str(task_input.companies_url)

	return dedent(
		f"""
		Tu es un agent spécialisé dans l'extraction d'entreprises depuis la page Station F HAL.

		Objectif CRITIQUE:
		- Navigue directement vers l'URL des entreprises: {companies_url}
		- IMPORTANT: Utilise l'action `navigate` pour aller directement à cette URL - NE FAIS PAS de recherche internet
		- {"Identifie et extrait TOUTES les entreprises présentes sur cette page, SANS AUCUNE EXCEPTION." if extract_all else f"Identifie et extrait jusqu'à {task_input.max_companies} entreprises présentes sur cette page."}
		- IMPORTANT: Ne filtre PAS les entreprises. Prends TOUTES les entreprises visibles sur la page.
		- Ne confonds PAS les titres de sections avec des entreprises réelles.
		- Pour chaque entreprise, capture:
		  • `name`: nom exact de l'entreprise tel qu'affiché
		  • `stationf_url`: URL complète vers la page Station F de l'entreprise (format: https://hal2.stationf.co/companies/...)
		  • `description`: description/tagline de l'entreprise si disponible
		  • `website`: URL du site web de l'entreprise si disponible
		  • `sector`: secteur/industrie si disponible
		  • `stage`: stade de développement (ex: Seed, Series A) si disponible
		  • `location`: localisation si disponible
		  • `founded_year`: année de fondation si disponible
		  • `employees`: nombre d'employés ou fourchette si disponible
		  • `tags`: tags/catégories visibles
		  • `logo_url`: URL du logo si disponible

		Processus pour Station F Companies:
		⚠️ IMPORTANT: La navigation et le scroll jusqu'en bas ont déjà été effectués automatiquement.
		Tu es déjà sur la page {companies_url} et TOUT le contenu a été chargé automatiquement.
		Le scroll est COMPLÈTEMENT TERMINÉ - tu n'as PAS besoin de scroller, c'est déjà fait!
		
		Tu dois maintenant UNIQUEMENT:
		1. EXTRACTION DES ENTREPRISES - MÉTHODE OBLIGATOIRE
		   - ⚠️ CRITIQUE: Le scroll est DÉJÀ FAIT - utilise `evaluate` IMMÉDIATEMENT, pas besoin d'attendre
		   - ⚠️ CRITIQUE: Utilise UNIQUEMENT l'action `evaluate` pour extraire le HTML des entreprises
		   - ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs directement
		   - Format exact de l'action `evaluate`: {{"evaluate": {{"code": "TON_CODE_JAVASCRIPT"}}}}
		   - Code JavaScript à utiliser (structure réelle de Station F):
		   (function(){{const elements = Array.from(document.querySelectorAll('[data-slot="drawer-trigger"]'));return JSON.stringify(elements.map(el => el.outerHTML));}})()
		   - Ce code va extraire le HTML complet de tous les éléments représentant des entreprises
		   - Le résultat sera une chaîne JSON contenant un tableau de chaînes HTML
		   - ⚠️ ATTENTION: `evaluate` utilise le paramètre `code` (PAS `query`, PAS `extract_links`)
		   - Une fois que tu as le résultat de `evaluate`, utilise directement `done` avec les données parsées dans le format StationFCompaniesReport

		2. Traitement des résultats de `evaluate`:
		   - Le résultat de `evaluate` sera une chaîne JSON contenant un tableau de chaînes HTML
		   - Parse cette chaîne JSON pour obtenir le tableau de HTML
		   - Pour chaque élément HTML du tableau, extrais les informations suivantes:
		     • `name`: texte du titre ou du lien principal
		     • `stationf_url`: attribut `href` du lien vers l'entreprise (normalise en URL absolue)
		     • `description`: texte de description si disponible
		     • `website`: lien externe si disponible
		     • `sector`, `stage`, `location`, `founded_year`, `employees`: métadonnées si visibles
		     • `tags`: tous les tags/badges visibles
		     • `logo_url`: URL de l'image du logo si disponible
		   - Construis un objet `StationFCompaniesReport` avec:
		     • `source_url`: "{companies_url}" (chaîne de caractères)
		     • `companies`: tableau de toutes les entreprises extraites dans l'ordre
		   - Limite la liste finale à {task_input.max_companies} entreprises maximum si nécessaire
		   - Utilise l'action `done` avec le champ `data` contenant l'objet `StationFCompaniesReport` complet

		Règles importantes:
		- ⚠️ NAVIGATION ET SCROLL: La navigation et le scroll ont déjà été effectués automatiquement - NE PAS naviguer ni scroller!
		- ⚠️ NE PAS utiliser `navigate` - tu es déjà sur la bonne page! (outil désactivé)
		- ⚠️ NE PAS utiliser `scroll` - tout le contenu a déjà été chargé! (outil désactivé)
		- ⚠️ NE PAS utiliser `wait` - pas besoin d'attendre, tout est déjà chargé! (outil désactivé)
		- ⚠️ EXTRACTION OBLIGATOIRE: Utilise UNIQUEMENT `evaluate` avec le paramètre `code` pour extraire le HTML
		- ⚠️ IMPORTANT: Le scroll est DÉJÀ FAIT - utilise `evaluate` IMMÉDIATEMENT, pas besoin d'attendre!
		- Format `evaluate`: {{"evaluate": {{"code": "(function(){{const elements = Array.from(document.querySelectorAll('article, [class*=\"company\"], [class*=\"card\"], a[href*=\"/companies/\"]'));return JSON.stringify(elements.map(el => el.outerHTML));}})()"}}}}
		- ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs directement
		- Le code JavaScript doit sélectionner tous les éléments représentant des entreprises
		- Retourne un tableau JSON de chaînes HTML (outerHTML de chaque élément)
		- Une fois le HTML extrait, parse-le pour construire le rapport StationFCompaniesReport
		- Si une info manque, laisse le champ null, mais ne l'invente pas
		- CRITIQUE: Ne visite JAMAIS les pages individuelles des entreprises - extrais toutes les informations depuis les cartes de la liste
		- CRITIQUE SÉRIALISATION: Lorsque tu appelles `done`, assure-toi que `source_url` est une chaîne de caractères (string), pas un objet URL
		- Utilise la vision et sois patient si le chargement est lent
		"""
	).strip()


async def _scroll_to_bottom_naive(browser, max_scrolls: int = 50) -> None:
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
	
	# Wait for initial page load (reduced for faster scrolling)
	print("   ⏳ Attente initiale de 2 secondes pour le chargement de la page...")
	await asyncio.sleep(2)
	
	# Check initial content - use the actual structure with data-slot
	try:
		initial_elements = await page.evaluate("() => document.querySelectorAll('[data-slot=\"drawer-trigger\"]').length")
		initial_elements = int(initial_elements) if initial_elements else 0
		print(f"   📊 Éléments initiaux trouvés: {initial_elements}")
	except Exception:
		initial_elements = 0
	
	# Scroll progressivement jusqu'en bas
	scroll_count = 0
	last_scroll_position = -1
	last_elements_count = initial_elements
	no_change_count = 0
	
	for i in range(max_scrolls):
		# Get current scroll position and content count
		try:
			current_position = await page.evaluate("() => window.pageYOffset || document.documentElement.scrollTop")
			current_position = int(current_position) if current_position else 0
			
			current_elements = await page.evaluate("() => document.querySelectorAll('[data-slot=\"drawer-trigger\"]').length")
			current_elements = int(current_elements) if current_elements else 0
		except Exception:
			current_position = 0
			current_elements = 0
		
		# Check if we're at the bottom (reduced threshold for faster completion)
		if scroll_count >= 5 and current_position == last_scroll_position and current_elements == last_elements_count:
			no_change_count += 1
			if no_change_count >= 2:  # Reduced from 3 to 2 for faster detection
				print(f"   ✅ Arrivé en bas après {scroll_count} scrolls ({current_elements} éléments au total)")
				break
		else:
			no_change_count = 0
			if current_elements > last_elements_count:
				print(f"   📊 {current_elements} éléments trouvés (nouveau contenu chargé)")
			elif scroll_count < 5:
				# Force continue scrolling for first 5 scrolls even if position doesn't change
				# This ensures we give the page time to load content
				no_change_count = 0
		
		last_scroll_position = current_position
		last_elements_count = current_elements
		
		# Scroll down by one page using JavaScript
		try:
			await page.evaluate(f"() => window.scrollBy(0, {viewport_height})")
			await page.evaluate(f"() => document.documentElement.scrollTop += {viewport_height}")
		except Exception as e:
			print(f"   ⚠️ Erreur lors du scroll: {e}")
			try:
				await page.evaluate(f"() => window.scrollTo(0, window.pageYOffset + {viewport_height})")
			except Exception:
				print(f"   ⚠️ Toutes les méthodes de scroll ont échoué")
				break
		
		scroll_count += 1
		
		# Wait for content to load (reduced delay for faster scrolling)
		await asyncio.sleep(1.5)  # Reduced from 4 to 1.5 seconds
		
		if scroll_count % 5 == 0:
			print(f"   📜 {scroll_count} scrolls effectués, {current_elements} éléments trouvés...")
	
	# Final wait to ensure everything is loaded (reduced for faster completion)
	print("   ⏳ Attente finale de 2 secondes pour le chargement complet...")
	await asyncio.sleep(2)
	
	# Final check of content - use the actual structure with data-slot
	try:
		final_elements = await page.evaluate("() => document.querySelectorAll('[data-slot=\"drawer-trigger\"]').length")
		final_elements = int(final_elements) if final_elements else 0
		print(f"   📊 Éléments finaux trouvés: {final_elements}")
	except Exception:
		final_elements = 0
	
	print(f"✅ Scroll terminé: {scroll_count} scrolls effectués au total, {final_elements} éléments trouvés")


async def _accept_cookies_with_llm(browser: Browser, llm) -> bool:
	"""Use LLM-powered agent to intelligently detect and accept cookie banners.
	
	Returns True if cookies were accepted, False otherwise.
	"""
	try:
		print("🤖 Utilisation du LLM pour détecter et accepter les cookies...")
		
		# Ultra-simple task: ONLY click "Accept" button, then STOP immediately
		task = """
		URGENT TASK: Find and click ONLY the "Accept" or "Accept All" button for cookies.
		
		INSTRUCTIONS:
		- Look at the page carefully for any cookie consent banner or popup (usually visible without scrolling)
		- Find the button labeled "Accept All", "Accept", "Accepter tout", "Accepter", "Agree", "OK", or similar
		- DO NOT click "Manage", "Preferences", "Settings", "Reject", "Decline", or any other button
		- DO NOT scroll - the cookie banner is usually visible at the top or bottom of the viewport
		- After clicking "Accept" or "Accept All", immediately signal completion with done action
		- Do nothing else after clicking Accept
		- If no cookie banner is visible in the current viewport, use done() to complete
		"""
		
		# Exclude scroll tool - we don't want the agent to scroll when accepting cookies
		tools = Tools(exclude_actions=['scroll', 'navigate', 'wait'])
		agent = Agent(
			task=task.strip(),
			llm=llm,
			browser=browser,
			tools=tools,
			max_steps=3,  # Very strict: click Accept (then stop immediately)
			flash_mode=True,
			use_thinking=False,
			use_vision='auto',  # Enable vision to see the banner
		)
		
		# Get browser session from agent before it closes
		# This ensures we have a reference to the browser even after agent.run() completes
		browser_session = agent.browser_session
		
		history = await agent.run()
		print_llm_usage_summary(history)
		print_llm_usage_summary(history)
		
		# Check if agent clicked "Accept" button
		clicked_accept = False
		actions = history.action_names()
		if any('click' in action.lower() for action in actions):
			# Check if the click was on an Accept button by looking at results
			action_results = history.action_results()
			for result in action_results:
				if result and result.extracted_content:
					content = str(result.extracted_content).lower()
					if 'accept' in content:
						clicked_accept = True
						break
		
		# Don't close the agent - the browser must stay open for the rest of the code
		# The browser will be closed at the end of run_stationf_companies
		# But we need to ensure the browser session is still valid
		# The browser_session reference should keep it alive
		
		if clicked_accept or history.is_successful():
			print("   ✅ Cookies acceptés via LLM")
			await asyncio.sleep(1.5)  # Brief wait for banner to disappear
			return True
		else:
			print("   ⚠️  LLM n'a peut-être pas cliqué sur Accept")
			return False
	except Exception as exc:
		print(f"⚠️  Échec de la détection LLM des cookies: {exc}")
		return False


async def _accept_cookies_pattern_only(browser: Browser) -> bool:
	"""Detect and accept cookie consent popups using ONLY pattern-based method (no LLM)."""
	print("🔍 Utilisation de la détection basée sur les motifs (sans LLM)...")
	page = await browser.get_current_page()
	if not page:
		return False
	
	script = """
	(() => {
		const keywords = [
			'accept all', 'accept cookies', 'accept', 'agree', 'allow all', 'allow cookies',
			'allow', 'consent', 'continue', 'ok', 'got it', 'i agree', 'i accept',
			"j'accepte", 'accepter', 'tout accepter', 'allow essential', 'accepter tout',
			'tous accepter', 'ok, j\'accepte', 'd\'accord', 'continuer', 'je comprends',
		];

		const normalize = text => (text || '').toLowerCase().replace(/\\s+/g, ' ').trim();

		const isVisible = (el) => {
			if (!el) return false;
			const style = window.getComputedStyle(el);
			return style.display !== 'none' && 
				style.visibility !== 'hidden' && 
				style.opacity !== '0' &&
				el.offsetParent !== null &&
				el.offsetWidth > 0 && 
				el.offsetHeight > 0;
		};

		// Quick check: if no cookie banner is visible, don't do anything
		const hasVisibleCookieBanner = () => {
			const cookieTexts = ['cookie', 'consent', 'privacy', 'gdpr'];
			const allElements = document.querySelectorAll('*');
			for (const el of allElements) {
				const elText = normalize(el.innerText || el.textContent || '');
				if (cookieTexts.some(text => elText.includes(text)) && isVisible(el)) {
					const buttons = el.querySelectorAll('button, [role="button"]');
					for (const btn of buttons) {
						const btnText = normalize(btn.innerText || btn.textContent || '');
						if (btnText.includes('accept') && isVisible(btn)) {
							return true;
						}
					}
				}
			}
			return false;
		};

		if (!hasVisibleCookieBanner()) {
			return false;
		}

		const attemptCookieAccept = (container = document) => {
			// Find banner-like elements
			const bannerSelectors = [
				'[id*="cookie" i]', '[id*="consent" i]', '[id*="privacy" i]', '[id*="gdpr" i]',
				'[class*="cookie" i]', '[class*="consent" i]', '[class*="privacy" i]', '[class*="gdpr" i]',
				'[class*="banner" i]', '[class*="popup" i]', '[class*="modal" i]',
				'[data-testid*="cookie" i]', '[data-testid*="consent" i]',
				'[aria-label*="cookie" i]', '[aria-label*="consent" i]',
				'[role="dialog"]', '[role="alertdialog"]'
			];
			for (const selector of bannerSelectors) {
				const banners = container.querySelectorAll(selector);
				for (const banner of banners) {
					if (!isVisible(banner)) continue;
					const buttons = banner.querySelectorAll('button, [role="button"], input[type="submit"], a, [onclick], [class*="btn"]');
					// Prioritize buttons matching 'accept all', 'accept', or keywords
					for (const btn of buttons) {
						const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
						if ((btnText === 'accept all' || btnText.includes('accept all')) && isVisible(btn)) {
							btn.scrollIntoView({ behavior: 'instant', block: 'center' });
							btn.click();
							return true;
						}
					}
					for (const btn of buttons) {
						const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
						if ((btnText.includes('accept') || keywords.some(kw => btnText.includes(kw))) && isVisible(btn)) {
							btn.scrollIntoView({ behavior: 'instant', block: 'center' });
							btn.click();
							return true;
						}
					}
				}
			}

			// Look for text-based banners
			const allElements = container.querySelectorAll('*');
			for (const el of allElements) {
				const elText = normalize(el.innerText || el.textContent || '');
				if (
					elText.includes('cookie consent') || elText.includes('cookies consent')
					|| (elText.includes('consent') && (elText.includes('cookie') || elText.includes('cookies')))
					|| (elText.includes('uses cookies') || elText.includes('website uses cookies'))
				) {
					if (isVisible(el)) {
						const buttons = el.querySelectorAll('button, [role="button"], input[type="submit"], a, [onclick], [class*="btn"]');
						for (const btn of buttons) {
							const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
							if (btnText.includes('accept all') && isVisible(btn)) {
								btn.scrollIntoView({ behavior: 'instant', block: 'center' });
								btn.click();
								return true;
							}
						}
						for (const btn of buttons) {
							const btnText = normalize(btn.innerText || btn.textContent || btn.value || btn.ariaLabel || btn.title || '');
							if (btnText.includes('accept') && isVisible(btn)) {
								btn.scrollIntoView({ behavior: 'instant', block: 'center' });
								btn.click();
								return true;
							}
						}
					}
				}
			}
			return false;
		};

		// Try in main document
		if (attemptCookieAccept(document)) {
			return true;
		}

		// Try in iframes
		const iframes = document.querySelectorAll('iframe');
		for (const iframe of iframes) {
			try {
				const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
				if (iframeDoc && attemptCookieAccept(iframeDoc)) {
					return true;
				}
			} catch (e) { }
		}

		return false;
	})();
	"""

	try:
		result = await page.evaluate(script)
		accepted = bool(result) if result else False
		if accepted:
			print("   ✅ Bannière de cookies détectée et acceptée (méthode pattern)")
			await asyncio.sleep(1.0)
			return True
		else:
			print("   ℹ️  Aucune bannière de cookies visible (déjà acceptée ou absente)")
			return False
	except Exception as exc:
		print(f"⚠️  Erreur lors de la vérification des cookies: {exc}")
		return False


async def _accept_cookies(browser: Browser, llm, max_retries: int = 2) -> bool:
	"""Accept cookies banner if present. Uses LLM first, then falls back to pattern-based method.
	
	Returns True if cookies were accepted, False otherwise.
	"""
	print("🍪 Recherche de la bannière de cookies...")
	
	# Wait a bit for the banner to appear (it might load asynchronously)
	await asyncio.sleep(3)
	
	# Try LLM first if available
	api_key_present = bool(os.getenv("OPENAI_API_KEY") or os.getenv("BROWSER_USE_API_KEY"))
	if api_key_present:
		llm_accepted = await _accept_cookies_with_llm(browser, llm)
		if llm_accepted:
			return True
		print("   🔄 Tentative avec la méthode pattern-based (sans LLM)...")
	
	# Fallback to pattern-based method
	return await _accept_cookies_pattern_only(browser)


async def _handle_login(browser: Browser, email: str, password: str, llm) -> bool:
	"""Handle login to Station F if credentials are provided. Uses LLM to fill form and submit."""
	print("🔐 Tentative de connexion avec LLM...")
	
	try:
		# Wait for login form to appear
		await asyncio.sleep(2)
		
		# Use LLM to handle login
		task = f"""
		URGENT TASK: Log in to Station F with the provided credentials.
		
		STEP-BY-STEP INSTRUCTIONS:
		
		1. FIND THE LOGIN FORM: Look for email and password input fields on the page.
		   - Email field: usually labeled "Email", "E-mail", or has type="email"
		   - Password field: usually has type="password"
		
		2. FILL THE EMAIL FIELD: Use the input action to fill the email field with: {email}
		   - Find the email input field (it might be the first text input or specifically labeled)
		   - Use input action with index number to fill it
		
		3. FILL THE PASSWORD FIELD: Use the input action to fill the password field with: {password}
		   - Find the password input field (usually has type="password")
		   - Use input action with index number to fill it
		
		4. CLICK THE LOGIN BUTTON: Find and click the login/submit button
		   - Look for buttons labeled "Log in", "Se connecter", "Sign in", "Login", or button with type="submit"
		   - Use click action with the button's index number
		
		5. WAIT FOR REDIRECT: After clicking, wait for the page to redirect (you should be redirected to /companies page)
		   - Use done() action once you see the URL contains "/companies" or you're on the companies page
		
		IMPORTANT:
		- Fill email first, then password, then click login button
		- Do NOT click any other buttons (like "Forgot password", "Sign up", etc.)
		- After clicking login, if a cookie banner appears, accept it first before signaling done
		- Only use done() when you're successfully logged in and on the companies page
		"""
		
		agent = Agent(
			task=task.strip(),
			llm=llm,
			browser=browser,
			max_steps=8,  # Allow enough steps for form filling and login
			flash_mode=True,
			use_thinking=False,
			use_vision='auto',  # Enable vision to see the form
		)
		
		history = await agent.run()
		
		# Don't close the agent - the browser must stay open for the rest of the code
		# The browser will be closed at the end of run_stationf_companies
		
		# Check if login was successful by looking at the URL
		await asyncio.sleep(2)  # Wait for redirect
		
		# Re-get page from browser to ensure it's still valid
		page = await browser.get_current_page()
		if not page:
			return False
		
		current_url = await page.get_url()
		
		if '/companies' in current_url or current_url.endswith('/companies'):
			print("   ✅ Connexion réussie via LLM")
			# CRITICAL: Accept cookies that may appear after redirect
			await asyncio.sleep(2)  # Wait for any modal to appear
			print("   🍪 Vérification des cookies après redirection...")
			cookies_accepted_after_redirect = await _accept_cookies(browser, llm, max_retries=5)
			if cookies_accepted_after_redirect:
				print("   ✅ Cookies acceptés après redirection")
			else:
				print("   ℹ️  Aucune bannière de cookies détectée après redirection")
			return True
		elif history.is_successful():
			# Agent says it's done, but verify URL
			await asyncio.sleep(3)  # Wait a bit more
			page = await browser.get_current_page()
			if not page:
				return False
			current_url = await page.get_url()
			if '/companies' in current_url or current_url.endswith('/companies'):
				print("   ✅ Connexion réussie via LLM (vérifiée)")
				await asyncio.sleep(2)
				cookies_accepted_after_redirect = await _accept_cookies(browser, llm, max_retries=5)
				if cookies_accepted_after_redirect:
					print("   ✅ Cookies acceptés après redirection")
				return True
			else:
				print(f"   ⚠️ Agent a signalé succès mais URL actuelle: {current_url}")
				# Wait a bit more and check again
				await asyncio.sleep(5)
				page = await browser.get_current_page()
				if not page:
					return False
				current_url = await page.get_url()
				if '/companies' in current_url or current_url.endswith('/companies'):
					print("   ✅ Connexion réussie après attente supplémentaire")
					await asyncio.sleep(2)
					cookies_accepted_after_redirect = await _accept_cookies(browser, llm, max_retries=5)
					return True
				return False
		else:
			print("   ⚠️ LLM n'a pas réussi à se connecter")
			return False
			
	except Exception as e:
		print(f"   ⚠️ Erreur lors de la connexion: {e}")
		import traceback
		traceback.print_exc()
		return False


async def run_stationf_companies(task_input: StationFCompaniesInput) -> StationFCompaniesReport | None:
	"""Execute the agent and return the structured list of companies."""

	print("🔧 Configuration du LLM...")
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
		page_extraction_llm = ChatBrowserUse()
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
		print(f"✅ Utilisation de ChatOpenAI avec le modèle: {model_name}")
		print(f"✅ Utilisation de ChatOpenAI pour l'extraction avec le modèle: {extraction_model}")
		if is_gemini:
			print("   ⚠️  Mode Gemini détecté: utilisation du schéma dans le prompt système")

	print("🌐 Création du navigateur...")
	browser = Browser(headless=False, keep_alive=True)  # Keep browser alive so agents don't close it
	await browser.start()
	
	try:
		# Navigate to the companies URL
		companies_url = str(task_input.companies_url)
		print(f"📍 Navigation vers: {companies_url}")
		navigate_event = NavigateToUrlEvent(url=companies_url, new_tab=True)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event
		
		# Get the current page after navigation
		page = await browser.get_current_page()
		if not page:
			page = await browser.new_page(companies_url)
		
		await asyncio.sleep(5)  # Wait for initial page load
		
		# CRITICAL: Accept cookies BEFORE login - must be done first
		print("🍪 Acceptation des cookies AVANT l'authentification...")
		cookies_accepted = await _accept_cookies(browser, llm, max_retries=8)
		if cookies_accepted:
			print("   ✅ Cookies acceptés avec succès avant l'authentification")
		else:
			print("   ℹ️  Aucune bannière de cookies détectée (peut-être déjà acceptée ou absente)")
		
		# Additional wait to ensure cookie modal is fully closed
		await asyncio.sleep(2)
		
		# Re-get page after cookies acceptance to ensure it's still valid
		# The agent may have closed the browser, so we need to check if it's still running
		try:
			page = await browser.get_current_page()
			if not page:
				# Browser might have been closed by the agent, try to navigate again
				await browser.navigate_to(companies_url)
				page = await browser.get_current_page()
		except (RuntimeError, AssertionError) as e:
			# Browser was closed, restart it
			print(f"   ⚠️ Navigateur fermé par l'agent, redémarrage...")
			try:
				await browser.start()
			except Exception:
				pass  # Already started
			await browser.navigate_to(companies_url)
			page = await browser.get_current_page()
		
		if not page:
			print("   ❌ Impossible de récupérer la page après l'acceptation des cookies")
			return _fallback_report(companies_url, "Impossible de récupérer la page après l'acceptation des cookies")
		
		# Handle login if credentials are provided - MUST succeed before scrolling
		if task_input.email and task_input.password:
			current_url = await page.get_url()
			if '/login' in current_url:
				login_success = await _handle_login(browser, task_input.email, task_input.password, llm)
				if not login_success:
					print("   ❌ Échec de la connexion. Impossible de continuer sans authentification.")
					return _fallback_report(companies_url, "Échec de la connexion: impossible d'accéder à la page des entreprises sans authentification.")
				
				# Re-get page after successful login
				page = await browser.get_current_page()
				if not page:
					page = await browser.new_page(companies_url)
				
				# Verify we're on the companies page
				await asyncio.sleep(2)
				final_url = await page.get_url()
				if '/companies' not in final_url and not final_url.endswith('/companies'):
					print(f"   ⚠️ Après connexion, URL actuelle: {final_url}")
					# Try navigating to companies page directly
					print("   📍 Tentative de navigation directe vers la page des entreprises...")
					navigate_event = NavigateToUrlEvent(url=companies_url, new_tab=False)
					await browser.event_bus.dispatch(navigate_event)
					await navigate_event
					await asyncio.sleep(3)
					page = await browser.get_current_page()
					if not page:
						page = await browser.new_page(companies_url)
				
				print("   ✅ Connexion réussie, prêt à extraire les entreprises")
			else:
				print("   ℹ️  Déjà sur la page des entreprises (pas de login nécessaire)")
		else:
			# Check if we need login but don't have credentials
			if not page:
				print("   ❌ Impossible de récupérer la page")
				return _fallback_report(companies_url, "Impossible de récupérer la page")
			current_url = await page.get_url()
			if '/login' in current_url:
				print("   ⚠️ La page nécessite une authentification mais aucun identifiant n'a été fourni.")
				print("   💡 Ajoutez STATIONF_EMAIL et STATIONF_PASSWORD dans votre fichier .env")
				return _fallback_report(companies_url, "Authentification requise mais non fournie.")
		
		# CRITICAL: Final check for cookies before scrolling (in case a modal appeared after navigation)
		print("🍪 Vérification finale des cookies avant le scroll...")
		await asyncio.sleep(2)  # Wait for any modal to appear
		final_page = await browser.get_current_page()
		if final_page:
			cookies_accepted_final = await _accept_cookies(browser, llm, max_retries=5)
			if cookies_accepted_final:
				print("   ✅ Cookies acceptés lors de la vérification finale")
			else:
				print("   ℹ️  Aucune bannière de cookies détectée lors de la vérification finale")
		
		# Scroll to bottom naively (without LLM)
		await _scroll_to_bottom_naive(browser, max_scrolls=50)
		
		# Verify content is loaded and extract directly using JavaScript
		print("🔍 Vérification que nous sommes bien en bas de la page avant extraction...")
		current_page = await browser.get_current_page()
		if not current_page:
			raise RuntimeError("No current page available after scroll")
		
		# Verify we're at the bottom of the page
		try:
			scroll_info_raw = await current_page.evaluate("""
				() => {
					const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
					const scrollHeight = document.documentElement.scrollHeight;
					const clientHeight = window.innerHeight || document.documentElement.clientHeight;
					const isAtBottom = (scrollTop + clientHeight >= scrollHeight - 10); // 10px tolerance
					return {
						scrollTop: scrollTop,
						scrollHeight: scrollHeight,
						clientHeight: clientHeight,
						isAtBottom: isAtBottom,
						distanceFromBottom: scrollHeight - (scrollTop + clientHeight)
					};
				}
			""")
			
			# Handle both dict and JSON string responses
			if isinstance(scroll_info_raw, str):
				try:
					scroll_info = json.loads(scroll_info_raw)
				except json.JSONDecodeError:
					scroll_info = {}
			else:
				scroll_info = scroll_info_raw if isinstance(scroll_info_raw, dict) else {}
			
			if scroll_info and not scroll_info.get('isAtBottom', False):
				distance = scroll_info.get('distanceFromBottom', 0)
				print(f"   ⚠️ Pas encore en bas de la page (distance: {distance}px), scroll supplémentaire...")
				# Scroll to absolute bottom
				await current_page.evaluate("() => window.scrollTo(0, document.documentElement.scrollHeight)")
				await asyncio.sleep(2)
				# Verify again
				scroll_info_raw2 = await current_page.evaluate("""
					() => {
						const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
						const scrollHeight = document.documentElement.scrollHeight;
						const clientHeight = window.innerHeight || document.documentElement.clientHeight;
						return {
							isAtBottom: (scrollTop + clientHeight >= scrollHeight - 10),
							distanceFromBottom: scrollHeight - (scrollTop + clientHeight)
						};
					}
				""")
				
				if isinstance(scroll_info_raw2, str):
					try:
						scroll_info = json.loads(scroll_info_raw2)
					except json.JSONDecodeError:
						scroll_info = {}
				else:
					scroll_info = scroll_info_raw2 if isinstance(scroll_info_raw2, dict) else {}
			
			if scroll_info and scroll_info.get('isAtBottom', False):
				print("   ✅ Confirmé: nous sommes bien en bas de la page")
			else:
				print("   ⚠️ Impossible de confirmer que nous sommes en bas, mais on continue quand même")
		except Exception as e:
			print(f"   ⚠️ Erreur lors de la vérification du scroll: {e}, on continue quand même")
		
		# Check if content is loaded - use the actual structure with data-slot
		try:
			elements_count = await current_page.evaluate(
				"() => document.querySelectorAll('[data-slot=\"drawer-trigger\"]').length"
			)
			elements_count = int(elements_count) if elements_count else 0
			print(f"   📊 {elements_count} éléments d'entreprises trouvés (data-slot=\"drawer-trigger\")")
			
			if elements_count == 0:
				print("   ⚠️ Aucun élément trouvé, attente supplémentaire...")
				await asyncio.sleep(5)
				elements_count = await current_page.evaluate(
					"() => document.querySelectorAll('[data-slot=\"drawer-trigger\"]').length"
				)
				elements_count = int(elements_count) if elements_count else 0
				print(f"   📊 Après attente: {elements_count} éléments trouvés")
		except Exception as e:
			print(f"   ⚠️ Erreur lors de la vérification: {e}")
			elements_count = 0
		
		# Extract HTML directly using JavaScript (bypass agent for reliability)
		extraction_successful = False
		if elements_count > 0:
			print("📥 Extraction directe du HTML des entreprises...")
			try:
				# Extract HTML elements using the actual structure with data-slot
				html_elements_json = await current_page.evaluate(
					"() => { const elements = Array.from(document.querySelectorAll('[data-slot=\"drawer-trigger\"]')); return JSON.stringify(elements.map(el => el.outerHTML)); }"
				)
				
				if html_elements_json and html_elements_json != '[]' and html_elements_json.strip():
					# Parse the JSON string
					html_elements = json.loads(html_elements_json)
					print(f"   ✅ {len(html_elements)} éléments HTML extraits")
					
					if html_elements and len(html_elements) > 0:
						# Parse HTML elements directly
						html_report = _parse_html_sections(html_elements, str(task_input.companies_url))
						if html_report and html_report.companies and len(html_report.companies) > 0:
							print(f"   ✅ {len(html_report.companies)} entreprises parsées depuis le HTML")
							
							# Limit to max_companies if needed
							if task_input.max_companies < 10000 and len(html_report.companies) > task_input.max_companies:
								html_report.companies = html_report.companies[:task_input.max_companies]
							
							extraction_successful = True
							return _sanitize_report(html_report)
						else:
							print(f"   ⚠️ Aucune entreprise parsée depuis le HTML (report: {html_report})")
					else:
						print(f"   ⚠️ Tableau HTML vide (length: {len(html_elements) if html_elements else 0})")
				else:
					print(f"   ⚠️ Aucune donnée extraite (result: {html_elements_json[:100] if html_elements_json else 'None'})")
			except json.JSONDecodeError as e:
				print(f"   ⚠️ Erreur de parsing JSON lors de l'extraction directe: {e}")
				print(f"   ⚠️ Contenu reçu: {html_elements_json[:200] if 'html_elements_json' in locals() else 'N/A'}")
			except Exception as e:
				print(f"   ⚠️ Erreur lors de l'extraction directe: {e}")
				import traceback
				traceback.print_exc()
		else:
			print(f"   ⚠️ Aucun élément trouvé (elements_count: {elements_count}), impossible d'extraire")
		
		# Only use agent as fallback if direct extraction completely failed
		if not extraction_successful:
			print("❌ Extraction directe échouée. Le contenu n'a peut-être pas été chargé correctement.")
			print("   💡 Vérifiez que le scroll s'est bien terminé et que la page a chargé le contenu.")
			return _fallback_report(str(task_input.companies_url), "Extraction directe échouée: aucun élément d'entreprise trouvé après le scroll.")
		
		# This should never be reached, but just in case:
		print("🤖 Utilisation de l'agent comme dernier recours...")
		# Exclude scroll and navigate tools since they're already done automatically
		tools = Tools(exclude_actions=['scroll', 'navigate', 'wait'])
		agent = Agent(
			task=build_task(task_input),
			llm=llm,
			page_extraction_llm=page_extraction_llm,
			browser=browser,
			tools=tools,
			output_model_schema=StationFCompaniesReport,
			use_vision='auto',
			vision_detail_level='auto',
			step_timeout=300,
			llm_timeout=180,
			max_failures=5,
			max_history_items=10,
			directly_open_url=False,
		)
		print("✅ Agent créé")

		print("▶️  Démarrage de l'exécution de l'agent (extraction uniquement)...")
		history = await agent.run()
		print("✅ Exécution terminée")
		print_llm_usage_summary(history)

		# Check if agent completed successfully
		agent_successful = history.is_successful()
		if not agent_successful and history.has_errors():
			print("⚠️  ATTENTION: Il semble y avoir eu un problème avec l'agent, mais on va essayer d'extraire les données quand même.")

		# Try to get structured output first
		if history.structured_output:
			return _sanitize_report(history.structured_output)  # type: ignore[arg-type]

		# Try to extract from final result
		final_result = history.final_result()
		if final_result:
			try:
				report = StationFCompaniesReport.model_validate_json(final_result)
				if not agent_successful:
					print("⚠️  Données récupérées depuis le résultat final malgré l'échec de l'agent.")
				return _sanitize_report(report)
			except ValidationError:
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if json_match:
					try:
						report = StationFCompaniesReport.model_validate_json(json_match.group(1))
						if not agent_successful:
							print("⚠️  Données récupérées depuis le résultat final (markdown) malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError:
						pass

		# Try to extract from evaluate actions (HTML sections)
		all_extracted_companies: list[StationFCompany] = []
		seen_names: set[str] = set()
		
		# Look for evaluate actions and their results
		action_results_list = history.action_results()
		model_actions_list = history.model_actions()
		
		# Match actions with their results
		for i, action_dict in enumerate(model_actions_list):
			if 'evaluate' in action_dict:
				if i < len(action_results_list):
					result = action_results_list[i]
					if result and result.extracted_content:
						try:
							content_str = result.extracted_content.strip()
							if content_str.startswith('```'):
								parts = content_str.split('```')
								if len(parts) >= 3:
									content_str = parts[1].strip()
									if '\n' in content_str:
										content_str = content_str.split('\n', 1)[1]
									else:
										content_str = parts[2].strip() if len(parts) > 2 else content_str
							
							evaluate_result = json.loads(content_str)
							html_report = _parse_html_sections(evaluate_result, str(task_input.companies_url))
							if html_report and html_report.companies:
								for company in html_report.companies:
									if company.name and company.name not in seen_names:
										all_extracted_companies.append(company)
										seen_names.add(company.name)
						except (json.JSONDecodeError, TypeError):
							try:
								html_report = _parse_html_sections(result.extracted_content, str(task_input.companies_url))
								if html_report and html_report.companies:
									for company in html_report.companies:
										if company.name and company.name not in seen_names:
											all_extracted_companies.append(company)
											seen_names.add(company.name)
							except Exception:
								pass
		
		# Try to extract from action results
		extracted_contents = history.extracted_content()
		
		for content in extracted_contents:
			if not content:
				continue
			
			# Try parsing as JSON first
			try:
				content_str = content.strip()
				if content_str.startswith('```'):
					parts = content_str.split('```')
					if len(parts) >= 3:
						content_str = parts[1].strip()
						if '\n' in content_str:
							content_str = content_str.split('\n', 1)[1]
						else:
							content_str = parts[2].strip() if len(parts) > 2 else content_str
				
				evaluate_result = json.loads(content_str)
				html_report = _parse_html_sections(evaluate_result, str(task_input.companies_url))
				if html_report and html_report.companies:
					for company in html_report.companies:
						if company.name and company.name not in seen_names:
							all_extracted_companies.append(company)
							seen_names.add(company.name)
					continue
			except (json.JSONDecodeError, TypeError):
				pass
			
			# Try parsing as HTML sections
			html_report = _parse_html_sections(content, str(task_input.companies_url))
			if html_report and html_report.companies:
				for company in html_report.companies:
					if company.name and company.name not in seen_names:
						all_extracted_companies.append(company)
						seen_names.add(company.name)
				continue

			# Fallback to markdown parsing
			markdown_report = _parse_extracted_markdown(content, str(task_input.companies_url))
			if markdown_report and markdown_report.companies:
				for company in markdown_report.companies:
					if company.name and company.name not in seen_names:
						all_extracted_companies.append(company)
						seen_names.add(company.name)

		# If we found companies, return them
		if all_extracted_companies:
			if task_input.max_companies < 10000:
				all_extracted_companies = all_extracted_companies[:task_input.max_companies]

			report = StationFCompaniesReport(
				source_url=AnyHttpUrl(str(task_input.companies_url)),
				companies=all_extracted_companies,
			)
			if not agent_successful:
				print(f"⚠️  Données récupérées depuis les résultats d'extraction malgré l'échec de l'agent. {len(all_extracted_companies)} entreprises trouvées.")
			return _sanitize_report(report)

		# Fallback: try individual content parsing
		for content in reversed(extracted_contents):
			if not content:
				continue

			markdown_report = _parse_extracted_markdown(content, str(task_input.companies_url))
			if markdown_report and markdown_report.companies:
				if not agent_successful:
					print(f"⚠️  Données récupérées depuis les résultats d'extraction (markdown) malgré l'échec de l'agent. {len(markdown_report.companies)} entreprises trouvées.")
				return _sanitize_report(markdown_report)

			try:
				report = StationFCompaniesReport.model_validate_json(content)
				if not agent_successful:
					print("⚠️  Données récupérées depuis les résultats d'extraction malgré l'échec de l'agent.")
				return _sanitize_report(report)
			except ValidationError:
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
				if json_match:
					try:
						report = StationFCompaniesReport.model_validate_json(json_match.group(1))
						if not agent_successful:
							print("⚠️  Données récupérées depuis les résultats d'extraction (markdown JSON) malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError:
						pass

		# Try to extract from model actions (look for done actions with data)
		for action_dict in reversed(history.model_actions()):
			if 'done' in action_dict:
				done_data = action_dict.get('done', {})
				if isinstance(done_data, dict) and 'data' in done_data:
					data = done_data['data']
					data_copy = copy.deepcopy(data) if isinstance(data, dict) else data

					def convert_urls_to_strings(obj: Any) -> Any:
						"""Recursively convert AnyHttpUrl objects to strings."""
						if isinstance(obj, dict):
							return {k: convert_urls_to_strings(v) for k, v in obj.items()}
						elif isinstance(obj, list):
							return [convert_urls_to_strings(item) for item in obj]
						elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
							if 'HttpUrl' in type(obj).__name__ or 'Url' in type(obj).__name__:
								return str(obj)
						return obj

					data_copy = convert_urls_to_strings(data_copy)

					try:
						report = StationFCompaniesReport.model_validate(data_copy)
						if not agent_successful:
							print("⚠️  Données récupérées depuis l'action 'done' malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError as e:
						try:
							json_str = json.dumps(data_copy, default=str)
							report = StationFCompaniesReport.model_validate_json(json_str)
							if not agent_successful:
								print("⚠️  Données récupérées depuis l'action 'done' (après conversion JSON) malgré l'échec de l'agent.")
							return _sanitize_report(report)
						except (ValidationError, json.JSONDecodeError):
							pass

		# If we get here, we couldn't extract any data
		if not agent_successful:
			print("❌ Impossible d'extraire les données malgré plusieurs tentatives.")
		return _fallback_report(str(task_input.companies_url), "L'agent a été interrompu avant de finaliser le JSON.")
	
	finally:
		# Close browser to free resources
		print("🧹 Fermeture du navigateur...")
		try:
			await browser.kill()
			print("✅ Navigateur fermé")
		except Exception as e:
			print(f"⚠️  Erreur lors de la fermeture du navigateur: {e}")


def parse_arguments() -> StationFCompaniesInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Extrait les entreprises depuis la page Station F HAL')
	parser.add_argument(
		'--url',
		default='https://hal2.stationf.co/companies',
		help='URL de la page des entreprises Station F (par défaut: https://hal2.stationf.co/companies)',
	)
	parser.add_argument(
		'--max-companies',
		type=int,
		default=1000,
		help='Nombre maximal d\'entreprises à extraire (par défaut: 1000)',
	)
	parser.add_argument(
		'--output',
		default='stationf_companies.json',
		help='Chemin du fichier JSON résultat (par défaut: ./stationf_companies.json)',
	)
	parser.add_argument(
		'--email',
		default=None,
		help='Email pour l\'authentification (optionnel, peut aussi être défini via STATIONF_EMAIL dans .env)',
	)
	parser.add_argument(
		'--password',
		default=None,
		help='Mot de passe pour l\'authentification (optionnel, peut aussi être défini via STATIONF_PASSWORD dans .env)',
	)
	args = parser.parse_args()
	
	# Read credentials from environment variables if not provided via CLI
	email = args.email or os.getenv('STATIONF_EMAIL')
	password = args.password or os.getenv('STATIONF_PASSWORD')
	
	return StationFCompaniesInput(
		url=args.url,
		max_companies=args.max_companies,
		output_path=Path(args.output),
		email=email,
		password=password,
	)


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		companies_url = str(task_input.companies_url)
		print(f"🚀 Démarrage de l'agent pour Station F")
		print(f"📍 URL des entreprises: {companies_url}")
		print(f"📊 Nombre max d'entreprises: {task_input.max_companies}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")
		if task_input.email:
			print(f"🔐 Authentification: {task_input.email}")

		result = await run_stationf_companies(task_input)

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

