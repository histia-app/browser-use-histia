"""
Agent designed to extract companies from Observatoire Deeptech companies pages.
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
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer, field_validator

from bs4 import BeautifulSoup

# Load environment variables immediately so the agent can access API keys.
load_dotenv()

# Configure timeouts BEFORE importing browser_use to ensure they're applied
# Use longer timeouts for heavy pages with lots of content
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')  # Increase scroll timeout for slow-loading pages

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class DeeptechCompaniesInput(BaseModel):
	"""User-provided parameters for the Deeptech companies extraction task."""

	url: str = Field(
		...,
		description='URL of the Deeptech companies page to extract from',
	)
	max_companies: int = Field(
		1000,
		ge=1,
		le=10000,
		description='Maximum number of companies to capture (use a high number like 1000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('deeptech_companies.json'),
		description='Destination for the JSON list of companies',
	)

	@field_validator('url')
	@classmethod
	def validate_url(cls, v: str) -> str:
		"""Validate URL format."""
		if not v.startswith(('http://', 'https://')):
			raise ValueError(f"Invalid URL format: {v}. Expected http:// or https://")
		return v


class DeeptechCompany(BaseModel):
	"""Structured information for each Deeptech company entry."""

	name: str = Field(..., description='Company name exactly as written on the page')
	company_url: str | None = Field(
		None,
		description='Complete URL to the company page (format: https://observatoire.lesdeeptech.fr/companies/...)',
	)
	description: str | None = Field(
		None,
		description='Company description/tagline if available',
	)
	logo_url: str | None = Field(
		None,
		description='URL of the company logo/image',
	)
	ranking: int | None = Field(
		None,
		description='Company ranking/score if available',
	)
	market: list[str] = Field(
		default_factory=list,
		description='Market focus (e.g., ["B2B"])',
	)
	industries: list[str] = Field(
		default_factory=list,
		description='Industries the company operates in (e.g., ["health", "biotechnology"])',
	)
	business_types: list[str] = Field(
		default_factory=list,
		description='Business types (e.g., ["commission", "manufacturing"])',
	)
	employees: str | None = Field(
		None,
		description='Number of employees (e.g., "2-10 employees")',
	)
	launch_date: str | None = Field(
		None,
		description='Company launch date (e.g., "2023")',
	)
	hq_location: str | None = Field(
		None,
		description='Headquarters location (e.g., "La Ciotat, France")',
	)
	status: str | None = Field(
		None,
		description='Company status (e.g., "operational")',
	)
	growth_stage: str | None = Field(
		None,
		description='Growth stage (e.g., "seed")',
	)
	web_visits: int | None = Field(
		None,
		description='Number of web visits if available',
	)
	total_funding: str | None = Field(
		None,
		description='Total funding amount if available',
	)
	valuation: str | None = Field(
		None,
		description='Company valuation if available',
	)
	last_funding: str | None = Field(
		None,
		description='Last funding round information (e.g., "N/A SPINOUT")',
	)


class DeeptechCompaniesReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='Source URL that was analysed')
	companies: list[DeeptechCompany] = Field(
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


def _fallback_report(source_url: str, reason: str) -> DeeptechCompaniesReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	return DeeptechCompaniesReport(
		source_url=AnyHttpUrl(source_url),
		companies=[
			DeeptechCompany(
				name='Informations indisponibles',
				company_url=source_url,
				description=None,
				logo_url=None,
				ranking=None,
				market=[],
				industries=[],
				business_types=[],
				employees=None,
				launch_date=None,
				hq_location=None,
				status=None,
				growth_stage=None,
				web_visits=None,
				total_funding=None,
				valuation=None,
				last_funding=None,
			)
		],
	)


def _normalize_deeptech_url(url: str | None, base_url: str) -> str | None:
	"""Convert relative URLs to absolute Deeptech URLs."""
	if not url:
		return None

	url = url.strip()
	if not url:
		return None

	# If it's already an absolute URL, return as is
	if url.startswith(('http://', 'https://')):
		# Ensure it's a Deeptech URL
		if 'lesdeeptech.fr' in url.lower() or 'observatoire.lesdeeptech.fr' in url.lower():
			return url
		return None

	# If it starts with /, make it relative to the base domain
	if url.startswith('/'):
		parsed_base = urlparse(base_url)
		return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"

	# Otherwise, try to resolve relative to base URL
	try:
		resolved = urljoin(base_url, url)
		if 'lesdeeptech.fr' in resolved.lower():
			return resolved
		return None
	except Exception:
		return None


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> DeeptechCompaniesReport | None:
	"""Parse HTML sections from evaluate action to build DeeptechCompaniesReport."""

	companies: list[DeeptechCompany] = []

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
			item = soup.find('div', class_='table-list-item')

			if not item:
				continue

			# Extract name and URL from the link
			name = None
			company_url = None
			name_link = item.select_one('a[data-testid="internal"][href^="/companies/"]')
			if name_link:
				name = name_link.get_text(strip=True)
				href = str(name_link.get('href', ''))
				if href:
					company_url = _normalize_deeptech_url(href, source_url)

			# Extract description
			description = None
			desc_p = item.select_one('p.tw\\:text-neutral-400, p.text-neutral-400')
			if desc_p:
				description = desc_p.get_text(strip=True)

			# Extract logo URL
			logo_url = None
			logo_img = item.select_one('img.responsive-img, img[itemprop="image"]')
			if logo_img:
				logo_url = str(logo_img.get('src', ''))

			# Extract ranking
			ranking = None
			ranking_p = item.select_one('p.tw\\:text-sm.tw\\:font-normal.tw\\:text-neutral-400, p.text-sm')
			if ranking_p:
				ranking_text = ranking_p.get_text(strip=True)
				ranking_match = re.search(r'(\d+)', ranking_text)
				if ranking_match:
					try:
						ranking = int(ranking_match.group(1))
					except ValueError:
						pass

			# Extract market (B2B, B2C, etc.)
			market = []
			market_links = item.select('div.companyMarket a[data-testid="internal"]')
			for link in market_links:
				market_text = link.get_text(strip=True)
				if market_text and market_text not in market:
					market.append(market_text)

			# Extract industries
			industries = []
			industry_links = item.select('div.companyMarket ul.item-list-column:not(.item-list-column--horizontal) a[data-testid="internal"]')
			for link in industry_links:
				industry_text = link.get_text(strip=True)
				if industry_text and industry_text not in industries:
					industries.append(industry_text)

			# Extract business types
			business_types = []
			type_links = item.select('div.business-type-column a[data-testid="internal"]')
			for link in type_links:
				type_text = link.get_text(strip=True)
				if type_text and type_text not in business_types:
					business_types.append(type_text)

			# Extract employees
			employees = None
			employees_elem = item.select_one('div.companyEmployees .growth-line-chart__hover-content .growth-line-chart__value')
			if employees_elem:
				employees_text = employees_elem.get_text(strip=True)
				if employees_text and employees_text != '-':
					employees = employees_text
			else:
				# Fallback: try direct value
				employees_elem = item.select_one('div.companyEmployees .growth-line-chart__value')
				if employees_elem:
					employees_text = employees_elem.get_text(strip=True)
					if employees_text and employees_text != '-':
						employees = employees_text

			# Extract launch date
			launch_date = None
			launch_time = item.select_one('div.launchDate time[datetime]')
			if launch_time:
				launch_date = str(launch_time.get('datetime', ''))
			else:
				# Fallback: try text content
				launch_elem = item.select_one('div.launchDate')
				if launch_elem:
					launch_text = launch_elem.get_text(strip=True)
					if launch_text and launch_text != '-':
						launch_date = launch_text

			# Extract HQ location
			hq_location = None
			hq_elem = item.select_one('div.hqLocations')
			if hq_elem:
				hq_text = hq_elem.get_text(strip=True)
				if hq_text and hq_text != '-':
					hq_location = hq_text

			# Extract status
			status = None
			status_elem = item.select_one('div.companyStatus')
			if status_elem:
				status_text = status_elem.get_text(strip=True)
				if status_text and status_text != '-':
					status = status_text

			# Extract growth stage
			growth_stage = None
			growth_elem = item.select_one('div.growthStage span')
			if growth_elem:
				growth_text = growth_elem.get_text(strip=True)
				if growth_text and growth_text != '-':
					growth_stage = growth_text

			# Extract web visits
			web_visits = None
			visits_elem = item.select_one('div.companyWebVisits .growth-line-chart__value')
			if visits_elem:
				visits_text = visits_elem.get_text(strip=True)
				visits_match = re.search(r'(\d+)', visits_text)
				if visits_match:
					try:
						web_visits = int(visits_match.group(1))
					except ValueError:
						pass

			# Extract total funding
			total_funding = None
			funding_elem = item.select_one('div.totalFunding')
			if funding_elem:
				funding_text = funding_elem.get_text(strip=True)
				if funding_text and funding_text != '-':
					total_funding = funding_text

			# Extract valuation
			valuation = None
			valuation_elem = item.select_one('div.valuation')
			if valuation_elem:
				valuation_text = valuation_elem.get_text(strip=True)
				if valuation_text and valuation_text != '-':
					valuation = valuation_text

			# Extract last funding
			last_funding = None
			last_funding_elem = item.select_one('div.lastFundingEnhanced')
			if last_funding_elem:
				last_funding_text = last_funding_elem.get_text(strip=True)
				if last_funding_text and last_funding_text != '-':
					last_funding = last_funding_text

			# Only add company if we have at least a name
			if name:
				companies.append(DeeptechCompany(
					name=name,
					company_url=company_url,
					description=description,
					logo_url=logo_url,
					ranking=ranking,
					market=market,
					industries=industries,
					business_types=business_types,
					employees=employees,
					launch_date=launch_date,
					hq_location=hq_location,
					status=status,
					growth_stage=growth_stage,
					web_visits=web_visits,
					total_funding=total_funding,
					valuation=valuation,
					last_funding=last_funding,
				))
		except Exception as e:
			# Skip malformed sections
			print(f"⚠️ Erreur lors du parsing d'une section: {e}")
			continue

	if companies:
		return DeeptechCompaniesReport(
			source_url=AnyHttpUrl(source_url),
			companies=companies,
		)
	return None


def _sanitize_report(report: DeeptechCompaniesReport) -> DeeptechCompaniesReport:
	"""Apply basic clean-up rules on top of the structured output."""

	base_url = str(report.source_url)
	for company in report.companies:
		# Normalize company URL
		if company.company_url:
			company_url_str = str(company.company_url)
			if not company_url_str.startswith('http://') and not company_url_str.startswith('https://'):
				company.company_url = _normalize_deeptech_url(company_url_str, base_url)
			elif 'lesdeeptech.fr' not in company_url_str.lower():
				# Invalid URL, set to None
				company.company_url = None

		# Clean up lists: remove empty strings and strip whitespace
		if company.market:
			company.market = [m.strip() for m in company.market if m.strip()]
		if company.industries:
			company.industries = [i.strip() for i in company.industries if i.strip()]
		if company.business_types:
			company.business_types = [bt.strip() for bt in company.business_types if bt.strip()]

	return report


def build_task(task_input: DeeptechCompaniesInput) -> str:
	"""Create the natural-language instructions fed to the agent, specialized for Deeptech companies extraction."""

	extract_all = task_input.max_companies >= 1000
	source_url = task_input.url

	return dedent(
		f"""
		Tu es un agent spécialisé dans l'extraction d'entreprises depuis la page Observatoire Deeptech.

		Objectif CRITIQUE:
		- Navigue directement vers l'URL: {source_url}
		- IMPORTANT: Utilise l'action `navigate` pour aller directement à cette URL - NE FAIS PAS de recherche internet
		- {"Identifie et extrait TOUTES les entreprises présentes sur cette page, SANS AUCUNE EXCEPTION." if extract_all else f"Identifie et extrait jusqu'à {task_input.max_companies} entreprises présentes sur cette page."}
		- IMPORTANT: Ne filtre PAS les entreprises. Prends TOUTES les entreprises visibles sur la page.
		- Ne confonds PAS les titres de sections avec des entreprises réelles.
		- Pour chaque entreprise, capture:
		  • `name`: nom exact de l'entreprise tel qu'affiché
		  • `company_url`: URL complète vers la page de l'entreprise (format: https://observatoire.lesdeeptech.fr/companies/...)
		  • `description`: description/tagline de l'entreprise
		  • `logo_url`: URL du logo de l'entreprise
		  • `ranking`: score/ranking si visible
		  • `market`: marché cible (ex: ["B2B"])
		  • `industries`: industries (ex: ["health", "biotechnology"])
		  • `business_types`: types de business (ex: ["commission", "manufacturing"])
		  • `employees`: nombre d'employés (ex: "2-10 employees")
		  • `launch_date`: date de lancement (ex: "2023")
		  • `hq_location`: localisation du siège (ex: "La Ciotat, France")
		  • `status`: statut (ex: "operational")
		  • `growth_stage`: stade de croissance (ex: "seed")
		  • `web_visits`: nombre de visites web si visible
		  • `total_funding`: financement total si visible
		  • `valuation`: valorisation si visible
		  • `last_funding`: dernier tour de financement si visible

		Processus pour Observatoire Deeptech:
		⚠️ IMPORTANT: La navigation a déjà été effectuée automatiquement.
		Tu es déjà sur la page {source_url}.
		
		Tu dois maintenant UNIQUEMENT:
		1. EXTRACTION DES ENTREPRISES - MÉTHODE OBLIGATOIRE
		   - ⚠️ CRITIQUE: Utilise UNIQUEMENT l'action `evaluate` pour extraire le code source HTML complet de la page
		   - ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs `data-testid` directement
		   - Format exact de l'action `evaluate`: {{"evaluate": {{"code": "TON_CODE_JAVASCRIPT"}}}}
		   - Code JavaScript à utiliser EXACTEMENT (copie-le tel quel):
		   (function(){{const items = Array.from(document.querySelectorAll('div.table-list-item'));return JSON.stringify(items.map(item => item.outerHTML));}})()
		   - Ce code va extraire le HTML complet de tous les éléments `<div class="table-list-item">` présents dans le code source de la page
		   - Le résultat sera une chaîne JSON contenant un tableau de chaînes HTML
		   - ⚠️ ATTENTION: `evaluate` utilise le paramètre `code` (PAS `query`, PAS `extract_links`, PAS `start_from_char`)
		   - Une fois que tu as le résultat de `evaluate`, utilise directement `done` avec les données parsées dans le format DeeptechCompaniesReport

		2. Traitement des résultats de `evaluate`:
		   - Le résultat de `evaluate` sera une chaîne JSON contenant un tableau de chaînes HTML
		   - Parse cette chaîne JSON pour obtenir le tableau de HTML
		   - Pour chaque élément HTML du tableau, extrais les informations suivantes:
		     • `name`: texte du lien `<a data-testid="internal" href="/companies/...">` dans l'item
		     • `company_url`: attribut `href` du même lien (normalise en URL absolue: https://observatoire.lesdeeptech.fr/companies/...)
		     • `description`: texte du paragraphe avec classe "text-neutral-400"
		     • `logo_url`: attribut `src` de l'image avec classe "responsive-img"
		     • `ranking`: nombre extrait du paragraphe avec classe "text-sm"
		     • `market`: texte de tous les liens dans `div.companyMarket` (première liste horizontale)
		     • `industries`: texte de tous les liens dans `div.companyMarket` (listes verticales)
		     • `business_types`: texte de tous les liens dans `div.business-type-column`
		     • `employees`: texte de `.growth-line-chart__hover-content .growth-line-chart__value` dans `div.companyEmployees`
		     • `launch_date`: attribut `datetime` ou texte de `div.launchDate time`
		     • `hq_location`: texte de `div.hqLocations`
		     • `status`: texte de `div.companyStatus`
		     • `growth_stage`: texte de `div.growthStage span`
		     • `web_visits`: nombre extrait de `div.companyWebVisits .growth-line-chart__value`
		     • `total_funding`: texte de `div.totalFunding`
		     • `valuation`: texte de `div.valuation`
		     • `last_funding`: texte de `div.lastFundingEnhanced`
		   - Construis un objet `DeeptechCompaniesReport` avec:
		     • `source_url`: "{source_url}" (chaîne de caractères, pas d'objet URL)
		     • `companies`: tableau de toutes les entreprises extraites dans l'ordre
		   - Limite la liste finale à {task_input.max_companies} entreprises maximum si nécessaire
		   - Utilise l'action `done` avec le champ `data` contenant l'objet `DeeptechCompaniesReport` complet
		   - Format exact: {{"done": {{"success": true, "data": {{"source_url": "{source_url}", "companies": [{{"name": "...", "company_url": "...", ...}}, ...]}}}}}}

		Règles importantes:
		- ⚠️ NAVIGATION: La navigation a déjà été effectuée automatiquement - NE PAS naviguer!
		- ⚠️ NE PAS utiliser `navigate` - tu es déjà sur la bonne page!
		- ⚠️ NE PAS utiliser `scroll` - récupère directement le code source HTML complet de la page!
		- ⚠️ EXTRACTION OBLIGATOIRE: Utilise UNIQUEMENT `evaluate` avec le paramètre `code` pour extraire le HTML
		- Format `evaluate`: {{"evaluate": {{"code": "(function(){{const items = Array.from(document.querySelectorAll('div.table-list-item'));return JSON.stringify(items.map(item => item.outerHTML));}})()"}}}}
		- ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs `data-testid` directement
		- Le code JavaScript doit sélectionner tous les éléments avec `document.querySelectorAll('div.table-list-item')` depuis le code source complet
		- Retourne un tableau JSON de chaînes HTML (outerHTML de chaque item)
		- Une fois le HTML extrait, parse-le pour construire le rapport DeeptechCompaniesReport
		- Si une info manque, laisse le champ null, mais ne l'invente pas
		- CRITIQUE: Ne visite JAMAIS les pages individuelles des entreprises - extrais toutes les informations depuis les items de la liste
		- CRITIQUE SÉRIALISATION: Lorsque tu appelles `done`, assure-toi que `source_url` est une chaîne de caractères (string), pas un objet URL
		- Exemple correct: "source_url": "{source_url}"
		"""
	).strip()


async def _accept_cookies(page) -> bool:
	"""Accept cookies banner if present. Returns True if cookies were accepted or no banner found."""
	
	print("🍪 Vérification de la bannière de cookies...")
	
	# Wait longer for the cookie banner to appear
	await asyncio.sleep(3)
	
	# First, check what cookie-related elements exist
	debug_info_raw = await page.evaluate(
		"""() => {
			const info = {
				buttons: document.querySelectorAll('button').length,
				links: document.querySelectorAll('a').length,
				cookieElements: document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"]').length,
				visibleBanners: 0,
				buttonTexts: []
			};
			
			// Check for visible banners
			const banners = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"], [class*="banner"]');
			for (const banner of banners) {
				const style = window.getComputedStyle(banner);
				if (style.display !== 'none' && style.visibility !== 'hidden' && parseFloat(style.opacity) > 0) {
					info.visibleBanners++;
				}
			}
			
			// Get text of first 10 buttons
			const buttons = Array.from(document.querySelectorAll('button')).slice(0, 10);
			info.buttonTexts = buttons.map(btn => btn.textContent?.trim() || '').filter(t => t);
			
			return JSON.stringify(info);
		}"""
	)
	
	# Parse JSON if it's a string
	try:
		if isinstance(debug_info_raw, str):
			debug_info = json.loads(debug_info_raw)
		else:
			debug_info = debug_info_raw
	except (json.JSONDecodeError, TypeError):
		debug_info = {}
	
	print(f"   🔍 Debug cookies - Boutons: {debug_info.get('buttons', 'N/A')}, Bannières visibles: {debug_info.get('visibleBanners', 'N/A')}")
	if debug_info.get('buttonTexts'):
		print(f"   📝 Textes des boutons: {debug_info.get('buttonTexts')[:5]}")
	
	# Try multiple strategies to find and click the accept button
	strategies = [
		# Strategy 1: Look for buttons with common accept text (more comprehensive)
		"""() => {
			const buttons = Array.from(document.querySelectorAll('button, a[role="button"], div[role="button"], span[role="button"]'));
			const acceptButton = buttons.find(btn => {
				const text = (btn.textContent || btn.innerText || '').toLowerCase().trim();
				const ariaLabel = (btn.getAttribute('aria-label') || '').toLowerCase();
				const title = (btn.getAttribute('title') || '').toLowerCase();
				const combined = text + ' ' + ariaLabel + ' ' + title;
				
				return combined.includes('accept') || 
				       combined.includes('accepter') || 
				       combined.includes('j accepte') ||
				       combined.includes('tout accepter') ||
				       combined.includes('accepter tout') ||
				       combined.includes('ok') ||
				       combined.includes('valider') ||
				       btn.getAttribute('data-test')?.toLowerCase().includes('cookie') ||
				       btn.getAttribute('data-testid')?.toLowerCase().includes('cookie') ||
				       btn.getAttribute('id')?.toLowerCase().includes('cookie') ||
				       btn.getAttribute('id')?.toLowerCase().includes('accept') ||
				       btn.getAttribute('class')?.toLowerCase().includes('accept') ||
				       btn.getAttribute('class')?.toLowerCase().includes('cookie');
			});
			if (acceptButton) {
				acceptButton.scrollIntoView({ behavior: 'smooth', block: 'center' });
				setTimeout(() => acceptButton.click(), 100);
				return JSON.stringify({ clicked: true, text: acceptButton.textContent?.trim() || 'no text' });
			}
			return JSON.stringify({ clicked: false });
		}""",
		# Strategy 2: Look for cookie consent overlay/div (more comprehensive)
		"""() => {
			const selectors = [
				'[class*="cookie"]',
				'[id*="cookie"]',
				'[class*="consent"]',
				'[id*="consent"]',
				'[class*="banner"]',
				'[id*="banner"]',
				'[class*="gdpr"]',
				'[id*="gdpr"]',
			];
			for (const selector of selectors) {
				const overlays = document.querySelectorAll(selector);
				for (const overlay of overlays) {
					const style = window.getComputedStyle(overlay);
					if (style.display !== 'none' && style.visibility !== 'hidden' && parseFloat(style.opacity) > 0) {
						const buttons = overlay.querySelectorAll('button, a[role="button"], div[role="button"]');
						for (const btn of buttons) {
							const text = (btn.textContent || btn.innerText || '').toLowerCase();
							if (text.includes('accept') || text.includes('accepter') || text.includes('j accepte') || text.includes('ok') || text.includes('valider')) {
								btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
								setTimeout(() => btn.click(), 100);
								return JSON.stringify({ clicked: true, text: btn.textContent?.trim() || 'no text' });
							}
						}
						// If no specific button found, try first button
						if (buttons.length > 0) {
							buttons[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
							setTimeout(() => buttons[0].click(), 100);
							return JSON.stringify({ clicked: true, text: 'first button in overlay' });
						}
					}
				}
			}
			return JSON.stringify({ clicked: false });
		}""",
		# Strategy 3: Look for any visible modal/overlay and try to find accept button
		"""() => {
			const modals = document.querySelectorAll('[class*="modal"], [class*="overlay"], [class*="popup"], [role="dialog"]');
			for (const modal of modals) {
				const style = window.getComputedStyle(modal);
				if (style.display !== 'none' && style.visibility !== 'hidden' && parseFloat(style.opacity) > 0) {
					const buttons = modal.querySelectorAll('button');
					for (const btn of buttons) {
						const text = (btn.textContent || '').toLowerCase();
						if (text.includes('accept') || text.includes('accepter') || text.includes('ok')) {
							btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
							setTimeout(() => btn.click(), 100);
							return JSON.stringify({ clicked: true, text: btn.textContent?.trim() || 'no text' });
						}
					}
				}
			}
			return JSON.stringify({ clicked: false });
		}""",
		# Strategy 4: Try clicking on any button that might be related to cookies (last resort)
		"""() => {
			const allButtons = Array.from(document.querySelectorAll('button'));
			for (const btn of allButtons) {
				const text = (btn.textContent || '').toLowerCase().trim();
				const id = (btn.id || '').toLowerCase();
				const className = (btn.className || '').toLowerCase();
				if (text.length < 50 && (text.includes('accept') || text.includes('accepter') || text.includes('ok') || id.includes('accept') || className.includes('accept'))) {
					const rect = btn.getBoundingClientRect();
					if (rect.width > 0 && rect.height > 0) {
						btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
						setTimeout(() => btn.click(), 100);
						return JSON.stringify({ clicked: true, text: btn.textContent?.trim() || 'no text' });
					}
				}
			}
			return JSON.stringify({ clicked: false });
		}""",
	]
	
	for i, strategy in enumerate(strategies, 1):
		try:
			print(f"   🔄 Essai stratégie {i}...")
			result_raw = await page.evaluate(strategy)
			await asyncio.sleep(1)  # Wait for click to register
			
			# Parse JSON if it's a string
			try:
				if isinstance(result_raw, str):
					result = json.loads(result_raw)
				else:
					result = result_raw
			except (json.JSONDecodeError, TypeError):
				result = {}
			
			if result and result.get('clicked'):
				button_text = result.get('text', 'unknown')
				print(f"   ✅ Bannière de cookies acceptée (stratégie {i}) - Bouton: '{button_text}'")
				# Wait for the banner to disappear
				await asyncio.sleep(3)
				# Verify the banner is gone
				banner_gone = await page.evaluate(
					"""() => {
						const banners = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"], [class*="banner"]');
						for (const banner of banners) {
							const style = window.getComputedStyle(banner);
							if (style.display !== 'none' && style.visibility !== 'hidden' && parseFloat(style.opacity) > 0) {
								return false;
							}
						}
						return true;
					}"""
				)
				if banner_gone:
					print("   ✅ Bannière de cookies confirmée comme fermée")
				else:
					print("   ⚠️ Bannière peut-être encore visible, mais clic effectué")
				return True
			else:
				print(f"   ❌ Stratégie {i} n'a pas trouvé de bouton")
		except Exception as e:
			print(f"   ⚠️ Erreur avec la stratégie {i}: {e}")
			import traceback
			traceback.print_exc()
			continue
	
	# Final check if cookie banner is still visible
	banner_visible = await page.evaluate(
		"""() => {
			const banners = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"], [class*="banner"]');
			for (const banner of banners) {
				const style = window.getComputedStyle(banner);
				if (style.display !== 'none' && style.visibility !== 'hidden' && parseFloat(style.opacity) > 0) {
					return true;
				}
			}
			return false;
		}"""
	)
	
	if not banner_visible:
		print("   ✅ Aucune bannière de cookies détectée (ou déjà fermée)")
		return True
	else:
		print("   ⚠️ Bannière de cookies toujours visible mais impossible de cliquer")
		print("   💡 La page peut nécessiter une interaction manuelle pour accepter les cookies")
		return False


async def _extract_companies_with_scroll(page, source_url: str, max_companies: int) -> list[DeeptechCompany]:
	"""Extract companies by scrolling through virtual-list and extracting at each step."""
	
	all_companies: list[DeeptechCompany] = []
	seen_names: set[str] = set()
	
	print("🔍 Détection du conteneur virtual-list/table-list...")
	
	# Try to find the scrollable container - find the one that contains table-list-item
	container_info_raw = await page.evaluate(
		"""() => {
			// First, find an item to get its parent container
			const firstItem = document.querySelector('div.table-list-item');
			if (!firstItem) {
				return JSON.stringify({ found: false, reason: 'no items found' });
			}
			
			// Find the scrollable parent that contains the items
			let parent = firstItem.parentElement;
			let scrollableContainer = null;
			let depth = 0;
			const maxDepth = 10;
			
			while (parent && depth < maxDepth) {
				const style = window.getComputedStyle(parent);
				const scrollHeight = parent.scrollHeight;
				const clientHeight = parent.clientHeight;
				const overflowY = style.overflowY || style.overflow;
				
				// Check if this parent is scrollable and contains the items
				if ((overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay') && 
				    scrollHeight > clientHeight) {
					// Verify it contains table-list-items
					const itemsInContainer = parent.querySelectorAll('div.table-list-item').length;
					if (itemsInContainer > 0) {
						scrollableContainer = parent;
						break;
					}
				}
				parent = parent.parentElement;
				depth++;
			}
			
			if (scrollableContainer) {
				return JSON.stringify({
					found: true,
					selector: scrollableContainer.className ? '.' + scrollableContainer.className.split(' ')[0] : 'parent',
					tagName: scrollableContainer.tagName.toLowerCase(),
					scrollHeight: scrollableContainer.scrollHeight,
					clientHeight: scrollableContainer.clientHeight,
					isScrollable: scrollableContainer.scrollHeight > scrollableContainer.clientHeight,
					itemsCount: scrollableContainer.querySelectorAll('div.table-list-item').length
				});
			}
			
			// Fallback: try common selectors
			const selectors = [
				'[class*="virtual-list"]',
				'[class*="table-list"]',
				'[class*="scroll"]',
				'[style*="overflow"]',
			];
			for (const selector of selectors) {
				const containers = document.querySelectorAll(selector);
				for (const container of containers) {
					const itemsInContainer = container.querySelectorAll('div.table-list-item').length;
					if (itemsInContainer > 0) {
						const scrollHeight = container.scrollHeight;
						const clientHeight = container.clientHeight;
						if (scrollHeight > clientHeight) {
							return JSON.stringify({
								found: true,
								selector: selector,
								tagName: container.tagName.toLowerCase(),
								scrollHeight: scrollHeight,
								clientHeight: clientHeight,
								isScrollable: true,
								itemsCount: itemsInContainer
							});
						}
					}
				}
			}
			
			return JSON.stringify({ found: false, reason: 'no scrollable container found' });
		}"""
	)
	
	# Parse JSON if it's a string
	try:
		if isinstance(container_info_raw, str):
			container_info = json.loads(container_info_raw)
		else:
			container_info = container_info_raw
	except (json.JSONDecodeError, TypeError):
		container_info = {}
	
	scroll_container_element = None
	scroll_container_selector = None
	if container_info and container_info.get('found'):
		# Try to get the actual element reference
		scroll_container_element = await page.evaluate(
			"""() => {
				const firstItem = document.querySelector('div.table-list-item');
				if (!firstItem) return null;
				
				let parent = firstItem.parentElement;
				let depth = 0;
				while (parent && depth < 10) {
					const style = window.getComputedStyle(parent);
					const overflowY = style.overflowY || style.overflow;
					if ((overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay') && 
					    parent.scrollHeight > parent.clientHeight) {
						const itemsInContainer = parent.querySelectorAll('div.table-list-item').length;
						if (itemsInContainer > 0) {
							return parent.className || parent.id || parent.tagName;
						}
					}
					parent = parent.parentElement;
					depth++;
				}
				return null;
			}"""
		)
		
		scroll_container_selector = container_info.get('selector')
		items_count = container_info.get('itemsCount', 0)
		print(f"   ✅ Conteneur scrollable trouvé: {scroll_container_selector}")
		print(f"   📏 Hauteur scrollable: {container_info.get('scrollHeight')}px, hauteur visible: {container_info.get('clientHeight')}px")
		print(f"   📊 Items dans le conteneur: {items_count}")
	else:
		print("   ⚠️ Conteneur spécifique non trouvé, recherche du conteneur parent des items...")
	
	# Extract initial companies
	print("📥 Extraction initiale des entreprises...")
	
	# First, check what's actually in the DOM
	debug_info = await page.evaluate(
		"""() => {
			const selectors = [
				'div.table-list-item',
				'.table-list-item',
				'[class*="table-list-item"]',
				'div[class*="list-item"]',
				'[data-testid*="company"]',
				'[data-testid*="item"]',
			];
			const results = {};
			for (const selector of selectors) {
				try {
					const items = document.querySelectorAll(selector);
					results[selector] = items.length;
				} catch (e) {
					results[selector] = 'error: ' + e.message;
				}
			}
			// Also check for virtual-list or table-list containers
			const containers = document.querySelectorAll('[class*="virtual-list"], [class*="table-list"], [class*="list"]');
			results.containers = containers.length;
			return results;
		}"""
	)
	print(f"   🔍 Debug - Éléments trouvés dans le DOM: {debug_info}")
	
	iteration = 0
	no_new_companies_count = 0
	max_iterations = 200  # Safety limit
	
	# Use a composite key (name + URL) to track companies more reliably
	seen_companies: set[tuple[str, str | None]] = set()
	
	while iteration < max_iterations:
		iteration += 1
		
		# Extract current companies from DOM - try multiple selectors
		try:
			# Try primary selector first
			html_items_json = await page.evaluate(
				"() => { const items = Array.from(document.querySelectorAll('div.table-list-item')); return JSON.stringify(items.map(item => item.outerHTML)); }"
			)
			
			# If no items found, try alternative selectors
			if not html_items_json or html_items_json == '[]' or not html_items_json.strip():
				print(f"   ⚠️ Aucun item trouvé avec 'div.table-list-item' à l'itération {iteration}, essai de sélecteurs alternatifs...")
				
				# Try alternative selectors
				alternative_selectors = [
					'.table-list-item',
					'[class*="table-list-item"]',
					'div[class*="list-item"]',
					'[data-testid*="company"]',
				]
				
				for alt_selector in alternative_selectors:
					try:
						html_items_json = await page.evaluate(
							f"() => {{ const items = Array.from(document.querySelectorAll('{alt_selector}')); return JSON.stringify(items.map(item => item.outerHTML)); }}"
						)
						if html_items_json and html_items_json != '[]' and html_items_json.strip():
							print(f"   ✅ Items trouvés avec le sélecteur alternatif: {alt_selector}")
							break
					except Exception as e:
						print(f"   ⚠️ Erreur avec le sélecteur {alt_selector}: {e}")
						continue
			
			if not html_items_json or html_items_json == '[]' or not html_items_json.strip():
				print(f"   ⚠️ Aucun item trouvé à l'itération {iteration} (tous les sélecteurs testés)")
				no_new_companies_count += 1
				if no_new_companies_count >= 5:
					print("   ✅ Plus d'entreprises à extraire (5 itérations sans nouvelles données)")
					break
				await asyncio.sleep(1)
				continue
			
			# Parse HTML items
			try:
				html_items = json.loads(html_items_json)
				print(f"   📦 {len(html_items)} items HTML extraits à l'itération {iteration}")
			except json.JSONDecodeError as e:
				print(f"   ⚠️ Erreur de parsing JSON: {e}")
				print(f"   📄 Premiers caractères: {html_items_json[:200] if html_items_json else 'None'}")
				no_new_companies_count += 1
				if no_new_companies_count >= 5:
					break
				await asyncio.sleep(1)
				continue
			
			current_batch = _parse_html_sections(html_items, source_url)
			
			if not current_batch or not current_batch.companies:
				print(f"   ⚠️ Aucune entreprise parsée à l'itération {iteration}")
				if iteration == 1:
					# On first iteration, show a sample of the HTML to debug
					if html_items and len(html_items) > 0:
						sample_html = html_items[0][:500] if len(html_items[0]) > 500 else html_items[0]
						print(f"   🔍 Échantillon HTML du premier item: {sample_html}...")
				no_new_companies_count += 1
				if no_new_companies_count >= 5:
					print("   ✅ Plus d'entreprises à extraire (5 itérations sans parsing)")
					break
				await asyncio.sleep(1)
				continue
			
			# Count new companies using composite key (name + URL)
			new_companies_count = 0
			for company in current_batch.companies:
				# Use composite key: (name, company_url) for more reliable tracking
				company_key = (company.name or '', company.company_url or '')
				if company.name and company_key not in seen_companies:
					all_companies.append(company)
					seen_companies.add(company_key)
					seen_names.add(company.name)  # Keep for backward compatibility
					new_companies_count += 1
			
			print(f"   📊 Itération {iteration}: {len(current_batch.companies)} entreprises trouvées, {new_companies_count} nouvelles (total: {len(all_companies)})")
			
			# Check if we've reached max_companies
			if max_companies < 10000 and len(all_companies) >= max_companies:
				print(f"   ✅ Limite de {max_companies} entreprises atteinte")
				break
			
			# If no new companies, increment counter
			if new_companies_count == 0:
				no_new_companies_count += 1
				# If we've scrolled but no new companies, try scrolling more aggressively
				if no_new_companies_count <= 3:
					print(f"   🔄 Aucune nouvelle entreprise ({no_new_companies_count}/10), scroll plus agressif dans le conteneur...")
					# Force scroll to absolute bottom of the container multiple times
					for scroll_attempt in range(5):  # Increased from 3 to 5
						await page.evaluate(
							"""() => {
								// Find the scrollable container
								const firstItem = document.querySelector('div.table-list-item');
								if (firstItem) {
									let parent = firstItem.parentElement;
									let depth = 0;
									while (parent && depth < 10) {
										const style = window.getComputedStyle(parent);
										const overflowY = style.overflowY || style.overflow;
										if ((overflowY === 'auto' || overflowY === 'scroll') && 
										    parent.scrollHeight > parent.clientHeight) {
											const itemsInContainer = parent.querySelectorAll('div.table-list-item').length;
											if (itemsInContainer > 0) {
												// Scroll to absolute bottom
												parent.scrollTop = parent.scrollHeight;
												parent.scrollTo({ top: parent.scrollHeight, behavior: 'auto' });
												// Also try instant scroll
												setTimeout(() => {
													parent.scrollTop = parent.scrollHeight;
												}, 100);
												parent.dispatchEvent(new Event('scroll', { bubbles: true }));
												parent.dispatchEvent(new Event('scrollend', { bubbles: true }));
												return 'container_scrolled';
											}
										}
										parent = parent.parentElement;
										depth++;
									}
								}
								// Fallback: scroll page to absolute bottom
								window.scrollTo(0, document.documentElement.scrollHeight);
								document.documentElement.scrollTop = document.documentElement.scrollHeight;
								document.body.scrollTop = document.body.scrollHeight;
								window.dispatchEvent(new Event('scroll'));
								window.dispatchEvent(new Event('scrollend'));
								return 'page_scrolled';
							}"""
						)
						await asyncio.sleep(2)
					print(f"   ✅ Scroll agressif effectué {5} fois")
					# Wait longer after aggressive scrolling
					await asyncio.sleep(3)
				elif no_new_companies_count >= 10:  # Increased from 5 to 10
					# Before giving up, do one final aggressive scroll to the absolute bottom
					print(f"   🔄 Dernière tentative: scroll jusqu'au bas absolu...")
					for final_attempt in range(5):
						await page.evaluate(
							"""() => {
								// Find the scrollable container and scroll to absolute bottom
								const firstItem = document.querySelector('div.table-list-item');
								if (firstItem) {
									let parent = firstItem.parentElement;
									let depth = 0;
									while (parent && depth < 10) {
										const style = window.getComputedStyle(parent);
										const overflowY = style.overflowY || style.overflow;
										if ((overflowY === 'auto' || overflowY === 'scroll') && 
										    parent.scrollHeight > parent.clientHeight) {
											const itemsInContainer = parent.querySelectorAll('div.table-list-item').length;
											if (itemsInContainer > 0) {
												// Force scroll to absolute bottom
												parent.scrollTop = parent.scrollHeight;
												return 'container_scrolled';
											}
										}
										parent = parent.parentElement;
										depth++;
									}
								}
								// Fallback: scroll page to absolute bottom
								window.scrollTo(0, document.documentElement.scrollHeight);
								document.documentElement.scrollTop = document.documentElement.scrollHeight;
								document.body.scrollTop = document.body.scrollHeight;
								return 'page_scrolled';
							}"""
						)
						await asyncio.sleep(1)
					await asyncio.sleep(5)  # Wait longer for final load
					print("   ✅ Plus d'entreprises à extraire (10 itérations sans nouvelles entreprises)")
					break
			else:
				no_new_companies_count = 0
			
			# Scroll down to load more content - use more aggressive scrolling
			# More aggressive scrolling strategy - find and scroll the actual container
			scroll_info = await page.evaluate(
				"""() => {
					// Find the scrollable container that contains table-list-items
					const firstItem = document.querySelector('div.table-list-item');
					if (!firstItem) {
						return JSON.stringify({ success: false, reason: 'no items' });
					}
					
					let scrollableContainer = null;
					let parent = firstItem.parentElement;
					let depth = 0;
					
					while (parent && depth < 10) {
						const style = window.getComputedStyle(parent);
						const overflowY = style.overflowY || style.overflow;
						const scrollHeight = parent.scrollHeight;
						const clientHeight = parent.clientHeight;
						
						if ((overflowY === 'auto' || overflowY === 'scroll' || overflowY === 'overlay') && 
						    scrollHeight > clientHeight) {
							const itemsInContainer = parent.querySelectorAll('div.table-list-item').length;
							if (itemsInContainer > 0) {
								scrollableContainer = parent;
								break;
							}
						}
						parent = parent.parentElement;
						depth++;
					}
					
					if (scrollableContainer) {
						const previousScrollTop = scrollableContainer.scrollTop;
						const scrollHeight = scrollableContainer.scrollHeight;
						const clientHeight = scrollableContainer.clientHeight;
						
						// More aggressive scrolling: scroll by 95% of viewport or to near bottom
						const scrollAmount = Math.max(clientHeight * 0.95, 800);
						const distanceToBottom = scrollHeight - scrollableContainer.scrollTop - clientHeight;
						
						// If we're close to bottom, scroll all the way to bottom
						let newScrollTop;
						if (distanceToBottom < clientHeight * 2) {
							// Close to bottom, scroll all the way
							newScrollTop = scrollHeight;
						} else {
							// Scroll by large amount
							newScrollTop = Math.min(scrollableContainer.scrollTop + scrollAmount, scrollHeight);
						}
						
						scrollableContainer.scrollTop = newScrollTop;
						// Also try smooth scroll
						scrollableContainer.scrollTo({ top: newScrollTop, behavior: 'auto' });
						
						// Trigger scroll events multiple times
						scrollableContainer.dispatchEvent(new Event('scroll', { bubbles: true }));
						scrollableContainer.dispatchEvent(new Event('scrollend', { bubbles: true }));
						
						return JSON.stringify({
							success: true,
							previousScrollTop: previousScrollTop,
							scrollTop: scrollableContainer.scrollTop,
							scrollHeight: scrollHeight,
							clientHeight: clientHeight,
							scrolled: scrollableContainer.scrollTop > previousScrollTop,
							containerTag: scrollableContainer.tagName,
							distanceToBottom: scrollHeight - scrollableContainer.scrollTop - clientHeight
						});
					}
					
					// Fallback: scroll page
					const previousScrollTop = window.pageYOffset || document.documentElement.scrollTop;
					const scrollHeight = document.documentElement.scrollHeight;
					const clientHeight = window.innerHeight;
					const scrollAmount = Math.max(clientHeight * 0.9, 500);
					const newScrollTop = Math.min(previousScrollTop + scrollAmount, scrollHeight);
					
					window.scrollTo(0, newScrollTop);
					document.documentElement.scrollTop = newScrollTop;
					document.body.scrollTop = newScrollTop;
					window.scrollTo({ top: newScrollTop, behavior: 'smooth' });
					
					return JSON.stringify({
						success: true,
						previousScrollTop: previousScrollTop,
						scrollTop: window.pageYOffset || document.documentElement.scrollTop,
						scrollHeight: scrollHeight,
						clientHeight: clientHeight,
						scrolled: (window.pageYOffset || document.documentElement.scrollTop) > previousScrollTop,
						containerTag: 'page'
					});
				}"""
			)
			
			# Parse scroll info
			try:
				if isinstance(scroll_info, str):
					scroll_result = json.loads(scroll_info)
				else:
					scroll_result = scroll_info
			except (json.JSONDecodeError, TypeError):
				scroll_result = {}
			
			# Wait longer for new content to load (virtual-list needs more time)
			await asyncio.sleep(4)  # Increased from 3 to 4
			
			# Try to trigger loading by scrolling a bit more and triggering events
			await page.evaluate(
				"""() => {
					// Small additional scroll to trigger lazy loading
					window.scrollBy(0, 100);
					document.documentElement.scrollTop += 100;
					// Trigger scroll events
					window.dispatchEvent(new Event('scroll'));
					window.dispatchEvent(new Event('scrollend'));
					// Also try touchmove for mobile-like events
					window.dispatchEvent(new TouchEvent('touchmove', { bubbles: true }));
				}"""
			)
			await asyncio.sleep(3)  # Increased from 2 to 3
			
			# Note: We don't check DOM item count anymore because virtual lists recycle elements
			# Instead, we rely on tracking new companies by name+URL in the next iteration
			
		except json.JSONDecodeError as e:
			print(f"   ⚠️ Erreur de parsing JSON à l'itération {iteration}: {e}")
			no_new_companies_count += 1
			if no_new_companies_count >= 3:
				break
			await asyncio.sleep(1)
		except Exception as e:
			print(f"   ⚠️ Erreur à l'itération {iteration}: {e}")
			no_new_companies_count += 1
			if no_new_companies_count >= 3:
				break
			await asyncio.sleep(1)
	
	print(f"✅ Extraction terminée: {len(all_companies)} entreprises extraites en {iteration} itérations")
	return all_companies


async def run_deeptech_companies(task_input: DeeptechCompaniesInput) -> DeeptechCompaniesReport | None:
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
			max_completion_tokens=15000,  # 18192 * 5
			add_schema_to_system_prompt=is_gemini,
			dont_force_structured_output=is_gemini,
		)
		extraction_model = os.getenv('PAGE_EXTRACTION_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		page_extraction_llm = ChatOpenAI(
			model=extraction_model,
			timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=20.0),
			max_retries=2,
			max_completion_tokens=15000,  # 18192 * 5
			add_schema_to_system_prompt=True,
			dont_force_structured_output=True,
		)
		print(f"✅ Utilisation de ChatOpenAI avec le modèle: {model_name}")
		print(f"✅ Utilisation de ChatOpenAI pour l'extraction avec le modèle: {extraction_model}")
		if is_gemini:
			print("   ⚠️  Mode Gemini détecté: utilisation du schéma dans le prompt système")

	print("🌐 Création du navigateur...")
	browser = Browser(headless=False)
	await browser.start()

	try:
		# Navigate to the companies URL in a new tab
		source_url = task_input.url
		print(f"📍 Navigation vers: {source_url} (nouvel onglet)")
		# Use NavigateToUrlEvent with new_tab=True to force opening in a new tab
		navigate_event = NavigateToUrlEvent(url=source_url, new_tab=True)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event

		# Get the current page after navigation
		page = await browser.get_current_page()
		if not page:
			# Fallback: try new_page if get_current_page doesn't work
			page = await browser.new_page(source_url)

		await asyncio.sleep(5)  # Wait for initial page load

		# Accept cookies before extraction (CRITICAL)
		cookies_accepted = await _accept_cookies(page)
		if not cookies_accepted:
			print("   ⚠️ Impossible d'accepter les cookies, continuation quand même...")
		else:
			# Wait a bit more after accepting cookies to ensure page is ready
			await asyncio.sleep(2)

		# Extract companies with scroll strategy for virtual-list
		print("📥 Extraction des entreprises avec scroll progressif...")
		current_page = await browser.get_current_page()
		if not current_page:
			raise RuntimeError("No current page available")

		# Extract companies using scroll strategy
		extraction_successful = False
		try:
			all_companies = await _extract_companies_with_scroll(
				current_page,
				str(task_input.url),
				task_input.max_companies
			)
			
			if all_companies and len(all_companies) > 0:
				# Limit to max_companies if needed
				if task_input.max_companies < 10000 and len(all_companies) > task_input.max_companies:
					all_companies = all_companies[:task_input.max_companies]
				
				report = DeeptechCompaniesReport(
					source_url=AnyHttpUrl(str(task_input.url)),
					companies=all_companies,
				)
				extraction_successful = True
				return _sanitize_report(report)
			else:
				print("   ⚠️ Aucune entreprise extraite")
				# Before giving up, try one more time with a longer wait
				print("   🔄 Dernière tentative après attente supplémentaire...")
				await asyncio.sleep(5)
				all_companies = await _extract_companies_with_scroll(
					current_page,
					str(task_input.url),
					task_input.max_companies
				)
				if all_companies and len(all_companies) > 0:
					if task_input.max_companies < 10000 and len(all_companies) > task_input.max_companies:
						all_companies = all_companies[:task_input.max_companies]
					report = DeeptechCompaniesReport(
						source_url=AnyHttpUrl(str(task_input.url)),
						companies=all_companies,
					)
					extraction_successful = True
					return _sanitize_report(report)
		except Exception as e:
			print(f"   ⚠️ Erreur lors de l'extraction avec scroll: {e}")
			import traceback
			traceback.print_exc()

		# Only use agent as fallback if direct extraction completely failed
		if not extraction_successful:
			print("❌ Extraction avec scroll échouée. Le contenu n'a peut-être pas été chargé correctement.")
			print("   💡 Vérifiez dans le navigateur que la page s'est bien chargée et que les entreprises sont visibles.")
			return _fallback_report(str(task_input.url), "Extraction avec scroll échouée: aucune entreprise trouvée.")

		# This should never be reached, but just in case:
		print("🤖 Utilisation de l'agent comme dernier recours...")
		agent = Agent(
			task=build_task(task_input),
			llm=llm,
			page_extraction_llm=page_extraction_llm,
			browser=browser,  # Use the browser we already set up
			output_model_schema=DeeptechCompaniesReport,
			use_vision='auto',  # Use 'auto' to reduce screenshot frequency and avoid timeouts
			vision_detail_level='auto',  # Use 'auto' instead of 'high' to reduce processing time
			step_timeout=300,
			llm_timeout=180,
			max_failures=5,
			max_history_items=10,
			directly_open_url=False,  # Don't navigate again, we already did
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
				report = DeeptechCompaniesReport.model_validate_json(final_result)
				if not agent_successful:
					print("⚠️  Données récupérées depuis le résultat final malgré l'échec de l'agent.")
				return _sanitize_report(report)
			except ValidationError:
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if json_match:
					try:
						report = DeeptechCompaniesReport.model_validate_json(json_match.group(1))
						if not agent_successful:
							print("⚠️  Données récupérées depuis le résultat final (markdown) malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError:
						pass

		# Try to extract from evaluate actions (HTML sections)
		all_extracted_companies: list[DeeptechCompany] = []
		seen_names: set[str] = set()

		# Look for evaluate actions and their results
		action_results_list = history.action_results()
		model_actions_list = history.model_actions()

		# Match actions with their results - evaluate results are in extracted_content
		for i, action_dict in enumerate(model_actions_list):
			if 'evaluate' in action_dict:
				# Try to get the result for this action
				if i < len(action_results_list):
					result = action_results_list[i]
					# Evaluate results are stored in extracted_content as JSON string
					if result and result.extracted_content:
						try:
							# Try to parse as JSON first (evaluate returns JSON-stringified arrays)
							content_str = result.extracted_content.strip()
							# Remove any markdown code blocks if present
							if content_str.startswith('```'):
								# Extract content between code blocks
								parts = content_str.split('```')
								if len(parts) >= 3:
									content_str = parts[1].strip()
									# Remove language identifier if present
									if '\n' in content_str:
										content_str = content_str.split('\n', 1)[1]
									else:
										content_str = parts[2].strip() if len(parts) > 2 else content_str

							evaluate_result = json.loads(content_str)
							html_report = _parse_html_sections(evaluate_result, str(task_input.url))
							if html_report and html_report.companies:
								for company in html_report.companies:
									if company.name and company.name not in seen_names:
										all_extracted_companies.append(company)
										seen_names.add(company.name)
						except (json.JSONDecodeError, TypeError) as e:
							# If not JSON, try parsing as string directly
							try:
								html_report = _parse_html_sections(result.extracted_content, str(task_input.url))
								if html_report and html_report.companies:
									for company in html_report.companies:
										if company.name and company.name not in seen_names:
											all_extracted_companies.append(company)
											seen_names.add(company.name)
							except Exception:
								# Skip if parsing fails
								pass

		# Try to extract from action results (especially extract and evaluate actions)
		extracted_contents = history.extracted_content()

		# Process extractions in chronological order (first to last) to preserve company order
		for content in extracted_contents:
			if not content:
				continue

			# Try parsing as JSON first (evaluate returns JSON-stringified arrays)
			try:
				content_str = content.strip()
				# Remove any markdown code blocks if present
				if content_str.startswith('```'):
					# Extract content between code blocks
					parts = content_str.split('```')
					if len(parts) >= 3:
						content_str = parts[1].strip()
						# Remove language identifier if present
						if '\n' in content_str:
							content_str = content_str.split('\n', 1)[1]
						else:
							content_str = parts[2].strip() if len(parts) > 2 else content_str

				evaluate_result = json.loads(content_str)
				html_report = _parse_html_sections(evaluate_result, str(task_input.url))
				if html_report and html_report.companies:
					for company in html_report.companies:
						if company.name and company.name not in seen_names:
							all_extracted_companies.append(company)
							seen_names.add(company.name)
					continue
			except (json.JSONDecodeError, TypeError):
				pass

			# Try parsing as HTML sections (from evaluate action returning raw HTML)
			html_report = _parse_html_sections(content, str(task_input.url))
			if html_report and html_report.companies:
				for company in html_report.companies:
					if company.name and company.name not in seen_names:
						all_extracted_companies.append(company)
						seen_names.add(company.name)
				continue

		# If we found companies from HTML parsing, return them
		if all_extracted_companies:
			if task_input.max_companies < 10000:
				all_extracted_companies = all_extracted_companies[:task_input.max_companies]

			report = DeeptechCompaniesReport(
				source_url=AnyHttpUrl(str(task_input.url)),
				companies=all_extracted_companies,
			)
			if not agent_successful:
				print(f"⚠️  Données récupérées depuis les résultats d'extraction malgré l'échec de l'agent. {len(all_extracted_companies)} entreprises trouvées.")
			return _sanitize_report(report)

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
						report = DeeptechCompaniesReport.model_validate(data_copy)
						if not agent_successful:
							print("⚠️  Données récupérées depuis l'action 'done' malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError as e:
						try:
							json_str = json.dumps(data_copy, default=str)
							report = DeeptechCompaniesReport.model_validate_json(json_str)
							if not agent_successful:
								print("⚠️  Données récupérées depuis l'action 'done' (après conversion JSON) malgré l'échec de l'agent.")
							return _sanitize_report(report)
						except (ValidationError, json.JSONDecodeError):
							pass

		# If we get here, we couldn't extract any data
		if not agent_successful:
			print("❌ Impossible d'extraire les données malgré plusieurs tentatives.")
		return _fallback_report(str(task_input.url), "L'agent a été interrompu avant de finaliser le JSON.")

	finally:
		# Close browser to free resources
		print("🧹 Fermeture du navigateur...")
		try:
			await browser.kill()
			print("✅ Navigateur fermé")
		except Exception as e:
			print(f"⚠️  Erreur lors de la fermeture du navigateur: {e}")


def parse_arguments() -> DeeptechCompaniesInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Extrait les entreprises depuis une page Observatoire Deeptech')
	parser.add_argument(
		'url',
		help='URL de la page des entreprises (ex: https://observatoire.lesdeeptech.fr/companies.startups/...)',
	)
	parser.add_argument(
		'--max-companies',
		type=int,
		default=1000,
		help='Nombre maximal d\'entreprises à extraire (par défaut: 1000)',
	)
	parser.add_argument(
		'--output',
		default='deeptech_companies.json',
		help='Chemin du fichier JSON résultat (par défaut: ./deeptech_companies.json)',
	)
	args = parser.parse_args()
	return DeeptechCompaniesInput(url=args.url, max_companies=args.max_companies, output_path=Path(args.output))


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		source_url = task_input.url
		print(f"🚀 Démarrage de l'agent pour l'URL: {source_url}")
		print(f"📊 Nombre max d'entreprises: {task_input.max_companies}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")

		result = await run_deeptech_companies(task_input)

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

