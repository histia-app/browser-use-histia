"""
Agent designed to extract products from Product Hunt leaderboard pages.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser

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
from browser_use.browser.events import NavigateToUrlEvent, ScrollEvent
from examples.histia import print_llm_usage_summary


class ProductHuntLeaderboardInput(BaseModel):
	"""User-provided parameters for the Product Hunt leaderboard task."""

	date: str = Field(..., description='Date for the leaderboard in format YYYY-MM-DD or YYYY/MM/DD (e.g., 2025-11-18)')
	max_products: int = Field(
		1000,
		ge=1,
		le=10000,
		description='Maximum number of products to capture from the leaderboard (use a high number like 1000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('product_hunt_leaderboard.json'),
		description='Destination for the JSON list of products',
	)

	@field_validator('date')
	@classmethod
	def validate_date(cls, v: str) -> str:
		"""Validate and normalize date format."""
		# Try to parse the date to ensure it's valid
		for fmt in ['%Y-%m-%d', '%Y/%m/%d']:
			try:
				# Normalize to YYYY-MM-DD format
				date_obj = datetime.strptime(v, fmt)
				return date_obj.strftime('%Y-%m-%d')
			except ValueError:
				continue
		raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD or YYYY/MM/DD")

	@property
	def url(self) -> AnyHttpUrl:
		"""Build the Product Hunt leaderboard URL from the date."""
		# Convert YYYY-MM-DD to YYYY/MM/DD for the URL
		date_parts = self.date.split('-')
		if len(date_parts) == 3:
			url_date = '/'.join(date_parts)
			url_str = f"https://www.producthunt.com/leaderboard/daily/{url_date}/all"
			return AnyHttpUrl(url_str)
		raise ValueError(f"Invalid date format: {self.date}")


class ProductHuntProduct(BaseModel):
	"""Structured information for each Product Hunt product entry."""

	name: str = Field(..., description='Product name exactly as written on the leaderboard')
	producthunt_url: str | None = Field(
		None,
		description='Complete URL to the Product Hunt product page (format: https://www.producthunt.com/products/...)',
	)
	rank: int | None = Field(
		None,
		description='Position/rank in the leaderboard (if available)',
	)
	description: str | None = Field(
		None,
		description='Product tagline/description if available on the leaderboard',
	)
	tags: list[str] = Field(
		default_factory=list,
		description='Product categories/tags visible on the card (e.g., ["Artificial Intelligence", "Productivity"])',
	)
	upvotes: int | None = Field(
		None,
		description='Number of upvotes if visible on the leaderboard card',
	)
	maker: str | None = Field(
		None,
		description='Maker name if visible on the leaderboard card',
	)
	comments_count: int | None = Field(
		None,
		description='Number of comments if visible on the leaderboard card',
	)


class ProductHuntLeaderboardReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='Leaderboard URL that was analysed')
	products: list[ProductHuntProduct] = Field(
		...,
		min_length=1,
		description='Product entries ordered as they appear on the leaderboard',
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


def _fallback_report(source_url: str, reason: str) -> ProductHuntLeaderboardReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	return ProductHuntLeaderboardReport(
		source_url=AnyHttpUrl(source_url),
		products=[
			ProductHuntProduct(
				name='Informations indisponibles',
				producthunt_url=source_url,
				rank=None,
				description=None,
				tags=[],
				upvotes=None,
				maker=None,
				comments_count=None,
			)
		],
	)


def _normalize_producthunt_url(url: str | None, base_url: str) -> str | None:
	"""Convert relative URLs to absolute Product Hunt URLs."""
	if not url:
		return None

	url = url.strip()
	if not url:
		return None

	# If it's already an absolute URL, return as is
	if url.startswith(('http://', 'https://')):
		# Ensure it's a Product Hunt URL
		if 'producthunt.com' in url.lower():
			return url
		return None

	# If it starts with /, make it relative to the base domain
	if url.startswith('/'):
		parsed_base = urlparse(base_url)
		return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"

	# Otherwise, try to resolve relative to base URL
	try:
		resolved = urljoin(base_url, url)
		if 'producthunt.com' in resolved.lower():
			return resolved
		return None
	except Exception:
		return None


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> ProductHuntLeaderboardReport | None:
	"""Parse HTML sections from evaluate action to build ProductHuntLeaderboardReport."""
	
	products: list[ProductHuntProduct] = []
	
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
			section = soup.find('section', {'data-test': re.compile(r'^post-item-')})
			
			if not section:
				continue
			
			# Extract rank from data-test attribute (e.g., "post-item-1039459" -> 1039459)
			rank = None
			data_test = str(section.get('data-test', ''))
			rank_match = re.search(r'post-item-(\d+)', data_test)
			if rank_match:
				try:
					rank = int(rank_match.group(1))
				except ValueError:
					pass
			
			# Extract name and URL from the link
			name = None
			producthunt_url = None
			name_link = section.select_one('a[href^="/products/"]')
			if name_link:
				name = name_link.get_text(strip=True)
				href = str(name_link.get('href', ''))
				if href:
					producthunt_url = _normalize_producthunt_url(href, source_url)
			
			# Extract description (text-secondary div)
			description = None
			desc_div = section.select_one('div.text-secondary, div.text-16.font-normal.text-dark-gray.text-secondary')
			if desc_div:
				description = desc_div.get_text(strip=True)
			
			# Extract tags from topic links
			tags = []
			topic_links = section.select('a[href^="/topics/"]')
			for link in topic_links:
				tag_text = link.get_text(strip=True)
				if tag_text:
					tags.append(tag_text)
			
			# Extract upvotes from vote button
			upvotes = None
			vote_button = section.select_one('button[data-test="vote-button"]')
			if vote_button:
				vote_text = vote_button.get_text(strip=True)
				upvotes_match = re.search(r'(\d+)', vote_text)
				if upvotes_match:
					try:
						upvotes = int(upvotes_match.group(1))
					except ValueError:
						pass
			
			# Extract comments count from comment button (look for button with comment icon)
			comments_count = None
			# Look for button with SVG path containing "M12.25 6.708" (comment icon pattern)
			comment_buttons = section.select('button')
			for button in comment_buttons:
				svg = button.find('svg')
				if svg:
					path = svg.find('path')
					if path and path.get('d'):
						path_d = str(path.get('d', ''))
						# Check if it's a comment icon (contains specific path pattern)
						if 'M12.25 6.708' in path_d or 'M5.833 1.75' in path_d:
							button_text = button.get_text(strip=True)
							comments_match = re.search(r'(\d+)', button_text)
							if comments_match:
								try:
									comments_count = int(comments_match.group(1))
									break
								except ValueError:
									pass
			
			# Maker is not typically visible in the HTML structure provided
			maker = None
			
			# Only add product if we have at least a name
			if name:
				products.append(ProductHuntProduct(
					name=name,
					producthunt_url=producthunt_url,
					rank=rank if rank else (idx + 1),  # Use index as fallback rank
					description=description,
					tags=tags,
					upvotes=upvotes,
					maker=maker,
					comments_count=comments_count,
				))
		except Exception as e:
			# Skip malformed sections
			continue
	
	if products:
		return ProductHuntLeaderboardReport(
			source_url=AnyHttpUrl(source_url),
			products=products,
		)
	return None


def _parse_extracted_markdown(content: str, source_url: str) -> ProductHuntLeaderboardReport | None:
	"""Parse markdown content from extract action to build ProductHuntLeaderboardReport."""

	products: list[ProductHuntProduct] = []
	current_product: dict[str, Any] | None = None

	# First, try to parse markdown tables (common format from Product Hunt leaderboard)
	# Look for table headers with Rank | Name | URL | Description | Tags | Upvotes | Maker | Comments
	table_pattern = re.compile(
		r'\|\s*Rank\s*\|\s*Name\s*\|\s*URL\s*\|\s*Description[^|]*\|\s*Tags[^|]*\|\s*(?:Upvotes|Votes)[^|]*\|\s*(?:Maker|Comments)?[^|]*\|\s*\n'
		r'\|[-\s|:]+\|\s*\n'
		r'((?:\|\s*\d+\s*\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*\n?)+)',
		re.IGNORECASE | re.MULTILINE
	)

	# More flexible pattern that matches any table with Rank column
	flexible_table_pattern = re.compile(
		r'\|\s*Rank\s*\|\s*Name\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*\n'
		r'\|[-\s|:]+\|\s*\n'
		r'((?:\|\s*\d+\s*\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*\n?)+)',
		re.IGNORECASE | re.MULTILINE
	)

	table_match = table_pattern.search(content) or flexible_table_pattern.search(content)
	if table_match:
		table_rows = table_match.group(1).strip().split('\n')
		for row in table_rows:
			if not row.strip() or not row.strip().startswith('|'):
				continue
			# Parse table row: | Rank | Name | URL | Description | Tags | Upvotes | Maker | Comments |
			parts = [p.strip() for p in row.split('|')]
			if len(parts) >= 5:
				try:
					rank_str = parts[1].strip() if len(parts) > 1 else ''
					name = parts[2].strip() if len(parts) > 2 else ''
					url = parts[3].strip() if len(parts) > 3 else ''
					description = parts[4].strip() if len(parts) > 4 else ''
					tags_str = parts[5].strip() if len(parts) > 5 else ''
					upvotes_str = parts[6].strip() if len(parts) > 6 else ''
					maker = parts[7].strip() if len(parts) > 7 else ''
					comments_str = parts[8].strip() if len(parts) > 8 else ''

					# Skip header rows
					if name.lower() in ['name', 'product', 'title'] or rank_str.lower() == 'rank' or (rank_str and not rank_str.isdigit()):
						continue

					if name:
						# Parse rank
						rank = None
						if rank_str and rank_str.isdigit():
							rank = int(rank_str)

						# Normalize URL
						producthunt_url = None
						if url:
							producthunt_url = _normalize_producthunt_url(url, source_url)

						# Parse tags
						tags = []
						if tags_str:
							tag_list = [t.strip() for t in re.split(r'[,;]', tags_str) if t.strip()]
							tags.extend(tag_list)

						# Parse upvotes
						upvotes = None
						if upvotes_str:
							# Extract number from string (e.g., "534" or "534 upvotes")
							upvotes_match = re.search(r'(\d+)', upvotes_str)
							if upvotes_match:
								upvotes = int(upvotes_match.group(1))

						# Parse comments
						comments_count = None
						if comments_str:
							comments_match = re.search(r'(\d+)', comments_str)
							if comments_match:
								comments_count = int(comments_match.group(1))

						# Clean maker
						maker_clean = maker if maker and maker.lower() not in ['null', 'n/a', ''] else None

						products.append(ProductHuntProduct(
							name=name,
							producthunt_url=producthunt_url,
							rank=rank,
							description=description if description and description.lower() not in ['null', 'n/a', ''] else None,
							tags=tags,
							upvotes=upvotes,
							maker=maker_clean,
							comments_count=comments_count,
						))
				except (IndexError, ValueError) as e:
					# Skip malformed rows
					continue

		if products:
			return ProductHuntLeaderboardReport(
				source_url=AnyHttpUrl(source_url),
				products=products,
			)

	# Fallback to line-by-line parsing
	lines = content.split('\n')

	for i, line in enumerate(lines):
		line_stripped = line.strip()

		# Skip empty lines
		if not line_stripped:
			continue

		# Detect new product entry - format 1: "1. **Name**: value" or "**Name**: value"
		name_match = re.match(r'^\d+\.\s*\*\*Name\*\*:\s*(.+)$', line_stripped)
		if name_match:
			# Save previous product if exists
			if current_product and current_product.get('name'):
				try:
					products.append(ProductHuntProduct.model_validate(current_product))
				except ValidationError:
					pass

			# Start new product
			name = name_match.group(1).strip()
			current_product = {
				'name': name,
				'producthunt_url': None,
				'rank': None,
				'description': None,
				'tags': [],
				'upvotes': None,
				'maker': None,
				'comments_count': None,
			}
			continue

		# Detect new product entry - format 2: "**Product Name**" (standalone)
		if line_stripped.startswith('**') and line_stripped.endswith('**') and len(line_stripped) > 4:
			# Check if it's a header or section name to skip
			if 'product' in line_stripped.lower() and ('list' in line_stripped.lower() or 'leaderboard' in line_stripped.lower()):
				continue

			# Save previous product if exists
			if current_product and current_product.get('name'):
				try:
					products.append(ProductHuntProduct.model_validate(current_product))
				except ValidationError:
					pass

			# Start new product
			name = line_stripped.strip('*').strip()
			current_product = {
				'name': name,
				'producthunt_url': None,
				'rank': None,
				'description': None,
				'tags': [],
				'upvotes': None,
				'maker': None,
				'comments_count': None,
			}
			continue

		# Only process fields if we have a current product with a valid name
		if current_product and current_product.get('name'):
			# Parse Rank
			if '**Rank:**' in line_stripped or '**Rank**:' in line_stripped:
				rank_match = re.search(r'\*\*Rank:\*\*\s*(\d+)', line_stripped)
				if rank_match:
					current_product['rank'] = int(rank_match.group(1))
				continue

			# Parse Product Hunt URL
			if '**Product Hunt URL:**' in line_stripped or '**URL:**' in line_stripped or '**ProductHunt URL:**' in line_stripped:
				url_match = re.search(r'https?://[^\s*]+|/products/[^\s*]+', line_stripped)
				if url_match:
					url = url_match.group(0)
					current_product['producthunt_url'] = _normalize_producthunt_url(url, source_url)
				continue

			# Parse Description
			if '**Description:**' in line_stripped:
				desc_match = re.search(r'\*\*Description:\*\*\s*(.+)', line_stripped)
				if desc_match:
					desc = desc_match.group(1).strip()
					if desc.lower() not in ['null', '(no description provided)', '(information not available)', '(not available)']:
						current_product['description'] = desc
				continue

			# Parse Tags
			if '**Tags:**' in line_stripped or '**Categories:**' in line_stripped:
				tags_match = re.search(r'\*\*Tags?:\*\*\s*(.+)', line_stripped)
				if tags_match:
					tags_str = tags_match.group(1).strip()
					if tags_str.lower() not in ['null', 'n/a', '']:
						tag_list = [t.strip() for t in re.split(r'[,;]', tags_str) if t.strip()]
						current_product['tags'] = tag_list
				continue

			# Parse Upvotes
			if '**Upvotes:**' in line_stripped or '**Votes:**' in line_stripped:
				upvotes_match = re.search(r'\*\*Upvotes?:\*\*\s*(\d+)', line_stripped)
				if upvotes_match:
					current_product['upvotes'] = int(upvotes_match.group(1))
				continue

			# Parse Maker
			if '**Maker:**' in line_stripped:
				maker_match = re.search(r'\*\*Maker:\*\*\s*(.+)', line_stripped)
				if maker_match:
					maker = maker_match.group(1).strip()
					if maker.lower() not in ['null', 'n/a', '']:
						current_product['maker'] = maker
				continue

			# Parse Comments Count
			if '**Comments:**' in line_stripped or '**Comments Count:**' in line_stripped:
				comments_match = re.search(r'\*\*Comments?:\*\*\s*(\d+)', line_stripped)
				if comments_match:
					current_product['comments_count'] = int(comments_match.group(1))
				continue

	# Don't forget the last product
	if current_product and current_product.get('name'):
		try:
			products.append(ProductHuntProduct.model_validate(current_product))
		except ValidationError:
			pass

	if products:
		return ProductHuntLeaderboardReport(
			source_url=AnyHttpUrl(source_url),
			products=products,
		)
	return None


def _sanitize_report(report: ProductHuntLeaderboardReport) -> ProductHuntLeaderboardReport:
	"""Apply basic clean-up rules on top of the structured output."""

	base_url = str(report.source_url)
	for product in report.products:
		# Normalize Product Hunt URL
		if product.producthunt_url:
			producthunt_url_str = str(product.producthunt_url)
			if not producthunt_url_str.startswith('http://') and not producthunt_url_str.startswith('https://'):
				product.producthunt_url = _normalize_producthunt_url(producthunt_url_str, base_url)
			elif 'producthunt.com' not in producthunt_url_str.lower():
				# Invalid URL, set to None
				product.producthunt_url = None

		# Clean up tags: remove empty strings and strip whitespace
		if product.tags:
			product.tags = [tag.strip() for tag in product.tags if tag.strip()]

	return report


def build_task(task_input: ProductHuntLeaderboardInput) -> str:
	"""Create the natural-language instructions fed to the agent, specialized for Product Hunt leaderboard."""

	extract_all = task_input.max_products >= 1000
	leaderboard_url = str(task_input.url)

	return dedent(
		f"""
		Tu es un agent spécialisé dans l'extraction de produits depuis la page leaderboard Product Hunt.

		Objectif CRITIQUE:
		- Navigue directement vers l'URL du leaderboard: {leaderboard_url}
		- IMPORTANT: Utilise l'action `navigate` pour aller directement à cette URL - NE FAIS PAS de recherche internet
		- {"Identifie et extrait TOUS les produits présents sur cette page leaderboard Product Hunt, SANS AUCUNE EXCEPTION." if extract_all else f"Identifie et extrait jusqu'à {task_input.max_products} produits présents sur cette page leaderboard Product Hunt."}
		- IMPORTANT: Ne filtre PAS les produits. Prends TOUS les produits visibles sur la page.
		- Ne confonds PAS les titres de sections avec des produits réels.
		- Pour chaque produit, capture:
		  • `name`: nom exact du produit tel qu'affiché sur la carte
		  • `producthunt_url`: URL complète vers la page Product Hunt du produit (format: https://www.producthunt.com/products/nom-du-produit)
		  • `rank`: position dans le leaderboard (si visible)
		  • `description`: tagline/description du produit affichée sur la carte
		  • `tags`: tags/catégories visibles (ex: ["Artificial Intelligence", "Productivity"])
		  • `upvotes`: nombre d'upvotes si visible
		  • `maker`: nom du maker si visible
		  • `comments_count`: nombre de commentaires si visible

		Processus pour Product Hunt Leaderboard:
		⚠️ IMPORTANT: La navigation et le scroll jusqu'en bas ont déjà été effectués automatiquement.
		Tu es déjà sur la page {leaderboard_url} et tout le contenu a été chargé.
		
		Tu dois maintenant UNIQUEMENT:
		1. EXTRACTION DES PRODUITS - MÉTHODE OBLIGATOIRE
		   - ⚠️ CRITIQUE: N'utilise `evaluate` QUE APRÈS avoir scrollé jusqu'en bas de la page et attendu 3 secondes
		   - ⚠️ NE PAS utiliser `evaluate` avant d'avoir fini de scroller - attends d'être vraiment en bas!
		   - ⚠️ CRITIQUE: Utilise UNIQUEMENT l'action `evaluate` pour extraire le HTML des produits
		   - ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs `data-test` directement
		   - Format exact de l'action `evaluate`: {{"evaluate": {{"code": "TON_CODE_JAVASCRIPT"}}}}
		   - Code JavaScript à utiliser EXACTEMENT (copie-le tel quel):
		   (function(){{const sections = Array.from(document.querySelectorAll('section[data-test^="post-item-"]'));return JSON.stringify(sections.map(section => section.outerHTML));}})()
		   - Ce code va extraire le HTML complet de tous les éléments `<section data-test="post-item-...">`
		   - Le résultat sera une chaîne JSON contenant un tableau de chaînes HTML
		   - ⚠️ ATTENTION: `evaluate` utilise le paramètre `code` (PAS `query`, PAS `extract_links`, PAS `start_from_char`)
		   - Une fois que tu as le résultat de `evaluate`, utilise directement `done` avec les données parsées dans le format ProductHuntLeaderboardReport

		2. Traitement des résultats de `evaluate`:
		   - Le résultat de `evaluate` sera une chaîne JSON contenant un tableau de chaînes HTML
		   - Parse cette chaîne JSON pour obtenir le tableau de HTML
		   - Pour chaque élément HTML du tableau, extrais les informations suivantes:
		     • `name`: texte du lien `<a href="/products/...">` dans la section
		     • `producthunt_url`: attribut `href` du même lien (normalise en URL absolue: https://www.producthunt.com/products/...)
		     • `rank`: nombre extrait de l'attribut `data-test` (ex: "post-item-1039459" -> 1039459)
		     • `description`: texte de la `<div>` avec classe "text-secondary" ou similaire
		     • `tags`: texte de tous les liens `<a href="/topics/...">` dans la section
		     • `upvotes`: nombre extrait du bouton avec `data-test="vote-button"`
		     • `comments_count`: nombre extrait du bouton avec l'icône de commentaire (SVG path contenant "M12.25 6.708")
		   - Construis un objet `ProductHuntLeaderboardReport` avec:
		     • `source_url`: "{leaderboard_url}" (chaîne de caractères, pas d'objet URL)
		     • `products`: tableau de tous les produits extraits dans l'ordre
		   - Limite la liste finale à {task_input.max_products} produits maximum si nécessaire
		   - Utilise l'action `done` avec le champ `data` contenant l'objet `ProductHuntLeaderboardReport` complet
		   - Format exact: {{"done": {{"success": true, "data": {{"source_url": "{leaderboard_url}", "products": [{{"name": "...", "producthunt_url": "...", ...}}, ...]}}}}}}

		Règles importantes:
		- ⚠️ NAVIGATION ET SCROLL: La navigation et le scroll ont déjà été effectués automatiquement - NE PAS naviguer ni scroller!
		- ⚠️ NE PAS utiliser `navigate` - tu es déjà sur la bonne page!
		- ⚠️ NE PAS utiliser `scroll` - tout le contenu a déjà été chargé!
		- ⚠️ NE PAS utiliser `wait` - pas besoin d'attendre, tout est déjà chargé!
		- ⚠️ EXTRACTION OBLIGATOIRE: Utilise UNIQUEMENT `evaluate` avec le paramètre `code` pour extraire le HTML
		- ⚠️ IMPORTANT: N'utilise `evaluate` QUE APRÈS avoir scrollé jusqu'en bas - pas avant!
		- Format `evaluate`: {{"evaluate": {{"code": "(function(){{const sections = Array.from(document.querySelectorAll('section[data-test^=\"post-item-\"]'));return JSON.stringify(sections.map(section => section.outerHTML));}})()"}}}}
		- ⚠️ NE PAS utiliser `extract` - il ne peut pas accéder aux attributs `data-test` directement
		- Le code JavaScript doit sélectionner tous les éléments avec `document.querySelectorAll('section[data-test^="post-item-"]')`
		- Retourne un tableau JSON de chaînes HTML (outerHTML de chaque section)
		- Une fois le HTML extrait, parse-le pour construire le rapport ProductHuntLeaderboardReport
		- Les `tags` doivent contenir les catégories/tags visibles sur la carte produit
		- Si une info manque, laisse le champ null, mais ne l'invente pas
		- CRITIQUE: Ne visite JAMAIS les pages individuelles des produits - extrais toutes les informations depuis les cartes du leaderboard
		- CRITIQUE SÉRIALISATION: Lorsque tu appelles `done`, assure-toi que `source_url` est une chaîne de caractères (string), pas un objet URL
		- Exemple correct: "source_url": "https://www.producthunt.com/leaderboard/daily/2025/11/18/all"
		- Exemple incorrect: "source_url": AnyHttpUrl("https://www.producthunt.com/leaderboard/daily/2025/11/18/all")
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
	
	# Wait for initial page load
	print("   ⏳ Attente initiale de 5 secondes pour le chargement de la page...")
	await asyncio.sleep(5)
	
	# Check initial content - try multiple selectors
	try:
		# Try the main selector
		initial_sections = await page.evaluate("() => document.querySelectorAll('section[data-test^=\"post-item-\"]').length")
		initial_sections = int(initial_sections) if initial_sections else 0
		
		# Also check for alternative selectors
		if initial_sections == 0:
			alt_selectors = [
				"section[data-test*=\"post-item\"]",
				"section[data-test*=\"post\"]",
				"[data-test^=\"post-item-\"]",
				"[data-test*=\"post-item\"]",
				"article",
				".post-item",
				"[class*=\"post-item\"]",
			]
			for selector in alt_selectors:
				try:
					count = await page.evaluate(f"() => document.querySelectorAll('{selector}').length")
					count = int(count) if count else 0
					if count > 0:
						print(f"   🔍 Sélecteur alternatif trouvé: '{selector}' avec {count} éléments")
						initial_sections = count
						break
				except Exception:
					pass
		
		print(f"   📊 Sections initiales trouvées: {initial_sections}")
	except Exception:
		initial_sections = 0
	
	# Scroll progressivement jusqu'en bas
	scroll_count = 0
	last_scroll_position = -1
	last_sections_count = initial_sections
	no_change_count = 0
	
	for i in range(max_scrolls):
		# Get current scroll position and content count
		try:
			current_position = await page.evaluate("() => window.pageYOffset || document.documentElement.scrollTop")
			current_position = int(current_position) if current_position else 0
			
			# Check how many sections are currently visible - try multiple selectors
			current_sections = await page.evaluate("() => document.querySelectorAll('section[data-test^=\"post-item-\"]').length")
			current_sections = int(current_sections) if current_sections else 0
			
			# If no sections found, try alternative selectors
			if current_sections == 0:
				alt_selectors = [
					"section[data-test*=\"post-item\"]",
					"[data-test^=\"post-item-\"]",
					"[data-test*=\"post-item\"]",
					"article",
				]
				for selector in alt_selectors:
					try:
						count = await page.evaluate(f"() => document.querySelectorAll('{selector}').length")
						count = int(count) if count else 0
						if count > 0:
							current_sections = count
							break
					except Exception:
						pass
		except Exception:
			current_position = 0
			current_sections = 0
		
		# Check if we're at the bottom (no change in scroll position AND no new content loaded)
		# But only stop if we've scrolled at least 10 times (to ensure we've tried to load content)
		if scroll_count >= 10 and current_position == last_scroll_position and current_sections == last_sections_count:
			no_change_count += 1
			if no_change_count >= 3:  # No change for 3 consecutive scrolls
				print(f"   ✅ Arrivé en bas après {scroll_count} scrolls ({current_sections} sections au total)")
				break
		else:
			no_change_count = 0
			if current_sections > last_sections_count:
				print(f"   📊 {current_sections} sections trouvées (nouveau contenu chargé)")
			elif scroll_count < 10:
				# Force continue scrolling for first 10 scrolls even if position doesn't change
				# This ensures we give the page time to load content
				no_change_count = 0
		
		last_scroll_position = current_position
		last_sections_count = current_sections
		
		# Scroll down by one page using JavaScript - use instant scroll to trigger loading
		# Smooth scroll might not trigger lazy loading properly
		try:
			# Use instant scroll instead of smooth to ensure content loads
			await page.evaluate(f"() => window.scrollBy(0, {viewport_height})")
			# Also try scrolling the document element directly
			await page.evaluate(f"() => document.documentElement.scrollTop += {viewport_height}")
		except Exception as e:
			print(f"   ⚠️ Erreur lors du scroll: {e}")
			# Try alternative scroll method
			try:
				await page.evaluate(f"() => window.scrollTo(0, window.pageYOffset + {viewport_height})")
			except Exception:
				print(f"   ⚠️ Toutes les méthodes de scroll ont échoué")
				break
		
		scroll_count += 1
		
		# Wait longer after each scroll for content to load (lazy loading needs time)
		# Product Hunt uses lazy loading, so we need to wait for content to appear
		await asyncio.sleep(4)  # Increased from 3 to 4 seconds
		
		if scroll_count % 5 == 0:
			print(f"   📜 {scroll_count} scrolls effectués, {current_sections} sections trouvées...")
	
	# Final wait to ensure everything is loaded
	print("   ⏳ Attente finale de 5 secondes pour le chargement complet...")
	await asyncio.sleep(5)
	
	# Final check of content
	try:
		final_sections = await page.evaluate("() => document.querySelectorAll('section[data-test^=\"post-item-\"]').length")
		final_sections = int(final_sections) if final_sections else 0
		print(f"   📊 Sections finales trouvées: {final_sections}")
	except Exception:
		final_sections = 0
	
	print(f"✅ Scroll terminé: {scroll_count} scrolls effectués au total, {final_sections} sections trouvées")


async def run_product_hunt_leaderboard(task_input: ProductHuntLeaderboardInput) -> ProductHuntLeaderboardReport | None:
	"""Execute the agent and return the structured list of products."""

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
		# Navigate to the leaderboard URL in a new tab
		leaderboard_url = str(task_input.url)
		print(f"📍 Navigation vers: {leaderboard_url} (nouvel onglet)")
		# Use NavigateToUrlEvent with new_tab=True to force opening in a new tab
		navigate_event = NavigateToUrlEvent(url=leaderboard_url, new_tab=True)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event
		
		# Get the current page after navigation
		page = await browser.get_current_page()
		if not page:
			# Fallback: try new_page if get_current_page doesn't work
			page = await browser.new_page(leaderboard_url)
		
		await asyncio.sleep(5)  # Wait longer for initial page load
		
		# Try to accept cookies if banner appears
		try:
			# Use evaluate to check for cookie banner and click it
			cookie_result = await page.evaluate(
				"(function() { const buttons = Array.from(document.querySelectorAll('button')); const acceptButton = buttons.find(btn => btn.textContent.toLowerCase().includes('accept') || btn.textContent.toLowerCase().includes('accepter') || btn.getAttribute('data-test')?.includes('cookie')); if (acceptButton) { acceptButton.click(); return 'clicked'; } return 'not_found'; })()"
			)
			if cookie_result and 'clicked' in str(cookie_result):
				print("   ✅ Bannière de cookies acceptée")
				await asyncio.sleep(2)
		except Exception:
			pass  # Cookie banner might not be present
		
		# Scroll to bottom naively (without LLM)
		# Browser is actually a BrowserSession, so we can use it directly
		await _scroll_to_bottom_naive(browser, max_scrolls=50)
		
		# Verify content is loaded and extract directly using JavaScript
		print("🔍 Vérification du contenu chargé et extraction directe...")
		current_page = await browser.get_current_page()
		if not current_page:
			raise RuntimeError("No current page available after scroll")
		
		# Check if content is loaded
		try:
			sections_count = await current_page.evaluate(
				"() => document.querySelectorAll('section[data-test^=\"post-item-\"]').length"
			)
			sections_count = int(sections_count) if sections_count else 0
			print(f"   📊 {sections_count} sections de produits trouvées")
			
			if sections_count == 0:
				print("   ⚠️ Aucune section trouvée, attente supplémentaire...")
				await asyncio.sleep(5)
				sections_count = await current_page.evaluate(
					"() => document.querySelectorAll('section[data-test^=\"post-item-\"]').length"
				)
				sections_count = int(sections_count) if sections_count else 0
				print(f"   📊 Après attente: {sections_count} sections trouvées")
		except Exception as e:
			print(f"   ⚠️ Erreur lors de la vérification: {e}")
			sections_count = 0
		
		# Extract HTML directly using JavaScript (bypass agent for reliability)
		extraction_successful = False
		if sections_count > 0:
			print("📥 Extraction directe du HTML des produits...")
			try:
				# Extract HTML sections - page.evaluate() adds () automatically, so don't include it
				html_sections_json = await current_page.evaluate(
					"() => { const sections = Array.from(document.querySelectorAll('section[data-test^=\"post-item-\"]')); return JSON.stringify(sections.map(section => section.outerHTML)); }"
				)
				
				if html_sections_json and html_sections_json != '[]' and html_sections_json.strip():
					# Parse the JSON string
					html_sections = json.loads(html_sections_json)
					print(f"   ✅ {len(html_sections)} sections HTML extraites")
					
					if html_sections and len(html_sections) > 0:
						# Parse HTML sections directly
						html_report = _parse_html_sections(html_sections, str(task_input.url))
						if html_report and html_report.products and len(html_report.products) > 0:
							print(f"   ✅ {len(html_report.products)} produits parsés depuis le HTML")
							
							# Limit to max_products if needed
							if task_input.max_products < 10000 and len(html_report.products) > task_input.max_products:
								html_report.products = html_report.products[:task_input.max_products]
							
							extraction_successful = True
							return _sanitize_report(html_report)
						else:
							print(f"   ⚠️ Aucun produit parsé depuis le HTML (report: {html_report})")
					else:
						print(f"   ⚠️ Tableau HTML vide (length: {len(html_sections) if html_sections else 0})")
				else:
					print(f"   ⚠️ Aucune donnée extraite (result: {html_sections_json[:100] if html_sections_json else 'None'})")
			except json.JSONDecodeError as e:
				print(f"   ⚠️ Erreur de parsing JSON lors de l'extraction directe: {e}")
				print(f"   ⚠️ Contenu reçu: {html_sections_json[:200] if 'html_sections_json' in locals() else 'N/A'}")
			except Exception as e:
				print(f"   ⚠️ Erreur lors de l'extraction directe: {e}")
				import traceback
				traceback.print_exc()
		else:
			print(f"   ⚠️ Aucune section trouvée (sections_count: {sections_count}), impossible d'extraire")
		
		# Only use agent as fallback if direct extraction completely failed
		if not extraction_successful:
			print("❌ Extraction directe échouée. Le contenu n'a peut-être pas été chargé correctement.")
			print("   💡 Vérifiez que le scroll s'est bien terminé et que la page a chargé le contenu.")
			return _fallback_report(str(task_input.url), "Extraction directe échouée: aucune section de produit trouvée après le scroll.")
		
		# This should never be reached, but just in case:
		print("🤖 Utilisation de l'agent comme dernier recours...")
		agent = Agent(
			task=build_task(task_input),
			llm=llm,
			page_extraction_llm=page_extraction_llm,
			browser=browser,  # Use the browser we already set up
			output_model_schema=ProductHuntLeaderboardReport,
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
				report = ProductHuntLeaderboardReport.model_validate_json(final_result)
				if not agent_successful:
					print("⚠️  Données récupérées depuis le résultat final malgré l'échec de l'agent.")
				return _sanitize_report(report)
			except ValidationError:
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if json_match:
					try:
						report = ProductHuntLeaderboardReport.model_validate_json(json_match.group(1))
						if not agent_successful:
							print("⚠️  Données récupérées depuis le résultat final (markdown) malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError:
						pass

		# Try to extract from evaluate actions (HTML sections)
		all_extracted_products: list[ProductHuntProduct] = []
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
							if html_report and html_report.products:
								for product in html_report.products:
									if product.name and product.name not in seen_names:
										all_extracted_products.append(product)
										seen_names.add(product.name)
						except (json.JSONDecodeError, TypeError) as e:
							# If not JSON, try parsing as string directly
							try:
								html_report = _parse_html_sections(result.extracted_content, str(task_input.url))
								if html_report and html_report.products:
									for product in html_report.products:
										if product.name and product.name not in seen_names:
											all_extracted_products.append(product)
											seen_names.add(product.name)
							except Exception:
								# Skip if parsing fails
								pass
		
		# Try to extract from action results (especially extract and evaluate actions)
		extracted_contents = history.extracted_content()
		
		# Process extractions in chronological order (first to last) to preserve product order
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
				if html_report and html_report.products:
					for product in html_report.products:
						if product.name and product.name not in seen_names:
							all_extracted_products.append(product)
							seen_names.add(product.name)
					continue
			except (json.JSONDecodeError, TypeError):
				pass
			
			# Try parsing as HTML sections (from evaluate action returning raw HTML)
			html_report = _parse_html_sections(content, str(task_input.url))
			if html_report and html_report.products:
				for product in html_report.products:
					if product.name and product.name not in seen_names:
						all_extracted_products.append(product)
						seen_names.add(product.name)
				continue

			# Fallback to markdown parsing
			markdown_report = _parse_extracted_markdown(content, str(task_input.url))
			if markdown_report and markdown_report.products:
				for product in markdown_report.products:
					if product.name and product.name not in seen_names:
						all_extracted_products.append(product)
						seen_names.add(product.name)

		# If we found products from markdown parsing, return them
		if all_extracted_products:
			if task_input.max_products < 10000:
				all_extracted_products = all_extracted_products[:task_input.max_products]

			report = ProductHuntLeaderboardReport(
				source_url=AnyHttpUrl(str(task_input.url)),
				products=all_extracted_products,
			)
			if not agent_successful:
				print(f"⚠️  Données récupérées depuis les résultats d'extraction (markdown) malgré l'échec de l'agent. {len(all_extracted_products)} produits trouvés.")
			return _sanitize_report(report)

		# Fallback: try individual content parsing
		for content in reversed(extracted_contents):
			if not content:
				continue

			markdown_report = _parse_extracted_markdown(content, str(task_input.url))
			if markdown_report and markdown_report.products:
				if not agent_successful:
					print(f"⚠️  Données récupérées depuis les résultats d'extraction (markdown) malgré l'échec de l'agent. {len(markdown_report.products)} produits trouvés.")
				return _sanitize_report(markdown_report)

			try:
				report = ProductHuntLeaderboardReport.model_validate_json(content)
				if not agent_successful:
					print("⚠️  Données récupérées depuis les résultats d'extraction malgré l'échec de l'agent.")
				return _sanitize_report(report)
			except ValidationError:
				json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
				if json_match:
					try:
						report = ProductHuntLeaderboardReport.model_validate_json(json_match.group(1))
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
						report = ProductHuntLeaderboardReport.model_validate(data_copy)
						if not agent_successful:
							print("⚠️  Données récupérées depuis l'action 'done' malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except ValidationError as e:
						try:
							json_str = json.dumps(data_copy, default=str)
							report = ProductHuntLeaderboardReport.model_validate_json(json_str)
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


def parse_arguments() -> ProductHuntLeaderboardInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Extrait les produits depuis une page leaderboard Product Hunt')
	parser.add_argument(
		'date',
		help='Date du leaderboard au format YYYY-MM-DD ou YYYY/MM/DD (ex: 2025-11-18)',
	)
	parser.add_argument(
		'--max-products',
		type=int,
		default=1000,
		help='Nombre maximal de produits à extraire (par défaut: 1000)',
	)
	parser.add_argument(
		'--output',
		default='product_hunt_leaderboard.json',
		help='Chemin du fichier JSON résultat (par défaut: ./product_hunt_leaderboard.json)',
	)
	args = parser.parse_args()
	return ProductHuntLeaderboardInput(date=args.date, max_products=args.max_products, output_path=Path(args.output))


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		leaderboard_url = str(task_input.url)
		print(f"🚀 Démarrage de l'agent pour la date: {task_input.date}")
		print(f"📍 URL du leaderboard: {leaderboard_url}")
		print(f"📊 Nombre max de produits: {task_input.max_products}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")

		result = await run_product_hunt_leaderboard(task_input)

		if result is None:
			print("❌ L'agent n'a retourné aucune donnée structurée.")
			return

		output_json = result.model_dump_json(indent=2, ensure_ascii=False)
		output_path = task_input.output_path
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(output_json, encoding='utf-8')

		#print(output_json)
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

