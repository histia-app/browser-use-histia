"""
Agent designed to build a lightweight list of startups from directories such as
Product Hunt, BetaList, FutureTools, etc.
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

# Load environment variables immediately so the agent can access API keys.
load_dotenv()

# Configure timeouts BEFORE importing browser_use to ensure they're applied
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '30')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '60')

from browser_use import Agent, ChatBrowserUse, ChatOpenAI
from examples.histia import print_llm_usage_summary


class StartupListingInput(BaseModel):
	"""User-provided parameters for the startup listing task."""

	url: AnyHttpUrl = Field(..., description='Product Hunt, BetaList, FutureTools, etc. listing URL')
	max_startups: int = Field(
		12,
		ge=1,
		le=1000,
		description='Maximum number of startups to capture from the page (use a high number like 1000 to extract all)',
	)
	output_path: Path = Field(
		default=Path('startup_listings.json'),
		description='Destination for the JSON list of startups',
	)


class StartupProfile(BaseModel):
	"""Minimal structured information for each startup entry."""

	name: str = Field(..., description='Startup name exactly as written on the listing')
	listing_url: str | None = Field(
		None,
		description='Direct URL to the startup page/product as exposed by the listing',
	)
	linkedin_url: str | None = Field(
		None,
		description='Public LinkedIn URL shown on the listing (keep None if absent)',
	)
	description: str | None = Field(
		None,
		description='Full description of the startup/product if available on the listing (keep None if absent)',
	)
	short_notes: list[str] = Field(
		default_factory=list,
		description='Two or three short bullet points from the listing (value proposition, positioning, tags, etc.)',
	)


class StartupListingReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='URL that was analysed')
	startups: list[StartupProfile] = Field(
		...,
		min_length=1,
		description='Startup entries ordered as they appear on the listing',
	)

	@field_serializer('source_url')
	def serialize_source_url(self, value: AnyHttpUrl, _info) -> str:
		"""Convert AnyHttpUrl to string for JSON serialization."""
		return str(value)

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Override model_dump to ensure AnyHttpUrl is converted to string."""
		# Always convert AnyHttpUrl to string for JSON compatibility
		result = super().model_dump(**kwargs)
		# Convert AnyHttpUrl to string if present (works even without mode='json')
		if 'source_url' in result:
			source_url = result['source_url']
			# Check if it's an AnyHttpUrl object (not a string)
			if hasattr(source_url, '__str__') and not isinstance(source_url, str):
				result['source_url'] = str(source_url)
		return result
	
	def model_dump_json(self, **kwargs) -> str:
		"""Override model_dump_json to ensure AnyHttpUrl is serialized correctly."""
		# Use model_dump with mode='json' to apply serializers
		return super().model_dump_json(**kwargs)


def _normalize_linkedin_url(value: str | None) -> str | None:
	"""Return a valid LinkedIn URL or None."""

	if not value:
		return None

	url = value.strip()
	if not url:
		return None

	if not url.lower().startswith(('http://', 'https://')):
		return None

	parsed = urlparse(url)
	if not parsed.netloc:
		return None

	if 'linkedin.com' not in parsed.netloc.lower():
		return None

	return url


def _fallback_report(source_url: str, reason: str) -> StartupListingReport:
	"""Return a minimal report when the agent cannot finish properly."""

	reason = reason.strip() or "Impossible d'obtenir un listing fiable depuis la page."
	from pydantic import AnyHttpUrl
	return StartupListingReport(
		source_url=AnyHttpUrl(source_url),
		startups=[
			StartupProfile(
				name='Informations indisponibles',
				listing_url=source_url,
				linkedin_url=None,
				description=None,
				short_notes=[
					reason,
					'Rapport généré automatiquement (agent interrompu avant la fin).',
				],
			)
		],
	)


def _normalize_listing_url(url: str | None, base_url: str) -> str | None:
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
		return url


def _parse_extracted_markdown(content: str, source_url: str) -> StartupListingReport | None:
	"""Parse markdown content from extract action to build StartupListingReport."""
	
	startups: list[StartupProfile] = []
	current_startup: dict[str, Any] | None = None
	in_description = False
	in_short_notes = False
	
	# First, try to parse markdown tables (common format from Product Hunt)
	# Look for table headers with Rank | Name | URL | Description | Tags/Categories
	table_pattern = re.compile(
		r'\|\s*Rank\s*\|\s*Name\s*\|\s*URL\s*\|\s*Description[^|]*\|\s*Tags/Categories\s*\|\s*\n'
		r'\|[-\s|:]+\|\s*\n'
		r'((?:\|\s*\d+\s*\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*\n?)+)',
		re.IGNORECASE | re.MULTILINE
	)
	
	# Also try a more flexible pattern that matches any table with Rank column
	flexible_table_pattern = re.compile(
		r'\|\s*Rank\s*\|\s*Name\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*[^|]+\s*\|\s*\n'
		r'\|[-\s|:]+\|\s*\n'
		r'((?:\|\s*\d+\s*\|\s*[^|]+\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*[^|]*\s*\|\s*\n?)+)',
		re.IGNORECASE | re.MULTILINE
	)
	
	table_match = table_pattern.search(content) or flexible_table_pattern.search(content)
	if table_match:
		table_rows = table_match.group(1).strip().split('\n')
		for row in table_rows:
			if not row.strip() or not row.strip().startswith('|'):
				continue
			# Parse table row: | Rank | Name | URL | Description | Tags |
			parts = [p.strip() for p in row.split('|')]
			if len(parts) >= 5:
				try:
					rank = parts[1].strip() if len(parts) > 1 else ''
					name = parts[2].strip() if len(parts) > 2 else ''
					url = parts[3].strip() if len(parts) > 3 else ''
					description = parts[4].strip() if len(parts) > 4 else ''
					tags = parts[5].strip() if len(parts) > 5 else ''
					
					# Skip header rows
					if name.lower() in ['name', 'product', 'title'] or rank.lower() == 'rank' or not rank.isdigit():
						continue
					
					if name:
						# Normalize URL
						listing_url = None
						if url:
							if url.startswith('/'):
								listing_url = f"https://www.producthunt.com{url}"
							elif url.startswith('http'):
								listing_url = url
							elif url.startswith('producthunt.com'):
								listing_url = f"https://www.{url}"
						
						# Parse tags
						short_notes = []
						if tags:
							# Tags might be comma-separated or space-separated
							tag_list = [t.strip() for t in re.split(r'[,;]', tags) if t.strip()]
							short_notes.extend(tag_list)
						
						startups.append(StartupProfile(
							name=name,
							listing_url=listing_url,
							linkedin_url=None,
							description=description if description and description.lower() not in ['null', 'n/a', ''] else None,
							short_notes=short_notes,
						))
				except (IndexError, ValueError) as e:
					# Skip malformed rows
					continue
		
		if startups:
			return StartupListingReport(
				source_url=AnyHttpUrl(source_url),
				startups=startups,
			)
	
	# Fallback to line-by-line parsing
	lines = content.split('\n')
	
	for i, line in enumerate(lines):
		original_line = line
		line_stripped = line.strip()
		indent_level = len(line) - len(line.lstrip())
		
		# Skip empty lines
		if not line_stripped:
			continue
		
		# Detect new startup entry - format 1: "1. **Name**: value" or "**Name**: value"
		name_match = re.match(r'^\d+\.\s*\*\*Name\*\*:\s*(.+)$', line_stripped)
		if name_match:
			# Save previous startup if exists
			if current_startup and current_startup.get('name'):
				try:
					startups.append(StartupProfile.model_validate(current_startup))
				except ValidationError:
					pass  # Skip invalid entries
			
			# Start new startup
			name = name_match.group(1).strip()
			current_startup = {
				'name': name,
				'listing_url': None,
				'linkedin_url': None,
				'description': None,
				'short_notes': [],
			}
			in_description = False
			in_short_notes = False
			continue
		
		# Detect new startup entry - format 2: "**Fiche de galerie 1**" or "**Name**" (standalone)
		if line_stripped.startswith('**') and line_stripped.endswith('**') and len(line_stripped) > 4:
			# Check if it's a header or section name to skip
			if 'Fiche' in line_stripped or 'galerie' in line_stripped.lower() or 'Short Notes' in line_stripped or 'Notes' in line_stripped:
				# This is a header or section, skip it
				continue
			
			# Save previous startup if exists
			if current_startup and current_startup.get('name'):
				try:
					startups.append(StartupProfile.model_validate(current_startup))
				except ValidationError:
					pass  # Skip invalid entries
			
			# Start new startup
			name = line_stripped.strip('*').strip()
			current_startup = {
				'name': name,
				'listing_url': None,
				'linkedin_url': None,
				'description': None,
				'short_notes': [],
			}
			in_description = False
			in_short_notes = False
			continue
		
		# Detect "**Name:** value" format (after header) - can be standalone or with current_startup
		if '**Name:**' in line_stripped or '**Name**:' in line_stripped:
			name_match = re.search(r'\*\*Name:\*\*\s*(.+)', line_stripped)
			if name_match:
				name_value = name_match.group(1).strip()
				# If we don't have a current startup or it doesn't have a name, start a new one
				if not current_startup or not current_startup.get('name'):
					# Save previous startup if exists
					if current_startup and current_startup.get('name'):
						try:
							startups.append(StartupProfile.model_validate(current_startup))
						except ValidationError:
							pass
					# Start new startup
					current_startup = {
						'name': name_value,
						'listing_url': None,
						'linkedin_url': None,
						'description': None,
						'short_notes': [],
					}
					in_description = False
					in_short_notes = False
				else:
					# Update existing startup name if it's empty
					current_startup['name'] = name_value
				continue
		
		# Only process fields if we have a current startup with a valid name
		if current_startup and current_startup.get('name'):
			# Parse Website URL
			if '**Website URL:**' in line_stripped or '**Website:**' in line_stripped or '**Website URL**:' in line_stripped:
				in_description = False
				in_short_notes = False
				url_match = re.search(r'https?://[^\s*]+|www\.[^\s*]+', line_stripped)
				if url_match:
					url = url_match.group(0)
					if not url.startswith('http'):
						url = 'https://' + url
					current_startup['listing_url'] = url
				continue
			
			# Parse Description
			if '**Description EN:**' in line_stripped or '**Description:**' in line_stripped or '**Description EN**:' in line_stripped:
				in_description = True
				in_short_notes = False
				# Extract description after the colon
				desc_match = re.search(r'\*\*Description(?:\s+EN)?:\*\*\s*(.+)', line_stripped)
				if desc_match:
					desc = desc_match.group(1).strip()
					# Handle special cases
					if desc.lower() in ['null', '(no description provided)', '(information not available)', '(not available)']:
						current_startup['description'] = None
					else:
						current_startup['description'] = desc
				else:
					current_startup['description'] = ''
				continue
			
			# Parse LinkedIn URL
			if '**LinkedIn' in line_stripped or '**LinkedIn URL:**' in line_stripped or '**LinkedIn URL**:' in line_stripped:
				in_description = False
				in_short_notes = False
				if 'null' in line_stripped.lower():
					current_startup['linkedin_url'] = None
				else:
					url_match = re.search(r'https?://[^\s*]+', line_stripped)
					if url_match:
						current_startup['linkedin_url'] = url_match.group(0)
				continue
			
			# Detect "Short Notes:" section - but don't create a new startup
			if '**Short Notes:**' in line_stripped or '**Short Notes**:' in line_stripped:
				in_description = False
				in_short_notes = True
				continue
			
			# Continue description if we're in description mode
			if in_description and line_stripped and not line_stripped.startswith('*') and '**' not in line_stripped:
				current_desc = current_startup.get('description')
				if current_desc:
					current_startup['description'] = str(current_desc) + ' ' + line_stripped
				else:
					current_startup['description'] = line_stripped
				continue
			
			# Parse other fields as short_notes (but only if not in description mode)
			if not in_description and not in_short_notes and (line_stripped.startswith('*') or indent_level > 0) and '**' in line_stripped and ':' in line_stripped:
				# Extract field name and value
				field_match = re.search(r'\*\*([^*]+)\*\*:\s*(.+)', line_stripped)
				if field_match:
					field_name = field_match.group(1).strip()
					field_value = field_match.group(2).strip()
					
					# Skip if it's a URL field or description (already handled)
					if 'URL' not in field_name and 'Website' not in field_name and 'LinkedIn' not in field_name and 'Description' not in field_name and 'Name' not in field_name:
						# Handle null values
						if field_value.lower() in ['null', '(not specified)', '(not available)', '(no description provided)']:
							continue
						if field_value:
							current_startup['short_notes'].append(f"{field_name}: {field_value}")
				continue
			
			# Handle indented short notes (under "Short Notes:" section)
			if in_short_notes and indent_level > 0 and line_stripped.startswith('*'):
				note_match = re.match(r'\*\s*(.+?):\s*(.+)', line_stripped)
				if note_match:
					note_name = note_match.group(1).strip()
					note_value = note_match.group(2).strip()
					if note_value.lower() not in ['null', '(not specified)', '(not available)']:
						current_startup['short_notes'].append(f"{note_name}: {note_value}")
				continue
			
			# If we hit a new field (starts with * **), we're no longer in description
			if in_description and line_stripped.startswith('*') and '**' in line_stripped:
				in_description = False
	
	# Don't forget the last startup
	if current_startup and current_startup.get('name'):
		try:
			startups.append(StartupProfile.model_validate(current_startup))
		except ValidationError:
			pass
	
	if startups:
		return StartupListingReport(
			source_url=AnyHttpUrl(source_url),
			startups=startups,
		)
	return None


def _sanitize_report(report: StartupListingReport) -> StartupListingReport:
	"""Apply basic clean-up rules on top of the structured output."""

	base_url = str(report.source_url)
	for startup in report.startups:
		startup.linkedin_url = _normalize_linkedin_url(startup.linkedin_url)
		# Only normalize listing_url if it's not already an absolute URL
		if startup.listing_url:
			listing_url_str = str(startup.listing_url)
			# Check if it's already a full URL
			if not listing_url_str.startswith('http://') and not listing_url_str.startswith('https://'):
				startup.listing_url = _normalize_listing_url(listing_url_str, base_url)
			elif listing_url_str != base_url:
				# This is likely a malformed URL, try to extract the real URL
				# If the URL contains www. or http, try to extract it
				url_match = re.search(r'(www\.[^\s/]+|https?://[^\s/]+)', listing_url_str)
				if url_match:
					extracted_url = url_match.group(0)
					if not extracted_url.startswith('http'):
						extracted_url = 'https://' + extracted_url
					startup.listing_url = extracted_url
		# Clean up short_notes: remove empty strings and strip whitespace
		if startup.short_notes:
			startup.short_notes = [note.strip() for note in startup.short_notes if note.strip()]
	return report


def build_task(task_input: StartupListingInput) -> str:
	"""Create the natural-language instructions fed to the agent, specialized for Product Hunt."""

	# Determine if we want all startups or a limited number
	extract_all = task_input.max_startups >= 100
	is_product_hunt = 'producthunt.com' in str(task_input.url).lower()
	
	# Product Hunt specific instructions
	ph_specific = ""
	if is_product_hunt:
		ph_specific = """
		INSTRUCTIONS SPÉCIFIQUES PRODUCT HUNT:
		- Product Hunt affiche les produits dans des cartes avec un titre, une description, et des tags/catégories
		- Les produits sont organisés par sections: "Top Products Launching Today", "Yesterday's Top Products", etc.
		- CRITIQUE ÉTAPE 1: Descendre pour trouver et cliquer sur "See all of today's products"
		  • Scrolle VERS LE BAS sur la page d'accueil pour trouver le bouton/lien "See all of today's products"
		  • Le bouton peut être en bas de la section "Top Products Launching Today" visible sur la page d'accueil
		  • Utilise `scroll` avec `down: true` pour descendre et chercher ce bouton
		  • Une fois trouvé, utilise `click` pour cliquer dessus
		  • Attends que la page se charge complètement (utilise `wait` avec `seconds: 3`)
		- CRITIQUE ÉTAPE 2: Remonter tout en haut après avoir cliqué
		  • Une fois sur la page "Top Products Launching Today", remonte TOUT EN HAUT de la page
		  • Utilise `scroll` avec `down: false` ou `send_keys` avec "Home" pour remonter rapidement
		  • Assure-toi d'être bien positionné au début de la liste des produits avant de commencer l'extraction
		- IMPORTANT: Extrait UNIQUEMENT les produits de la section "Top Products Launching Today"
		- IGNORE complètement les autres sections comme "Yesterday's Top Products", "Last Week's Top Products", etc.
		- CONTINUE la boucle extrait/scroll jusqu'à voir clairement "Yesterday's Top Products", même si tu as déjà capturé {task_input.max_startups} produits
		- Pour chaque produit, capture:
		  • Le nom exact du produit (titre de la carte)
		  • L'URL complète vers la page Product Hunt du produit (https://www.producthunt.com/products/...)
		  • La description/tagline du produit
		  • Les tags/catégories visibles (ex: "Artificial Intelligence", "Productivity", etc.)
		- Les produits sont généralement dans des cartes avec une image, un titre cliquable, et une description
		- ⚠️ STRATÉGIE D'EXTRACTION ITÉRATIVE OBLIGATOIRE:
		- Tu DOIS faire PLUSIEURS extractions successives (une par écran scrollé)
		- Chaque extraction doit capturer UNIQUEMENT les produits visibles à l'écran au moment de l'appel
		- Ne fais JAMAIS une seule extraction qui essaie de capturer tous les produits d'un coup
		- Pattern: Extract (écran 1) -> Scroll -> Wait -> Extract (écran 2) -> Scroll -> Wait -> Extract (écran 3) -> etc.
		- Continue jusqu'à avoir collecté {task_input.max_startups} produits OU jusqu'à voir "Yesterday's Top Products"
		"""
	
	return dedent(
		f"""
		Tu es un agent spécialisé dans l'extraction de listings de produits depuis Product Hunt.

		Objectif CRITIQUE:
		{"- Identifie et extrait TOUS les produits présents sur cette page Product Hunt, SANS AUCUNE EXCEPTION." if extract_all else f"- Identifie et extrait jusqu'à {task_input.max_startups} produits présents sur cette page Product Hunt."}
		- IMPORTANT: Ne filtre PAS les produits. Prends TOUS les produits visibles sur la page.
		- Ne confonds PAS les titres de sections (comme "Top Products Launching Today") avec des produits réels.
		- Pour chaque produit, capture:
		  • `name`: nom exact du produit tel qu'affiché sur la carte
		  • `listing_url`: URL complète vers la page Product Hunt du produit (format: https://www.producthunt.com/products/nom-du-produit)
		  • `linkedin_url`: URL LinkedIn si visible sur la carte (laisse null sinon - généralement pas disponible sur Product Hunt)
		  • `description`: tagline/description du produit affichée sur la carte
		  • `short_notes`: tags/catégories visibles (ex: ["Artificial Intelligence", "Productivity"])

		Processus recommandé pour Product Hunt:
		1. ÉTAPE CRITIQUE: Descendre pour trouver et cliquer sur "See all of today's products"
		   - Scrolle VERS LE BAS sur la page d'accueil pour trouver le bouton/lien "See all of today's products"
		   - Le bouton peut être en bas de la section "Top Products Launching Today" visible sur la page d'accueil
		   - Utilise l'action `scroll` avec `down: true` pour descendre et chercher ce bouton
		   - Une fois trouvé, utilise l'action `click` pour cliquer sur ce lien/bouton
		   - Attends que la page se charge complètement après le clic (utilise `wait` avec `seconds: 3`)
		2. IMPORTANT: Remonter tout en haut après avoir cliqué
		   - Une fois sur la page "Top Products Launching Today", tu dois être en haut de la liste
		   - Utilise l'action `scroll` avec `down: false` ou `find_text` pour remonter tout en haut
		   - Tu peux aussi utiliser `send_keys` avec "Home" pour remonter rapidement en haut de la page
		   - Assure-toi d'être bien positionné au début de la liste des produits avant de commencer l'extraction
		3. BOUCLE D'EXTRACTION ITÉRATIVE (Répète jusqu'à avoir {task_input.max_startups} produits OU jusqu'à la fin de la section):
		   
		   ⚠️ CRITIQUE: Tu DOIS faire PLUSIEURS extractions successives. Ne fais PAS une seule grande extraction!
		   
		   RÉPÈTE cette séquence jusqu'à avoir collecté {task_input.max_startups} produits OU jusqu'à voir "Yesterday's Top Products":
		   
		   a. EXTRAIS UNIQUEMENT les produits visibles ACTUELLEMENT à l'écran (pas tous les produits de la page).
		      - Utilise `extract` avec `extract_links=true`.
		      - Query: "Extract ONLY the products currently visible on THIS SCREEN (not all products on the page). Capture: name, URL, description, tags. Stop extraction if you see 'Yesterday's Top Products'."
		      - Cette extraction ne doit capturer QUE les produits visibles maintenant (généralement 5-10 produits par écran).
		   
		   b. COMPTE combien de produits tu as collecté au total jusqu'à présent.
		      - Si tu as déjà {task_input.max_startups} produits ou plus -> passe à l'étape 4.
		      - Si tu vois "Yesterday's Top Products" -> passe à l'étape 4.
		   
		   c. SCROLLE vers le bas pour charger la suite.
		      - Utilise `scroll` avec `down: true` et `pages: 1`.
		      - Utilise `wait` avec `seconds: 1` pour laisser le contenu se charger.
		   
		   d. RETOURNE à l'étape 'a' pour faire une NOUVELLE extraction des produits maintenant visibles.
		   
		   RÈGLE ABSOLUE: 
		   - Chaque appel à `extract` doit capturer UNIQUEMENT les produits visibles à l'écran au moment de l'appel
		   - Tu dois faire PLUSIEURS appels à `extract` (un par écran scrollé)
		   - Ne fais JAMAIS une seule extraction qui essaie de capturer tous les produits d'un coup
		   - Pattern: Extract (écran 1) -> Scroll -> Extract (écran 2) -> Scroll -> Extract (écran 3) -> etc.
		   - Continue jusqu'à avoir {task_input.max_startups} produits OU jusqu'à voir "Yesterday's Top Products"
		   - Même si tu as déjà {task_input.max_startups} produits, poursuis la boucle jusqu'à ce que "Yesterday's Top Products" apparaisse clairement à l'écran

		4. Une fois que tu as atteint la section suivante ("Yesterday's Top Products") ou que tu as collecté {task_input.max_startups} produits:
		   - Lis TOUTES les extractions que tu as faites précédemment (elles sont disponibles dans ton historique)
		   - COMBINE toutes les extractions pour créer une liste complète et ordonnée de produits
		   - Les produits doivent être dans l'ordre où ils apparaissent sur la page (du premier au dernier)
		   - Limite la liste finale à {task_input.max_startups} produits maximum
		   - Utilise l'action `done` avec le champ `data` contenant l'objet `StartupListingReport` complet avec TOUS les produits combinés
		   - IMPORTANT: Le champ `source_url` doit être une CHAÎNE DE CARACTÈRES (string), pas un objet URL
		   - Format: {{"done": {{"success": true, "data": {{"source_url": "{task_input.url}", "startups": [...]}}}}}}

		Règles importantes:
		- ÉTAPE CRITIQUE 1: Descendre pour trouver et cliquer sur "See all of today's products" avant d'extraire
		  • Scrolle VERS LE BAS sur la page d'accueil pour trouver le bouton/lien "See all of today's products"
		  • Utilise `scroll` avec `down: true` pour descendre et chercher ce bouton
		  • Une fois trouvé, utilise `click` pour cliquer dessus
		  • Attends que la page se charge complètement (utilise `wait` avec `seconds: 3`)
		- ÉTAPE CRITIQUE 2: Remonter tout en haut après avoir cliqué
		  • Une fois sur la page "Top Products Launching Today", remonte TOUT EN HAUT de la page
		  • Utilise `scroll` avec `down: false` ou `send_keys` avec "Home" pour remonter rapidement
		  • Assure-toi d'être bien positionné au début de la liste des produits avant de commencer l'extraction
		- Reste strictement sur la page "Top Products Launching Today" après avoir cliqué
		- ⚠️ EXTRACTION ITÉRATIVE OBLIGATOIRE: Tu DOIS faire PLUSIEURS extractions successives (une par écran scrollé)
		- Chaque extraction doit capturer UNIQUEMENT les produits visibles à l'écran au moment de l'appel
		- Ne fais JAMAIS une seule extraction qui essaie de capturer tous les produits d'un coup
		- Pattern obligatoire: Extract (écran 1) -> Scroll -> Wait -> Extract (écran 2) -> Scroll -> Wait -> Extract (écran 3) -> etc.
		- Continue jusqu'à avoir collecté {task_input.max_startups} produits OU jusqu'à voir "Yesterday's Top Products"
		- Ne confonds PAS les titres de sections avec des produits réels
		- N'extrait QUE les produits de "Top Products Launching Today" - ignore "Yesterday's Top Products", "Last Week's Top Products", etc.
		- Les `short_notes` doivent contenir les tags/catégories visibles sur la carte produit
		- Si une info manque, laisse le champ null, mais ne l'invente pas
		- CRITIQUE SÉRIALISATION: Lorsque tu appelles `done`, assure-toi que `source_url` est une chaîne de caractères (string), pas un objet URL
		- Exemple correct: "source_url": "https://www.producthunt.com/"
		- Exemple incorrect: "source_url": AnyHttpUrl("https://www.producthunt.com/")
		- Utilise la vision et sois patient si le chargement est lent
		- Lorsque tu appelles `done`, combine TOUTES les extractions que tu as faites pour créer la liste complète de {task_input.max_startups} produits
		{ph_specific}
		"""
	).strip()


async def run_startup_listing(task_input: StartupListingInput) -> StartupListingReport | None:
	"""Execute the agent and return the structured list of startups."""

	print("🔧 Configuration du LLM...")
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
		# Pour l'extraction de page, utiliser un modèle plus petit et fiable
		page_extraction_llm = ChatBrowserUse()
		print("✅ Utilisation de ChatBrowserUse")
	else:
		# Utiliser gemini-2.5-flash-lite-preview-09-2025 par défaut (peut être surchargé par OPENAI_MODEL)
		model_name = os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		# Forcer l'utilisation de gemini-2.5-flash-lite-preview-09-2025 si un modèle non-Gemini est détecté
		if 'gemini' not in model_name.lower():
			model_name = 'gemini-2.5-flash-lite-preview-09-2025'
			print(f"⚠️  Modèle non-Gemini détecté, utilisation de {model_name} à la place")
		# Pour les modèles Gemini via LiteLLM, utiliser add_schema_to_system_prompt
		# pour éviter les problèmes de schéma JSON avec response_format
		is_gemini = 'gemini' in model_name.lower()
		llm = ChatOpenAI(
			model=model_name,
			timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
			max_retries=3,  # Augmenté pour plus de robustesse avec Gemini
			max_completion_tokens=18192,  # Augmenté pour éviter les troncatures JSON avec Gemini Flash Lite
			add_schema_to_system_prompt=is_gemini,  # Évite les problèmes de schéma avec Gemini
			dont_force_structured_output=is_gemini,  # Gemini via LiteLLM a des problèmes avec response_format
		)
		# Utiliser un modèle plus petit et rapide pour l'extraction de page
		# Cela réduit les erreurs de parsing JSON et améliore la performance
		extraction_model = os.getenv('PAGE_EXTRACTION_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		page_extraction_llm = ChatOpenAI(
			model=extraction_model,
			timeout=httpx.Timeout(120.0, connect=30.0, read=120.0, write=20.0),
			max_retries=2,
			max_completion_tokens=18192,  # Augmenté pour éviter les troncatures JSON avec Gemini Flash Lite
			add_schema_to_system_prompt=True,  # Toujours utiliser le schéma dans le prompt pour l'extraction
			dont_force_structured_output=True,  # Ne pas forcer le structured output pour éviter les erreurs
		)
		print(f"✅ Utilisation de ChatOpenAI avec le modèle: {model_name}")
		print(f"✅ Utilisation de ChatOpenAI pour l'extraction avec le modèle: {extraction_model}")
		if is_gemini:
			print("   ⚠️  Mode Gemini détecté: utilisation du schéma dans le prompt système")
			print("   💡 Note: Gemini peut parfois générer du JSON mal formé, mais les données seront récupérées depuis les extractions.")

	print("🤖 Création de l'agent...")
	agent = Agent(
		task=build_task(task_input),
		llm=llm,
		page_extraction_llm=page_extraction_llm,  # Utiliser un LLM dédié pour l'extraction
		output_model_schema=StartupListingReport,
		use_vision=True,
		vision_detail_level='high',
		step_timeout=300,
		llm_timeout=180,
		max_failures=5,
		max_history_items=10,  # Limiter l'historique pour éviter les réponses JSON trop longues
		directly_open_url=True,
	)
	print("✅ Agent créé")

	print("▶️  Démarrage de l'exécution de l'agent...")
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
			# Try to parse as JSON
			report = StartupListingReport.model_validate_json(final_result)
			if not agent_successful:
				print("⚠️  Données récupérées depuis le résultat final malgré l'échec de l'agent.")
			return _sanitize_report(report)
		except ValidationError:
			# Try to extract JSON from markdown code blocks
			json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
			if json_match:
				try:
					report = StartupListingReport.model_validate_json(json_match.group(1))
					if not agent_successful:
						print("⚠️  Données récupérées depuis le résultat final (markdown) malgré l'échec de l'agent.")
					return _sanitize_report(report)
				except ValidationError:
					pass

	# Try to extract from action results (especially extract actions)
	extracted_contents = history.extracted_content()
	# Try to combine all extracted contents if multiple extractions were made
	all_extracted_startups: list[StartupProfile] = []
	seen_names: set[str] = set()  # Avoid duplicates
	
	# Process extractions in chronological order (first to last) to preserve product order
	# This ensures products appear in the same order as they were extracted
	for content in extracted_contents:  # Process from first to last extraction
		if not content:
			continue
		
		# First try to parse markdown format from extract action
		markdown_report = _parse_extracted_markdown(content, str(task_input.url))
		if markdown_report and markdown_report.startups:
			# Add unique startups (avoid duplicates by name)
			# Process in order to preserve the ranking/order of products
			for startup in markdown_report.startups:
				if startup.name and startup.name not in seen_names:
					all_extracted_startups.append(startup)
					seen_names.add(startup.name)
	
	# If we found startups from markdown parsing, return them
	if all_extracted_startups:
		# Limit to max_startups if specified
		if task_input.max_startups < 1000:
			all_extracted_startups = all_extracted_startups[:task_input.max_startups]
		
		report = StartupListingReport(
			source_url=AnyHttpUrl(str(task_input.url)),
			startups=all_extracted_startups,
		)
		if not agent_successful:
			print(f"⚠️  Données récupérées depuis les résultats d'extraction (markdown) malgré l'échec de l'agent. {len(all_extracted_startups)} startups trouvées.")
		return _sanitize_report(report)
	
	# Fallback: try individual content parsing
	for content in reversed(extracted_contents):
		if not content:
			continue
		
		# Try to parse markdown format from extract action
		markdown_report = _parse_extracted_markdown(content, str(task_input.url))
		if markdown_report and markdown_report.startups:
			if not agent_successful:
				print(f"⚠️  Données récupérées depuis les résultats d'extraction (markdown) malgré l'échec de l'agent. {len(markdown_report.startups)} startups trouvées.")
			return _sanitize_report(markdown_report)
		
		try:
			# Try to parse as JSON directly
			report = StartupListingReport.model_validate_json(content)
			if not agent_successful:
				print("⚠️  Données récupérées depuis les résultats d'extraction malgré l'échec de l'agent.")
			return _sanitize_report(report)
		except ValidationError:
			# Try to extract JSON from markdown code blocks
			json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
			if json_match:
				try:
					report = StartupListingReport.model_validate_json(json_match.group(1))
					if not agent_successful:
						print("⚠️  Données récupérées depuis les résultats d'extraction (markdown JSON) malgré l'échec de l'agent.")
					return _sanitize_report(report)
				except ValidationError:
					pass
			# Try to find JSON object in the content
			json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content, re.DOTALL)
			if json_match:
				try:
					report = StartupListingReport.model_validate_json(json_match.group(1))
					if not agent_successful:
						print("⚠️  Données récupérées depuis les résultats d'extraction (JSON brut) malgré l'échec de l'agent.")
					return _sanitize_report(report)
				except ValidationError:
					pass

	# Try to extract from model actions (look for done actions with data)
	for action_dict in reversed(history.model_actions()):
		if 'done' in action_dict:
			done_data = action_dict.get('done', {})
			if isinstance(done_data, dict) and 'data' in done_data:
				data = done_data['data']
				# Convert AnyHttpUrl to string if needed (deep copy to avoid modifying original)
				data_copy = copy.deepcopy(data) if isinstance(data, dict) else data
				
				# Recursive function to convert AnyHttpUrl to string in nested structures
				def convert_urls_to_strings(obj: Any) -> Any:
					"""Recursively convert AnyHttpUrl objects to strings."""
					if isinstance(obj, dict):
						return {k: convert_urls_to_strings(v) for k, v in obj.items()}
					elif isinstance(obj, list):
						return [convert_urls_to_strings(item) for item in obj]
					elif hasattr(obj, '__str__') and not isinstance(obj, (str, int, float, bool, type(None))):
						# Check if it's an AnyHttpUrl-like object
						if 'HttpUrl' in type(obj).__name__ or 'Url' in type(obj).__name__:
							return str(obj)
					return obj
				
				data_copy = convert_urls_to_strings(data_copy)
				
				try:
					report = StartupListingReport.model_validate(data_copy)
					if not agent_successful:
						print("⚠️  Données récupérées depuis l'action 'done' malgré l'échec de l'agent.")
					return _sanitize_report(report)
				except ValidationError as e:
					# Try with JSON serialization first
					try:
						json_str = json.dumps(data_copy, default=str)
						report = StartupListingReport.model_validate_json(json_str)
						if not agent_successful:
							print("⚠️  Données récupérées depuis l'action 'done' (après conversion JSON) malgré l'échec de l'agent.")
						return _sanitize_report(report)
					except (ValidationError, json.JSONDecodeError):
						pass

	# If we get here, we couldn't extract any data
	if not agent_successful:
		print("❌ Impossible d'extraire les données malgré plusieurs tentatives.")
	return _fallback_report(str(task_input.url), "L'agent a été interrompu avant de finaliser le JSON.")


def parse_arguments() -> StartupListingInput:
	"""Validate CLI arguments via Pydantic before launching the agent."""

	parser = argparse.ArgumentParser(description='Construit un listing de startups depuis une page Product Hunt/BetaList/etc.')
	parser.add_argument('url', help='URL du listing (Product Hunt, BetaList, FutureTools, etc.)')
	parser.add_argument(
		'--max-startups',
		type=int,
		default=500,
		help='Nombre maximal de startups à extraire (par défaut: 12)',
	)
	parser.add_argument(
		'--output',
		default='startup_listings.json',
		help='Chemin du fichier JSON résultat (par défaut: ./startup_listings.json)',
	)
	args = parser.parse_args()
	return StartupListingInput(url=args.url, max_startups=args.max_startups, output_path=Path(args.output))


async def main() -> None:
	"""CLI entry point."""

	try:
		task_input = parse_arguments()
		print(f"🚀 Démarrage de l'agent pour: {task_input.url}")
		print(f"📊 Nombre max de startups: {task_input.max_startups}")
		print(f"💾 Fichier de sortie: {task_input.output_path}")
		
		result = await run_startup_listing(task_input)

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
