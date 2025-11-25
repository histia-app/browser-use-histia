"""
Agent that extracts structured company and product information from launch listings
such as Product Hunt or BetaList.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError

# Load environment variables before importing browser_use so timeout overrides apply.
load_dotenv()
os.environ.setdefault('TIMEOUT_ScreenshotEvent', '25')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '45')

from browser_use import Agent, ChatBrowserUse, ChatOpenAI
from examples.histia import print_llm_usage_summary


class ProductResearchInput(BaseModel):
	"""User provided parameters for the research task."""

	url: AnyHttpUrl = Field(..., description='Product Hunt, BetaList or similar listing URL')
	max_products: int = Field(
		3,
		ge=1,
		le=10,
		description='Maximum number of products or variants to summarize',
	)
	output_path: Path = Field(
		default=Path('product_research_report.json'),
		description='Destination path for the generated JSON report',
	)


class CompanyProfile(BaseModel):
	"""Structured representation of the company behind the launch."""

	name: str = Field(..., description="Official company name exactly as written on the listing")
	logo_url: str | None = Field(None, description='Absolute URL to the company logo if available')
	description: str | None = Field(None, description='Short positioning statement about the company')
	official_website: str | None = Field(None, description='Primary website URL for the company')
	linkedin_page: str | None = Field(None, description='Public LinkedIn company page URL')
	other_facts: list[str] = Field(
		default_factory=list,
		description='Short bullet-style facts such as funding, metrics, or founding date',
	)


class ProductOverview(BaseModel):
	"""Structured information about a single product or plan."""

	product_name: str = Field(..., description='Official product or plan name')
	what_it_does: str = Field(..., description='One sentence that explains the core value proposition')
	go_to_market: str | None = Field(
		None,
		description='Business model classification such as B2B, B2C, B2G, marketplace, etc.',
	)
	target_audience: str | None = Field(
		None,
		description='Primary personas or industries targeted by the product',
	)
	description: str | None = Field(
		None,
		description='A slightly longer description of the product, features, or differentiators',
	)


class ProductResearchReport(BaseModel):
	"""Complete JSON report returned by the agent."""

	company: CompanyProfile
	products: list[ProductOverview] = Field(
		...,
		min_length=1,
		description='Top products or plans mentioned in the listing ordered by relevance',
	)


def _fallback_report(source_url: str, reason: str) -> ProductResearchReport:
	"""Return a minimal, but schema-compliant report when the agent fails."""

	reason = reason.strip() or "Impossible d'obtenir les données demandées."
	company = CompanyProfile(
		name='',
		logo_url=None,
		description=None,
		official_website=str(source_url),
		linkedin_page=None,
		other_facts=[
			reason,
			'Rapport généré automatiquement sans données fiables (agent interrompu avant la fin).',
		],
	)
	placeholder_product = ProductOverview(
		product_name='Informations indisponibles',
		what_it_does='Le produit n’a pas pu être extrait (l’agent a été interrompu).',
		go_to_market=None,
		target_audience=None,
		description=reason,
	)
	return ProductResearchReport(company=company, products=[placeholder_product])


def _normalize_linkedin_url(value: str | None) -> str | None:
	"""Ensure the linkedin_page field only contains a real LinkedIn URL."""

	if not value:
		return None

	url = value.strip()
	if not url:
		return None

	# Require full URL with https/http schema
	if not url.lower().startswith(('http://', 'https://')):
		return None

	parsed = urlparse(url)
	if not parsed.netloc:
		return None

	if 'linkedin.com' not in parsed.netloc.lower():
		return None

	return url


def _sanitize_report(report: ProductResearchReport) -> ProductResearchReport:
	"""Apply post-processing rules (e.g., LinkedIn URL validation)."""

	report.company.linkedin_page = _normalize_linkedin_url(report.company.linkedin_page)
	return report


def build_task(task_input: ProductResearchInput) -> str:
	"""Create the natural language instructions for the agent."""

	return dedent(
		f"""
		Analyse the listing available at {task_input.url}.

		Steps:
		1. Capture the exact company name, logo, a short description, official website, and the LinkedIn company page.
		   Only collect the raw links that are already exposed on the listing (no external searches).
		2. Provide up to {task_input.max_products} distinct products or plans. Focus on the core value, business model
		   (B2B/B2C/B2G/etc.), audience, and a concise description for each using only the text from the original listing page.
		3. Add a list of other short facts (funding round, launch metrics, founders, notable partnerships, etc.) sourced from
		   the listing itself.

		Important:
		- Do NOT follow external links for LinkedIn, website, or product descriptions. Simply record the links that the listing provides.
		- Factorise les offres: s'il existe plusieurs formules d'un même produit (par ex. version gratuite, abonnement annuel, lifetime), regroupe-les dans une seule fiche produit et détaille les variantes dans la description ou les autres faits. Ne duplique pas artificiellement les produits.
		- Only keep `linkedin_page` when the page exposes a real LinkedIn URL (must include https://...linkedin.com/). Otherwise set it to null.
		- Pour chaque URL (LinkedIn, site officiel, pages produits, promos…), récupère systématiquement l'attribut `href` exact depuis le code source/DOM (ex: via `extract_links=true`) et colle l'URL complète au lieu du texte du bouton.
		- When scrolling, always call the `scroll` action with BOTH `down` (True for scrolling down, False for scrolling up) and `pages` (use decimals for partial pages). Example: `{{"scroll": {{"down": true, "pages": 1}}}}`.
		- Sois extrêmement exhaustif: fais défiler toute la page, explore tous les onglets utiles du listing, et n'oublie aucun fait notable disponible.
		- Every action response MUST be valid JSON that strictly follows the agent schema (double-check commas, quotes, and brackets). Pour les actions `extract`, FOURNIS TOUJOURS un champ `query` explicite et `extract_links=true` si tu dois récupérer des URLs — ne renvoie JAMAIS `{{"extract": {{"extract_links": true}}}}` sans prompt.
		- **Ne termine jamais tant que tu n'as pas construit un objet `ProductResearchReport`.** Si tu es bloqué ou que des données manquent, remplis les champs avec des chaînes/listes vides, documente le blocage dans `other_facts`, mets `success=false`, puis appelle `done` avec `data=<rapport>`.
		- Utilise la capture de screenshots/vision pour valider ce que tu lis, mais en cas de lenteur, patiente et réessaie plutôt que d'abandonner.
		- Keep every field short, fact-based, and in French when possible.
		- Return the final answer strictly as JSON that validates the ProductResearchReport schema.
		"""
	).strip()


async def run_product_research(task_input: ProductResearchInput) -> ProductResearchReport | None:
	"""Execute the agent and return the structured report."""

	# Try ChatBrowserUse first, fallback to ChatOpenAI if API key not available
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
	else:
		# Fallback to ChatOpenAI (works with LiteLLM, OpenAI, etc.)
		llm = ChatOpenAI(
			model=os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-lite-preview-09-2025'),
			timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
			max_retries=2,
			add_schema_to_system_prompt=False,
			remove_min_items_from_schema=True,
			remove_defaults_from_schema=True,
			dont_force_structured_output=True,
		)

	agent = Agent(
		task=build_task(task_input),
		llm=llm,
		output_model_schema=ProductResearchReport,
		use_vision=True,
		vision_detail_level='high',
		step_timeout=300,
		llm_timeout=180,
	)

	history = await agent.run()
	print_llm_usage_summary(history)
	if history.structured_output:
		return _sanitize_report(history.structured_output)  # type: ignore[arg-type]

	final_result = history.final_result()
	if not final_result:
		return _fallback_report(
			str(task_input.url),
			"L'agent a été interrompu avant de produire le JSON structuré attendu.",
		)

	try:
		return _sanitize_report(ProductResearchReport.model_validate_json(final_result))
	except ValidationError as exc:
		return _fallback_report(
			str(task_input.url),
			f'Impossible de parser le JSON structuré retourné par le modèle ({exc.__class__.__name__}).',
		)


def parse_arguments() -> ProductResearchInput:
	"""Validate CLI arguments with Pydantic before running the agent."""

	parser = argparse.ArgumentParser(
		description='Collect company and product insights from Product Hunt, BetaList, or similar listings.',
	)
	parser.add_argument('url', help='Product Hunt, BetaList, or landing page URL to analyse')
	parser.add_argument(
		'--max-products',
		type=int,
		default=3,
		help='Maximum number of products/variants to summarize (default: 3)',
	)
	parser.add_argument(
		'--output',
		default='product_research_report.json',
		help='Path where the resulting JSON report will be written (default: ./product_research_report.json)',
	)
	args = parser.parse_args()
	return ProductResearchInput(url=args.url, max_products=args.max_products, output_path=Path(args.output))


async def main() -> None:
	"""CLI entrypoint."""

	task_input = parse_arguments()
	result = await run_product_research(task_input)

	if result is None:
		print('No structured output was returned by the agent.')
		return

	output_json = result.model_dump_json(indent=2, ensure_ascii=False)
	output_path = task_input.output_path
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(output_json, encoding='utf-8')

	print(output_json)
	print(f'\nReport saved to: {output_path.resolve()}')


if __name__ == '__main__':
	asyncio.run(main())
