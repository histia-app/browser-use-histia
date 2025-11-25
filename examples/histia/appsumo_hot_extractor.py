"""
Agent designed to extract trending products from AppSumo's “What's hot” collection.
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
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from bs4.element import Tag
from dotenv import load_dotenv
from pydantic import AnyHttpUrl, BaseModel, Field, ValidationError, field_serializer

# Ensure environment variables (API keys, timeouts, etc.) are available to the agent.
load_dotenv()

os.environ.setdefault('TIMEOUT_ScreenshotEvent', '45')
os.environ.setdefault('TIMEOUT_BrowserStateRequestEvent', '90')
os.environ.setdefault('TIMEOUT_ScrollEvent', '15')

from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI
from browser_use.browser.events import NavigateToUrlEvent
from examples.histia import print_llm_usage_summary


class AppSumoHotInput(BaseModel):
	"""User-provided parameters for the AppSumo extraction task."""

	url: AnyHttpUrl = Field(
		default=AnyHttpUrl('https://appsumo.com/collections/whats-hot/'),
		description="URL of the AppSumo “What's hot” collection page",
	)
	max_products: int = Field(
		default=200,
		ge=1,
		le=2000,
		description='Maximum number of product cards to capture',
	)
	output_path: Path = Field(
		default=Path('appsumo_hot_products.json'),
		description='Destination file for the JSON list of products',
	)


class AppSumoProduct(BaseModel):
	"""Structured information for each AppSumo deal entry."""

	name: str = Field(..., description='Product name exactly as written on the card')
	product_url: str | None = Field(None, description='Absolute URL to the product detail page')
	category: str | None = Field(None, description='Product category label (e.g., “Video”)')
	category_url: str | None = Field(None, description='Absolute category URL if available')
	description: str | None = Field(None, description='Marketing blurb/description shown on the card')
	price: str | None = Field(None, description='Current AppSumo price, including symbol')
	price_suffix: str | None = Field(None, description='Price suffix such as “/lifetime”')
	original_price: str | None = Field(None, description='Strikethrough/original price text')
	reviews_count: int | None = Field(None, description='Number of AppSumo reviews')
	rating_value: float | None = Field(None, description='Average star rating (numeric)')
	rating_text: str | None = Field(None, description='Raw rating string (e.g., “4.7 stars”)')
	image_url: str | None = Field(None, description='Hero image URL')
	badges: list[str] = Field(default_factory=list, description='Deal badges (e.g., price increases info)')
	appsumo_select: bool = Field(False, description='True when the AppSumo Select badge is present')


class AppSumoHotReport(BaseModel):
	"""Complete response returned by the agent."""

	source_url: AnyHttpUrl = Field(..., description='AppSumo collection URL that was analysed')
	products: list[AppSumoProduct] = Field(
		...,
		min_length=1,
		description='Product entries ordered as they appear on the page',
	)

	@field_serializer('source_url')
	def serialize_source_url(self, value: AnyHttpUrl, _info) -> str:
		"""Convert AnyHttpUrl to string for JSON serialization."""
		return str(value)

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Override model_dump to ensure AnyHttpUrl values serialize as strings."""
		result = super().model_dump(**kwargs)
		source_url = result.get('source_url')
		if source_url and not isinstance(source_url, str):
			result['source_url'] = str(source_url)
		return result

	def model_dump_json(self, **kwargs) -> str:
		"""Override model_dump_json for consistent serialization."""
		return super().model_dump_json(**kwargs)


def _normalize_url(url: str | None, base_url: str) -> str | None:
	"""Convert relative URLs to absolute URLs."""
	if not url:
		return None

	url = url.strip()
	if not url:
		return None

	if url.startswith(('http://', 'https://')):
		return url

	if url.startswith('/'):
		parsed_base = urlparse(base_url)
		return f'{parsed_base.scheme}://{parsed_base.netloc}{url}'

	try:
		return urljoin(base_url, url)
	except Exception:
		return None


def _clean_text(value: str | None) -> str | None:
	"""Remove excessive whitespace and return None for empty strings."""
	if not value:
		return None
	text = re.sub(r'\s+', ' ', value).strip()
	return text or None


def _parse_reviews_count(text: str | None) -> int | None:
	if not text:
		return None
	match = re.search(r'(\d[\d,]*)', text)
	if not match:
		return None
	try:
		return int(match.group(1).replace(',', ''))
	except ValueError:
		return None


def _parse_rating_value(text: str | None) -> float | None:
	if not text:
		return None
	match = re.search(r'([0-9]+(?:\.[0-9]+)?)', text)
	if not match:
		return None
	try:
		return float(match.group(1))
	except ValueError:
		return None


def _extract_badges(soup: BeautifulSoup) -> list[str]:
	"""Retrieve badge texts such as “Price increases in 4 days”."""
	badges: list[str] = []
	for span in soup.select('div span'):
		text = _clean_text(span.get_text())
		if not text:
			continue
		lower = text.lower()
		if any(keyword in lower for keyword in ('price', 'black friday', 'ending soon')):
			badges.append(text)
	return badges


def _safe_attr(element: Tag | None, attribute: str) -> str | None:
	"""Return a string attribute value even when BeautifulSoup yields list types."""
	if not element:
		return None
	value = element.get(attribute)
	if isinstance(value, list):
		text = ' '.join(item for item in value if isinstance(item, str)).strip()
		return text or None
	if isinstance(value, str):
		return value
	return None


def _extract_rating_info(card: BeautifulSoup) -> tuple[float | None, str | None]:
	rating_img = card.find('img', alt=lambda value: bool(value and 'star' in value.lower()))
	if rating_img:
		raw_text = _clean_text(_safe_attr(rating_img, 'alt'))
		return _parse_rating_value(raw_text), raw_text

	rating_container = card.find(string=re.compile(r'\bstars?\b', re.IGNORECASE))
	if rating_container:
		text = _clean_text(str(rating_container))
		return _parse_rating_value(text), text
	return None, None


def _parse_product_card(html_section: str, source_url: str) -> AppSumoProduct | None:
	"""Parse a single product card HTML snippet."""
	if not html_section:
		return None

	soup = BeautifulSoup(html_section, 'html.parser')

	name = None
	name_selectors = [
		'span.sr-only',
		'span.font-bold',
		'a[aria-label]',
	]
	for selector in name_selectors:
		element = soup.select_one(selector)
		if element:
			name = _clean_text(element.get_text())
			if name:
				break

	if not name:
		return None

	link_elem = soup.select_one('a[href^="/products/"]')
	product_url = _normalize_url(_safe_attr(link_elem, 'href'), source_url)

	category_link = soup.select_one('span a[href*="/software/"], span a[href*="/courses/"], span a[href*="/creative/"]')
	category = _clean_text(category_link.get_text() if category_link else None)
	category_url = _normalize_url(_safe_attr(category_link, 'href'), source_url)

	description_elem = soup.select_one('div[class*="line-clamp"], div[class*="text-center"] div[class*="line-clamp"]')
	description = _clean_text(description_elem.get_text() if description_elem else None)

	price_elem = soup.select_one('#deal-price')
	price = _clean_text(price_elem.get_text() if price_elem else None)

	price_suffix_elem = soup.select_one('#deal-price-suffix')
	price_suffix = _clean_text(price_suffix_elem.get_text() if price_suffix_elem else None)

	original_price_elem = soup.select_one('#deal-price-original')
	original_price = _clean_text(original_price_elem.get_text() if original_price_elem else None)

	reviews_link = soup.select_one('a[href*="#reviews"] span, a[href*="#reviews"]')
	reviews_count = _parse_reviews_count(reviews_link.get_text() if reviews_link else None)

	rating_value, rating_text = _extract_rating_info(soup)

	image_elem = None
	image_selectors = [
		'img.aspect-sku-card',
		'img.rounded-t',
		f'img[alt="{name}"]',
		'img[decoding="async"]',
	]
	for selector in image_selectors:
		image_elem = soup.select_one(selector)
		if image_elem:
			break
	if not image_elem:
		image_elem = soup.select_one('img[alt][src^="http"], img[data-nimg]')
	image_url = _normalize_url(_safe_attr(image_elem, 'src'), source_url)

	appsumo_select = bool(soup.select_one('img[alt="AppSumo Select"]'))
	badges = _extract_badges(soup)

	return AppSumoProduct(
		name=name,
		product_url=product_url,
		category=category,
		category_url=category_url,
		description=description,
		price=price,
		price_suffix=price_suffix,
		original_price=original_price,
		reviews_count=reviews_count,
		rating_value=rating_value,
		rating_text=rating_text,
		image_url=image_url,
		badges=badges,
		appsumo_select=appsumo_select,
	)


def _parse_html_sections(html_sections: list[str] | str, source_url: str) -> AppSumoHotReport | None:
	"""Parse HTML sections from the evaluate action to build AppSumoHotReport."""
	products: list[AppSumoProduct] = []
	seen_names: set[str] = set()

	if isinstance(html_sections, str):
		try:
			html_sections = json.loads(html_sections)
		except json.JSONDecodeError:
			html_sections = [html_sections]

	if not isinstance(html_sections, list):
		return None

	for html_section in html_sections:
		if not isinstance(html_section, str):
			continue
		product = _parse_product_card(html_section, source_url)
		if not product:
			continue
		name_key = product.name.lower()
		if name_key in seen_names:
			continue
		seen_names.add(name_key)
		products.append(product)

	if not products:
		return None

	return AppSumoHotReport(
		source_url=AnyHttpUrl(source_url),
		products=products,
	)


async def _scroll_to_bottom(browser: Browser, max_scrolls: int = 20) -> None:
	"""Perform a deterministic scroll to trigger lazy loading."""
	page = await browser.get_current_page()
	if not page:
		raise RuntimeError('No current page available for scrolling')

	viewport_height = await page.evaluate('() => window.innerHeight || document.documentElement.clientHeight') or 900
	viewport_height = int(viewport_height)

	await asyncio.sleep(3)
	last_position = -1
	retries_without_progress = 0

	for _ in range(max_scrolls):
		current_position = await page.evaluate('() => window.pageYOffset || document.documentElement.scrollTop') or 0
		current_position = int(current_position)

		if current_position == last_position:
			retries_without_progress += 1
			if retries_without_progress >= 3:
				break
		else:
			retries_without_progress = 0

		last_position = current_position
		await page.evaluate(f'() => window.scrollBy(0, {viewport_height})')
		await asyncio.sleep(1.5)

	await asyncio.sleep(2)


async def run_appsumo_hot_extraction(task_input: AppSumoHotInput) -> AppSumoHotReport | None:
	"""Execute the agent flow and return the structured list of products."""
	if os.getenv('BROWSER_USE_API_KEY'):
		llm = ChatBrowserUse()
	else:
		model_name = os.getenv('OPENAI_MODEL', 'gemini-2.5-flash-lite-preview-09-2025')
		if 'gemini' not in model_name.lower():
			model_name = 'gemini-2.5-flash-lite-preview-09-2025'
		llm = ChatOpenAI(
			model=model_name,
			timeout=httpx.Timeout(180.0, connect=60.0, read=180.0, write=30.0),
			max_retries=3,
			max_completion_tokens=15000,
			add_schema_to_system_prompt='gemini' in model_name.lower(),
			dont_force_structured_output='gemini' in model_name.lower(),
		)

	browser = Browser(headless=False)
	await browser.start()

	try:
		appsumo_url = str(task_input.url)
		navigate_event = NavigateToUrlEvent(url=appsumo_url, new_tab=False)
		await browser.event_bus.dispatch(navigate_event)
		await navigate_event

		page = await browser.get_current_page()
		if not page:
			page = await browser.new_page(appsumo_url)

		await asyncio.sleep(5)
		await _scroll_to_bottom(browser)

		if not page:
			raise RuntimeError('No active page for extraction')

		extraction_code = """() => {
			const cards = Array.from(document.querySelectorAll('div.relative.h-full'))
				.filter(card => card.querySelector('a[href^="/products/"]'));
			return JSON.stringify(cards.map(card => card.outerHTML));
		}"""

		html_sections_json = await page.evaluate(extraction_code)
		html_sections = json.loads(html_sections_json) if html_sections_json else []

		if html_sections:
			report = _parse_html_sections(html_sections, appsumo_url)
			if report and report.products:
				if task_input.max_products < len(report.products):
					report.products = report.products[: task_input.max_products]
				return report

		task = dedent(
			f"""
			Tu es un agent spécialisé dans l'extraction de données AppSumo.

			Objectif:
			- Tu es déjà sur la page: {appsumo_url}
			- Utilise l'action `evaluate` pour récupérer le HTML de chaque carte produit.
			- Pour chaque produit, capture:
			  • name
			  • product_url
			  • category & category_url
			  • description
			  • price, price_suffix, original_price
			  • reviews_count, rating_value
			  • badges (text tels que "Price increases in 4 days")
			  • booléen appsumo_select

			Important:
			- N'effectue aucune navigation supplémentaire.
			- N'utilise pas `scroll` (le contenu est déjà chargé).
			- Utilise `evaluate` pour renvoyer un tableau JSON contenant `outerHTML` de chaque carte.
			- Utilise `done` avec un objet AppSumoHotReport complet dans `data`.
			"""
		).strip()

		agent = Agent(
			task=task,
			llm=llm,
			browser=browser,
			output_model_schema=AppSumoHotReport,
			use_vision='auto',
			vision_detail_level='auto',
			step_timeout=300,
			llm_timeout=180,
			max_failures=5,
			max_history_items=10,
			directly_open_url=False,
		)
		history = await agent.run()
		print_llm_usage_summary(history)

		if history.structured_output:
			return history.structured_output  # type: ignore[arg-type]

		final_result = history.final_result()
		if final_result:
			try:
				return AppSumoHotReport.model_validate_json(final_result)
			except ValidationError:
				match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', final_result, re.DOTALL)
				if match:
					try:
						return AppSumoHotReport.model_validate_json(match.group(1))
					except ValidationError:
						return None
		return None
	finally:
		try:
			await browser.kill()
		except Exception:
			pass


def parse_arguments() -> AppSumoHotInput:
	"""Validate CLI arguments via Pydantic before launching the agent.

	Examples:
	--------
	python examples/histia/appsumo_hot_extractor.py --max-products 50 --output appsumo.json
	"""

	parser = argparse.ArgumentParser(description='Extract products from AppSumo “What’s hot” page')
	parser.add_argument(
		'--url',
		default='https://appsumo.com/collections/whats-hot/',
		help='AppSumo collection URL (default: https://appsumo.com/collections/whats-hot/)',
	)
	parser.add_argument(
		'--max-products',
		type=int,
		default=200,
		help='Maximum number of products to extract (default: 200)',
	)
	parser.add_argument(
		'--output',
		default='appsumo_hot_products.json',
		help='Output JSON file path (default: ./appsumo_hot_products.json)',
	)
	args = parser.parse_args()
	return AppSumoHotInput(url=AnyHttpUrl(args.url), max_products=args.max_products, output_path=Path(args.output))


async def main() -> None:
	"""CLI entry point."""
	try:
		task_input = parse_arguments()
		report = await run_appsumo_hot_extraction(task_input)

		if report is None:
			print('❌ Aucun produit structuré retourné.')
			return

		report_json = report.model_dump_json(indent=2, ensure_ascii=False)
		task_input.output_path.parent.mkdir(parents=True, exist_ok=True)
		task_input.output_path.write_text(report_json, encoding='utf-8')

		print(report_json)
		print(f'\n✅ Listing sauvegardé dans: {task_input.output_path.resolve()}')
	except KeyboardInterrupt:
		print('\n⚠️  Interruption utilisateur détectée.')
	except Exception as exc:
		print(f"❌ Erreur lors de l'extraction: {exc}")
		raise


if __name__ == '__main__':
	asyncio.run(main())


