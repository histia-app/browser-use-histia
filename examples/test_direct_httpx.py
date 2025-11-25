"""Test direct avec httpx pour comparer avec curl."""

import asyncio
import os

import httpx
from dotenv import load_dotenv

from browser_use.llm.openai.utils import normalize_openai_base_url

load_dotenv()


async def test_direct_httpx():
	"""Test direct avec httpx (comme curl)."""
	base_url = normalize_openai_base_url(os.getenv('OPENAI_API_URL'))
	if not base_url:
		print('❌ OPENAI_API_URL non défini')
		return False

	url = base_url.rstrip('/') + '/chat/completions'
	api_key = os.getenv('OPENAI_API_KEY')
	
	print(f'📤 Test direct avec httpx')
	print(f'   Base URL normalisée: {base_url}')
	print(f'   Endpoint complet: {url}')
	print(f'   Timeout: 30s')
	
	async with httpx.AsyncClient(timeout=30.0) as client:
		try:
			response = await client.post(
				url,
				headers={
					'Authorization': f'Bearer {api_key}',
					'Content-Type': 'application/json',
				},
				json={
					'model': 'gemini-2.5-flash-lite-preview-09-2025',
					'messages': [{'role': 'user', 'content': 'Say hello'}],
					'max_tokens': 10,
				},
			)
			print(f'\n✅ Réponse reçue: HTTP {response.status_code}')
			print(f'   Temps: < 30s')
			print(f'   Contenu: {response.text[:200]}')
			return True
		except httpx.TimeoutException:
			print(f'\n❌ Timeout après 30s')
			print('   httpx timeout - le serveur ne répond pas')
			return False
		except Exception as e:
			print(f'\n❌ Erreur: {type(e).__name__}: {str(e)}')
			return False


if __name__ == '__main__':
	asyncio.run(test_direct_httpx())











