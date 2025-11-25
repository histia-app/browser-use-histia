"""
Test ultra-simple de connexion LLM sans aucun schéma structuré.
"""

from browser_use import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import httpx
import time
import os

load_dotenv()
### Load the environment variable

async def test_simple():
	"""Test le plus simple possible - juste un appel chat sans schéma."""
	print("🔍 Test ultra-simple (sans schéma structuré)...")
	print(f"   URL: {os.getenv('OPENAI_API_URL', 'Non défini')}")
	print(f"   Modèle: gemini-2.5-flash-lite-preview-09-2025")
	print()
	
	# Configuration minimale - pas de schéma, pas de structured output
	# IMPORTANT: Utiliser un timeout HTTP plus long pour la connexion
	# Le client OpenAI peut avoir besoin de plus de temps pour établir la connexion
	llm = ChatOpenAI(
		model="gemini-2.5-flash-lite-preview-09-2025",
		timeout=httpx.Timeout(120.0, connect=60.0, read=120.0, write=30.0),  # Timeouts détaillés
		# Pas de schéma du tout
		add_schema_to_system_prompt=False,
		dont_force_structured_output=True,
		max_retries=1,  # Réduire les retries pour éviter les timeouts cumulés
	)
	
	# Message ultra-simple
	from browser_use.llm.messages import UserMessage
	messages = [UserMessage(content="Say hello")]
	
	print("📤 Envoi du message simple (timeout: 120s)...")
	start_time = time.time()
	
	try:
		response = await asyncio.wait_for(
			llm.ainvoke(messages),  # Pas de output_format = pas de schéma
			timeout=125.0  # Légèrement plus que le timeout HTTP
		)
		elapsed = time.time() - start_time
		print(f"\n✅ Réussi en {elapsed:.2f} secondes!")
		print(f"   Réponse: {response.completion}")
		return True
	except asyncio.TimeoutError:
		elapsed = time.time() - start_time
		print(f"\n❌ Timeout après {elapsed:.2f} secondes")
		print("   Le serveur LiteLLM ne répond pas même pour un appel simple.")
		print("   💡 Problème probable:")
		print("      - Latence réseau très élevée vers le serveur")
		print("      - Serveur LiteLLM surchargé")
		print("      - Problème de configuration du serveur")
		return False
	except Exception as e:
		elapsed = time.time() - start_time
		print(f"\n❌ Erreur après {elapsed:.2f} secondes")
		print(f"   {type(e).__name__}: {str(e)}")
		return False


if __name__ == "__main__":
	asyncio.run(test_simple())

