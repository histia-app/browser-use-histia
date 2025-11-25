"""
Histia agents collection.

This directory contains specialized agents for various web automation tasks.
"""

from __future__ import annotations

from browser_use.agent.views import AgentHistoryList


def _format_currency(amount: float) -> str:
	"""Return a USD currency string with 4 decimal places."""
	return f'${amount:.4f}'


def print_llm_usage_summary(history: AgentHistoryList | None) -> None:
	"""
	Print a concise LLM consumption summary for a given agent history.

	Example:
		>>> history = await agent.run()
		>>> print_llm_usage_summary(history)
	"""

	if history is None:
		print("ℹ️ Impossible d'afficher la consommation LLM: historique indisponible.")
		return

	usage = history.usage
	if usage is None:
		print("ℹ️ Aucune donnée de consommation LLM n'a été renvoyée par l'agent.")
		return

	print("\n💰 Consommation LLM")
	print(
		f"   Coût total : {_format_currency(usage.total_cost)}"
		f" pour {usage.total_tokens:,} tokens"
	)
	print(
		f"   Prompt      : {usage.total_prompt_tokens:,} tokens"
		f" ({_format_currency(usage.total_prompt_cost)})"
	)
	print(
		f"   Completion  : {usage.total_completion_tokens:,} tokens"
		f" ({_format_currency(usage.total_completion_cost)})"
	)

