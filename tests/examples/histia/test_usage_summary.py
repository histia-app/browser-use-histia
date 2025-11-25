"""
Unit tests for the shared Histia LLM usage printing helper.
"""

from __future__ import annotations

from browser_use.agent.views import AgentHistoryList
from browser_use.tokens.views import ModelUsageStats, UsageSummary

from examples.histia import print_llm_usage_summary


def _make_usage_summary() -> UsageSummary:
	"""Return a UsageSummary fixture with deterministic values."""

	return UsageSummary(
		total_prompt_tokens=123,
		total_prompt_cost=0.0123,
		total_prompt_cached_tokens=0,
		total_prompt_cached_cost=0.0,
		total_completion_tokens=456,
		total_completion_cost=0.0456,
		total_tokens=579,
		total_cost=0.0579,
		entry_count=3,
		by_model={
			'ChatBrowserUse': ModelUsageStats(
				model='ChatBrowserUse',
				prompt_tokens=100,
				completion_tokens=200,
				total_tokens=300,
				cost=0.03,
				invocations=2,
				average_tokens_per_invocation=150.0,
			)
		},
	)


def test_print_llm_usage_summary_displays_cost(capsys) -> None:
	"""Ensure helper prints aggregated cost and token counts."""

	history = AgentHistoryList(history=[], usage=_make_usage_summary())

	print_llm_usage_summary(history)

	captured = capsys.readouterr().out
	assert 'Coût total' in captured
	assert '$0.0579' in captured
	assert '579 tokens' in captured


def test_print_llm_usage_summary_handles_missing_usage(capsys) -> None:
	"""Ensure helper degrades gracefully when usage data is absent."""

	history = AgentHistoryList(history=[], usage=None)
	print_llm_usage_summary(history)

	captured = capsys.readouterr().out
	assert 'Aucune donnée de consommation LLM' in captured

