"""A real FunctionAgent with TopK tools picks the right search and uses long-term memory.

Needs TOPK_API_KEY and ANTHROPIC_API_KEY. Set TOPK_TEST_LLM to try another Claude model.
"""

import pytest
from llama_index.core.agent.workflow import FunctionAgent, ToolCall

from llama_index.tools.topk import TopKMemoryToolSpec, TopKToolSpec

pytestmark = [pytest.mark.integration, pytest.mark.agent]

SYSTEM = "You answer questions using the search tools. Always search before answering. Be brief."


async def run(agent, message):
    handler = agent.run(message)
    calls = [ev async for ev in handler.stream_events() if isinstance(ev, ToolCall)]
    return str(await handler), calls


@pytest.fixture
def search_agent(runbook_store, llm):
    tools = TopKToolSpec(
        runbook_store,
        corpus_description="an engineering runbook: error codes, incidents and product notes",
        metadata_fields={"topic": "payments, checkout, inventory, performance or product", "year": "integer"},
    ).to_tool_list()
    return FunctionAgent(tools=tools, llm=llm, system_prompt=SYSTEM)


async def test_agent_uses_keyword_search_for_an_exact_error_code(search_agent):
    answer, calls = await run(search_agent, "What does error E-4021 mean?")
    assert calls[0].tool_name == "keyword_search", [c.tool_name for c in calls]
    assert "E-4021" in calls[0].tool_kwargs["query"]
    assert "gateway" in answer.lower()


async def test_agent_uses_meaning_based_search_for_a_paraphrase(search_agent):
    answer, calls = await run(search_agent, "Why does the site feel sluggish right after we ship a release?")
    assert calls[0].tool_name in {"semantic_search", "hybrid_search"}, [c.tool_name for c in calls]
    assert "cdn" in answer.lower() or "cache" in answer.lower()


async def test_agent_filters_by_metadata(search_agent):
    answer, calls = await run(search_agent, "List the payments error codes from 2024 only.")
    filtered = [c for c in calls if c.tool_kwargs.get("filters")]
    assert filtered, [(c.tool_name, c.tool_kwargs) for c in calls]
    assert "E-4021" in answer and "E-4012" in answer
    assert "E-4201" not in answer


async def test_agent_remembers_across_sessions_per_user(store, llm):
    def agent_for(user_id):
        tools = TopKMemoryToolSpec(store, user_id=user_id).to_tool_list()
        prompt = "You are a personal assistant with long-term memory tools. Be brief."
        return FunctionAgent(tools=tools, llm=llm, system_prompt=prompt)

    _, calls = await run(agent_for("user-alice"), "Please remember that I'm allergic to peanuts.")
    assert "remember" in [c.tool_name for c in calls]

    # A brand-new agent run (no shared chat history) for the same user.
    answer, calls = await run(agent_for("user-alice"), "Suggest a snack for me. Check what you know about me first.")
    assert "recall" in [c.tool_name for c in calls]
    assert "peanut" in answer.lower()

    # Another user's agent can't see Alice's memories.
    answer, calls = await run(agent_for("user-bob"), "Do I have any food allergies? Check your memory.")
    assert "recall" in [c.tool_name for c in calls]
    assert "peanut" not in answer.lower()
