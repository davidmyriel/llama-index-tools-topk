"""Live tests of the search tools, memory tools and memory block (no LLM needed)."""

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import Memory

from llama_index.tools.topk import TopKMemoryBlock, TopKMemoryToolSpec, TopKToolSpec

pytestmark = pytest.mark.integration


def test_keyword_search_finds_the_exact_code(runbook_store):
    out = TopKToolSpec(runbook_store).keyword_search("E-4021")
    assert out.startswith("[1] id=E-4021")
    assert "E-4012" not in out


def test_keyword_search_says_when_nothing_matches(runbook_store):
    assert TopKToolSpec(runbook_store).keyword_search("E-9999") == "No matching documents."


def test_semantic_search_matches_paraphrases(runbook_store):
    out = TopKToolSpec(runbook_store).semantic_search("why do people stop paying for the product?", top_k=1)
    assert "id=churn" in out


def test_hybrid_search_with_filters(runbook_store):
    spec = TopKToolSpec(runbook_store)
    out = spec.hybrid_search("payment problems", filters={"topic": "payments", "year": {"$gte": 2024}})
    assert {"id=E-4021", "id=E-4012"} <= set(line.split()[1] for line in out.splitlines() if line.startswith("["))
    assert "E-4201" not in out


async def test_async_tools(runbook_store):
    spec = TopKToolSpec(runbook_store)
    assert (await spec.akeyword_search("E-2041")).startswith("[1] id=E-2041")
    assert "id=slow-pages" in await spec.ahybrid_search("site feels sluggish right after we ship", top_k=1)


def test_memory_tools_remember_recall_forget(store):
    alice = TopKMemoryToolSpec(store, user_id="user-alice")
    bob = TopKMemoryToolSpec(store, user_id="user-bob")
    alice.remember("The user is allergic to peanuts.", category="health")
    alice.remember("The user's order number is 88412.", category="account")
    bob.remember("The user prefers aisle seats.")

    assert "peanuts" in alice.recall("what foods should I avoid?", top_k=1)
    assert "88412" in alice.recall("88412", top_k=1)
    assert "aisle" not in alice.recall("seat preference")
    assert "peanuts" not in bob.recall("allergies")

    memory_id = alice.recall("order number", top_k=1).split("(")[1].split(",")[0]
    alice.forget(memory_id)
    assert "88412" not in alice.recall("order number")


async def test_memory_tools_async(store):
    spec = TopKMemoryToolSpec(store, user_id="user-carol")
    await spec.aremember("The user works night shifts.")
    assert "night shifts" in await spec.arecall("work schedule")


async def test_memory_block_round_trip(store):
    block = TopKMemoryBlock(vector_store=store, user_id="user-dana")
    await block.aput([ChatMessage(role="user", content="My dog is called Biscuit and he's a beagle.")])
    got = await block.aget([ChatMessage(role="user", content="What breed is my dog?")])
    assert "Biscuit" in got
    other = TopKMemoryBlock(vector_store=store, user_id="user-eve")
    assert await other.aget([ChatMessage(role="user", content="What breed is my dog?")]) == ""


async def test_memory_block_carries_across_sessions(store):
    def memory(session_id):
        block = TopKMemoryBlock(vector_store=store, user_id="user-finn")
        # A tiny token limit makes Memory flush old messages to the long-term block.
        return Memory.from_defaults(
            session_id=session_id,
            token_limit=60,
            chat_history_token_ratio=0.1,
            token_flush_size=10,
            memory_blocks=[block],
        )

    first = memory("session-1")
    await first.aput_messages(
        [
            ChatMessage(role="user", content="I live in Lisbon and I'm vegetarian."),
            ChatMessage(role="assistant", content="Noted: Lisbon, vegetarian."),
            ChatMessage(role="user", content="Also I run every morning before work."),
            ChatMessage(role="assistant", content="Got it, morning runs."),
        ]
    )
    await first.aget()  # triggers the flush into TopK

    second = memory("session-2")
    await second.aput(ChatMessage(role="user", content="Where do I live and what do I eat?"))
    prompt = "\n".join(str(m.content) for m in await second.aget())
    assert "Lisbon" in prompt
