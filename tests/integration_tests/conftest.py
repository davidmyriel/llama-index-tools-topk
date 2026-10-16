import os
import uuid

import pytest
from llama_index.core.schema import TextNode
from llama_index.vector_stores.topk import TopKVectorStore
from topk_sdk import AsyncClient, Client

# Near-identical error codes: a pure semantic search can't tell them apart, BM25 can.
RUNBOOK = [
    ("E-4021", "Error code E-4021 means the payment gateway timed out. Retry after 30 seconds.", "payments", 2024),
    ("E-4012", "Error code E-4012 means the card was declined by the issuing bank.", "payments", 2024),
    ("E-4201", "Error code E-4201 means the shipping address failed validation.", "checkout", 2023),
    ("E-2041", "Error code E-2041 means the inventory service returned stale stock levels.", "inventory", 2025),
    (
        "slow-pages",
        "Pages render slowly when the CDN cache is cold after a deploy; warm it with the prefetch job.",
        "performance",
        2025,
    ),
    ("churn", "Customers cancel subscriptions mostly because onboarding takes too long.", "product", 2024),
]


def runbook_nodes():
    return [
        TextNode(id_=node_id, text=text, metadata={"topic": topic, "year": year})
        for node_id, text, topic, year in RUNBOOK
    ]


@pytest.fixture(scope="session")
def creds() -> dict:
    if not os.environ.get("TOPK_API_KEY"):
        pytest.skip("TOPK_API_KEY not set")
    return {"api_key": os.environ["TOPK_API_KEY"], "region": os.environ.get("TOPK_REGION", "aws-us-east-1-elastica")}


@pytest.fixture
def collection_name(creds):
    # A fresh name per test: TopK doesn't support reusing a just-deleted collection name.
    name = f"li_tools_test_{uuid.uuid4().hex[:10]}"
    yield name
    try:
        Client(**creds).collections().delete(name)
    except Exception:
        pass


@pytest.fixture
def store(creds, collection_name) -> TopKVectorStore:
    return TopKVectorStore(collection_name=collection_name, client=Client(**creds), async_client=AsyncClient(**creds))


@pytest.fixture
def runbook_store(store) -> TopKVectorStore:
    store.add(runbook_nodes())
    return store


@pytest.fixture
def llm():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from llama_index.llms.anthropic import Anthropic

    return Anthropic(model=os.environ.get("TOPK_TEST_LLM", "claude-sonnet-5-5"), temperature=0.0)
