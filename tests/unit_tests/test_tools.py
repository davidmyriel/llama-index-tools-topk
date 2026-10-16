import pytest
from llama_index.core.vector_stores.types import FilterOperator
from llama_index.vector_stores.topk import TopKVectorStore

from llama_index.tools.topk import TopKMemoryBlock, TopKMemoryToolSpec, TopKToolSpec
from llama_index.tools.topk._filters import dict_to_filters


class FakeClient:
    def collections(self):
        raise AssertionError("should not create a collection")


@pytest.fixture
def store():
    return TopKVectorStore(client=FakeClient(), create_collection=False)


def test_search_tools_describe_when_to_use_them(store):
    tools = {t.metadata.name: t for t in TopKToolSpec(store).to_tool_list()}
    assert set(tools) == {"keyword_search", "semantic_search", "hybrid_search"}
    assert "error code" in tools["keyword_search"].metadata.description
    assert "paraphrased" in tools["semantic_search"].metadata.description
    assert "default" in tools["hybrid_search"].metadata.description
    assert all(t.async_fn is not None for t in tools.values())


def test_corpus_and_metadata_go_into_descriptions(store):
    spec = TopKToolSpec(store, corpus_description="incident runbooks", metadata_fields={"year": "integer"})
    for tool in spec.to_tool_list():
        assert "incident runbooks" in tool.metadata.description
        assert "year: integer" in tool.metadata.description


def test_byo_store_needs_embed_model():
    byo = TopKVectorStore(client=FakeClient(), server_side_embeddings=False)
    with pytest.raises(ValueError, match="embed_model"):
        TopKToolSpec(byo)
    with pytest.raises(ValueError, match="embed_model"):
        TopKMemoryToolSpec(byo, user_id="u")


def test_memory_tools_use_user_partition(store):
    spec = TopKMemoryToolSpec(store, user_id="user-42")
    assert spec.store.partition == "user-42"
    assert store.partition is None
    assert {t.metadata.name for t in spec.to_tool_list()} == {"remember", "recall", "forget"}


def test_memory_block_is_pydantic(store):
    block = TopKMemoryBlock(vector_store=store, user_id="u1")
    assert block._partition().partition == "u1"


def test_query_clamps_top_k_and_weight(store):
    q = TopKToolSpec(store)._query("x", 999, None, "hybrid", alpha=-3)
    assert q.similarity_top_k == 20 and q.alpha == 0.0


def test_dict_to_filters():
    assert dict_to_filters(None) is None
    f = dict_to_filters({"source": "runbook", "year": {"$gte": 2024, "$lt": 2026}})
    assert [(x.key, x.operator, x.value) for x in f.filters] == [
        ("source", FilterOperator.EQ, "runbook"),
        ("year", FilterOperator.GTE, 2024),
        ("year", FilterOperator.LT, 2026),
    ]
    with pytest.raises(ValueError, match="Unknown filter operator"):
        dict_to_filters({"year": {"$regex": "x"}})
    with pytest.raises(ValueError):
        dict_to_filters("source=runbook")
