"""Search tools that let an agent choose keyword, semantic or hybrid retrieval over TopK."""

from __future__ import annotations

import json
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode, VectorStoreQueryResult
from llama_index.vector_stores.topk import TopKVectorStore

from llama_index.tools.topk._filters import dict_to_filters

_MAX_TOP_K = 20


class TopKToolSpec(BaseToolSpec):
    """
    Search tools over a TopK collection for LlamaIndex agents.

    Gives the agent three tools and tells it when to use each:

    * ``keyword_search``: BM25 for exact strings (error codes, IDs, SKUs, names)
    * ``semantic_search``: meaning-based search for paraphrased questions
    * ``hybrid_search``: both in one ranked query, the default choice

    All three accept a metadata ``filters`` object. Describe your metadata with
    ``metadata_fields`` so the agent knows what it can filter on.

    Example:
        .. code-block:: python

            from llama_index.core.agent.workflow import FunctionAgent
            from llama_index.tools.topk import TopKToolSpec
            from llama_index.vector_stores.topk import TopKVectorStore

            tools = TopKToolSpec(
                TopKVectorStore(collection_name="docs"),
                corpus_description="product docs and incident runbooks",
                metadata_fields={"source": "'runbook' or 'docs'", "year": "integer"},
            ).to_tool_list()
            agent = FunctionAgent(tools=tools, llm=llm)
    """

    spec_functions = [
        ("keyword_search", "akeyword_search"),
        ("semantic_search", "asemantic_search"),
        ("hybrid_search", "ahybrid_search"),
    ]

    def __init__(
        self,
        vector_store: TopKVectorStore,
        *,
        embed_model: BaseEmbedding | None = None,
        corpus_description: str | None = None,
        metadata_fields: dict[str, str] | None = None,
        default_top_k: int = 5,
        max_chars_per_result: int = 1500,
    ) -> None:
        """
        Create the tools.

        Args:
            vector_store: The ``TopKVectorStore`` to search (scoped to its partition, if any).
            embed_model: Query embedding model. Only needed when the store uses your own
                embeddings (``server_side_embeddings=False``).
            corpus_description: What the collection contains, added to every tool description.
            metadata_fields: Filterable metadata fields and what values they hold, added to
                every tool description, e.g. ``{"source": "'runbook' or 'docs'"}``.
            default_top_k: Results per search when the agent doesn't ask for a number.
            max_chars_per_result: Truncate each result's text to keep tool output small.

        """
        if not vector_store.server_side_embeddings and embed_model is None:
            msg = "This store uses your own embeddings; pass the same embed_model to TopKToolSpec"
            raise ValueError(msg)
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.corpus_description = corpus_description
        self.metadata_fields = metadata_fields or {}
        self.default_top_k = default_top_k
        self.max_chars_per_result = max_chars_per_result

    def to_tool_list(self, *args: Any, **kwargs: Any) -> list[FunctionTool]:
        tools = super().to_tool_list(*args, **kwargs)
        extra = self._context()
        if extra:
            for tool in tools:
                tool.metadata.description = f"{tool.metadata.description}\n{extra}"
        return tools

    def _context(self) -> str:
        lines = []
        if self.corpus_description:
            lines.append(f"The collection contains: {self.corpus_description}.")
        if self.metadata_fields:
            fields = "; ".join(f"{k}: {v}" for k, v in self.metadata_fields.items())
            lines.append(f"Metadata fields you can filter on: {fields}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ tools

    def keyword_search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> str:
        """
        Exact keyword search (BM25). Use this when the user mentions a specific string that must
        appear in the document: an error code like E-4021, an ID, a SKU, a ticket number, a
        function or product name, or a quoted phrase. Only documents that contain the query terms
        are returned, so an empty result means the term doesn't appear anywhere.

        Args:
            query: The exact terms to look for, e.g. "E-4021" or "rotate_api_key".
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.
                Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains.

        """
        return self._format(self.vector_store.query(self._query(query, top_k, filters, "text_search")))

    async def akeyword_search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> str:
        return self._format(await self.vector_store.aquery(await self._aquery(query, top_k, filters, "text_search")))

    def semantic_search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> str:
        """
        Semantic search by meaning. Use this for conceptual or paraphrased questions where the
        documents may use different words than the user, e.g. "why are customers churning?" or
        "how do I make deploys safer?". Don't use it to look up exact codes or IDs.

        Args:
            query: A natural-language question or description.
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.
                Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains.

        """
        return self._format(self.vector_store.query(self._query(query, top_k, filters, "default")))

    async def asemantic_search(self, query: str, top_k: int | None = None, filters: dict | None = None) -> str:
        return self._format(await self.vector_store.aquery(await self._aquery(query, top_k, filters, "default")))

    def hybrid_search(
        self, query: str, top_k: int | None = None, filters: dict | None = None, keyword_weight: float = 0.5
    ) -> str:
        """
        Hybrid search: meaning and exact keywords ranked together in one query. This is the best
        default when a question mixes both, e.g. "why does checkout fail with E-4021?", or when
        you're unsure which search fits.

        Args:
            query: The question, including any exact terms.
            top_k: How many results to return (default 5, max 20).
            filters: Optional metadata filter, e.g. {"source": "runbook"} or {"year": {"$gte": 2024}}.
                Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $contains.
            keyword_weight: 0 to 1. Raise it when exact terms matter more than meaning (default 0.5).

        """
        q = self._query(query, top_k, filters, "hybrid", alpha=1.0 - keyword_weight)
        return self._format(self.vector_store.query(q))

    async def ahybrid_search(
        self, query: str, top_k: int | None = None, filters: dict | None = None, keyword_weight: float = 0.5
    ) -> str:
        q = await self._aquery(query, top_k, filters, "hybrid", alpha=1.0 - keyword_weight)
        return self._format(await self.vector_store.aquery(q))

    # ------------------------------------------------------------------ helpers

    def _base_query(
        self, query: str, top_k: int | None, filters: dict | None, mode: str, alpha: float | None
    ) -> VectorStoreQuery:
        k = max(1, min(top_k or self.default_top_k, _MAX_TOP_K))
        return VectorStoreQuery(
            query_str=query,
            similarity_top_k=k,
            mode=VectorStoreQueryMode(mode),
            alpha=None if alpha is None else max(0.0, min(1.0, alpha)),
            filters=dict_to_filters(filters),
        )

    def _needs_embedding(self, mode: str) -> bool:
        return self.embed_model is not None and mode != "text_search"

    def _query(
        self, query: str, top_k: int | None, filters: dict | None, mode: str, alpha: float | None = None
    ) -> VectorStoreQuery:
        q = self._base_query(query, top_k, filters, mode, alpha)
        if self._needs_embedding(mode):
            q.query_embedding = self.embed_model.get_query_embedding(query)
        return q

    async def _aquery(
        self, query: str, top_k: int | None, filters: dict | None, mode: str, alpha: float | None = None
    ) -> VectorStoreQuery:
        q = self._base_query(query, top_k, filters, mode, alpha)
        if self._needs_embedding(mode):
            q.query_embedding = await self.embed_model.aget_query_embedding(query)
        return q

    def _format(self, result: VectorStoreQueryResult) -> str:
        if not result.nodes:
            return "No matching documents."
        blocks = []
        for i, (node, score) in enumerate(zip(result.nodes, result.similarities or [], strict=True), start=1):
            text = node.get_content()
            if len(text) > self.max_chars_per_result:
                text = text[: self.max_chars_per_result] + "…"
            meta = json.dumps(node.metadata, default=str, ensure_ascii=False) if node.metadata else "{}"
            blocks.append(f"[{i}] id={node.node_id} score={score:.3f} metadata={meta}\n{text}")
        return "\n\n".join(blocks)
