"""Long-term agent memory on TopK: one isolated partition per user."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.schema import TextNode
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode, VectorStoreQueryResult
from llama_index.vector_stores.topk import TopKVectorStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TopKMemoryToolSpec(BaseToolSpec):
    """
    ``remember`` / ``recall`` / ``forget`` tools backed by a per-user TopK partition.

    The agent decides what's worth keeping and saves it with ``remember``. Later, in
    any session, ``recall`` finds it with hybrid search, so both paraphrases ("what
    can't I eat?") and exact terms ("peanuts", an order number) match. Each user's
    memories live in their own partition, so one user's agent can never read
    another's.

    Example:
        .. code-block:: python

            store = TopKVectorStore(collection_name="agent-memory")
            tools = TopKMemoryToolSpec(store, user_id="user-42").to_tool_list()
    """

    spec_functions = [("remember", "aremember"), ("recall", "arecall"), ("forget", "aforget")]

    def __init__(
        self,
        vector_store: TopKVectorStore,
        user_id: str,
        *,
        embed_model: BaseEmbedding | None = None,
        default_top_k: int = 5,
    ) -> None:
        """
        Create the tools.

        Args:
            vector_store: A ``TopKVectorStore``; memories go in its ``user_id`` partition.
            user_id: Whose memories these are. Used as the TopK partition name.
            embed_model: Only needed when the store uses your own embeddings.
            default_top_k: Memories returned by ``recall`` when the agent doesn't ask for a number.

        """
        if not vector_store.server_side_embeddings and embed_model is None:
            msg = "This store uses your own embeddings; pass the same embed_model to TopKMemoryToolSpec"
            raise ValueError(msg)
        self.store = vector_store.with_partition(user_id)
        self.user_id = user_id
        self.embed_model = embed_model
        self.default_top_k = default_top_k

    def remember(self, memory: str, category: str | None = None) -> str:
        """
        Save a fact about the user for future conversations. Use it when the user shares a lasting
        preference, constraint, personal detail, decision or goal (e.g. "I'm allergic to peanuts",
        "we deploy on Fridays", "my order number is 88412"). Write the memory as a short,
        self-contained sentence in the third person. Don't save small talk or one-off requests.

        Args:
            memory: The fact to remember, e.g. "The user is allergic to peanuts."
            category: Optional label such as "preference", "health", "work" or "account".

        """
        self.store.add([self._node(memory, category, self._embed(memory))])
        return "Saved."

    async def aremember(self, memory: str, category: str | None = None) -> str:
        await self.store.async_add([self._node(memory, category, await self._aembed(memory))])
        return "Saved."

    def recall(self, query: str, top_k: int | None = None) -> str:
        """
        Look up what you know about the user from earlier conversations. Use it before answering
        anything that could depend on the user's preferences, history or personal details, and
        whenever the user asks what you remember. Matches both by meaning and by exact terms.

        Args:
            query: What you need to know, e.g. "food allergies" or "order 88412".
            top_k: How many memories to return (default 5).

        """
        return _format_memories(self.store.query(self._query(query, top_k, self._embed(query))))

    async def arecall(self, query: str, top_k: int | None = None) -> str:
        q = self._query(query, top_k, await self._aembed(query))
        return _format_memories(await self.store.aquery(q))

    def forget(self, memory_id: str) -> str:
        """
        Delete a memory that is wrong or that the user asked you to forget. Get the id from recall.

        Args:
            memory_id: The id shown by recall, e.g. "mem-3f2a...".

        """
        self.store.delete_nodes(node_ids=[memory_id])
        return "Deleted."

    async def aforget(self, memory_id: str) -> str:
        await self.store.adelete_nodes(node_ids=[memory_id])
        return "Deleted."

    def _node(self, memory: str, category: str | None, embedding: list[float] | None) -> TextNode:
        metadata: dict[str, Any] = {"user_id": self.user_id, "created_at": _now()}
        if category:
            metadata["category"] = category
        return TextNode(id_=f"mem-{uuid.uuid4().hex[:12]}", text=memory, metadata=metadata, embedding=embedding)

    def _query(self, query: str, top_k: int | None, embedding: list[float] | None) -> VectorStoreQuery:
        return VectorStoreQuery(
            query_str=query,
            query_embedding=embedding,
            similarity_top_k=max(1, min(top_k or self.default_top_k, 20)),
            mode=VectorStoreQueryMode.HYBRID,
        )

    def _embed(self, text: str) -> list[float] | None:
        return self.embed_model.get_text_embedding(text) if self.embed_model else None

    async def _aembed(self, text: str) -> list[float] | None:
        return await self.embed_model.aget_text_embedding(text) if self.embed_model else None


def _format_memories(result: VectorStoreQueryResult) -> str:
    if not result.nodes:
        return "No memories found."
    lines = []
    for node in result.nodes:
        when = node.metadata.get("created_at", "")
        category = node.metadata.get("category")
        tag = f" [{category}]" if category else ""
        lines.append(f"- ({node.node_id}, {when}{tag}) {node.get_content()}")
    return "\n".join(lines)


class TopKMemoryBlock(BaseMemoryBlock[str]):
    """
    A long-term memory block for LlamaIndex ``Memory``, stored in a per-user TopK partition.

    When short-term chat history overflows, ``Memory`` flushes old messages here.
    On every turn the block searches them with hybrid (semantic + BM25) search
    using the latest messages, and inserts what it finds into the prompt. Because
    the partition is per user rather than per session, memories carry across
    sessions.

    Example:
        .. code-block:: python

            from llama_index.core.memory import Memory

            block = TopKMemoryBlock(vector_store=TopKVectorStore(collection_name="agent-memory"), user_id="user-42")
            memory = Memory.from_defaults(session_id="chat-1", memory_blocks=[block])
    """

    name: str = Field(default="TopKMemory", description="The name of the memory block.")
    vector_store: TopKVectorStore = Field(description="TopK store; memories go in the user's partition.")
    user_id: str | None = Field(default=None, description="Partition for this user's memories.")
    embed_model: BaseEmbedding | None = Field(
        default=None, description="Only needed when the store uses your own embeddings."
    )
    similarity_top_k: int = Field(default=3, description="Memories retrieved per turn.")
    retrieval_context_window: int = Field(default=3, description="Recent messages used as the search query.")
    query_mode: str = Field(default="hybrid", description="'hybrid', 'default' or 'text_search'.")

    _store: TopKVectorStore | None = PrivateAttr(default=None)

    def _partition(self) -> TopKVectorStore:
        if self._store is None:
            self._store = self.vector_store.with_partition(self.user_id) if self.user_id else self.vector_store
        return self._store

    async def _aget(self, messages: list[ChatMessage] | None = None, **block_kwargs: Any) -> str:
        if not messages:
            return ""
        query_text = " ".join(_text(m) for m in messages[-self.retrieval_context_window :]).strip()
        if not query_text:
            return ""
        embedding = await self.embed_model.aget_query_embedding(query_text) if self.embed_model else None
        result = await self._partition().aquery(
            VectorStoreQuery(
                query_str=query_text,
                query_embedding=embedding,
                similarity_top_k=self.similarity_top_k,
                mode=VectorStoreQueryMode(self.query_mode),
            )
        )
        return "\n".join(f"- {node.get_content()}" for node in result.nodes or [])

    async def _aput(self, messages: list[ChatMessage]) -> None:
        lines = [f"{m.role.value}: {_text(m)}" for m in messages if _text(m)]
        if not lines:
            return
        text = "\n".join(lines)
        session_id = next((m.additional_kwargs.get("session_id") for m in messages if m.additional_kwargs), None)
        metadata: dict[str, Any] = {"created_at": _now()}
        if self.user_id:
            metadata["user_id"] = self.user_id
        if session_id:
            metadata["session_id"] = str(session_id)
        embedding = await self.embed_model.aget_text_embedding(text) if self.embed_model else None
        node = TextNode(id_=f"mem-{uuid.uuid4().hex[:12]}", text=text, metadata=metadata, embedding=embedding)
        await self._partition().async_add([node])


def _text(message: ChatMessage) -> str:
    return " ".join(b.text for b in message.blocks if isinstance(b, TextBlock)).strip()
