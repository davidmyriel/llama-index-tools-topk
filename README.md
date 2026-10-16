# llama-index-tools-topk

**Search and long-term memory tools for LlamaIndex agents, backed by [TopK](https://topk.io).** Give a `FunctionAgent` keyword (BM25), semantic and hybrid search tools that say when to use each, plus `remember` / `recall` memory that is isolated per user. TopK embeds text on the server, so you don't need an embedding model.

```bash
pip install llama-index-tools-topk
```

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.anthropic import Anthropic
from llama_index.tools.topk import TopKToolSpec
from llama_index.vector_stores.topk import TopKVectorStore

tools = TopKToolSpec(
    TopKVectorStore(collection_name="runbook"),  # reads TOPK_API_KEY, TOPK_REGION
    corpus_description="an engineering runbook: error codes, incidents and product notes",
    metadata_fields={"topic": "payments, checkout or inventory", "year": "integer"},
).to_tool_list()

agent = FunctionAgent(tools=tools, llm=Anthropic(model="claude-sonnet-5-5"))
await agent.run("What does error E-4021 mean?")  # the agent calls keyword_search("E-4021")
```

Set `TOPK_API_KEY` (create one at [console.topk.io](https://console.topk.io)) and `TOPK_REGION` (for example `aws-us-east-1-elastica`, see [regions](https://docs.topk.io/regions)). Load data with [`llama-index-vector-stores-topk`](https://github.com/topk-io/llama-index-vector-stores-topk), which this package installs.

## What's in the package

| Class | What the agent gets | Use it for |
|---|---|---|
| `TopKToolSpec` | `keyword_search`, `semantic_search`, `hybrid_search`, all with metadata `filters` | Agentic RAG over docs, runbooks, tickets, code |
| `TopKMemoryToolSpec` | `remember`, `recall`, `forget` in a per-user partition | Long-term memory the agent manages itself |
| `TopKMemoryBlock` | A `BaseMemoryBlock` for LlamaIndex `Memory` | Automatic long-term memory from flushed chat history |

## Why give an agent three search tools?

Agents don't know which retrieval fits a question, but they can read tool descriptions. Each tool's description says when to use it:

- **`keyword_search`** (BM25): exact strings such as error codes, IDs, SKUs, ticket numbers and function names. It only returns documents that contain the terms, so an empty result tells the agent the term isn't there. With vector search alone, `E-4021` and `E-4012` look the same.
- **`semantic_search`**: paraphrased or conceptual questions where the documents use different words ("why do customers leave?" finds "subscriptions are cancelled because onboarding is slow").
- **`hybrid_search`**: both ranked together in [one TopK query](https://docs.topk.io/guides/true-hybrid-search), the default when a question mixes meaning and exact terms. `keyword_weight` lets the agent lean toward exact matches.

All three take `filters`, a small JSON object the model can write: `{"topic": "payments", "year": {"$gte": 2024}}`. Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`; dots address nested metadata. `corpus_description` and `metadata_fields` are appended to every tool description so the agent knows what it's searching and what it can filter on.

Results come back as compact text with each hit's id, score, metadata and (truncated) content. Every tool also has an async version, which `FunctionAgent` uses automatically.

## Long-term memory per user

```python
from llama_index.tools.topk import TopKMemoryToolSpec

memories = TopKVectorStore(collection_name="agent-memory")


def assistant_for(user_id: str) -> FunctionAgent:
    tools = TopKMemoryToolSpec(memories, user_id=user_id).to_tool_list()
    return FunctionAgent(
        tools=tools, llm=llm, system_prompt="You are a personal assistant with long-term memory tools."
    )


await assistant_for("user-alice").run("Please remember that I'm allergic to peanuts.")  # -> remember(...)
await assistant_for("user-alice").run("Suggest a snack for me.")  # -> recall("food allergies")
await assistant_for("user-bob").run("Do I have any allergies?")  # Bob's partition: nothing
```

- **Isolated by design.** Each user's memories live in their own TopK [partition](https://docs.topk.io/concepts#multi-tenancy). There's no `user_id` filter that a prompt or a bug can drop.
- **Recall finds paraphrases and exact terms.** `recall` runs hybrid search, so "what can't I eat?" finds the peanut allergy and "88412" finds the order number.
- **The agent curates.** `remember` asks for short, self-contained facts; `forget` deletes by the id that `recall` shows.

### Or let `Memory` do it automatically

```python
from llama_index.core.memory import Memory
from llama_index.tools.topk import TopKMemoryBlock

memory = Memory.from_defaults(
    session_id="chat-1",
    memory_blocks=[TopKMemoryBlock(vector_store=memories, user_id="user-alice", similarity_top_k=3)],
)
await agent.run("I live in Lisbon and I'm vegetarian.", memory=memory)
```

When short-term history overflows, `Memory` flushes old messages into the block. On every turn the block searches them (hybrid by default) using the latest messages and adds the hits to the prompt. Because the partition is per user, a new session for the same user starts with those memories.

## Bring your own embeddings

Both tool specs and the memory block work with a store that uses your own embeddings. Pass the same embedding model:

```python
from llama_index.embeddings.openai import OpenAIEmbedding

store = TopKVectorStore(collection_name="docs", server_side_embeddings=False)
tools = TopKToolSpec(store, embed_model=OpenAIEmbedding()).to_tool_list()
```

## FAQ

**How do I give a LlamaIndex agent keyword search and vector search?** Use `TopKToolSpec(...).to_tool_list()`. The agent gets separate BM25, semantic and hybrid tools, and the descriptions tell it which to use.

**How do I add long-term memory to a LlamaIndex agent?** Either give it `TopKMemoryToolSpec` (the agent decides what to remember) or add `TopKMemoryBlock` to `Memory` (flushed chat history becomes searchable). Both keep each user in a separate partition.

**Does this need an embedding model?** No. TopK embeds on the server by default. You can bring any LlamaIndex embedding model instead.

**Why isn't this in the llama_index monorepo?** LlamaIndex no longer accepts new integration packages in its repository; new integrations are published to PyPI on their own. This package uses the standard `llama_index.tools.*` namespace.

## Development

```bash
uv venv && uv pip install -e . --group test
pytest tests/unit_tests
TOPK_API_KEY=... TOPK_REGION=aws-us-east-1-elastica pytest tests/integration_tests   # live TopK
ANTHROPIC_API_KEY=... pytest tests/integration_tests/test_agent.py                 # real FunctionAgent
```

`examples/topk_agent_tools.ipynb` walks through everything with a real agent.

## Links

- TopK docs: https://docs.topk.io
- Vector store package: https://github.com/topk-io/llama-index-vector-stores-topk
- TopK Python SDK: https://pypi.org/project/topk-sdk/
