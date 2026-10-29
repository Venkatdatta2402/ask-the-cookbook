---
# the Hugging Face Space's settings (https://huggingface.co/docs/hub/spaces-config-reference); GitHub shows them
# as a table. sdk_version must match gradio in requirements.txt. Python 3.12: the Space runs on ZeroGPU, which
# takes only 3.12 or 3.10
title: Ask the Cookbook
emoji: 🍳
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Questions on agentic-forge-cookbook, answered with sources
---

# Ask the Cookbook

**A chat assistant that answers questions about the
[Agentic Forge cookbook](https://github.com/Venkatdatta2402/agentic-forge-cookbook) and links every claim to the
exact lines it came from.**

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/Venkatdatta/ask-the-cookbook)
[![Tests](https://github.com/Venkatdatta2402/ask-the-cookbook/actions/workflows/tests.yml/badge.svg)](https://github.com/Venkatdatta2402/ask-the-cookbook/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

Agentic Forge is a cookbook that builds AI agents from first principles: an agent loop, tools, memory, context
engineering and multi-agent orchestration, written from scratch, then compared with LangGraph, CrewAI, AutoGen,
LlamaIndex and others. This assistant answers questions about it, and its agentic mode runs on **the cookbook's
own agent runtime**. The project asks the cookbook questions with the agent the cookbook teaches you to build.

**Try it:** [huggingface.co/spaces/Venkatdatta/ask-the-cookbook](https://huggingface.co/spaces/Venkatdatta/ask-the-cookbook)
· **Use it from code:** [the API](#api), with
[interactive docs](https://venkatdatta-ask-the-cookbook.hf.space/api/docs)

---

## Contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [API](#api)
- [Concurrency and performance](#concurrency-and-performance)
- [Designed for a free tier](#designed-for-a-free-tier)
- [Measured results](#measured-results)
- [Evaluation](#evaluation)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Keeping the index current](#keeping-the-index-current)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [Where the code comes from](#where-the-code-comes-from)
- [License](#license)

---

## What it does

You ask a question, and you pick how it is answered:

| | **RAG** | **Agentic RAG** |
|---|---|---|
| How | One hybrid search, then one model call that answers from the best 5 passages | An agent that searches and opens files as many times as the question needs (up to 8 tool calls) |
| Best for | A question about one idea or one chapter | A question that spans chapters, or compares two parts |
| Time | A few seconds | A few minutes |
| Cost | About 4,000 tokens | About 15,000 to 40,000 tokens |
| Key | The demo's shared key, or your own | Your own [Groq API key](https://console.groq.com/keys) (free) |

Both modes use `openai/gpt-oss-120b` on Groq. Answers come with **citations that open the cited lines on
GitHub**, and a list of sources named by file, lines and section.

It's also a **chat**:

- **Ask about.** Select any part of an answer, and an *Ask about* button appears beside it. The passage goes
  into your next question, so the model is sure to see the part you mean.
- **Token meter.** Shows how many tokens the chat holds, next to the 5,000 that one agent call can carry.
- **Compact now.** Replaces older turns with a summary and the facts pulled out of them, and keeps the last three
  exchanges word for word.
- **Nothing stored.** The chat lasts for your browser session. Your key is used for your questions only.

And it's an **API**: the same answers over HTTP, streamed as they're worked out, from an async FastAPI backend
that runs on the same server as the page.

## How it works

```mermaid
flowchart LR
    Q([Question]) --> M{Mode}
    M -- RAG --> H["Hybrid search<br/>vectors + BM25, fused"]
    H --> F{"Anything close<br/>enough?"}
    F -- no --> N(["'The cookbook does not cover that.'<br/>(no model call)"])
    F -- yes --> W["gpt-oss-120b answers<br/>from the top 5 passages"]
    M -- Agentic RAG --> L["Agent loop<br/>Think, Act, Observe"]
    L -- "search_cookbook<br/>open_file" --> L
    L --> A["Answer from<br/>what it found"]
    W --> C([Answer with links<br/>to the cited lines])
    A --> C
```

### 1. The index

The corpus is what a reader of the cookbook would read: the root and chapter READMEs, the **prose** of every
notebook (markdown cells only, not code or outputs), and every Python module. Tests, stubs and the evaluation
chapters' answer keys are left out, so the assistant can't read off the answers it's graded on. The current file
and chunk counts are under [Keeping the index current](#keeping-the-index-current), updated by every refresh.

- **Prose** is split along its markdown headings, then into chunks of up to 800 tokens.
- **Code** is split by its syntax tree into chunks of up to 80 lines, so a function stays in one piece.
- Every chunk keeps its file and its line numbers. That's what lets an answer link to the exact lines.
- Chunks are embedded with Gemini `gemini-embedding-001` (768 dimensions) and stored as a LlamaIndex index.

### 2. RAG: hybrid search with a refusal floor

- **Two searches.** Vector search finds passages that mean the same thing as the question. BM25 keyword search
  finds exact names, like `_doom_loop` or `max_tool_calls`, that embeddings tend to blur.
- **Fused.** The two lists (10 passages each) are merged by reciprocal rank fusion, and the top 5 are kept.
- **Refusal floor.** If no passage scores at least **0.66** on raw vector similarity, the question is turned away
  before any model call. The floor was measured: the lowest-scoring answerable test question scored 0.702, the
  highest off-topic one 0.624. It's applied *before* fusion, because after fusion a score only reflects rank.
- **Citations.** The model cites passages by number, in several formats (`[1]`, `[Source 1]`, `【1†L9-L15】`).
  All of them are turned into one clickable `[n]` per source, pointing at the passage's lines on GitHub.
  Duplicate sources are merged, and code such as `pairs[2]` is never mistaken for a citation.

### 3. Agentic RAG: the cookbook's own agent

The agent is assembled from the cookbook's chapters 1 to 6, not from a framework:

- **The loop** (chapter 2) decides, acts and observes. Its two tools are `search_cookbook` (the same hybrid
  search, with no floor, returning passages labelled with file and lines) and `open_file` (reads up to 100 lines
  of a file the search pointed to).
- **The context system** (chapter 5) builds each model call. It pins the instructions and the question, always
  keeps the two newest items, and fills the rest of a **5,000-token budget** by relevance, kind and recency.
  Results that don't fit drop to one-line labels.
- **A ledger of calls made** is pinned into every call: each search and file read so far, and what it returned.
  This stops the agent redoing work it no longer has in view.
- **Repeat guards** refuse an exact repeat, and a near-duplicate search (the same words reordered or
  re-inflected), naming the earlier search.
- **A forced finish.** After 8 tool calls the agent must answer from what it found. That final call gets a
  7,000-token budget, enough for nearly everything the run read.
- **Isolation.** Each run gets its own model client on the visitor's key, so runs on different keys proceed at
  the same time without touching each other.

### 4. The chat

The chat keeps each question and final answer; the agent's tool calls are discarded once it answers. A RAG
answer is kept with its `[n]` citations rewritten as `[file, lines a-b]`, so a later agentic turn can open
exactly those lines. Compaction uses chapter 5's compressor, and is undone if the result wouldn't be smaller.

## API

Base URL: `https://venkatdatta-ask-the-cookbook.hf.space` ·
[interactive docs](https://venkatdatta-ask-the-cookbook.hf.space/api/docs)

| Endpoint | What it does |
|---|---|
| `POST /api/ask` | Answers a question: streamed as server-sent events (default), or as one JSON reply with `"stream": false` |
| `GET /api/health` | Status, the model, the cookbook commit the index holds, the limits |
| `GET /api/docs` | Interactive docs (OpenAPI) |

**Your keys pay for your answers.** The API never uses the demo key.

| Header | |
|---|---|
| `X-Groq-Key` | Required. The answer is written on your [Groq](https://console.groq.com/keys) account |
| `X-Gemini-Key` | Required for `"mode": "agentic"`, whose up to 8 searches each embed a query. Optional for RAG |

Each key is checked with its provider before any work starts, using a free call, and is never logged or stored.

**Request body:** `question` (up to 500 characters), `mode` (`"rag"` or `"agentic"`), `quote` (optional: a
passage the question is about), `stream` (default `true`).

**Streaming with curl** (`-N` prints events as they arrive):

```bash
curl -N https://venkatdatta-ask-the-cookbook.hf.space/api/ask \
  -H "Content-Type: application/json" \
  -H "X-Groq-Key: $GROQ_API_KEY" \
  -d '{"question": "What counts as a doom loop, and what does the runtime do when it detects one?"}'
```

```
event: progress
data: {"message": "Searching the cookbook"}

event: progress
data: {"message": "Reading 5 passages and writing the answer"}

event: answer
data: {"answer": "... [[1]](https://github.com/.../runtime.py#L181-L218) ...", "sources": [...], ...}

event: done
data: {"seconds": ..., "tokens": ...}
```

**From Python, asynchronously**, with `httpx`:

```python
import asyncio, json, os
import httpx

URL = "https://venkatdatta-ask-the-cookbook.hf.space/api/ask"

async def ask(question, mode="rag"):
    headers = {"X-Groq-Key": os.environ["GROQ_API_KEY"]}
    if mode == "agentic":
        headers["X-Gemini-Key"] = os.environ["GEMINI_API_KEY"]
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", URL, json={"question": question, "mode": mode},
                                 headers=headers) as response:
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    data = json.loads(line.removeprefix("data: "))
                    if event == "progress":
                        print("...", data["message"])
                    elif event == "answer":
                        return data
                    elif event == "error":
                        raise RuntimeError(f'{data["status"]}: {data["message"]}')

print(asyncio.run(ask("Why shouldn't agent memory start with a vector database?"))["answer"])
```

| Response | When |
|---|---|
| `401` | A key is missing, malformed, or refused by its provider |
| `422` | The question is empty or over 500 characters |
| `429` | More than 5 requests a minute from one address (see `Retry-After`), or your key's own limit |
| `error` event | Something failed mid-answer: `{"status", "message"}` |

## Concurrency and performance

### One server for the page and the API

The Space runs on **ZeroGPU**, the only Gradio hardware a free account gets. ZeroGPU only accepts an app started
by Gradio's own `launch()`. Four set-ups were tried on a throwaway Space:

| Set-up | Result on ZeroGPU |
|---|---|
| A FastAPI app of our own, Gradio mounted on it, run by uvicorn | Port taken: Gradio's server-side renderer starts first |
| The same, renderer off | Started, then **shut down by ZeroGPU after 2 seconds** |
| Gradio's `launch()`, with our routes in the FastAPI app it creates | Runs, but the renderer in front answers `/api/*` itself |
| **The same, renderer off** | **Works**: API, streaming, page and Gradio events, all on one server |

So `app.py` hands the API's routes to Gradio: `launch(app_kwargs={"routes": api.router.routes}, ssr_mode=False)`.
FastAPI takes `routes` when it's created, so the API is part of the server from the start.

### Async where the waiting is

A question spends almost all its time **waiting**: on Gemini to embed it, and on Groq to write the answer.

- **RAG is async end to end.** The embedding and the answer are awaited, so while one request waits, the event
  loop serves the others. No thread is held per request.
- **The agent stays synchronous.** It's the cookbook's own loop, which deliberately uses threads (its chapter 2
  explains why). It runs with `asyncio.to_thread`, and the event loop follows its progress and streams each step.
- **Caps on work at once.** At most 8 RAG answers and 2 agent runs are worked out at once (`asyncio.Semaphore`).
  A request over the cap waits its turn, and is told so in its stream; it isn't refused. This protects the
  free-tier quotas that every visitor shares.
- **Stopping when the caller leaves.** If the caller disconnects, RAG stops at once, because its awaits are
  cancelled. A thread can't be killed, so an agent run is stopped where its next model call would go out: at most
  the call already in flight is spent. Its slot is held until the thread has actually ended.
- **Abuse guards.** Each address may send 5 requests a minute, and each key is checked with its provider before
  any work. A made-up Groq key would otherwise get its question embedded on the owner's Gemini key before Groq
  refused it.

### Load test

[`benchmarks/load_test.py`](benchmarks/load_test.py) sends **100 questions at once** over HTTP to a real uvicorn
server. The model is a stand-in that takes 1 second to reply, so the test costs no tokens. Everything else is
the real code: retrieval over the real index, the citation engine, sources and links.

| Serving | Before the fixes below | After |
|---|---|---|
| Threads, the page's way (8 workers) | 71.7 s | 14.0 s |
| Async API, cap of 8 (as deployed) | 51.3 s | 13.8 s |
| Async API, cap lifted | 54.9 s | **3.0 s** (33 answers a second, on one event-loop thread) |

With waiting alone, the cap-lifted run should take about 1 second, so the first numbers meant something else was
using CPU. Profiling found **~500 ms of CPU per question**, which no event loop can overlap:

| Found | Fix | CPU per question |
|---|---|---|
| The BM25 keyword index was rebuilt over every chunk for every question | Built once per index and shared; it's read-only | −390 ms |
| The vector search ran twice: once for the refusal floor, again inside the fusion | The fusion reuses the first search's results | −45 ms |
| LlamaIndex's in-memory store rebuilt its numpy matrix from every stored vector on every query | The matrix is kept between questions; one matrix multiply | −45 ms |
| The async search made the Gemini embedding twice, and its sync fallback blocked the event loop | An async retriever that embeds once and awaits it | 1 embedding, not 2 |

After the fixes, search costs **3.9 ms of CPU per question**, down from about 500 ms. That also shortens every
RAG answer on the live page. Tests check that each fix returns exactly the same passages, in the same order,
with the same scores.

## Designed for a free tier

Every number above comes from a limit of Groq's free tier for `gpt-oss-120b`:

| Limit | What it shaped |
|---|---|
| A request over **8,000 tokens** is refused | Each agent call carries at most 5,000 tokens of context, which leaves room for the tool schemas, the instructions and the model's reply |
| **8,000 tokens a minute** | Agentic RAG takes minutes: each step waits for the minute's budget |
| **200,000 tokens a day**, per account | The demo key covers about 50 RAG questions a day for all visitors together. Agentic RAG runs on the visitor's own key |
| Free embeddings: 1,000 texts a day | The index is refreshed incrementally: only new or changed files are embedded |

When Groq refuses the demo key, the visitor is told which limit was hit. For the daily limit, they're asked to
paste their own key or come back tomorrow. For the per-minute limit, they're asked to try again in a minute.

## Measured results

From live runs against the real models:

| Run | Tokens | Time | Notes |
|---|---|---|---|
| RAG question | 4,027 | 3.9 s | Cited `runtime.py` lines 181-218: the exact function |
| Ask about a passage | 4,048 | 2.9 s | Found the exact code the selection referred to |
| Agentic RAG, multi-part question, before fixes | 39,174 | minutes | 8 tool calls, two searches nearly identical, one part never searched |
| The same question, after the ledger and guards | **14,503** | minutes | **4 tool calls**, every part answered correctly |
| Compact now | 1,941 | 17.6 s | Two model calls: a summary and the facts |

## Evaluation

Retrieval is measured against a **hand-graded golden set**, offline, on every push.

**The golden set.** 30 questions (24 RAG, 6 agentic). Every chunk in the index (1,063 of them) was read once,
with no search involved, and graded against all 30 questions: 3 states the answer, 2 is a needed part of it,
1 helps a little, 0 is not relevant. Searching for candidates first would only have graded what the search
found, which is the thing being measured. That gave 510 graded passages, each with a quote from the chunk and
the reason for its grade. Tests check that every passage is a real chunk and every quote is in it.

**The metrics**, on the search RAG answers from (k = 5):

| | Precision@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| All 30 questions | 0.773 | 0.421 | 0.967 | 0.731 |
| RAG questions (24) | 0.800 | 0.471 | 0.958 | 0.760 |
| Agentic questions (6), as one search | 0.667 | 0.225 | 1.000 | 0.615 |

Precision counts any relevant passage (grade 1+), recall counts the passages an answer needs (grade 2+), and
NDCG uses every grade. Each question's embedding is kept with the golden set, so the run calls no API and gives
the same numbers every time.

**The gate.** These numbers are the committed baseline. The test suite fails if any of them drops, so a change to
the search, the splitter or the floor that loses relevant passages can't be merged unnoticed. Breaking the search
on purpose (a stricter floor, a weaker keyword search) failed it both times.

**Checks on answers.** Plain code, no model: an answer must cite, every citation must point to a source it was
given, every source must be real lines of a real file, and at least one cited source must be a passage graded 3.

```bash
python -m evals.run_evals                         # retrieval against the baseline (free)
python -m evals.generation.collect_answers        # real RAG answers to check: spends about 90k Groq tokens
```

[evals/README.md](evals/README.md) has the details.

## Getting started

**Requirements:** Python 3.12, a [Groq API key](https://console.groq.com/keys) and a
[Gemini API key](https://aistudio.google.com/apikey). Both are free.

```bash
conda env create -f environment.yml        # or: pip install -r requirements-dev.txt
conda activate ask-the-cookbook
cp .env.example .env                       # then fill in the two keys
python app.py                              # http://127.0.0.1:7860
```

| Variable | Used for |
|---|---|
| `GROQ_API_KEY` | The demo key: RAG questions from visitors who haven't pasted their own |
| `GEMINI_API_KEY` | Embedding each question to search the index (and embedding the cookbook on a refresh) |

**Tests** run offline, with stand-in models and fake clients: no keys, no tokens.

```bash
pytest                                     # also run on every push by the Tests workflow
python benchmarks/load_test.py             # the load test above, against a stand-in model
```

They cover what's checked before any model call, citation parsing and linking, the agent's context budget and
guards, concurrent runs on different keys, compaction, and the index refresh. For the API, they also cover the
key checks, streaming order, the limits, requests running at once, stopping work when the caller leaves, and
that the search fixes return exactly the same passages and scores as before.

## Project structure

```
app.py                     the page (Gradio), and the one server that serves it and the API
ask_the_cookbook/
    answers.py             answering, for the page and the API: RAG (sync and async), the agent, citations
    api.py                 the API (FastAPI): streaming, key checks, limits, stopping when the caller leaves
    corpus.py              which cookbook files are read, and how (notebooks as their prose)
    search.py              the index, hybrid search with the refusal floor, the RAG engine, the agent's tools
    agent.py               the agent: the cookbook's loop and context system over those tools
    session.py             a visitor's chat: history, Ask about, the meter, compaction
    refresh.py             brings the index up to date with the cookbook's latest commit
    runtime/               the agent's parts, from the cookbook's chapters 1-6
data/
    index/                 the saved LlamaIndex index (the vector store is in Git LFS)
    corpus/                the cookbook's text exactly as indexed; answers link to these lines
tests/                     offline tests
evals/                     the golden set, retrieval metrics and their baseline, checks on answers
benchmarks/                the load test
.github/workflows/         the tests, the sync to the Space, and the index refresh
```

## Keeping the index current

<!-- index-commit -->
Built from the cookbook at commit [`43c7bf5`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/43c7bf5e3ac6f8db886a81790b45e9f71febbf04): 124 files, 1,063 chunks.
<!-- /index-commit -->

```bash
python -m ask_the_cookbook.refresh --dry-run    # what would change, and how many embeddings it would cost
python -m ask_the_cookbook.refresh              # do it
```

The refresh checks the cookbook's latest commit and stops if it's already indexed. Otherwise it fetches that
commit and compares it with the index **file by file, by content hash**, not by git diff. A git diff would flag
changes the index never sees, such as edits to notebook code cells. It then embeds only new or changed files,
removes deleted ones, and rewrites `data/corpus` and the line above. A run that would need more than 900
embeddings is refused, to stay inside the free daily quota.

The **Refresh the index** workflow runs the same module on GitHub from a button in the Actions tab. It commits
the result and copies it to the Space.

## Deployment

The app runs on [Hugging Face Spaces](https://huggingface.co/spaces/Venkatdatta/ask-the-cookbook). GitHub is the
source of truth: on every push to `main`, the **Sync to the Space** workflow uploads the repo to the Space with
Hugging Face's [`hub-sync`](https://github.com/huggingface/hub-sync) action, and the Space rebuilds.

| Where | Setting |
|---|---|
| Space secrets | `GROQ_API_KEY`, `GEMINI_API_KEY` |
| GitHub Actions secret | `HF_TOKEN`: a Hugging Face token with write access to the Space |
| GitHub Actions variable | `HF_SPACE`: `Venkatdatta/ask-the-cookbook` |

The Space runs on ZeroGPU, the hardware a free account gets for Gradio apps. The app needs no GPU, since its
models run on Groq and Gemini. One server serves both the page and the API (see
[One server for the page and the API](#one-server-for-the-page-and-the-api)).

## Limitations

- **The refusal floor has a thin margin** (0.702 vs 0.624). An unusual phrasing of an answerable question can be
  turned away; naming the chapter or idea usually fixes it. Re-measure it if the corpus or embedding model changes.
- **Notebook citations open the notebook at the top.** GitHub shows notebooks in a frame that no link can scroll,
  so they're named by section instead of linked to lines.
- **One model.** The agent's prompts and tool format were tuned on `gpt-oss-120b`. Another model would need
  testing first: other model families format tool calls differently.
- **Free-tier limits.** The demo key's daily allowance is shared by every visitor, and agentic answers take
  minutes.
- **The index is refreshed by hand**, so it can trail the cookbook until the workflow is run.
- **The per-address limit is a guard rail.** Behind Hugging Face's proxy, the caller's address comes from the
  `X-Forwarded-For` header, which a caller can set. The key checks are the real protection.
- **An agent run stops at its next model call**, not instantly, when its caller leaves: the cookbook's loop runs
  on a thread, and Python can't kill a thread.
- **Limits and remembered keys live in one process's memory.** A restart or a second replica starts them afresh.

## Where the code comes from

The assistant was first built inside the cookbook, in `08_frameworks/05_llamaindex`, and moved here at commit
[`fb39ada`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/fb39ada) as a package:

| Here | In the cookbook |
|---|---|
| `ask_the_cookbook/corpus.py` | `08_frameworks/05_llamaindex/cookbook.py` |
| `ask_the_cookbook/search.py` | `08_frameworks/05_llamaindex/ask_index.py` |
| `ask_the_cookbook/agent.py` | `08_frameworks/05_llamaindex/agent_scratch.py` |
| `ask_the_cookbook/runtime/` | `01_foundations` (llm, chat), `02_agent_runtime` (think, act, observe, runtime), `03_tools` (tools, registry), `04_memory/recall.py`, `05_context` (context, selection, focus, compression, system), `06_orchestration/supervisor.py` |

What changed:

- **Trimmed to what the app can reach.** Code that nothing in the app or the refresh can call was removed: the
  other reasoning strategies, the `Loop` and `Graph` runners, the supervisor agents, memory retrieval, the
  LlamaIndex agent and the evaluation helpers. What remains keeps its code and its teaching comments.
- **Per-run model clients.** An agent run can be given its own client, and every model call it makes goes
  through it, including the repair of a failed tool call and compaction. Given none, the code behaves as in the
  cookbook.
- **Built for the app:** the chat, the ledger of calls made, the near-duplicate guard and the larger final-answer
  budget.
- **Faster search, same results:** BM25 built once and shared, the vector search run once on a matrix kept between
  questions, and an async retriever that embeds a question once. An agent run's searches can also use another
  Gemini key (an API caller's).

## License

[MIT](LICENSE). The indexed text in `data/corpus/` is from
[agentic-forge-cookbook](https://github.com/Venkatdatta2402/agentic-forge-cookbook), by the same author.
