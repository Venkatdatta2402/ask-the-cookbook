---
# the Hugging Face Space's settings (https://huggingface.co/docs/hub/spaces-config-reference); GitHub shows them
# as a table. sdk_version must match gradio in requirements.txt. Python 3.12: the Space runs on ZeroGPU, which
# takes only 3.12 or 3.10
title: Ask the Cookbook
emoji: 🍳
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Questions on agentic-forge-cookbook, answered with sources
---

# Ask the Cookbook

Ask questions about [agentic-forge-cookbook](https://github.com/Venkatdatta2402/agentic-forge-cookbook) and get
answers drawn from its READMEs, notebook prose and code, with the files they came from cited.

Two modes:

- **RAG**: one retrieval from the cookbook, then one model call that answers from the best five passages and
  cites them. A few seconds and about 4,000 tokens. Free on the demo's key until Groq's daily limit for it is
  reached (about 50 questions, shared by all visitors), or on your own key.
- **Agentic RAG**: retrieval is a tool the model calls as many times as the question needs, with a second tool to
  open files and read further, inside an agent loop. For questions that span chapters. A few minutes and 30-40k
  tokens, so it runs on your own [Groq API key](https://console.groq.com/keys) (free), which is used for your
  questions only and never stored.

It is a chat. To follow up on part of an answer, select that text: an **Ask about** button appears beside it, and
the passage goes into your next question, so the model is sure to get it, in either mode. Without a selection, RAG
answers the question as asked (retrieval runs on its words alone, so name the chapter or idea), while in agentic
RAG the earlier turns are candidates for each model call's context, and the cookbook's chapter 5 context system
selects what fits. Under the chat, a meter shows how many tokens it holds against what one call can carry (Groq's free tier
refuses requests over 8,000 tokens). **Compact now** replaces the older turns with a summary and the facts pulled
out of them, keeping the last three exchanges word for word. The chat lasts for your browser session; nothing is
stored.

Both use `openai/gpt-oss-120b` on Groq. Questions are matched against the cookbook with Gemini embeddings
(`gemini-embedding-001`, 768 dimensions) and BM25, and a question nothing in the cookbook is close to is turned
away rather than answered.

## Project layout

```
app.py                     the Gradio app (the Hugging Face Space runs this)
ask_the_cookbook/
    corpus.py              which cookbook files are read, and how
    search.py              the index, hybrid retrieval with a refusal floor, the one-shot engine
    agent.py               the agent: the cookbook's own loop and context system, over the same search
    session.py             a visitor's chat: follow-ups, the meter, Compact now
    refresh.py             brings the index up to date with the cookbook's latest commit
    runtime/               the agent's parts, from the cookbook's chapters 1-6
data/
    index/                 the saved LlamaIndex index (the vector store is in Git LFS)
    corpus/                the cookbook's text exactly as indexed; the app reads files from here
tests/                     offline tests: stand-in models and fake clients, no keys, no tokens
.github/workflows/         the sync to the Space, and the index refresh (run by hand from the Actions tab)
```

## The index

<!-- index-commit -->
Built from the cookbook at commit [`fb39ada`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/fb39adac66b77dcd2a6ed4f17300e6b415632b63): 124 files, 1,015 chunks.
<!-- /index-commit -->

`python -m ask_the_cookbook.refresh` brings it up to date with the cookbook's latest commit on main. It compares
the cookbook file by file against the index, embeds only new or changed files, removes deleted ones, and rewrites
`data/corpus`. Add `--dry-run` to see what would change, and how many embeddings it would cost, without a key.
The **Refresh the index** workflow runs the same thing on GitHub and commits the result.

## Running it locally

```
conda env create -f environment.yml
conda activate ask-the-cookbook
```

Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` and `GEMINI_API_KEY`. Then:

```
python app.py              # the app, at http://127.0.0.1:7860 (questions spend tokens on your keys)
pytest                     # the tests, offline
```

Without conda: `pip install -r requirements-dev.txt` on Python 3.12 (the Space's).

## Where the code comes from

The assistant was built in the cookbook, in `08_frameworks/05_llamaindex`. Its code was copied from commit
[`fb39ada`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/fb39ada) and arranged as a package:

| Here | In the cookbook |
|---|---|
| `ask_the_cookbook/corpus.py` | `08_frameworks/05_llamaindex/cookbook.py` |
| `ask_the_cookbook/search.py` | `08_frameworks/05_llamaindex/ask_index.py` |
| `ask_the_cookbook/agent.py` | `08_frameworks/05_llamaindex/agent_scratch.py` |
| `ask_the_cookbook/runtime/` | `01_foundations` (llm, chat), `02_agent_runtime` (think, act, observe, runtime), `03_tools` (tools, registry), `04_memory/recall.py`, `05_context` (context, selection, focus, compression, system), `06_orchestration/supervisor.py` |

What changed:
- **Trimmed to what the app can reach.** Code nothing in the app or the index refresh can call was removed: the
  other reasoning strategies (planner, tree search, reflection), the `Loop` and `Graph` runners, chapter 6's
  supervisor agents, chapter 4's ranking, the LlamaIndex agent and the evaluation helpers. What remains keeps its
  code and its comments, some of which still mention the removed parts as the reason for a design.
- **Imports** are relative, within the package; **paths** put the index and corpus under `data/`.
- **One addition**, so agent runs on different keys can go at once: an agent run can be given its own model
  client, and every model call it makes goes through it -- chapter 2's `Think`, `Act`'s repair of a failed
  tool call (`DebugThink`, via chapter 1's `extract`/`chat`), and chapter 5's compaction. Given none, they use
  chapter 1's shared client, as in the cookbook. `agent.run` also takes a chat's earlier turns.
