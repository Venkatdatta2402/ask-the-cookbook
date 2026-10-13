# Ask the Cookbook

A question-answering assistant over [agentic-forge-cookbook](https://github.com/Venkatdatta2402/agentic-forge-cookbook), the repo it was built in
(`08_frameworks/05_llamaindex`). It answers from the cookbook's READMEs, notebook prose and modules, and cites
the files it used.

## Where the code comes from

Every file here is copied unchanged from the cookbook at commit [`fb39ada`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/fb39ada), at the same path, so the
scratch agent finds its chapters the way it does there:

| Path | What it is |
|---|---|
| `08_frameworks/05_llamaindex/` | the corpus rules, the index and its search, and the scratch agent |
| `01_foundations` | the model client |
| `02_agent_runtime` | the agent loop |
| `03_tools` | tools and their registry |
| `04_memory/recall.py` | ranking, which chapter 5's selection uses |
| `05_context` | what each model call is sent |
| `06_orchestration/supervisor.py` | `Think` and `verbatim` |

## The index

<!-- index-commit -->
Built from the cookbook at commit [`fb39ada`](https://github.com/Venkatdatta2402/agentic-forge-cookbook/tree/fb39adac66b77dcd2a6ed4f17300e6b415632b63): 124 files, 1,015 chunks.
<!-- /index-commit -->

`refresh_index.py` brings it up to date with the cookbook's latest commit: it embeds only new or changed
files, removes deleted ones, and writes `corpus/`, the cookbook's text exactly as indexed (notebooks as their
markdown cells), which the app reads files from.

## Setup

```
conda env create -f environment.yml
conda activate ask-the-cookbook
```

Copy `.env.example` to `.env` and fill in the two keys.
