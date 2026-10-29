"""Load test: 100 questions at once, against a stand-in model that takes LATENCY seconds -- as Groq and Gemini make a
real answer wait -- so it costs no tokens and no key. Three ways of serving them:

    threads        the page's way: a pool of 8 worker threads, each holding a question until its answer comes back
    async, cap 8   the API as deployed: asyncio, with at most 8 RAG answers worked out at once (api.RUNS_AT_ONCE),
                   which protects the free-tier quotas everyone shares -- the rest wait their turn
    async, no cap  the same API with the cap lifted: what one event-loop thread can hold open at once

The API runs in a real uvicorn server, and the questions are real HTTP requests. Each answer goes through the real
code -- retrieval over the real index (a stand-in embedding), the citation engine, sources and links -- and only the
model's reply is faked.

    python benchmarks/load_test.py
"""

import asyncio
import os
import pathlib
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GROQ_API_KEY"] = "gsk_placeholder_owner"        # nothing reaches Groq or Gemini: the models are faked
os.environ["GEMINI_API_KEY"] = "placeholder"
os.environ.pop("GOOGLE_API_KEY", None)

from llama_index.core import MockEmbedding  # noqa: E402
from llama_index.core.base.llms.types import CompletionResponse  # noqa: E402
from llama_index.core.llms import MockLLM  # noqa: E402

from ask_the_cookbook import search  # noqa: E402

search.embed_model = lambda key=None: MockEmbedding(embed_dim=search.EMBED_DIM)

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from ask_the_cookbook import answers, api  # noqa: E402

REQUESTS, LATENCY, THREADS = 100, 1.0, 8
ANSWER = "A doom loop is the same call again and again [1]."


class SlowModel(MockLLM):
    """A model that takes LATENCY seconds to reply: blocking its thread when called the blocking way, awaiting when
    awaited -- as the real client does on each path."""

    def complete(self, prompt, formatted=False, **kwargs):
        time.sleep(LATENCY)
        return CompletionResponse(text=ANSWER)

    async def acomplete(self, prompt, formatted=False, **kwargs):
        await asyncio.sleep(LATENCY)
        return CompletionResponse(text=ANSWER)


def threads():
    """The page's way: questions on a pool of THREADS worker threads, blocking calls."""
    started = time.monotonic()
    with ThreadPoolExecutor(THREADS) as pool:
        results = list(pool.map(lambda n: answers.rag(f"What counts as a doom loop? ({n})", "gsk_x", {}),
                                range(REQUESTS)))
    assert all(r["response"] is not None for r in results)
    return time.monotonic() - started, THREADS


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def through_the_api(cap):
    """REQUESTS real HTTP requests at once to the API in a uvicorn server, with at most `cap` RAG answers at once."""
    api.GATES = api.Gates({"rag": cap, "agentic": 2})
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(FastAPI(routes=api.router.routes), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    most_threads = []

    def count_threads():                    # the server process's threads, sampled while the requests are open
        while not done.is_set():
            most_threads.append(threading.active_count())
            time.sleep(0.05)
    done = threading.Event()
    threading.Thread(target=count_threads, daemon=True).start()

    async def go():
        limits = httpx.Limits(max_connections=REQUESTS * 2, max_keepalive_connections=REQUESTS * 2)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=300, limits=limits) as client:
            return await asyncio.gather(*(client.post(
                "/api/ask", json={"question": f"What counts as a doom loop? ({n})", "stream": False},
                headers={"X-Groq-Key": "gsk_x", "X-Forwarded-For": f"10.0.{n // 250}.{n % 250}"})
                for n in range(REQUESTS)))
    started = time.monotonic()
    replies = asyncio.run(go())
    took = time.monotonic() - started
    done.set()
    server.should_exit = True
    thread.join()
    assert all(r.status_code == 200 and r.json()["sources"] for r in replies), [r.text[:200] for r in replies[:3]]
    return took, max(most_threads) - 2                   # less this script's main thread and the sampler


def main():
    answers.chat_model = lambda key: SlowModel()
    answers.A.FLOOR = -1                                  # a stand-in embedding is near nothing: no floor here

    async def accepted(kind, key):
        return True
    api.check_key = accepted                              # no key to check: nothing reaches Groq
    api.LIMIT = api.PerMinute(REQUESTS * 10)              # one machine sends them all: no per-address limit here

    rows = [("threads (the page's way)", *threads(), f"{THREADS} worker threads"),
            ("async API, cap 8 (as deployed)", *through_the_api(8), "1 event loop"),
            ("async API, cap lifted", *through_the_api(REQUESTS), "1 event loop")]
    ideal = REQUESTS * LATENCY
    print(f"\n{REQUESTS} questions at once, each answer {LATENCY:g} s of waiting on the model "
          f"({ideal:g} s if served one by one)\n")
    print("| Serving | Total time | Answers per second | Server threads (most) | Handles the waiting with |")
    print("|---|---|---|---|---|")
    for name, took, threads_used, how in rows:
        print(f"| {name} | {took:.1f} s | {REQUESTS / took:.1f} | {threads_used} | {how} |")


if __name__ == "__main__":
    main()
