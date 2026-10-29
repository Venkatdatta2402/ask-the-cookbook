"""The API: ask the cookbook from code, with the answers the page gives, over HTTP.

    POST /api/ask         a question. Answered as server-sent events -- what is happening as it happens, then the
                          answer with its sources -- or, with "stream": false, as one JSON reply
    GET  /api/health      up or not, the model, the cookbook commit the index holds, the limits
    GET  /api/docs        interactive docs (OpenAPI), for these routes only

Async throughout: a request waiting on Gemini or Groq holds no thread while it waits, so one process keeps many open
at once. RAG is asyncio all the way down (answers.arag). The agent is the cookbook's own loop, synchronous by design
(its chapter 2 README says why): it runs on a thread, and the API follows its progress from the event loop.

Keys come with each request and are used for it alone -- never logged, never kept (a key that passed its check is
remembered for a while as a hash):
    X-Groq-Key     required. The answer is written on the caller's own Groq account; the API never uses the demo key
    X-Gemini-Key   required for agentic mode, whose up to 8 searches are each an embedding; optional for RAG, which
                   otherwise embeds its one search on the owner's key
A key is checked with its provider before any work (a free call: listing a model): a made-up Groq key would
otherwise have had its question embedded on the owner's Gemini key before Groq turned it away.

Limits, so one caller cannot use up what the page's visitors share: PER_MINUTE requests a minute from one address,
and at most RUNS_AT_ONCE answers per mode worked out at once -- a request over it waits its turn, it is not refused.
A caller who disconnects stops the work: RAG at once (its awaits are cancelled), an agent run before its next model
call (answers.agent_parts' `stop`), its slot held until its thread has ended.
"""

import asyncio
import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from typing import Literal

import openai
from fastapi import APIRouter, Header, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import answers as ans
from . import search as A
from .session import QUOTE_LIMIT, Session, asking

PER_MINUTE = 5                              # requests a minute from one address
RUNS_AT_ONCE = {"rag": 8, "agentic": 2}     # answers worked out at once, per mode; more wait their turn
TICK = 0.5                                  # seconds between looks at an agent run's progress
KEEPALIVE = 15                              # seconds of silence before a comment line keeps the stream open
KEY_GOOD_FOR = 600                          # seconds a key that passed its check is not checked again

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- what is asked, what comes back

class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=ans.MAX_QUESTION, description="The question.",
                          examples=["What counts as a doom loop, and what does the runtime do when it detects one?"])
    mode: Literal["rag", "agentic"] = Field("rag", description="`rag`: one search, an answer in seconds. `agentic`: "
                                            "searches and reads files as often as needed; minutes.")
    quote: str = Field("", max_length=QUOTE_LIMIT, description="A passage the question is about (the page's Ask about).")
    stream: bool = Field(True, description="Server-sent events as it goes (true), or one JSON reply at the end.")

    @field_validator("question", "quote", mode="before")
    @classmethod
    def stripped(cls, value):
        return value.strip() if isinstance(value, str) else value


class Source(BaseModel):
    n: int
    name: str
    url: str


class ToolCall(BaseModel):
    tool: str
    args: str


class Answer(BaseModel):
    answer: str = Field(description="Markdown; each citation a link to the cited lines on GitHub.")
    sources: list[Source]
    tool_calls: list[ToolCall] = Field(description="The agent's searches and file reads (agentic mode).")
    refused: bool = Field(description="Nothing in the cookbook was close enough to the question to answer from.")
    tokens: int | None = Field(description="Groq tokens spent, on the caller's key.")
    seconds: float


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- limits

class PerMinute:
    """At most `limit` requests in any 60 seconds from one address: a sliding window of their times. Only the event
    loop's thread touches it, and `wait` never awaits, so it needs no lock."""

    def __init__(self, limit, clock=time.monotonic):
        self.limit, self.clock, self.seen = limit, clock, defaultdict(deque)

    def wait(self, who):
        """0 if a request from `who` may go now (and it is counted); otherwise the seconds until one may."""
        now = self.clock()
        if len(self.seen) > 10_000:                     # addresses not heard from in a minute are forgotten
            self.seen = defaultdict(deque, {k: v for k, v in self.seen.items() if v and now - v[-1] < 60})
        times = self.seen[who]
        while times and now - times[0] >= 60:
            times.popleft()
        if len(times) >= self.limit:
            return 60 - (now - times[0])
        times.append(now)
        return 0


class Gates:
    """RUNS_AT_ONCE as semaphores. Made for the running event loop: an asyncio semaphore belongs to the loop it was
    first waited in, and each test runs in a loop of its own."""

    def __init__(self, sizes):
        self.sizes, self.loop, self.gates = sizes, None, {}

    def __call__(self, mode):
        loop = asyncio.get_running_loop()
        if loop is not self.loop:
            self.loop, self.gates = loop, {m: asyncio.Semaphore(n) for m, n in self.sizes.items()}
        return self.gates[mode]


LIMIT = PerMinute(PER_MINUTE)
GATES = Gates(RUNS_AT_ONCE)


def address(request):
    """Who is asking. Behind Hugging Face's proxy every request comes from the proxy, and the caller's address is the
    first in X-Forwarded-For. A caller can write an address of its own there: this limit is a guard rail against a
    runaway script, not a security boundary -- the keys are that."""
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


# --------------------------------------------------------------------------- keys

_ACCEPTED = {}          # sha256 of a key that passed its check -> when it stops counting as checked


async def check_key(kind, key):
    """True if the provider accepts the key, False if it refuses it. A free call each: Groq lists its models, Gemini
    describes the embedding model. Any other failure (the provider down) raises."""
    digest = hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()
    if _ACCEPTED.get(digest, 0) > time.monotonic():
        return True
    if kind == "groq":
        try:
            async with openai.AsyncOpenAI(api_key=key, base_url=A.GROQ, max_retries=2) as groq:
                await groq.models.list()
        except openai.AuthenticationError:
            return False
    else:
        from google import genai
        from google.genai import errors
        try:
            await genai.Client(api_key=key).aio.models.get(model=A.EMBED_MODEL)
        except errors.ClientError as error:
            if error.code in (400, 401, 403):
                return False
            raise
    _ACCEPTED[digest] = time.monotonic() + KEY_GOOD_FOR
    return True


def problem(error, own_gemini):
    """(HTTP status, what went wrong in words) for an error while answering. Never the key."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if type(error).__module__.startswith("google"):
        if status == 429:
            return 429, ("Your Gemini key has hit its limit." if own_gemini else
                         "The search's embedding quota for today (Gemini's free tier) is used up: send your own "
                         "X-Gemini-Key, or try again tomorrow.")
        if own_gemini and status in (400, 401, 403):
            return 401, "That Gemini key was refused."
    if isinstance(error, openai.AuthenticationError):
        return 401, "That Groq key was refused."
    if isinstance(error, openai.RateLimitError):
        return 429, ("Your Groq key has reached its daily limit." if "per day" in str(error) else
                     "Your Groq key has hit its per-minute limit: try again in a minute.")
    if isinstance(error, openai.APIStatusError) and status == 413:
        return 413, "That request was too large for the model's limit. Try a shorter or narrower question."
    print(f"api error: {type(error).__name__}: {str(error)[:300]}", flush=True)
    return 500, "Something went wrong answering that."


# --------------------------------------------------------------------------- the work

class Stage(dict):
    """answers.arag's `stage`, which says what is happening by setting "now": here each one also goes into the
    progress queue. Set from the event loop's own thread, so put_nowait is safe."""

    def __init__(self, progress):
        super().__init__()
        self.progress = progress

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "now":
            self.progress.put_nowait(value)


async def answer(body, groq, gemini, progress):
    """One question's answer, as Answer's fields; what is happening goes into `progress` on the way."""
    started, message = time.monotonic(), asking(body.question, body.quote)
    embed = ans.embedding(gemini) if gemini else None
    gate = GATES(body.mode)
    if gate.locked():
        progress.put_nowait("Waiting for a free slot: other answers are being worked out")

    if body.mode == "rag":
        async with gate:
            result = await ans.arag(message, groq, Stage(progress), embed)
        if result["refused"]:
            return {"answer": f"{A.NOT_COVERED} Nothing in it is close enough to the question to answer from.",
                    "sources": [], "tool_calls": [], "refused": True, "tokens": result["tokens"],
                    "seconds": round(time.monotonic() - started, 2)}
        parts = ans.rag_parts(result)
        return {"answer": parts["answer"], "sources": parts["sources"], "tool_calls": [], "refused": False,
                "tokens": result["tokens"], "seconds": round(time.monotonic() - started, 2)}

    # agentic: the cookbook's loop on a thread. Its slot is held until the thread has ended -- not just until this
    # coroutine is cancelled -- so a caller who leaves cannot free a slot while their run is still spending
    await gate.acquire()
    stop, stage = threading.Event(), {}
    try:
        job = asyncio.ensure_future(asyncio.to_thread(ans.agent_parts, message, groq, Session(), stage, embed, stop))
    except BaseException:
        gate.release()
        raise
    # ...and whatever it ends with is collected, so a run stopped after its caller left is not reported as unread
    job.add_done_callback(lambda done: (gate.release(), done.cancelled() or done.exception()))
    said = None
    try:
        while True:
            now = ans.doing(ans.AGENTIC, stage)
            if now != said:
                said = now
                progress.put_nowait(now)
            finished, _ = await asyncio.wait({job}, timeout=TICK)
            if finished:
                break
    except asyncio.CancelledError:
        stop.set()          # the caller left: the run ends before its next model call
        raise
    parts = job.result()
    return {"answer": parts["answer"], "sources": [], "refused": False, "tokens": parts["tokens"],
            "tool_calls": [{"tool": c["tool"], "args": str(c["args"])} for c in parts["calls"]],
            "seconds": round(time.monotonic() - started, 2)}


async def events(body, groq, gemini):
    """The answer as server-sent events: `progress` as it happens, then `answer` and `done` -- or `error`. A comment
    line every KEEPALIVE seconds of silence keeps proxies from closing a long agent run's stream. When the caller
    disconnects, the server cancels this generator; the `finally` passes that on to the work."""
    progress = asyncio.Queue()
    work = asyncio.ensure_future(answer(body, groq, gemini, progress))
    try:
        while not work.done():
            getter = asyncio.ensure_future(progress.get())
            done, _ = await asyncio.wait({work, getter}, timeout=KEEPALIVE, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                yield sse("progress", {"message": getter.result()})
            else:
                getter.cancel()
                if not done:
                    yield ": keep-alive\n\n"
        while not progress.empty():                     # what was said as the work finished
            yield sse("progress", {"message": progress.get_nowait()})
        try:
            result = work.result()
        except Exception as error:  # noqa: BLE001 -- sent to the caller as words, not a trace
            status, message = problem(error, own_gemini=bool(gemini))
            yield sse("error", {"status": status, "message": message})
            return
        yield sse("answer", {k: v for k, v in result.items() if k != "seconds"})
        yield sse("done", {"seconds": result["seconds"], "tokens": result["tokens"]})
    finally:
        if not work.done():
            work.cancel()


# --------------------------------------------------------------------------- the routes

def turned_away(status, message, headers=None):
    return JSONResponse({"detail": message}, status_code=status, headers=headers)


ASK_DOC = """Ask a question about the cookbook.

**Streaming** (the default) answers with `text/event-stream`:
- `progress` `{"message"}`: what is happening, as it happens
- `answer`: the answer, its sources and tool calls (the `Answer` fields but `seconds`)
- `done` `{"seconds", "tokens"}`; or, instead of both, `error` `{"status", "message"}`

With `"stream": false`, one JSON reply: `Answer`.

Disconnecting stops the work: RAG at once, an agent run before its next model call."""


@router.post("/ask", summary="Ask the cookbook", description=ASK_DOC, response_model=Answer,
             responses={200: {"content": {"text/event-stream": {}}},
                        401: {"description": "A key is missing, malformed, or refused by its provider"},
                        422: {"description": "The question is empty, or too long"},
                        429: {"description": "Over the rate limit (see Retry-After), or a key's own limit"}})
async def ask(body: Ask, request: Request,
              x_groq_key: str | None = Header(None, description="Your Groq API key (gsk_...): it pays for the answer."),
              x_gemini_key: str | None = Header(None, description="Your Gemini API key: required for agentic mode; "
                                                                  "optional for RAG.")):
    groq, gemini = (x_groq_key or "").strip(), (x_gemini_key or "").strip() or None
    if not groq.startswith("gsk_"):
        return turned_away(401, "Send your Groq API key (gsk_...) as X-Groq-Key: the answer is written on it.")
    if body.mode == "agentic" and gemini is None:
        return turned_away(401, "Agentic mode needs your Gemini API key as X-Gemini-Key: each of its searches is an "
                            "embedding.")
    wait = LIMIT.wait(address(request))
    if wait:
        return turned_away(429, f"Over {PER_MINUTE} requests a minute: try again in {wait:.0f} s.",
                       {"Retry-After": str(int(wait) + 1)})
    try:
        if not await check_key("groq", groq):
            return turned_away(401, "That Groq key was refused.")
        if gemini is not None and not await check_key("gemini", gemini):
            return turned_away(401, "That Gemini key was refused.")
    except Exception as error:  # noqa: BLE001 -- the provider could not be reached
        print(f"api key check: {type(error).__name__}: {str(error)[:200]}", flush=True)
        return turned_away(503, "Could not reach Groq or Gemini to check the key. Try again shortly.")

    if body.stream:
        return StreamingResponse(events(body, groq, gemini), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # one JSON reply: no stream for the server to cancel when the caller leaves, so it is watched for here
    work = asyncio.ensure_future(answer(body, groq, gemini, asyncio.Queue()))
    while not work.done():
        await asyncio.wait({work}, timeout=1)
        if not work.done() and await request.is_disconnected():
            work.cancel()
            return Response(status_code=499)            # nobody left to send it to
    try:
        return work.result()
    except Exception as error:  # noqa: BLE001 -- sent to the caller as words, not a trace
        status, message = problem(error, own_gemini=gemini is not None)
        return turned_away(status, message)


@router.get("/health", summary="Up, and what it serves")
async def health():
    return {"status": "ok", "model": A.MODEL,
            "index": {"commit": ans.BUILT["commit"], "files": ans.BUILT["files"], "chunks": ans.BUILT["chunks"]},
            "limits": {"requests_per_minute": PER_MINUTE, "at_once": RUNS_AT_ONCE}}


@router.get("/openapi.json", include_in_schema=False)
async def openapi():
    return get_openapi(title="Ask the Cookbook API", version="1.0.0", routes=router.routes,
                       description="Questions about agentic-forge-cookbook, answered with links to the lines cited. "
                                   "Your own keys, sent as headers, pay for each answer.")


@router.get("/docs", include_in_schema=False)
async def docs():
    return get_swagger_ui_html(openapi_url="/api/openapi.json", title="Ask the Cookbook API")
