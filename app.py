"""Ask the Cookbook, as a Gradio app: the Hugging Face Space.

    RAG           one retrieval, then one answer with citations (search.engine), on the owner's Groq key:
                  DAILY_LIMIT questions a day, all visitors together, counted here -- the count starts again if the
                  Space restarts, and Groq's own daily limit still stands behind it
    your own key  a Groq key (gsk_...): questions on the visitor's own Groq account and its limits, not counted
                  here, and agentic RAG -- retrieval as a tool the model calls as
                  often as it decides, in the agent loop (agent: chapter 2's loop, chapter 5's context). The same
                  model either way, gpt-oss-120b, the one the agent was tuned on. Groq only: Hugging Face's router
                  serves the model too, but bills from credit a free account no longer has (402 "no remaining
                  credits", measured), and a Groq key is free

It is a chat (session.py), kept for the browser session only. Select text in an answer and an "Ask about" button
appears: the passage goes into the question, which RAG retrieves with and agentic RAG has pinned in its context.
Without a selection, RAG answers the question as asked; agentic RAG has the earlier turns as candidates for each
call's context, and chapter 5's selection decides what fits. A meter shows how much the chat holds, and Compact
now replaces older turns with a summary (chapter 5's compaction). Every message, and every compaction, on the
owner's key counts against DAILY_LIMIT.

Every question also costs one Gemini embedding (an agent question one per search) on the owner's key, whichever
key answers it: the index was embedded with Gemini, and a question has to be too.

The cookbook's text is read from data/corpus (ask_the_cookbook.refresh writes it).
"""

import concurrent.futures
import html
import json
import os
import random
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import openai

from ask_the_cookbook import agent as S
from ask_the_cookbook import search as A
from ask_the_cookbook import session as chat
from ask_the_cookbook.session import Session

COOKBOOK = "https://github.com/Venkatdatta2402/agentic-forge-cookbook"
BUILT = json.loads((A.INDEX_DIR / "cookbook.json").read_text(encoding="utf-8"))

DAILY_LIMIT = 50            # questions a day on the owner's key: ~3,200 Groq tokens each, of 200k a day
MAX_QUESTION = 500          # characters: a question, not a document to summarise on someone else's key

RAG, AGENTIC = "RAG", "Agentic RAG"

INDEX = A.open_index()


# --------------------------------------------------------------------------- the daily limit

class Daily:
    """Questions answered on the owner's key today (UTC), and none past the limit -- one count for every visitor
    and session. Taken before the question is answered, so visitors asking at once cannot go past it together.

    Kept in a file as well as in memory (`path`): restarting the app kept starting the count again at the limit.
    A Space's own disk is wiped when the Space restarts (unless it has persistent storage), so there it survives
    everything but that."""

    def __init__(self, limit, path=None):
        self.limit, self.day, self.used, self.path = limit, None, 0, path
        self.lock = threading.Lock()
        if path is not None and path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.day, self.used = datetime.fromisoformat(saved["day"]).date(), int(saved["used"])
            except (ValueError, KeyError, TypeError):
                pass                                    # unreadable: start the day's count again

    def _today(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.used = today, 0

    def _save(self):
        if self.path is None:
            return
        # written beside, then moved into place: a crash mid-write cannot leave half a count
        partial = self.path.with_suffix(".partial")
        partial.write_text(json.dumps({"day": self.day.isoformat(), "used": self.used}), encoding="utf-8")
        os.replace(partial, self.path)

    def take(self):
        with self.lock:
            self._today()
            if self.used >= self.limit:
                return False
            self.used += 1
            self._save()
            return True

    def left(self):
        with self.lock:
            self._today()
            return self.limit - self.used


# on a Space with persistent storage, point DAILY_FILE at it (/data/...); by default, beside the app
DAILY_FILE = Path(os.environ.get("DAILY_FILE", Path(__file__).resolve().parent / "daily_count.json"))
DAILY = Daily(DAILY_LIMIT, DAILY_FILE)


# --------------------------------------------------------------------------- the models

_SWAP = threading.Lock()


def chat_model(key):
    """ask_index.chat_model on any Groq key. It reads the key from GROQ_API_KEY when it is called and takes none as
    an argument -- the copied code stays as it is -- so the key is set for the call and put back, under a lock.
    The clients it makes keep the key they were given."""
    with _SWAP:
        saved = os.environ["GROQ_API_KEY"]
        os.environ["GROQ_API_KEY"] = key
        try:
            return A.chat_model(budget=A.Budget(label="engine"))
        finally:
            os.environ["GROQ_API_KEY"] = saved


def client(key):
    """A client for one agent run, or one compaction, alone: so calls at once on different keys stay apart."""
    return openai.OpenAI(api_key=key, base_url=A.GROQ, max_retries=8)


# --------------------------------------------------------------------------- answering

def blob(file):
    return f"{COOKBOOK}/blob/{BUILT['commit']}/{file}"


# A citation as gpt-oss writes it: [1], [1, 2], [Source 1], once live [Source 1, commentary in the same file], its
# own trained form 【1†L9-L15】 (source 1, lines 9-15 of the passage as it was shown) -- and, in prose, a bare
# "shown in Source 1". Each is placed by our own count of the passage's lines in its file, not by the model's.
# Not after a name or a number: `pairs[2]` is an index, not a citation (live, inside a table's code, it was linked).
_CITATION = re.compile(r"(?<!\w)\[(?:Source\s*)?(\d+(?:\s*,\s*(?:Source\s*)?\d+)*)([^\[\]]*)\]|【(\d+)(?:†[^】]*)?】"
                       r"|(?<!\[)\bSource\s+(\d+)\b")
# Code -- `inline` or ``` fenced -- is the model quoting the cookbook, and nothing in it is a citation
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)


def outside_code(text, change):
    """`text` with `change` applied to everything but its code."""
    parts, last = [], 0
    for code in _CODE.finditer(text):
        parts += [change(text[last:code.start()]), code.group()]
        last = code.end()
    return "".join(parts) + change(text[last:])


def citations(text, render):
    """Every numbered citation in `text` -- outside code -- replaced by render(numbers, rest), where it returns
    something."""
    def one(match):
        numbers = [int(n) for n in re.findall(r"\d+", match[1] or match[3] or match[4])]
        return render(numbers, match[2] or "") or match[0]
    return outside_code(text, lambda prose: _CITATION.sub(one, prose))


# gpt-oss sometimes ends an answer with a "Sources" list of its own, above the page's real one: a heading, then list
# items, to the end. Dropped when that is all it is; a heading followed by more prose is left alone.
_OWN_SOURCES = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*Sources?[ \t]*\**[ \t]*:?[ \t]*\**[ \t]*$", re.M | re.I)


def without_own_sources(answer):
    heads = list(_OWN_SOURCES.finditer(answer))
    if not heads:
        return answer
    after = [line for line in answer[heads[-1].end():].splitlines() if line.strip()]
    if after and all(re.match(r"\s*(?:[-*•]|\d+[.)])\s", line) for line in after):
        return answer[:heads[-1].start()].rstrip()
    return answer


def cited(text, places):
    """The answer the chat keeps: each citation as "file, lines a-b", which means something later."""
    return citations(text, lambda numbers, rest: all(n in places for n in numbers)
                     and "[" + "; ".join(places[n] for n in numbers) + rest + "]")


def linked(text, links, renumber=None):
    """The answer the page shows: each citation as a small [n] that opens the cited lines on GitHub -- numbered as
    the Sources list numbers it (`renumber`, where passages of one section were merged into one source)."""
    renumber = renumber or {}

    def render(numbers, rest):
        if not all(n in links for n in numbers):
            return None
        shown = dict.fromkeys((renumber.get(n, n), links[n]) for n in numbers)      # merged sources once
        return "".join(f"[[{m}]]({url})" for m, url in shown)
    return citations(text, render)


def span(node):
    """The lines of its file a passage covers, counted as the agent's open_file and search results count them (a
    notebook by its prose), or None if its text is no longer found there."""
    start = A.first_line(node)
    if start is None:
        return None
    lines = node.get_content().splitlines()
    # first_line() places the passage by its first line of 12 or more characters -- past the "Source 1:" header
    # CitationQueryEngine puts on each passage -- so the count starts there too
    body = lines[next(n for n, line in enumerate(lines) if len(line.strip()) >= 12):]
    while body and not body[-1].strip():
        body.pop()
    return start, start + len(body) - 1


def where(node):
    """A source passage as "file, lines a-b", for the chat to keep. Just the file if its lines cannot be placed."""
    file, lines = node.metadata["file"], span(node)
    return f"{file}, lines {lines[0]}-{lines[1]}" if lines else file


def heading_above(file, line):
    """The heading a passage sits under: the nearest markdown heading at or above its first line, in the file's text
    as indexed (a notebook's prose). Its own `header_path` will not do -- it lists the headings ABOVE the passage's
    section, and a passage that continues a section has no heading of its own."""
    heading, fenced = None, False
    for text in A.file_text(file).splitlines()[:line]:
        if text.lstrip().startswith("```"):
            fenced = not fenced                     # a "# comment" in a code example is not a heading
        elif not fenced and re.match(r"#{1,6} ", text):
            heading = text.lstrip("#").strip()
    return heading


def source(node):
    """(how the page names a source passage, the link that opens it). A .py or .md file opens on GitHub at the
    passage's lines -- a README in its raw form, the one GitHub can point at a line in. A notebook opens on GitHub
    from the top: GitHub shows notebooks in a frame no link can scroll, and nbviewer, which could, was refused by
    GitHub's rate limit when tried live. It is named by the heading the passage sits under, to find it by, and
    carries no line numbers: ours count its prose as indexed, which GitHub does not show as such."""
    file, lines = node.metadata["file"], span(node)
    heading = heading_above(file, lines[0]) if lines and not file.endswith(".py") else None
    section = f" § {heading}" if heading else ""
    if file.endswith(".ipynb"):
        return file + section, blob(file)
    if not lines:
        return file, blob(file)
    raw = "?plain=1" if file.endswith(".md") else ""
    return f"{file}, lines {lines[0]}-{lines[1]}{section}", f"{blob(file)}{raw}#L{lines[0]}-L{lines[1]}"


FOLLOW_UP = ("To follow up on an earlier answer, select the part you mean and press **Ask about**, or ask a question "
             "that names the chapter or idea.")


def rag(question, key, stage):
    """search.ask in its two halves, so the page can say which is happening (`stage["now"]`): the search, then the
    answer. A question the floor turns away is never sent to the model at all."""
    from llama_index.core import QueryBundle
    engine = A.engine(INDEX, chat_model(key))
    stage["now"] = "Searching the cookbook"
    nodes = engine.retrieve(QueryBundle(question))
    if not nodes:
        return {"refused": True, "answer": A.NOT_COVERED, "response": None}
    stage["now"] = f"Reading {len(nodes)} passages and writing the answer"
    response = engine.synthesize(QueryBundle(question), nodes)
    return {"refused": not response.source_nodes, "answer": str(response).strip(), "response": response}


def ask_engine(question, key, session, stage=None):
    """(what the page shows, what the chat keeps). Retrieval runs on the question as asked -- with the passage, when
    the visitor asked about one. The chat keeps the answer with its numbered citations turned into "file, lines a-b",
    which mean something later, and are lines a later agentic turn can open."""
    result = rag(question, key, stage if stage is not None else {})
    if result["refused"]:
        text = f"{A.NOT_COVERED} Nothing in it is close enough to the question to answer from."
        # a follow-up turned away is most often one that leaned on the chat: retrieval saw only its own words
        return (f"{text}\n\n{FOLLOW_UP}" if session.history else text), text
    nodes = dict(enumerate((s.node for s in result["response"].source_nodes), 1))
    named = {n: source(node) for n, node in nodes.items()}
    # passages named and linked alike -- two cut from one notebook section -- are one source in the list
    listed, renumber = [], {}
    for n, entry in named.items():
        if entry not in listed:
            listed.append(entry)
        renumber[n] = listed.index(entry) + 1
    sources = "\n".join(f"{m}. [{name}]({url})" for m, (name, url) in enumerate(listed, 1))
    answer = without_own_sources(result["answer"])
    shown = linked(answer, {n: url for n, (_, url) in named.items()}, renumber)
    kept = cited(answer, {n: where(node) for n, node in nodes.items()})
    return f"{shown}\n\n**Sources**\n\n{sources}", kept


# The agent cites files its own way, copying its search results' labels more or less: [04_memory/recall.py],
# [ap_checks.py, line 4‑5], [05_context/notebooks/02_context_selection.ipynb § Context Selection, lines 64-74]. A bare
# name is found among the cookbook's files when it names exactly one of them. A markdown link is left alone.
# Live, it also wrote them in gpt-oss's own brackets: 【08_frameworks/02_langgraph/ap_checks.py, lines 5‑6】.
_CITED = re.compile(r"[\[【]([^\[\]【】]+?\.(?:py|md|ipynb))([^\[\]【】]*)[\]】](?!\()")
# "lines 5-6" with any dash, and gpt-oss's own "†L19-L24"
_LINES = re.compile(r"(?:lines?\s*|†L)(\d+)(?:\s*[-‐‑‒–—]\s*L?(\d+))?")
FILES = [p.relative_to(A.CB.REPO).as_posix() for p in A.CB.files()]


def resolve(name):
    """A cited file as its path in the cookbook, or None if it names none, or more than one."""
    name = name.strip().lstrip("./")
    if name in FILES:
        return name
    found = [f for f in FILES if f.endswith("/" + name)]
    return found[0] if len(found) == 1 else None


def file_link(file, start=None, end=None):
    """The link that opens a cited file: on GitHub at its lines when it has them (a README raw, the form GitHub can
    point at a line in); a notebook from the top, whose lines no GitHub view shows as ours count them."""
    if start is None or file.endswith(".ipynb"):
        return blob(file)
    raw = "?plain=1" if file.endswith(".md") else ""
    return f"{blob(file)}{raw}#L{start}-L{end or start}"


def link_files(answer):
    """The agent's file citations as links, each shown as the agent wrote it."""
    def one(match):
        file = resolve(match[1])
        if file is None:
            return match[0]
        lines = _LINES.search(match[2])
        url = file_link(file, *(int(n) if n else None for n in lines.groups())) if lines else file_link(file)
        return f"[[{match[1]}{match[2]}]]({url})"
    return outside_code(answer, lambda prose: _CITED.sub(one, prose))


def log_run(result):
    """An agentic run, into the server's log (the Space's logs, which only its owner reads): each call's context in
    tokens and routes, every tool call, why it stopped, and what the last call was given -- labels and counts, never
    the text, and never a key. What the per-call selection actually let through is otherwise invisible."""
    lines = [f"agentic run: {len(result['calls'])} tool calls, {result['usage']['total']:,} tokens, "
             f"stopped: {result['stopped'] or 'no'}"]
    lines += [f"  call {n}: {trace}" for n, trace in enumerate(result["context per call"], 1)]
    lines += ["  model calls, tokens in / out (Groq checks a request's input; both count per minute): "
              + ", ".join(f"{i:,}/{o:,}" for i, o in result["usage"]["per call (in, out)"])]
    lines += [f"  tool: {c['tool']} {str(c['args'])[:120]}" for c in result["calls"]]
    lines += ["  last call was given:"] + [f"    {item}" for item in result["last call"]]
    print("\n".join(lines), flush=True)


def ask_agent(question, key, session, stage=None):
    """(what the page shows, what the chat keeps). The run starts from the chat so far."""
    own = client(key)
    send, refused = own.chat.completions.create, []

    # agent.run reports a refused request as why the run stopped, in words; the error itself is kept here, so a bad
    # key or a spent limit can be told apart. The client is this run's alone, so this touches no other run
    def create(*args, **kwargs):
        try:
            return send(*args, **kwargs)
        except openai.APIStatusError as error:
            refused.append(error)
            raise

    own.chat.completions.create = create
    stage = stage if stage is not None else {}
    result = S.run(INDEX, question, A.Budget(label="agent"), client=own, history=session.history, notes=session.notes,
                   watch=lambda conversation: stage.update(conversation=conversation))
    log_run(result)
    answer = result["answer"]
    if not answer:
        if (result["stopped"] or "").startswith("BudgetExceeded"):
            text = "The agent used its token budget before it had an answer. Try a narrower question."
            return text, text
        if refused:
            raise refused[-1]
        raise RuntimeError(f"the agent stopped without an answer ({result['stopped']})")
    linked = link_files(answer)
    steps = "\n".join(f"- `{c['tool']}` {str(c['args'])[:150]}" for c in result["calls"])
    usage = result["usage"]
    return (f"{linked}\n\n<details><summary>How it got there: {len(result['calls'])} tool calls, "
            f"{usage['total']:,} tokens on your key</summary>\n\n{steps}\n\n</details>"), answer


def trouble(error, own_key):
    """What went wrong, in words a visitor can act on. Never the key, and never a stack trace."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    gemini = type(error).__module__.startswith("google")
    if gemini and status == 429:
        return "The search has used up its quota for today (Gemini's free tier). Please come back tomorrow."
    if isinstance(error, openai.AuthenticationError) or status in (401, 403):
        return "That key was refused. Check that it is a current Groq API key."
    if isinstance(error, openai.RateLimitError):
        if own_key:
            return "Your key has hit its rate limit. Wait a little, or until its daily limit resets."
        return "The demo's Groq key has reached its daily limit. Paste your own key in the sidebar, or come back tomorrow."
    if isinstance(error, openai.APIStatusError) and status == 413:
        return "That request was too large for the model's limit. Try a shorter or narrower question."
    print(f"error: {type(error).__name__}: {str(error)[:300]}", flush=True)
    return "Something went wrong answering that. Please try again."


def status(session, note=""):
    """The line under the chat: a note when there is one, then how much the chat holds."""
    return ((f"{note}\n\n" if note else "")
            + f"Chat holds **{session.held():,} tokens**. The agent can take up to **{chat.AGENT_CALL:,}** per model "
              f"call.")


def allowance(mine=False):
    """The sidebar's line on the demo key, beside the key box it is the alternative to. With a visitor's own key in
    the box, their questions are not counted -- and the line says so, rather than a count that does not move."""
    if mine:
        return ("Your key is in use: your questions go on your own Groq account, under its limits, and don't use "
                "the shared demo key.")
    # how much is left, not a number: the count is the app's own, and starts again when the Space restarts
    left = DAILY.left()
    state = ("used up for today -- paste your own key above, or come back tomorrow" if left <= 0
             else "running low" if left <= DAILY_LIMIT * LOW else "plenty left today")
    return f"Without a key, questions use a shared demo key: **{state}**."


LOW = 0.2           # the share of the day's demo-key questions left at which the sidebar says "running low"


def own(key):
    return bool((key or "").strip())


def whose_key(key, mode=None):
    """(the key to use, own key?, a problem to show instead), taking a daily question when it is the demo key's."""
    key = (key or "").strip()
    if key:
        if not key.startswith("gsk_"):
            return None, True, "That isn't a Groq API key: those start `gsk_`."
        return key, True, None
    if mode == AGENTIC:
        return None, False, "Agentic RAG runs on your own key: paste a Groq API key in the sidebar."
    if not DAILY.take():
        return None, False, ("Today's free questions are used up. Paste your own Groq API key in the sidebar to keep "
                             "asking, or come back tomorrow.")
    return os.environ["GROQ_API_KEY"], False, None


def respond(question, quote, key, mode, session):
    """One message, from the page: a question, and the passage it is about if the visitor pressed Ask about.
    Yields (the chat, the session, the status line, the demo key's allowance, the question box, the quote box)."""
    question, quote = (question or "").strip(), (quote or "").strip()
    mine = own(key)

    def turned_away(note, keep_question=True):
        return session.shown, session, status(session, note), allowance(mine), question if keep_question else "", quote

    if not question:
        yield turned_away("Type a question first.", keep_question=False)
        return
    if len(question) > MAX_QUESTION:
        yield turned_away(f"Please keep the question under {MAX_QUESTION} characters.")
        return
    if len(quote) > chat.QUOTE_LIMIT:
        yield turned_away(f"Please select at most {chat.QUOTE_LIMIT:,} characters to ask about "
                          f"(this is {len(quote):,}).")
        return
    key, own_key, problem = whose_key(key, mode)
    if problem:
        yield turned_away(problem)
        return

    message, stage, started, order = chat.asking(question, quote), {}, time.monotonic(), word_order()
    said = doing(mode, stage)
    session.shown += [{"role": "user", "content": chat.showing(question, quote)},
                      {"role": "assistant", "content": working(mode, stage, 0.0, order)}]
    yield session.shown, session, status(session), allowance(mine), "", ""
    # The answer is worked out in the background. Meanwhile the message carries the turning words (CSS, in the
    # browser) and a line on what is really happening -- re-sent only when that line changes. A re-sent message is
    # drawn afresh, so it carries how far into the words' cycle it is, and they go on where they were.
    job = WORKERS.submit(ask_agent if mode == AGENTIC else ask_engine, message, key, session, stage)
    while not concurrent.futures.wait([job], timeout=TICK).done:
        if doing(mode, stage) != said:
            said = doing(mode, stage)
            session.shown[-1] = {"role": "assistant",
                                 "content": working(mode, stage, time.monotonic() - started, order)}
            yield session.shown, session, status(session), allowance(mine), "", ""
    try:
        shown, kept = job.result()
        session.add(message, kept)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        shown = trouble(error, own_key)
    session.shown[-1] = {"role": "assistant", "content": shown}
    yield session.shown, session, status(session), allowance(mine), "", ""


WORKERS = concurrent.futures.ThreadPoolExecutor(max_workers=16)      # answers being worked out, behind the page
TICK = 0.5                                                           # seconds between looks at what is happening


def doing(mode, stage):
    """What is really happening: for RAG the search or the writing; for agentic RAG the step it is on, and how many
    calls in."""
    if mode != AGENTIC:
        return stage.get("now", "Searching the cookbook")
    conversation = stage.get("conversation")
    calls = [m for m in (conversation.messages if conversation else [])
             if m.get("role") == "decision" and m.get("type") == "tool_call"]
    if not calls:
        now = "Planning its first search"
    elif "result" in calls[-1]:
        now = f"Read what call {len(calls)} found; choosing the next step"
    else:
        now = step(calls[-1])
    return (f"{now} · call {len(calls)} of up to {S.A.MAX_TOOL_CALLS} · each step waits for Groq's per-minute token "
            "limit, so this takes a few minutes")


def working(mode, stage, elapsed=0.0, order=None):
    """The waiting message: the turning words, in `order`, `elapsed` seconds into their cycle, and under them what is
    really happening. One paragraph with a line break: as two, a paragraph's spacing hung between them. Escaped: the
    step can carry the model's own search words."""
    return f'{cooking(elapsed, order)}<br><span class="doing">{html.escape(doing(mode, stage))}</span>'


def step(call):
    """A tool call as the visitor would say it."""
    try:
        args = json.loads(call["content"])
    except (TypeError, ValueError):
        args = {}
    if call["tool"] == "search_cookbook":
        return f"Searching for “{args.get('query', '')}”"
    if call["tool"] == "open_file":
        return f"Reading {args.get('path', 'a file')}, lines {args.get('line_start', 1)}–{args.get('line_end', '')}"
    return f"Calling {call['tool']}"


def compact_now(key, session):
    """Compact now: older turns replaced by a summary and facts, by chapter 5's compaction. Yields (the chat, the
    session, the status line, the demo key's allowance). On the demo key it counts as a question; on a visitor's key
    the calls go on theirs."""
    mine = own(key)
    if not session.can_compact():
        note = (f"Not worth compacting yet: {session.compactable():,} tokens are older than the last "
                f"{chat.KEEP_RECENT // 2} exchanges (which are always kept as they are), and a summary is only "
                f"smaller once at least {chat.COMPACT_WORTH:,} are.")
        yield session.shown, session, status(session, note), allowance(mine)
        return
    key, own_key, problem = whose_key(key)
    if problem:
        yield session.shown, session, status(session, problem), allowance(mine)
        return
    yield session.shown, session, status(session, "*Compacting older turns...*"), allowance(mine)
    try:
        before, after = session.compact(client(key) if own_key else None)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        yield session.shown, session, status(session, trouble(error, own_key)), allowance(mine)
        return
    if after >= before:
        note = "Compacting would not have made the chat any smaller, so it was left as it was."
        yield session.shown, session, status(session, note), allowance(mine)
        return
    note = (f"*Older turns compacted: the chat now holds {after:,} tokens, down from {before:,}. The transcript above "
            "stays; the model gets a summary of the older part.*")
    session.shown.append({"role": "assistant", "content": note})
    yield session.shown, session, status(session), allowance(mine)


def new_chat(key):
    fresh = Session()
    return fresh.shown, fresh, status(fresh), allowance(own(key))


def example(choice: gr.SelectData):
    """An example question, clicked in the empty chat, into the question box -- not sent, so the mode can be chosen."""
    return choice.value["text"]


# --------------------------------------------------------------------------- the page

EXAMPLES = [
    "What counts as a doom loop, and what does the runtime do when it detects one?",
    "Why shouldn't agent memory start with a vector database?",
    "What is the difference between handing off a conversation and delegating to another agent?",
    "Why is CrewAI pinned to an old version in this repo?",
]

# "Ask about": select text in the chat and a small button appears beside it; pressing it puts the selection in the
# Asking about box. The page script keeps the selection and presses a hidden button whose event copies it into the box
# (an event's `js` with no Python function fills its outputs straight from the browser).
ASK_ABOUT_JS = """
() => {
  let chosen = "";
  const chip = document.createElement("button");
  chip.id = "ask-about-chip";
  chip.type = "button";
  chip.textContent = "Ask about \\u2197";
  chip.style.display = "none";
  document.body.appendChild(chip);

  const inChat = (node) => {
    const chat = document.getElementById("chat");
    return Boolean(chat && node && chat.contains(node.nodeType === 1 ? node : node.parentNode));
  };
  const hide = () => { chip.style.display = "none"; };

  document.addEventListener("mouseup", (event) => {
    if (event.target === chip) return;
    setTimeout(() => {                                   // after the selection has settled
      const selection = window.getSelection();
      const text = selection ? selection.toString().trim() : "";
      if (!text || !inChat(selection.anchorNode) || !inChat(selection.focusNode)) { hide(); return; }
      chosen = text;
      const box = selection.getRangeAt(0).getBoundingClientRect();
      chip.style.top = `${box.bottom + 6}px`;
      chip.style.left = `${Math.max(8, Math.min(box.right - 40, window.innerWidth - 130))}px`;
      chip.style.display = "block";
    }, 0);
  });
  document.addEventListener("scroll", hide, true);
  chip.addEventListener("mousedown", (event) => event.preventDefault());   // keep the selection while clicking
  chip.addEventListener("click", () => {
    window.askAboutText = chosen;
    hide();
    const trigger = document.getElementById("ask-about-trigger");
    (trigger.tagName === "BUTTON" ? trigger : trigger.querySelector("button")).click();
    window.getSelection().removeAllRanges();
    setTimeout(() => document.querySelector("#question textarea")?.focus(), 100);
  });
}
"""

# The words a waiting message turns through, WORD_SECONDS each, fading out and in at every change -- Ask the
# Cookbook, so kitchen words. Animated by CSS in the browser: the server never re-sends a message to move them.
COOKING = ["Preheating", "Gathering ingredients", "Leafing through chapters", "Reading the recipe", "Measuring",
           "Whisking", "Simmering", "Stirring", "Seasoning", "Tasting", "Reducing", "Folding in", "Marinating",
           "Plating", "Garnishing", "Letting it rest"]
WORD_SECONDS = 4.5          # 2, then 3, read as too fast, live

# All the words are in the message, stacked in one place; each fades in, holds, and fades out in its own slot of the
# cycle. Only opacity and position are animated: a first version swapped the text itself (`content` in keyframes),
# and the browser showed nothing at all.
# In a random order, the owner's choice: one per question, kept for all of it -- a re-drawn message must go on with
# the same sequence. ROUNDS shuffles end to end, so the words run ~3 minutes before the sequence repeats, an agentic
# run's usual length; where two shuffles meet, never the same word twice.
ROUNDS = 4
_SLOTS = len(COOKING) * ROUNDS
_CYCLE = _SLOTS * WORD_SECONDS


def word_order(rng=random):
    order = []
    while len(order) < _SLOTS:
        round_ = rng.sample(COOKING, len(COOKING))
        if order and round_[0] == order[-1]:
            round_[0], round_[-1] = round_[-1], round_[0]
        order += round_
    return order


def cooking(elapsed=0.0, order=None):
    """The words, in `order` (one random sequence per question), `elapsed` seconds into their cycle (--phase). A
    message re-sent because the step line changed is drawn afresh, and its animation would start again from the first
    word's fade-in -- live, the word seemed to transition into itself at every step. Started this far in, it carries
    on where the old drawing was."""
    words = "".join(f"<span>{word}…</span>" for word in (order or word_order()))
    return f'<span class="cooking" style="--phase: -{elapsed % _CYCLE:.2f}s">{words}</span>'


_SLOT = 100 / _SLOTS                             # one word's share of the cycle, in percent
COOKING_CSS = (
    ".cooking { display: inline-grid; font-style: italic; font-size: 1.05em; }\n"
    # background-IMAGE, not the `background` shorthand: Gradio rebuilds custom CSS from the browser's own reading of
    # it, and a shorthand holding var() with longhands set after it cannot be read back -- it vanished, leaving the
    # text transparent on nothing (found by rendering the page headless and reading the computed style)
    ".cooking > span { grid-area: 1 / 1; opacity: 0; white-space: nowrap;"
    " background-image: linear-gradient(90deg, var(--body-text-color-subdued) 0%, var(--body-text-color) 50%,"
    " var(--body-text-color-subdued) 100%); background-size: 200% 100%;"
    " -webkit-background-clip: text; background-clip: text; color: transparent;"
    f" animation: cooking-word {_CYCLE}s ease-in-out infinite, cooking-shimmer {WORD_SECONDS}s linear infinite; }}\n"
    + "".join(f".cooking > span:nth-child({n + 1}) {{ animation-delay: calc(var(--phase, 0s) + {n * WORD_SECONDS}s),"
              f" var(--phase, 0s); }}\n" for n in range(_SLOTS))
    + f"@keyframes cooking-word {{ 0% {{ opacity: 0; transform: translateY(4px); }}"
      f" {_SLOT * 0.2:.3f}% {{ opacity: 1; transform: none; }} {_SLOT * 0.8:.3f}% {{ opacity: 1; transform: none; }}"
      f" {_SLOT:.3f}% {{ opacity: 0; transform: translateY(-4px); }} 100% {{ opacity: 0; }} }}\n"
    "@keyframes cooking-shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }\n"
    "@media (prefers-reduced-motion: reduce) { .cooking > span { animation: none; }"
    " .cooking > span:first-child { opacity: 1; } }\n"
    ".doing { font-size: var(--text-sm); color: var(--body-text-color-subdued); }\n")

# Answers come with tables, often with code in a cell. Code does not wrap, so one long line could stretch a table
# past its message (live: a citation link written out inside code did); it wraps here, and a table still too wide
# scrolls inside its message instead.
TABLE_CSS = """
#chat .message-content table { display: block; max-width: 100%; overflow-x: auto; }
#chat .message-content table code { white-space: pre-wrap; word-break: break-word; }
"""

CSS = COOKING_CSS + TABLE_CSS + """
#ask-about-chip {
  position: fixed; z-index: 1000; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
  border: 1px solid var(--button-primary-border-color); background: var(--button-primary-background-fill);
  color: var(--button-primary-text-color); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}
#ask-about-trigger { display: none !important; }
/* the passage being asked about shows only once there is one */
#quote-box:has(textarea:placeholder-shown) { display: none !important; }
#quote-box { border-left: 3px solid var(--color-accent); }
/* the chat's state, quiet and on one line with the two things that act on it */
#chat-state { align-items: center; }
#meter, #meter p { font-size: var(--text-sm); color: var(--body-text-color-subdued); margin: 0; }
#allowance, #index { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
"""

with gr.Blocks(title="Ask the Cookbook") as demo:
    session = gr.State(Session())

    with gr.Sidebar(width=320):
        gr.Markdown(f"## Ask the Cookbook\nQuestions about [agentic-forge-cookbook]({COOKBOOK}), answered from its "
                    "READMEs, notebooks and code, with the files cited.")
        mode = gr.Radio([RAG, AGENTIC], value=RAG, label="Mode",
                        info="RAG: one search, an answer in seconds. Agentic RAG: searches as many times as the "
                             "question needs and reads files, for questions that span chapters; a few minutes.")
        key = gr.Textbox(label="Your Groq API key", type="password", placeholder="gsk_...",
                         info="Needed for Agentic RAG. With it, every question goes on your own Groq account, under "
                              "its limits, instead of the shared demo key. Used for your questions only, never stored.")
        gr.Markdown("Get one free at [console.groq.com/keys](https://console.groq.com/keys).")
        allowance_line = gr.Markdown(elem_id="allowance")
        gr.Markdown(f"Index: the cookbook at [`{BUILT['commit'][:7]}`]({COOKBOOK}/tree/{BUILT['commit']}), "
                    f"{BUILT['files']} files. The chat lasts for this browser session; nothing is stored.",
                    elem_id="index")

    conversation = gr.Chatbot(show_label=False, height="62vh", elem_id="chat",
                              examples=[{"text": q} for q in EXAMPLES],
                              placeholder="**Ask anything about the cookbook.** To follow up on part of an answer, "
                                          "select it and press **Ask about**.")
    with gr.Row(elem_id="chat-state"):
        meter = gr.Markdown(elem_id="meter")
        compact = gr.Button("Compact now", size="sm", scale=0, min_width=120)
        clear = gr.Button("New chat", size="sm", scale=0, min_width=100)
    with gr.Group(elem_id="quote-box"):
        with gr.Row():
            quote = gr.Textbox(label="Asking about", placeholder=" ", lines=1, max_lines=6, scale=8,
                               elem_id="quote")
            gr.ClearButton(quote, value="✕", size="sm", scale=0, min_width=40)
    question = gr.Textbox(show_label=False, placeholder="Ask about the cookbook...", lines=1, max_lines=6,
                          submit_btn=True, elem_id="question")
    ask_about = gr.Button("Ask about", elem_id="ask-about-trigger")

    # private: not offered to API clients, so the demo key's daily questions are asked through this page
    question.submit(respond, [question, quote, key, mode, session],
                    [conversation, session, meter, allowance_line, question, quote],
                    concurrency_limit=8, api_visibility="private")
    ask_about.click(None, None, quote, js="() => window.askAboutText || ''", api_visibility="private")
    conversation.example_select(example, None, question, api_visibility="private")
    compact.click(compact_now, [key, session], [conversation, session, meter, allowance_line], concurrency_limit=8,
                  api_visibility="private")
    clear.click(new_chat, key, [conversation, session, meter, allowance_line], api_visibility="private")
    key.change(lambda k: allowance(own(k)), key, allowance_line, api_visibility="private")
    demo.load(lambda s, k: (status(s), allowance(own(k))), [session, key], [meter, allowance_line],
              api_visibility="private")

demo.queue(max_size=32)

if __name__ == "__main__":
    demo.launch(js=ASK_ABOUT_JS, css=CSS, theme=gr.themes.Soft())
