"""Answering a question about the cookbook, for the page (app.py) and the API (api.py) alike.

    rag, arag       RAG: one retrieval, then one answer with numbered citations (search.engine). `arag` is the same
                    on asyncio, for the API: the search's embedding and the answer await the network instead of
                    holding a thread while they wait
    rag_parts       a RAG answer as both show it: each citation a link to the cited lines on GitHub, the sources
                    listed by file, lines and section, and what a chat keeps
    agent_parts     agentic RAG: the agent (agent.py: chapter 2's loop, chapter 5's context) on one key's own client,
                    so runs at once on different keys stay apart. `stop` ends a run before its next model call
    doing           what is really happening, in words, while an answer is worked out

The model is gpt-oss-120b on Groq, on whichever key a question comes with. The search embeds every question with
Gemini -- the index was embedded with it -- on the owner's key unless another is given (`embed`: an API caller's).
The cookbook's text is read from data/corpus (ask_the_cookbook.refresh writes it).
"""

import json
import os
import re
import threading

import openai

from . import agent as S
from . import search as A

COOKBOOK = "https://github.com/Venkatdatta2402/agentic-forge-cookbook"
BUILT = json.loads((A.INDEX_DIR / "cookbook.json").read_text(encoding="utf-8"))

MAX_QUESTION = 500          # characters: a question, not a document to summarise on someone else's key

RAG, AGENTIC = "RAG", "Agentic RAG"

INDEX = A.open_index()


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


def embedding(key):
    """The search's embedding model on a Gemini key other than the owner's: an API caller's."""
    return A.embed_model(key)


# --------------------------------------------------------------------------- citations and sources

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


# --------------------------------------------------------------------------- RAG

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


async def arag(question, key, stage, embed=None):
    """`rag` on asyncio: the question's embedding (Gemini) and the answer (Groq) are awaited, so while they are on
    their way the server serves other requests. `embed`: the search's embedding model, if not the owner's."""
    from llama_index.core import QueryBundle
    llm = chat_model(key)
    engine = A.engine(INDEX, llm, embed=embed)
    stage["now"] = "Searching the cookbook"
    nodes = await engine.aretrieve(QueryBundle(question))
    if not nodes:
        return {"refused": True, "answer": A.NOT_COVERED, "response": None, "tokens": 0}
    stage["now"] = f"Reading {len(nodes)} passages and writing the answer"
    response = await engine.asynthesize(QueryBundle(question), nodes)
    budget = getattr(llm, "budget", None)               # a stand-in model in the tests has none
    return {"refused": not response.source_nodes, "answer": str(response).strip(), "response": response,
            "tokens": budget.spent if budget is not None else None}


def rag_parts(result):
    """A RAG answer, for the page and the API alike: `answer`, with each citation a link to the cited lines;
    `sources`, numbered as the answer cites them; and `kept`, the answer as a chat keeps it -- each citation turned
    into "file, lines a-b", which mean something later and are lines a later agentic turn can open."""
    nodes = dict(enumerate((s.node for s in result["response"].source_nodes), 1))
    named = {n: source(node) for n, node in nodes.items()}
    # passages named and linked alike -- two cut from one notebook section -- are one source in the list
    listed, renumber = [], {}
    for n, entry in named.items():
        if entry not in listed:
            listed.append(entry)
        renumber[n] = listed.index(entry) + 1
    answer = without_own_sources(result["answer"])
    return {"answer": linked(answer, {n: url for n, (_, url) in named.items()}, renumber),
            "sources": [{"n": m, "name": name, "url": url} for m, (name, url) in enumerate(listed, 1)],
            "kept": cited(answer, {n: where(node) for n, node in nodes.items()})}


def ask_engine(question, key, session, stage=None):
    """(what the page shows, what the chat keeps). Retrieval runs on the question as asked -- with the passage, when
    the visitor asked about one."""
    result = rag(question, key, stage if stage is not None else {})
    if result["refused"]:
        text = f"{A.NOT_COVERED} Nothing in it is close enough to the question to answer from."
        # a follow-up turned away is most often one that leaned on the chat: retrieval saw only its own words
        return (f"{text}\n\n{FOLLOW_UP}" if session.history else text), text
    parts = rag_parts(result)
    sources = "\n".join(f"{s['n']}. [{s['name']}]({s['url']})" for s in parts["sources"])
    return f"{parts['answer']}\n\n**Sources**\n\n{sources}", parts["kept"]


# --------------------------------------------------------------------------- agentic RAG

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


class Cancelled(Exception):
    """Whoever asked has gone (an API caller disconnected): the run stops before its next model call."""


def agent_parts(question, key, session, stage=None, embed=None, stop=None):
    """An agentic run, starting from the chat so far: `answer` with its file citations as links, `kept` for the
    chat, the tool `calls`, the `tokens` spent, and whether it `finished` with an answer of its own.

    The run is the cookbook's loop, synchronous -- the caller runs it on a thread. A thread cannot be killed, so
    `stop` (a threading.Event) is checked where every model call goes out, on the run's own client: once it is set,
    the next call raises Cancelled instead, and the run ends. At most the call already in flight is spent."""
    own = client(key)
    send, refused = own.chat.completions.create, []

    # agent.run reports a refused request as why the run stopped, in words; the error itself is kept here, so a bad
    # key or a spent limit can be told apart. The client is this run's alone, so this touches no other run
    def create(*args, **kwargs):
        if stop is not None and stop.is_set():
            raise Cancelled("the caller left")
        try:
            return send(*args, **kwargs)
        except openai.APIStatusError as error:
            refused.append(error)
            raise

    own.chat.completions.create = create
    stage = stage if stage is not None else {}
    result = S.run(INDEX, question, A.Budget(label="agent"), client=own, history=session.history, notes=session.notes,
                   watch=lambda conversation: stage.update(conversation=conversation), embed=embed)
    log_run(result)
    answer = result["answer"]
    parts = {"calls": result["calls"], "tokens": result["usage"]["total"]}
    if not answer:
        if (result["stopped"] or "").startswith("BudgetExceeded"):
            text = "The agent used its token budget before it had an answer. Try a narrower question."
            return {**parts, "answer": text, "kept": text, "finished": False}
        if refused:
            raise refused[-1]
        raise RuntimeError(f"the agent stopped without an answer ({result['stopped']})")
    return {**parts, "answer": link_files(answer), "kept": answer, "finished": True}


def ask_agent(question, key, session, stage=None):
    """(what the page shows, what the chat keeps). The run starts from the chat so far."""
    parts = agent_parts(question, key, session, stage)
    if not parts["finished"]:
        return parts["answer"], parts["kept"]
    steps = "\n".join(f"- `{c['tool']}` {str(c['args'])[:150]}" for c in parts["calls"])
    return (f"{parts['answer']}\n\n<details><summary>How it got there: {len(parts['calls'])} tool calls, "
            f"{parts['tokens']:,} tokens on your key</summary>\n\n{steps}\n\n</details>"), parts["kept"]


# --------------------------------------------------------------------------- what is happening

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
