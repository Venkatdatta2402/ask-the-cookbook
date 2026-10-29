"""Ask the Cookbook, as a Gradio app: the Hugging Face Space.

    RAG           one retrieval, then one answer with citations (search.engine), on the owner's Groq key, until
                  Groq's own daily limit for it is reached (200k tokens: about 50 questions, all visitors together)
    your own key  a Groq key (gsk_...): questions on the visitor's own Groq account and its limits, and agentic
                  RAG -- retrieval as a tool the model calls as
                  often as it decides, in the agent loop (agent: chapter 2's loop, chapter 5's context). The same
                  model either way, gpt-oss-120b, the one the agent was tuned on. Groq only: Hugging Face's router
                  serves the model too, but bills from credit a free account no longer has (402 "no remaining
                  credits", measured), and a Groq key is free

It is a chat (session.py), kept for the browser session only. Select text in an answer and an "Ask about" button
appears: the passage goes into the question, which RAG retrieves with and agentic RAG has pinned in its context.
Without a selection, RAG answers the question as asked; agentic RAG has the earlier turns as candidates for each
call's context, and chapter 5's selection decides what fits. A meter shows how much the chat holds, and Compact
now replaces older turns with a summary (chapter 5's compaction).

Every question also costs one Gemini embedding (an agent question one per search) on the owner's key, whichever
key answers it: the index was embedded with Gemini, and a question has to be too.

The answering itself -- RAG, the agent, citations and sources -- is ask_the_cookbook/answers.py, shared with the
API (ask_the_cookbook/api.py). The same server serves both: the API's routes go into the FastAPI app Gradio makes.
"""

# First: on a ZeroGPU Space (the one Gradio hardware a free HF account gets), `spaces` must be imported before
# anything else. Anywhere else it does nothing.
import spaces  # noqa: I001

import concurrent.futures
import html
import os
import random
import time

import gradio as gr
import openai

from ask_the_cookbook import answers as ans
from ask_the_cookbook import api
from ask_the_cookbook import session as chat
from ask_the_cookbook.answers import AGENTIC, BUILT, COOKBOOK, MAX_QUESTION, RAG, doing
from ask_the_cookbook.session import Session


@spaces.GPU(duration=1)
def _zerogpu():
    """Never called. ZeroGPU refuses to start a Space with no @spaces.GPU function, and this app needs no GPU: its
    models run on Groq and Gemini."""


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
        # Groq's refusal names the limit: "... on tokens per day (TPD)", or per minute when visitors ask at once
        if "per day" in str(error):
            return ("The demo's Groq key has reached its daily limit. Paste your own key in the sidebar, or come back "
                    "tomorrow.")
        return "The demo's Groq key is busy (Groq's per-minute limit). Try again in a minute, or paste your own key."
    if isinstance(error, openai.APIStatusError) and status == 413:
        return "That request was too large for the model's limit. Try a shorter or narrower question."
    print(f"error: {type(error).__name__}: {str(error)[:300]}", flush=True)
    return "Something went wrong answering that. Please try again."


def status(session, note=""):
    """The line under the chat: a note when there is one, then how much the chat holds."""
    return ((f"{note}\n\n" if note else "")
            + f"Chat holds **{session.held():,} tokens**. The agent can take up to **{chat.AGENT_CALL:,}** per model "
              f"call.")


def whose(mine=False):
    """The sidebar's line on which key answers, beside the key box. It follows the box."""
    if mine:
        return ("Your key is in use: your questions go on your own Groq account, under its limits, and don't use "
                "the shared demo key.")
    return ("Without a key, RAG questions use a shared demo key, until Groq's daily limit for it is reached. "
            "[Get a free key](https://console.groq.com/keys).")


def own(key):
    return bool((key or "").strip())


def whose_key(key, mode=None):
    """(the key to use, own key?, a problem to show instead)."""
    key = (key or "").strip()
    if key:
        if not key.startswith("gsk_"):
            return None, True, "That isn't a Groq API key: those start `gsk_`."
        return key, True, None
    if mode == AGENTIC:
        return None, False, "Agentic RAG runs on your own key: paste a Groq API key in the sidebar."
    return os.environ["GROQ_API_KEY"], False, None


def respond(question, quote, key, mode, session):
    """One message, from the page: a question, and the passage it is about if the visitor pressed Ask about.
    Yields (the chat, the session, the status line, the question box, the quote box)."""
    question, quote = (question or "").strip(), (quote or "").strip()

    def turned_away(note, keep_question=True):
        return session.shown, session, status(session, note), question if keep_question else "", quote

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
    yield session.shown, session, status(session), "", ""
    # The answer is worked out in the background. Meanwhile the message carries the turning words (CSS, in the
    # browser) and a line on what is really happening -- re-sent only when that line changes. A re-sent message is
    # drawn afresh, so it carries how far into the words' cycle it is, and they go on where they were.
    job = WORKERS.submit(ans.ask_agent if mode == AGENTIC else ans.ask_engine, message, key, session, stage)
    while not concurrent.futures.wait([job], timeout=TICK).done:
        if doing(mode, stage) != said:
            said = doing(mode, stage)
            session.shown[-1] = {"role": "assistant",
                                 "content": working(mode, stage, time.monotonic() - started, order)}
            yield session.shown, session, status(session), "", ""
    try:
        shown, kept = job.result()
        session.add(message, kept)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        shown = trouble(error, own_key)
    session.shown[-1] = {"role": "assistant", "content": shown}
    yield session.shown, session, status(session), "", ""


WORKERS = concurrent.futures.ThreadPoolExecutor(max_workers=16)      # answers being worked out, behind the page
TICK = 0.5                                                           # seconds between looks at what is happening


def working(mode, stage, elapsed=0.0, order=None):
    """The waiting message: the turning words, in `order`, `elapsed` seconds into their cycle, and under them what is
    really happening. One paragraph with a line break: as two, a paragraph's spacing hung between them. Escaped: the
    step can carry the model's own search words."""
    return f'{cooking(elapsed, order)}<br><span class="doing">{html.escape(doing(mode, stage))}</span>'



def compact_now(key, session):
    """Compact now: older turns replaced by a summary and facts, by chapter 5's compaction. Yields (the chat, the
    session, the status line). On the demo key the calls go on the owner's key; on a visitor's key on theirs."""
    if not session.can_compact():
        note = (f"Not worth compacting yet: {session.compactable():,} tokens are older than the last "
                f"{chat.KEEP_RECENT // 2} exchanges (which are always kept as they are), and a summary is only "
                f"smaller once at least {chat.COMPACT_WORTH:,} are.")
        yield session.shown, session, status(session, note)
        return
    key, own_key, problem = whose_key(key)
    if problem:
        yield session.shown, session, status(session, problem)
        return
    yield session.shown, session, status(session, "*Compacting older turns...*")
    try:
        before, after = session.compact(ans.client(key) if own_key else None)
    except Exception as error:  # noqa: BLE001 -- shown to the visitor as words, not a trace
        yield session.shown, session, status(session, trouble(error, own_key))
        return
    if after >= before:
        note = "Compacting would not have made the chat any smaller, so it was left as it was."
        yield session.shown, session, status(session, note)
        return
    note = (f"*Older turns compacted: the chat now holds {after:,} tokens, down from {before:,}. The transcript above "
            "stays; the model gets a summary of the older part.*")
    session.shown.append({"role": "assistant", "content": note})
    yield session.shown, session, status(session)


def new_chat():
    fresh = Session()
    return fresh.shown, fresh, status(fresh)


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
/* the chat's own bin (no setting turns it off): it cleared the screen but not what the model gets; New chat does both */
#chat button[aria-label="Clear"] { display: none !important; }
/* the passage being asked about shows only once there is one */
#quote-box:has(textarea:placeholder-shown) { display: none !important; }
#quote-box { border-left: 3px solid var(--color-accent); }
/* the chat's state, quiet and on one line with the two things that act on it */
#chat-state { align-items: center; }
#meter, #meter p { font-size: var(--text-sm); color: var(--body-text-color-subdued); margin: 0; }
#whose-key, #whose-key p, #index, #index p { font-size: var(--text-sm); color: var(--body-text-color-subdued); }
#index { text-align: center; }
"""

with gr.Blocks(title="Ask the Cookbook") as demo:
    session = gr.State(Session())

    with gr.Sidebar(width=320):
        # kept short: at 1366x768, a common laptop screen, the sidebar must fit without a scrollbar
        gr.Markdown(f"## Ask the Cookbook\nQuestions about [agentic-forge-cookbook]({COOKBOOK}), answered from its "
                    "READMEs, notebooks and code, with sources.")
        mode = gr.Radio([RAG, AGENTIC], value=RAG, label="Mode",
                        info="RAG: one search, an answer in seconds. Agentic RAG: searches and reads files as often "
                             "as needed, for questions across chapters; a few minutes.")
        key = gr.Textbox(label="Your Groq API key", type="password", placeholder="gsk_...",
                         info="Needed for Agentic RAG. With it, all your questions go on your own Groq account, under "
                              "its limits. Never stored.")
        whose_line = gr.Markdown(elem_id="whose-key")

    # no corner buttons (share, copy all): only a copy button under each message
    conversation = gr.Chatbot(show_label=False, height="62vh", elem_id="chat", buttons=["copy"],
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
    gr.Markdown(f"Index: the cookbook at [`{BUILT['commit'][:7]}`]({COOKBOOK}/tree/{BUILT['commit']}), "
                f"{BUILT['files']} files. The chat lasts for this browser session; nothing is stored.",
                elem_id="index")
    ask_about = gr.Button("Ask about", elem_id="ask-about-trigger")

    # private: not offered to API clients, so the demo key is used through this page only
    question.submit(respond, [question, quote, key, mode, session], [conversation, session, meter, question, quote],
                    concurrency_limit=8, api_visibility="private")
    ask_about.click(None, None, quote, js="() => window.askAboutText || ''", api_visibility="private")
    conversation.example_select(example, None, question, api_visibility="private")
    compact.click(compact_now, [key, session], [conversation, session, meter], concurrency_limit=8,
                  api_visibility="private")
    clear.click(new_chat, None, [conversation, session, meter], api_visibility="private")
    key.change(lambda k: whose(own(k)), key, whose_line, api_visibility="private")
    demo.load(lambda s, k: (status(s), whose(own(k))), [session, key], [meter, whose_line], api_visibility="private")

demo.queue(max_size=32)

# The API's routes go into the FastAPI app Gradio makes for the page (FastAPI(routes=...), through launch's
# app_kwargs), so one server started by Gradio serves both: the one arrangement ZeroGPU ran, tried on a test Space --
# a FastAPI app of our own with Gradio mounted on it was shut down 2 s after it started. SSR off: its Node server in
# front answered /api/* itself (the page, or 405) instead of passing it on.
LAUNCH = {"js": ASK_ABOUT_JS, "css": CSS, "theme": gr.themes.Soft(), "ssr_mode": False,
          "app_kwargs": {"routes": api.router.routes}}

if __name__ == "__main__":
    demo.launch(**LAUNCH)
