"""A visitor's chat: what they see, what the model gets, and the two things done to it.

    shown         the whole transcript, as the page shows it. Never shortened
    history       the earlier turns the model gets: {"role": "user" | "assistant", "content"}
    notes         what a compaction kept of the turns it replaced: a summary, and the facts pulled out of them

    asking()      a question about a passage the visitor selected ("Ask about"): the passage, then the question.
                  It is the whole of what RAG retrieves with and answers, and it is pinned in agentic RAG's
                  context -- the one way a visitor can be sure the part they mean reaches the model
    compact()     chapter 5's compaction over the history, when the visitor asks for it: the last KEEP_RECENT
                  messages word for word, everything older replaced by notes
    held()        how many tokens the chat holds: everything that competes for a place in each model call

A RAG follow-up with no selection is answered as asked: retrieval runs on the question's words alone, so the page
asks the visitor to select the part they mean, or to name the chapter or idea. Agentic RAG has the earlier turns as
candidates for every call's context. Kept for the browser session only; nothing is stored.
"""

from dataclasses import dataclass, field

from . import agent
from .runtime.chat import count_tokens
from .runtime.compression import compact, compressible
from .runtime.context import Item

AGENT_CALL = agent.CONTEXT_TOKENS      # what one agent model call may carry, the chat included
KEEP_RECENT = 6                        # messages compaction leaves word for word: the last three exchanges
QUOTE_LIMIT = 1000                     # characters a selection may be: a few sentences, ~250 tokens


def asking(question, quote=""):
    """What the model gets for a question about a selected passage, and what the chat keeps of it."""
    quote = (quote or "").strip()
    return f'About this part of an earlier answer:\n"{quote}"\n\n{question}' if quote else question


def showing(question, quote=""):
    """The same, as the visitor's message on the page: the passage as a quote above the question."""
    quote = (quote or "").strip()
    return "\n".join(f"> {line}" for line in quote.splitlines()) + f"\n\n{question}" if quote else question


@dataclass
class Session:
    shown: list = field(default_factory=list)
    history: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def held(self):
        return sum(count_tokens(m["content"]) for m in self.history) + sum(count_tokens(n) for n in self.notes)

    def add(self, question, answer):
        """One exchange into what the model gets next time."""
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    def items(self):
        """The chat as chapter 5 items: notes first (they stand for the oldest turns), then each message."""
        notes = [Item("fact", note, 0, source="notes") for note in self.notes]
        turns = [Item("message", f"{m['role']}: {m['content']}", n) for n, m in enumerate(self.history, 1)]
        return notes + turns

    def can_compact(self):
        return bool(compressible(self.items(), keep_recent=KEEP_RECENT))

    def compact(self, client=None):
        """Replace everything older than the last KEEP_RECENT messages with a summary and facts. Returns the tokens
        held before and after. `client`: whose key the 2-3 model calls go on (None: chapter 1's shared one)."""
        before = self.held()
        result = compact(self.items(), keep_recent=KEEP_RECENT, using=client)
        self.history = [self.history[i.turn - 1] for i in result.items if i.kind == "message"]
        # the summary first: it tells the story the facts belong to
        self.notes = ([i.content for i in result.items if i.kind == "summary"]
                      + [i.content for i in result.items if i.kind in ("fact", "constraint")])
        return before, self.held()
