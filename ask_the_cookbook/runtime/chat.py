import os
from pathlib import Path

os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).parent / ".tiktoken_cache"))

import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(_encoding.encode(text))


class Conversation:
    """The messages an agent is carrying, plus the live things it needs to do its work.

    `messages` and `notes` are data -- they can be written to a file and read back. `resources`
    is the opposite: connections, sessions, database handles, things that are *open* and have
    to be closed. Keeping them here rather than in a closure is what makes them reachable,
    which is what makes them replaceable.

    Two chapters asked for this independently. `04_memory` notebook 5 found that nothing on a
    Conversation could hold a memory store, so `remember` had to close over one. `07_mcp`
    notebook 6 found the same thing for a live MCP session, where it is worse: a session whose
    server has died needs replacing, and a thing sealed inside a closure cannot be replaced by
    anyone.
    """

    def __init__(self, system=None):
        self.messages = []
        self.notes = []
        # name -> a live thing. The names are the caller's to choose ("mcp", "store"), because
        # nothing here needs to know what is in it.
        self.resources = {}
        if system:
            self.messages.append({"role": "system", "content": system})
