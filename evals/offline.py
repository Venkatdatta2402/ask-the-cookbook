"""The evals' one setup: no key and no network.

Placeholder keys where none are set, and the saved index opened with a stand-in embedding model. A question's own
embedding comes from data/query_embeddings.json instead -- made once on Gemini (RETRIEVAL_QUERY, the task the app's
search uses) and kept -- so a run spends nothing and gives the same numbers every time, in CI as on a laptop.
"""

import os

os.environ.setdefault("GROQ_API_KEY", "gsk_placeholder")
os.environ.setdefault("GEMINI_API_KEY", "placeholder")

from llama_index.core import MockEmbedding  # noqa: E402

from ask_the_cookbook import search  # noqa: E402

# Replaced only if nothing has replaced it already: the app's own tests put a stand-in there that records what it
# was made for, and swapping it here would hide those records from them
if search.embed_model.__module__ == search.__name__:
    search.embed_model = lambda key=None: MockEmbedding(embed_dim=search.EMBED_DIM)

from ask_the_cookbook import answers  # noqa: E402 -- opens the saved index, with the stand-in

INDEX = answers.INDEX
