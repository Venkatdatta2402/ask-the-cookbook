"""The agent's parts, from agentic-forge-cookbook at fb39ada, cut down to what the app can reach.

    chapter 1   llm (the model client), chat (conversations, token counts)
    chapter 2   think, act, observe, runtime: the agent loop
    chapter 3   tools, registry
    chapter 4   recall: only `unit`, the normalisation chapter 5's selection uses
    chapter 5   context, selection, focus, compression, system: what each model call is sent
    chapter 6   supervisor: `Think` with an output ceiling, and `verbatim`

Removed: whatever nothing in the app or the index refresh can reach -- the other reasoning strategies (planner,
tree search, reflection), Loop and Graph, the supervisor's agents and chat, chapter 4's ranking, the chat-style
helpers. Comments that mention those keep their cookbook meaning: they say why the code that remains is shaped as
it is. Imports are relative, and think.Think can be given its own model client (`client=`), so agent runs on
different keys can go at once; given none, it uses llm.client, as in the cookbook.
"""
