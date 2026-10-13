"""Ask the Cookbook: question answering over agentic-forge-cookbook, with the files it used cited.

    corpus    which of the cookbook's files are read, and how (READMEs, notebook prose, modules)
    search    the index (Gemini embeddings), hybrid retrieval behind a floor, and the one-shot engine
    agent     the agent: the cookbook's own loop and context system, over the same search
    refresh   bring the index up to date with the cookbook's latest commit
    runtime   the agent's parts, from the cookbook's chapters 1-6
"""
