# Checking the judge's marks

26 claims: every one the judge marked against an answer, and 20 it passed, chosen at random. For each claim: "right": true if every mark the judge gave it is right, false if any is wrong -- and then "note": which, and why. Write it into `evals/data/owner_review.json`.

Also worth a word in a note: a claim split oddly, or a reference point that is wrong.

## Q1#2 (passed)

*The README says extract() asks the API to enforce the schema first. If the API enforces it, why validate as well?*

**Claim:** That alone does not guarantee a correct answer.

> but that alone is not enough to guarantee a correct answer.

- against the key: **supported** (foundations/llm.py 90-94)
- on the question: **yes**
- against the passages: **other** (passage 1)
- the key's evidence: [01_foundations/llm.py 86-122](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/01_foundations/llm.py#L86-L122)

## Q1#7 (passed)

*The README says extract() asks the API to enforce the schema first. If the API enforces it, why validate as well?*

**Claim:** Validation is kept as a second line of defence to catch semantic errors, handle unsupported schema enforcement, and give feedback for retries.

> Therefore, validation is kept as a second line of defence to catch semantic errors, handle models that don’t support schema enforcement, and to give the model feedback for corrective retries.

- against the key: **supported** (foundations/llm.py 90-94)
- on the question: **yes**
- against the passages: **other** (passage 3)
- the key's evidence: [01_foundations/llm.py 86-122](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/01_foundations/llm.py#L86-L122)

## Q2#1 (flagged)

*What counts as a doom loop, and what does the runtime do when it detects one?*

**Claim:** A doom‑loop is any pattern that shows the agent is stuck repeating the same work without making progress.

> A **doom‑loop** is any pattern that shows the agent is stuck repeating the same work without making progress.

- against the key: **not in the key**
- on the question: **yes**
- against the passages: **none** -- Passages only list specific doom‑loop patterns (tool repeat, error repeat, oscillation, similar‑result) and do not give a generic definition of a doom‑loop as any pattern of repeated work without progress.
- the key's evidence: [02_agent_runtime/runtime.py 181-249](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/runtime.py#L181-L249)

## Q2#2 (passed)

*What counts as a doom loop, and what does the runtime do when it detects one?*

**Claim:** The runtime’s `_doom_loop` routine flags a loop when it sees one of the following situations.

> The runtime’s `_doom_loop` routine flags a loop when it sees one of the following situations [5]:

- against the key: **supported** (runtime.py 181-218)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [02_agent_runtime/runtime.py 181-249](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/runtime.py#L181-L249)
- the passages it cites: [[5] 02_agent_runtime/runtime.py 181-218](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/runtime.py#L181-L218)

## Q2#7 (passed)

*What counts as a doom loop, and what does the runtime do when it detects one?*

**Claim:** When any of these conditions is met, the outer `Runtime` does not simply abort.

> When any of these conditions is met, the outer `Runtime` does **not** simply abort.

- against the key: **supported** (runtime.py 223-226)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [02_agent_runtime/runtime.py 181-249](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/runtime.py#L181-L249)
- the passages it cites: [[1] 02_agent_runtime/README.md 94-98](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L94-L98)

## Q3#2 (passed)

*What is the difference between a tool that refuses and one that is broken, and which does the model get to see?*

**Claim:** The error message usually contains a reason or a suggestion (e.g., “No weather station for Vestby. Try a larger nearby city.”).

> The error message usually contains a reason or a suggestion (e.g., “No weather station for Vestby. Try a larger nearby city.”).

- against the key: **not in the key**
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [03_tools/invoke.py 50-72](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/invoke.py#L50-L72), [03_tools/invoke.py 128-181](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/invoke.py#L128-L181), [03_tools/README.md 77-84](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L77-L84)
- the passages it cites: [[2] 03_tools/notebooks/03_invocation.ipynb 59-83](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb), [[4] 03_tools/README.md 62-74](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L62-L74)

## Q3#7 (flagged)

*What is the difference between a tool that refuses and one that is broken, and which does the model get to see?*

**Claim:** The conversation only receives a generic notice that the tool could not be completed, while the real error is left for the developer to inspect.

> The conversation only receives a generic notice that the tool could not be completed, while the real error is left for the developer to inspect

- against the key: **supported** (point 3)
- on the question: **yes**
- against the passages: **other** (passage 1) -- Passage 1 states that broken results give a generic notice while the cited passages do not mention this.
- the key's evidence: [03_tools/invoke.py 50-72](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/invoke.py#L50-L72), [03_tools/invoke.py 128-181](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/invoke.py#L128-L181), [03_tools/README.md 77-84](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L77-L84)
- the passages it cites: [[2] 03_tools/notebooks/03_invocation.ipynb 59-83](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb), [[4] 03_tools/README.md 62-74](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L62-L74)

## Q4#5 (passed)

*Why is recall injected into the loop while remembering is a tool the agent calls?*

**Claim:** Injecting the recall step prevents the agent from needing to remember to ask for it and avoids the failure mode of a fluent answer that consulted nothing.

> By injecting the recall step, the agent never has to remember to ask for it, and the failure mode of “a fluent answer that consulted nothing” is avoided [Source 4].

- against the key: **supported** (04_memory/README.md L135-L136)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [04_memory/memory_tools.py 1-20](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/memory_tools.py#L1-L20), [04_memory/README.md 125-139](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/README.md?plain=1#L125-L139)
- the passages it cites: [[4] 04_memory/notebooks/05_memory_in_the_loop.ipynb 1-19](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/05_memory_in_the_loop.ipynb)

## Q4#7 (passed)

*Why is recall injected into the loop while remembering is a tool the agent calls?*

**Claim:** `remember` is implemented as a tool that the model calls when it thinks the content should outlive the session.

> Therefore `remember` is implemented as a tool that the model calls when it thinks the content should outlive the session.

- against the key: **supported** (04_memory/memory_tools.py L8-L9)
- on the question: **yes**
- against the passages: **other** (passage 5)
- the key's evidence: [04_memory/memory_tools.py 1-20](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/memory_tools.py#L1-L20), [04_memory/README.md 125-139](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/README.md?plain=1#L125-L139)

## Q5#10 (passed)

*What does compaction leave untouched, and why?*

**Claim:** `compact()` never alters pinned rules, task‑state information, the latest few turns, or external reference items.

> it never alters pinned rules, task‑state information, the latest few turns, or external reference items

- against the key: **supported** (05_context/compression.py 258-267)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [05_context/compression.py 258-282](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/compression.py#L258-L282)
- the passages it cites: [[1] 05_context/notebooks/03_context_compression.ipynb 93-114](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/03_context_compression.ipynb), [[2] 05_context/compression.py 258-331](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/compression.py#L258-L331), [[5] 05_context/notebooks/03_context_compression.ipynb 84-91](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/03_context_compression.ipynb)

## Q12#4 (passed)

*Why is TaskGroup safer than asyncio.gather when one of several concurrent calls fails, and how do you limit how many run at once?*

**Claim:** `Semaphore(n)` caps the number of tasks that may be *in progress* at any moment; additional tasks wait until a slot is released.

> A `Semaphore(n)` caps the number of tasks that may be *in progress* at any moment; additional tasks wait until a slot is released.

- against the key: **supported** (point 3)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [01_foundations/notebooks/08_asyncio.ipynb 37-47](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/01_foundations/notebooks/08_asyncio.ipynb)
- the passages it cites: [[2] 01_foundations/notebooks/08_asyncio.ipynb 43-47](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/01_foundations/notebooks/08_asyncio.ipynb)

## Q14#3 (passed)

*What is the difference between ReWOO and LLM Compiler?*

**Claim:** ReWOO uses a solver – a plain Think that receives only the evidence gathered by the graph and must produce the final answer

> ReWOO uses a solver – a plain `Think` that receives only the evidence gathered by the graph and must produce the final answer

- against the key: **supported** (point 2)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [02_agent_runtime/README.md 80-81](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L80-L81), [02_agent_runtime/notebooks/06_agent_patterns.ipynb 87-104](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)
- the passages it cites: [[1] 02_agent_runtime/notebooks/06_agent_patterns.ipynb 85-112](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)

## Q14#6 (passed)

*What is the difference between ReWOO and LLM Compiler?*

**Claim:** ReWOO terminates by construction

> ReWOO **terminates by construction**

- against the key: **supported** (point 4)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [02_agent_runtime/README.md 80-81](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L80-L81), [02_agent_runtime/notebooks/06_agent_patterns.ipynb 87-104](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)
- the passages it cites: [[1] 02_agent_runtime/notebooks/06_agent_patterns.ipynb 85-112](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)

## Q14#15 (flagged)

*What is the difference between ReWOO and LLM Compiler?*

**Claim:** LLM Compiler can observe the results

> the LLM Compiler can observe the results

- against the key: **supported** (point 2)
- on the question: **yes**
- against the passages: **none** -- No passage states that LLM Compiler can observe the results
- the key's evidence: [02_agent_runtime/README.md 80-81](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L80-L81), [02_agent_runtime/notebooks/06_agent_patterns.ipynb 87-104](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)
- the passages it cites: [[2] 02_agent_runtime/README.md 77-86](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L77-L86), [[4] 02_agent_runtime/notebooks/04_planning.ipynb 211-250](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/04_planning.ipynb)

## Q14#16 (passed)

*What is the difference between ReWOO and LLM Compiler?*

**Claim:** LLM Compiler can decide to continue

> decide to continue

- against the key: **supported** (point 2)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [02_agent_runtime/README.md 80-81](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L80-L81), [02_agent_runtime/notebooks/06_agent_patterns.ipynb 87-104](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/06_agent_patterns.ipynb)
- the passages it cites: [[2] 02_agent_runtime/README.md 77-86](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/README.md?plain=1#L77-L86), [[4] 02_agent_runtime/notebooks/04_planning.ipynb 211-250](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/02_agent_runtime/notebooks/04_planning.ipynb)

## Q17#3 (passed)

*Why does chapter 3 add a per-call timeout when the runtime already has max_seconds, and when is retrying after a timeout unsafe?*

**Claim:** Wrapping each tool call in invoke(timeout=…) lets the system regain control as soon as the timeout expires, even if work continues elsewhere

> By wrapping each tool call in `invoke(timeout=…)` the system can regain control as soon as the timeout expires, even though the underlying work may still be running in its own thread or process

- against the key: **supported** (03_tools/notebooks/03_invocation.ipynb 43)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [03_tools/README.md 86-89](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L86-L89), [03_tools/notebooks/03_invocation.ipynb 39-43](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb), [03_tools/notebooks/03_invocation.ipynb 57-57](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb), [03_tools/notebooks/03_invocation.ipynb 99-105](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb)
- the passages it cites: [[1] 03_tools/README.md 86-89](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/README.md?plain=1#L86-L89), [[2] 03_tools/notebooks/03_invocation.ipynb 26-57](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/03_tools/notebooks/03_invocation.ipynb)

## Q19#5 (passed)

*Why is a 50/50 weighting of similarity and recency not really 50/50, and how is it fixed?*

**Claim:** Activation moves across nearly 0.8

> Activation moves across nearly 0.8

- against the key: **supported** (Reference point 1)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [04_memory/README.md 86-91](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/README.md?plain=1#L86-L91), [04_memory/notebooks/03_operations.ipynb 211-217](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/03_operations.ipynb)
- the passages it cites: [[2] 04_memory/notebooks/03_operations.ipynb 205-217](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/03_operations.ipynb)

## Q21#2 (flagged)

*Why does the add / update / ignore write rule work for facts but destroy episodic memory?*

**Claim:** When a fact changes, the duplicate check sees the same proposition and the update action replaces the old entry with the new one, keeping the store consistent and preventing duplicate storage of the fact.

> When a fact changes (e.g., “the checkout pool is 200 connections” → “the checkout pool is 210 connections”), the duplicate check sees the same proposition and the **update** action replaces the old entry with the new one, keeping the store consistent and preventing the same fact from being stored many times — exactly what we want for facts

- against the key: **supported** (04_memory/notebooks/04_types_of_memory.ipynb L32-33)
- on the question: **yes**
- against the passages: **none** -- Citation 2 does not mention the duplicate check seeing the same proposition nor that the update action prevents duplicate storage; other passages do not fully cover this claim.
- the key's evidence: [04_memory/README.md 104-112](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/README.md?plain=1#L104-L112), [04_memory/notebooks/04_types_of_memory.ipynb 13-13](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/04_types_of_memory.ipynb), [04_memory/notebooks/04_types_of_memory.ipynb 32-36](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/04_types_of_memory.ipynb), [04_memory/notebooks/04_types_of_memory.ipynb 198-202](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/notebooks/04_types_of_memory.ipynb)
- the passages it cites: [[2] 04_memory/README.md 103-107](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/04_memory/README.md?plain=1#L103-L107)

## Q22#2 (passed)

*How much did separating instructions from data reduce prompt injection, and did labels help?*

**Claim:** Mixed instructions and data resulted in 11 out of 24 injections (≈46%).

> When instructions and data were mixed in a single message the injection was obeyed 11 times out of 24 (≈46 %).

- against the key: **supported** (point 1)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [05_context/README.md 102-106](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/README.md?plain=1#L102-L106), [05_context/notebooks/04_context_construction.ipynb 85-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/04_context_construction.ipynb), [05_context/construction.py 24-28](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/construction.py#L24-L28)
- the passages it cites: [[5] 05_context/README.md 79-114](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/README.md?plain=1#L79-L114)

## Q23#1 (passed)

*Why does selection supersede old versions before scoring, and what does keep_last guarantee?*

**Claim:** Selection runs supersede step before relevance scoring because older items are outright wrong

> Selection first runs the supersede step (the “identity” step) before any relevance scoring because older items are not merely less useful – they are outright wrong.

- against the key: **supported** (point 1)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [05_context/README.md 88-93](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/README.md?plain=1#L88-L93), [05_context/notebooks/02_context_selection.ipynb 66-74](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/02_context_selection.ipynb), [05_context/notebooks/02_context_selection.ipynb 120-128](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/02_context_selection.ipynb)
- the passages it cites: [[2] 05_context/notebooks/02_context_selection.ipynb 64-74](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/02_context_selection.ipynb), [[3] 05_context/notebooks/02_context_selection.ipynb 178-189](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/05_context/notebooks/02_context_selection.ipynb)

## Q27#5 (passed)

*How does the repo keep an agent working when an MCP server dies mid-run?*

**Claim:** On each iteration it probes the server (using the cheap `tools/list` call, since the protocol’s `ping` always fails) and, if the connection is broken, attempts a limited number of repairs.

> On each iteration it probes the server (using the cheap `tools/list` call, since the protocol’s `ping` always fails) and, if the connection is broken, attempts a limited number of repairs.

- against the key: **supported** (point 4)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [07_mcp/README.md 117-133](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/README.md?plain=1#L117-L133), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 64-68](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 94-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)
- the passages it cites: [[1] 07_mcp/README.md 117-133](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/README.md?plain=1#L117-L133), [[2] 07_mcp/notebooks/06_build_an_mcp_agent.ipynb 70-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)

## Q27#8 (passed)

*How does the repo keep an agent working when an MCP server dies mid-run?*

**Claim:** When the server actually dies, the transport raises an exception that is caught as a “broken” result.

> When the server actually dies, the transport raises an exception that is caught as a “broken” result.

- against the key: **supported** (point 3)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [07_mcp/README.md 117-133](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/README.md?plain=1#L117-L133), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 64-68](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 94-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)
- the passages it cites: [[4] 07_mcp/notebooks/06_build_an_mcp_agent.ipynb 58-68](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)

## Q27#11 (flagged)

*How does the repo keep an agent working when an MCP server dies mid-run?*

**Claim:** The mutable `Connection` resource, the health‑check‑via‑`tools/list`, the bounded repair budget, and the silent‑repair policy keep the agent running even if the MCP server crashes in the middle of a run.

> Together, the mutable `Connection` resource, the health‑check‑via‑`tools/list`, the bounded repair budget, and the silent‑repair policy keep the agent running even if the MCP server crashes in the middle of a run.

- against the key: **supported** (point 1, point 4, point 5, point 6)
- on the question: **yes**
- against the passages: **none** -- no single passage states the combined claim of mutable Connection, tools/list health check, bounded repair budget, and silent‑repair policy together
- the key's evidence: [07_mcp/README.md 117-133](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/README.md?plain=1#L117-L133), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 64-68](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb), [07_mcp/notebooks/06_build_an_mcp_agent.ipynb 94-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)
- the passages it cites: [[1] 07_mcp/README.md 117-133](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/README.md?plain=1#L117-L133), [[2] 07_mcp/notebooks/06_build_an_mcp_agent.ipynb 70-94](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb), [[4] 07_mcp/notebooks/06_build_an_mcp_agent.ipynb 58-68](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb), [[5] 07_mcp/notebooks/06_build_an_mcp_agent.ipynb 46-56](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/07_mcp/notebooks/06_build_an_mcp_agent.ipynb)

## Q28#3 (passed)

*Why is CrewAI pinned to an old version in this repo?*

**Claim:** CrewAI releases ≥0.201 force chromadb~=1.1

> Every CrewAI release ≥ 0.201 forces `chromadb~=1.1`

- against the key: **supported** (Reference point 2)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [08_frameworks/README.md 188-192](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L188-L192)
- the passages it cites: [[1] 08_frameworks/README.md 173-195](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L173-L195), [[2] 08_frameworks/03_crewai/security_questionnaire.ipynb 413-432](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/03_crewai/security_questionnaire.ipynb)

## Q28#6 (passed)

*Why is CrewAI pinned to an old version in this repo?*

**Claim:** Downgrading mcp to 1.28 breaks the 07_mcp chapter

> (which breaks the 07_mcp chapter)

- against the key: **supported** (Reference point 3)
- on the question: **yes**
- against the passages: **cited**
- the key's evidence: [08_frameworks/README.md 188-192](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L188-L192)
- the passages it cites: [[1] 08_frameworks/README.md 173-195](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L173-L195), [[2] 08_frameworks/03_crewai/security_questionnaire.ipynb 413-432](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/03_crewai/security_questionnaire.ipynb)

## Q28#10 (flagged)

*Why is CrewAI pinned to an old version in this repo?*

**Claim:** Pinning to 0.193.2 keeps other chapters functional

> and keeps the other chapters functional

- against the key: **supported** (Reference point 4)
- on the question: **yes**
- against the passages: **none** -- No passage states that pinning to 0.193.2 keeps other chapters functional.
- the key's evidence: [08_frameworks/README.md 188-192](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L188-L192)
- the passages it cites: [[1] 08_frameworks/README.md 173-195](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/README.md?plain=1#L173-L195), [[2] 08_frameworks/03_crewai/security_questionnaire.ipynb 413-432](https://github.com/Venkatdatta2402/agentic-forge-cookbook/blob/43c7bf5e3ac6f8db886a81790b45e9f71febbf04/08_frameworks/03_crewai/security_questionnaire.ipynb)
