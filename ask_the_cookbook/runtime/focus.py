"""Giving each component in a loop its own context, without the runtime knowing.

Chapter 2 hands every component the same `conversation` and lets each one decide what to do
with it. `Think` replays all of it in the API's native tool-call shape; `Observe` flattens all
of it to text and appends a raw tool result; `ReflectionThink` and `SelectThink` each call
`render()` over all of it again. Four different context decisions, each written inside a
`run()`, none of them visible from the loop where the components are declared.

That is the thing this file moves. A component still gets a conversation; it just gets a
NARROWED one:

    Focus(Observe(), kinds=("request", "observation"), budget=120)

`Focus` runs selection (and anything else the caller asks for) over the real conversation,
rebuilds the survivors into a conversation-shaped `View`, and calls the component with that.
The component is unchanged and unaware. `Runtime` is unchanged and unaware -- it still sees
something with `.run(conversation)`.

Two things make this safe rather than clever:

  - `View` overrides exactly one attribute, `messages`, and reads everything else from the real
    conversation. `pending`, `step_outputs`, `branches`, budgets and the rest are machinery, not
    context. (In the cookbook a View also passed writes through, for components that set
    `conversation.pending`; here only `Think` is focused, and it writes nothing.)
  - the component's return value is passed straight back, so whatever it produces is appended
    to the REAL conversation by `Runtime`, exactly as before.

What a narrowed view cannot do is narrow what a component takes from somewhere other than
`messages`. `Observe` reads the raw tool result from `conversation.pending`; trimming that is
`invoke(max_tokens=)` in 03_tools or `truncate()` in this chapter, not this file's business.
"""

from .context import available
from .selection import select


class View:
    """A conversation with different `messages` and the same everything else."""

    def __init__(self, conversation, messages):
        object.__setattr__(self, "conversation", conversation)
        object.__setattr__(self, "messages", list(messages))

    def __getattr__(self, name):
        # only reached for attributes View does not have: the machinery
        return getattr(self.conversation, name)


def to_messages(items):
    """Items back into the runtime's own message shapes.

    Not the API's shapes: a component expects a `conversation`, and will do its own
    translation -- `Think._build_messages()` rebuilds native tool calls from exactly the
    `decision` + `tool` pairing produced here, which is the structure chapter 2 found models
    recognise as work already done.

    An `observation` item carries the call it came from in `about` ("get_metrics {...}"), so
    the pair can be rebuilt. Items that were never messages -- facts, memories, summaries,
    scratchpad state, raw results -- have no shape to go back to, and are gathered into one
    labelled user message instead of being dressed as turns somebody took.
    """
    messages, notes = [], []
    for item in items:
        if item.kind == "instruction":
            messages.append({"role": "system", "content": item.content})
        elif item.kind == "constraint":
            messages.append({"role": "system", "content": item.content})
        elif item.kind == "request":
            messages.append({"role": "user", "content": item.content})
        elif item.kind == "message":
            role, _, said = item.content.partition(": ")
            messages.append({"role": role if role in ("user", "assistant") else "user",
                             "content": said or item.content})
        elif item.kind == "observation" and item.about:
            tool, _, args = item.about.partition(" ")
            messages.append({"role": "decision", "type": "tool_call", "tool": tool,
                             "content": args or "{}", "applied": True})
            messages.append({"role": "tool", "content": item.content})
        elif item.kind == "decision":
            # the call in flight, with no observation after it -- exactly how it sat in the
            # conversation before `available()` turned it into an item
            tool, _, args = item.about.partition(" ")
            messages.append({"role": "decision", "type": "tool_call", "tool": tool,
                             "content": args or "{}", "applied": True})
        elif item.kind == "observation":
            messages.append({"role": "tool", "content": item.content})
        elif item.kind == "state" and item.about == "plan":
            # kind="goals" because that is the only plan a Think can act on; a plan of tool
            # calls belongs to Graph, which reads it from the real conversation anyway
            messages.append({"role": "plan", "kind": "goals", "content": item.content,
                             "steps": [{"goal": line} for line in item.content.splitlines()[1:]]})
        else:
            notes.append(f"- {item.content}")
    if notes:
        messages.append({"role": "user", "content":
                         "[What is known so far. Reference material, not a message from the "
                         "user.]\n" + "\n".join(notes)})
    return messages


class Focus:
    """One component, given a context chosen for it.

    `scratchpad` and `constraints` are what `available()` needs; everything else is passed to
    `select()`, so a policy reads as what the component is allowed to see:

        Focus(Think(tools=tools), budget=800)
        Focus(Observe(), kinds=("request", "observation"), budget=120)

    `prepare(items, conversation)` is for everything selection cannot say: a boundary like the
    cookbook's `current_pass`, a compression, a retrieval. It runs per call, which is the point -- what a
    component gets is decided when it runs, not when the loop is declared. With no `policy`,
    selection is skipped entirely and `prepare` sees everything `available()` found:

        Focus(Observe(), prepare=current_pass)
    """

    def __init__(self, component, scratchpad=None, constraints=(), prepare=None,
                 profile=None, catalogue=(), **policy):
        self.component = component
        self.scratchpad = scratchpad
        self.constraints = constraints
        self.profile = profile
        self.catalogue = catalogue
        self.prepare = prepare
        self.policy = policy
        self.last = None            # what this component was given, for inspection

    def items(self, conversation):
        items = available(conversation, self.scratchpad, self.constraints,
                          profile=self.profile, catalogue=self.catalogue)
        chosen = select(items, **self.policy).kept if self.policy else items
        return self.prepare(chosen, conversation) if self.prepare else chosen

    def run(self, conversation):
        chosen = self.items(conversation)
        self.last = chosen
        return self.component.run(View(conversation, to_messages(chosen)))
