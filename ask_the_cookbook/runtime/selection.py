"""Choosing which of the information you already have goes into one operation's context.

Selection works on what is already in hand -- `context.available()`'s list:

    select()      [Item] in, [Item] out. Local, cheap, and it can only ever drop things.

(The cookbook also has retrieval from a chapter 4 memory store, run beside selection. This app keeps no
memories between questions, so it is not here.)

`select()` runs in a fixed order, and the order is the design:

    1. supersede  items with the same `about` are versions of one thing; keep the newest.
                  An identity question, not a scoring one -- an old plan is not "less
                  relevant" than the new one, it is wrong.
    2. pin        pinned items go in, always. If they alone overflow the budget, selection
                  refuses: it can drop things, but it cannot make a constraint smaller.
                  That is compression's job.
    3. score      what is left is ranked by relevance to the request, importance of its kind,
                  and recency -- each min-max normalised across the candidates, as in chapter
                  4's `Recall`, so a weight is a share of the decision.
    4. fill       best first, until the budget is spent.

The result keeps the original order. Selection decides WHAT goes in; where each thing sits in
the model's input is construction's decision, and doing it here would decide it by accident.
"""

import re
from dataclasses import dataclass, field

from .context import route, tokens
from .recall import unit

# Importance is a property of what KIND of thing an item is, fixed in advance. Not rated by a
# model per item: chapter 4 measured that, and a model asked "how important is this, 1-10"
# answers 6-8 for nearly everything -- a call per item that moves no ranking.
#
# Pinned kinds are listed for completeness; they never reach scoring.
IMPORTANCE = {
    "instruction": 1.0,
    "request": 1.0,
    "constraint": 1.0,
    "state": 0.8,        # where the task stands -- what the next step starts from
    "decision": 0.8,     # the call in flight: what the next observation is about
    "fact": 0.7,
    "summary": 0.7,      # stands in for a whole stretch of the run that is no longer there
    "observation": 0.6,  # a sentence per tool call, already phrased for reading
    "memory": 0.6,
    "message": 0.4,
    "tool_result": 0.3,  # exact but large; the observation usually says what matters in it
}

STOPWORDS = frozenset(
    "a an and are as at be but by can did do does for from had has have how i if in is it its "
    "me my of on or so that the them then there this to was we were what when where which who "
    "why will with you your about into just should would could".split()
)


def words(text):
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower())
            if len(w) > 2 and w not in STOPWORDS}


def overlap(query, text):
    """Share of the query's words that appear in the text. The cheapest relevance there is.

    Knows nothing about meaning: "slow" and "latency" score zero against each other. Any
    callable `(query, text) -> float` can replace it -- see `select(relevance=)`.
    """
    q = words(query)
    return len(q & words(text)) / len(q) if q else 0.0


@dataclass
class Selection:
    kept: list
    dropped: list = field(default_factory=list)   # [(item, why)]

    @property
    def tokens(self):
        return tokens(self.kept)


def supersede(items):
    """Newest version of every `about`, and everything with no `about` at all.

    Returns (current, [(old_item, why)]).
    """
    newest = {}
    for item in items:
        if item.about:
            # the same `about` can hold an observation AND a raw result; they are versions
            # only of their own kind, never of each other
            newest[(item.kind, item.about)] = item.turn
    current, dropped = [], []
    for item in items:
        latest = newest.get((item.kind, item.about)) if item.about else None
        if latest is not None and item.turn < latest:
            dropped.append((item, f"superseded at t{latest}"))
        else:
            current.append(item)
    return current, dropped


def select(items, budget=None, query=None, kinds=None, relevance=overlap, keep_last=0,
           floors=None, w_relevance=0.5, w_importance=0.3, w_recency=0.2, half_life=8):
    """The items this operation should be given, within `budget` tokens.

    `query` is what relevance is measured against. Defaults to the request item's content,
    which is the right thing for a Think; an Observe or a summarizer may want something else.

    `kinds`, if given, is the only kinds this operation can see at all -- a filter applied
    before anything is scored. An operation checking a number wants `tool_result`; one
    answering the user usually does not.

    `half_life` is in turns: an item that many turns old scores half on recency.

    `floors` is the other guarantee: `{"catalogue": 2, "state": 1}` means at least that many
    items from that group survive, whatever they score, with the group's best-scoring ones
    chosen. A name is matched against the item's route (`context.route`) or its kind, so it can
    speak about either.

    It exists because of a measured failure. Once a skills catalogue became items, selection
    dropped **all of it** at a 700-token budget: the entries are unpinned, they scored below the
    incident's own findings, and the model was then asked what to do about a connection pool
    with `resize_pool` invisible to it. A dropped fact makes an answer worse; a dropped
    catalogue entry removes something the agent could do, and nothing in the answer says so.

    Like the pinned items, a floor that does not fit the budget is refused rather than quietly
    broken.

    `keep_last` is a GUARANTEE, and `w_recency` is not. Recency is one term in a score: an old
    item with strong relevance outranks a recent one, and the greedy fill can skip a recent item
    that does not fit while taking a smaller older one. Weighting recency higher does not fix
    that -- it only shifts the odds. `keep_last=N` takes the N most recent items (after
    superseding) and treats them exactly like pinned ones: in, whatever they score. If they do
    not fit the budget, selection refuses rather than quietly dropping one, for the same reason
    it refuses on pinned items.
    """
    current, dropped = supersede(items)

    if kinds is not None:
        dropped += [(i, f"kind {i.kind} not used here") for i in current
                    if i.kind not in kinds and not i.pinned]
        current = [i for i in current if i.kind in kinds or i.pinned]

    # the last `keep_last` items join the pinned ones as must-keeps, in their own order
    recent = sorted(current, key=lambda i: i.turn)[-keep_last:] if keep_last else []
    recent_ids = set(map(id, recent))
    pinned = [i for i in current if i.pinned or id(i) in recent_ids]
    candidates = [i for i in current if not (i.pinned or id(i) in recent_ids)]

    if budget is not None and tokens(pinned) > budget:
        what = "pinned items" if not keep_last else f"pinned items and the last {keep_last}"
        raise ValueError(
            f"{what} alone are {tokens(pinned)} tokens, over the budget of {budget}. "
            "Selection can leave things out; it cannot make what must go in any smaller. "
            "Raise the budget or compress them."
        )

    if query is None:
        query = next((i.content for i in reversed(items) if i.kind == "request"), "")

    chosen = set(map(id, pinned))
    if candidates:
        now = max(i.turn for i in items)
        rel = unit([relevance(query, i.content) for i in candidates])
        imp = unit([IMPORTANCE[i.kind] for i in candidates])
        rec = unit([0.5 ** ((now - i.turn) / half_life) for i in candidates])
        scored = sorted(
            ((w_relevance * r + w_importance * m + w_recency * c, n, item)
             for n, (item, r, m, c) in enumerate(zip(candidates, rel, imp, rec))),
            key=lambda s: (-s[0], s[1]),
        )
        # floors, chosen AFTER scoring so each group keeps its best rather than its first
        promised = []
        for name, least in (floors or {}).items():
            group = [item for _, _, item in scored
                     if name in (route(item), item.kind) and id(item) not in chosen]
            for item in group[:least]:
                chosen.add(id(item))
                promised.append(item)
        if budget is not None and tokens(pinned) + tokens(promised) > budget:
            raise ValueError(
                f"pinned items and the floors {floors} come to "
                f"{tokens(pinned) + tokens(promised)} tokens, over the budget of {budget}. A "
                "floor is a promise, so this is refused rather than half-kept. Raise the budget, "
                "lower a floor, or compress what they hold."
            )

        spent = tokens(pinned) + tokens(promised)
        for score, _, item in scored:
            if id(item) in chosen:
                continue
            if budget is None or spent + item.tokens <= budget:
                chosen.add(id(item))
                spent += item.tokens
            else:
                dropped.append((item, f"over budget (score {score:.2f})"))

    kept = [i for i in current if id(i) in chosen]
    return Selection(kept, sorted(dropped, key=lambda d: d[0].turn))
