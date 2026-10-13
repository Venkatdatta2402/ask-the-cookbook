"""Chapter 6's two pieces the agent uses: `Think` with an output ceiling, and `verbatim`."""

from . import think


class Think(think.Think):
    """Chapter 2's Think with a ceiling on output that can be set.

    gpt-oss-20b stops at 2,048 output tokens unless told otherwise, and its hidden reasoning
    counts. On a long task it can spend the lot thinking and return an empty answer.
    """

    def __init__(self, *args, max_tokens=16000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens

    def _sampling_kwargs(self):
        return {**super()._sampling_kwargs(), "max_tokens": self.max_tokens}


def verbatim(effect):
    # What a tool returned goes into the conversation as it is. Observe's default is to have a
    # model restate it in one sentence, which is right for a sensor reading and wrong for another
    # agent's reply: the reply IS the work, and a one-sentence paraphrase of it throws the work away.
    return effect["content"]
