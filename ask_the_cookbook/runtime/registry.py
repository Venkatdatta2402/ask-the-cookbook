import copy
import re

from .tools import Tool

# What the API accepts as a tool name: letters, digits, underscore, hyphen, up to 64 characters.
# Checked when a tool is registered rather than when a request is sent, because the API rejects
# the WHOLE request over one bad name -- so a single mistyped tool takes every other tool with it,
# and the error arrives with no clue about which agent or which turn caused it.
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Registry:
    """The tools an agent can use, and the three views of them the runtime needs.

    Notebook 1 ended by building those three views by hand, in three comprehensions over one
    list. That works until the list gets long enough to fall out of sync with itself -- which is
    the same problem notebook 1 opened with, one level up.

    A note on the names. `tools` here means the JSON the model reads, and `schemas` means the
    Pydantic models that check arguments. That is backwards from how it sounds, and it is
    deliberate: those are the names chapter 2's `Think(tools=...)` and `Act(schemas=...)` already
    use, so the wiring reads straight at the call site. The actual `Tool` objects are reached by
    name -- `registry["get_weather"]` -- or by iterating the registry.
    """

    def __init__(self, tools=None):
        self._tools = {}
        self._tags = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool, name=None, prefix=None, tags=None):
        # A plain function is wrapped on the way in, so registering never needs the @tool
        # decorator to have been used at the definition site -- which matters for functions
        # coming from somewhere you don't control.
        if not isinstance(tool, Tool):
            tool = Tool(tool)

        name = name or tool.name
        if prefix:
            name = f"{prefix}_{name}"
        if not NAME.match(name):
            raise ValueError(
                f"'{name}' cannot be used as a tool name: only letters, digits, underscores and "
                "hyphens are allowed, up to 64 characters."
            )
        if name in self._tools:
            # A dict would take the second one silently. Two tools called `search` from two
            # different places is an ordinary thing to happen, and the quiet version of it means
            # every `search` call goes to whichever was registered last -- a tool that answers
            # confidently with the wrong thing, which is worse than one that fails.
            raise ValueError(
                f"a tool named '{name}' is already registered. Pass prefix= or name= to give "
                "this one a different name."
            )

        if name != tool.name:
            # A copy, because the name is part of what the model is shown: the schema is built
            # from tool.name, and it has to match the key the call comes back under. Renaming the
            # original would change it for everyone else holding a reference to it.
            tool = copy.copy(tool)
            tool.name = name

        self._tools[name] = tool
        self._tags[name] = set(tags or ())
        return tool

    @property
    def tools(self):
        """The JSON the model reads -- for chapter 2's `Think(tools=...)`."""
        return [tool.schema for tool in self._tools.values()]

    @property
    def functions(self):
        """Name -> the plain function -- for chapter 2's `Act(functions=...)`."""
        return {name: tool.function for name, tool in self._tools.items()}

    @property
    def schemas(self):
        """Name -> the Pydantic model that checks arguments -- for `Act(schemas=...)`."""
        return {name: tool.args for name, tool in self._tools.items()}

    @property
    def names(self):
        return list(self._tools)
