"""Agent Development Kit tools on Datacircle's open API, https://api.datacircle.dev/v1: no key, no account."""

from __future__ import annotations

import asyncio
import urllib.parse
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.genai import types

from adk_datacircle import _api

if TYPE_CHECKING:
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.tools.tool_context import ToolContext

SORTS = ("score", "provider", "price")


def _number(value: Any, name: str, low: int, high: Optional[int]) -> Optional[int]:
    """A model's limit or offset as a whole number in its range, or None when left out."""
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise _api.DatacircleError(f"{name} must be a whole number, not {value!r}") from None
    if number != value and str(number) != str(value).strip():
        raise _api.DatacircleError(f"{name} must be a whole number, not {value!r}")
    if number < low or (high is not None and number > high):
        raise _api.DatacircleError(f"{name} must be {low} or more" + (f" and {high} or less" if high is not None else ""))
    return number


def _words(value: Any) -> Optional[str]:
    """A model's text argument, or None when left out."""
    return None if value is None else str(value)


def _answer(make: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """What the API gave, or {"error": ...} with its words for the model to read."""
    try:
        return make()
    except _api.DatacircleError as error:
        return {"error": str(error)}


def _best(plans: Any) -> Optional[Dict[str, Any]]:
    """A provider's best plan on an API: its lowest rank, a plan with no score having none."""
    ranked = sorted(plans, key=lambda plan: plan.get("rank") if isinstance(plan.get("rank"), (int, float)) else float("inf"))
    return _api.brief(ranked[0], ["plan", "score", "perThousand", "page"]) if ranked else None


class Datacircle:
    """Datacircle's benchmark of GTM APIs (https://datacircle.dev): every API it lists, who is best for each, every plan
    and each provider's ranks, from its open API. Each method makes one HTTPS GET and gives the answer as a dict, or
    {"error": ...} with the API's words.

    Args:
        base_url: The API's address. Default https://api.datacircle.dev/v1.
        timeout: Seconds each HTTP request may take. Default 30.
    """

    def __init__(self, base_url: str = _api.BASE_URL, timeout: float = 30):
        self.base_url = base_url
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return _api.get(path, params, base_url=self.base_url, timeout=self.timeout)

    def list_apis(self, query: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Lists the GTM APIs on Datacircle's sheet, each with its slug, how many plans and providers it has and its
        best plan: the ones whose name holds the most of query's words first. Gives total and up to limit (20) APIs."""

        def make() -> Dict[str, Any]:
            most = _number(limit, "limit", 1, 200) or 20
            apis = self._get("/apis")
            words = [word for word in str(query or "").lower().replace("-", " ").split() if word]
            held = []
            for index, api in enumerate(apis):
                name = f"{api.get('api', '')} {str(api.get('slug', '')).replace('-', ' ')}".lower()
                matched = sum(word in name for word in words)
                if matched or not words:
                    held.append((-matched, index, api))
            found = [_api.brief(api, _api.API_KEYS) for _, _, api in sorted(held, key=lambda h: h[:2])]
            return {"total": len(found), "apis": found[:most]}

        return _answer(make)

    def best_plans(self, slug: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """Gives an API's plans best first by their score out of 100, each with its rank, provider, plan, $ per 1,000
        calls and its page: slug is the API's slug or name, limit 1 to 50 (10)."""

        def make() -> Dict[str, Any]:
            made = _api.slug(slug)
            if not made:
                raise _api.DatacircleError("Give an API's slug, like 'email-finder': datacircle_list_apis gives every slug")
            answer = self._get(f"/apis/{urllib.parse.quote(made)}/best", {"limit": _number(limit, "limit", 1, 50)})
            return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}

        return _answer(make)

    def search_plans(
        self,
        api: Optional[str] = None,
        query: Optional[str] = None,
        sort: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Searches every plan, or one API's, by provider or plan name, sorted by score (the default), provider or
        price. Gives total, offset and up to limit (10) plans."""

        def make() -> Dict[str, Any]:
            order = str(sort).strip().lower() if sort not in (None, "") else None
            if order is not None and order not in SORTS:
                raise _api.DatacircleError(f"sort must be score, provider or price, not {sort!r}")
            params = {"api": _words(api), "query": _words(query), "sort": order,
                      "limit": _number(limit, "limit", 1, 50), "offset": _number(offset, "offset", 0, None)}
            answer = self._get("/plans", params)
            return {**answer, "plans": [_api.brief(plan) for plan in answer.get("plans") or []]}

        return _answer(make)

    def provider(self, slug: Optional[str] = None) -> Dict[str, Any]:
        """Gives each API one provider sells and its rank on each, with its best plan there: slug is the provider's
        slug or name."""

        def make() -> Dict[str, Any]:
            made = _api.slug(slug)
            if not made:
                raise _api.DatacircleError("Give a provider's slug or name, like 'bright-data' or 'Bright Data'")
            answer = self._get(f"/providers/{urllib.parse.quote(made)}")
            apis = [{**_api.brief(entry, ["api", "slug", "rank", "bestPage"]), "plans": len(entry.get("plans") or []),
                     "best": _best(entry.get("plans") or [])} for entry in answer.get("apis") or []]
            return {**answer, "apis": apis}

        return _answer(make)


def _text(description: str, **more: Any) -> types.Schema:
    return types.Schema(type=types.Type.STRING, description=description, **more)


def _whole(description: str, **more: Any) -> types.Schema:
    return types.Schema(type=types.Type.INTEGER, description=description, **more)


def _inputs(required: Sequence[str] = (), **properties: types.Schema) -> types.Schema:
    return types.Schema(type=types.Type.OBJECT, properties=properties, required=list(required) or None)


PER_THOUSAND = "perThousand is a plan's price of 1,000 items in dollars, pay the plan's whole price in its currency."
LIMIT = "How many plans to give, 1 to 50. Leave it out for 10."

# Each tool: its name, what the model reads of it, its inputs, and the method of Datacircle it calls.
TOOLS = [
    ("datacircle_list_apis",
     "Lists the GTM APIs on Datacircle's sheet (https://datacircle.dev), with each API's slug, how many plans and "
     "providers it has and its best plan. Call it first to find an API's slug. Gives total, the APIs found, and apis, "
     "up to limit of them.",
     _inputs(query=_text("Words of the API's name or slug, like 'email' or 'linkedin profile': the APIs that hold the "
                         "most of them come first. Leave it out for the sheet's order."),
             limit=_whole("How many APIs to give, 1 to 200. Leave it out for 20: the sheet has about 150.",
                          minimum=1, maximum=200)),
     "list_apis"),
    ("datacircle_best_plans",
     "Gives an API's plans best first by their score out of 100 on Datacircle's benchmark (https://datacircle.dev), "
     "each with its rank, provider, plan, $ per 1,000 calls and its page. " + PER_THOUSAND,
     _inputs(["slug"],
             slug=_text("The API's slug, like 'email-finder' or 'linkedin-profile', as datacircle_list_apis gives it."),
             limit=_whole(LIMIT, minimum=1, maximum=50)),
     "best_plans"),
    ("datacircle_search_plans",
     "Searches every plan on Datacircle's sheet (https://datacircle.dev), or one API's, by provider or plan name, "
     "sorted by score, provider or price. Gives total, offset and up to limit plans. " + PER_THOUSAND,
     _inputs(api=_text("Only this API's plans: its name or slug, like 'email-finder'. A provider's name goes in query, "
                       "never here. Leave it out for every API."),
             query=_text("Words that must all appear in the provider's or plan's name, like 'bright data'. Leave it "
                         "out for any."),
             sort=_text("score (best first), provider (A to Z) or price ($ per 1,000, lowest first). Leave it out for "
                        "score.", enum=list(SORTS)),
             limit=_whole(LIMIT, minimum=1, maximum=50),
             offset=_whole("How many found plans to skip first. Leave it out for 0.", minimum=0)),
     "search_plans"),
    ("datacircle_provider",
     "Gives each API one provider sells and its rank on each on Datacircle's benchmark (https://datacircle.dev), "
     "with its best plan there.",
     _inputs(["slug"],
             slug=_text("The provider's slug, like 'bright-data', or its name, like 'Bright Data': the name is made a "
                        "slug.")),
     "provider"),
]


class DatacircleTool(BaseTool):
    """One of Datacircle's four tools: one HTTPS GET to the open API, its answer a dict for the model, or
    {"error": ...} with the API's words or what an argument must be, so the model can call again."""

    def __init__(self, name: str, description: str, parameters: types.Schema, call: Callable[..., Dict[str, Any]]):
        super().__init__(name=name, description=description)
        self.parameters = parameters
        self.call = call

    def _get_declaration(self) -> types.FunctionDeclaration:
        return types.FunctionDeclaration(name=self.name, description=self.description,
                                         parameters=self.parameters.model_copy(deep=True))

    async def run_async(self, *, args: Dict[str, Any], tool_context: Optional[ToolContext] = None) -> Dict[str, Any]:
        # A null is an argument left out: some models send every input, null when they mean none.
        given = {key: value for key, value in (args or {}).items() if value is not None}
        unknown = [key for key in given if key not in (self.parameters.properties or {})]
        if unknown:
            return {"error": f"{self.name} takes {', '.join(self.parameters.properties or {})}, "
                             f"not {', '.join(unknown)}"}
        return await asyncio.to_thread(self.call, **given)


def datacircle_tools(base_url: str = _api.BASE_URL, timeout: float = 30) -> List[DatacircleTool]:
    """The four tools, for an agent's tools: datacircle_list_apis, datacircle_best_plans, datacircle_search_plans and
    datacircle_provider."""
    datacircle = Datacircle(base_url, timeout)
    return [DatacircleTool(name, description, parameters, getattr(datacircle, method))
            for name, description, parameters, method in TOOLS]


class DatacircleToolset(BaseToolset):
    """Datacircle's benchmark of GTM APIs (https://datacircle.dev) as an ADK toolset, for
    LlmAgent(tools=[DatacircleToolset()]): four tools, each one HTTPS GET to the open API giving its answer as a dict.

    Args:
        base_url: The API's address. Default https://api.datacircle.dev/v1.
        timeout: Seconds each HTTP request may take. Default 30.
        tool_filter: The names of the tools to give, or a predicate of ADK's. Default all four.
        tool_name_prefix: A prefix ADK puts before each tool's name (ADK 1.6 and later).
    """

    def __init__(
        self,
        *,
        base_url: str = _api.BASE_URL,
        timeout: float = 30,
        tool_filter: Optional[Union[Callable[..., bool], List[str]]] = None,
        tool_name_prefix: Optional[str] = None,
    ):
        given: Dict[str, Any] = {"tool_filter": tool_filter}
        if tool_name_prefix:
            given["tool_name_prefix"] = tool_name_prefix
        super().__init__(**given)
        self.tools = datacircle_tools(base_url, timeout)

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        chosen = self.tool_filter
        if not chosen:
            return list(self.tools)
        if isinstance(chosen, (list, tuple, set)):
            return [tool for tool in self.tools if tool.name in chosen]
        return [tool for tool in self.tools if chosen(tool, readonly_context)]

    async def close(self) -> None:
        """Nothing to release: each call is a request of its own. Abstract in ADK 1.10 and before, so defined here."""

    @classmethod
    def from_config(cls, config: Any, config_abs_path: str) -> "DatacircleToolset":
        """The toolset of an agent's YAML config: name adk_datacircle.DatacircleToolset, its args any of base_url,
        timeout, tool_filter and tool_name_prefix."""
        given = config.model_dump() if hasattr(config, "model_dump") else dict(config or {})
        unknown = sorted(set(given) - {"base_url", "timeout", "tool_filter", "tool_name_prefix"})
        if unknown:
            raise ValueError(f"DatacircleToolset takes base_url, timeout, tool_filter and tool_name_prefix, not {', '.join(unknown)}")
        return cls(**given)


_default = Datacircle()


def list_apis(query: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Every GTM API on Datacircle's sheet, the best match of query first, as a dict."""
    return _default.list_apis(query, limit)


def best_plans(slug: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """One API's plans, best first by score out of 100, as a dict."""
    return _default.best_plans(slug, limit)


def search_plans(api: Optional[str] = None, query: Optional[str] = None, sort: Optional[str] = None,
                 limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
    """Plans by provider or plan name, on every API or one, as a dict."""
    return _default.search_plans(api, query, sort, limit, offset)


def provider(slug: str) -> Dict[str, Any]:
    """The APIs a provider sells and its rank on each, as a dict."""
    return _default.provider(slug)
