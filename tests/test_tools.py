"""The tools with no network: what each sends to the API, what a model reads of each, and how answers and refusals
reach it, by a tool's own call and through ADK's runner on a scripted model."""

import asyncio
import json
from typing import AsyncGenerator

import pytest
import requests
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from adk_datacircle import DatacircleToolset, _api, best_plans, datacircle_tools, list_apis, provider, search_plans
from adk_datacircle.tools import TOOLS

NAMES = ["datacircle_list_apis", "datacircle_best_plans", "datacircle_search_plans", "datacircle_provider"]


def tools(**options):
    return {tool.name: tool for tool in asyncio.run(DatacircleToolset(**options).get_tools())}


def call(name: str, **args):
    """The tool called as ADK calls it, with a model's arguments."""
    return asyncio.run(tools()[name].run_async(args=args, tool_context=None))


@pytest.fixture
def api(monkeypatch):
    """requests.get answering from a dict of path to (status, body), each request kept."""
    sent, answers = [], {}

    def get(url, params=None, headers=None, timeout=None, allow_redirects=None):
        sent.append((url, dict(params or {})))
        status, body = answers[url.split("/v1", 1)[1]]
        answer = requests.Response()
        answer.status_code, answer._content = status, json.dumps(body).encode()
        answer.headers["Retry-After"] = "1"
        return answer

    monkeypatch.setattr(_api.requests, "get", get)
    monkeypatch.setattr(_api.time, "sleep", lambda seconds: None)
    return sent, answers


def test_the_four_tools_as_a_model_reads_them():
    found = {name: tool._get_declaration().model_dump(mode="json", exclude_none=True) for name, tool in tools().items()}
    assert list(found) == NAMES
    assert all("https://datacircle.dev" in declaration["description"] for declaration in found.values())
    search = found["datacircle_search_plans"]["parameters"]
    assert list(search["properties"]) == ["api", "query", "sort", "limit", "offset"]
    assert "required" not in search
    assert search["properties"]["sort"]["enum"] == ["score", "provider", "price"]
    assert search["properties"]["limit"]["type"] == "INTEGER"
    assert found["datacircle_best_plans"]["parameters"]["required"] == ["slug"]
    assert found["datacircle_provider"]["parameters"]["required"] == ["slug"]
    assert all(prop.get("description") for declaration in found.values()
               for prop in declaration["parameters"]["properties"].values())
    assert [tool.name for tool in datacircle_tools()] == NAMES


def test_list_apis_puts_the_best_match_first(api):
    sent, answers = api
    answers["/apis"] = (200, [{"api": "Employee profile", "slug": "linkedin-profile", "plans": 9},
                              {"api": "Employee email", "slug": "email-finder", "plans": 5, "extra": "cut"},
                              {"api": "Email verification", "slug": "email-verification", "plans": 3}])
    assert list_apis(query="email finder", limit=1) == {"total": 2, "apis": [{"api": "Employee email", "slug": "email-finder", "plans": 5}]}
    assert call("datacircle_list_apis", query=None, limit=None)["total"] == 3
    assert call("datacircle_list_apis")["total"] == 3


def test_best_plans_makes_a_slug_and_keeps_the_keys_that_choose_a_plan(api):
    sent, answers = api
    answers["/apis/email-finder/best"] = (200, {"api": "Employee email", "total": 1, "plans": [
        {"rank": 1, "provider": "Emelia", "perThousand": 4.99, "aggregators": ["x"], "latencyMs": None}]})
    out = call("datacircle_best_plans", slug="Email Finder", limit="3")
    assert sent == [("https://api.datacircle.dev/v1/apis/email-finder/best", {"limit": 3})]
    assert out["plans"] == [{"rank": 1, "provider": "Emelia", "perThousand": 4.99}]
    assert call("datacircle_best_plans", slug="email-finder", limit=2.0)["plans"]


def test_search_plans_sends_only_what_is_given(api):
    sent, answers = api
    answers["/plans"] = (200, {"total": 1, "offset": 0, "plans": [{"provider": "Hunter", "plan": "Starter", "pitch": "cut"}]})
    out = call("datacircle_search_plans", api=None, query="hunter", sort="Price", limit=None, offset=0)
    assert sent == [("https://api.datacircle.dev/v1/plans", {"query": "hunter", "sort": "price", "offset": 0})]
    assert out == {"total": 1, "offset": 0, "plans": [{"provider": "Hunter", "plan": "Starter"}]}
    assert search_plans(query="hunter")["total"] == 1


def test_provider_gives_each_api_its_best_plan(api):
    sent, answers = api
    answers["/providers/bright-data"] = (200, {"provider": "Bright Data", "slug": "bright-data", "apis": [
        {"api": "Employee profile", "slug": "linkedin-profile", "rank": 3, "plans": [
            {"plan": "Scale", "rank": 9, "score": 50}, {"plan": "Pay as you go", "rank": 3, "score": 61, "pitch": "cut"}]}]})
    out = call("datacircle_provider", slug="Bright Data")
    assert sent[0][0] == "https://api.datacircle.dev/v1/providers/bright-data"
    assert out["apis"] == [{"api": "Employee profile", "slug": "linkedin-profile", "rank": 3, "plans": 2,
                            "best": {"plan": "Pay as you go", "score": 61}}]
    assert provider("bright-data")["provider"] == "Bright Data"


def test_the_model_reads_the_api_refusals_and_its_own_mistakes(api):
    sent, answers = api
    answers["/apis/nope/best"] = (404, {"error": "no API has the slug nope", "slugs": ["email-finder", "linkedin-profile"]})
    answers["/apis"] = (429, {"error": "slow down"})
    assert call("datacircle_best_plans", slug="nope")["error"] == ("Datacircle's API answered 404: no API has the slug "
                                                                  "nope; the slugs: email-finder, linkedin-profile")
    assert call("datacircle_list_apis")["error"] == "Datacircle's API answered 429 twice: wait a minute and call again"
    assert call("datacircle_best_plans")["error"].startswith("Give an API's slug")
    assert call("datacircle_best_plans", slug="email-finder", limit=2.5)["error"] == "limit must be a whole number, not 2.5"
    assert call("datacircle_best_plans", slug="email-finder", limit=99)["error"] == "limit must be 1 or more and 50 or less"
    assert call("datacircle_search_plans", sort="cheapest")["error"] == "sort must be score, provider or price, not 'cheapest'"
    assert call("datacircle_search_plans", provider="Hunter")["error"] == (
        "datacircle_search_plans takes api, query, sort, limit, offset, not provider")
    assert call("datacircle_provider", slug="  ")["error"].startswith("Give a provider's slug")
    assert len(sent) == 3


def test_an_unreachable_api_is_an_answer_too(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(_api.requests, "get", down)
    assert best_plans("email-finder") == {"error": "Datacircle's API could not be reached: no route"}
    assert call("datacircle_provider", slug="x")["error"] == "Datacircle's API could not be reached: no route"


def test_a_base_url_and_a_filter_of_its_own(api):
    sent, answers = api
    answers["/apis"] = (200, [])
    chosen = tools(base_url="http://127.0.0.1:9/v1/", timeout=5, tool_filter=["datacircle_list_apis"])
    assert list(chosen) == ["datacircle_list_apis"]
    asyncio.run(chosen["datacircle_list_apis"].run_async(args={}, tool_context=None))
    assert sent == [("http://127.0.0.1:9/v1/apis", {})]
    assert len(asyncio.run(DatacircleToolset(tool_filter=lambda tool, context: "plans" in tool.name).get_tools())) == 2


def test_the_toolset_of_a_yaml_config():
    made = DatacircleToolset.from_config(type("Args", (), {"model_dump": lambda self: {"tool_filter": ["datacircle_provider"]}})(), "x")
    assert [tool.name for tool in asyncio.run(made.get_tools())] == ["datacircle_provider"]
    with pytest.raises(ValueError, match="not key"):
        DatacircleToolset.from_config({"key": "x"}, "x")


class Scripted(BaseLlm):
    """A model that calls each tool once, two at a time at last, then answers with what they gave."""

    async def generate_content_async(self, llm_request, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        turns = [[("datacircle_list_apis", {"query": "email finder", "limit": 1})],
                 [("datacircle_best_plans", {"slug": "email-finder", "limit": 1})],
                 [("datacircle_search_plans", {"query": "hunter", "sort": "price", "limit": 1, "offset": None}),
                  ("datacircle_provider", {"slug": "Bright Data"})]]
        done = sum(1 for content in llm_request.contents if content.role == "model")
        offered = sorted(llm_request.tools_dict)
        if done < len(turns):
            parts = [types.Part(function_call=types.FunctionCall(name=name, args=args)) for name, args in turns[done]]
        else:
            given = [part.function_response.response for content in llm_request.contents for part in content.parts or []
                     if part.function_response]
            parts = [types.Part(text=json.dumps({"offered": offered, "given": given}))]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def test_an_agent_calls_each_tool_through_adk_runner(api):
    sent, answers = api
    answers["/apis"] = (200, [{"api": "Employee email", "slug": "email-finder", "plans": 5}])
    answers["/apis/email-finder/best"] = (200, {"api": "Employee email", "plans": [{"rank": 1, "provider": "Emelia"}]})
    answers["/plans"] = (200, {"total": 1, "offset": 0, "plans": [{"provider": "Hunter", "perThousand": 12}]})
    answers["/providers/bright-data"] = (200, {"provider": "Bright Data", "apis": []})
    agent = LlmAgent(model=Scripted(model="scripted"), name="gtm", instruction="Answer from the tools.",
                     tools=[DatacircleToolset()])

    async def run():
        runner = InMemoryRunner(agent=agent)
        session = await runner.session_service.create_session(app_name=runner.app_name, user_id="me")
        question = types.Content(role="user", parts=[types.Part(text="Who is best for email finding?")])
        return [event async for event in runner.run_async(user_id="me", session_id=session.id, new_message=question)]

    final = json.loads(asyncio.run(run())[-1].content.parts[0].text)
    assert final["offered"] == sorted(NAMES)
    assert [given.get("total", given.get("provider")) for given in final["given"]] == [1, None, 1, "Bright Data"]
    assert [url.split("/v1")[1] for url, _ in sent] == ["/apis", "/apis/email-finder/best", "/plans", "/providers/bright-data"]
    assert sent[2][1] == {"query": "hunter", "sort": "price", "limit": 1}
    assert len(TOOLS) == 4
