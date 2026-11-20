# adk-datacircle

Find the best GTM APIs. We benchmark all the GTM APIs and we put the results on our website: https://datacircle.dev

This package gives [Agent Development Kit](https://github.com/google/adk-python) (ADK) agents four tools on Datacircle's open API, https://api.datacircle.dev/v1. You need no key and no account.

```bash
pip install adk-datacircle
```

| Tool | What it gives | Call |
| --- | --- | --- |
| `datacircle_list_apis` | Every GTM API on Datacircle, with its slug, number of plans and providers, and its best plan | `GET /apis` |
| `datacircle_best_plans` | One API's plans, best first by score out of 100, each with its rank and price per 1,000 calls | `GET /apis/{slug}/best` |
| `datacircle_search_plans` | Plans by provider or plan name, on every API or one, sorted by score, provider or price | `GET /plans` |
| `datacircle_provider` | The APIs a provider sells and its rank on each | `GET /providers/{slug}` |

## Use

In your agent's `agent.py`:

```python
from google.adk.agents import LlmAgent
from adk_datacircle import DatacircleToolset

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="gtm_assistant",
    instruction="Answer from the Datacircle tools.",
    tools=[DatacircleToolset()],
)
```

Then `adk run`, `adk web` or `adk api_server` as with any agent, or a runner in your own code:

```python
import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types


async def main():
    runner = InMemoryRunner(agent=root_agent)
    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="me")
    question = types.Content(role="user", parts=[types.Part(text="Which email finder API is best for under $5 per 1,000?")])
    async for event in runner.run_async(user_id="me", session_id=session.id, new_message=question):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)


asyncio.run(main())
```

Any model ADK runs works, Gemini or another through `LiteLlm`. `DatacircleToolset(tool_filter=["datacircle_best_plans"])` gives only the tools named, and `datacircle_tools()` gives the four as a list.

In an agent's YAML config (ADK 1.11 and later):

```yaml
name: gtm_assistant
model: gemini-2.5-flash
instruction: Answer from the Datacircle tools.
tools:
  - name: adk_datacircle.DatacircleToolset
```

Each tool gives the model the API's answer as a dict. Each plan comes with its rank, provider, plan, score, price per 1,000 (`perThousand`, in dollars) and its page on datacircle.dev, which lists every field. An unknown slug or a value out of range comes back as `{"error": ...}` with the API's words (an unknown API's answer lists every slug), so the model can call again.

Each is also a plain function that gives a dict:

```python
from adk_datacircle import best_plans

best_plans(slug="email-finder", limit=3)
```

## Network

Each call makes one HTTPS GET to https://api.datacircle.dev/v1 with the tool's arguments and nothing else. The API takes up to 60 calls a minute from one address; the tools wait once, a minute at most, when it asks them to. `DatacircleToolset(timeout=10)` sets how long a call may take. Every plan is also one file at https://datacircle.dev/datacircle.json.

## Links

- Datacircle: https://datacircle.dev
- The API: https://datacircle.dev/api
- Its spec: https://api.datacircle.dev/v1/openapi.json
- What each key means: https://datacircle.dev/fields

MIT license, ADK 1.2.1 and later, Python 3.10 and later.
