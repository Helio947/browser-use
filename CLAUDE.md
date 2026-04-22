# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`browser-use` is a Python library that lets an LLM-driven agent control a real browser via Playwright. An `Agent` iterates in a loop: it reads a summarized DOM + screenshot from a `BrowserContext`, asks an LLM for the next action(s), and executes them through a `Controller` whose `Registry` maps function names to async callables. The library is published to PyPI (currently `0.1.40`) and is installed by end users; treat the public surface in `browser_use/__init__.py` as API.

Python requirement is `>=3.11,<4.0` (see `.python-version`, `pyproject.toml`). Playwright is a hard dependency and needs `playwright install` after `pip install`.

## Commands

Package management uses either `pip` or `uv`. Dev extras live under `[project.optional-dependencies].dev` in `pyproject.toml`.

```bash
# install editable with dev extras
pip install -e ".[dev]"
playwright install

# run the full test suite (pytest picks up pytest.ini automatically)
pytest

# run a single test file / test
pytest tests/test_service.py
pytest tests/test_service.py::test_name -s

# skip slow tests
pytest -m "not slow"

# lint/format (ruff is configured via pyproject.toml: line-length=130, single quotes, tab indent)
ruff check .
ruff format .

# pre-commit (installs ruff + whitespace/yaml/toml hooks)
pre-commit install
pre-commit run --all-files

# try the library end-to-end
python examples/simple.py
```

Tests rely on environment variables from a local `.env` (see `.env.example`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `AZURE_OPENAI_KEY`/`AZURE_OPENAI_ENDPOINT` (used by `tests/conftest.py`), `BROWSER_USE_LOGGING_LEVEL` (`result` | `info` | `debug`), `ANONYMIZED_TELEMETRY`. `tests/conftest.py` instantiates a real `ChatOpenAI`/`AzureChatOpenAI` — tests that require an LLM will fail without keys.

There are no CI test workflows; `.github/workflows/publish.yml` only publishes to PyPI on GitHub release. `codebeaver.yml` runs `pytest` with dummy keys in Codebeaver.

## Architecture

### The agent loop (`browser_use/agent/service.py`)

`Agent.run()` drives repeated `step()` calls until an action returns `ActionResult(is_done=True)` or `max_steps`/`max_failures` is hit. Each step:

1. `BrowserContext.get_state()` → `DomService` injects `browser_use/dom/buildDomTree.js` into the page, which returns a hash map of DOM nodes annotated with `highlightIndex`, visibility, interactivity, etc. Python rebuilds a `DOMElementNode` tree plus a `SelectorMap` (index → element).
2. `MessageManager` appends the state + prior results to the history, enforcing a token budget (`max_input_tokens`, default 128k; approximate, chars/token heuristic).
3. LLM is called via LangChain `BaseChatModel`. `ToolCallingMethod` (`function_calling` | `json_mode` | `raw` | `auto`) is chosen from the model class; the agent falls back to parsing JSON from raw content if tool calling is not supported.
4. The response is validated as `AgentOutput`, which is **dynamically extended** with the current action union — see `AgentOutput.type_with_custom_actions` and `Registry.create_action_model`. This means the schema that the LLM sees is rebuilt from whatever actions are registered (including user-defined ones) and passed as the structured-output schema.
5. Each action in `AgentOutput.action` is executed through `Controller.act()` → `Registry.execute_action()`.

Optional side-features layered onto the loop: a planner LLM (`PlannerPrompt`, runs every `planner_interval` steps), a second-pass validator (`validate_output`), conversation logging (`save_conversation_path`), and GIF generation from screenshots (`agent/gif.py`, triggered by `generate_gif`).

### Action registry (`browser_use/controller/`)

`Controller.__init__` registers the default actions (search_google, go_to_url, click_element, input_text, scroll, extract_content, done, etc.) using `@self.registry.action(description, param_model=...)`. The same decorator is how users add custom actions:

```python
controller = Controller()

@controller.action('Description the LLM sees', param_model=MyArgs)
async def my_action(params: MyArgs, browser: BrowserContext):
    ...
    return ActionResult(extracted_content=..., include_in_memory=True)
```

`Registry._create_param_model` auto-derives a Pydantic param model from the function signature when `param_model` isn't supplied. **Reserved parameter names injected by the registry at call time** (skipped when building the param model): `browser` (the `BrowserContext`), `page_extraction_llm` (LLM passed down from `Agent.settings.page_extraction_llm`), `available_file_paths`. Adding another parameter name to this injection list requires editing both `Registry._create_param_model` and `Registry.execute_action`.

`Controller(output_model=MyModel)` swaps the default `done` action for one that emits a user-defined structured output (`success` is appended internally and stripped from the serialized result).

`exclude_actions=[...]` on `Controller` prevents registration entirely — if you add a new default action, double-check it respects this.

### Browser / BrowserContext (`browser_use/browser/`)

`Browser` wraps Playwright's browser launch — `BrowserConfig` supports headless/security flags, a local Chrome binary (`chrome_instance_path`), or remote connections (`wss_url`, `cdp_url`). `BrowserContext` is the per-session object the agent actually drives: it owns tabs, cookies, viewport, screenshots, element highlighting, and exposes helpers like `get_current_page`, `get_state`, `click_element_node`, `input_text_element_node`. `BrowserContextConfig` tunes load-timing heuristics (`minimum_wait_page_load_time`, `wait_for_network_idle_page_load_time`, `maximum_wait_page_load_time`, `wait_between_actions`) that directly affect how stable action sequences are.

`context.py` is a ~1000-line file — resist splitting it unless a change genuinely warrants it; the surrounding docs (`browser_use/README.md`) call out a ~500-line threshold before splitting services.

### DOM extraction (`browser_use/dom/`)

Almost all heavy lifting happens in `buildDomTree.js`, which runs in the browser. `DomService` in Python just loads that file via `importlib.resources`, passes args (`doHighlightElements`, `focusHighlightIndex`, `viewportExpansion`, `debugMode`), and reconstructs typed nodes. When changing element selection / highlighting behavior, edit the JS file; Python-side changes usually only affect parsing. `HistoryTreeProcessor` (`dom/history_tree_processor/service.py`) maps interacted elements into serializable `DOMHistoryElement`s for replay.

### Message manager (`browser_use/agent/message_manager/`)

Owns `MessageManagerState.history`, a list of messages with attached `MessageMetadata` tracking approximate token counts. `utils.convert_input_messages` converts LangChain messages to the format the model expects, including stripping images when `use_vision=False` and reshaping messages for models that need tool-call syntax. `save_conversation` dumps the exchange for debugging.

### Module conventions (`browser_use/README.md`)

Each subpackage follows the Netflix Dispatch layout: `service.py` (logic), `views.py` (Pydantic models, split into "All / Requests / Responses" sections), `prompts.py` where applicable, `utils.py`. Split a `service.py` into `_subservice/` only when it exceeds ~500 lines. Keep `views.py` as the single home for types used across a module.

### Public API

`browser_use/__init__.py` re-exports: `Agent`, `Browser`, `BrowserConfig`, `Controller`, `DomService`, `SystemPrompt`, `ActionResult`, `ActionModel`, `AgentHistoryList`, `BrowserContextConfig`. Preserve these names; downstream users import them directly.

### Telemetry

`ProductTelemetry` (`browser_use/telemetry/service.py`) posts anonymized events to PostHog. Disabled when `ANONYMIZED_TELEMETRY=false`. Events: agent run/step/end, registered controller functions. New top-level features should emit matching events in `telemetry/views.py` only when it's useful for product metrics — don't add events casually.

### Logging (`browser_use/logging_config.py`)

A custom `RESULT` log level (35) sits above `INFO`. `setup_logging()` reads `BROWSER_USE_LOGGING_LEVEL` and silences noisy third-party loggers (`httpx`, `playwright`, `langchain`, `openai`, `PIL`, etc.). Both `conftest.py` files call it on import so logs render consistently during tests.

## Style / conventions

- Ruff config in `pyproject.toml` is authoritative: line length 130, **tab indentation**, **single quotes**, `select = ["E", "F", "I"]`, `fix = true`. Pre-commit runs the same rules. Keep existing tab indentation when editing; don't convert to spaces.
- Async-first: almost all framework code is `async def`. Sync action functions registered via `@registry.action` are auto-wrapped with `asyncio.to_thread` — prefer async where possible to avoid the thread hop.
- Pydantic v2 everywhere (`>=2.10.4`). Use `model_config = ConfigDict(...)` rather than the legacy `class Config`. `AgentOutput` uses `arbitrary_types_allowed=True` so LangChain types can be fields.
- LangChain provider choice is explicit at the call site; core code depends only on `langchain_core.language_models.chat_models.BaseChatModel`. Adding provider-specific behavior (e.g. Anthropic vs OpenAI tool calling) goes in `agent/service.py::_set_tool_calling_method` and `message_manager/utils.py::convert_input_messages`, not in the core loop.
- System prompt content lives in `browser_use/agent/system_prompt.md` (loaded as a package resource). Don't inline-edit in Python; edit the markdown and keep `{max_actions}` as the only template variable.
