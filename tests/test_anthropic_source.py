import json
from types import SimpleNamespace

import pytest
from anthropic import BadRequestError
from anthropic import _base_client as anthropic_base_client
from anthropic.types import Message, MessageDeltaUsage, TextBlock, Usage

from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.provider.entities import TokenUsage
from astrbot.core.provider.sources.anthropic_source import ProviderAnthropic

# The SDK bundles its own httpx (``httpx`` before 1.0.0, ``httpx2`` after).
sdk_httpx = getattr(
    anthropic_base_client, "httpx", getattr(anthropic_base_client, "httpx2", None)
)


def _provider() -> ProviderAnthropic:
    return ProviderAnthropic.__new__(ProviderAnthropic)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_headers", "expected_key"),
    [({}, "test-key"), ({"x-api-key": "override-key"}, "override-key")],
)
async def test_anthropic_request_sends_single_api_key_header(
    monkeypatch, custom_headers, expected_key
):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)

    def handle(request):
        assert request.headers.get_list("x-api-key") == [expected_key]
        return sdk_httpx.Response(200, json={})

    async with sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(handle)
    ) as http_client:
        monkeypatch.setattr(
            ProviderAnthropic,
            "_create_http_client",
            lambda self, config: http_client,
        )
        provider = ProviderAnthropic(
            provider_config={
                "key": ["test-key"],
                "api_base": "https://opencode.ai/zen/go",
                "custom_headers": custom_headers,
            },
            provider_settings={},
        )
        provider.client.max_retries = 0
        await provider.client.messages.with_raw_response.create(
            model="claude-test",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_anthropic_extract_usage_counts_cache_creation_input():
    provider = _provider()

    usage = provider._extract_usage(
        Usage(
            input_tokens=10,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
            output_tokens=20,
        )
    )

    # Anthropic's input_tokens excludes cache writes, so cache_creation
    # must be folded into input_other to keep total input accurate.
    assert usage.input_other == 60
    assert usage.input_cached == 100
    assert usage.input == 160
    assert usage.output == 20


def test_anthropic_extract_usage_without_cache_breakpoints():
    provider = _provider()

    usage = provider._extract_usage(Usage(input_tokens=30, output_tokens=10))

    assert usage.input_other == 30
    assert usage.input_cached == 0
    assert usage.input == 30
    assert usage.output == 10


def test_anthropic_extract_usage_none_returns_empty():
    provider = _provider()

    assert provider._extract_usage(None) == TokenUsage()


def test_anthropic_update_usage_counts_cache_creation_input():
    provider = _provider()
    token_usage = TokenUsage(input_other=5, input_cached=0, output=0)

    provider._update_usage(
        token_usage,
        MessageDeltaUsage(
            input_tokens=10,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
            output_tokens=20,
        ),
    )

    assert token_usage.input_other == 60
    assert token_usage.input_cached == 100
    assert token_usage.input == 160
    assert token_usage.output == 20


def test_anthropic_update_usage_omitted_fields_are_preserved():
    provider = _provider()
    token_usage = TokenUsage(input_other=5, input_cached=0, output=0)

    # message_delta usage only carries output tokens in practice.
    provider._update_usage(token_usage, MessageDeltaUsage(output_tokens=7))

    assert token_usage.input_other == 5
    assert token_usage.input_cached == 0
    assert token_usage.output == 7


@pytest.mark.asyncio
async def test_query_merges_custom_server_tools_with_registered_tools():
    provider = ProviderAnthropic(
        provider_config={
            "id": "anthropic-test",
            "type": "anthropic_chat_completion",
            "model": "claude-test",
            "key": ["test-key"],
            "custom_extra_body": {
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "custom_flag": True,
            },
        },
        provider_settings={},
    )

    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_1",
                content=[TextBlock(text="ok", type="text")],
                model="claude-test",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    provider.client = SimpleNamespace(messages=FakeMessages())

    tool_set = ToolSet(
        [
            FunctionTool(
                name="get_time",
                description="Get the current time.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    await provider._query({"model": "claude-test", "messages": []}, tool_set)

    # Server-side tools from the custom body must coexist with the
    # registered function tools instead of replacing them.
    assert [tool["name"] for tool in captured["tools"]] == ["get_time", "web_search"]
    assert "tools" not in captured["extra_body"]
    assert captured["extra_body"]["custom_flag"] is True


@pytest.mark.asyncio
async def test_query_custom_tools_without_registered_tools_pass_through():
    provider = ProviderAnthropic(
        provider_config={
            "id": "anthropic-test",
            "type": "anthropic_chat_completion",
            "model": "claude-test",
            "key": ["test-key"],
            "custom_extra_body": {
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            },
        },
        provider_settings={},
    )

    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_1",
                content=[TextBlock(text="ok", type="text")],
                model="claude-test",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    provider.client = SimpleNamespace(messages=FakeMessages())

    await provider._query({"model": "claude-test", "messages": []}, None)

    # No registered tools: the custom body keeps its previous override
    # semantics and goes out via extra_body untouched.
    assert "tools" not in captured
    assert captured["extra_body"]["tools"] == [
        {"type": "web_search_20250305", "name": "web_search"}
    ]


@pytest.mark.asyncio
async def test_query_custom_tool_overrides_registered_tool_with_same_name():
    provider = ProviderAnthropic(
        provider_config={
            "id": "anthropic-test",
            "type": "anthropic_chat_completion",
            "model": "claude-test",
            "key": ["test-key"],
            "custom_extra_body": {
                "tools": [
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
                ],
            },
        },
        provider_settings={},
    )

    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_1",
                content=[TextBlock(text="ok", type="text")],
                model="claude-test",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    provider.client = SimpleNamespace(messages=FakeMessages())

    tool_set = ToolSet(
        [
            FunctionTool(
                name="web_search",
                description="Client-side web search.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    await provider._query({"model": "claude-test", "messages": []}, tool_set)

    # A name collision must not duplicate the tool (the API rejects
    # duplicates); the custom body keeps its override authority.
    assert captured["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    ]


@pytest.mark.asyncio
async def test_query_null_custom_extra_body_does_not_break_request():
    provider = ProviderAnthropic(
        provider_config={
            "id": "anthropic-test",
            "type": "anthropic_chat_completion",
            "model": "claude-test",
            "key": ["test-key"],
            "custom_extra_body": None,
        },
        provider_settings={},
    )

    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_1",
                content=[TextBlock(text="ok", type="text")],
                model="claude-test",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    provider.client = SimpleNamespace(messages=FakeMessages())

    tool_set = ToolSet(
        [
            FunctionTool(
                name="get_time",
                description="Get the current time.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    await provider._query({"model": "claude-test", "messages": []}, tool_set)

    assert [tool["name"] for tool in captured["tools"]] == ["get_time"]
    assert captured["extra_body"] == {}


def _tool_use_stream(
    partial_json_chunks: list[str], start_input: dict | None = None
) -> bytes:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_time",
                "input": start_input or {},
            },
        },
        *(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": chunk},
            }
            for chunk in partial_json_chunks
        ),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "start_input", "expected_args"),
    [
        ([""], None, {}),
        ([], None, {}),
        (["", '{"tz": "UTC"}'], None, {"tz": "UTC"}),
        ([], {"tz": "UTC"}, {"tz": "UTC"}),
    ],
)
async def test_anthropic_stream_keeps_tool_call_with_empty_input(
    monkeypatch, chunks, start_input, expected_args
):
    body = _tool_use_stream(chunks, start_input)
    transport = sdk_httpx.MockTransport(
        lambda request: sdk_httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )
    )
    monkeypatch.setattr(
        ProviderAnthropic,
        "_create_http_client",
        lambda self, provider_config: sdk_httpx.AsyncClient(transport=transport),
    )
    provider = ProviderAnthropic(
        {
            "id": "test",
            "type": "anthropic_chat_completion",
            "key": ["sk-test"],
            "model": "claude-test",
            "api_base": "https://api.anthropic.test",
        },
        {},
    )
    tools = ToolSet(
        [
            FunctionTool(
                name="get_time",
                description="Return the current time.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )
    payloads = {
        "messages": [{"role": "user", "content": "what time is it?"}],
        "model": "claude-test",
    }

    try:
        responses = [r async for r in provider._query_stream(payloads, tools)]
    finally:
        await provider.terminate()

    tool_chunks = [r for r in responses if r.role == "tool"]
    assert len(tool_chunks) == 1
    final = responses[-1]
    assert not final.is_chunk
    for response in (tool_chunks[0], final):
        assert response.tools_call_name == ["get_time"]
        assert response.tools_call_args == [expected_args]
        assert response.tools_call_ids == ["toolu_01"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True], ids=["standard", "stream"])
@pytest.mark.parametrize(
    "custom_tool",
    [{}, {"name": ""}, {"name": None}, None, "invalid", 0, False],
    ids=["missing-name", "empty-name", "null-name", "null", "string", "zero", "false"],
)
async def test_malformed_custom_tools_reach_api_validation(
    monkeypatch, streaming, custom_tool
):
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return sdk_httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "Invalid tool"},
            },
        )

    monkeypatch.setattr(
        ProviderAnthropic,
        "_create_http_client",
        lambda self, config: sdk_httpx.AsyncClient(
            transport=sdk_httpx.MockTransport(handle)
        ),
    )
    custom_tools = [custom_tool, {"type": "unnamed"}]
    provider = ProviderAnthropic(
        {
            "id": "test",
            "type": "anthropic_chat_completion",
            "key": ["test-key"],
            "model": "claude-test",
            "api_base": "https://offline.invalid",
            "custom_extra_body": {"tools": custom_tools},
        },
        {},
    )
    tools = ToolSet(
        [
            FunctionTool(
                name="get_time",
                description="Offline tool schema.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )
    payload = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    try:
        for _ in range(2):
            with pytest.raises(BadRequestError):
                if streaming:
                    async for _ in provider._query_stream(
                        payload, tools, request_max_retries=1
                    ):
                        pass
                else:
                    await provider._query(payload, tools, request_max_retries=1)
    finally:
        await provider.terminate()

    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert captured[0]["tools"][0]["name"] == "get_time"
    assert captured[0]["tools"][1:] == custom_tools
    assert provider.provider_config["custom_extra_body"]["tools"] == custom_tools


def _capturing_provider(extra_config=None):
    provider = ProviderAnthropic(
        provider_config={
            "id": "anthropic-test",
            "type": "anthropic_chat_completion",
            "model": "claude-test",
            "key": ["test-key"],
            **(extra_config or {}),
        },
        provider_settings={},
    )
    captured: dict[str, object] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return Message(
                id="msg_1",
                content=[TextBlock(text="ok", type="text")],
                model="claude-test",
                role="assistant",
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    provider.client = SimpleNamespace(messages=FakeMessages())
    return provider, captured


@pytest.mark.asyncio
async def test_query_lifts_openai_system_messages_into_system_param():
    # Callers such as plugins hand `_query` an OpenAI-shaped payload directly;
    # the Messages API rejects a leading system-role message.
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [
                {"role": "system", "content": "You are Gemini."},
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Reply in JSON."}],
                },
                {"role": "user", "content": "hi"},
            ],
        },
        None,
    )

    assert [m["role"] for m in captured["messages"]] == ["user"]
    texts = [block["text"] for block in captured["system"]]
    assert texts == ["You are Gemini.\n\nReply in JSON."]


@pytest.mark.asyncio
async def test_query_appends_lifted_system_after_existing_system_blocks():
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "system": [{"type": "text", "text": "base"}],
            "messages": [
                {"role": "system", "content": "extra"},
                {"role": "user", "content": "hi"},
            ],
        },
        None,
    )

    assert [block["text"] for block in captured["system"]] == ["base", "extra"]


@pytest.mark.asyncio
async def test_query_converts_openai_tool_calls_and_results():
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what time is it"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_time", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ],
        },
        None,
    )

    assistant, result = captured["messages"][1], captured["messages"][2]
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["id"] == "call_1"
    assert result["role"] == "user"
    assert result["content"][0]["type"] == "tool_result"
    assert result["content"][0]["tool_use_id"] == "call_1"


@pytest.mark.asyncio
async def test_query_json_object_response_format_asks_for_json_only():
    # Anthropic has no JSON mode; the request must still say so instead of
    # silently losing the caller's format requirement.
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        },
        None,
    )

    assert "response_format" not in captured
    assert "format" not in (captured.get("output_config") or {})
    instruction = captured["system"][-1]["text"]
    assert "JSON object" in instruction


@pytest.mark.asyncio
async def test_query_json_object_instruction_follows_lifted_system_prompt():
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [
                {"role": "system", "content": "You are Gemini."},
                {"role": "user", "content": "hi"},
            ],
            "response_format": {"type": "json_object"},
        },
        None,
    )

    texts = [block["text"] for block in captured["system"]]
    assert texts[0] == "You are Gemini."
    assert "JSON object" in texts[-1]


@pytest.mark.asyncio
async def test_query_maps_json_schema_response_format_to_output_config():
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"effort": "low"},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string", "minLength": 1},
                            "score": {"type": "integer", "minimum": 0, "maximum": 5},
                            "tags": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"t": {"type": "string"}},
                                },
                            },
                        },
                        "required": ["decision"],
                    },
                },
            },
        },
        None,
    )

    assert "response_format" not in captured
    output_config = captured["output_config"]
    assert output_config["effort"] == "low"
    fmt = output_config["format"]
    assert fmt["type"] == "json_schema"
    schema = fmt["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["tags"]["items"]["additionalProperties"] is False
    assert "minLength" not in schema["properties"]["decision"]
    assert "minimum" not in schema["properties"]["score"]
    assert "maximum" not in schema["properties"]["score"]
    # Unsupported constraints are kept as hints instead of being lost.
    assert schema["properties"]["decision"]["description"] == "{minLength: 1}"
    assert schema["properties"]["score"]["description"] == "{minimum: 0, maximum: 5}"


@pytest.mark.asyncio
async def test_output_schema_keeps_descriptions_type_lists_and_small_min_items():
    provider, captured = _capturing_provider()

    await provider._query(
        {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "reflection",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": ["string", "null"]},
                            "confidence": {
                                "type": "number",
                                "description": "How sure.",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "handles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 3,
                            },
                            "anniversaries": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                            },
                        },
                        "required": [
                            "item_id",
                            "confidence",
                            "handles",
                            "anniversaries",
                        ],
                    },
                },
            },
        },
        None,
    )

    props = captured["output_config"]["format"]["schema"]["properties"]
    assert props["item_id"] == {"type": ["string", "null"]}
    assert props["confidence"]["description"] == "How sure.\n\n{minimum: 0, maximum: 1}"
    assert props["handles"]["minItems"] == 1
    assert props["handles"]["description"] == "{maxItems: 3}"
    assert "minItems" not in props["anniversaries"]
    assert props["anniversaries"]["description"] == "{minItems: 2}"


@pytest.mark.asyncio
async def test_query_leaves_anthropic_shaped_payload_untouched():
    provider, captured = _capturing_provider()
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": "again"},
    ]

    await provider._query(
        {
            "model": "claude-test",
            "system": [{"type": "text", "text": "base"}],
            "messages": [dict(m) for m in messages],
        },
        None,
    )

    assert captured["messages"] == messages
    assert [block["text"] for block in captured["system"]] == ["base"]
