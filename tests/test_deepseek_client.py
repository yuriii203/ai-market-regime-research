from __future__ import annotations

import json

import requests

from src.clients.deepseek import (
    DeepSeekClient,
    DeepSeekResponseError,
    DeepSeekTransportError,
)


class FakeResponse:
    def __init__(self, body: object, status_code: int = 200) -> None:
        self.body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self) -> object:
        return self.body


class FakeSession:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.response = response
        self.error = error
        self.last_url = ""
        self.last_json: dict | None = None
        self.last_timeout = None

    def post(self, url: str, *, json: dict, timeout):
        self.last_url = url
        self.last_json = json
        self.last_timeout = timeout
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def test_client_requests_json_output_and_parses_response() -> None:
    content = {"state_label": "综合偏弱"}
    session = FakeSession(
        FakeResponse(
            {
                "id": "chat-123",
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(content, ensure_ascii=False)},
                    }
                ],
                "usage": {"total_tokens": 100},
            }
        )
    )
    client = DeepSeekClient("secret", session=session)

    result = client.create_json(system_prompt="输出json", user_prompt="解释")

    assert result.data == content
    assert result.request_id == "chat-123"
    assert session.last_url == "https://api.deepseek.com/chat/completions"
    assert session.last_json["response_format"] == {"type": "json_object"}
    assert session.last_json["model"] == "deepseek-flash"
    assert session.last_json["thinking"] == {"type": "disabled"}
    assert session.last_json["reasoning_effort"] == "none"
    assert session.headers["Authorization"] == "Bearer secret"


def test_client_maps_timeout_to_safe_error() -> None:
    client = DeepSeekClient(
        "secret", session=FakeSession(error=requests.Timeout("slow"))
    )

    try:
        client.create_json(system_prompt="json", user_prompt="test")
    except DeepSeekTransportError as exc:
        assert "超时" in str(exc)
    else:
        raise AssertionError("expected DeepSeekTransportError")


def test_client_rejects_truncated_or_non_stopped_output() -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": "{}"}}
                ]
            }
        )
    )
    client = DeepSeekClient("secret", session=session)

    try:
        client.create_json(system_prompt="json", user_prompt="test")
    except DeepSeekResponseError as exc:
        assert "未正常结束" in str(exc)
    else:
        raise AssertionError("expected DeepSeekResponseError")
