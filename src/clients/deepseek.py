from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


class DeepSeekError(RuntimeError):
    """Base class for safe, user-facing DeepSeek failures."""


class DeepSeekConfigurationError(DeepSeekError):
    pass


class DeepSeekTransportError(DeepSeekError):
    pass


class DeepSeekResponseError(DeepSeekError):
    pass


@dataclass(frozen=True)
class DeepSeekJsonResponse:
    data: dict[str, Any]
    request_id: str
    model: str
    usage: dict[str, Any]


class DeepSeekClient:
    """Minimal Chat Completions client with JSON-output enforcement."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: tuple[float, float] = (5.0, 30.0),
        session: requests.Session | None = None,
    ) -> None:
        clean_key = api_key.strip()
        if not clean_key:
            raise DeepSeekConfigurationError("缺少 DEEPSEEK_API_KEY")
        clean_model = model.strip()
        if not clean_model:
            raise DeepSeekConfigurationError("DEEPSEEK_MODEL 不能为空")
        self.base_url = base_url.rstrip("/")
        self.model = clean_model
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {clean_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "market-regime-research/0.1",
            }
        )

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(
            os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        )

    def create_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2500,
    ) -> DeepSeekJsonResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
            "max_tokens": max_tokens,
            "stream": False,
        }
        endpoint = "/chat/completions"
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise DeepSeekTransportError("DeepSeek解释请求超时，请稍后重试") from exc
        except requests.RequestException as exc:
            raise DeepSeekTransportError(
                f"无法连接DeepSeek（{exc.__class__.__name__}）"
            ) from exc
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise DeepSeekTransportError(
                f"DeepSeek接口返回HTTP {response.status_code}"
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise DeepSeekResponseError("DeepSeek返回了非JSON响应") from exc
        if not isinstance(body, dict):
            raise DeepSeekResponseError("DeepSeek响应不是JSON对象")
        try:
            finish_reason = body["choices"][0]["finish_reason"]
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekResponseError("DeepSeek响应缺少有效choices内容") from exc
        if finish_reason != "stop":
            raise DeepSeekResponseError(
                f"DeepSeek输出未正常结束：finish_reason={finish_reason}"
            )
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekResponseError("DeepSeek返回了空解释")
        import json

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DeepSeekResponseError("DeepSeek解释不是合法JSON") from exc
        if not isinstance(parsed, dict):
            raise DeepSeekResponseError("DeepSeek解释JSON不是对象")
        usage = body.get("usage")
        return DeepSeekJsonResponse(
            data=parsed,
            request_id=str(body.get("id", "")),
            model=str(body.get("model", self.model)),
            usage=usage if isinstance(usage, dict) else {},
        )
