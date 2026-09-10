import itertools
import json as _json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ._agent import VoiceAgent
from ._exceptions import ConfigurationError
from ._tool import Tool

# Two contexts are two sessions, so a test cannot pass by inheriting state from
# the one before it.
_SESSIONS = itertools.count(1)


@dataclass(frozen=True)
class RecordedCall:
    method: str
    url: str
    headers: dict
    json: Any
    params: Any


@dataclass(frozen=True)
class LogRecord:
    level: str
    event: str
    fields: dict


class StubbedResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json: Any = None,
        text: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._json = json
        if text is not None:
            self.text = text
        else:
            self.text = "" if json is None else _json.dumps(json)

    def json(self) -> Any:
        if self._json is None:
            raise ValueError(
                f"the stub for this response has no JSON body (status "
                f"{self.status_code}); pass json= to ctx.http.stub to give it one"
            )
        return self._json


class StubHttp:
    def __init__(self) -> None:
        self.calls: list = []
        self._routes: dict = {}

    def stub(
        self,
        method: str,
        url: str,
        *,
        status_code: int = 200,
        json: Any = None,
        text: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._routes[(method.upper(), url)] = StubbedResponse(
            status_code=status_code, json=json, text=text, headers=headers
        )

    async def request(self, method: str, url: str, **kwargs) -> StubbedResponse:
        route = (method.upper(), url)
        self.calls.append(
            RecordedCall(
                method=route[0],
                url=url,
                headers=dict(kwargs.get("headers") or {}),
                json=kwargs.get("json"),
                params=kwargs.get("params"),
            )
        )
        response = self._routes.get(route)
        if response is None:
            raise ConfigurationError(
                f"{route[0]} {url} is not stubbed. A test double refuses a request "
                f"it was not told about, so a tool cannot reach the network by "
                f"accident: {_stubbed(self._routes)}"
            )
        return response

    async def get(self, url: str, **kwargs) -> StubbedResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> StubbedResponse:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> StubbedResponse:
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> StubbedResponse:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> StubbedResponse:
        return await self.request("DELETE", url, **kwargs)


def _stubbed(routes: Mapping) -> str:
    if not routes:
        return "nothing is stubbed yet — call ctx.http.stub(method, url, ...) first."
    listed = ", ".join(f"{method} {url}" for method, url in sorted(routes))
    return f"stubbed so far: {listed}."


class CapturingLog:
    def __init__(self) -> None:
        self.records: list = []

    def debug(self, event: str, **fields) -> None:
        self.records.append(LogRecord("debug", event, fields))

    def info(self, event: str, **fields) -> None:
        self.records.append(LogRecord("info", event, fields))

    def warning(self, event: str, **fields) -> None:
        self.records.append(LogRecord("warning", event, fields))

    def error(self, event: str, **fields) -> None:
        self.records.append(LogRecord("error", event, fields))


class ToolContextDouble:
    def __init__(
        self,
        *,
        secrets: Optional[Mapping[str, str]] = None,
        aborted: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        self.http = StubHttp()
        self.log = CapturingLog()
        self.session_id = session_id or f"sess_test_{next(_SESSIONS)}"
        self.aborted = aborted
        self._secrets = dict(secrets or {})

    def secret(self, name: str) -> str:
        if name not in self._secrets:
            known = ", ".join(f"`{key}`" for key in sorted(self._secrets)) or "none"
            raise ConfigurationError(
                f"secret `{name}` is not set on this context. Pass it as "
                f"create_tool_context(secrets={{'{name}': ...}}). Set here: {known}."
            )
        return self._secrets[name]


def create_tool_context(
    *,
    secrets: Optional[Mapping[str, str]] = None,
    aborted: bool = False,
    session_id: Optional[str] = None,
) -> ToolContextDouble:
    return ToolContextDouble(secrets=secrets, aborted=aborted, session_id=session_id)


def get_tool(agent: Any, name: str) -> Tool:
    if not isinstance(agent, VoiceAgent):
        raise ConfigurationError(
            f"get_tool wants the VoiceAgent declaration and got "
            f"{type(agent).__name__}. Pass the module-level object your project "
            f"assigns from `VoiceAgent(...)`."
        )
    declared = {declared.name: declared for declared in agent.tools or ()}
    if not declared:
        raise ConfigurationError(
            f"agent `{agent.name}` lists no tools, so `{name}` cannot be reached. "
            f"Tools are an explicit list — pass them as `VoiceAgent(tools=[...])`."
        )
    if name not in declared:
        known = ", ".join(f"`{key}`" for key in sorted(declared))
        raise ConfigurationError(
            f"agent `{agent.name}` lists no tool under `{name}`, which is the name "
            f"the model calls it by. Listed: {known}."
        )
    return declared[name]
