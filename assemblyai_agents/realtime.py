import asyncio
import base64
import json
import random
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import websockets.exceptions
from pydantic import BaseModel, ConfigDict
from websockets.asyncio.client import ClientConnection, connect

from ._exceptions import RealtimeError
from .models.ws import (
    Code,
    InputSpeechStarted,
    InputSpeechStopped,
    ReplyAudio,
    ReplyDone,
    ReplyStarted,
    SessionEnded,
    SessionError,
    SessionReady,
    SessionUpdatedEvent,
    ToolCall,
    TranscriptAgent,
    TranscriptAgentDelta,
    TranscriptUser,
)

_WS_SCHEME = {"https": "wss", "http": "ws", "ws": "ws", "wss": "wss"}

_RESUME_BUDGET_SEC = 25.0

_TERMINAL_RESUME_CODES = frozenset(
    {
        Code.session_not_found,
        Code.session_expired,
        Code.session_forbidden,
        Code.unauthorized,
    }
)


def _ws_url_from_base(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = _WS_SCHEME.get(parts.scheme, parts.scheme)
    return urlunsplit((scheme, parts.netloc, "/v1/ws", "", ""))


_SERVER_EVENTS: dict[str, type[BaseModel]] = {
    "session.ready": SessionReady,
    "session.updated": SessionUpdatedEvent,
    "session.error": SessionError,
    "session.ended": SessionEnded,
    "input.speech.started": InputSpeechStarted,
    "input.speech.stopped": InputSpeechStopped,
    "reply.audio": ReplyAudio,
    "reply.done": ReplyDone,
    "reply.started": ReplyStarted,
    "tool.call": ToolCall,
    "transcript.user": TranscriptUser,
    "transcript.agent": TranscriptAgent,
    "transcript.agent.delta": TranscriptAgentDelta,
}


class UnknownEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = None
    raw: dict[str, Any]


def _parse_frame(frame) -> BaseModel:
    if isinstance(frame, bytes):
        frame = frame.decode()
    try:
        data = json.loads(frame)
    except ValueError:
        return UnknownEvent(type=None, raw={"__raw__": frame})
    if not isinstance(data, dict):
        return UnknownEvent(type=None, raw={"__raw__": data})
    model_cls = _SERVER_EVENTS.get(data.get("type"))
    if model_cls is None:
        return UnknownEvent(type=data.get("type"), raw=data)
    try:
        return model_cls.model_validate(data)
    except Exception:
        return UnknownEvent(type=data.get("type"), raw=data)


class AsyncRealtimeSession:
    def __init__(
        self,
        ws: ClientConnection,
        *,
        url: Optional[str] = None,
        open_timeout: Optional[float] = None,
        auto_resume: bool = False,
        max_resume_attempts: int = 5,
    ) -> None:
        self._ws = ws
        self._url = url
        self._open_timeout = open_timeout
        self._auto_resume = auto_resume
        self._max_resume_attempts = max_resume_attempts
        self._close_code: Optional[int] = None
        self._closed = False
        self._client_ended = False
        self._session_id: Optional[str] = None
        self._resume_token: Optional[str] = None
        self._pending_frame = None

    async def _send(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))

    async def update(
        self,
        *,
        agent_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        greeting: Optional[str] = None,
        input: Optional[dict] = None,
        output: Optional[dict] = None,
        tools: Optional[list] = None,
        webhook: Optional[dict] = None,
    ) -> None:
        candidates = {
            "agent_id": agent_id,
            "system_prompt": system_prompt,
            "greeting": greeting,
            "input": input,
            "output": output,
            "tools": tools,
            "webhook": webhook,
        }
        session = {k: v for k, v in candidates.items() if v is not None}
        await self._send({"type": "session.update", "session": session})

    async def resume(self, session_id: str) -> None:
        await self._send({"type": "session.resume", "session_id": session_id})

    async def end(self) -> None:
        self._client_ended = True
        await self._send({"type": "session.end"})

    async def send_audio(self, audio: bytes) -> None:
        await self._send(
            {"type": "input.audio", "audio": base64.b64encode(audio).decode()}
        )

    async def cancel_reply(self, reply_id: str) -> None:
        await self._send({"type": "reply.cancel", "reply_id": reply_id})

    async def create_reply(self, instructions: Optional[str] = None) -> None:
        payload: dict = {"type": "reply.create"}
        if instructions is not None:
            payload["instructions"] = instructions
        await self._send(payload)

    async def send_tool_result(
        self, call_id: str, result: str, *, is_error: bool = False
    ) -> None:
        await self._send(
            {
                "type": "tool.result",
                "call_id": call_id,
                "result": result,
                "is_error": is_error,
            }
        )

    async def send_message(self, content: str, *, role: str = "user") -> None:
        await self._send(
            {"type": "conversation.message", "role": role, "content": content}
        )

    def __aiter__(self) -> "AsyncRealtimeSession":
        return self

    async def __anext__(self) -> BaseModel:
        while True:
            if self._pending_frame is not None:
                frame = self._pending_frame
                self._pending_frame = None
            else:
                try:
                    frame = await self._ws.recv()
                except websockets.exceptions.ConnectionClosed as exc:
                    drop_code = exc.rcvd.code if exc.rcvd else None
                    abnormal = isinstance(
                        exc, websockets.exceptions.ConnectionClosedError
                    )
                    if (
                        not abnormal
                        or not self._auto_resume
                        or self._client_ended
                        or not self._resume_token
                    ):
                        self._close_code = drop_code
                        self._closed = True
                        raise StopAsyncIteration
                    await self._reconnect_and_resume(drop_code)
                    continue
            event = _parse_frame(frame)
            if isinstance(event, SessionReady):
                self._session_id = event.session_id
                self._resume_token = event.resume_token
            return event

    async def _reconnect_and_resume(self, drop_code: Optional[int]) -> None:
        """Bounded backoff reconnect against the server's 30s resume grace.

        Per attempt: exponential backoff (small base) with jitter, capped, and
        abandoned the moment the next sleep would push cumulative elapsed past
        the grace budget — a late reconnect only earns a wasted
        session_not_found. A live new socket sends session.resume{session_id}
        with the held resume_token as the Bearer; the server's first reply is
        peeked: a terminal session.error code is unrecoverable and raised at
        once, otherwise the frame is buffered and the iterator resumes.
        """
        elapsed = 0.0
        for attempt in range(self._max_resume_attempts):
            delay = min(0.25 * 2**attempt, 5.0)
            delay += random.uniform(0, 0.25 * delay)
            if elapsed + delay > _RESUME_BUDGET_SEC:
                break
            await asyncio.sleep(delay)
            elapsed += delay

            try:
                new_session = await _connect(
                    self._url, self._resume_token, self._open_timeout
                )
            except (RealtimeError, OSError, asyncio.TimeoutError):
                continue
            new_ws = new_session._ws

            try:
                await new_ws.send(
                    json.dumps(
                        {"type": "session.resume", "session_id": self._session_id}
                    )
                )
                first = await new_ws.recv()
            except websockets.exceptions.ConnectionClosed:
                await new_ws.close()
                continue

            event = _parse_frame(first)
            if isinstance(event, SessionError) and event.code in _TERMINAL_RESUME_CODES:
                await new_ws.close()
                raise RealtimeError(
                    f"auto-resume rejected: {event.code.value}",
                    close_code=drop_code,
                )

            self._pending_frame = first
            self._ws = new_ws
            self._close_code = None
            return

        raise RealtimeError(
            f"auto-resume failed after {self._max_resume_attempts} attempts",
            close_code=drop_code,
        )

    async def __aenter__(self) -> "AsyncRealtimeSession":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._client_ended = True
        await self._ws.close()

    @property
    def close_code(self) -> Optional[int]:
        return self._close_code

    def __repr__(self) -> str:
        return f"AsyncRealtimeSession(close_code={self._close_code!r})"


async def _connect(
    url: str,
    token: str,
    open_timeout: Optional[float],
    *,
    auto_resume: bool = False,
    max_resume_attempts: int = 5,
) -> AsyncRealtimeSession:
    try:
        ws = await connect(
            url,
            additional_headers={"Authorization": f"Bearer {token}"},
            open_timeout=open_timeout,
            close_timeout=5,
        )
    except websockets.exceptions.InvalidHandshake as exc:
        raise RealtimeError(f"WebSocket handshake failed: {exc}") from exc
    return AsyncRealtimeSession(
        ws,
        url=url,
        open_timeout=open_timeout,
        auto_resume=auto_resume,
        max_resume_attempts=max_resume_attempts,
    )
