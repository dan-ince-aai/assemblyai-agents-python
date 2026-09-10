"""Drive a deployed agent from a script, with no microphone and no phone line.

The cheapest useful test of a voice agent is a whole call: hand it a list of
things a caller says and read back what it said and which tools ran. That is
what this does, over the real WebSocket, against the agent you actually
deployed.

    from assemblyai_agents.drive import scripted_call

    transcript = await scripted_call(agent_id, [
        "Hi, I need to book a check-up.",
        "It's four four seven one.",
        "Yes, this is Maria Delgado.",
    ])
    assert "booked in for" in transcript.spoken

Each line is sent as a real `user` turn rather than as an instruction, which
matters twice: a reply engine of your own reads the transcript and sees it, and
those turns persist, so a value collected over several turns holds together.

Device audio is off, so nothing is played and no microphone is opened. The
platform still produces reply audio; it is simply not listened to.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

from ._client import AsyncClient
from .connection import AgentConnection

DEFAULT_TURN_TIMEOUT = 45.0


@dataclass
class Transcript:
    """What happened on a scripted call."""

    session_id: str = ""
    turns: list = field(default_factory=list)  # (speaker, text) in order
    errors: list = field(default_factory=list)
    timed_out: bool = False

    @property
    def agent_lines(self) -> list:
        return [text for speaker, text in self.turns if speaker == "agent"]

    @property
    def caller_lines(self) -> list:
        return [text for speaker, text in self.turns if speaker == "caller"]

    @property
    def spoken(self) -> str:
        """Everything the agent said, joined, for a quick assertion."""
        return " ".join(self.agent_lines)

    @property
    def ok(self) -> bool:
        return not self.errors and not self.timed_out

    def __str__(self) -> str:
        lines = []
        for speaker, text in self.turns:
            lines.append(("AGENT : " if speaker == "agent" else "CALLER: ") + text)
        for error in self.errors:
            lines.append("ERROR : " + error)
        return "\n".join(lines)


async def scripted_call(
    agent_id: str,
    lines: Sequence[str],
    *,
    tools: Optional[Mapping[str, Callable]] = None,
    api_key: Optional[str] = None,
    client: Optional[AsyncClient] = None,
    turn_timeout: float = DEFAULT_TURN_TIMEOUT,
    linger: float = 2.0,
    on_turn: Optional[Callable[[str, str], None]] = None,
    url: Optional[str] = None,
) -> Transcript:
    """Say each line to the agent, waiting for its reply in between.

    Returns once the script is finished, the agent stops replying, or the
    session ends. `on_turn(speaker, text)` is called as each turn happens, for
    printing progress.

    `tools` are the handlers this process answers with, keyed by the name the
    model calls. Pass them whenever the agent has tools declared without an
    `http=` config: those are resolved by whoever is connected, so without a
    handler here every call comes back to the model as an error.
    """
    transcript = Transcript()
    replies: asyncio.Queue = asyncio.Queue()
    conn = AgentConnection(
        agent_id=agent_id, api_key=api_key, client=client, tools=tools, audio=False, url=url
    )

    def record(speaker: str, text: str) -> None:
        transcript.turns.append((speaker, text))
        if on_turn is not None:
            on_turn(speaker, text)

    @conn.on_ready
    def _ready(event) -> None:
        transcript.session_id = event.session_id

    @conn.on_agent_transcript
    def _agent(text: str) -> None:
        record("agent", text)
        replies.put_nowait(text)

    @conn.on_error
    def _error(event) -> None:
        transcript.errors.append(f"{event.code.value}: {event.message}")
        replies.put_nowait("")

    async def script() -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            # The greeting is the agent's first turn; wait for it before speaking.
            await asyncio.wait_for(replies.get(), timeout=turn_timeout)
            for line in lines:
                record("caller", line)
                # Both, because the two kinds of agent read a different one.
                # A reply engine of your own reads the raw transcript and sees
                # the user message; the platform's own model does not reliably
                # act on a conversation.message, and needs the line quoted in
                # the reply instructions instead. Sending both drives either.
                await conn.say(line)
                await conn.session.create_reply(
                    f'The caller just said: "{line}". Respond to the caller, '
                    f"calling your tools as needed."
                )
                try:
                    await asyncio.wait_for(replies.get(), timeout=turn_timeout)
                except asyncio.TimeoutError:
                    transcript.timed_out = True
                    return
            await asyncio.sleep(linger)
        except ConnectionClosed:
            # The session was refused or dropped; on_error carries the reason.
            return

    async with conn:
        run = asyncio.create_task(conn.run())
        driver = asyncio.create_task(script())
        _, pending = await asyncio.wait({run, driver}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await conn.aclose()
    return transcript


def call(agent_id: str, lines: Sequence[str], **kwargs) -> Transcript:
    """`scripted_call` from synchronous code, for a script or a test."""
    return asyncio.run(scripted_call(agent_id, lines, **kwargs))
