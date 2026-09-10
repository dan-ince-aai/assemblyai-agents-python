"""Talk to a deployed agent through your microphone and speakers.

Requires the audio extra (PortAudio + pyaudio):

    pip install "assemblyai-agents[audio] @ git+https://github.com/dan-ince-aai/assemblyai-agents-python.git"
    export ASSEMBLYAI_API_KEY=...
    python examples/talk.py $AGENT_ID

With ``PUBLIC_BASE_URL`` unset when the agent was deployed, its tools are
client-resident and run in this process: pass them to
``AgentConnection(tools=...)`` under the names the model calls. With HTTP tools
the platform calls server.py instead and the ``tools=`` mapping is simply unused.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from assemblyai_agents import AgentConnection
from pizza_line import cancel_order, lookup_order


async def main(agent_id: str) -> None:
    conn = AgentConnection(
        agent_id=agent_id,
        tools={"lookup_order": lookup_order, "cancel_order": cancel_order},
    )

    @conn.on_ready
    def _ready(event):
        print(f"connected (session {event.session_id}); start talking, Ctrl-C to hang up")

    @conn.on_user_transcript
    def _user(text):
        print(f"you:   {text}")

    @conn.on_agent_transcript
    def _agent(text):
        print(f"agent: {text}")

    @conn.on_error
    def _error(event):
        print(f"error: {event.code.value}: {event.message}")

    async with conn:
        await conn.run()  # returns when the session ends


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/talk.py AGENT_ID")
    try:
        asyncio.run(main(sys.argv[1]))
    except KeyboardInterrupt:
        pass
