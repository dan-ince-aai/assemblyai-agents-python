from ._agent import VoiceAgent
from ._client import AsyncClient, Client
from ._context import ToolContext
from ._exceptions import (
    APIError,
    AssemblyAIAgentsError,
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    ConflictError,
    ErrorCode,
    NotFoundError,
    RealtimeError,
    ResponseError,
    ServerError,
    ValidationError,
    WebhookSignatureError,
    WebhookTimestampError,
    WebhookVerificationError,
)
from ._io import AudioFormat, AudioInput, AudioOutput
from ._pagination import AsyncPager, SyncPager
from ._response import RawResponse
from ._telephony import Captured, Header, HumanTransfer, PreConnectRequest
from ._tool import Tool, tool
from .deploy import deploy
from ._version import __version__
from .audio import (
    alaw_to_pcm16,
    base64_to_pcm,
    pcm16_to_alaw,
    pcm16_to_ulaw,
    pcm_to_base64,
    ulaw_to_pcm16,
)
from .audio_io import DeviceAudioNotInstalledError, PlaybackSink, microphone_stream
from .connection import AgentConnection

# The generated request models a customer has to construct to reach the raw
# escape hatch. Only these five are re-exported: the rest of `models.rest` is
# reachable as `assemblyai_agents.models.rest.*` and lifting all ~60 of them
# here would bury the names above.
from .models.rest import (
    AgentCreateRequest,
    AgentUpdateRequest,
    LlmConfigRequest,
    PlaintextToolDefinition,
    VoiceConfig,
)
from .models.ws import (
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
from .realtime import AsyncRealtimeSession, UnknownEvent
from .webhooks import verify

__all__ = [
    "__version__",
    "Client",
    "AsyncClient",
    "RawResponse",
    "SyncPager",
    "AsyncPager",
    "AssemblyAIAgentsError",
    "ConfigurationError",
    "ResponseError",
    "RealtimeError",
    "APIError",
    "ErrorCode",
    "BadRequestError",
    "AuthenticationError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "ServerError",
    "WebhookVerificationError",
    "WebhookSignatureError",
    "WebhookTimestampError",
    "AsyncRealtimeSession",
    "UnknownEvent",
    "InputSpeechStarted",
    "InputSpeechStopped",
    "ReplyAudio",
    "ReplyDone",
    "ReplyStarted",
    "SessionEnded",
    "SessionError",
    "SessionReady",
    "SessionUpdatedEvent",
    "ToolCall",
    "TranscriptAgent",
    "TranscriptAgentDelta",
    "TranscriptUser",
    "AgentCreateRequest",
    "AgentUpdateRequest",
    "LlmConfigRequest",
    "PlaintextToolDefinition",
    "VoiceConfig",
    "verify",
    "pcm_to_base64",
    "base64_to_pcm",
    "pcm16_to_ulaw",
    "ulaw_to_pcm16",
    "pcm16_to_alaw",
    "alaw_to_pcm16",
    "DeviceAudioNotInstalledError",
    "PlaybackSink",
    "microphone_stream",
    "AgentConnection",
    "VoiceAgent",
    "AudioFormat",
    "AudioInput",
    "AudioOutput",
    "HumanTransfer",
    "PreConnectRequest",
    "Header",
    "Captured",
    "tool",
    "Tool",
    "ToolContext",
    "deploy",
]
