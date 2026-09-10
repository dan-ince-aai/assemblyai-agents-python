import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# The declaration reads these at import, and the tests import it.
os.environ.setdefault("PUBLIC_BASE_URL", "https://starter.test.local")
os.environ.setdefault("TOOL_SECRET", "test-tool-secret")
os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("BYO_LLM", "1")
os.environ.setdefault("MODEL", "off")

import pytest

import reply as reply_module
import store as store_module


@pytest.fixture(autouse=True)
def fresh_call():
    """Each test is a new call: nothing booked, nothing remembered."""
    store_module.reset()
    reply_module.memo.forget()
    yield
    store_module.reset()
    reply_module.memo.forget()
