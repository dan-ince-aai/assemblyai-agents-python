"""Give a local port a public HTTPS address, with ngrok.

The platform reaches your tools and your reply endpoint over HTTPS, and it has
to: a phone call has no client on the other end to ask. So code running on your
machine needs an address, and today that means a tunnel.

This file is the whole of that. It is a script in the examples and not part of
`assemblyai_agents` on purpose:

* Nothing in the SDK knows a tunnel exists. An agent's tool URLs come from
  `PUBLIC_BASE_URL` and it does not care what put the value there.
* It is a workaround, not a product. When agent code can be deployed directly,
  this one function is what gets deleted and everything above it is unchanged.

    from expose import public_address

    with public_address(8000) as base_url:
        ...   # base_url reaches http://127.0.0.1:8000

`PUBLIC_BASE_URL` wins if it is already set, so a staging host, your own tunnel
or a real deployment needs none of this.

ngrok rather than cloudflared because one is enough, and because cloudflared's
quick tunnels can take minutes to become resolvable or never resolve at all,
which is a bad first five minutes. Swapping is a few lines below if you prefer
it: `cloudflared tunnel --url http://127.0.0.1:PORT` and read the
`trycloudflare.com` address out of its output.

    brew install ngrok && ngrok config add-authtoken <token>
"""

import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.request import urlopen

READY_TIMEOUT = 40
# ngrok's local API, which is how you find out what address it handed you.
INSPECTOR = "http://127.0.0.1:4040/api/tunnels"


def _address(port: int) -> Optional[str]:
    try:
        with urlopen(INSPECTOR, timeout=3) as response:
            tunnels = json.load(response).get("tunnels", [])
    except Exception:
        return None
    for tunnel in tunnels:
        url = tunnel.get("public_url", "")
        addr = (tunnel.get("config") or {}).get("addr", "")
        if url.startswith("https://") and addr.endswith(f":{port}"):
            return url
    return None


@contextmanager
def public_address(port: int, *, log=lambda message: print(message, flush=True)) -> Iterator[str]:
    """An HTTPS address that reaches `port`, for as long as the block runs."""
    already = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if already:
        log(f"using PUBLIC_BASE_URL: {already}")
        yield already
        return

    if not shutil.which("ngrok"):
        raise RuntimeError(
            "ngrok is not on PATH. Either install it and run "
            "`ngrok config add-authtoken <token>`, or set PUBLIC_BASE_URL to an "
            f"address that already reaches port {port}."
        )

    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"ngrok-{port}.log"
    handle = open(log_path, "w")
    process = subprocess.Popen(
        ["ngrok", "http", str(port), "--log", "stdout", "--log-format", "json"],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    try:
        url = None
        for _ in range(READY_TIMEOUT):
            time.sleep(1)
            url = _address(port)
            if url:
                break
        if not url:
            raise RuntimeError(
                f"ngrok did not report an address for port {port}. Its log is at "
                f"{log_path}. A free account allows one tunnel at a time, so close "
                f"any other one first."
            )
        log(f"ngrok: {url} -> http://127.0.0.1:{port}")
        yield url.rstrip("/")
    finally:
        process.terminate()
        try:
            process.wait(5)
        except Exception:
            process.kill()
        handle.close()
