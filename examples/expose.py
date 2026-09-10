"""Give a local port a public HTTPS address.

The platform reaches your tools over HTTPS, and it has to, because a phone call
has no client on the other end to ask. So code running on a laptop needs an
address, and today that means a tunnel.

This is the temporary part. It is a script in the examples rather than anything
in `assemblyai_agents` because it is a workaround, not part of the product: when
there is a way to deploy your agent's code directly, this one function is what
gets replaced, and nothing above it changes.

    from expose import public_address

    with public_address(8000) as base_url:
        ...   # base_url reaches http://127.0.0.1:8000

`PUBLIC_BASE_URL` wins if it is already set, so a staging host, a tunnel you
run yourself, or a deployment needs none of this.
"""

import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional
from urllib.request import urlopen

READY_TIMEOUT = 40


def _ngrok_url() -> Optional[str]:
    try:
        with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=3) as response:
            tunnels = json.load(response).get("tunnels", [])
    except Exception:
        return None
    return next((t["public_url"] for t in tunnels if t["public_url"].startswith("https://")), None)


@contextmanager
def public_address(port: int, *, prefer: Optional[str] = None, log=print) -> Iterator[str]:
    """An HTTPS address that reaches `port`, for as long as the block runs."""
    existing = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if existing:
        log(f"using PUBLIC_BASE_URL: {existing}")
        yield existing
        return

    which = prefer or ("ngrok" if shutil.which("ngrok") else "cloudflared" if shutil.which("cloudflared") else "")
    if not which:
        raise RuntimeError(
            "No public address. Either set PUBLIC_BASE_URL to something that reaches "
            f"port {port}, or install ngrok (and run `ngrok config add-authtoken ...`) "
            "or cloudflared."
        )

    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"expose-{which}-{port}.log"
    handle = open(log_path, "w")
    if which == "ngrok":
        command = ["ngrok", "http", str(port), "--log", "stdout", "--log-format", "json"]
    else:
        command = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)

    try:
        url = None
        for _ in range(READY_TIMEOUT):
            time.sleep(1)
            if which == "ngrok":
                url = _ngrok_url()
            else:
                match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text())
                url = match.group(0) if match else None
            if url:
                break
        if not url:
            raise RuntimeError(f"{which} did not report an address; see {log_path}")
        log(f"{which}: {url} -> http://127.0.0.1:{port}")
        yield url.rstrip("/")
    finally:
        process.terminate()
        try:
            process.wait(5)
        except Exception:
            process.kill()
        handle.close()
