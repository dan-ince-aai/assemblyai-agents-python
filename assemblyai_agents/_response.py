import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class RawResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    request_id: Optional[str]

    def json(self) -> Any:
        return json.loads(self.content)
