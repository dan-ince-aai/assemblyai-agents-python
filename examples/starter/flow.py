"""Two small pieces of organisation. Both are opinions, so they live here.

Neither is in the SDK, and neither has to be in your project either. They are
here because a call of any length wants them, and forty lines you can read is
better than a framework you cannot.

`Stages` splits a call into parts, each finished when its own test says so. A
call has a shape: the ends are usually a script, the middle is a conversation.
One handler trying to be both fills up with conditions.

`Memo` remembers what has already been said. Nearly everything else can be read
back out of the transcript and should be, but not this: a caller who talks over
a long scripted line leaves that turn interrupted, and it does not come back in
the messages, so "have I said this yet?" answers no and the line repeats.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Stage:
    name: str
    handler: Callable
    until: Optional[Callable] = None


class Stages:
    """Ordered handlers. The first one not yet finished answers the turn.

        stages = Stages()

        @stages.stage("identify", until=lambda turn: verified(turn))
        def identify(turn): ...

        @stages.stage("close")          # no `until`: the last one
        def close(turn): ...

    A handler returns whatever the platform should hear next, or None to fall
    through to the stage after it.
    """

    def __init__(self) -> None:
        self._stages: list[Stage] = []

    def stage(self, name: str, until: Optional[Callable] = None) -> Callable:
        def register(handler: Callable) -> Callable:
            self._stages.append(Stage(name, handler, until))
            return handler

        return register

    def current(self, turn) -> Optional[Stage]:
        return next((s for s in self._stages if s.until is None or not s.until(turn)), None)

    def decide(self, turn):
        for stage in self._stages:
            if stage.until is not None and stage.until(turn):
                continue
            answer = stage.handler(turn)
            if answer is not None:
                return answer
        return None

    @property
    def names(self) -> tuple:
        return tuple(stage.name for stage in self._stages)


class Memo:
    """A note of what has happened on a call, keyed however you like.

    Process-local, which is enough for one worker. A fleet would keep the same
    notes wherever it keeps session state, and forget them per call.
    """

    def __init__(self) -> None:
        self._notes: dict[str, set] = {}

    def note(self, key: str, *markers: str) -> None:
        self._notes.setdefault(key or "-", set()).update(markers)

    def has(self, key: str, marker: str) -> bool:
        return marker in self._notes.get(key or "-", set())

    def forget(self, key: Optional[str] = None) -> None:
        self._notes.clear() if key is None else self._notes.pop(key, None)
