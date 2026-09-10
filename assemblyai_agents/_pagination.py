from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
)

T = TypeVar("T")

_TERMINAL = ""


def _next_cursor(page: Any) -> str:
    if not isinstance(page, dict):
        return _TERMINAL
    metadata = page.get("response_metadata")
    if not isinstance(metadata, dict):
        return _TERMINAL
    cursor = metadata.get("next_cursor")
    if cursor is None:
        return _TERMINAL
    return cursor


def _page_items(page: Any, item_key: str, item_factory: Callable[[Any], T]) -> list:
    if not isinstance(page, dict):
        return []
    return [item_factory(item) for item in page.get(item_key, [])]


def _params_with_cursor(
    base_params: Optional[Mapping[str, Any]], cursor: Optional[str]
) -> dict:
    params = dict(base_params or {})
    if cursor:
        params["cursor"] = cursor
    return params


class SyncPager(Generic[T]):
    def __init__(
        self,
        fetch: Callable[..., Any],
        path: str,
        item_key: str,
        params: Optional[Mapping[str, Any]],
        item_factory: Callable[[Any], T],
    ) -> None:
        self._fetch = fetch
        self._path = path
        self._item_key = item_key
        self._base_params = params
        self._item_factory = item_factory
        self._cursor: Optional[str] = None
        self._exhausted = False

    @property
    def has_more(self) -> bool:
        return not self._exhausted

    def next_page(self) -> Optional[list]:
        if self._exhausted:
            return None
        used_cursor = self._cursor
        page = self._fetch(
            "GET",
            self._path,
            params=_params_with_cursor(self._base_params, used_cursor),
        )
        items = _page_items(page, self._item_key, self._item_factory)
        cursor = _next_cursor(page)
        # A server that echoes the same non-empty cursor it was just handed would
        # loop forever; treat a repeated cursor as terminal.
        if cursor == _TERMINAL or cursor == used_cursor:
            self._exhausted = True
        else:
            self._cursor = cursor
        return items

    def __iter__(self) -> Iterator[T]:
        while True:
            page = self.next_page()
            if page is None:
                return
            yield from page


class AsyncPager(Generic[T]):
    def __init__(
        self,
        fetch: Callable[..., Any],
        path: str,
        item_key: str,
        params: Optional[Mapping[str, Any]],
        item_factory: Callable[[Any], T],
    ) -> None:
        self._fetch = fetch
        self._path = path
        self._item_key = item_key
        self._base_params = params
        self._item_factory = item_factory
        self._cursor: Optional[str] = None
        self._exhausted = False

    @property
    def has_more(self) -> bool:
        return not self._exhausted

    async def next_page(self) -> Optional[list]:
        if self._exhausted:
            return None
        used_cursor = self._cursor
        page = await self._fetch(
            "GET",
            self._path,
            params=_params_with_cursor(self._base_params, used_cursor),
        )
        items = _page_items(page, self._item_key, self._item_factory)
        # A server that echoes the same non-empty cursor it was just handed would
        # loop forever; treat a repeated cursor as terminal.
        cursor = _next_cursor(page)
        if cursor == _TERMINAL or cursor == used_cursor:
            self._exhausted = True
        else:
            self._cursor = cursor
        return items

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            page = await self.next_page()
            if page is None:
                return
            for item in page:
                yield item
