from assemblyai_agents import SyncPager

from .conftest import Recorder


def _page(items: list, cursor, *, has_more: bool, with_metadata: bool = True) -> dict:
    body = {"agents": items, "has_more": has_more}
    if with_metadata:
        body["response_metadata"] = {"next_cursor": cursor}
    return body


def _make_pager(client) -> SyncPager:
    # The pager binds the client's low-level request method, the list path, the
    # item key, and base params (assumption R5). PR11 item_factory is identity.
    return client.paginate("/v1/agents", item_key="agents", params={"limit": 1})


def test_pager_stops_on_empty_string_cursor_yields_all(make_client, recorder: Recorder):
    client = make_client(
        [
            _page(["a", "b"], "c1", has_more=True),
            _page(["c"], "", has_more=False),  # empty string == terminal
        ],
        recorder,
    )

    items = list(_make_pager(client))

    assert items == ["a", "b", "c"]
    assert recorder.count == 2  # exactly two GETs, no infinite loop


def test_pager_passes_cursor_query_param_on_next_page(make_client, recorder: Recorder):
    client = make_client(
        [
            _page(["a", "b"], "c1", has_more=True),
            _page(["c"], "", has_more=False),
        ],
        recorder,
    )

    list(_make_pager(client))

    # First GET has no cursor; second GET carries cursor=c1.
    first_q = dict(recorder.requests[0].url.params)
    second_q = dict(recorder.requests[1].url.params)
    assert "cursor" not in first_q
    assert second_q.get("cursor") == "c1"
    # Every request is a GET and carries no Idempotency-Key (GETs are idempotent).
    assert all(r.method == "GET" for r in recorder.requests)
    assert all(r.headers.get("Idempotency-Key") is None for r in recorder.requests)


def test_pager_terminates_on_missing_response_metadata(make_client, recorder: Recorder):
    # No response_metadata at all -> treated as terminal (defensive ==  "").
    client = make_client(
        [_page(["a", "b"], None, has_more=True, with_metadata=False)], recorder
    )

    items = list(_make_pager(client))

    assert items == ["a", "b"]
    assert recorder.count == 1, "missing response_metadata must terminate, not loop"


def test_pager_stops_if_cursor_repeats(make_client, recorder: Recorder):
    # A buggy server echoes the SAME non-empty cursor it was just handed; without
    # a guard the pager would loop forever. It must stop after consuming the page
    # that first repeats the cursor.
    client = make_client(
        [
            _page(["a"], "c1", has_more=True),
            _page(["b"], "c1", has_more=True),  # same cursor again -> stop
        ],
        recorder,
    )

    items = list(_make_pager(client))

    assert items == ["a", "b"]
    assert recorder.count == 2, "a repeated cursor must terminate, not loop"
