import time

from src.api.sessions import SessionStore


def test_create_returns_a_server_generated_id():
    store = SessionStore()
    session_id, conversation = store.create()
    assert len(session_id) >= 16
    assert conversation.messages == []


def test_ids_are_unique_and_unguessable():
    store = SessionStore()
    ids = {store.create()[0] for _ in range(50)}
    assert len(ids) == 50


def test_get_returns_the_same_conversation():
    store = SessionStore()
    session_id, conversation = store.create()
    conversation.add_user("chicken")
    assert store.get(session_id) is conversation


def test_get_of_an_unknown_id_is_none():
    assert SessionStore().get("nope") is None


def test_expired_sessions_are_dropped():
    store = SessionStore(ttl_seconds=0.05)
    session_id, _ = store.create()
    time.sleep(0.1)
    assert store.get(session_id) is None


def test_get_or_create_recovers_from_an_expired_id():
    """An idle client should keep chatting, not get an error."""
    store = SessionStore(ttl_seconds=0.05)
    old_id, _ = store.create()
    time.sleep(0.1)
    new_id, conversation = store.get_or_create(old_id)
    assert new_id != old_id
    assert conversation.messages == []


def test_store_is_bounded_so_ids_cannot_exhaust_memory():
    store = SessionStore(max_sessions=10)
    ids = [store.create()[0] for _ in range(25)]
    assert len(store) == 10
    assert store.get(ids[0]) is None    # oldest evicted
    assert store.get(ids[-1]) is not None


def test_active_sessions_survive_eviction():
    store = SessionStore(max_sessions=3)
    keep, _ = store.create()
    for _ in range(3):
        store.get(keep)                  # touch it
        store.create()
    assert store.get(keep) is not None


def test_delete_removes_and_reports():
    store = SessionStore()
    session_id, _ = store.create()
    assert store.delete(session_id) is True
    assert store.delete(session_id) is False
