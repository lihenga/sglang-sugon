import threading
from collections import Counter

import pytest

from sglang.srt.mem_cache.storage.mooncake_store.session_prefetch import SessionPrefetch


class Store:
    def __init__(self):
        self.started = Counter()
        self.ended = Counter()
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.fail_prefetch = False
        self.partial_start = False

    def batch_get_session_start(self, keys):
        self.started.update(keys)
        return [(-1 if self.partial_start and key == "bad" else 0) for key in keys]

    def batch_get_session_prefetch(self, keys):
        self.entered.set()
        assert self.proceed.wait(5)
        if self.fail_prefetch:
            raise OSError("storage temporarily unavailable")
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        self.ended.update(keys)


def test_shared_key_survives_one_waiter_cancel_and_active_load():
    store = Store()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["shared"])
        assert manager.submit("b", ["shared"])
        manager.release("a")
        with manager.acquire(["shared"]):
            manager.release("b")
            assert not store.ended
            assert store.started["shared"] == 1
        assert store.ended["shared"] == 1
        assert not manager.keys
    finally:
        manager.close()


def test_cancel_inflight_retains_source_until_worker_finishes():
    store = Store()
    store.proceed.clear()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["page"])
        assert store.entered.wait(5)
        manager.release("a")
        assert not store.ended
        assert "page" in manager.keys
        assert manager.submit("b", ["other"])
        manager.release("b")
        store.proceed.set()
        manager.reset()
        assert store.ended["page"] == 1
        assert not manager.keys
    finally:
        store.proceed.set()
        manager.close()


def test_shared_batch_is_retained_until_its_last_key_is_released():
    store = Store()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["x", "y"])
        assert manager.submit("b", ["y"])
        manager.release("a")
        assert set(manager.keys) == {"x", "y"}
        assert not store.ended
        assert manager.submit("c", ["z"])
        manager.release("c")
        manager.release("b")
        manager.reset()
        assert not manager.keys
    finally:
        manager.close()


def test_prefetch_failure_keeps_normal_session_usable():
    store = Store()
    store.fail_prefetch = True
    manager = SessionPrefetch(store)
    try:
        manager.submit("a", ["page"])
        with manager.acquire(["page"]):
            assert store.started["page"] == 1
        manager.release("a")
        manager.reset()
        assert manager.stats["prefetch_errors"] == 1
        assert store.ended["page"] == 1
    finally:
        manager.close()


def test_native_arena_capacity_failure_keeps_normal_read_fallback():
    class FullArenaStore(Store):
        def batch_get_session_prefetch(self, keys):
            return [-1] * len(keys)

    store = FullArenaStore()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["page"])
        with manager.acquire(["page"]):
            assert store.started["page"] == 1
        assert manager.stats["prefetch_failed_keys"] == 1
        manager.release("a")
        manager.reset()
        assert store.ended["page"] == 1
        assert not manager.keys
    finally:
        manager.close()


def test_duplicate_keys_share_one_native_session_and_release_once():
    store = Store()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["page", "page"])
        with manager.acquire(["page", "page"]):
            manager.release("a")
            assert store.started["page"] == 1
            assert not store.ended
        assert store.ended["page"] == 1
        assert not manager.keys
    finally:
        manager.close()


def test_partial_start_is_reported_and_all_sessions_are_cleaned():
    store = Store()
    store.partial_start = True
    manager = SessionPrefetch(store)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            with manager.acquire(["good", "bad"]):
                pytest.fail("partial session must not be consumed")
        assert store.ended == Counter({"good": 1, "bad": 1})
    finally:
        manager.close()


def test_request_limit_and_repeated_lookup_do_not_leak_refs():
    manager = SessionPrefetch(Store(), max_requests=1)
    try:
        assert manager.submit("a", ["page"])
        assert manager.submit("a", ["page"])
        assert not manager.submit("b", ["other"])
        manager.release("a")
        manager.reset()
        assert not manager.keys
        assert "mooncake_direct_prefetch_waiting_requests 0" in manager.metrics()
    finally:
        manager.close()


def test_memory_only_session_does_not_issue_dfs_prefetch():
    class MemoryStore(Store):
        def batch_get_session_start_with_sources(self, keys):
            return self.batch_get_session_start(keys), ["memory"] * len(keys)

    store = MemoryStore()
    manager = SessionPrefetch(store)
    try:
        assert manager.submit("a", ["page"])
        with manager.acquire(["page"]):
            assert not store.entered.is_set()
        manager.release("a")
        assert store.ended["page"] == 1
    finally:
        manager.close()


def test_expired_waiting_lease_is_refreshed_without_restarting_prefetch():
    class RefreshStore(Store):
        refreshed = []

        def batch_get_session_refresh(self, keys):
            self.refreshed.extend(keys)
            return [0] * len(keys)

    store = RefreshStore()
    manager = SessionPrefetch(store)
    try:
        manager.submit("waiting", ["page"])
        with manager.acquire(["page"]):
            assert store.refreshed == ["page"]
            assert store.started["page"] == 1
        manager.release("waiting")
    finally:
        manager.close()
