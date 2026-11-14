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
        assert manager.submit("a", {"shared": 4096})
        assert manager.submit("b", {"shared": 4096})
        manager.release("a")
        with manager.acquire({"shared": 4096}):
            manager.release("b")
            assert not store.ended
            assert store.started["shared"] == 1
        assert store.ended["shared"] == 1
        assert manager.bytes == 0
    finally:
        manager.close()


def test_cancel_inflight_retains_source_and_budget_until_worker_finishes():
    store = Store()
    store.proceed.clear()
    manager = SessionPrefetch(store, max_bytes=4096)
    try:
        assert manager.submit("a", {"page": 4096})
        assert store.entered.wait(5)
        manager.release("a")
        assert not store.ended
        assert manager.bytes == 4096
        assert not manager.submit("b", {"other": 4096})
        store.proceed.set()
        manager.reset()
        assert store.ended["page"] == 1
        assert manager.bytes == 0
    finally:
        store.proceed.set()
        manager.close()


def test_shared_arena_is_charged_until_its_last_key_is_released():
    manager = SessionPrefetch(Store(), max_bytes=8192)
    try:
        assert manager.submit("a", {"x": 4096, "y": 4096})
        assert manager.submit("b", {"y": 4096})
        manager.release("a")
        assert manager.bytes == 8192
        assert not manager.submit("c", {"z": 4096})
        manager.release("b")
        manager.reset()
        assert manager.bytes == 0
    finally:
        manager.close()


def test_prefetch_failure_keeps_normal_session_usable():
    store = Store()
    store.fail_prefetch = True
    manager = SessionPrefetch(store)
    try:
        manager.submit("a", {"page": 4096})
        with manager.acquire({"page": 4096}):
            assert store.started["page"] == 1
        manager.release("a")
        manager.reset()
        assert manager.stats["prefetch_errors"] == 1
        assert store.ended["page"] == 1
    finally:
        manager.close()


def test_partial_start_is_reported_and_all_sessions_are_cleaned():
    store = Store()
    store.partial_start = True
    manager = SessionPrefetch(store)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            with manager.acquire({"good": 4096, "bad": 4096}):
                pytest.fail("partial session must not be consumed")
        assert store.ended == Counter({"good": 1, "bad": 1})
    finally:
        manager.close()


def test_request_limit_and_repeated_lookup_do_not_leak_refs():
    manager = SessionPrefetch(Store(), max_requests=1)
    try:
        assert manager.submit("a", {"page": 4096})
        assert "mooncake_direct_prefetch_reserved_bytes 4096" in manager.metrics()
        assert manager.submit("a", {"page": 4096})
        assert not manager.submit("b", {"other": 4096})
        manager.release("a")
        manager.reset()
        assert manager.bytes == 0
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
        assert manager.submit("a", {"page": 4096})
        with manager.acquire({"page": 4096}):
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
        manager.submit("waiting", {"page": 4096})
        with manager.acquire({"page": 4096}):
            assert store.refreshed == ["page"]
            assert store.started["page"] == 1
        manager.release("waiting")
    finally:
        manager.close()
