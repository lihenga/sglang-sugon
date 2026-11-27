import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.storage.mooncake_store import session_prefetch
from sglang.srt.mem_cache.storage.mooncake_store.session_prefetch import (
    SessionPrefetch,
    session_refresh_age_from_env,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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


def samples(manager):
    from prometheus_client.parser import text_string_to_metric_families

    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(manager.metrics())
        for sample in family.samples
    }


def test_prefetch_metrics_count_reuse_claim_and_ready_lead_once(clock):
    manager = SessionPrefetch(Store(), enable_metrics=True)
    try:
        prepare(manager)
        clock.now += 2
        assert manager.submit("waiting", ["page"])
        manager.claim("waiting")
        manager.claim("waiting")
        manager.release("waiting")
        metric = samples(manager)
        prefix = "mooncake_direct_prefetch_"
        assert metric[prefix + "events_total", (("event", "submitted"),)] == 1
        assert metric[prefix + "events_total", (("event", "reused_requests"),)] == 1
        assert metric[prefix + "events_total", (("event", "claimed"),)] == 1
        assert metric[prefix + "ready_lead_seconds_count", ()] == 1
        assert metric[prefix + "ready_lead_seconds_sum", ()] == 2
        assert (
            metric[prefix + "duration_seconds_count", (("result", "prefetched"),)] == 1
        )
        assert manager.stats["discarded_ready"] == 0
    finally:
        manager.close()


def test_prefetch_metrics_cancel_inflight_once_and_keep_ownership():
    store = Store()
    store.proceed.clear()
    manager = SessionPrefetch(store, enable_metrics=True)
    try:
        manager.submit("waiting", ["page"])
        assert store.entered.wait(5)
        manager.release("waiting")
        assert not store.ended
        store.proceed.set()
        manager.reset()
        metric = samples(manager)
        prefix = "mooncake_direct_prefetch_duration_seconds_count"
        assert metric[prefix, (("result", "cancelled"),)] == 1
        assert (prefix, (("result", "prefetched"),)) not in metric
        assert store.ended["page"] == 1
    finally:
        store.proceed.set()
        manager.close()


def test_prefetch_metrics_shared_ready_requests_release_independently():
    store = Store()
    manager = SessionPrefetch(store, enable_metrics=True)
    try:
        prepare(manager, "first")
        prepare(manager, "second")
        assert samples(manager)["mooncake_direct_prefetch_ready_requests", ()] == 2
        manager.release("first")
        assert not store.ended
        manager.claim("second")
        manager.release("second")
        assert manager.stats["discarded_ready"] == 1
        assert manager.stats["claimed"] == 1
        assert store.started["page"] == store.ended["page"] == 1
    finally:
        manager.close()


def test_shared_batch_failure_does_not_mark_an_unaffected_request_failed():
    store = Store()
    store.partial_start = True
    manager = SessionPrefetch(store, enable_metrics=True)
    try:
        prepare(manager, "mixed", ("bad", "good"))
        prepare(manager, "good-only", ("good",))
        metric = samples(manager)
        prefix = "mooncake_direct_prefetch_duration_seconds_count"
        assert metric[prefix, (("result", "failed"),)] == 1
        assert metric[prefix, (("result", "prefetched"),)] == 1
        manager.claim("good-only")
        assert manager.stats["claimed_without_ready_prefetch"] == 0
    finally:
        manager.close()


def test_prefetch_request_borrowing_demand_session_is_not_prefetched():
    manager = SessionPrefetch(Store(), enable_metrics=True)
    try:
        with manager.acquire(["page"]):
            prepare(manager)
            manager.claim("waiting")
            metric = samples(manager)
            assert (
                metric[
                    "mooncake_direct_prefetch_duration_seconds_count",
                    (("result", "session_only"),),
                ]
                == 1
            )
            assert metric["mooncake_direct_prefetch_ready_lead_seconds_count", ()] == 0
            manager.release("waiting")
    finally:
        manager.close()


def test_prefetch_metrics_claim_before_ready_does_not_invent_lead():
    store = Store()
    store.proceed.clear()
    manager = SessionPrefetch(store, enable_metrics=True)
    try:
        manager.submit("waiting", ["page"])
        assert store.entered.wait(5)
        manager.claim("waiting")
        store.proceed.set()
        with manager.acquire(["page"]):
            pass
        manager.release("waiting")
        assert manager.stats["claimed_without_ready_prefetch"] == 1
        assert (
            samples(manager)["mooncake_direct_prefetch_ready_lead_seconds_count", ()]
            == 0
        )
        assert manager.stats["discarded_ready"] == 0
    finally:
        store.proceed.set()
        manager.close()


@pytest.mark.parametrize("failure", ["exception", "return_code", "session_start"])
def test_prefetch_metrics_failures_are_distinct_from_admission_rejection(failure):
    class FailedStore(Store):
        def batch_get_session_prefetch(self, keys):
            if failure == "return_code":
                return [-1] * len(keys)
            return super().batch_get_session_prefetch(keys)

    store = FailedStore()
    store.fail_prefetch = failure == "exception"
    store.partial_start = failure == "session_start"
    manager = SessionPrefetch(store, enable_metrics=True, max_requests=1)
    try:
        prepare(manager, keys=("bad",))
        assert not manager.submit("rejected", ["other"])
        assert manager.stats["rejected_limit"] == 1
        assert (
            samples(manager)[
                "mooncake_direct_prefetch_duration_seconds_count",
                (("result", "failed"),),
            ]
            == 1
        )
        if failure != "session_start":
            with manager.acquire(["bad"]):
                pass  # Failed speculative reads still allow the normal load.
    finally:
        manager.close()


def test_prefetch_metrics_reset_records_pending_cancellation(monkeypatch):
    store = Store()
    store.proceed.clear()
    manager = SessionPrefetch(store, enable_metrics=True)
    try:
        manager.submit("waiting", ["page"])
        assert store.entered.wait(5)
        release = manager.release

        def release_and_unblock(rid):
            release(rid)
            store.proceed.set()

        monkeypatch.setattr(manager, "release", release_and_unblock)
        manager.reset()
        manager.reset()
        assert (
            samples(manager)[
                "mooncake_direct_prefetch_duration_seconds_count",
                (("result", "cancelled"),),
            ]
            == 1
        )
        assert samples(manager)["mooncake_direct_prefetch_waiting_requests", ()] == 0
    finally:
        store.proceed.set()
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


def test_memory_only_prefetch_metrics_do_not_count_dfs_readiness():
    class MemoryStore(Store):
        def batch_get_session_start_with_sources(self, keys):
            return self.batch_get_session_start(keys), ["memory"] * len(keys)

    manager = SessionPrefetch(MemoryStore(), enable_metrics=True)
    try:
        prepare(manager)
        manager.claim("waiting")
        manager.release("waiting")
        metric = samples(manager)
        assert (
            metric[
                "mooncake_direct_prefetch_duration_seconds_count",
                (("result", "no_prefetch_needed"),),
            ]
            == 1
        )
        assert metric["mooncake_direct_prefetch_ready_lead_seconds_count", ()] == 0
        assert manager.stats["discarded_ready"] == 0
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


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class RefreshStore(Store):
    def __init__(self):
        super().__init__()
        self.refreshed = []
        self.refresh_result = None
        self.refresh_error = None

    def batch_get_session_refresh(self, keys):
        self.refreshed.append(list(keys))
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.refresh_result is not None:
            return self.refresh_result
        return [0] * len(keys)


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(session_prefetch, "time", SimpleNamespace(monotonic=clock))
    return clock


def prepare(manager, rid="waiting", keys=("page",)):
    assert manager.submit(rid, keys)
    for key in keys:
        manager.keys[key].future.result(timeout=5)


def test_young_private_prefetch_skips_refresh_and_preserves_ownership(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        clock.now += 4.9
        with manager.acquire(["page"]):
            manager.release("waiting")
            assert not store.ended
        assert not store.refreshed
        assert store.started["page"] == store.ended["page"] == 1
        assert manager.stats["refresh_skipped_keys"] == 1
    finally:
        manager.close()


def test_age_boundary_refreshes_then_young_session_can_skip(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        clock.now += 5
        with manager.acquire(["page"]):
            pass
        clock.now += 0.1
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"]]
        assert manager.stats["refreshed_keys"] == 1
        assert manager.stats["refresh_skipped_keys"] == 1
    finally:
        manager.close()


@pytest.mark.parametrize("phase", ["query", "prefetch"])
def test_rpc_and_prefetch_time_count_toward_age(clock, phase):
    class SlowStore(RefreshStore):
        def batch_get_session_start(self, keys):
            if phase == "query":
                clock.now += 6
            return super().batch_get_session_start(keys)

        def batch_get_session_prefetch(self, keys):
            if phase == "prefetch":
                clock.now += 6
            return super().batch_get_session_prefetch(keys)

    store = SlowStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"]]
    finally:
        manager.close()


def test_timestamp_is_taken_when_worker_queries_not_when_queued(clock):
    store = RefreshStore()
    store.proceed.clear()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        assert manager.submit("blocking", ["other"])
        assert store.entered.wait(5)
        assert manager.submit("waiting", ["page"])
        clock.now += 10
        store.proceed.set()
        manager.keys["page"].future.result(timeout=5)
        with manager.acquire(["page"]):
            pass
        assert not store.refreshed
    finally:
        store.proceed.set()
        manager.close()


def test_cross_request_shared_key_keeps_refreshing_after_one_cancels(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager, "first")
        prepare(manager, "second")
        manager.release("first")
        with manager.acquire(["page"]):
            pass
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"], ["page"]]
        assert store.started["page"] == 1
    finally:
        manager.close()


def test_overlapping_loads_disable_skipping_even_after_one_finishes(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        with manager.acquire(["page"]):
            with manager.acquire(["page"]):
                assert not store.ended
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"], ["page"]]
    finally:
        manager.close()


@pytest.mark.parametrize("refresh_delay", [0, 28])
def test_any_stale_key_refreshes_entire_batch(clock, refresh_delay):
    class LeaseStore(RefreshStore):
        def __init__(self):
            super().__init__()
            self.deadlines = {}

        def batch_get_session_start(self, keys):
            self.deadlines.update({key: clock.now + 30 for key in keys})
            return super().batch_get_session_start(keys)

        def batch_get_session_refresh(self, keys):
            # Native BatchQuery counts the RPC's time against the returned TTL.
            self.deadlines.update({key: clock.now + 30 for key in keys})
            clock.now += refresh_delay
            return super().batch_get_session_refresh(keys)

    store = LeaseStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager, "old", ("old",))
        clock.now += 4
        prepare(manager, "fresh", ("fresh",))
        clock.now += 2
        with manager.acquire(["old", "fresh"]):
            assert all(clock.now < deadline for deadline in store.deadlines.values())
        assert store.refreshed == [["old", "fresh"]]
        assert manager.stats["refresh_skipped_keys"] == 0
        assert manager.stats["refreshed_keys"] == 2
    finally:
        manager.close()


def test_entire_young_private_batch_can_skip_refresh(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager, keys=("a", "b"))
        clock.now += 4.9
        with manager.acquire(["a", "b"]):
            pass
        assert not store.refreshed
        assert manager.stats["refresh_skipped_keys"] == 2
    finally:
        manager.close()


@pytest.mark.parametrize("reason", ["shared", "unknown"])
def test_one_unsafe_key_forces_refresh_of_young_batch(clock, reason):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager, keys=("a", "b"))
        if reason == "shared":
            prepare(manager, "other", ("a",))
        else:
            manager.keys["a"].lease_bases.pop("a")
        with manager.acquire(["a", "b"]):
            pass
        assert store.refreshed == [["a", "b"]]
        assert manager.stats["refresh_skipped_keys"] == 0
    finally:
        manager.close()


def test_partial_batch_refresh_failure_invalidates_all_lease_bases(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager, "old", ("old",))
        clock.now += 4
        prepare(manager, "fresh", ("fresh",))
        clock.now += 2
        store.refresh_result = [-1, 0]
        with pytest.raises(RuntimeError, match="session refresh failed"):
            with manager.acquire(["old", "fresh"]):
                pytest.fail("failed refresh must not admit a load")
        assert store.refreshed == [["old", "fresh"]]
        assert not manager.keys["old"].lease_bases
        assert not manager.keys["fresh"].lease_bases
    finally:
        manager.close()


def test_unknown_lease_base_always_refreshes(clock):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        manager.keys["page"].lease_bases.clear()
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"]]
    finally:
        manager.close()


def test_failed_cleanup_forces_refresh_before_reusing_session(clock):
    class FailedEndStore(RefreshStore):
        def batch_get_session_end(self, keys):
            return -1

    store = FailedEndStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        manager.release("waiting")
        assert "page" in manager.keys
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"]]
    finally:
        manager.close()


@pytest.mark.parametrize("result", [[], [-1], [0, 0], "exception"])
def test_failed_refresh_invalidates_age_and_never_admits_load(clock, result):
    store = RefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        clock.now += 5
        if result == "exception":
            store.refresh_error = OSError("master unavailable")
        else:
            store.refresh_result = result
        with pytest.raises((RuntimeError, OSError)):
            with manager.acquire(["page"]):
                pytest.fail("failed refresh must not admit a load")
        assert "page" not in manager.keys["page"].lease_bases
        store.refresh_error = store.refresh_result = None
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"], ["page"]]
    finally:
        manager.close()


def test_refresh_rpc_time_is_included_in_new_age(clock):
    class SlowRefreshStore(RefreshStore):
        def batch_get_session_refresh(self, keys):
            clock.now += 6
            return super().batch_get_session_refresh(keys)

    store = SlowRefreshStore()
    manager = SessionPrefetch(store, refresh_age_s=5)
    try:
        prepare(manager)
        clock.now += 5
        with manager.acquire(["page"]):
            pass
        with manager.acquire(["page"]):
            pass
        assert store.refreshed == [["page"], ["page"]]
    finally:
        manager.close()


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "invalid"])
def test_invalid_refresh_age_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", value)
    with pytest.raises(ValueError):
        session_refresh_age_from_env()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_lease_ttl_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", "5")
    monkeypatch.setenv("SGLANG_MOONCAKE_LEASE_TTL_S", value)
    with pytest.raises(ValueError):
        session_refresh_age_from_env()


def test_refresh_skip_configuration_guards_ttl_and_groups(monkeypatch):
    monkeypatch.delenv("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", raising=False)
    monkeypatch.delenv("SGLANG_MOONCAKE_LEASE_TTL_S", raising=False)
    assert session_refresh_age_from_env() == 0
    monkeypatch.setenv("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", "5")
    monkeypatch.setenv("SGLANG_MOONCAKE_LEASE_TTL_S", "30")
    assert session_refresh_age_from_env() == 5
    assert session_refresh_age_from_env(group_semantics=True) == 0
    monkeypatch.setenv("SGLANG_MOONCAKE_LEASE_TTL_S", "5")
    assert session_refresh_age_from_env() == 0
    monkeypatch.setenv("SGLANG_MOONCAKE_LEASE_TTL_S", "4")
    assert session_refresh_age_from_env() == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
