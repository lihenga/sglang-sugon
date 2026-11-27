"""Bounded host prefetch with shared ownership of native key sessions."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def session_refresh_age_from_env(*, group_semantics=False):
    """Read the opt-in age bound and the deployment-declared Master lease TTL."""
    age = float(os.environ.get("SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S", "0") or 0)
    ttl = float(os.environ.get("SGLANG_MOONCAKE_LEASE_TTL_S", "10") or 10)
    if not math.isfinite(age) or age < 0:
        raise ValueError(
            "SGLANG_MOONCAKE_SESSION_REFRESH_AGE_S must be finite and >= 0"
        )
    if not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("SGLANG_MOONCAKE_LEASE_TTL_S must be finite and > 0")
    if age > 0 and (age >= ttl or group_semantics):
        # Group replies may overstate the remaining lease; never trust their age.
        logger.warning(
            "Disabling Mooncake session refresh skipping: age=%s, TTL=%s, groups=%s",
            age,
            ttl,
            group_semantics,
        )
        return 0.0
    return age


@dataclass(eq=False)
class _Batch:
    keys: tuple[str, ...]
    refs: Counter = field(default_factory=Counter)
    future: object = None
    lease_bases: dict[str, float] = field(default_factory=dict)
    shared: set[str] = field(default_factory=set)
    loading: Counter = field(default_factory=Counter)
    results: dict[str, str] = field(default_factory=dict)
    completed_at: float = 0.0


@dataclass
class _Request:
    owners: list
    submitted_at: float | None = None
    claimed: bool = False
    result: str | None = None
    ready_at: float = 0.0


class SessionPrefetch:
    """Keep native sessions alive until all waiting and loading users release.

    Retain a whole batch until its final key is released: native prefetch views
    can share one arena. Capacity is managed by Mooncake's native arena.
    """

    def __init__(
        self,
        store,
        *,
        workers=1,
        max_requests=12,
        refresh_age_s=0.0,
        enable_metrics=False,
    ):
        if workers < 1 or max_requests < 1:
            raise ValueError("Mooncake prefetch limits must be positive.")
        if not math.isfinite(refresh_age_s) or refresh_age_s < 0:
            raise ValueError("Mooncake session refresh age must be finite and >= 0.")
        for name in ("batch_get_session_start", "batch_get_session_end"):
            if not callable(getattr(store, name, None)):
                raise ValueError(f"Mooncake is missing {name}.")
        self.store = store
        self.max_requests = max_requests
        self.refresh_age_s = refresh_age_s
        self.trace_enabled = os.environ.get("MC_STORE_DFS_READ_TRACE", "0").lower() in (
            "1",
            "true",
            "on",
        )
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mooncake-prefetch"
        )
        self.lock = threading.RLock()
        self.keys = {}
        self.requests = {}
        self.closed = False
        self.stats = Counter()
        self.metric_registry = None
        if enable_metrics:
            from prometheus_client import CollectorRegistry, Histogram

            # Export through the existing client snapshot, not the worker registry.
            self.metric_registry = CollectorRegistry()
            buckets = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 300)
            self.duration = Histogram(
                "mooncake_direct_prefetch_duration_seconds",
                "Request admission to session/prefetch preparation or cancellation",
                ("result",),
                buckets=buckets,
                registry=self.metric_registry,
            )
            self.ready_lead = Histogram(
                "mooncake_direct_prefetch_ready_lead_seconds",
                "Completed native prefetch to load-worker claim; excludes unready claims",
                buckets=buckets,
                registry=self.metric_registry,
            )

    def _prepare(self, batch, prefetch):
        try:
            return self._prepare_sessions(batch, prefetch)
        except BaseException:
            if self.metric_registry is not None:
                batch.results = dict.fromkeys(batch.keys, "failed")
            raise
        finally:
            if self.metric_registry is not None:
                batch.completed_at = time.monotonic()

    def _prepare_sessions(self, batch, prefetch):
        trace_start = time.perf_counter() if self.trace_enabled else None
        keys = list(batch.keys)
        start_with_sources = getattr(
            self.store, "batch_get_session_start_with_sources", None
        )
        # Include the query RPC and prefetch time in the lease's age.
        lease_base = time.monotonic()
        if callable(start_with_sources):
            results, sources = start_with_sources(keys)
            results, sources = list(results), list(sources)
            if len(sources) != len(keys):
                raise RuntimeError(
                    "Mooncake session start returned an invalid source count."
                )
        else:
            results = list(self.store.batch_get_session_start(keys))
            sources = ["unknown"] * len(keys)
        if len(results) != len(keys):
            raise RuntimeError("Mooncake session start returned an invalid count.")
        if self.refresh_age_s:
            with self.lock:
                batch.lease_bases.update(
                    (key, lease_base)
                    for key, result in zip(keys, results)
                    if result == 0 and key not in batch.shared
                )
        good = [
            key
            for key, result, source in zip(keys, results, sources)
            if result == 0 and source in ("dfs", "unknown")
        ]
        if trace_start is not None:
            logger.info(
                "Mooncake linker trace: stage=session_start keys=%d prefetch=%s sources=%s ms=%.3f",
                len(keys),
                prefetch,
                dict(Counter(sources)),
                (time.perf_counter() - trace_start) * 1000,
            )
            trace_start = time.perf_counter()
        if self.metric_registry is not None:
            batch.results = {
                key: (
                    "failed"
                    if result != 0 or (prefetch and source in ("dfs", "unknown"))
                    else "no_prefetch_needed"
                    if prefetch
                    else "session_only"
                )
                for key, result, source in zip(keys, results, sources)
            }
        if prefetch and good:
            try:
                copied = list(self.store.batch_get_session_prefetch(good))
                if self.metric_registry is not None and len(copied) == len(good):
                    batch.results.update(
                        (key, "prefetched" if result == 0 else "failed")
                        for key, result in zip(good, copied)
                    )
                with self.lock:
                    self.stats["prefetched_keys"] += sum(x == 0 for x in copied)
                    self.stats["prefetch_failed_keys"] += len(good) - sum(
                        x == 0 for x in copied
                    )
            except Exception:
                with self.lock:
                    self.stats["prefetch_errors"] += 1
                # A speculative read failure must not prevent normal recovery.
                logger.warning(
                    "Mooncake host prefetch failed; using normal reads", exc_info=True
                )
        if trace_start is not None and prefetch and good:
            logger.info(
                "Mooncake linker trace: stage=host_prefetch keys=%d ms=%.3f",
                len(good),
                (time.perf_counter() - trace_start) * 1000,
            )
        return dict(zip(keys, results))

    def _retain(self, keys, prefetch):
        new = tuple(key for key in keys if key not in self.keys)
        if new:
            batch = _Batch(new)
            for key in new:
                self.keys[key] = batch
            batch.future = self.executor.submit(self._prepare, batch, prefetch)
        owners = []
        for key in keys:
            batch = self.keys[key]
            if self.refresh_age_s and prefetch and batch.refs[key]:
                batch.shared.add(key)
                batch.lease_bases.pop(key, None)
            batch.refs[key] += 1
            owners.append((key, batch))
        return owners

    def submit(self, rid, keys):
        """Admit a speculative request; false means ordinary loads still work."""
        keys = tuple(dict.fromkeys(keys))
        with self.lock:
            previous = self.requests.get(rid)
            if previous is not None and keys == tuple(k for k, _ in previous.owners):
                self.stats["reused_requests"] += 1
                return True
            self.release(rid)
            if self.closed or not keys or len(self.requests) >= self.max_requests:
                self.stats["not_admitted"] += 1
                reason = "closed" if self.closed else "empty" if not keys else "limit"
                self.stats[f"rejected_{reason}"] += 1
                return False
            submitted_at = (
                time.monotonic() if self.metric_registry is not None else None
            )
            request = _Request(self._retain(keys, True), submitted_at)
            self.requests[rid] = request
            self.stats["submitted"] += 1
            return True

    def _observe_request(self, request):
        with self.lock:
            if request.submitted_at is None or request.result is not None:
                return
            batches = {batch for _, batch in request.owners}
            if not all(batch.future.done() for batch in batches):
                return
            # Only this request's keys determine its outcome in a shared batch.
            results = {
                batch.results.get(key, "failed") for key, batch in request.owners
            }
            if "failed" in results:
                result = "failed"
            elif "session_only" in results:
                result = "session_only"
            elif "prefetched" in results:
                result = "prefetched"
            else:
                result = "no_prefetch_needed"
            request.result = result
            # A new owner may reuse a batch that finished before it was admitted.
            request.ready_at = max(
                request.submitted_at, max(batch.completed_at for batch in batches)
            )
            self.duration.labels(result=result).observe(
                request.ready_at - request.submitted_at
            )

    def claim(self, rid):
        """Record the first load-worker claim, not successful device consumption."""
        with self.lock:
            request = self.requests.get(rid)
            if request is None or request.submitted_at is None or request.claimed:
                return
            self._observe_request(request)
            request.claimed = True
            self.stats["claimed"] += 1
            if request.result == "prefetched":
                self.ready_lead.observe(time.monotonic() - request.ready_at)
            else:
                self.stats["claimed_without_ready_prefetch"] += 1

    def release(self, rid):
        with self.lock:
            request = self.requests.pop(rid, None)
            if request is None:
                return
            self.stats["released_requests"] += 1
            self._observe_request(request)
            if request.submitted_at is not None:
                if request.result is None:
                    request.result = "cancelled"
                    self.duration.labels(result="cancelled").observe(
                        time.monotonic() - request.submitted_at
                    )
                elif not request.claimed and request.result == "prefetched":
                    self.stats["discarded_ready"] += 1
            self._release(request.owners)

    def _release(self, owners):
        batches = set()
        for key, batch in owners:
            batch.refs[key] -= 1
            batches.add(batch)
        for batch in batches:
            if not any(batch.refs.values()):
                # Keep the key mapping until cleanup completes. A new owner may
                # retain this same session while the worker is still reading.
                batch.future.add_done_callback(lambda _, b=batch: self._cleanup(b))

    def _cleanup(self, batch):
        with self.lock:
            if any(batch.refs.values()) or not any(
                self.keys.get(key) is batch for key in batch.keys
            ):
                return
            try:
                result = self.store.batch_get_session_end(list(batch.keys))
                if result not in (0, None):
                    raise RuntimeError(f"Mooncake session end failed: {result}")
            except Exception:
                # Retain ownership after a failed end so later users reuse the
                # same session instead of starting an untracked replacement.
                batch.lease_bases.clear()
                self.stats["cleanup_errors"] += 1
                logger.exception("Mooncake session cleanup failed")
                return
            for key in batch.keys:
                if self.keys.get(key) is batch:
                    del self.keys[key]

    @contextmanager
    def acquire(self, keys):
        """Hold sessions through the caller's synchronous device-copy boundary."""
        trace_start = time.perf_counter() if self.trace_enabled else None
        keys = tuple(dict.fromkeys(keys))
        with self.lock:
            if self.closed:
                raise RuntimeError("Mooncake session manager is closed.")
            owners = self._retain(keys, False)
            if self.refresh_age_s:
                for key, batch in owners:
                    if batch.loading[key]:
                        batch.shared.add(key)
                        batch.lease_bases.pop(key, None)
                    batch.loading[key] += 1
        trace_wait_start = time.perf_counter() if trace_start is not None else None
        try:
            for key, batch in owners:
                if batch.future.result()[key] != 0:
                    raise RuntimeError(f"Mooncake session start failed for {key}")
            trace_wait_ms = (
                (time.perf_counter() - trace_wait_start) * 1000
                if trace_wait_start is not None
                else 0.0
            )
            trace_refresh_ms = 0.0
            refresh = getattr(self.store, "batch_get_session_refresh", None)
            if keys and callable(refresh):
                with self.lock:
                    now = time.monotonic()
                    needs_refresh = any(
                        key in batch.shared
                        or key not in batch.lease_bases
                        or not 0 <= now - batch.lease_bases[key] < self.refresh_age_s
                        for key, batch in owners
                    )
                    if not needs_refresh:
                        self.stats["refresh_skipped_keys"] += len(keys)
                if needs_refresh:
                    # Refresh the whole batch: a slow partial refresh could
                    # outlive a lease that was young before the RPC started.
                    lease_base = time.monotonic()
                    try:
                        trace_refresh_start = (
                            time.perf_counter() if trace_start is not None else None
                        )
                        results = list(refresh(list(keys)))
                        if trace_refresh_start is not None:
                            trace_refresh_ms = (
                                time.perf_counter() - trace_refresh_start
                            ) * 1000
                        if len(results) != len(keys) or any(x != 0 for x in results):
                            raise RuntimeError(
                                f"Mooncake session refresh failed: {results}"
                            )
                    except BaseException:
                        with self.lock:
                            for key, batch in owners:
                                batch.lease_bases.pop(key, None)
                        raise
                    with self.lock:
                        for key, batch in owners:
                            if self.refresh_age_s and key not in batch.shared:
                                batch.lease_bases[key] = lease_base
                        self.stats["refreshed_keys"] += len(keys)
            if trace_start is not None:
                logger.info(
                    "Mooncake linker trace: stage=session_acquire keys=%d future_wait_ms=%.3f refresh_ms=%.3f ms=%.3f",
                    len(keys),
                    trace_wait_ms,
                    trace_refresh_ms,
                    (time.perf_counter() - trace_start) * 1000,
                )
            yield
        finally:
            with self.lock:
                if self.refresh_age_s:
                    for key, batch in owners:
                        batch.loading[key] -= 1
                self._release(owners)

    def reset(self):
        with self.lock:
            for rid in list(self.requests):
                self.release(rid)
            batches = set(self.keys.values())
        for batch in batches:
            try:
                batch.future.result()
            except Exception:
                pass
            self._cleanup(batch)

    def metrics(self):
        with self.lock:
            for request in self.requests.values():
                self._observe_request(request)
            ready = sum(
                request.result == "prefetched" and not request.claimed
                for request in self.requests.values()
            )
            lines = [
                "# HELP mooncake_direct_prefetch_waiting_requests Waiting/loading requests retaining native sessions",
                "# TYPE mooncake_direct_prefetch_waiting_requests gauge",
                f"mooncake_direct_prefetch_waiting_requests {len(self.requests)}",
                "# HELP mooncake_direct_prefetch_ready_requests Unclaimed requests with completed native prefetch",
                "# TYPE mooncake_direct_prefetch_ready_requests gauge",
                f"mooncake_direct_prefetch_ready_requests {ready}",
                "# HELP mooncake_direct_prefetch_events_total Session admission and API results, not DFS-consumption bytes",
                "# TYPE mooncake_direct_prefetch_events_total counter",
            ]
            lines.extend(
                f'mooncake_direct_prefetch_events_total{{event="{name}"}} {value}'
                for name, value in sorted(self.stats.items())
            )
            text = "\n".join(lines) + "\n"
            if self.metric_registry is not None:
                from prometheus_client import generate_latest

                text += generate_latest(self.metric_registry).decode()
            return text

    def close(self):
        with self.lock:
            self.closed = True
        self.reset()
        self.executor.shutdown(wait=True)
        logger.info("Mooncake prefetch stats: %s", dict(self.stats))
