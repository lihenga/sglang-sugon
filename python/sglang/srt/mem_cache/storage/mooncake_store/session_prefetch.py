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


class SessionPrefetch:
    """Keep native sessions alive until all waiting and loading users release.

    Retain a whole batch until its final key is released: native prefetch views
    can share one arena. Capacity is managed by Mooncake's native arena.
    """

    def __init__(self, store, *, workers=1, max_requests=12, refresh_age_s=0.0):
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
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mooncake-prefetch"
        )
        self.lock = threading.RLock()
        self.keys = {}
        self.requests = {}
        self.closed = False
        self.stats = Counter()

    def _prepare(self, batch, prefetch):
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
        if prefetch and good:
            try:
                copied = list(self.store.batch_get_session_prefetch(good))
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
            if previous is not None and keys == tuple(k for k, _ in previous):
                return True
            self.release(rid)
            if self.closed or not keys or len(self.requests) >= self.max_requests:
                self.stats["not_admitted"] += 1
                return False
            self.requests[rid] = self._retain(keys, True)
            self.stats["submitted"] += 1
            return True

    def release(self, rid):
        with self.lock:
            owners = self.requests.pop(rid, [])
            if owners:
                self.stats["released_requests"] += 1
            self._release(owners)

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
        try:
            for key, batch in owners:
                if batch.future.result()[key] != 0:
                    raise RuntimeError(f"Mooncake session start failed for {key}")
            refresh = getattr(self.store, "batch_get_session_refresh", None)
            if keys and callable(refresh):
                with self.lock:
                    now = time.monotonic()
                    stale = [
                        (key, batch)
                        for key, batch in owners
                        if key in batch.shared
                        or key not in batch.lease_bases
                        or not 0 <= now - batch.lease_bases[key] < self.refresh_age_s
                    ]
                    self.stats["refresh_skipped_keys"] += len(keys) - len(stale)
                if stale:
                    lease_base = time.monotonic()
                    try:
                        results = list(refresh([key for key, _ in stale]))
                        if len(results) != len(stale) or any(x != 0 for x in results):
                            raise RuntimeError(
                                f"Mooncake session refresh failed: {results}"
                            )
                    except BaseException:
                        with self.lock:
                            for key, batch in stale:
                                batch.lease_bases.pop(key, None)
                        raise
                    with self.lock:
                        for key, batch in stale:
                            if self.refresh_age_s and key not in batch.shared:
                                batch.lease_bases[key] = lease_base
                        self.stats["refreshed_keys"] += len(stale)
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
            lines = [
                "# HELP mooncake_direct_prefetch_waiting_requests Waiting requests retaining native sessions",
                "# TYPE mooncake_direct_prefetch_waiting_requests gauge",
                f"mooncake_direct_prefetch_waiting_requests {len(self.requests)}",
                "# HELP mooncake_direct_prefetch_events_total Session admission and API results, not DFS-consumption bytes",
                "# TYPE mooncake_direct_prefetch_events_total counter",
            ]
            lines.extend(
                f'mooncake_direct_prefetch_events_total{{event="{name}"}} {value}'
                for name, value in sorted(self.stats.items())
            )
            return "\n".join(lines) + "\n"

    def close(self):
        with self.lock:
            self.closed = True
        self.reset()
        self.executor.shutdown(wait=True)
        logger.info("Mooncake prefetch stats: %s", dict(self.stats))
