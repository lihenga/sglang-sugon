"""Bounded host prefetch with shared ownership of native key sessions."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _Batch:
    keys: tuple[str, ...]
    refs: Counter = field(default_factory=Counter)
    future: object = None


class SessionPrefetch:
    """Keep native sessions alive until all waiting and loading users release.

    Retain a whole batch until its final key is released: native prefetch views
    can share one arena. Capacity is managed by Mooncake's native arena.
    """

    def __init__(self, store, *, workers=1, max_requests=12):
        if workers < 1 or max_requests < 1:
            raise ValueError("Mooncake prefetch limits must be positive.")
        for name in ("batch_get_session_start", "batch_get_session_end"):
            if not callable(getattr(store, name, None)):
                raise ValueError(f"Mooncake is missing {name}.")
        self.store = store
        self.max_requests = max_requests
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="mooncake-prefetch"
        )
        self.lock = threading.RLock()
        self.keys = {}
        self.requests = {}
        self.closed = False
        self.stats = Counter()

    def _prepare(self, keys, prefetch):
        start_with_sources = getattr(
            self.store, "batch_get_session_start_with_sources", None
        )
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
            batch.future = self.executor.submit(self._prepare, list(new), prefetch)
        owners = []
        for key in keys:
            batch = self.keys[key]
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
        try:
            for key, batch in owners:
                if batch.future.result()[key] != 0:
                    raise RuntimeError(f"Mooncake session start failed for {key}")
            refresh = getattr(self.store, "batch_get_session_refresh", None)
            if keys and callable(refresh):
                results = list(refresh(list(keys)))
                if len(results) != len(keys) or any(result != 0 for result in results):
                    raise RuntimeError(f"Mooncake session refresh failed: {results}")
            yield
        finally:
            with self.lock:
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
