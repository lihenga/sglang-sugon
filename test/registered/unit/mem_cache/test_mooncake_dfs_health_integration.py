"""Exercise health capability negotiation and the Store/cache wiring on CPU."""

import argparse
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.arg_groups.fields.memory import Memory
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.mooncake_store.compat import supports_dfs_health
from sglang.srt.mem_cache.storage.mooncake_store.dfs_health import (
    MooncakeDfsHealthManager,
    resolve_dfs_health_config,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class NativeStore:
    def __init__(self):
        self.policies = []

    def get_capabilities(self):
        return {"features": {"dfs_health": True}}

    def setup(self, *args, **kwargs):
        return 0

    def probe_dfs_read(self):
        return 0

    def probe_dfs_write(self):
        return 0

    def set_dfs_access_policy(self, read, write):
        self.policies.append((read, write))
        return 0


def wait_for_policy(store, expected):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if store.policies[-1] == expected:
            return
        time.sleep(0.01)
    raise AssertionError(f"Policy not applied: {store.policies}, expected {expected}")


@pytest.mark.parametrize("declaration", [None, {}, {"features": {"dfs_health": True}}])
def test_complete_legacy_or_declared_api_is_supported(declaration):
    store = NativeStore()
    store.get_capabilities = None if declaration is None else lambda: declaration
    assert supports_dfs_health(store)


def test_explicit_false_and_incomplete_api_are_not_supported():
    store = NativeStore()
    store.get_capabilities = lambda: {"features": {"dfs_health": False}}
    assert not supports_dfs_health(store)
    store.get_capabilities = lambda: {"features": {"dfs_health": True}}
    store.probe_dfs_write = None
    assert not supports_dfs_health(store)


def test_memory_namespace_keeps_environment_and_cli_precedence(monkeypatch):
    monkeypatch.setenv("SGLANG_MOONCAKE_DFS_HEALTH_PROBE_TIMEOUT_S", "7")
    config = Memory()
    assert resolve_dfs_health_config(server_args=config)["probe_timeout_s"] == 7
    config.mooncake_dfs_health_probe_timeout_s = 9
    assert resolve_dfs_health_config(server_args=config)["probe_timeout_s"] == 9
    assert (
        resolve_dfs_health_config({"probe_timeout_s": 11}, config)["probe_timeout_s"]
        == 11
    )


def test_health_options_are_registered_on_the_launch_cli():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        [
            "--model-path",
            "unused",
            "--mooncake-dfs-health-probe-interval-s",
            "4",
            "--mooncake-dfs-health-failure-threshold",
            "2",
            "--mooncake-dfs-health-success-threshold",
            "5",
            "--mooncake-dfs-health-probe-timeout-s",
            "7",
        ]
    )
    assert parsed.mooncake_dfs_health_probe_interval_s == 4
    assert parsed.mooncake_dfs_health_failure_threshold == 2
    assert parsed.mooncake_dfs_health_success_threshold == 5
    assert parsed.mooncake_dfs_health_probe_timeout_s == 7


@pytest.mark.parametrize(
    "tp_size,cp_size,pp_size", [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)]
)
def test_store_fails_closed_before_warmup_and_defers_rank_policy(
    monkeypatch, tp_size, cp_size, pp_size
):
    monkeypatch.setenv(
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "distributed_storage_backend"
    )
    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=cp_size,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1 << 20,
            "_mooncake_dfs_health_config": {"success_threshold": 1},
        },
    )

    def warmup(store):
        assert store.store.policies == [(False, False)]
        assert store._dfs_health_manager._thread is None

    with (
        patch.object(MooncakeStore, "_import_mooncake_store", return_value=NativeStore),
        patch.object(MooncakeStore, "warmup", warmup),
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(MooncakeDfsHealthManager, "start") as start,
    ):
        store = MooncakeStore(config)
    manager = store._dfs_health_manager
    start.assert_called_once()
    assert manager._defer_policy_application == (
        tp_size > 1 or cp_size > 1 or pp_size > 1
    )
    manager.probe_once()
    assert store.get_local_dfs_health_policy() == (True, True)
    assert store.store.policies[-1] == (
        (False, False) if manager._defer_policy_application else (True, True)
    )
    store.apply_dfs_health_policy(False, True)
    wait_for_policy(store.store, (False, True))
    store.close()
    assert manager._stop.is_set()
    assert store.get_local_dfs_health_policy() is None


def test_legacy_package_without_health_keeps_existing_behavior(monkeypatch):
    monkeypatch.setenv(
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "distributed_storage_backend"
    )

    class LegacyStore(NativeStore):
        def get_capabilities(self):
            return {"features": {"dfs_health": False}}

    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
        },
    )
    with (
        patch.object(MooncakeStore, "_import_mooncake_store", return_value=LegacyStore),
        patch.object(MooncakeStore, "warmup"),
        patch.object(MooncakeDfsHealthManager, "start") as start,
    ):
        store = MooncakeStore(config)
    start.assert_not_called()
    assert store.get_local_dfs_health_policy() is None
    assert store.store.policies == []
    store.close()


@pytest.mark.parametrize("health_supported", [True, False])
def test_dfs_only_startup_does_not_write_before_health_opens(
    monkeypatch, health_supported
):
    monkeypatch.setenv(
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "distributed_storage_backend"
    )

    class Store(NativeStore):
        def get_capabilities(self):
            return {"features": {"dfs_health": health_supported}}

    config = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=2,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=None,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "memory_replica_num": 0,
            "dfs_replica_num": 1,
            "_mooncake_dfs_health_config": {"success_threshold": 1},
        },
    )
    with (
        patch.object(MooncakeStore, "_import_mooncake_store", return_value=Store),
        patch.object(MooncakeStore, "warmup") as warmup,
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(MooncakeDfsHealthManager, "start"),
    ):
        store = MooncakeStore(config)
    if health_supported:
        warmup.assert_not_called()
        assert store.store.policies == [(False, False)]
        store._dfs_health_manager.probe_once()
        assert store.get_local_dfs_health_policy() == (True, True)
        # A local success must still wait for the attention-rank verdict.
        assert store.store.policies == [(False, False)]
        store.apply_dfs_health_policy(True, True)
        wait_for_policy(store.store, (True, True))
    else:
        warmup.assert_called_once()
        assert store.store.policies == []
    store.close()


@pytest.mark.parametrize("use_linker", [True, False])
def test_cache_applies_attention_group_minimum(use_linker):
    applied = []
    owner = SimpleNamespace(
        get_local_dfs_health_policy=lambda: (True, True),
        apply_dfs_health_policy=lambda read, write: applied.append((read, write)),
    )
    if use_linker:
        wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
        wrapper.cache_linker = owner
        owner = wrapper
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = owner if use_linker else None
    cache.cache_controller = SimpleNamespace(storage_backend=owner)
    cache.attn_cp_group, cache.attn_tp_group = object(), object()
    cache.tp_world_size = 2
    groups = []

    def reduce(tensor, *, op, group):
        assert op == torch.distributed.ReduceOp.MIN
        groups.append(group)
        if tensor.tolist() == [1, -1]:
            return
        if group is cache.attn_cp_group:
            tensor[0] = 0
        else:
            tensor[1] = 0

    with (
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_reduce", side_effect=reduce),
    ):
        cache._sync_dfs_health_policy()
    assert groups == [cache.attn_cp_group, cache.attn_tp_group] * 2
    assert applied == [(False, False)]


def test_backends_without_health_only_check_capability_once():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = None
    cache.cache_controller = SimpleNamespace(storage_backend=object())
    cache._all_reduce_attn_groups = Mock()
    cache._sync_dfs_health_policy()
    cache._all_reduce_attn_groups.reset_mock()
    cache._sync_dfs_health_policy()
    cache._all_reduce_attn_groups.assert_not_called()


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize(
    "operation,code", [("write", -1606), ("read", -1607), ("read", -1608)]
)
def test_business_io_timeouts_report_health_but_admission_backpressure_does_not(
    multi, operation, code
):
    store = MooncakeStore.__new__(MooncakeStore)
    store._dfs_health_manager = Mock()
    store._use_group_semantics = False
    store.store = Mock()
    store.store.batch_put_from.return_value = [code]
    store.store.batch_put_from_multi_buffers.return_value = [code]
    store.store.batch_get_into.return_value = [code]
    store.store.batch_get_into_multi_buffers.return_value = [code]
    store._replicate_config_cls = SimpleNamespace
    buffers, sizes = ([[123]], [[4]]) if multi else ([123], [4])
    method = (
        store._put_batch_zero_copy_impl
        if operation == "write"
        else store._get_batch_zero_copy_impl
    )
    assert method(["key"], buffers, sizes) == [code]
    report = store._dfs_health_manager.report_operation_timeout
    if code == -1608:
        report.assert_not_called()
    else:
        report.assert_called_once_with(operation, code)


def test_controller_retains_backend_when_native_close_is_incomplete():
    from sglang.srt.managers.cache_controller import HiCacheController

    controller = HiCacheController.__new__(HiCacheController)
    controller._stop_storage_threads = Mock()
    controller._destroy_sync_groups = Mock()
    owner = Mock()
    owner.close.side_effect = RuntimeError("native work pending")
    controller.storage_backend = owner
    controller.enable_storage = True
    controller.prefetch_hits_sync_groups = [object()]
    controller.prefetch_completion_sync_groups = [object()]
    controller.storage_stop_event = Mock()
    with pytest.raises(RuntimeError, match="native work pending"):
        controller.detach_storage_backend()
    assert controller.storage_backend is owner
    assert not controller.enable_storage
    controller._destroy_sync_groups.assert_not_called()

    owner.close.side_effect = None
    controller.detach_storage_backend()
    assert controller.storage_backend is None
    assert not controller.enable_storage


def test_failed_health_close_disables_cache_storage_admission():
    from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment

    controller = SimpleNamespace(enable_storage=True)

    def detach():
        controller.enable_storage = False
        raise RuntimeError("native work pending")

    controller.detach_storage_backend = detach
    cache = SimpleNamespace(
        cache_controller=controller,
        enable_storage=True,
        enable_storage_metrics=True,
        drain_storage_control_queues_local=Mock(),
    )
    ok, reason = StorageAttachment(cache).detach()
    assert not ok
    assert "native work pending" in reason
    assert not cache.enable_storage
    assert not cache.enable_storage_metrics
