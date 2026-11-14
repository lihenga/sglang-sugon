import ctypes

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStore,
    _local_buffer_size,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LayerWiseLoadCounter


class HostStore:
    def __init__(self, objects):
        self.objects = objects
        self.reads = 0
        self.started = []
        self.ended = []

    def batch_get_session_start(self, keys):
        self.started.extend(keys)
        return [0 if key in self.objects else -1 for key in keys]

    def batch_get_session_end(self, keys):
        self.ended.extend(keys)
        return 0

    def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
        self.reads += 1
        for key, destinations, lengths, starts in zip(keys, ptrs, sizes, offsets):
            data = self.objects[key]
            for ptr, length, start in zip(destinations, lengths, starts):
                ctypes.memmove(ptr, data[start : start + length], length)
        return [sum(parts) for parts in sizes]


def make_linker(page_wise):
    buffers = [torch.zeros((4, 8), dtype=torch.uint8) for _ in range(3)]
    pool = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[buffers],
        layer_mapping={0: 0, 1: 1, 2: 2},
        page_size=1,
        rows_are_pages=True,
    )
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.pools = {PoolName.KV: pool}
    linker.num_layers = 3
    linker.page_wise_load = page_wise
    linker.sessions = None
    linker.storage = MooncakeStore.__new__(MooncakeStore)
    linker.storage.registered_pools = linker.pools
    linker.storage.mla_suffix = "tp0_cp0_pp0"
    linker.storage.config_prefix = None
    transfer = PoolTransfer(
        name=PoolName.KV, keys=["p0", "p1"], host_indices=torch.tensor([3, 1])
    )
    keys, _ = linker.storage._get_hybrid_page_component_keys(transfer.keys, transfer)
    linker.storage.store = HostStore(
        {
            keys[0]: bytes(range(24)),
            keys[1]: bytes(range(24, 48)),
        }
    )
    linker.layer_done_counter = LayerWiseLoadCounter(3)
    return linker, transfer, buffers


@pytest.mark.parametrize("page_wise", [False, True])
def test_page_and_layer_reads_restore_noncontiguous_destinations(page_wise):
    linker, transfer, buffers = make_linker(page_wise)
    counter = linker.layer_done_counter.update_producer()
    linker.load_layer_wise(counter, [[transfer]])
    assert linker.layer_done_counter.finish(counter)
    for layer, buf in enumerate(buffers):
        assert buf[3].tolist() == list(range(layer * 8, layer * 8 + 8))
        assert buf[1].tolist() == list(range(24 + layer * 8, 32 + layer * 8))
        assert not buf[0].any()
        assert not buf[2].any()
    assert linker.storage.store.reads == (1 if page_wise else 3)
    assert linker.storage.store.started == linker.storage.store.ended


def test_partial_start_failure_ends_successful_sessions():
    linker, transfer, _ = make_linker(True)
    linker.storage.store.objects.pop(next(iter(linker.storage.store.objects)))
    counter = linker.layer_done_counter.update_producer()
    linker.load_layer_wise(counter, [[transfer]])
    assert not linker.layer_done_counter.finish(counter)
    assert linker.storage.store.started == linker.storage.store.ended


def test_local_buffer_config_is_resolved_for_each_client(monkeypatch):
    monkeypatch.setenv("SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE", str(1 << 30))
    assert _local_buffer_size() == 1 << 30
    monkeypatch.setenv("SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE", "0")
    assert _local_buffer_size() == 0
    for value in ["-1", "garbage"]:
        monkeypatch.setenv("SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE", value)
        with pytest.raises(ValueError):
            _local_buffer_size()


def test_replica_policy_survives_group_and_session_attribution():
    class Config:
        replica_num = 1
        dfs_replica_num = 0

    class Native:
        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config):
            self.config = config
            return [0]

    storage = MooncakeStore.__new__(MooncakeStore)
    storage._replicate_config_cls = Config
    storage.write_replica_config = {"replica_num": 0, "dfs_replica_num": 1}
    storage._can_use_group_semantics = lambda: True
    storage.store = Native()
    assert storage._put_batch_zero_copy_impl(
        ["p"], [[1, 2]], [[8, 8]], ["group"], ["session"]
    ) == [0]
    config = storage.store.config
    assert config.replica_num == 0 and config.dfs_replica_num == 1
    assert config.group_ids == ["group"]
    assert config.session_ids == ["session"]
