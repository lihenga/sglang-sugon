import pytest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStore,
    _local_buffer_size,
)


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
