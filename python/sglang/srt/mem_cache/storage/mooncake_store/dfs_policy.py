"""Coordinate optional DFS health policies before issuing native updates."""

import torch


class DfsHealthPolicySync:
    def __init__(self, reduce_policy):
        self._reduce = reduce_policy
        self._owner = None
        self._enabled = None

    def poll(self, owner) -> None:
        if owner is not self._owner:
            self._owner = owner
            self._enabled = None
        getter = getattr(owner, "get_local_dfs_health_policy", None)
        apply_policy = getattr(owner, "apply_dfs_health_policy", None)
        local = getter() if callable(getter) and callable(apply_policy) else None
        if self._enabled is None:
            available = int(local is not None)
            agreement = torch.tensor([available, -available], dtype=torch.int)
            self._reduce(agreement)
            if int(agreement[0]) != -int(agreement[1]):
                raise ValueError(
                    "Mooncake DFS health capability/configuration differs across ranks"
                )
            self._enabled = bool(agreement[0])
        if not self._enabled:
            return
        if local is None:
            raise RuntimeError("Mooncake DFS health manager disappeared")
        effective = torch.tensor(local, dtype=torch.int)
        self._reduce(effective)
        read, write = map(bool, effective.tolist())
        apply_policy(read and write, write)
