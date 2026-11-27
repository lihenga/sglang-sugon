"""Negotiate optional DFS health APIs across Mooncake packages."""


def supports_dfs_health(store) -> bool:
    query = getattr(store, "get_capabilities", None)
    declared = query() if callable(query) else {}
    features = declared.get("features", {}) if isinstance(declared, dict) else {}
    if isinstance(features, dict) and features.get("dfs_health") is False:
        return False
    return all(
        callable(getattr(store, name, None))
        for name in ("probe_dfs_read", "probe_dfs_write", "set_dfs_access_policy")
    )
