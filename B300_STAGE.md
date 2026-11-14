# B300 migration: DCU validation stage

This branch adds the missing direct-cache behaviors to SGLang v0.5.23 while
preserving the upstream hybrid-pool interface. It is not a B300 performance
certification. The current validation uses HIP/DCU and POSIX DFS on one host.

## Direct Mooncake options

```bash
--enable-unified-cache-external-linker \
--mooncake-enable-page-wise-load \
--mooncake-enable-waiting-queue-dfs-prefetch \
--mooncake-waiting-queue-dfs-prefetch-workers 1 \
--mooncake-waiting-queue-dfs-prefetch-max-requests 12 \
--mooncake-waiting-queue-dfs-prefetch-max-bytes 536870912
```

All new behaviors are opt-in. Page-wise loading restores every physical buffer
of each page before publishing readiness. The normal layer-wise path remains
available. Waiting-queue prefetch starts only after the existing cross-rank
restorable-prefix intersection; admission failures use the normal load path.

`SGLANG_MOONCAKE_READ_PLAN=1` uses the native `create_read_plan` interface.
The layer counter delegates readiness to the native plan, retaining its device
buffer owners until completion. A missing requested native capability fails at
startup instead of silently ignoring the option. Session refresh on admission
uses the native API when available and preserves its read-view checks.

The prefetch manager retains a shared key session until its waiting and loading
users release it. Cancellation cannot end an in-flight prefetch. The whole
submitted allocation batch remains charged until its final key is released.
The Python byte limit counts aligned payload; native unified-buffer allocator
rounding and admission limits also apply. It is not a process RSS limit.

## Store configuration

`SGLANG_MOONCAKE_LOCAL_BUFFER_SIZE` sets the local fallback buffer for each new
embedded client. The default remains 16MiB. Negative or malformed values fail.
Account for every rank's local buffer, unified DFS buffer, model pools and other
allocations when sizing a machine.

For an explicitly DFS-only test namespace:

```bash
--unified-cache-external-linker-config \
'{"memory_replica_num":0,"dfs_replica_num":1,"extra_backend_tag":"isolated-model-revision-layout"}'
```

Without explicit replica counts, the existing native defaults are preserved.
Warmup uses the same requested replica policy and waits for asynchronous DFS
publication. The storage master/client DFS configuration must still match.
Use a different backend tag for incompatible model revisions, tokenizers, KV
dtypes, page layouts or parallel topologies. This patch does not automatically
make persisted KV portable across those configurations.

## Models and supported boundaries

Plain unscaled MHA/MLA device pools now support direct linking. This path rejects
CP/DCP, scaled KV and packed draft pools; existing hybrid strategies retain their
own checks. K3's FULL+MAMBA strategy still requires plain MLA without DCP or
packed draft KV. Do not confuse ordinary HiCache's DCP support with these
direct-linker limits. Unified virtual memory remains excluded upstream.

The stage validates synthetic KV+state pools and real Qwen3-8B TP1/TP2 requests.
It does not validate a real Kimi checkpoint, KDA computation, CUDA kernels or
multi-node inference. An older AITER build can import unrelated BF16/cache code
because Quark schemes now load on demand; requesting an unsupported Quark
scheme still raises its actual dependency error.

## Metrics

With `--enable-metrics`, embedded client snapshots are forwarded to the existing
HTTP/HTTP-sidecar metrics endpoint. The receiver uses existing API-key and TLS
settings; reports include host, process, rank and client identity. A reporter
waits for receiver readiness before posting. Successful internal reports do not
produce access-log spam; errors remain visible.

The snapshot includes native Mooncake counters and:

- `mooncake_direct_prefetch_reserved_bytes`: aligned payload admission charge;
- `mooncake_direct_prefetch_waiting_requests`: requests retaining sessions;
- `mooncake_direct_prefetch_events_total`: admission and API outcomes.

These API outcomes do not represent successful consumption bytes. Use native
DFS/prefetch/H2D counters and request evidence to identify the actual source.

## Reproducing this stage

The isolated scripts, commands, native source identities, model results and logs
are under `/stortest/zhy_spcae/fixbug/b300-stage-20261114` in `guofy-4B` on
`10.211.11.28`. They do not overwrite the globally installed SGLang or Mooncake.
DCU model probes disable incompatible installed plugins and use the existing
pure-Torch fused-op fallback; their timings are functional observations only.

Future validation needs the actual Kimi config/revision, target CUDA stack,
two B300 nodes, the target IB/RoCE topology and the storage API. The current
single-host SHCA test is not evidence of inter-node or RoCE behavior.
