# B300 direct-cache integration

This branch targets the single-host NVIDIA B300 environment and adds direct-cache
behaviors to SGLang v0.5.23 while preserving the upstream hybrid-pool interface.
It keeps the page restore, ReadPlan, session prefetch and client metrics changes.
The DCU-only Quark optional-import patch has been removed; quantization imports
match the upstream baseline.

## Direct Mooncake options

```bash
--enable-unified-cache-external-linker \
--mooncake-enable-page-wise-load \
--mooncake-enable-waiting-queue-dfs-prefetch \
--mooncake-waiting-queue-dfs-prefetch-workers 1 \
--mooncake-waiting-queue-dfs-prefetch-max-requests 12
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
submitted batch remains owned until its final key is released. SGLang limits
workers and waiting requests; Mooncake's native arena controls prefetch capacity
through `MC_STORE_DFS_PREFETCH_ARENA_SIZE_BYTES`. Native prefetch failures retain
normal read fallback; no Python byte budget is applied.

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

The current B300 container uses the Kimi-K3 MXFP4 checkpoint at
`/jsyw/models/Kimi-K3`, CUDA 13 and ordinary mlx5 RDMA devices. Build Mooncake
with CUDA, not HIP/HYGON or the SHCA ABI. The image
`lmsysorg/sglang:v0.5.22` initially contains
`mooncake-transfer-engine-cuda13==0.3.13.post1`, which lacks the native ReadPlan
and DFS session-prefetch APIs required by the corresponding options. Use a
matching build of `mooncake-das/v0.3.14_dev_l4_b300`; an older package importing
successfully does not make these capabilities available.

## Metrics

With `--enable-metrics`, embedded client snapshots are forwarded to the existing
HTTP/HTTP-sidecar metrics endpoint. The receiver uses existing API-key and TLS
settings; reports include host, process, rank and client identity. A reporter
waits for receiver readiness before posting. Successful internal reports do not
produce access-log spam; errors remain visible.

The snapshot includes native Mooncake counters and:

- `mooncake_direct_prefetch_waiting_requests`: requests retaining sessions;
- `mooncake_direct_prefetch_events_total`: admission and API outcomes.

These API outcomes do not represent successful consumption bytes. Use native
DFS/prefetch/H2D counters and request evidence to identify the actual source.

## B300 validation

Validation files belong under
`/jsyw/zhy_spcae/mooncake-b300-audit-20261116` on `GPU-10-130-10-8`.
The isolated container mounts this directory at `/audit`; temporary files,
compiler outputs and caches use that directory. Validation does not replace
packages in the user's `sglang-b300` container.

After removing the Quark patch, the three focused suites for page restore,
shared session prefetch and client metrics pass in the CUDA image:
38 tests and 22 subtests. These exercise Python behavior with synthetic pools
and test doubles; they do not establish native DFS, RDMA or model performance.

The matching Mooncake branch's accelerator sources were separately compiled
against the container's CUDA SDK. CUDA batch-copy correctness passed on all
eight B300 GPUs. The native Store/TE extensions and Master also compile and
link in the isolated environment. Real POSIX DFS tests using those extensions
and this SGLang source pass on CPU and all eight B300 GPUs, covering six restore
combinations per device: layer/page reads, prefetch and native ReadPlan, plus
missing-key rejection. The payloads are synthetic KV/state pools and transport
is TCP. A separate Mooncake RDMA-protocol check with an independent memory
provider restored data into CPU and all eight CUDA destinations on this host.
Neither result establishes Kimi inference, inter-node bandwidth or target
storage-side RoCE performance.

The running Kimi configuration uses TP8+DCP8 with HiCache disabled. Do not add
the direct-linker options above to it unchanged: the direct-linker pool layouts
still reject DCP/packed draft KV, independently of ordinary HiCache support.
Keep this limitation explicit when preparing a separate L4 experiment.
