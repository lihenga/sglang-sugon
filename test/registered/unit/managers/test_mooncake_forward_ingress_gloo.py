"""Real CPU collectives while a different rank's forward is still running."""

import sys
import threading
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as module
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components import request_receiver as receiver_module
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.mem_cache.storage.mooncake_store.session_prefetch import SessionPrefetch

register_cpu_ci(est_time=300, suite="base-a-test-cpu")


def _rank(rank, rendezvous, layout):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    singleton_groups = [dist.new_group([i], backend="gloo") for i in range(2)]
    singleton = singleton_groups[rank]
    world = dist.group.WORLD
    cp_mode = layout == "cp"

    def coord(group, ranks):
        return SimpleNamespace(
            cpu_group=group,
            ranks=ranks,
            rank=rank,
            world_size=len(ranks),
            rank_in_group=ranks.index(rank),
        )

    tp = coord(world, [0, 1])
    cp = coord(world if cp_mode else singleton, [0, 1] if cp_mode else [rank])
    attn_tp = coord(singleton if cp_mode else world, [rank] if cp_mode else [0, 1])
    parallel = SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        tp_size=2,
        tp_group=tp,
        world_group=tp,
        attn_dp_enabled=cp_mode,
        attn_dp_rank=0,
        attn_cp_group=cp,
        attn_tp_group=attn_tp,
        attn_cp_rank=cp.rank_in_group,
        attn_tp_rank=attn_tp.rank_in_group,
        attn_cp_size=cp.world_size,
        attn_tp_size=attn_tp.world_size,
        enable_dp_attention_local_control_broadcast=True,
    )
    release, prefetched = threading.Event(), threading.Event()
    store = SimpleNamespace(
        batch_get_session_start=lambda keys: [0] * len(keys),
        batch_get_session_prefetch=lambda keys: prefetched.set() or [0] * len(keys),
        batch_get_session_end=lambda keys: [0] * len(keys),
    )
    sessions = SessionPrefetch(store)
    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.enable_overlap = layout == "cp"
    s.device_module = SimpleNamespace(Event=Mock, StreamContext=lambda _: nullcontext())
    s.forward_stream, s.schedule_stream = Mock(), Mock()
    s.is_generation = True
    s.attn_dp_enabled = cp_mode
    s.enable_pdmux = False
    s.spec_algorithm = SimpleNamespace(
        is_none=lambda: not s.enable_overlap, is_dspark=lambda: s.enable_overlap
    )
    s.schedule_policy = "fcfs"
    s.disaggregation_mode = DisaggregationMode.NULL
    s.mm_receiver = s.input_blocker = s.scripted_scheduler_hook = None
    s.tp_cpu_group, s.attn_cp_cpu_group, s.attn_tp_cpu_group = (
        world,
        cp.cpu_group,
        attn_tp.cpu_group,
    )
    s.tree_cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=SimpleNamespace(prefetch_enabled=True))
    )
    s._deferred_input_requests = []
    s.request_receiver = SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        server_args=None,
        model_config=None,
        max_recv_per_poll=-1,
        stream_output=None,
        get_last_batch=lambda: None,
    )
    try:
        with (
            patch.object(module, "get_parallel", return_value=parallel),
            patch.object(receiver_module, "get_parallel", return_value=parallel),
            patch.object(
                module,
                "get_memory",
                return_value=SimpleNamespace(
                    unified_cache_external_linker_backend="mooncake"
                ),
            ),
            patch.object(
                module,
                "get_observability",
                return_value=SimpleNamespace(kv_canary="none"),
            ),
            envs.SGLANG_MOONCAKE_FORWARD_INGRESS.override(
                not (layout == "mixed" and rank == 1)
            ),
        ):
            s.init_forward_ingress()
            if layout == "mixed":
                assert not s._forward_ingress_enabled
                assert s._forward_launch_executor is None
                return
            assert s._forward_ingress_enabled
            assert s.request_receiver.tp_cpu_group is not world
            assert s.tree_cache.attn_cp_group is not cp.cpu_group
            assert s.tree_cache.attn_tp_group is not attn_tp.cpu_group
            receiver = s.request_receiver
            sent = False
            owner = threading.get_ident()

            def recv():
                nonlocal sent
                reqs = [] if rank == 0 else None
                if rank == 0 and not sent:
                    sent = True
                    reqs.append(
                        TokenizedGenerateReqInput(
                            rid="new",
                            input_text=None,
                            input_ids=None,
                            input_embeds=None,
                            mm_inputs=None,
                            token_type_ids=None,
                            sampling_params=SimpleNamespace(beam_width=1),
                            return_logprob=False,
                            logprob_start_len=-1,
                            top_logprobs_num=0,
                            token_ids_logprob=None,
                            stream=False,
                        )
                    )
                return receiver._broadcast_reqs_across_ranks(reqs)

            seen = []

            def prepare(req):
                assert threading.get_ident() == owner
                cache_req = SimpleNamespace(
                    session=None,
                    to_finish=None,
                    init_next_round_input=lambda *a, **k: sessions.submit(
                        req.rid, ["key"]
                    ),
                )
                Scheduler._prefetch_kvcache(s, cache_req)
                assert prefetched.wait(5)
                seen.append(req.rid)
                release.set()

            def forward(batch):
                # Original model communicator stays usable concurrently with ingress.
                value = torch.tensor([rank + 1])
                dist.all_reduce(value, group=world)
                assert value.item() == 3
                if rank == 1:
                    assert release.wait(10)
                return "done"

            s.request_receiver = SimpleNamespace(recv_requests=recv)
            s.handle_generate_request = prepare
            s.model_worker = SimpleNamespace(forward_batch_generation=forward)
            assert s._forward_with_waiting_queue_ingress(None) == "done"
            assert seen == ["new"]
    finally:
        if s._forward_launch_executor is not None:
            s._forward_launch_executor.shutdown(wait=True)
        sessions.close()
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="Validated on Linux CPU Gloo")
@pytest.mark.parametrize("layout", ["tp", "cp", "mixed"])
def test_gloo_forward_ingress(tmp_path, layout):
    mp.start_processes(
        _rank,
        args=((tmp_path / "rendezvous").as_uri(), layout),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
