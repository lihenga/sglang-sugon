"""Forward-time ingress ownership, ordering, and A/B switch regressions."""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as module
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def request(rid="new", **kwargs):
    return TokenizedGenerateReqInput(
        rid=rid,
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
        **kwargs,
    )


def scheduler(monkeypatch):
    s = Scheduler.__new__(Scheduler)
    s.device = "cpu"
    s.enable_overlap = False
    s.device_module = SimpleNamespace(Event=Mock, StreamContext=lambda _: nullcontext())
    s.forward_stream = Mock()
    s.schedule_stream = Mock()
    s._forward_ingress_group = object()
    s._forward_ingress_enabled = True
    s._waiting_queue_prefetch_enabled = True
    s.attn_dp_enabled = False
    s.return_health_check_ipcs = []
    s._deferred_input_requests = []
    s.handle_generate_request = Mock()
    s.handle_batch_generate_request = Mock()
    monkeypatch.setattr(module.torch.distributed, "all_reduce", lambda *a, **k: None)
    return s


def target_topology():
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, rank=0, ranks=[0], cpu_group=object()
    )
    return SimpleNamespace(
        pp_size=1,
        pp_rank=0,
        tp_size=1,
        tp_group=group,
        world_group=group,
        attn_dp_enabled=False,
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_dp_rank=0,
        attn_cp_size=1,
        attn_tp_size=1,
        attn_cp_group=group,
        attn_tp_group=group,
        enable_dp_attention_local_control_broadcast=True,
    )


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_receives_and_prefetches_while_forward_blocked(monkeypatch, batched, overlap):
    s = scheduler(monkeypatch)
    s.enable_overlap = overlap
    entered, prefetched = threading.Event(), threading.Event()
    owner = threading.get_ident()
    trace = contextvars.ContextVar("ingress-test")
    trace.set("request-context")
    seen = []

    def forward(batch):
        assert threading.get_ident() != owner
        assert trace.get() == "request-context"
        entered.set()
        assert prefetched.wait(5), "prefetch did not start during forward"
        return "result"

    def prepare(req):
        assert entered.wait(5)
        assert threading.get_ident() == owner
        # This is the same queue hook used by handle_generate_request.
        cache_req = SimpleNamespace(
            session=None,
            to_finish=None,
            init_next_round_input=lambda *a, **k: prefetched.set(),
        )
        s.tree_cache = SimpleNamespace(
            linker=SimpleNamespace(cache_linker=SimpleNamespace(prefetch_enabled=True))
        )
        Scheduler._prefetch_kvcache(s, cache_req)
        seen.append(req)

    req = request()
    received = BatchTokenizedGenerateReqInput(batch=[req]) if batched else req
    health = request(HEALTH_CHECK_RID_PREFIX + "-probe", http_worker_ipc="health")
    polls = iter([[health, received]])
    s.request_receiver = SimpleNamespace(recv_requests=lambda: next(polls, []))
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    s.handle_generate_request = prepare
    s.handle_batch_generate_request = lambda batch: [prepare(req) for req in batch]
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        assert s._forward_with_waiting_queue_ingress(object()) == "result"
    assert seen == [req]
    assert s.return_health_check_ipcs == ["health"]
    assert not s._deferred_input_requests
    if overlap:
        s.schedule_stream.wait_stream.assert_not_called()
    else:
        s.schedule_stream.wait_stream.assert_called_once_with(s.forward_stream)
    s.forward_stream.wait_event.assert_called_once()


@pytest.mark.parametrize(
    "unsupported", [object(), request(session_id="s"), request(lora_id="lora")]
)
def test_defers_unsupported_and_preserves_following_order(monkeypatch, unsupported):
    s = scheduler(monkeypatch)
    release = threading.Event()
    after = request("after")
    first = True

    def recv():
        nonlocal first
        if first:
            first = False
            release.set()
            return [unsupported, after]
        return []

    def forward(batch):
        assert release.wait(5)
        return "done"

    s.request_receiver = SimpleNamespace(recv_requests=recv)
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        assert s._forward_with_waiting_queue_ingress(None) == "done"
    assert s._deferred_input_requests == [unsupported, after]
    s.handle_generate_request.assert_not_called()


def test_disabled_path_stays_on_scheduler_thread(monkeypatch):
    s = scheduler(monkeypatch)
    s._forward_ingress_enabled = False
    s.model_worker = SimpleNamespace(
        forward_batch_generation=lambda batch: threading.get_ident()
    )
    assert s._forward_with_waiting_queue_ingress(None) == threading.get_ident()


@pytest.mark.parametrize("failure", [False, True])
def test_dspark_callbacks_keep_scheduler_ownership(monkeypatch, failure):
    s = scheduler(monkeypatch)
    s.enable_overlap = True
    owner = threading.get_ident()
    received, published = threading.Event(), threading.Event()

    def recv():
        received.set()
        return []

    def barrier():
        assert threading.get_ident() == owner
        if failure:
            raise ValueError("grammar failed")
        return "grammar-ready"

    def publish(value):
        assert threading.get_ident() != owner
        assert value == "published"
        published.set()

    def forward(batch, on_publish, grammar_barrier):
        assert received.wait(5)
        assert grammar_barrier() == "grammar-ready"
        on_publish("published")
        return "result"

    s.request_receiver = SimpleNamespace(recv_requests=recv)
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        if failure:
            with pytest.raises(ValueError, match="grammar failed"):
                s._forward_with_waiting_queue_ingress(
                    None, on_publish=publish, grammar_barrier=barrier
                )
        else:
            assert (
                s._forward_with_waiting_queue_ingress(
                    None, on_publish=publish, grammar_barrier=barrier
                )
                == "result"
            )
            assert published.is_set()
    s.schedule_stream.wait_stream.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA stream semantics"
)
@pytest.mark.parametrize("overlap", [False, True])
def test_forward_ingress_cuda_streams(monkeypatch, overlap):
    s = scheduler(monkeypatch)
    s.device = "cuda"
    s.device_module = torch.cuda
    s.enable_overlap = overlap
    s.tp_worker = SimpleNamespace(gpu_id=torch.cuda.current_device())
    s.schedule_stream, s.forward_stream = torch.cuda.Stream(), torch.cuda.Stream()
    entered, received = threading.Event(), threading.Event()
    with torch.cuda.stream(s.schedule_stream):
        value = torch.full((1,), 7, device="cuda")

    def forward(batch):
        assert torch.cuda.current_stream() == s.forward_stream
        entered.set()
        assert received.wait(5)
        return value + 1

    def recv():
        assert entered.wait(5)
        assert torch.cuda.current_stream() == s.schedule_stream
        received.set()
        return []

    s.request_receiver = SimpleNamespace(recv_requests=recv)
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        # run_batch's overlap branch enters the forward stream before calling us.
        with torch.cuda.stream(s.forward_stream if overlap else s.schedule_stream):
            output = s._forward_with_waiting_queue_ingress(None)
    s.forward_stream.synchronize()
    assert output.item() == 8


def test_forward_failure_is_propagated(monkeypatch):
    s = scheduler(monkeypatch)
    s.model_worker = SimpleNamespace(
        forward_batch_generation=Mock(side_effect=ValueError("forward failed"))
    )
    s.request_receiver = SimpleNamespace(recv_requests=lambda: [])
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        with pytest.raises(ValueError, match="forward failed"):
            s._forward_with_waiting_queue_ingress(None)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_startup_respects_switch_with_overlap(monkeypatch, overlap, enabled):
    s = scheduler(monkeypatch)
    s.is_generation = True
    s.enable_overlap = overlap
    s.enable_pdmux = False
    s.spec_algorithm = SpeculativeAlgorithm.DSPARK
    s.schedule_policy = "fcfs"
    s.disaggregation_mode = DisaggregationMode.NULL
    s.mm_receiver = s.input_blocker = s.scripted_scheduler_hook = None
    s.tree_cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=SimpleNamespace(prefetch_enabled=True))
    )
    s.tp_cpu_group = s.attn_cp_cpu_group = s.attn_tp_cpu_group = object()
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
    monkeypatch.setattr(
        module,
        "get_parallel",
        target_topology,
    )
    monkeypatch.setattr(
        module,
        "get_memory",
        lambda: SimpleNamespace(unified_cache_external_linker_backend="mooncake"),
    )
    monkeypatch.setattr(
        module, "get_observability", lambda: SimpleNamespace(kv_canary="none")
    )
    monkeypatch.setattr(
        module.torch.distributed, "get_process_group_ranks", lambda _: [0]
    )
    monkeypatch.setattr(
        module, "create_custom_parallel_group", lambda *a, **kw: object()
    )
    with envs.SGLANG_MOONCAKE_FORWARD_INGRESS.override(enabled):
        s.init_forward_ingress()
    try:
        assert s._forward_ingress_enabled is enabled
        assert s._waiting_queue_prefetch_enabled is enabled
    finally:
        if s._forward_launch_executor is not None:
            s._forward_launch_executor.shutdown(wait=True)


def test_receiver_keeps_target_topology_during_dspark_draft(monkeypatch):
    import zmq

    from sglang.srt.distributed.parallel_state import patch_tensor_parallel_group
    from sglang.srt.managers.scheduler_components import (
        request_receiver as receiver_module,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import published_topology

    def group(size):
        return SimpleNamespace(
            world_size=size,
            rank_in_group=0,
            rank=0,
            ranks=list(range(size)),
            cpu_group=object(),
        )

    with published_topology(tp_size=4, attn_dp_size=2, ranks=dict(dp_rank=0)):
        parallel = get_parallel()
        with parallel.override(
            tp_group=group(4),
            world_group=group(4),
            attn_tp_group=group(2),
            attn_cp_group=group(1),
        ):
            target = SimpleNamespace(
                **{
                    name: getattr(parallel, name)
                    for name in vars(target_topology())
                    if name != "pp_size"
                }
            )
            target.attn_dp_enabled = True
            target.enable_dp_attention_local_control_broadcast = True
            receiver = SchedulerRequestReceiver(
                recv_from_tokenizer=object(),
                recv_from_rpc=object(),
                recv_skipper=None,
                input_blocker=None,
                mm_receiver=None,
                server_args=None,
                model_config=SimpleNamespace(is_multimodal=False),
                max_recv_per_poll=-1,
                stream_output=None,
                get_last_batch=lambda: None,
                parallel=target,
                tp_cpu_group=object(),
                attn_cp_cpu_group=object(),
                attn_tp_cpu_group=object(),
            )
            pull = Mock(side_effect=zmq.Again())
            monkeypatch.setattr(receiver_module, "sock_recv", pull)
            broadcasts = []

            def broadcast(reqs, rank, cpu_group, src):
                assert cpu_group is receiver.attn_tp_cpu_group
                broadcasts.append(reqs)
                return reqs

            monkeypatch.setattr(receiver_module, "broadcast_pyobj", broadcast)
            entered, release = threading.Event(), threading.Event()

            def draft():
                with patch_tensor_parallel_group(group(2), owns_attention=True):
                    entered.set()
                    assert release.wait(5)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(draft)
                try:
                    assert entered.wait(5)
                    assert parallel.attn_cp_group is None
                    assert receiver.recv_requests() == []
                    assert pull.call_count == 2
                    assert len(broadcasts) == 2
                finally:
                    release.set()
                    future.result(timeout=5)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("ingress", [False, True])
def test_run_batch_dspark_receives_before_forward_returns(
    monkeypatch, overlap, ingress
):
    @dataclass
    class Batch:
        sampling_info: object = None
        spec_algorithm: object = SpeculativeAlgorithm.DSPARK
        req_pool_indices: object = None
        spec_info: object = None
        input_ids: object = "input"

    s = scheduler(monkeypatch)
    s.enable_overlap = overlap
    s._forward_ingress_enabled = ingress
    s.scheduler_stage_metrics = None
    s.metrics_reporter = Mock()
    s.forward_ct = s.processed_tokens_counter = 0
    s._sched_idled = False
    s.scripted_scheduler_hook = None
    s.profiler_manager = Mock()
    s.forward_sleep_time = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.is_generation = True
    s.enable_pdmux = s.enable_unified_memory = (
        s.enable_unified_cache_external_linker
    ) = False
    s._confidence_budget_prepare = None
    s.spec_algorithm = SpeculativeAlgorithm.DSPARK
    s.future_map = Mock()
    s.forward_stream_ctx = nullcontext()
    s.batch_record_buf, s.batch_record_ct = [None, None], 0
    s.result_queue = []
    s.pp_group = SimpleNamespace(is_last_rank=True)
    s._relay_forward_payload = Mock()
    s.update_cache_from_scheduler = Mock()
    s._copy_result_to_cpu = Mock()
    s._copy_auxiliary_output_to_cpu = Mock()
    monkeypatch.setattr(module, "resolve_forward_inputs", lambda *a: None)
    monkeypatch.setattr(module, "get_parallel", lambda: SimpleNamespace(pp_size=1))
    batch = Batch()
    batch.forward_mode = SimpleNamespace(
        is_split_prefill=lambda: False,
        is_prebuilt=lambda: False,
        is_extend=lambda: True,
    )
    batch.reqs, batch.extend_num_tokens = [], 0
    batch.return_logprob = batch.return_hidden_states = False
    result = SimpleNamespace(
        extra_keep_alive_refs=[],
        delay_sample_func=None,
        has_sampled_token_ids=False,
        next_draft_input=SimpleNamespace(dsa_topk_indices=None),
        new_seq_lens=None,
        copy_done=None,
        copy_to_cpu=Mock(),
    )
    prefetched = threading.Event()
    req = request()
    polls = iter([[req]])
    s.request_receiver = SimpleNamespace(recv_requests=lambda: next(polls, []))
    owner = threading.get_ident()

    def prepare(req):
        assert threading.get_ident() == owner
        prefetched.set()

    def forward(batch, on_publish=None, grammar_barrier=None, **kwargs):
        if ingress:
            assert threading.get_ident() != owner
            assert prefetched.wait(5)
        else:
            assert threading.get_ident() == owner
            assert not prefetched.is_set()
        if grammar_barrier:
            grammar_barrier()
        if on_publish:
            on_publish("seq_lens")
        batch.input_ids = "worker-mutation"
        return result

    s.handle_generate_request = prepare
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    if not ingress:
        s._forward_with_waiting_queue_ingress = Mock(
            side_effect=AssertionError(
                "Disabled run_batch must call model forward directly"
            )
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        s._forward_launch_executor = executor
        assert s.run_batch(batch) is result
    assert batch.input_ids is None
    assert batch.spec_info is result.next_draft_input
    if overlap:
        s.future_map.publish.assert_called_once_with(None, "seq_lens")
        s.schedule_stream.wait_stream.assert_not_called()
    else:
        s.future_map.publish.assert_not_called()
        if ingress:
            s.schedule_stream.wait_stream.assert_called_once_with(s.forward_stream)
        else:
            s.schedule_stream.wait_stream.assert_not_called()


def test_disabled_switch_preserves_original_queue_prefetch_path(monkeypatch):
    s = scheduler(monkeypatch)
    s._waiting_queue_prefetch_enabled = False
    s.enable_hicache_storage = s.enable_lmcache = False
    s.tree_cache = SimpleNamespace(
        linker=SimpleNamespace(cache_linker=SimpleNamespace(prefetch_enabled=True))
    )
    req = SimpleNamespace(session=None, to_finish=None, init_next_round_input=Mock())
    assert s._prefetch_kvcache(req) is None
    req.init_next_round_input.assert_not_called()


@pytest.mark.parametrize(
    "disabled",
    [
        "env",
        "pp",
        "canary",
        "prefetch",
        "spec",
        "pdmux",
        "decode",
        "policy",
        "blocker",
        "peer",
    ],
)
def test_startup_gates_do_not_create_groups(monkeypatch, disabled):
    s = scheduler(monkeypatch)
    s.is_generation = True
    s.enable_overlap = disabled == "overlap"
    s.enable_pdmux = disabled == "pdmux"
    s.spec_algorithm = SimpleNamespace(
        is_none=lambda: disabled != "spec", is_dspark=lambda: False
    )
    s.schedule_policy = "lpm" if disabled == "policy" else "fcfs"
    s.disaggregation_mode = (
        DisaggregationMode.DECODE if disabled == "decode" else DisaggregationMode.NULL
    )
    s.mm_receiver = s.scripted_scheduler_hook = None
    s.input_blocker = object() if disabled == "blocker" else None
    s.tree_cache = SimpleNamespace(
        linker=SimpleNamespace(
            cache_linker=SimpleNamespace(prefetch_enabled=disabled != "prefetch")
        )
    )
    monkeypatch.setattr(
        module,
        "get_parallel",
        lambda: SimpleNamespace(
            pp_size=2 if disabled == "pp" else 1,
            world_group=SimpleNamespace(
                world_size=2 if disabled == "peer" else 1, cpu_group=object()
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "get_memory",
        lambda: SimpleNamespace(unified_cache_external_linker_backend="mooncake"),
    )
    monkeypatch.setattr(
        module,
        "get_observability",
        lambda: SimpleNamespace(kv_canary="check" if disabled == "canary" else "none"),
    )
    create = Mock(side_effect=AssertionError("must not create a group"))
    monkeypatch.setattr(module, "create_custom_parallel_group", create)
    if disabled == "peer":
        monkeypatch.setattr(
            module.torch.distributed, "all_reduce", lambda flags, **kw: flags.zero_()
        )
    with envs.SGLANG_MOONCAKE_FORWARD_INGRESS.override(disabled != "env"):
        s.init_forward_ingress()
    assert not s._forward_ingress_enabled
    assert s._forward_launch_executor is None
    assert s._waiting_queue_prefetch_enabled == (
        disabled not in ("env", "pp", "decode", "prefetch", "peer")
    )
    create.assert_not_called()


def test_switch_defaults_on(monkeypatch):
    monkeypatch.delenv("SGLANG_MOONCAKE_FORWARD_INGRESS", raising=False)
    assert envs.SGLANG_MOONCAKE_FORWARD_INGRESS.get()


@pytest.mark.parametrize("rejected", [False, True])
def test_early_lookup_preserves_active_session(monkeypatch, rejected):
    from sglang.srt.managers.schedule_batch import ReqKvInfo
    from sglang.srt.session.session_controller import Session
    from sglang.srt.session.streaming_session import SessionSlot, StreamingSession

    s = scheduler(monkeypatch)
    session = Session(4096, session_id="active", streaming=True)
    session._inflight = True
    streaming = StreamingSession(SimpleNamespace())
    slot = SessionSlot(kv=ReqKvInfo(req_pool_idx=0))
    streaming.slots[session.session_id] = slot
    req = SimpleNamespace(
        session=session,
        to_finish=object() if rejected else None,
        release_mm_inputs_on_abort=Mock(),
    )

    def match(*args, **kwargs):
        streaming.find_active_slot(req)
        pytest.fail("Session requests must not enter early prefix matching")

    req.init_next_round_input = match
    s._prefetch_kvcache(req)
    assert session._inflight
    assert req.session is session
    assert streaming.slots[session.session_id] is slot


def test_early_lookup_skips_preaborted_request(monkeypatch):
    s = scheduler(monkeypatch)
    req = SimpleNamespace(
        session=None, to_finish=object(), init_next_round_input=Mock()
    )
    s._prefetch_kvcache(req)
    req.init_next_round_input.assert_not_called()
