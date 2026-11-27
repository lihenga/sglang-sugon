"""Forward-time ingress ownership, ordering, and A/B switch regressions."""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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


@pytest.mark.parametrize("batched", [False, True])
def test_receives_and_prefetches_while_forward_blocked(monkeypatch, batched):
    s = scheduler(monkeypatch)
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


@pytest.mark.parametrize(
    "disabled",
    [
        "env",
        "overlap",
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
    s.spec_algorithm = SimpleNamespace(is_none=lambda: disabled != "spec")
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
        disabled not in ("pp", "decode", "prefetch", "peer")
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
