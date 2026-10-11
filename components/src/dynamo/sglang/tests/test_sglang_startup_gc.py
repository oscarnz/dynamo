# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup GC must precede serving and never use the synchronous Engine API."""

import asyncio
import gc
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.sglang import init_llm

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(10),
]


@pytest.fixture
def startup(monkeypatch):
    """Replace engine/runtime boundaries while exercising the real init flow."""
    events = []
    state = {"frozen": False, "warm": False, "failure": False, "attached": False}

    async def freeze_gc():
        events.append("freeze-start")
        await asyncio.sleep(0)
        if state["failure"]:
            raise RuntimeError("freeze failed")
        state["frozen"] = True
        events.append("freeze-end")

    server_args = SimpleNamespace(
        node_rank=0,
        enable_trace=False,
        enable_forward_pass_metrics=False,
    )
    engine = SimpleNamespace(
        server_args=server_args,
        tokenizer_manager=SimpleNamespace(freeze_gc=freeze_gc),
        freeze_gc=Mock(side_effect=AssertionError("sync Engine API used")),
        shutdown=Mock(),
    )
    config = SimpleNamespace(
        server_args=server_args,
        dynamo_args=SimpleNamespace(
            namespace="unit",
            component="backend",
            endpoint="generate",
            frontend_decoding=False,
            use_sglang_tokenizer=False,
            endpoint_types="chat",
            custom_jinja_template=None,
            freeze_gc_after_init=False,
        ),
        use_resolved_server_args=lambda value: value,
        serving_mode=DisaggregationMode.AGGREGATED,
    )

    async def serving(*args, **kwargs):
        assert state["frozen"] is not state["attached"]
        events.append("serve")

    async def register(*args, **kwargs):
        assert state["frozen"] is not state["attached"]
        events.append("register")
        kwargs["readiness_gate"].set()

    async def warmup(*args):
        assert not state["frozen"]
        state["warm"] = True
        events.append("warmup")

    endpoint = SimpleNamespace(
        first_token_source=AsyncMock(return_value=None), serve_endpoint=serving
    )
    runtime = SimpleNamespace(endpoint=lambda _: endpoint)
    handler = SimpleNamespace(
        generate=Mock(),
        load_lora=Mock(),
        unload_lora=Mock(),
        list_loras=Mock(),
        clear_kv_blocks=Mock(),
        register_engine_routes=Mock(),
        follow_shared_pause_state=Mock(),
        cleanup=Mock(),
    )
    publisher = SimpleNamespace(
        component_gauges=SimpleNamespace(set_model_load_time=Mock())
    )
    case = SimpleNamespace(
        config=config,
        engine=engine,
        runtime=runtime,
        handler=handler,
        metrics_task=None,
        state=state,
        events=events,
    )

    async def setup_metrics(*args):
        case.metrics_task = asyncio.create_task(asyncio.Event().wait())
        return publisher, case.metrics_task, {}

    monkeypatch.setattr(init_llm.sgl, "Engine", Mock(return_value=engine))
    monkeypatch.setattr(init_llm, "set_forward_pass_metrics_worker_id", Mock())
    monkeypatch.setattr(init_llm, "gateway_worker_count", lambda *args: 1)
    monkeypatch.setattr(init_llm, "attached_engine_load_time", lambda: 0.0)
    monkeypatch.setattr(init_llm, "setup_sgl_metrics", setup_metrics)
    monkeypatch.setattr(init_llm, "DecodeWorkerHandler", Mock(return_value=handler))
    monkeypatch.setattr(init_llm, "PrefillWorkerHandler", Mock(return_value=handler))
    monkeypatch.setattr(init_llm, "register_model_with_readiness_gate", register)
    monkeypatch.setattr(init_llm, "_warmup_prefill_engine", warmup)
    payload = SimpleNamespace(to_dict=lambda: {})
    for name in (
        "SglangHealthCheckPayload",
        "SglangDisaggHealthCheckPayload",
        "SglangPrefillHealthCheckPayload",
    ):
        monkeypatch.setattr(init_llm, name, Mock(return_value=payload))
    return case


async def run_startup(startup, role, source="owned"):
    startup.config.serving_mode = role
    if source == "attached":
        startup.state["attached"] = True
    kwargs = {} if source == "owned" else {f"{source}_engine": startup.engine}
    init = (
        init_llm.init_prefill
        if role == DisaggregationMode.PREFILL
        else init_llm.init_decode
    )
    try:
        await init(startup.runtime, startup.config, asyncio.Event(), [], **kwargs)
    finally:
        if startup.metrics_task is not None:
            startup.metrics_task.cancel()
            await asyncio.gather(startup.metrics_task, return_exceptions=True)


@pytest.mark.parametrize(
    "role",
    [
        DisaggregationMode.AGGREGATED,
        DisaggregationMode.DECODE,
        DisaggregationMode.PREFILL,
    ],
)
@pytest.mark.parametrize("source", ["owned", "snapshot"])
def test_owned_worker_freezes_before_registration_and_serving(startup, role, source):
    before = (gc.isenabled(), gc.get_threshold())
    asyncio.run(run_startup(startup, role, source))
    assert startup.events.count("freeze-end") == 1
    assert startup.events.index("freeze-end") < startup.events.index("register")
    assert startup.events.index("freeze-end") < startup.events.index("serve")
    if role == DisaggregationMode.PREFILL:
        assert startup.events.index("warmup") < startup.events.index("freeze-start")
    startup.engine.freeze_gc.assert_not_called()
    startup.handler.cleanup.assert_called_once()
    assert startup.metrics_task.cancelled()
    assert (gc.isenabled(), gc.get_threshold()) == before


@pytest.mark.parametrize(
    "role",
    [
        DisaggregationMode.AGGREGATED,
        DisaggregationMode.DECODE,
        DisaggregationMode.PREFILL,
    ],
)
def test_freeze_failure_prevents_serving_and_cleans_up(startup, role):
    startup.state["failure"] = True
    with pytest.raises(RuntimeError, match="freeze failed"):
        asyncio.run(run_startup(startup, role))
    assert "serve" not in startup.events and "register" not in startup.events
    startup.handler.cleanup.assert_called_once()
    assert startup.metrics_task.cancelled()


@pytest.mark.parametrize(
    "role",
    [
        DisaggregationMode.AGGREGATED,
        DisaggregationMode.DECODE,
        DisaggregationMode.PREFILL,
    ],
)
def test_attached_gateway_does_not_refreeze_shared_engine(startup, role):
    asyncio.run(run_startup(startup, role, "attached"))
    assert "freeze-start" not in startup.events
    assert "serve" in startup.events and "register" in startup.events
    startup.handler.follow_shared_pause_state.assert_called_once()
    startup.handler.cleanup.assert_called_once()
    assert startup.metrics_task.cancelled()
