# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MingStreamingDecodeScheduler and text stream output builder."""

from __future__ import annotations

import queue as _queue_mod
from unittest.mock import MagicMock

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage


class _FakeTokenizer:
    def __init__(self, eos_token_id=100):
        self.eos_token_id = eos_token_id

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(min(t, 0x7F)) for t in token_ids)


def _msg_new_request(request_id: str, stream: bool = False) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="new_request",
        data=StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs="test",
                params={"stream": stream} if stream else {},
            ),
            data={},
        ),
    )


def _msg_stream_chunk(request_id: str, token_id: int) -> IncomingMessage:
    import torch

    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=torch.tensor([token_id], dtype=torch.long),
    )


class TestMingStreamingDecodeScheduler:
    def _make_scheduler(self, tokenizer=None, eos_token_id=None):
        from sglang_omni.models.ming_omni.components.streaming_decode import (
            MingStreamingDecodeScheduler as Scheduler,
        )

        if tokenizer is None:
            tokenizer = _FakeTokenizer()
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        return Scheduler(tokenizer=tokenizer, eos_token_id=eos_token_id)

    def _drain(self, scheduler) -> list[OutgoingMessage]:
        msgs = []
        while True:
            try:
                msgs.append(scheduler.outbox.get_nowait())
            except _queue_mod.Empty:
                break
        return msgs

    # ── non-streaming ────────────────────────────────────────────────

    def test_new_request_non_streaming_emits_result(self):
        scheduler = self._make_scheduler()
        scheduler._start = lambda: None

        scheduler._on_new_request("req", _msg_new_request("req", stream=False).data)
        msgs = self._drain(scheduler)
        assert len(msgs) == 1
        assert msgs[0].type == "result"
        assert "req" not in scheduler._state

    # ── streaming ────────────────────────────────────────────────────

    def test_stream_chunk_incremental_emit(self):
        scheduler = self._make_scheduler()
        scheduler._start = lambda: None

        scheduler._on_new_request("req", _msg_new_request("req", stream=True).data)
        assert self._drain(scheduler) == []

        scheduler._on_stream_chunk("req", _msg_stream_chunk("req", 72).data)
        msgs = self._drain(scheduler)
        assert len(msgs) == 1
        assert msgs[0].type == "stream"
        assert msgs[0].target is None  # terminal -> coordinator
        assert msgs[0].data["text"] == "H"
        assert msgs[0].data["modality"] == "text"

    def test_abort_cleans_state(self):
        scheduler = self._make_scheduler()
        scheduler._start = lambda: None

        scheduler._on_new_request("req", _msg_new_request("req", stream=True).data)
        assert "req" in scheduler._state
        scheduler.abort("req")
        assert "req" not in scheduler._state

    def test_stream_done_finalizes(self):
        scheduler = self._make_scheduler()
        scheduler._start = lambda: None

        scheduler._on_new_request("req", _msg_new_request("req", stream=True).data)
        assert self._drain(scheduler) == []

        scheduler._on_stream_chunk("req", _msg_stream_chunk("req", 101).data)
        msgs = self._drain(scheduler)
        assert len(msgs) == 1
        assert msgs[0].type == "stream"

        scheduler._on_stream_done("req")
        final_msgs = self._drain(scheduler)
        assert any(m.type == "result" for m in final_msgs)
        assert "req" not in scheduler._state

    def test_error_isolation_single_request(self):
        scheduler = self._make_scheduler()
        scheduler._on_new_request("A", _msg_new_request("A", stream=True).data)
        scheduler._on_new_request("B", _msg_new_request("B", stream=True).data)

        scheduler._on_stream_chunk("A", _msg_stream_chunk("A", 72).data)
        msgs = self._drain(scheduler)
        assert any(m.request_id == "A" and m.type == "stream" for m in msgs)

        scheduler._on_stream_chunk("B", _msg_stream_chunk("B", 73).data)
        msgs = self._drain(scheduler)
        assert any(m.request_id == "B" and m.type == "stream" for m in msgs)


class TestTextStreamOutputBuilder:
    def test_emits_token_ids_to_decode(self):
        from sglang_omni.models.ming_omni.bootstrap import (
            make_text_stream_output_builder,
        )

        tokenizer = _FakeTokenizer(eos_token_id=100)
        builder = make_text_stream_output_builder(
            tokenizer=tokenizer,
            eos_token_id=100,
        )

        req = MagicMock()
        req.is_chunked = 0
        req_data = MagicMock()
        req_data.req = req
        req_output = MagicMock()
        req_output.data = 42

        msgs = builder("req-1", req_data, req_output)
        assert len(msgs) == 1
        assert msgs[0].type == "stream"
        assert msgs[0].target == "decode"
        assert msgs[0].metadata["token_id"] == 42
        assert msgs[0].metadata["is_eos"] is False

    def test_emits_eos_flag(self):
        from sglang_omni.models.ming_omni.bootstrap import (
            make_text_stream_output_builder,
        )

        tokenizer = _FakeTokenizer(eos_token_id=100)
        builder = make_text_stream_output_builder(
            tokenizer=tokenizer,
            eos_token_id=100,
        )

        req = MagicMock()
        req.is_chunked = 0
        req_data = MagicMock()
        req_data.req = req
        req_output = MagicMock()
        req_output.data = 100  # eos token

        msgs = builder("req-1", req_data, req_output)
        assert msgs[0].metadata["is_eos"] is True

    def test_suppresses_chunked_prefill(self):
        from sglang_omni.models.ming_omni.bootstrap import (
            make_text_stream_output_builder,
        )

        builder = make_text_stream_output_builder(
            tokenizer=_FakeTokenizer(),
            eos_token_id=100,
        )

        req = MagicMock()
        req.is_chunked = 1
        req_data = MagicMock()
        req_data.req = req
        req_output = MagicMock()
        req_output.data = 42

        msgs = builder("req-1", req_data, req_output)
        assert msgs == []
