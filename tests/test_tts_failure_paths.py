"""Offline TTS failure regressions: no soundcard, credentials, or network."""
import asyncio
import base64
import contextlib
import io
import json
import os
import time
import unittest
from unittest.mock import patch

from core.audio_devices import AudioDeviceInfo, AudioRoutingPlan
from core import elevenlabs_tts as tts


class MemoryLogger:
    session_id = "offline-tts-regression"

    def __init__(self, callback=None):
        self.callback = callback
        self.events = []
        self.failure_seen = asyncio.Event()

    def emit_event(self, event):
        event = {**event, "type": event.get("type", event.get("event"))}
        self.events.append(event)
        if self.callback is not None:
            self.callback(event)
        if event["type"] == "playback_error":
            self.failure_seen.set()

    async def log(self, event):
        self.emit_event(event)


class FakeOutput:
    def __init__(self, fail=False, fail_close=False, on_write=None):
        self.fail = fail
        self.fail_close = fail_close
        self.writes = 0
        self.closed = False
        self.on_write = on_write

    def write(self, samples):
        self.writes += 1
        if self.fail:
            raise RuntimeError("synthetic output device disappeared")
        if self.on_write is not None:
            self.on_write()

    def stop(self):
        pass

    def close(self):
        self.closed = True
        if self.fail_close:
            raise RuntimeError("synthetic cleanup failure")


class FakeSocket:
    def __init__(self, frames, final_gate=None, audio_written=None):
        self.frames = frames
        self.final_gate = final_gate
        self.exited = False
        self.cancelled = False
        self.audio_written = audio_written

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.exited = True

    async def send(self, text):
        pass

    async def __aiter__(self):
        try:
            for frame in self.frames:
                yield json.dumps(frame)
                if frame.get("audio") and self.audio_written is not None:
                    await self.audio_written.wait()
            if self.final_gate is not None:
                await self.final_gate.wait()
                yield json.dumps({"is_final": True})
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def routing(default=False):
    return AudioRoutingPlan(
        AudioDeviceInfo(42, "Offline fake output", 0, 1, 24000),
        default, False, None,
    )


def audio_frame(chunk=b"\x01\x00\x02\x00"):
    return {"audio": base64.b64encode(chunk).decode("ascii")}


class TTSFailurePathTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, frames, *, default=False, fail_output=False,
                       fallback_error=None, final_gate=None, observe=None,
                       final_segment=True, end_input=True, fail_close=False,
                       wait_for_write=False, event_callback=None):
        queue = asyncio.Queue()
        await queue.put({
            "event": "segment", "sentence_id": 7, "segment_index": 1,
            "text": "One synthetic sentence.", "is_final_segment": final_segment,
            "segment_ready_at": time.perf_counter(),
            "trace": {"trace_id": "offline-7"},
        })
        if end_input:
            await queue.put(None)
        logger = MemoryLogger(event_callback)
        written = asyncio.Event() if wait_for_write else None
        loop = asyncio.get_running_loop()
        output = FakeOutput(
            fail_output, fail_close,
            (lambda: loop.call_soon_threadsafe(written.set)) if written else None,
        )
        websocket = FakeSocket(frames, final_gate, written)
        error = None
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "offline-key"}), \
             patch.object(tts.websockets, "connect", return_value=websocket), \
             patch.object(tts, "_open_audio_output_streams",
                          return_value=tts.AudioOutputStreams(output)), \
             patch.object(tts, "speak_sync", return_value=time.perf_counter(),
                          side_effect=fallback_error) as fallback, \
             contextlib.redirect_stdout(io.StringIO()):
            task = asyncio.create_task(tts.elevenlabs_tts_worker(queue, logger, routing(default)))
            try:
                if observe is not None:
                    await observe(logger, task, websocket)
                await asyncio.wait_for(task, 1.0)
            except Exception as caught:
                error = caught
            finally:
                if final_gate is not None:
                    final_gate.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return logger.events, fallback.call_count, output, websocket, error

    def assert_fatal(self, events, category):
        failures = [e for e in events if e["type"] == "playback_error"]
        self.assertEqual(len(failures), 1, events)
        self.assertEqual(failures[0]["status"], "Error")
        self.assertEqual(failures[0]["error_category"], category)
        self.assertTrue(failures[0].get("request_stop"))
        self.assertFalse(any(e.get("status") == "Listening" for e in events), events)

    async def test_zero_audio_service_error_with_explicit_route_is_fatal(self):
        events, calls, output, websocket, error = await self.run_case([
            {"error": "synthetic upstream unavailable"},
        ])
        self.assert_fatal(events, "tts_service_failed")
        self.assertEqual(calls, 0)
        self.assertTrue(output.closed)
        self.assertTrue(websocket.exited)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_partial_audio_service_error_never_replays_sentence(self):
        events, calls, output, websocket, error = await self.run_case([
            audio_frame(), {"error": "synthetic interrupted stream"},
        ], default=True, wait_for_write=True)
        self.assert_fatal(events, "tts_partial_failed")
        self.assertEqual(calls, 0)
        self.assertEqual(output.writes, 1)
        self.assertTrue(output.closed)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_service_error_does_not_wait_for_translation_tail(self):
        events, calls, output, websocket, error = await self.run_case([
            {"error": "synthetic upstream unavailable"},
        ], final_segment=False, end_input=False)
        self.assert_fatal(events, "tts_service_failed")
        self.assertTrue(output.closed)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_device_error_interrupts_wait_for_next_translation(self):
        events, calls, output, websocket, error = await self.run_case([
            audio_frame(), {"is_final": True},
        ], fail_output=True, end_input=False)
        self.assert_fatal(events, "audio_output_failed")
        self.assertTrue(output.closed)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_cleanup_error_cannot_replace_service_error(self):
        events, calls, output, websocket, error = await self.run_case([
            {"error": "synthetic upstream unavailable"},
        ], fail_close=True)
        self.assert_fatal(events, "tts_service_failed")
        self.assertTrue(output.closed)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)
        self.assertIn("Speech synthesis failed", str(error))

    async def test_successful_default_fallback_is_recorded_separately(self):
        events, calls, output, websocket, error = await self.run_case([
            {"error": "synthetic upstream unavailable"},
        ], default=True)
        self.assertIsNone(error)
        self.assertEqual(calls, 1)
        self.assertFalse(any(e["type"] == "playback_error" for e in events))
        fallbacks = [e for e in events if e["type"] == "tts_fallback"]
        self.assertEqual([e["fallback_status"] for e in fallbacks], ["started", "succeeded"])
        self.assertTrue(all(e["failure_category"] == "tts_service_failed" for e in fallbacks))
        summary = next(e for e in events if e["type"] == "trace_summary")
        self.assertEqual(summary["fallback_provider"], "pyttsx3")
        self.assertEqual(summary["fallback_status"], "succeeded")
        self.assertTrue(summary["service_failure"])
        self.assertFalse(summary["playback_error"])

    async def test_failed_fallback_is_fatal_and_not_listening(self):
        events, calls, output, websocket, error = await self.run_case([
            {"error": "synthetic upstream unavailable"},
        ], default=True, fallback_error=RuntimeError("synthetic local failure"))
        self.assert_fatal(events, "audio_output_failed")
        self.assertEqual(calls, 1)
        self.assertIn("failed", [e.get("fallback_status") for e in events])
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_output_write_error_stops_without_remote_final(self):
        gate = asyncio.Event()

        async def observe(logger, task, websocket):
            await asyncio.wait_for(logger.failure_seen.wait(), 0.25)
            self.assertFalse(gate.is_set())
            await asyncio.wait_for(asyncio.shield(task), 0.25)

        events, calls, output, websocket, error = await self.run_case(
            [audio_frame()], default=True, fail_output=True,
            final_gate=gate, observe=observe,
        )
        self.assert_fatal(events, "audio_output_failed")
        self.assertTrue(websocket.cancelled)
        self.assertTrue(output.closed)
        self.assertEqual(output.writes, 1)
        self.assertEqual(calls, 0)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)

    async def test_malformed_pcm_stops_without_remote_final(self):
        gate = asyncio.Event()

        async def observe(logger, task, websocket):
            await asyncio.wait_for(logger.failure_seen.wait(), 0.25)
            await asyncio.wait_for(asyncio.shield(task), 0.25)

        events, calls, output, websocket, error = await self.run_case(
            [audio_frame(b"\x01")], final_gate=gate, observe=observe,
        )
        self.assert_fatal(events, "audio_output_failed")
        self.assertTrue(websocket.cancelled)
        self.assertEqual(output.writes, 0)
        self.assertIsInstance(error, tts.ElevenLabsTTSError)


if __name__ == "__main__":
    unittest.main()
