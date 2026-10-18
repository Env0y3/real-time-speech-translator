"""Synthetic playback summaries; no output stream, network, or model is used."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from core.elevenlabs_tts import (
    SentenceMetrics, SentenceState, _print_and_log_session,
)
from core.performance_logger import PerformanceLogger


class PlaybackErrorEventTests(unittest.IsolatedAsyncioTestCase):
    async def _events(self, failed):
        events = []
        state = SentenceState(
            sentence_id=7,
            metrics=SentenceMetrics(first_segment_ready_at=10, playback_error=failed),
        )
        with tempfile.TemporaryDirectory() as directory:
            logger = PerformanceLogger(Path(directory) / "events.jsonl", "test", events.append)
            with contextlib.redirect_stdout(io.StringIO()):
                await _print_and_log_session(logger, state)
        return events

    async def test_playback_failure_is_error_after_summary_not_listening(self):
        events = await self._events(True)
        self.assertEqual(events[-2]["type"], "trace_summary")
        self.assertTrue(events[-2]["playback_error"])
        self.assertEqual(events[-1]["type"], "playback_error")
        self.assertEqual(events[-1]["status"], "Error")
        self.assertEqual(events[-1]["error_category"], "audio_output_failed")
        self.assertEqual(events[-1]["sentence_id"], 7)
        self.assertEqual(events[-1]["message"],
                         "Audio playback failed. Check the selected output device.")
        self.assertFalse(any(event.get("status") == "Listening" for event in events))

    async def test_success_keeps_listening_status(self):
        events = await self._events(False)
        self.assertFalse(events[-2]["playback_error"])
        self.assertEqual(events[-1]["type"], "status")
        self.assertEqual(events[-1]["status"], "Listening")
        self.assertFalse(any(event["type"] == "playback_error" for event in events))


if __name__ == "__main__":
    unittest.main()
