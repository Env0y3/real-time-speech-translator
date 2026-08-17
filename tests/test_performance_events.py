import asyncio
import tempfile
import unittest
from pathlib import Path

from core.performance_logger import PerformanceLogger


class PerformanceEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_log_is_forwarded_directly_to_callback(self) -> None:
        events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = PerformanceLogger(
                Path(temp_dir) / "latency.jsonl",
                "test-session",
                event_callback=events.append,
            )
            await logger.log(
                {
                    "event": "trace_summary",
                    "speech_end_to_first_playback_ms": 1234,
                }
            )

        self.assertEqual(events[0]["type"], "trace_summary")
        self.assertEqual(events[0]["session_id"], "test-session")
        self.assertEqual(
            events[0]["speech_end_to_first_playback_ms"],
            1234,
        )

    async def test_ui_only_event_is_not_written_to_log(self) -> None:
        events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "latency.jsonl"
            logger = PerformanceLogger(
                log_path,
                "test-session",
                event_callback=events.append,
            )
            logger.emit_event({"type": "status", "status": "Listening"})
            await asyncio.sleep(0)
            self.assertFalse(log_path.exists())

        self.assertEqual(events[0]["status"], "Listening")


if __name__ == "__main__":
    unittest.main()
