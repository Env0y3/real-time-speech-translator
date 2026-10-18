import os
import unittest
from unittest.mock import patch
from core.pipeline import NormalPipelineOptions, validate_normal_pipeline_options


class LocalTTSRoutingTests(unittest.TestCase):
    def check(self, options):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "synthetic"}), \
             patch("core.pipeline.TTS_PROVIDER", "pyttsx3"), \
             patch("core.pipeline.NORMAL_ASR_PROVIDER", "sensevoice"), \
             patch("core.pipeline.resolve_input_device"):
            return validate_normal_pipeline_options(options)

    def test_default_without_monitor_is_supported(self):
        self.assertIsNone(self.check(NormalPipelineOptions()))

    def test_explicit_output_cannot_be_ignored(self):
        with self.assertRaisesRegex(ValueError, "system default"):
            self.check(NormalPipelineOptions(translation_output_device=2))

    def test_monitor_cannot_be_ignored(self):
        with self.assertRaisesRegex(ValueError, "local monitor"):
            self.check(NormalPipelineOptions(local_monitor_enabled=True))


class LocalTTSFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_is_reported_and_worker_stops(self):
        import asyncio
        from pathlib import Path
        from core.performance_logger import PerformanceLogger
        from core.tts import tts_worker
        queue = asyncio.Queue()
        await queue.put(("Synthetic phrase", 1.0, 1, 0))
        events = []
        logger = PerformanceLogger(Path("unused.jsonl"), "synthetic", events.append)
        with patch("core.tts.speak_sync", side_effect=RuntimeError("synthetic-private-detail")):
            with self.assertRaisesRegex(RuntimeError, "Local speech playback failed"):
                await tts_worker(queue, logger)
        self.assertEqual(events[-1]["type"], "playback_error")
        self.assertNotIn("synthetic-private-detail", str(events))
