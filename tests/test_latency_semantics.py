"""Offline latency contracts: synthetic clocks and temporary JSONL only."""
import json
import tempfile
import unittest
from pathlib import Path

from core.performance_logger import (
    PerformanceLogger, calculate_trace_latencies, elapsed_ms,
    summarize_trace_records,
)


class LatencyDefinitionTests(unittest.TestCase):
    def test_endpoint_and_last_voice_are_distinct(self):
        trace = {
            "last_voice_at": 10.0, "endpoint_triggered_at": 10.6,
            "asr_result_at": 10.8, "translation_first_token_at": 10.9,
            "translation_first_segment_at": 11.0,
            "tts_first_audio_received_at": 11.4, "first_playback_at": 11.5,
        }
        metrics = calculate_trace_latencies(trace)
        expected = {
            "endpoint_wait_ms": 600, "asr_stage_ms": 200,
            "translation_after_asr_ms": 200, "tts_after_first_segment_ms": 400,
            "speech_end_to_first_playback_ms": 900,
            "endpoint_trigger_to_first_playback_ms": 900,
            "last_voice_to_first_playback_ms": 1500,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value, msg=name)
        self.assertAlmostEqual(
            metrics["last_voice_to_first_playback_ms"],
            metrics["endpoint_wait_ms"] + metrics["endpoint_trigger_to_first_playback_ms"],
        )
        self.assertAlmostEqual(
            metrics["endpoint_trigger_to_first_playback_ms"],
            metrics["asr_stage_ms"] + metrics["translation_after_asr_ms"]
            + metrics["tts_after_first_segment_ms"]
            + elapsed_ms(trace["tts_first_audio_received_at"], trace["first_playback_at"]),
        )

    def test_unknown_is_not_zero_or_fabricated(self):
        self.assertTrue(all(value is None for value in calculate_trace_latencies({}).values()))
        for start, end in [(None, 2), (1, None), (True, 2), (1, float("nan")),
                           (float("inf"), 2), (3, 2), ("1", 2)]:
            self.assertIsNone(elapsed_ms(start, end))
        self.assertEqual(elapsed_ms(2, 2), 0)

    def test_statistics_require_explicit_session_and_keep_legacy_alias(self):
        records = [
            {"event": "trace_summary", "session_id": "historical",
             "speech_end_to_first_playback_ms": 2110},
            {"event": "trace_summary", "session_id": "current",
             "speech_end_to_first_playback_ms": 900},
            {"event": "trace_summary", "session_id": "current",
             "endpoint_trigger_to_first_playback_ms": 1100,
             "last_voice_to_first_playback_ms": 1700},
            {"event": "trace_summary", "session_id": "current",
             "endpoint_trigger_to_first_playback_ms": float("nan")},
            {"event": "tts_session", "session_id": "current",
             "endpoint_trigger_to_first_playback_ms": 99999},
        ]
        result = summarize_trace_records(records, session_id="current")
        self.assertEqual(result["trace_count"], 3)
        self.assertEqual(result["metrics"]["endpoint_trigger_to_first_playback_ms"], {
            "count": 2, "mean_ms": 1000, "median_ms": 1000, "p95_ms": 1100,
        })
        self.assertEqual(result["metrics"]["last_voice_to_first_playback_ms"]["count"], 1)
        self.assertIsNone(result["metrics"]["endpoint_wait_ms"]["mean_ms"])
        self.assertEqual(summarize_trace_records(records, session_id="absent")["trace_count"], 0)
        with self.assertRaises(TypeError):
            summarize_trace_records(records)
        self.assertNotIn("endpoint_trigger_to_first_playback_ms", records[1])


class LatencyLogCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_append_preserves_old_record_and_emits_new_fields(self):
        old = {"event": "trace_summary", "session_id": "old",
               "speech_end_to_first_playback_ms": 2110}
        old_line = json.dumps(old) + "\n"
        events = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance.jsonl"
            path.write_text(old_line, encoding="utf-8")
            logger = PerformanceLogger(path, "new", events.append)
            metrics = calculate_trace_latencies({
                "last_voice_at": 1, "endpoint_triggered_at": 1.5,
                "first_playback_at": 2,
            })
            await logger.log({"event": "trace_summary", **metrics,
                              "first_playback_measurement": "output_stream_write_start",
                              "first_playback_is_acoustic": False})
            contents = path.read_text(encoding="utf-8")
        self.assertTrue(contents.startswith(old_line))
        rows = [json.loads(line) for line in contents.splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], old)
        self.assertEqual(rows[1]["speech_end_to_first_playback_ms"], 500)
        self.assertEqual(rows[1]["endpoint_trigger_to_first_playback_ms"], 500)
        self.assertEqual(rows[1]["last_voice_to_first_playback_ms"], 1000)
        self.assertEqual(events[0]["first_playback_measurement"], "output_stream_write_start")
        self.assertFalse(events[0]["first_playback_is_acoustic"])


if __name__ == "__main__":
    unittest.main()
