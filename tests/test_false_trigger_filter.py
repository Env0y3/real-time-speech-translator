import asyncio
import unittest

from core.false_trigger_filter import (
    false_trigger_filter_worker,
    get_speech_duration_ms,
    should_drop_false_trigger,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.records = []

    async def log(self, record: dict) -> None:
        self.records.append(record)


class FalseTriggerRuleTests(unittest.TestCase):
    def test_short_fillers_are_dropped(self) -> None:
        for text in ("嗯", "嗯。", "啊", "呃", "额", "嗯嗯"):
            with self.subTest(text=text):
                self.assertEqual(
                    should_drop_false_trigger(text, 180),
                    (True, "short_filler"),
                )

    def test_natural_length_filler_is_kept(self) -> None:
        self.assertEqual(
            should_drop_false_trigger("嗯", 500),
            (False, None),
        )

    def test_threshold_is_inclusive_and_conservative(self) -> None:
        self.assertEqual(
            should_drop_false_trigger("嗯", 350),
            (True, "short_filler"),
        )
        self.assertEqual(
            should_drop_false_trigger("嗯", 351),
            (False, None),
        )

    def test_filler_inside_sentence_is_kept(self) -> None:
        self.assertEqual(
            should_drop_false_trigger("嗯，我觉得可以。", 180),
            (False, None),
        )

    def test_real_short_phrases_are_kept(self) -> None:
        for text in (
            "你好",
            "谢谢",
            "可以",
            "不用",
            "好的",
            "不是",
            "对",
            "行",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    should_drop_false_trigger(text, 180),
                    (False, None),
                )

    def test_missing_duration_is_kept(self) -> None:
        self.assertEqual(
            should_drop_false_trigger("嗯", None),
            (False, None),
        )

    def test_timestamp_duration_fallback_uses_monotonic_values(self) -> None:
        self.assertAlmostEqual(
            get_speech_duration_ms(
                {"speech_started_at": 10.0, "last_voice_at": 10.18}
            ),
            180,
        )


class FalseTriggerWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_drop_never_reaches_downstream_and_is_logged(self) -> None:
        source_queue = asyncio.Queue()
        downstream_queue = asyncio.Queue()
        logger = RecordingLogger()
        dropped_item = {
            "trace_id": "session-1",
            "sentence_id": 1,
            "text": "嗯",
            "speech_duration_ms": 180,
            "voice_chunk_count": 1,
        }
        kept_item = {
            "trace_id": "session-2",
            "sentence_id": 2,
            "text": "你好",
            "speech_duration_ms": 180,
            "voice_chunk_count": 1,
        }
        await source_queue.put(dropped_item)
        await source_queue.put(kept_item)
        await source_queue.put(None)

        await false_trigger_filter_worker(
            source_queue,
            downstream_queue,
            logger,
        )

        self.assertEqual(await downstream_queue.get(), kept_item)
        self.assertIsNone(await downstream_queue.get())
        self.assertEqual(len(logger.records), 1)
        self.assertEqual(logger.records[0]["action"], "drop")
        self.assertEqual(logger.records[0]["reason"], "short_filler")


if __name__ == "__main__":
    unittest.main()
