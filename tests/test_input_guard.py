import asyncio
import unittest

from core.input_guard import input_validity_guard_worker
from core.text_utils import has_meaningful_content


class RecordingLogger:
    def __init__(self) -> None:
        self.records = []

    async def log(self, record: dict) -> None:
        self.records.append(record)


class MeaningfulContentTests(unittest.TestCase):
    def test_empty_whitespace_and_punctuation_are_invalid(self) -> None:
        for text in (
            "",
            " ",
            "。",
            "！！",
            "，？。",
            "@#$%^&*()",
            None,
        ):
            with self.subTest(text=text):
                self.assertFalse(has_meaningful_content(text))

    def test_chinese_english_and_digits_are_valid(self) -> None:
        for text in ("你好。", "DeepSeek", "2。", "Python", "嗯"):
            with self.subTest(text=text):
                self.assertTrue(has_meaningful_content(text))


class InputValidityGuardWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_input_is_logged_and_never_published(self) -> None:
        source_queue = asyncio.Queue()
        valid_queue = asyncio.Queue()
        logger = RecordingLogger()
        invalid_item = {
            "trace_id": "session-1",
            "sentence_id": 1,
            "text": "。",
        }
        valid_item = {
            "trace_id": "session-2",
            "sentence_id": 2,
            "text": "DeepSeek",
        }
        await source_queue.put(invalid_item)
        await source_queue.put(valid_item)
        await source_queue.put(None)

        await input_validity_guard_worker(
            source_queue,
            valid_queue,
            logger,
        )

        self.assertEqual(await valid_queue.get(), valid_item)
        self.assertIsNone(await valid_queue.get())
        self.assertEqual(len(logger.records), 1)
        self.assertEqual(logger.records[0]["event"], "input_guard")
        self.assertEqual(
            logger.records[0]["reason"],
            "no_meaningful_content",
        )


if __name__ == "__main__":
    unittest.main()
