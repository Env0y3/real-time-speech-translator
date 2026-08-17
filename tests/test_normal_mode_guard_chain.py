import asyncio
import unittest

from core.false_trigger_filter import false_trigger_filter_worker
from core.hotwords import hotword_correction_worker
from core.input_guard import input_validity_guard_worker


class RecordingLogger:
    def __init__(self) -> None:
        self.records = []

    async def log(self, record: dict) -> None:
        self.records.append(record)


class NormalModeGuardChainSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_normal_mode_guard_wiring(self) -> None:
        asr_queue = asyncio.Queue()
        valid_queue = asyncio.Queue()
        filtered_queue = asyncio.Queue()
        translation_input_queue = asyncio.Queue()
        logger = RecordingLogger()
        samples = [
            (1, "。", 200),
            (2, "嗯", 200),
            (3, "嗯", 600),
            (4, "嗯，我觉得可以", 200),
            (5, "你好", 200),
            (6, "2", 200),
            (7, "DeepSeek", 200),
        ]
        for sentence_id, text, speech_duration_ms in samples:
            await asr_queue.put(
                {
                    "trace_id": f"smoke-{sentence_id}",
                    "sentence_id": sentence_id,
                    "raw_text": text,
                    "text": text,
                    "speech_duration_ms": speech_duration_ms,
                }
            )
        await asr_queue.put(None)

        await asyncio.gather(
            input_validity_guard_worker(
                asr_queue,
                valid_queue,
                logger,
            ),
            false_trigger_filter_worker(
                valid_queue,
                filtered_queue,
                logger,
            ),
            hotword_correction_worker(
                filtered_queue,
                translation_input_queue,
                [],
            ),
        )

        published_items = []
        while True:
            item = await translation_input_queue.get()
            if item is None:
                break
            published_items.append(item)

        self.assertEqual(
            [item["sentence_id"] for item in published_items],
            [3, 4, 5, 6, 7],
        )
        self.assertEqual(
            [item["text"] for item in published_items],
            ["嗯", "嗯，我觉得可以", "你好", "2", "DeepSeek"],
        )
        self.assertEqual(
            [
                (record["event"], record["reason"])
                for record in logger.records
            ],
            [
                ("input_guard", "no_meaningful_content"),
                ("false_trigger_filter", "short_filler"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
