import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.input_guard import input_validity_guard_worker
from core.translation import (
    has_translation_output,
    streaming_translation_worker,
    translation_worker,
)


class FakeEmptyStream:
    def __init__(self) -> None:
        self._chunks = iter(
            [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="   ")
                        )
                    ]
                )
            ]
        )

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def close(self) -> None:
        return None


class FakeCompletions:
    def __init__(self, streaming: bool) -> None:
        self.streaming = streaming
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        if self.streaming:
            return FakeEmptyStream()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="   ")
                )
            ]
        )


class FakeClient:
    last_instance = None
    streaming = False

    def __init__(self, **kwargs) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions(self.streaming)
        )
        FakeClient.last_instance = self

    async def close(self) -> None:
        return None


class TranslationOutputTests(unittest.TestCase):
    def test_only_non_whitespace_output_is_valid(self) -> None:
        self.assertFalse(has_translation_output(""))
        self.assertFalse(has_translation_output("   \n"))
        self.assertTrue(has_translation_output("Hello"))


class TranslationWorkerGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.environment_patch = patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "test-key"},
        )
        self.environment_patch.start()

    async def asyncTearDown(self) -> None:
        self.environment_patch.stop()

    async def test_empty_full_translation_never_reaches_tts_queue(self) -> None:
        FakeClient.streaming = False
        source_queue = asyncio.Queue()
        translated_queue = asyncio.Queue()
        await source_queue.put("你好")
        await source_queue.put(None)

        with patch("core.translation.AsyncOpenAI", FakeClient):
            await translation_worker(source_queue, translated_queue)

        self.assertIsNone(await translated_queue.get())
        self.assertTrue(translated_queue.empty())

    async def test_punctuation_input_never_calls_deepseek(self) -> None:
        FakeClient.streaming = False
        asr_queue = asyncio.Queue()
        guarded_queue = asyncio.Queue()
        translated_queue = asyncio.Queue()
        await asr_queue.put(
            {
                "trace_id": "guard-1",
                "sentence_id": 1,
                "text": "。",
            }
        )
        await asr_queue.put(None)

        with patch("core.translation.AsyncOpenAI", FakeClient):
            await asyncio.gather(
                input_validity_guard_worker(asr_queue, guarded_queue),
                translation_worker(guarded_queue, translated_queue),
            )

        self.assertIsNone(await translated_queue.get())
        self.assertTrue(translated_queue.empty())
        self.assertEqual(
            FakeClient.last_instance.chat.completions.call_count,
            0,
        )

    async def test_empty_stream_does_not_fallback_or_reach_tts(self) -> None:
        FakeClient.streaming = True
        source_queue = asyncio.Queue()
        translated_queue = asyncio.Queue()
        await source_queue.put("你好")
        await source_queue.put(None)

        with patch("core.translation.AsyncOpenAI", FakeClient):
            await streaming_translation_worker(
                source_queue,
                translated_queue,
            )

        self.assertIsNone(await translated_queue.get())
        self.assertTrue(translated_queue.empty())
        self.assertEqual(
            FakeClient.last_instance.chat.completions.call_count,
            1,
        )


if __name__ == "__main__":
    unittest.main()
