import asyncio

from config import INPUT_VALIDITY_GUARD_ENABLED
from core.performance_logger import PerformanceLogger
from core.text_utils import has_meaningful_content


NO_MEANINGFUL_CONTENT_REASON = "no_meaningful_content"


async def input_validity_guard_worker(
    asr_text_queue: asyncio.Queue,
    valid_text_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """在 False Trigger/Hotword 前阻止没有可翻译内容的 ASR 文本。"""
    while True:
        trace_item = await asr_text_queue.get()

        if trace_item is None:
            await valid_text_queue.put(None)
            print("Input Validity Guard Worker 已结束")
            break

        raw_text = (
            trace_item.get("text", "")
            if isinstance(trace_item, dict)
            else trace_item
        )
        text = raw_text if isinstance(raw_text, str) else ""
        if (
            not INPUT_VALIDITY_GUARD_ENABLED
            or has_meaningful_content(text)
        ):
            await valid_text_queue.put(trace_item)
            continue

        if performance_logger is not None:
            await performance_logger.log(
                {
                    "event": "input_guard",
                    "trace_id": (
                        trace_item.get("trace_id")
                        if isinstance(trace_item, dict)
                        else None
                    ),
                    "sentence_id": (
                        trace_item.get("sentence_id")
                        if isinstance(trace_item, dict)
                        else None
                    ),
                    "text": text,
                    "action": "drop",
                    "reason": NO_MEANINGFUL_CONTENT_REASON,
                }
            )

        print("\n[Input Guard]")
        print(f"Text: {text}")
        print("Action: DROP")
        print(f"Reason: {NO_MEANINGFUL_CONTENT_REASON}")
