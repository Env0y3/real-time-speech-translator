import asyncio

from config import (
    FALSE_TRIGGER_FILLERS,
    FALSE_TRIGGER_FILTER_ENABLED,
    FALSE_TRIGGER_MAX_SPEECH_SECONDS,
)
from core.performance_logger import PerformanceLogger
from core.text_utils import normalize_text


SHORT_FILLER_REASON = "short_filler"
NORMALIZED_FALSE_TRIGGER_FILLERS = frozenset(
    normalize_text(filler) for filler in FALSE_TRIGGER_FILLERS
)
MAX_SPEECH_DURATION_MS = FALSE_TRIGGER_MAX_SPEECH_SECONDS * 1000


def should_drop_false_trigger(
    text: str,
    speech_duration_ms: float | None,
) -> tuple[bool, str | None]:
    """保守过滤极短 filler，单独出现短文本或 filler 均不足以触发。"""
    if not FALSE_TRIGGER_FILTER_ENABLED or speech_duration_ms is None:
        return False, None

    if (
        normalize_text(str(text)) in NORMALIZED_FALSE_TRIGGER_FILLERS
        and speech_duration_ms <= MAX_SPEECH_DURATION_MS
    ):
        return True, SHORT_FILLER_REASON
    return False, None


def get_speech_duration_ms(trace_item: dict) -> float | None:
    """优先读取 ASR 的有效发声时长，兼容同一 monotonic clock 的时间戳。"""
    speech_duration_ms = trace_item.get("speech_duration_ms")
    if isinstance(speech_duration_ms, (int, float)):
        return float(speech_duration_ms)

    speech_started_at = trace_item.get("speech_started_at")
    last_voice_at = trace_item.get("last_voice_at")
    if isinstance(speech_started_at, (int, float)) and isinstance(
        last_voice_at,
        (int, float),
    ):
        return max(0.0, (last_voice_at - speech_started_at) * 1000)
    return None


async def false_trigger_filter_worker(
    asr_text_queue: asyncio.Queue,
    filtered_text_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """在 ASR 与 Hotword/Translation 之间丢弃明确的极短 filler。"""
    while True:
        trace_item = await asr_text_queue.get()

        if trace_item is None:
            await filtered_text_queue.put(None)
            print("False Trigger Filter Worker 已结束")
            break

        # Normal SenseVoice 使用 dict；旧字符串输入没有时长，安全放行。
        if not isinstance(trace_item, dict):
            await filtered_text_queue.put(trace_item)
            continue

        text = str(trace_item.get("text", ""))
        speech_duration_ms = get_speech_duration_ms(trace_item)
        should_drop, reason = should_drop_false_trigger(
            text,
            speech_duration_ms,
        )
        if not should_drop:
            await filtered_text_queue.put(trace_item)
            continue

        if performance_logger is not None:
            await performance_logger.log(
                {
                    "event": "false_trigger_filter",
                    "trace_id": trace_item.get("trace_id"),
                    "sentence_id": trace_item.get("sentence_id"),
                    "text": text,
                    "speech_duration_ms": round(speech_duration_ms, 1),
                    "voice_chunk_count": trace_item.get(
                        "voice_chunk_count"
                    ),
                    "action": "drop",
                    "reason": reason,
                }
            )

        print("\n[False Trigger Filter]")
        print(f"Text: {text}")
        print(f"Speech Duration: {speech_duration_ms:.0f} ms")
        print("Action: DROP")
        print(f"Reason: {reason}")
