import asyncio
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EventCallback = Callable[[dict[str, Any]], None]


def create_session_id() -> str:
    """为每次 Normal Mode 运行生成易读的 Session ID（会话编号）。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


# Legacy speech_end_* names start at the endpoint trigger, NOT last voice.
LATENCY_INTERVALS = {
    "endpoint_wait_ms": ("last_voice_at", "endpoint_triggered_at"),
    "speech_end_to_asr_result_ms": ("endpoint_triggered_at", "asr_result_at"),
    "speech_end_to_translation_first_token_ms": (
        "endpoint_triggered_at", "translation_first_token_at"),
    "speech_end_to_translation_first_segment_ms": (
        "endpoint_triggered_at", "translation_first_segment_at"),
    "speech_end_to_tts_first_audio_ms": (
        "endpoint_triggered_at", "tts_first_audio_received_at"),
    "speech_end_to_first_playback_ms": (
        "endpoint_triggered_at", "first_playback_at"),
    "last_voice_to_first_playback_ms": ("last_voice_at", "first_playback_at"),
    "translation_after_asr_ms": ("asr_result_at", "translation_first_segment_at"),
    "tts_after_first_segment_ms": (
        "translation_first_segment_at", "tts_first_audio_received_at"),
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def elapsed_ms(start_at: Any, end_at: Any) -> float | None:
    """Same-process perf_counter interval; missing/invalid/reversed is unknown."""
    if not _finite_number(start_at) or not _finite_number(end_at):
        return None
    if end_at < start_at:
        return None
    return (end_at - start_at) * 1000


def calculate_trace_latencies(trace: dict[str, Any]) -> dict[str, float | None]:
    """Pure interval definitions, retaining all legacy speech_end_* fields."""
    values = {
        name: elapsed_ms(trace.get(start), trace.get(end))
        for name, (start, end) in LATENCY_INTERVALS.items()
    }
    values["endpoint_trigger_to_first_playback_ms"] = values[
        "speech_end_to_first_playback_ms"
    ]
    values["asr_stage_ms"] = values["speech_end_to_asr_result_ms"]
    return values


def summarize_trace_records(
    records: list[dict[str, Any]], *, session_id: str
) -> dict[str, Any]:
    """Summarize only the requested run; never mix append-only historical runs.

    Missing metrics are excluded, never converted to zero. p95 uses nearest rank.
    Old logs retain their values and only their unambiguous endpoint alias is read.
    """
    selected = [
        row for row in records
        if row.get("event", row.get("type")) == "trace_summary"
        and row.get("session_id") == session_id
    ]
    result: dict[str, Any] = {
        "session_id": session_id, "trace_count": len(selected), "metrics": {}
    }
    names = (
        "endpoint_wait_ms", "asr_stage_ms", "translation_after_asr_ms",
        "tts_after_first_segment_ms", "tts_queue_wait_ms",
        "endpoint_trigger_to_first_playback_ms", "last_voice_to_first_playback_ms",
    )
    for name in names:
        values = []
        for row in selected:
            value = row.get(name)
            if name == "endpoint_trigger_to_first_playback_ms" and value is None:
                value = row.get("speech_end_to_first_playback_ms")
            if _finite_number(value) and value >= 0:
                values.append(value)
        values.sort()
        result["metrics"][name] = {
            "count": len(values),
            "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "p95_ms": values[math.ceil(len(values) * 0.95) - 1] if values else None,
        }
    return result


class PerformanceLogger:
    """以 JSONL 追加方式保存同一运行会话的延迟记录。"""

    def __init__(
        self,
        log_path: Path,
        session_id: str,
        event_callback: EventCallback | None = None,
    ) -> None:
        self.log_path = log_path
        self.session_id = session_id
        self.event_callback = event_callback
        self._write_lock = asyncio.Lock()

    def emit_event(self, record: dict[str, Any]) -> None:
        """把结构化事件同步转发给可选 UI；回调失败不影响 Pipeline。"""
        if self.event_callback is None:
            return
        event = {
            **record,
            "type": record.get("type", record.get("event", "event")),
            "session_id": self.session_id,
        }
        try:
            self.event_callback(event)
        except Exception as error:
            print(f"[Event Callback Warning] {type(error).__name__}")

    def _append_sync(self, record: dict[str, Any]) -> None:
        """同步追加一行 JSON；由后台线程调用，避免阻塞 Event Loop。"""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            json.dump(record, log_file, ensure_ascii=False)
            log_file.write("\n")

    async def log(self, record: dict[str, Any]) -> None:
        """安全写入记录；日志失败不能影响实时 Pipeline（流水线）。"""
        record_with_session = {
            **record,
            "session_id": self.session_id,
        }
        self.emit_event(record_with_session)
        try:
            # Lock（锁）防止 Translation 与 TTS 同时写入造成行内容交错。
            async with self._write_lock:
                await asyncio.to_thread(
                    self._append_sync,
                    record_with_session,
                )
        except Exception as error:
            print(
                "[Performance Log Warning] "
                f"{type(error).__name__}"
            )
