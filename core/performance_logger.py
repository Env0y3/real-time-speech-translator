import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


EventCallback = Callable[[dict[str, Any]], None]


def create_session_id() -> str:
    """为每次 Normal Mode 运行生成易读的 Session ID（会话编号）。"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


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
