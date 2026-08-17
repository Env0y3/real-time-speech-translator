import asyncio
import threading

from PySide6.QtCore import QThread, Signal

from core.pipeline import NormalPipelineOptions, run_normal_pipeline


class PipelineThread(QThread):
    """在专用线程中运行 asyncio，Qt 主线程只处理界面事件。"""

    event_received = Signal(dict)
    pipeline_error = Signal(str)

    def __init__(
        self,
        options: NormalPipelineOptions,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.options = options
        self._stop_signal = threading.Event()

    def request_stop(self) -> None:
        self._stop_signal.set()

    def run(self) -> None:
        try:
            asyncio.run(
                run_normal_pipeline(
                    options=self.options,
                    stop_signal=self._stop_signal,
                    event_callback=self.event_received.emit,
                )
            )
        except Exception as error:
            message = str(error).strip() or type(error).__name__
            self.pipeline_error.emit(message)
