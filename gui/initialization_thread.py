from PySide6.QtCore import QThread, Signal

from asr.sensevoice_asr import load_sensevoice_model
from config import MODEL_PATH, NORMAL_ASR_PROVIDER


class InitializationThread(QThread):
    """在后台执行真实 ASR 初始化，避免阻塞 Qt Event Loop。"""

    progress = Signal(str)
    initialization_ready = Signal()
    initialization_error = Signal(str, str)

    def run(self) -> None:
        try:
            self.progress.emit("checking_model")
            if NORMAL_ASR_PROVIDER == "sensevoice":
                self.progress.emit("loading_model")
                load_sensevoice_model()
            elif NORMAL_ASR_PROVIDER == "vosk":
                if not MODEL_PATH.exists():
                    self.initialization_error.emit("model_not_found", "")
                    return
            self.initialization_ready.emit()
        except FileNotFoundError:
            self.initialization_error.emit("model_not_found", "")
        except Exception as error:
            detail = str(error).strip() or type(error).__name__
            self.initialization_error.emit("model_load_failed", detail)
