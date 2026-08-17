import os
from datetime import datetime

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEEPSEEK_MODEL,
    LOCAL_MONITOR_DEVICE,
    LOCAL_MONITOR_ENABLED,
    NORMAL_ASR_PROVIDER,
    SENSEVOICE_MODEL_NAME,
    TTS_PROVIDER,
    TRANSLATION_OUTPUT_DEVICE,
)
from core.audio_devices import format_device, list_audio_devices
from core.pipeline import (
    NormalPipelineOptions,
    validate_normal_pipeline_options,
)
from gui.pipeline_thread import PipelineThread


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline_thread: PipelineThread | None = None
        self._stop_requested = False
        self._close_requested = False
        self._status = "Idle"

        self.setWindowTitle("Real-Time Speech Translator")
        self.resize(900, 650)
        self.setMinimumSize(760, 580)
        self._build_ui()
        self._apply_style()
        self.refresh_api_status()
        self.refresh_devices(use_config_defaults=True)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("Real-Time Speech Translator")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.DemiBold))
        header.addWidget(title)
        header.addStretch()
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.status_label)
        root.addLayout(header)

        top_grid = QGridLayout()
        top_grid.setHorizontalSpacing(14)
        top_grid.addWidget(self._build_audio_group(), 0, 0)
        top_grid.addWidget(self._build_pipeline_group(), 0, 1)
        top_grid.setColumnStretch(0, 3)
        top_grid.setColumnStretch(1, 2)
        root.addLayout(top_grid)

        text_grid = QGridLayout()
        text_grid.setHorizontalSpacing(14)
        self.chinese_text = self._create_text_area("等待识别中文语音…")
        self.english_text = self._create_text_area("Waiting for translation…")
        text_grid.addWidget(QLabel("Chinese"), 0, 0)
        text_grid.addWidget(QLabel("English"), 0, 1)
        text_grid.addWidget(self.chinese_text, 1, 0)
        text_grid.addWidget(self.english_text, 1, 1)
        text_grid.setColumnStretch(0, 1)
        text_grid.setColumnStretch(1, 1)
        root.addLayout(text_grid, 1)

        latency_frame = QFrame()
        latency_frame.setObjectName("latencyFrame")
        latency_layout = QHBoxLayout(latency_frame)
        latency_layout.setContentsMargins(14, 10, 14, 10)
        self.latency_total = QLabel("Speech End → First Playback: —")
        self.latency_total.setObjectName("latencyTotal")
        self.latency_detail = QLabel(
            "Endpoint: —   ASR: —   Translation: —   TTS: —"
        )
        latency_layout.addWidget(self.latency_total)
        latency_layout.addStretch()
        latency_layout.addWidget(self.latency_detail)
        root.addWidget(latency_frame)

        self.error_label = QLabel("")
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        root.addWidget(self.error_label)

        controls = QHBoxLayout()
        controls.addStretch()
        self.start_button = QPushButton("Start Translation")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_translation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_translation)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        root.addLayout(controls)

        self.setCentralWidget(central)
        self._set_status("Idle")

    def _build_audio_group(self) -> QGroupBox:
        group = QGroupBox("Audio Devices")
        layout = QFormLayout(group)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.monitor_checkbox = QCheckBox("Enable Local Monitor")
        self.monitor_checkbox.setChecked(LOCAL_MONITOR_ENABLED)
        self.monitor_checkbox.toggled.connect(self._update_monitor_enabled)
        self.monitor_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh Devices")
        self.refresh_button.clicked.connect(self.refresh_devices)

        layout.addRow("Input Microphone:", self.input_combo)
        layout.addRow("Translation Output:", self.output_combo)
        layout.addRow("", self.monitor_checkbox)
        layout.addRow("Monitor Device:", self.monitor_combo)
        layout.addRow("", self.refresh_button)
        return group

    def _build_pipeline_group(self) -> QGroupBox:
        group = QGroupBox("Pipeline")
        layout = QVBoxLayout(group)
        provider_form = QFormLayout()
        asr_name = (
            "SenseVoiceSmall"
            if NORMAL_ASR_PROVIDER == "sensevoice"
            else NORMAL_ASR_PROVIDER
        )
        provider_form.addRow("ASR:", QLabel(asr_name))
        provider_form.addRow("Translation:", QLabel(f"DeepSeek ({DEEPSEEK_MODEL})"))
        provider_form.addRow("TTS:", QLabel(TTS_PROVIDER.title()))
        layout.addLayout(provider_form)
        layout.addSpacing(6)
        self.deepseek_api_label = QLabel()
        self.elevenlabs_api_label = QLabel()
        layout.addWidget(self.deepseek_api_label)
        layout.addWidget(self.elevenlabs_api_label)
        layout.addStretch()
        return group

    @staticmethod
    def _create_text_area(placeholder: str) -> QPlainTextEdit:
        area = QPlainTextEdit()
        area.setReadOnly(True)
        area.setPlaceholderText(placeholder)
        area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        return area

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #f4f6f8; }
            QLabel { color: #263238; }
            QGroupBox {
                background: white;
                border: 1px solid #dce2e7;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QComboBox, QPlainTextEdit {
                background: white;
                border: 1px solid #cbd4dc;
                border-radius: 5px;
                padding: 6px;
            }
            QComboBox { min-height: 24px; }
            QPlainTextEdit { font-size: 14px; }
            QPushButton {
                min-height: 34px;
                padding: 0 16px;
                border: 1px solid #b8c2cc;
                border-radius: 6px;
                background: white;
            }
            QPushButton:hover { background: #eef3f7; }
            QPushButton:disabled { color: #9aa5ad; background: #edf0f2; }
            #startButton { background: #1976d2; color: white; border: none; }
            #startButton:hover { background: #1565c0; }
            #titleLabel { color: #15202b; }
            #statusLabel { font-size: 14px; font-weight: 600; }
            #latencyFrame { background: white; border: 1px solid #dce2e7; border-radius: 7px; }
            #latencyTotal { font-weight: 600; }
            #errorLabel { color: #b3261e; background: #fdecea; padding: 8px; border-radius: 5px; }
            """
        )

    def refresh_api_status(self) -> None:
        deepseek_ready = bool(os.environ.get("DEEPSEEK_API_KEY"))
        elevenlabs_ready = bool(os.environ.get("ELEVENLABS_API_KEY"))
        self.deepseek_api_label.setText(
            f"DeepSeek API: {'✓ Ready' if deepseek_ready else '✗ Missing'}"
        )
        self.elevenlabs_api_label.setText(
            "ElevenLabs API: "
            f"{'✓ Ready' if elevenlabs_ready else '✗ Missing'}"
        )
        ready_style = "color: #207a3c;"
        missing_style = "color: #b3261e;"
        self.deepseek_api_label.setStyleSheet(
            ready_style if deepseek_ready else missing_style
        )
        self.elevenlabs_api_label.setStyleSheet(
            ready_style if elevenlabs_ready else missing_style
        )

    def refresh_devices(self, checked=False, use_config_defaults=False) -> None:
        del checked
        current_input = None if use_config_defaults else self.input_combo.currentData()
        current_output = (
            TRANSLATION_OUTPUT_DEVICE
            if use_config_defaults
            else self.output_combo.currentData()
        )
        current_monitor = (
            LOCAL_MONITOR_DEVICE
            if use_config_defaults
            else self.monitor_combo.currentData()
        )
        try:
            devices = list_audio_devices()
            input_devices = [d for d in devices if d.max_input_channels > 0]
            output_devices = [d for d in devices if d.max_output_channels > 0]
            self._populate_device_combo(
                self.input_combo,
                input_devices,
                current_input,
            )
            self._populate_device_combo(
                self.output_combo,
                output_devices,
                current_output,
            )
            self._populate_device_combo(
                self.monitor_combo,
                output_devices,
                current_monitor,
            )
            self._clear_error()
        except Exception as error:
            self._show_error(f"Device refresh failed: {error}")
        self._update_monitor_enabled(self.monitor_checkbox.isChecked())

    @staticmethod
    def _populate_device_combo(
        combo: QComboBox,
        devices,
        selected_device: int | str | None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("System Default", None)
        for device in devices:
            combo.addItem(format_device(device), device.index)

        selected_index = 0
        if isinstance(selected_device, int):
            found_index = combo.findData(selected_device, role=Qt.ItemDataRole.UserRole)
            if found_index >= 0:
                selected_index = found_index
        elif isinstance(selected_device, str):
            query = selected_device.casefold()
            for index in range(1, combo.count()):
                if query in combo.itemText(index).casefold():
                    selected_index = index
                    break
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _runtime_options(self) -> NormalPipelineOptions:
        return NormalPipelineOptions(
            input_device=self.input_combo.currentData(),
            translation_output_device=self.output_combo.currentData(),
            local_monitor_enabled=self.monitor_checkbox.isChecked(),
            monitor_device=self.monitor_combo.currentData(),
        )

    def start_translation(self) -> None:
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            return
        self.refresh_api_status()
        self._clear_error()
        options = self._runtime_options()
        try:
            validate_normal_pipeline_options(options)
        except Exception as error:
            self._set_status("Error")
            self._show_error(str(error))
            return

        self._stop_requested = False
        self.pipeline_thread = PipelineThread(options, self)
        self.pipeline_thread.event_received.connect(self._handle_pipeline_event)
        self.pipeline_thread.pipeline_error.connect(self._handle_pipeline_error)
        self.pipeline_thread.finished.connect(self._pipeline_finished)
        self._set_running_controls(True)
        self._set_status("Running")
        self.pipeline_thread.start()

    def stop_translation(self) -> None:
        if self.pipeline_thread is None or not self.pipeline_thread.isRunning():
            return
        self._stop_requested = True
        self._set_status("Stopping")
        self.stop_button.setEnabled(False)
        self.pipeline_thread.request_stop()

    def _handle_pipeline_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "status":
            self._set_status(str(event.get("status", "Running")))
        elif event_type == "asr_result":
            self._set_status("Recognizing")
            self._append_text(self.chinese_text, str(event.get("text", "")))
        elif event_type == "translation_segment":
            self._set_status("Translating")
            self._append_text(self.english_text, str(event.get("text", "")))
        elif event_type == "tts_started":
            self._set_status("Speaking")
        elif event_type == "trace_summary":
            self._update_latency(event)
        elif event_type == "error":
            self._set_status("Error")
            self._show_error(str(event.get("message", "Pipeline error")))

    def _handle_pipeline_error(self, message: str) -> None:
        self._set_status("Error")
        self._show_error(message)

    def _pipeline_finished(self) -> None:
        failed = self._status == "Error" and not self._stop_requested
        thread = self.pipeline_thread
        self.pipeline_thread = None
        self._set_running_controls(False)
        if not failed:
            self._set_status("Idle")
        if thread is not None:
            thread.deleteLater()
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    def _set_running_controls(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.monitor_checkbox.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.monitor_combo.setEnabled(
            not running and self.monitor_checkbox.isChecked()
        )

    def _update_monitor_enabled(self, enabled: bool) -> None:
        running = self.pipeline_thread is not None and self.pipeline_thread.isRunning()
        self.monitor_combo.setEnabled(enabled and not running)

    def _set_status(self, status: str) -> None:
        self._status = status
        color = {
            "Idle": "#66727c",
            "Listening": "#207a3c",
            "Recognizing": "#7b5c00",
            "Translating": "#1555a2",
            "Speaking": "#7a3ea1",
            "Error": "#b3261e",
            "Stopping": "#66727c",
            "Running": "#207a3c",
        }.get(status, "#66727c")
        self.status_label.setText(f"● {status}")
        self.status_label.setStyleSheet(f"color: {color};")

    @staticmethod
    def _append_text(area: QPlainTextEdit, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        area.appendPlainText(f"[{timestamp}] {clean_text}")
        document = area.document()
        while document.blockCount() > 30:
            cursor = area.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
        area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())

    def _update_latency(self, event: dict) -> None:
        total = event.get("speech_end_to_first_playback_ms")
        endpoint = event.get("endpoint_wait_ms")
        asr = event.get("asr_stage_ms")
        translation = event.get("translation_after_asr_ms")
        tts = event.get("tts_after_first_segment_ms")
        self.latency_total.setText(
            "Speech End → First Playback: " + self._format_latency(total)
        )
        self.latency_detail.setText(
            "Endpoint: "
            f"{self._format_latency(endpoint)}   ASR: {self._format_latency(asr)}   "
            f"Translation: {self._format_latency(translation)}   "
            f"TTS: {self._format_latency(tts)}"
        )

    @staticmethod
    def _format_latency(value) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        if value >= 1000:
            return f"{value / 1000:.2f} s"
        return f"{value:.0f} ms"

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()

    def _clear_error(self) -> None:
        self.error_label.clear()
        self.error_label.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            self._close_requested = True
            self.stop_translation()
            event.ignore()
            return
        event.accept()
