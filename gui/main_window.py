import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont
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
    TTS_PROVIDER,
    TRANSLATION_OUTPUT_DEVICE,
)
from core.audio_devices import AudioDeviceInfo, format_device, list_audio_devices
from core.pipeline import (
    NormalPipelineOptions,
    validate_normal_pipeline_options,
)
from gui.i18n import normalize_language, tr
from gui.initialization_thread import InitializationThread
from gui.pipeline_thread import PipelineThread
from gui.settings import (
    default_settings_path,
    load_user_settings,
    save_user_settings,
)


DEVICE_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1
MAX_TEXT_HISTORY = 100


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings_path: Path | None = None,
        auto_initialize: bool = True,
    ) -> None:
        super().__init__()
        self.settings_path = settings_path or default_settings_path()
        self.user_settings = load_user_settings(self.settings_path)
        self.language = normalize_language(
            self.user_settings.get("ui_language")
        )
        self.pipeline_thread: PipelineThread | None = None
        self.initialization_thread: InitializationThread | None = None
        self._model_ready = False
        self._stop_requested = False
        self._close_requested = False
        self._settings_ready = False
        self._status = "LOADING"
        self._startup_detail_key = "initializing"
        self._notice_keys: set[str] = set()
        self._last_latency_event: dict = {}
        self._error_key: str | None = None
        self._error_values: dict = {}

        self.resize(940, 680)
        self.setMinimumSize(800, 600)
        self._build_ui()
        self._apply_style()
        self.refresh_api_status()
        self.refresh_devices(use_saved_settings=True)
        self._settings_ready = True
        self._connect_settings_signals()
        self._retranslate_ui()
        self._set_status("LOADING")
        self.start_button.setEnabled(False)

        if auto_initialize:
            QTimer.singleShot(0, self._start_initialization)

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        header = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setObjectName("titleLabel")
        self.title_label.setFont(
            QFont("Segoe UI", 20, QFont.Weight.DemiBold)
        )
        header.addWidget(self.title_label)
        header.addStretch()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        language_index = self.language_combo.findData(self.language)
        self.language_combo.setCurrentIndex(max(0, language_index))
        self.status_label = QLabel()
        self.status_label.setObjectName("statusLabel")
        header.addWidget(self.language_label)
        header.addWidget(self.language_combo)
        header.addSpacing(14)
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
        self.chinese_label = QLabel()
        self.english_label = QLabel()
        self.chinese_text = self._create_text_area()
        self.english_text = self._create_text_area()
        text_grid.addWidget(self.chinese_label, 0, 0)
        text_grid.addWidget(self.english_label, 0, 1)
        text_grid.addWidget(self.chinese_text, 1, 0)
        text_grid.addWidget(self.english_text, 1, 1)
        text_grid.setColumnStretch(0, 1)
        text_grid.setColumnStretch(1, 1)
        root.addLayout(text_grid, 1)

        latency_frame = QFrame()
        latency_frame.setObjectName("latencyFrame")
        latency_layout = QHBoxLayout(latency_frame)
        latency_layout.setContentsMargins(14, 10, 14, 10)
        self.latency_heading = QLabel()
        self.latency_heading.setObjectName("latencyHeading")
        self.latency_total = QLabel()
        self.latency_total.setObjectName("latencyTotal")
        self.latency_detail = QLabel()
        latency_layout.addWidget(self.latency_heading)
        latency_layout.addSpacing(8)
        latency_layout.addWidget(self.latency_total)
        latency_layout.addStretch()
        latency_layout.addWidget(self.latency_detail)
        root.addWidget(latency_frame)

        self.notice_label = QLabel()
        self.notice_label.setObjectName("noticeLabel")
        self.notice_label.setWordWrap(True)
        self.notice_label.hide()
        root.addWidget(self.notice_label)

        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        root.addWidget(self.error_label)

        controls = QHBoxLayout()
        controls.addStretch()
        self.start_button = QPushButton()
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_translation)
        self.stop_button = QPushButton()
        self.stop_button.clicked.connect(self.stop_translation)
        self.stop_button.setEnabled(False)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        root.addLayout(controls)

        self.setCentralWidget(central)

    def _build_audio_group(self) -> QGroupBox:
        self.audio_group = QGroupBox()
        layout = QFormLayout(self.audio_group)
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.input_label = QLabel()
        self.output_label = QLabel()
        self.monitor_label = QLabel()
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        self.monitor_checkbox = QCheckBox()
        self.monitor_checkbox.setChecked(
            bool(
                self.user_settings.get(
                    "local_monitor_enabled",
                    LOCAL_MONITOR_ENABLED,
                )
            )
        )
        self.monitor_combo = QComboBox()
        self.refresh_button = QPushButton()
        self.refresh_button.clicked.connect(self.refresh_devices)

        layout.addRow(self.input_label, self.input_combo)
        layout.addRow(self.output_label, self.output_combo)
        layout.addRow("", self.monitor_checkbox)
        layout.addRow(self.monitor_label, self.monitor_combo)
        layout.addRow("", self.refresh_button)
        return self.audio_group

    def _build_pipeline_group(self) -> QGroupBox:
        self.pipeline_group = QGroupBox()
        layout = QVBoxLayout(self.pipeline_group)
        provider_form = QFormLayout()
        self.asr_label = QLabel()
        self.translation_label = QLabel()
        self.tts_label = QLabel()
        asr_name = (
            "SenseVoiceSmall"
            if NORMAL_ASR_PROVIDER == "sensevoice"
            else NORMAL_ASR_PROVIDER
        )
        provider_form.addRow(self.asr_label, QLabel(asr_name))
        provider_form.addRow(
            self.translation_label,
            QLabel(f"DeepSeek ({DEEPSEEK_MODEL})"),
        )
        provider_form.addRow(self.tts_label, QLabel(TTS_PROVIDER.title()))
        layout.addLayout(provider_form)
        layout.addSpacing(6)
        self.deepseek_api_label = QLabel()
        self.elevenlabs_api_label = QLabel()
        self.startup_detail_label = QLabel()
        self.startup_detail_label.setWordWrap(True)
        self.startup_detail_label.setObjectName("startupDetailLabel")
        layout.addWidget(self.deepseek_api_label)
        layout.addWidget(self.elevenlabs_api_label)
        layout.addSpacing(5)
        layout.addWidget(self.startup_detail_label)
        layout.addStretch()
        return self.pipeline_group

    @staticmethod
    def _create_text_area() -> QPlainTextEdit:
        area = QPlainTextEdit()
        area.setReadOnly(True)
        area.document().setMaximumBlockCount(MAX_TEXT_HISTORY)
        area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        return area

    def _connect_settings_signals(self) -> None:
        self.language_combo.currentIndexChanged.connect(
            self._language_changed
        )
        self.input_combo.currentIndexChanged.connect(self._save_settings)
        self.output_combo.currentIndexChanged.connect(self._save_settings)
        self.monitor_combo.currentIndexChanged.connect(self._save_settings)
        self.monitor_checkbox.toggled.connect(
            self._monitor_setting_changed
        )

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
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
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
            #startButton:disabled {
                color: #9aa5ad;
                background: #dfe4e8;
            }
            #titleLabel { color: #15202b; }
            #statusLabel { font-size: 14px; font-weight: 600; }
            #startupDetailLabel { color: #54636e; }
            #latencyFrame {
                background: white;
                border: 1px solid #dce2e7;
                border-radius: 7px;
            }
            #latencyTotal { font-weight: 600; }
            #latencyHeading { font-weight: 600; }
            #noticeLabel {
                color: #765800;
                background: #fff7d6;
                padding: 8px;
                border-radius: 5px;
            }
            #errorLabel {
                color: #b3261e;
                background: #fdecea;
                padding: 8px;
                border-radius: 5px;
            }
            """
        )

    def _language_changed(self) -> None:
        self.language = normalize_language(self.language_combo.currentData())
        self._retranslate_ui()
        self._save_settings()

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(tr("window_title", self.language))
        self.title_label.setText(tr("window_title", self.language))
        self.language_label.setText(tr("language", self.language))
        self.audio_group.setTitle(tr("audio_devices", self.language))
        self.input_label.setText(tr("input_microphone", self.language))
        self.output_label.setText(tr("translation_output", self.language))
        self.monitor_checkbox.setText(
            tr("enable_local_monitor", self.language)
        )
        self.monitor_label.setText(tr("monitor_device", self.language))
        self.refresh_button.setText(tr("refresh_devices", self.language))
        self.pipeline_group.setTitle(tr("pipeline", self.language))
        self.asr_label.setText(tr("asr", self.language))
        self.translation_label.setText(tr("translation", self.language))
        self.tts_label.setText(tr("tts", self.language))
        self.chinese_label.setText(tr("chinese", self.language))
        self.english_label.setText(tr("english", self.language))
        self.latency_heading.setText(tr("latency", self.language))
        self.chinese_text.setPlaceholderText(
            tr("chinese_placeholder", self.language)
        )
        self.english_text.setPlaceholderText(
            tr("english_placeholder", self.language)
        )
        self.start_button.setText(tr("start_translation", self.language))
        self.stop_button.setText(tr("stop_translation", self.language))
        self._translate_default_device_items()
        self.refresh_api_status()
        self._set_status(self._status)
        self._set_startup_detail(self._startup_detail_key)
        self._update_latency(self._last_latency_event)
        self._update_notices()
        self._refresh_error_text()

    def _translate_default_device_items(self) -> None:
        for combo in (
            self.input_combo,
            self.output_combo,
            self.monitor_combo,
        ):
            if combo.count():
                combo.setItemText(0, tr("system_default", self.language))

    def refresh_api_status(self) -> None:
        deepseek_ready = bool(os.environ.get("DEEPSEEK_API_KEY"))
        elevenlabs_ready = bool(os.environ.get("ELEVENLABS_API_KEY"))
        self.deepseek_api_label.setText(
            "DeepSeek API: "
            + tr(
                "api_ready" if deepseek_ready else "api_missing",
                self.language,
            )
        )
        self.elevenlabs_api_label.setText(
            "ElevenLabs API: "
            + tr(
                "api_ready" if elevenlabs_ready else "api_missing",
                self.language,
            )
        )
        ready_style = "color: #207a3c;"
        missing_style = "color: #b3261e;"
        self.deepseek_api_label.setStyleSheet(
            ready_style if deepseek_ready else missing_style
        )
        self.elevenlabs_api_label.setStyleSheet(
            ready_style if elevenlabs_ready else missing_style
        )

    def _saved_device_selection(
        self,
        key: str,
        config_default: int | str | None,
    ):
        saved = self.user_settings.get(key)
        if saved is not None:
            return saved
        return {"index": config_default, "name": None}

    def refresh_devices(
        self,
        checked: bool = False,
        use_saved_settings: bool = False,
    ) -> None:
        del checked
        if use_saved_settings:
            input_selection = self._saved_device_selection(
                "input_device",
                None,
            )
            output_selection = self._saved_device_selection(
                "translation_output_device",
                TRANSLATION_OUTPUT_DEVICE,
            )
            monitor_selection = self._saved_device_selection(
                "monitor_device",
                LOCAL_MONITOR_DEVICE,
            )
        else:
            input_selection = self._device_selection(self.input_combo)
            output_selection = self._device_selection(self.output_combo)
            monitor_selection = self._device_selection(self.monitor_combo)

        for key in (
            "saved_input_missing",
            "saved_output_missing",
            "saved_monitor_missing",
        ):
            self._notice_keys.discard(key)

        try:
            devices = list_audio_devices()
            input_devices = [d for d in devices if d.max_input_channels > 0]
            output_devices = [d for d in devices if d.max_output_channels > 0]
            input_restored = self._populate_device_combo(
                self.input_combo,
                input_devices,
                input_selection,
            )
            output_restored = self._populate_device_combo(
                self.output_combo,
                output_devices,
                output_selection,
            )
            monitor_restored = self._populate_device_combo(
                self.monitor_combo,
                output_devices,
                monitor_selection,
            )
            if not input_restored:
                self._notice_keys.add("saved_input_missing")
            if not output_restored:
                self._notice_keys.add("saved_output_missing")
            if not monitor_restored:
                self._notice_keys.add("saved_monitor_missing")

            virtual_detected = any(
                "cable input" in device.name.casefold()
                for device in output_devices
            )
            if virtual_detected:
                self._notice_keys.discard("virtual_device_missing")
            else:
                self._notice_keys.add("virtual_device_missing")
            self._clear_error()
        except Exception as error:
            self._show_error_key(
                "device_refresh_failed",
                detail=str(error),
            )
        self._update_monitor_enabled(self.monitor_checkbox.isChecked())
        self._update_notices()
        if self._settings_ready:
            self._save_settings()

    def _populate_device_combo(
        self,
        combo: QComboBox,
        devices: list[AudioDeviceInfo],
        selected_device,
    ) -> bool:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr("system_default", self.language), None)
        for device in devices:
            combo.addItem(format_device(device), device.index)
            combo.setItemData(combo.count() - 1, device.name, DEVICE_NAME_ROLE)

        selected_index, selected_name = self._selection_parts(selected_device)
        explicit_selection = selected_index is not None or bool(selected_name)
        restored_combo_index = 0
        name_matches = []
        if selected_name:
            name_matches = [
                index
                for index in range(1, combo.count())
                if combo.itemData(index, DEVICE_NAME_ROLE).casefold()
                == str(selected_name).casefold()
            ]
        if name_matches:
            matching_saved_index = next(
                (
                    index
                    for index in name_matches
                    if combo.itemData(index) == selected_index
                ),
                None,
            )
            restored_combo_index = matching_saved_index or name_matches[0]
        elif selected_index is not None:
            found_index = combo.findData(selected_index)
            if found_index >= 0:
                restored_combo_index = found_index

        combo.setCurrentIndex(restored_combo_index)
        combo.blockSignals(False)
        return not explicit_selection or restored_combo_index > 0

    @staticmethod
    def _selection_parts(selection) -> tuple[int | None, str | None]:
        if isinstance(selection, dict):
            index = selection.get("index")
            name = selection.get("name")
            return (
                index if isinstance(index, int) and not isinstance(index, bool) else None,
                name if isinstance(name, str) and name.strip() else None,
            )
        if isinstance(selection, int) and not isinstance(selection, bool):
            return selection, None
        if isinstance(selection, str) and selection.strip():
            return None, selection.strip()
        return None, None

    @staticmethod
    def _device_selection(combo: QComboBox) -> dict:
        index = combo.currentData()
        name = combo.currentData(DEVICE_NAME_ROLE) if index is not None else None
        return {"index": index, "name": name}

    def _monitor_setting_changed(self, enabled: bool) -> None:
        self._update_monitor_enabled(enabled)
        self._save_settings()

    def _save_settings(self) -> None:
        if not self._settings_ready:
            return
        settings = {
            "ui_language": self.language,
            "input_device": self._device_selection(self.input_combo),
            "translation_output_device": self._device_selection(
                self.output_combo
            ),
            "local_monitor_enabled": self.monitor_checkbox.isChecked(),
            "monitor_device": self._device_selection(self.monitor_combo),
        }
        self.user_settings = settings
        try:
            save_user_settings(settings, self.settings_path)
        except OSError as error:
            print(f"[User Settings Warning] {type(error).__name__}")

    def _start_initialization(self) -> None:
        if (
            self.initialization_thread is not None
            and self.initialization_thread.isRunning()
        ):
            return
        self._model_ready = False
        self._set_status("LOADING")
        self._set_startup_detail("checking_model")
        self.start_button.setEnabled(False)
        self.initialization_thread = InitializationThread(self)
        self.initialization_thread.progress.connect(
            self._set_startup_detail
        )
        self.initialization_thread.initialization_ready.connect(
            self._on_initialization_ready
        )
        self.initialization_thread.initialization_error.connect(
            self._on_initialization_error
        )
        self.initialization_thread.finished.connect(
            self._initialization_finished
        )
        self.initialization_thread.start()

    def _on_initialization_ready(self) -> None:
        self._model_ready = True
        self._clear_error()
        self._set_startup_detail("ready_detail")
        self._set_status("READY")
        self.start_button.setEnabled(True)

    def _on_initialization_error(self, key: str, detail: str) -> None:
        self._model_ready = False
        self._set_status("ERROR")
        self.start_button.setEnabled(False)
        values = {"detail": detail} if detail else {}
        self._show_error_key(key, **values)

    def _initialization_finished(self) -> None:
        thread = self.initialization_thread
        self.initialization_thread = None
        if thread is not None:
            thread.deleteLater()
        if self._close_requested and self.pipeline_thread is None:
            QTimer.singleShot(0, self.close)

    def _set_startup_detail(self, key: str) -> None:
        self._startup_detail_key = key
        self.startup_detail_label.setText(tr(key, self.language))

    def _runtime_options(self) -> NormalPipelineOptions:
        return NormalPipelineOptions(
            input_device=self.input_combo.currentData(),
            translation_output_device=self.output_combo.currentData(),
            local_monitor_enabled=self.monitor_checkbox.isChecked(),
            monitor_device=self.monitor_combo.currentData(),
        )

    def start_translation(self) -> None:
        if not self._model_ready:
            return
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            return
        self.refresh_api_status()
        self._clear_error()
        if not os.environ.get("DEEPSEEK_API_KEY"):
            self._set_status("ERROR")
            self._show_error_key("missing_deepseek_api")
            return
        if TTS_PROVIDER == "elevenlabs" and not os.environ.get(
            "ELEVENLABS_API_KEY"
        ):
            self._set_status("ERROR")
            self._show_error_key("missing_elevenlabs_api")
            return

        options = self._runtime_options()
        try:
            validate_normal_pipeline_options(options)
        except Exception as error:
            self._set_status("ERROR")
            self._show_error_key("pipeline_error", detail=str(error))
            return

        self._save_settings()
        self._stop_requested = False
        self.pipeline_thread = PipelineThread(options, self)
        self.pipeline_thread.event_received.connect(
            self._handle_pipeline_event
        )
        self.pipeline_thread.pipeline_error.connect(
            self._handle_pipeline_error
        )
        self.pipeline_thread.finished.connect(self._pipeline_finished)
        self._set_running_controls(True)
        self._set_status("LISTENING")
        self.pipeline_thread.start()

    def stop_translation(self) -> None:
        if self.pipeline_thread is None or not self.pipeline_thread.isRunning():
            return
        self._stop_requested = True
        self._set_status("STOPPING")
        self.stop_button.setEnabled(False)
        self.pipeline_thread.request_stop()

    def _handle_pipeline_event(self, event: dict) -> None:
        event_type = event.get("type")
        if event_type == "status":
            status = self._canonical_status(event.get("status"))
            if not (self._stop_requested and status == "READY"):
                self._set_status(status)
        elif event_type == "asr_result":
            self._set_status("RECOGNIZING")
            self._append_text(
                self.chinese_text,
                str(event.get("text", "")),
            )
        elif event_type == "translation_segment":
            self._set_status("TRANSLATING")
            self._append_text(
                self.english_text,
                str(event.get("text", "")),
            )
        elif event_type == "tts_started":
            self._set_status("SPEAKING")
        elif event_type == "trace_summary":
            self._update_latency(event)
        elif event_type == "error":
            self._set_status("ERROR")
            self._show_error_key(
                "pipeline_error",
                detail=str(event.get("message", "")),
            )

    def _handle_pipeline_error(self, message: str) -> None:
        self._set_status("ERROR")
        self._show_error_key("pipeline_error", detail=message)

    def _pipeline_finished(self) -> None:
        failed = self._status == "ERROR" and not self._stop_requested
        thread = self.pipeline_thread
        self.pipeline_thread = None
        self._set_running_controls(False)
        if not failed:
            self._set_status("READY")
        self._stop_requested = False
        if thread is not None:
            thread.deleteLater()
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    def _set_running_controls(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self._model_ready)
        self.stop_button.setEnabled(running)
        self.input_combo.setEnabled(not running)
        self.output_combo.setEnabled(not running)
        self.monitor_checkbox.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.language_combo.setEnabled(True)
        self.monitor_combo.setEnabled(
            not running and self.monitor_checkbox.isChecked()
        )

    def _update_monitor_enabled(self, enabled: bool) -> None:
        running = (
            self.pipeline_thread is not None
            and self.pipeline_thread.isRunning()
        )
        self.monitor_combo.setEnabled(enabled and not running)

    @staticmethod
    def _canonical_status(status) -> str:
        normalized = str(status or "READY").strip().upper()
        aliases = {
            "RUNNING": "LISTENING",
        }
        return aliases.get(normalized, normalized)

    def _set_status(self, status: str) -> None:
        canonical = self._canonical_status(status)
        valid = {
            "IDLE",
            "LOADING",
            "READY",
            "LISTENING",
            "RECOGNIZING",
            "TRANSLATING",
            "SPEAKING",
            "STOPPING",
            "ERROR",
        }
        self._status = canonical if canonical in valid else "READY"
        color = {
            "IDLE": "#66727c",
            "LOADING": "#7b5c00",
            "READY": "#207a3c",
            "LISTENING": "#207a3c",
            "RECOGNIZING": "#7b5c00",
            "TRANSLATING": "#1555a2",
            "SPEAKING": "#7a3ea1",
            "STOPPING": "#66727c",
            "ERROR": "#b3261e",
        }[self._status]
        status_text = tr(
            f"status_{self._status.lower()}",
            self.language,
        )
        self.status_label.setText(
            f"{tr('status', self.language)}: ● {status_text}"
        )
        self.status_label.setStyleSheet(f"color: {color};")

    @staticmethod
    def _append_text(area: QPlainTextEdit, text: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        area.appendPlainText(f"[{timestamp}] {clean_text}")
        area.verticalScrollBar().setValue(
            area.verticalScrollBar().maximum()
        )

    def _update_latency(self, event: dict) -> None:
        self._last_latency_event = dict(event)
        total = self._format_latency(
            event.get("speech_end_to_first_playback_ms")
        )
        endpoint = self._format_latency(event.get("endpoint_wait_ms"))
        asr = self._format_latency(event.get("asr_stage_ms"))
        translation = self._format_latency(
            event.get("translation_after_asr_ms")
        )
        tts = self._format_latency(
            event.get("tts_after_first_segment_ms")
        )
        self.latency_total.setText(
            tr(
                "speech_end_first_playback",
                self.language,
                value=total,
            )
        )
        self.latency_detail.setText(
            "   ".join(
                [
                    tr("endpoint", self.language, value=endpoint),
                    tr("asr_latency", self.language, value=asr),
                    tr(
                        "translation_latency",
                        self.language,
                        value=translation,
                    ),
                    tr("tts_latency", self.language, value=tts),
                ]
            )
        )

    @staticmethod
    def _format_latency(value) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        if value >= 1000:
            return f"{value / 1000:.2f} s"
        return f"{value:.0f} ms"

    def _update_notices(self) -> None:
        messages = [
            tr(key, self.language)
            for key in sorted(self._notice_keys)
        ]
        self.notice_label.setText("\n".join(messages))
        self.notice_label.setVisible(bool(messages))

    def _show_error_key(self, key: str, **values) -> None:
        self._error_key = key
        self._error_values = values
        self._refresh_error_text()

    def _refresh_error_text(self) -> None:
        if self._error_key is None:
            return
        self.error_label.setText(
            tr(self._error_key, self.language, **self._error_values)
        )
        self.error_label.show()

    def _clear_error(self) -> None:
        self._error_key = None
        self._error_values = {}
        self.error_label.clear()
        self.error_label.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self.pipeline_thread is not None and self.pipeline_thread.isRunning():
            self._close_requested = True
            self.stop_translation()
            event.ignore()
            return
        if (
            self.initialization_thread is not None
            and self.initialization_thread.isRunning()
        ):
            self._close_requested = True
            self._set_status("STOPPING")
            self.start_button.setEnabled(False)
            event.ignore()
            return
        event.accept()
