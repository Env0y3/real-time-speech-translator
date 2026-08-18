import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from core.audio_devices import AudioDeviceInfo
    from gui.main_window import MAX_TEXT_HISTORY, MainWindow

    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "user_settings.json"
        self.devices = [
            AudioDeviceInfo(0, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(1, "CABLE Input", 0, 2, 48_000),
            AudioDeviceInfo(2, "Speakers", 0, 2, 48_000),
        ]
        self.device_patch = patch(
            "gui.main_window.list_audio_devices",
            side_effect=lambda: self.devices,
        )
        self.device_patch.start()
        self.window = self._new_window()

    def tearDown(self) -> None:
        self.window.close()
        self.device_patch.stop()
        self.temp_dir.cleanup()

    def _new_window(self, auto_initialize: bool = False) -> MainWindow:
        return MainWindow(
            settings_path=self.settings_path,
            auto_initialize=auto_initialize,
        )

    def _make_ready(self, window: MainWindow | None = None) -> None:
        (window or self.window)._on_initialization_ready()

    def test_startup_is_loading_and_start_is_disabled(self) -> None:
        self.assertEqual(self.window._status, "LOADING")
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertIn("加载", self.window.status_label.text())

    def test_real_initialization_signal_enables_start(self) -> None:
        self.window.close()
        with patch(
            "gui.initialization_thread.load_sensevoice_model",
            return_value=object(),
        ) as load_model:
            self.window = self._new_window(auto_initialize=True)
            self.assertEqual(self.window._status, "LOADING")
            self._wait_until(lambda: self.window._status == "READY")

        load_model.assert_called_once_with()
        self.assertTrue(self.window.start_button.isEnabled())

    def test_language_switch_updates_ui_and_is_persisted(self) -> None:
        english_index = self.window.language_combo.findData("en")
        self.window.language_combo.setCurrentIndex(english_index)

        self.assertEqual(
            self.window.windowTitle(),
            "Real-Time Speech Translator",
        )
        self.assertEqual(self.window.audio_group.title(), "Audio Devices")
        self.assertEqual(self.window.start_button.text(), "Start Translation")
        self.assertIn("Loading", self.window.status_label.text())

        restored = self._new_window()
        try:
            self.assertEqual(restored.language, "en")
            self.assertEqual(restored.language_combo.currentData(), "en")
            self.assertEqual(restored.audio_group.title(), "Audio Devices")
        finally:
            restored.close()

    def test_device_lists_filter_by_channel_direction(self) -> None:
        self.assertEqual(self.window.input_combo.count(), 2)
        self.assertEqual(self.window.output_combo.count(), 3)
        self.assertIn("Microphone", self.window.input_combo.itemText(1))
        self.assertNotIn("Microphone", self.window.output_combo.itemText(1))

    def test_device_selection_is_saved_and_restored_by_name(self) -> None:
        self.window.input_combo.setCurrentIndex(
            self.window.input_combo.findData(0)
        )
        self.window.output_combo.setCurrentIndex(
            self.window.output_combo.findData(1)
        )
        self.window.monitor_checkbox.setChecked(True)
        self.window.monitor_combo.setCurrentIndex(
            self.window.monitor_combo.findData(2)
        )

        settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(settings["input_device"]["name"], "Microphone")
        self.assertEqual(
            settings["translation_output_device"]["name"],
            "CABLE Input",
        )
        self.assertNotIn("DEEPSEEK_API_KEY", settings)

        self.devices = [
            AudioDeviceInfo(8, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(9, "CABLE Input", 0, 2, 48_000),
            AudioDeviceInfo(10, "Speakers", 0, 2, 48_000),
        ]
        restored = self._new_window()
        try:
            self.assertEqual(restored.input_combo.currentData(), 8)
            self.assertEqual(restored.output_combo.currentData(), 9)
            self.assertEqual(restored.monitor_combo.currentData(), 10)
            self.assertTrue(restored.monitor_checkbox.isChecked())
        finally:
            restored.close()

    def test_missing_saved_device_falls_back_without_crashing(self) -> None:
        self.window.output_combo.setCurrentIndex(
            self.window.output_combo.findData(1)
        )
        self.devices = [
            AudioDeviceInfo(0, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(2, "Speakers", 0, 2, 48_000),
        ]

        restored = self._new_window()
        try:
            self.assertIsNone(restored.output_combo.currentData())
            self.assertFalse(restored.notice_label.isHidden())
            self.assertIn("输出设备不可用", restored.notice_label.text())
        finally:
            restored.close()

    def test_virtual_cable_absence_is_nonfatal_notice(self) -> None:
        self.devices = [
            AudioDeviceInfo(0, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(2, "Speakers", 0, 2, 48_000),
        ]
        self.window.refresh_devices()
        self.assertEqual(self.window._status, "LOADING")
        self.assertIn("虚拟音频设备", self.window.notice_label.text())

    def test_monitor_dropdown_tracks_checkbox(self) -> None:
        self.window.monitor_checkbox.setChecked(False)
        self.assertFalse(self.window.monitor_combo.isEnabled())
        self.window.monitor_checkbox.setChecked(True)
        self.assertTrue(self.window.monitor_combo.isEnabled())

    def test_missing_api_key_is_localized_without_starting(self) -> None:
        self._make_ready()
        with patch.dict(os.environ, {}, clear=True):
            self.window.refresh_api_status()
            self.window.start_translation()
        self.assertIn("未配置", self.window.deepseek_api_label.text())
        self.assertEqual(self.window._status, "ERROR")
        self.assertIsNone(self.window.pipeline_thread)

    def test_start_stop_and_restart_keep_ui_responsive(self) -> None:
        async def fake_pipeline(
            options,
            stop_signal,
            event_callback,
            interactive_stop=False,
        ) -> None:
            del options, interactive_stop
            event_callback({"type": "status", "status": "Listening"})
            while not stop_signal.is_set():
                await asyncio.sleep(0.01)
            event_callback({"type": "status", "status": "Ready"})

        self._make_ready()
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test",
                    "ELEVENLABS_API_KEY": "test",
                },
                clear=True,
            ),
            patch(
                "gui.main_window.validate_normal_pipeline_options",
                return_value=None,
            ),
            patch(
                "gui.pipeline_thread.run_normal_pipeline",
                side_effect=fake_pipeline,
            ),
        ):
            for _ in range(2):
                QTest.mouseClick(
                    self.window.start_button,
                    Qt.MouseButton.LeftButton,
                )
                self._wait_until(
                    lambda: self.window._status == "LISTENING"
                )
                self.assertFalse(self.window.start_button.isEnabled())
                self.assertTrue(self.window.stop_button.isEnabled())
                self.assertFalse(self.window.input_combo.isEnabled())

                QTest.mouseClick(
                    self.window.stop_button,
                    Qt.MouseButton.LeftButton,
                )
                self._wait_until(
                    lambda: self.window.pipeline_thread is None
                )
                self.assertEqual(self.window._status, "READY")
                self.assertTrue(self.window.start_button.isEnabled())
                self.assertFalse(self.window.stop_button.isEnabled())
                self.assertTrue(self.window.input_combo.isEnabled())

    def test_pipeline_events_update_text_and_trace_latency(self) -> None:
        self.window._handle_pipeline_event(
            {"type": "asr_result", "text": "我正在测试实时翻译。"}
        )
        self.window._handle_pipeline_event(
            {
                "type": "translation_segment",
                "text": "I'm testing real-time translation.",
            }
        )
        self.window._handle_pipeline_event(
            {
                "type": "trace_summary",
                "speech_end_to_first_playback_ms": 2430,
                "endpoint_wait_ms": 600,
                "asr_stage_ms": 160,
                "translation_after_asr_ms": 1080,
                "tts_after_first_segment_ms": 590,
            }
        )

        self.assertIn("我正在测试", self.window.chinese_text.toPlainText())
        self.assertIn("I'm testing", self.window.english_text.toPlainText())
        self.assertIn("2.43 s", self.window.latency_total.text())
        self.assertIn("600 ms", self.window.latency_detail.text())

    def test_text_history_is_limited(self) -> None:
        for index in range(MAX_TEXT_HISTORY + 25):
            self.window._append_text(
                self.window.chinese_text,
                f"message-{index}",
            )
        text = self.window.chinese_text.toPlainText()
        self.assertLessEqual(
            self.window.chinese_text.document().blockCount(),
            MAX_TEXT_HISTORY,
        )
        self.assertNotIn("message-0\n", text)
        self.assertIn("message-124", text)

    def test_close_while_running_requests_graceful_stop(self) -> None:
        async def fake_pipeline(
            options,
            stop_signal,
            event_callback,
            interactive_stop=False,
        ) -> None:
            del options, interactive_stop
            event_callback({"type": "status", "status": "Listening"})
            while not stop_signal.is_set():
                await asyncio.sleep(0.01)

        self._make_ready()
        with (
            patch.dict(
                os.environ,
                {
                    "DEEPSEEK_API_KEY": "test",
                    "ELEVENLABS_API_KEY": "test",
                },
                clear=True,
            ),
            patch(
                "gui.main_window.validate_normal_pipeline_options",
                return_value=None,
            ),
            patch(
                "gui.pipeline_thread.run_normal_pipeline",
                side_effect=fake_pipeline,
            ),
        ):
            self.window.show()
            self.window.start_translation()
            self._wait_until(lambda: self.window._status == "LISTENING")
            self.window.close()
            self.assertTrue(self.window._close_requested)
            self._wait_until(lambda: self.window.pipeline_thread is None)
            QApplication.processEvents()
            self.assertFalse(self.window.isVisible())

    @staticmethod
    def _wait_until(predicate, attempts: int = 200) -> None:
        for _ in range(attempts):
            QApplication.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        raise AssertionError("Timed out waiting for GUI state")


if __name__ == "__main__":
    unittest.main()
