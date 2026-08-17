import asyncio
import os
import unittest
from unittest.mock import patch

try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from core.audio_devices import AudioDeviceInfo
    from gui.main_window import MainWindow

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
        self.devices = [
            AudioDeviceInfo(0, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(1, "CABLE Input", 0, 2, 48_000),
            AudioDeviceInfo(2, "Speakers", 0, 2, 48_000),
        ]
        self.device_patch = patch(
            "gui.main_window.list_audio_devices",
            return_value=self.devices,
        )
        self.device_patch.start()
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()
        self.device_patch.stop()

    def test_device_lists_filter_by_channel_direction(self) -> None:
        self.assertEqual(self.window.input_combo.count(), 2)
        self.assertEqual(self.window.output_combo.count(), 3)
        self.assertIn("Microphone", self.window.input_combo.itemText(1))
        self.assertNotIn("Microphone", self.window.output_combo.itemText(1))

    def test_monitor_dropdown_tracks_checkbox(self) -> None:
        self.window.monitor_checkbox.setChecked(False)
        self.assertFalse(self.window.monitor_combo.isEnabled())
        self.window.monitor_checkbox.setChecked(True)
        self.assertTrue(self.window.monitor_combo.isEnabled())

    def test_missing_api_key_is_shown_without_starting(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.window.refresh_api_status()
            self.window.start_translation()
        self.assertIn("Missing", self.window.deepseek_api_label.text())
        self.assertEqual(self.window._status, "Error")
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
            event_callback({"type": "status", "status": "Idle"})

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
                    lambda: self.window._status == "Listening"
                )
                self.assertFalse(self.window.start_button.isEnabled())
                self.assertTrue(self.window.stop_button.isEnabled())

                QTest.mouseClick(
                    self.window.stop_button,
                    Qt.MouseButton.LeftButton,
                )
                self._wait_until(
                    lambda: self.window.pipeline_thread is None
                )
                self.assertEqual(self.window._status, "Idle")
                self.assertTrue(self.window.start_button.isEnabled())
                self.assertFalse(self.window.stop_button.isEnabled())

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
            self._wait_until(lambda: self.window._status == "Listening")
            self.window.close()
            self.assertTrue(self.window._close_requested)
            self._wait_until(lambda: self.window.pipeline_thread is None)
            QApplication.processEvents()
            self.assertFalse(self.window.isVisible())

    @staticmethod
    def _wait_until(predicate, attempts: int = 100) -> None:
        for _ in range(attempts):
            QApplication.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        raise AssertionError("Timed out waiting for GUI state")


if __name__ == "__main__":
    unittest.main()
