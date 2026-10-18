"""Offline regressions for current choices versus persisted device identities."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import PySide6
except ModuleNotFoundError as error:
    if error.name != "PySide6":
        raise
    PYSIDE_AVAILABLE = False
else:
    from PySide6.QtWidgets import QApplication

    from core.audio_devices import AudioDeviceInfo, AudioRoutingError
    from gui.main_window import MainWindow, UNAVAILABLE_ROLE

    PYSIDE_AVAILABLE = True


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class CurrentDeviceSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.temp_dir.name) / "settings.json"
        self.devices = [
            AudioDeviceInfo(0, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(1, "Microphone", 1, 0, 48_000),
            AudioDeviceInfo(2, "Speakers", 0, 2, 48_000),
            AudioDeviceInfo(3, "Speakers", 0, 2, 48_000),
        ]
        self.device_patch = patch(
            "gui.main_window.list_audio_devices",
            side_effect=lambda: list(self.devices),
        )
        self.device_patch.start()
        self.window = self._new_window()

    def tearDown(self) -> None:
        self.window.pipeline_thread = None
        self.window.close()
        self.device_patch.stop()
        self.temp_dir.cleanup()

    def _new_window(self) -> MainWindow:
        return MainWindow(settings_path=self.settings_path, auto_initialize=False)

    def _select_output(self, index: int = 3) -> None:
        self.window.output_combo.setCurrentIndex(
            self.window.output_combo.findData(index)
        )

    def _start_with_mocks(self):
        self.window._on_initialization_ready()
        worker = Mock()
        with (
            patch.dict(os.environ, {
                "DEEPSEEK_API_KEY": "offline-placeholder",
                "ELEVENLABS_API_KEY": "offline-placeholder",
            }, clear=True),
            patch("gui.main_window.validate_normal_pipeline_options") as validate,
            patch("gui.main_window.PipelineThread", return_value=worker) as create,
        ):
            self.window.start_translation()
        return worker, validate, create

    def test_explicit_duplicate_output_survives_unchanged_prestart_refresh(self):
        self._select_output()
        worker, validate, create = self._start_with_mocks()
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0].translation_output_device, 3)
        validate.assert_called_once()
        worker.start.assert_called_once_with()
        self.assertFalse(self.window.output_combo.currentData(UNAVAILABLE_ROLE))

    def test_explicit_duplicate_input_and_monitor_survive_repeated_refresh(self):
        self.window.input_combo.setCurrentIndex(self.window.input_combo.findData(1))
        self.window.monitor_combo.setCurrentIndex(self.window.monitor_combo.findData(2))
        self.window.monitor_checkbox.setChecked(True)
        for _ in range(2):
            self.window.refresh_devices()
            options = self.window._runtime_options()
            self.assertEqual(options.input_device, 1)
            self.assertEqual(options.monitor_device, 2)

    def test_historical_duplicate_is_unavailable_even_at_saved_index(self):
        self._select_output()
        restored = self._new_window()
        try:
            for _ in range(2):
                restored.refresh_devices()
                self.assertTrue(restored.output_combo.currentData(UNAVAILABLE_ROLE))
                with self.assertRaises(AudioRoutingError):
                    restored._runtime_options()
        finally:
            restored.close()

    def test_historical_duplicate_can_be_explicitly_reselected_then_started(self):
        self._select_output()
        self.window.close()
        self.window = self._new_window()
        self.assertTrue(self.window.output_combo.currentData(UNAVAILABLE_ROLE))
        self._select_output(2)
        worker, _, create = self._start_with_mocks()
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0].translation_output_device, 2)
        worker.start.assert_called_once_with()

    def test_device_changes_fail_closed_until_explicit_reselection(self):
        self._select_output()
        self.devices[3] = AudioDeviceInfo(3, "Replacement speaker", 0, 2, 48_000)
        for _ in range(2):
            self.window.refresh_devices()
            self.assertTrue(self.window.output_combo.currentData(UNAVAILABLE_ROLE))
            with self.assertRaisesRegex(AudioRoutingError, "重新选择"):
                self.window._runtime_options()
        self.assertIn("device_reselection_required", self.window._notice_keys)
        self._select_output(2)
        self.window.refresh_devices()
        self.assertEqual(self.window._runtime_options().translation_output_device, 2)

    def test_remaining_same_name_peer_is_not_an_automatic_replacement(self):
        self._select_output()
        self.devices = self.devices[:3]
        for _ in range(2):
            self.window.refresh_devices()
            self.assertTrue(self.window.output_combo.currentData(UNAVAILABLE_ROLE))
            with self.assertRaises(AudioRoutingError):
                self.window._runtime_options()

    def test_changed_device_properties_invalidate_duplicate_choice(self):
        self._select_output()
        self.devices[3] = AudioDeviceInfo(3, "Speakers", 0, 1, 44_100)
        self.window.refresh_devices()
        self.assertTrue(self.window.output_combo.currentData(UNAVAILABLE_ROLE))
        with self.assertRaises(AudioRoutingError):
            self.window._runtime_options()

    def test_failed_enumeration_invalidates_duplicate_choice(self):
        self._select_output()
        with patch("gui.main_window.list_audio_devices", side_effect=RuntimeError("offline")):
            worker, validate, create = self._start_with_mocks()
        create.assert_not_called()
        validate.assert_not_called()
        worker.start.assert_not_called()
        self.window.refresh_devices()
        self.assertTrue(self.window.output_combo.currentData(UNAVAILABLE_ROLE))
        with self.assertRaises(AudioRoutingError):
            self.window._runtime_options()
        self._select_output()
        self.window.refresh_devices()
        self.assertEqual(self.window._runtime_options().translation_output_device, 3)

    def test_system_default_is_explicit_recovery_after_snapshot_changes(self):
        self._select_output()
        self.devices = self.devices[:3]
        self.window.refresh_devices()
        self.window.output_combo.setCurrentIndex(0)
        self.window.refresh_devices()
        self.assertIsNone(self.window._runtime_options().translation_output_device)

    def test_live_selection_proof_is_not_persisted(self):
        self._select_output()
        self.window.refresh_devices()
        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["translation_output_device"], {
            "index": 3, "name": "Speakers",
        })
        restored = self._new_window()
        try:
            self.assertTrue(restored.output_combo.currentData(UNAVAILABLE_ROLE))
        finally:
            restored.close()


if __name__ == "__main__":
    unittest.main()
