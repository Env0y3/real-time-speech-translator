"""Offline failure-event to Qt stop-state regression; no network/audio device."""
from __future__ import annotations
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
try:
    import PySide6
except ModuleNotFoundError as error:
    if error.name != 'PySide6':
        raise
    QT = False
else:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    from gui.main_window import MainWindow
    from gui.pipeline_thread import PipelineThread
    from core.pipeline import NormalPipelineOptions
    from core.audio_devices import AudioDeviceInfo
    QT = True

@unittest.skipUnless(QT, 'PySide6 is not installed')
class FailureGUIFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        cls.app = QApplication.instance() or QApplication([])

    def test_error_callback_requests_stop_before_gui_delivery(self):
        checks = []
        async def pipeline(*, stop_signal, event_callback, **kwargs):
            event_callback({'type': 'playback_error', 'error_category': 'audio_output_failed'})
            checks.append(stop_signal.is_set())
        with patch('gui.pipeline_thread.run_normal_pipeline', side_effect=pipeline):
            thread = PipelineThread(NormalPipelineOptions())
            thread.run()
        self.assertEqual(checks, [True])

    def test_service_and_partial_errors_survive_actual_thread_cleanup(self):
        for category in ('tts_service_failed', 'tts_partial_failed', 'audio_output_failed'):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                stopped = []
                async def pipeline(*, stop_signal, event_callback, **kwargs):
                    event_callback({'type': 'playback_error', 'error_category': category})
                    for _ in range(200):
                        if stop_signal.is_set():
                            break
                        await asyncio.sleep(.005)
                    stopped.append(stop_signal.is_set())
                    event_callback({'type': 'status', 'status': 'Listening'})
                    event_callback({'type': 'status', 'status': 'Ready'})
                with patch('gui.main_window.list_audio_devices', return_value=[AudioDeviceInfo(0,'synthetic',1,2,48000)]), patch('gui.pipeline_thread.run_normal_pipeline', side_effect=pipeline):
                    window = MainWindow(settings_path=Path(tmp)/'settings.json', auto_initialize=False)
                    window._on_initialization_ready()
                    thread = PipelineThread(NormalPipelineOptions(), window)
                    window.pipeline_thread = thread
                    thread.event_received.connect(window._handle_pipeline_event)
                    thread.finished.connect(window._pipeline_finished)
                    thread.pipeline_error.connect(window._handle_pipeline_error)
                    window._set_running_controls(True)
                    thread.start()
                    for _ in range(300):
                        self.app.processEvents()
                        if window.pipeline_thread is None:
                            break
                        QTest.qWait(5)
                    self.assertIsNone(window.pipeline_thread)
                    self.assertEqual(stopped, [True])
                    self.assertEqual(window._status, 'ERROR')
                    self.assertEqual(window._error_key, category)
                    self.assertFalse(window.stop_button.isEnabled())
                    self.assertTrue(window.start_button.isEnabled())
                    window.close()


    def test_actual_tts_failures_reach_gui_and_stop_without_overwrite(self):
        # Real TTS orchestration/playback consumer and real Qt thread/signals;
        # only remote websocket and physical output are replaced by fault fakes.
        from test_tts_failure_paths import TTSFailurePathTests, audio_frame
        cases = (
            ("tts_service_failed", [{"error": "offline service failure"}], {}),
            ("tts_partial_failed", [audio_frame(), {"error": "offline disconnect"}],
             {"default": True, "wait_for_write": True}),
            ("audio_output_failed", [audio_frame()], {"fail_output": True}),
        )
        for category, frames, options in cases:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                observations = []
                async def pipeline(*, stop_signal, event_callback, **kwargs):
                    settings = dict(options)
                    gate = None
                    if category == "audio_output_failed":
                        gate = asyncio.Event()
                        settings["final_gate"] = gate
                    helper = TTSFailurePathTests()
                    result = await helper.run_case(frames, event_callback=event_callback, **settings)
                    observations.append((stop_signal.is_set(), result[4], result[2].closed))
                    event_callback({"type": "status", "status": "Listening"})
                    event_callback({"type": "playback_error", "error_category": "late_cleanup_error"})
                    if result[4] is not None:
                        raise result[4]
                with patch('gui.main_window.list_audio_devices', return_value=[AudioDeviceInfo(0,'synthetic',1,2,48000)]), patch('gui.pipeline_thread.run_normal_pipeline', side_effect=pipeline):
                    window = MainWindow(settings_path=Path(tmp)/'settings.json', auto_initialize=False)
                    window._on_initialization_ready()
                    thread = PipelineThread(NormalPipelineOptions(), window)
                    window.pipeline_thread = thread
                    thread.event_received.connect(window._handle_pipeline_event)
                    thread.finished.connect(window._pipeline_finished)
                    thread.pipeline_error.connect(window._handle_pipeline_error)
                    window._set_running_controls(True)
                    thread.start()
                    for _ in range(600):
                        self.app.processEvents()
                        if window.pipeline_thread is None:
                            break
                        QTest.qWait(5)
                    self.assertIsNone(window.pipeline_thread)
                    self.assertEqual(len(observations), 1)
                    self.assertTrue(observations[0][0])
                    self.assertIsNotNone(observations[0][1])
                    self.assertNotIsInstance(observations[0][1], TimeoutError)
                    self.assertTrue(observations[0][2])
                    self.assertEqual(window._status, 'ERROR')
                    self.assertEqual(window._error_key, category)
                    window.close()
