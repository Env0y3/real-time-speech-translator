import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from core.audio_devices import (
    AudioDeviceInfo,
    AudioRoutingError,
    AudioRoutingPlan,
)
from core.elevenlabs_tts import (
    AudioOutputStreams,
    _open_output_stream,
    _should_use_local_tts_fallback,
    _write_routed_audio_chunk,
    SentenceMetrics,
    SentenceState,
    elevenlabs_tts_worker,
)


class FakeOutputStream:
    def __init__(self, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.writes = []
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def write(self, samples: np.ndarray) -> None:
        if self.fail_write:
            raise RuntimeError("monitor unavailable")
        self.writes.append(samples.copy())

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class OutputStreamCreationTests(unittest.TestCase):
    def test_output_stream_uses_selected_device_and_existing_pcm_format(
        self,
    ) -> None:
        stream = FakeOutputStream()
        with patch(
            "core.elevenlabs_tts.sd.OutputStream",
            return_value=stream,
        ) as output_stream:
            result = _open_output_stream(8, "Translation output")

        self.assertIs(result, stream)
        self.assertTrue(stream.started)
        output_stream.assert_called_once_with(
            device=8,
            samplerate=24_000,
            channels=1,
            dtype="int16",
        )


class RoutedAudioWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_off_writes_primary_once(self) -> None:
        primary = FakeOutputStream()
        streams = AudioOutputStreams(translation=primary)
        samples = np.array([[1], [2], [3]], dtype=np.int16)

        await _write_routed_audio_chunk(streams, samples)

        self.assertEqual(len(primary.writes), 1)
        np.testing.assert_array_equal(primary.writes[0], samples)

    async def test_monitor_on_writes_same_pcm_to_both_streams(self) -> None:
        primary = FakeOutputStream()
        monitor = FakeOutputStream()
        streams = AudioOutputStreams(
            translation=primary,
            monitor=monitor,
            monitor_label="[4] Speakers",
        )
        samples = np.array([[1], [2], [3]], dtype=np.int16)

        await _write_routed_audio_chunk(streams, samples)

        self.assertEqual(len(primary.writes), 1)
        self.assertEqual(len(monitor.writes), 1)
        np.testing.assert_array_equal(primary.writes[0], monitor.writes[0])

    async def test_monitor_failure_disables_only_monitor(self) -> None:
        primary = FakeOutputStream()
        monitor = FakeOutputStream(fail_write=True)
        streams = AudioOutputStreams(
            translation=primary,
            monitor=monitor,
            monitor_label="[4] Speakers",
        )
        samples = np.array([[1], [2], [3]], dtype=np.int16)

        await _write_routed_audio_chunk(streams, samples)

        self.assertEqual(len(primary.writes), 1)
        self.assertIsNone(streams.monitor)
        self.assertTrue(monitor.stopped)
        self.assertTrue(monitor.closed)


class ExplicitOutputFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_output_failure_never_falls_back_to_speakers(
        self,
    ) -> None:
        plan = AudioRoutingPlan(
            translation_device=AudioDeviceInfo(
                index=8,
                name="CABLE Input",
                max_input_channels=0,
                max_output_channels=2,
                default_samplerate=48_000,
            ),
            translation_uses_default=False,
            monitor_requested=False,
            monitor_device=None,
        )
        translated_queue = asyncio.Queue()
        fallback_worker = AsyncMock()

        with (
            patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}),
            patch(
                "core.elevenlabs_tts._open_audio_output_streams",
                side_effect=RuntimeError("device unavailable"),
            ),
            patch("core.elevenlabs_tts.tts_worker", fallback_worker),
        ):
            with self.assertRaises(AudioRoutingError):
                await elevenlabs_tts_worker(
                    translated_queue,
                    audio_routing=plan,
                )

        fallback_worker.assert_not_awaited()

    async def test_explicit_routing_disables_local_sentence_fallback(
        self,
    ) -> None:
        explicit_plan = AudioRoutingPlan(
            translation_device=AudioDeviceInfo(
                index=8,
                name="CABLE Input",
                max_input_channels=0,
                max_output_channels=2,
                default_samplerate=48_000,
            ),
            translation_uses_default=False,
            monitor_requested=False,
            monitor_device=None,
        )
        state = SentenceState(
            sentence_id=1,
            metrics=SentenceMetrics(first_segment_ready_at=1.0),
            segments={1: "Hello."},
        )

        self.assertFalse(
            _should_use_local_tts_fallback(state, explicit_plan)
        )


if __name__ == "__main__":
    unittest.main()
