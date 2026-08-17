import unittest
from unittest.mock import patch

from core.audio_devices import (
    AudioRoutingError,
    build_audio_routing_plan,
    list_audio_devices,
    resolve_audio_device,
    validate_output_device,
)


FAKE_DEVICES = [
    {
        "name": "Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48_000,
    },
    {
        "name": "Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48_000,
    },
    {
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48_000,
    },
]


class AudioDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query_patch = patch(
            "core.audio_devices.sd.query_devices",
            return_value=FAKE_DEVICES,
        )
        self.check_patch = patch(
            "core.audio_devices.sd.check_output_settings"
        )
        self.query_patch.start()
        self.check_output_settings = self.check_patch.start()

    def tearDown(self) -> None:
        self.check_patch.stop()
        self.query_patch.stop()

    def test_list_audio_devices_exposes_required_fields(self) -> None:
        devices = list_audio_devices()
        self.assertEqual(devices[2].index, 2)
        self.assertIn("CABLE Input", devices[2].name)
        self.assertEqual(devices[2].max_input_channels, 0)
        self.assertEqual(devices[2].max_output_channels, 2)
        self.assertEqual(devices[2].default_samplerate, 48_000)

    def test_integer_and_unique_name_selection(self) -> None:
        by_index, _ = resolve_audio_device(2, "Translation output")
        by_name, _ = resolve_audio_device(
            "VB-Audio Virtual Cable",
            "Translation output",
        )
        self.assertEqual(by_index.index, 2)
        self.assertEqual(by_name.index, 2)

    def test_invalid_device_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(AudioRoutingError, "does not exist"):
            validate_output_device(
                999,
                "Translation output",
                24_000,
                1,
                "int16",
            )

    def test_input_only_device_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            AudioRoutingError,
            "has no output channels",
        ):
            validate_output_device(
                0,
                "Translation output",
                24_000,
                1,
                "int16",
            )

    def test_pcm_settings_are_checked(self) -> None:
        validate_output_device(
            2,
            "Translation output",
            24_000,
            1,
            "int16",
        )
        self.check_output_settings.assert_called_once_with(
            device=2,
            samplerate=24_000,
            channels=1,
            dtype="int16",
        )

    def test_invalid_monitor_is_disabled_without_affecting_primary(self) -> None:
        plan = build_audio_routing_plan(
            2,
            True,
            999,
            24_000,
            1,
            "int16",
        )
        self.assertEqual(plan.translation_device.index, 2)
        self.assertFalse(plan.monitor_enabled)
        self.assertIn("does not exist", plan.monitor_warning)


if __name__ == "__main__":
    unittest.main()
