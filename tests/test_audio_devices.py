import unittest
from unittest.mock import patch

from core.audio_devices import (
    AudioDeviceInfo,
    AudioRoutingError,
    restore_saved_device,
    build_audio_routing_plan,
    list_audio_devices,
    resolve_audio_device,
    resolve_input_device,
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


class SavedDeviceTests(unittest.TestCase):
    def setUp(self):
        self.devices = [
            AudioDeviceInfo(4, "Other speaker", 0, 2, 48000),
            AudioDeviceInfo(9, "Saved cable", 0, 2, 48000),
        ]

    def test_safe_default_does_not_depend_on_machine_index(self):
        from config import TRANSLATION_OUTPUT_DEVICE
        self.assertIsNone(TRANSLATION_OUTPUT_DEVICE)
        self.assertIsNone(restore_saved_device(self.devices, None, None))

    def test_name_wins_when_index_changes(self):
        self.assertEqual(restore_saved_device(self.devices, 4, "saved CABLE").index, 9)

    def test_missing_name_never_uses_recycled_index(self):
        with self.assertRaises(AudioRoutingError):
            restore_saved_device(self.devices, 4, "Disconnected speaker")

    def test_duplicate_name_is_ambiguous_even_at_old_index(self):
        self.devices.append(AudioDeviceInfo(10, "Saved cable", 0, 2, 48000))
        with self.assertRaises(AudioRoutingError):
            restore_saved_device(self.devices, 9, "Saved cable")

    def test_legacy_index_only_selection_requires_reselection(self):
        for index in (4, 123):
            with self.subTest(index=index), self.assertRaises(AudioRoutingError):
                restore_saved_device(self.devices, index, None)

    def test_explicit_current_config_index_can_select_device(self):
        self.assertEqual(
            restore_saved_device(self.devices, 4, None, allow_index=True).index, 4
        )
        with self.assertRaises(AudioRoutingError):
            restore_saved_device(self.devices, 123, None, allow_index=True)

    def test_name_does_not_partially_match_another_device(self):
        with self.assertRaises(AudioRoutingError):
            restore_saved_device(self.devices, None, "Saved")


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

    def test_input_device_selection_reuses_shared_device_list(self) -> None:
        input_device, uses_default = resolve_input_device(0)
        self.assertEqual(input_device.index, 0)
        self.assertFalse(uses_default)

    def test_output_only_device_is_rejected_as_input(self) -> None:
        with self.assertRaisesRegex(
            AudioRoutingError,
            "has no input channels",
        ):
            resolve_input_device(2)

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
