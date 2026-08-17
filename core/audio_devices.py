from dataclasses import dataclass

import sounddevice as sd


class AudioRoutingError(Exception):
    """音频输出设备配置无法安全用于当前 PCM 格式。"""


@dataclass(frozen=True)
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


@dataclass(frozen=True)
class AudioRoutingPlan:
    translation_device: AudioDeviceInfo
    translation_uses_default: bool
    monitor_requested: bool
    monitor_device: AudioDeviceInfo | None
    monitor_warning: str | None = None

    @property
    def monitor_enabled(self) -> bool:
        return self.monitor_requested and self.monitor_device is not None


def list_audio_devices() -> list[AudioDeviceInfo]:
    """返回 PortAudio 当前可见的全部输入和输出设备。"""
    return [
        AudioDeviceInfo(
            index=index,
            name=str(device["name"]),
            max_input_channels=int(device["max_input_channels"]),
            max_output_channels=int(device["max_output_channels"]),
            default_samplerate=float(device["default_samplerate"]),
        )
        for index, device in enumerate(sd.query_devices())
    ]


def _default_output_index() -> int:
    try:
        output_index = int(sd.default.device[1])
    except (IndexError, TypeError, ValueError) as error:
        raise AudioRoutingError(
            "System default output device is unavailable."
        ) from error
    if output_index < 0:
        raise AudioRoutingError(
            "System default output device is unavailable."
        )
    return output_index


def _default_input_index() -> int:
    try:
        input_index = int(sd.default.device[0])
    except (IndexError, TypeError, ValueError) as error:
        raise AudioRoutingError(
            "System default input device is unavailable."
        ) from error
    if input_index < 0:
        raise AudioRoutingError(
            "System default input device is unavailable."
        )
    return input_index


def resolve_input_device(
    device: int | str | None,
    role: str = "Input microphone",
) -> tuple[AudioDeviceInfo, bool]:
    """把输入设备选择解析为 PortAudio 设备，并验证输入声道。"""
    devices = list_audio_devices()
    uses_default = device is None
    if device is None:
        device_index = _default_input_index()
    elif isinstance(device, bool):
        raise AudioRoutingError(f"{role} device must be an index or name.")
    elif isinstance(device, int):
        device_index = device
    elif isinstance(device, str) and device.strip():
        query = device.strip().casefold()
        exact_matches = [
            item for item in devices if item.name.casefold() == query
        ]
        partial_matches = [
            item for item in devices if query in item.name.casefold()
        ]
        matches = exact_matches or partial_matches
        if len(matches) != 1:
            match_indexes = ", ".join(str(item.index) for item in matches)
            detail = f"; matching indexes: {match_indexes}" if matches else ""
            raise AudioRoutingError(
                f'{role} device name "{device}" is not unique or was not '
                f"found{detail}."
            )
        device_index = matches[0].index
    else:
        raise AudioRoutingError(f"{role} device must be an index or name.")

    if device_index < 0 or device_index >= len(devices):
        raise AudioRoutingError(
            f"{role} device {device_index} does not exist."
        )
    device_info = devices[device_index]
    if device_info.max_input_channels <= 0:
        raise AudioRoutingError(
            f"{role} device [{device_info.index}] {device_info.name} "
            "has no input channels."
        )
    return device_info, uses_default


def resolve_audio_device(
    device: int | str | None,
    role: str,
) -> tuple[AudioDeviceInfo, bool]:
    """把 None、索引或唯一名称解析为明确的 PortAudio 设备。"""
    devices = list_audio_devices()
    uses_default = device is None
    if device is None:
        device_index = _default_output_index()
    elif isinstance(device, bool):
        raise AudioRoutingError(f"{role} device must be an index or name.")
    elif isinstance(device, int):
        device_index = device
    elif isinstance(device, str) and device.strip():
        query = device.strip().casefold()
        exact_matches = [
            item for item in devices if item.name.casefold() == query
        ]
        partial_matches = [
            item for item in devices if query in item.name.casefold()
        ]
        matches = exact_matches or partial_matches
        if len(matches) != 1:
            match_indexes = ", ".join(
                str(item.index) for item in matches
            )
            detail = (
                f"; matching indexes: {match_indexes}"
                if matches
                else ""
            )
            raise AudioRoutingError(
                f'{role} device name "{device}" is not unique or was not '
                f"found{detail}."
            )
        device_index = matches[0].index
    else:
        raise AudioRoutingError(f"{role} device must be an index or name.")

    if device_index < 0 or device_index >= len(devices):
        raise AudioRoutingError(
            f"{role} device {device_index} does not exist."
        )
    device_info = devices[device_index]
    if device_info.max_output_channels <= 0:
        raise AudioRoutingError(
            f"{role} device [{device_info.index}] {device_info.name} "
            "has no output channels."
        )
    return device_info, uses_default


def validate_output_device(
    device: int | str | None,
    role: str,
    samplerate: int,
    channels: int,
    dtype: str,
) -> tuple[AudioDeviceInfo, bool]:
    """验证设备方向及其对当前 ElevenLabs PCM 格式的支持。"""
    device_info, uses_default = resolve_audio_device(device, role)
    try:
        sd.check_output_settings(
            device=device_info.index,
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
        )
    except Exception as error:
        raise AudioRoutingError(
            f"{role} device [{device_info.index}] {device_info.name} "
            f"cannot use {samplerate} Hz, {channels} channel(s), {dtype}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return device_info, uses_default


def build_audio_routing_plan(
    translation_device: int | str | None,
    monitor_enabled: bool,
    monitor_device: int | str | None,
    samplerate: int,
    channels: int,
    dtype: str,
) -> AudioRoutingPlan:
    """主输出必须有效；Monitor 无效时仅禁用 Monitor。"""
    translation_info, translation_uses_default = validate_output_device(
        translation_device,
        "Translation output",
        samplerate,
        channels,
        dtype,
    )
    resolved_monitor = None
    monitor_warning = None
    if monitor_enabled:
        try:
            resolved_monitor, _ = validate_output_device(
                monitor_device,
                "Local monitor",
                samplerate,
                channels,
                dtype,
            )
            if resolved_monitor.index == translation_info.index:
                monitor_warning = (
                    "Local monitor resolves to the translation output "
                    "device and was disabled to avoid duplicate playback."
                )
                resolved_monitor = None
        except AudioRoutingError as error:
            monitor_warning = str(error)

    return AudioRoutingPlan(
        translation_device=translation_info,
        translation_uses_default=translation_uses_default,
        monitor_requested=monitor_enabled,
        monitor_device=resolved_monitor,
        monitor_warning=monitor_warning,
    )


def format_device(device: AudioDeviceInfo) -> str:
    return f"[{device.index}] {device.name}"


def print_audio_devices() -> None:
    print("[Audio Devices]")
    print()
    for device in list_audio_devices():
        print(f"[{device.index}] {device.name}")
        print(f"    input={device.max_input_channels}")
        print(f"    output={device.max_output_channels}")
        print(f"    default_samplerate={device.default_samplerate:.0f}")
        print()


def print_audio_routing(plan: AudioRoutingPlan) -> None:
    print("========== Audio Routing ==========")
    default_label = " (Default)" if plan.translation_uses_default else ""
    print("Translation Output:")
    print(f"{format_device(plan.translation_device)}{default_label}")
    print()
    print("Local Monitor:")
    print("ON" if plan.monitor_enabled else "OFF")
    print()
    print("Monitor Device:")
    if plan.monitor_device is not None:
        print(format_device(plan.monitor_device))
    else:
        print("None")
    if plan.monitor_warning:
        print()
        print(f"[Audio Routing Warning] {plan.monitor_warning}")
    print("===================================")
