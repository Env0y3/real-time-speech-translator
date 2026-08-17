import asyncio
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Protocol

from vosk import SetLogLevel

from asr.sensevoice_asr import sensevoice_asr_worker
from asr.vosk_asr import vosk_asr_worker
from config import (
    ELEVENLABS_AUDIO_QUEUE_MAXSIZE,
    ELEVENLABS_CHANNELS,
    ELEVENLABS_DTYPE,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SAMPLE_RATE,
    ELEVENLABS_VOICE_ID,
    FALSE_TRIGGER_FILLERS,
    FALSE_TRIGGER_FILTER_ENABLED,
    FALSE_TRIGGER_MAX_SPEECH_SECONDS,
    HOTWORDS_PATH,
    INPUT_VALIDITY_GUARD_ENABLED,
    LOCAL_MONITOR_DEVICE,
    LOCAL_MONITOR_ENABLED,
    MODEL_PATH,
    NORMAL_ASR_PROVIDER,
    NORMAL_HOTWORD_CORRECTION_ENABLED,
    RUNTIME_LATENCY_LOG_PATH,
    STREAMING_TRANSLATION_ENABLED,
    STREAMING_TRANSLATION_MAX_CHARS,
    STREAMING_TRANSLATION_MIN_CHARS,
    STREAMING_TRANSLATION_TARGET_CHARS,
    TTS_PROVIDER,
    TRANSLATION_OUTPUT_DEVICE,
)
from core.audio import audio_worker, wait_for_stop
from core.audio_devices import (
    AudioRoutingPlan,
    build_audio_routing_plan,
    print_audio_routing,
    resolve_input_device,
)
from core.elevenlabs_tts import elevenlabs_tts_worker
from core.false_trigger_filter import false_trigger_filter_worker
from core.hotwords import hotword_correction_worker, load_hotwords
from core.input_guard import input_validity_guard_worker
from core.performance_logger import PerformanceLogger, create_session_id
from core.translation import streaming_translation_worker, translation_worker
from core.tts import tts_worker


PipelineEventCallback = Callable[[dict[str, Any]], None]


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class NormalPipelineOptions:
    """Normal Mode 可在运行时覆盖的设备选项；不会修改 config.py。"""

    input_device: int | str | None = None
    translation_output_device: int | str | None = TRANSLATION_OUTPUT_DEVICE
    local_monitor_enabled: bool = LOCAL_MONITOR_ENABLED
    monitor_device: int | str | None = LOCAL_MONITOR_DEVICE


def validate_normal_pipeline_options(
    options: NormalPipelineOptions,
) -> AudioRoutingPlan | None:
    """启动前验证 Provider、API Key、输入方向和输出 PCM 路由。"""
    if NORMAL_ASR_PROVIDER not in {"vosk", "sensevoice"}:
        raise ValueError('NORMAL_ASR_PROVIDER 只支持 "vosk" 或 "sensevoice"')
    if NORMAL_ASR_PROVIDER == "vosk" and not MODEL_PATH.exists():
        raise ValueError(f"找不到 Vosk 中文模型：{MODEL_PATH}")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("Missing DEEPSEEK_API_KEY")
    if TTS_PROVIDER not in {"pyttsx3", "elevenlabs"}:
        raise ValueError('TTS_PROVIDER 只支持 "pyttsx3" 或 "elevenlabs"')
    if TTS_PROVIDER == "elevenlabs" and not os.environ.get(
        "ELEVENLABS_API_KEY"
    ):
        raise ValueError("Missing ELEVENLABS_API_KEY")

    resolve_input_device(options.input_device)
    if TTS_PROVIDER != "elevenlabs":
        return None
    return build_audio_routing_plan(
        options.translation_output_device,
        options.local_monitor_enabled,
        options.monitor_device,
        ELEVENLABS_SAMPLE_RATE,
        ELEVENLABS_CHANNELS,
        ELEVENLABS_DTYPE,
    )


async def _watch_external_stop(
    external_stop: StopSignal,
    pipeline_stop: asyncio.Event,
) -> None:
    while not external_stop.is_set() and not pipeline_stop.is_set():
        await asyncio.sleep(0.05)
    if external_stop.is_set():
        pipeline_stop.set()


async def _emit_listening_when_ready(
    asr_ready: asyncio.Event,
    performance_logger: PerformanceLogger,
) -> None:
    await asr_ready.wait()
    performance_logger.emit_event(
        {"type": "status", "status": "Listening"}
    )


def _session_start_record(
    options: NormalPipelineOptions,
    audio_routing_plan: AudioRoutingPlan | None,
) -> dict[str, Any]:
    return {
        "event": "session_start",
        "streaming_translation_enabled": STREAMING_TRANSLATION_ENABLED,
        "tts_provider": TTS_PROVIDER,
        "audio_playback_queue_maxsize": (
            ELEVENLABS_AUDIO_QUEUE_MAXSIZE
            if TTS_PROVIDER == "elevenlabs"
            else None
        ),
        "translation_min_chars": STREAMING_TRANSLATION_MIN_CHARS,
        "translation_target_chars": STREAMING_TRANSLATION_TARGET_CHARS,
        "translation_max_chars": STREAMING_TRANSLATION_MAX_CHARS,
        "false_trigger_filter_enabled": FALSE_TRIGGER_FILTER_ENABLED,
        "input_validity_guard_enabled": INPUT_VALIDITY_GUARD_ENABLED,
        "false_trigger_max_speech_ms": (
            FALSE_TRIGGER_MAX_SPEECH_SECONDS * 1000
        ),
        "false_trigger_fillers": sorted(FALSE_TRIGGER_FILLERS),
        "input_device": options.input_device,
        "translation_output_device": (
            audio_routing_plan.translation_device.index
            if audio_routing_plan is not None
            else None
        ),
        "translation_output_device_name": (
            audio_routing_plan.translation_device.name
            if audio_routing_plan is not None
            else None
        ),
        "translation_output_uses_default": (
            audio_routing_plan.translation_uses_default
            if audio_routing_plan is not None
            else None
        ),
        "local_monitor_enabled": (
            audio_routing_plan.monitor_enabled
            if audio_routing_plan is not None
            else False
        ),
        "monitor_device": (
            audio_routing_plan.monitor_device.index
            if audio_routing_plan is not None
            and audio_routing_plan.monitor_device is not None
            else None
        ),
    }


async def run_normal_pipeline(
    options: NormalPipelineOptions | None = None,
    stop_signal: StopSignal | None = None,
    event_callback: PipelineEventCallback | None = None,
    interactive_stop: bool = False,
) -> None:
    """运行 CLI 与 GUI 共用的 Normal Pipeline 编排。"""
    if not interactive_stop and stop_signal is None:
        raise ValueError("A stop_signal is required outside interactive CLI mode")

    runtime_options = options or NormalPipelineOptions()
    audio_routing_plan = validate_normal_pipeline_options(runtime_options)
    if audio_routing_plan is not None:
        print_audio_routing(audio_routing_plan)

    SetLogLevel(-1)
    session_id = create_session_id()
    performance_logger = PerformanceLogger(
        RUNTIME_LATENCY_LOG_PATH,
        session_id,
        event_callback=event_callback,
    )
    await performance_logger.log(
        _session_start_record(runtime_options, audio_routing_plan)
    )
    print(f"Latency Session ID: {session_id}")

    audio_queue = asyncio.Queue(maxsize=5)
    raw_text_queue = asyncio.Queue(maxsize=5)
    translated_queue = asyncio.Queue(
        maxsize=20 if STREAMING_TRANSLATION_ENABLED else 5
    )
    pipeline_stop = asyncio.Event()
    microphone_ready = asyncio.Event()
    asr_ready = asyncio.Event()

    selected_translation_worker = (
        streaming_translation_worker
        if STREAMING_TRANSLATION_ENABLED
        else translation_worker
    )
    selected_tts_worker = (
        partial(elevenlabs_tts_worker, audio_routing=audio_routing_plan)
        if TTS_PROVIDER == "elevenlabs"
        else tts_worker
    )

    print(
        "Streaming Translation: "
        f"{'ON' if STREAMING_TRANSLATION_ENABLED else 'OFF'}"
    )
    print(f"TTS Provider: {TTS_PROVIDER}")
    if TTS_PROVIDER == "elevenlabs":
        print(f"Voice: {ELEVENLABS_VOICE_ID}")
        print(f"Model: {ELEVENLABS_MODEL_ID}")
        print(f"Output: {ELEVENLABS_OUTPUT_FORMAT}")

    worker_coroutines = [
        audio_worker(
            audio_queue,
            pipeline_stop,
            microphone_ready,
            input_device=runtime_options.input_device,
        ),
    ]
    if stop_signal is not None:
        worker_coroutines.append(
            _watch_external_stop(stop_signal, pipeline_stop)
        )
    if interactive_stop:
        worker_coroutines.append(
            wait_for_stop(pipeline_stop, microphone_ready)
        )

    if NORMAL_ASR_PROVIDER == "vosk":
        print("Normal ASR: Vosk")
        worker_coroutines.extend(
            [
                _emit_listening_when_ready(asr_ready, performance_logger),
                vosk_asr_worker(
                    audio_queue,
                    raw_text_queue,
                    asr_ready=asr_ready,
                ),
                selected_translation_worker(
                    raw_text_queue,
                    translated_queue,
                    performance_logger,
                ),
                selected_tts_worker(translated_queue, performance_logger),
            ]
        )
    else:
        print("Normal ASR: SenseVoiceSmall")
        print(
            "Hotword Post Correction: "
            f"{'ON' if NORMAL_HOTWORD_CORRECTION_ENABLED else 'OFF'}"
        )
        print(
            "Input Validity Guard: "
            f"{'ON' if INPUT_VALIDITY_GUARD_ENABLED else 'OFF'}"
        )
        print(
            "False Trigger Filter: "
            f"{'ON' if FALSE_TRIGGER_FILTER_ENABLED else 'OFF'}"
        )
        valid_text_queue = asyncio.Queue(maxsize=5)
        filtered_text_queue = asyncio.Queue(maxsize=5)
        translation_input_queue = filtered_text_queue
        worker_coroutines.extend(
            [
                sensevoice_asr_worker(
                    audio_queue,
                    raw_text_queue,
                    asr_ready,
                    benchmark_mode=False,
                    trace_session_id=session_id,
                    event_callback=performance_logger.emit_event,
                ),
                input_validity_guard_worker(
                    raw_text_queue,
                    valid_text_queue,
                    performance_logger,
                ),
                false_trigger_filter_worker(
                    valid_text_queue,
                    filtered_text_queue,
                    performance_logger,
                ),
            ]
        )
        if NORMAL_HOTWORD_CORRECTION_ENABLED:
            corrected_text_queue = asyncio.Queue(maxsize=5)
            translation_input_queue = corrected_text_queue
            worker_coroutines.append(
                hotword_correction_worker(
                    filtered_text_queue,
                    corrected_text_queue,
                    load_hotwords(HOTWORDS_PATH),
                )
            )
        worker_coroutines.extend(
            [
                selected_translation_worker(
                    translation_input_queue,
                    translated_queue,
                    performance_logger,
                ),
                selected_tts_worker(translated_queue, performance_logger),
            ]
        )

    try:
        await asyncio.gather(*worker_coroutines)
    except Exception as error:
        performance_logger.emit_event(
            {
                "type": "error",
                "message": f"{type(error).__name__}: {error}",
            }
        )
        pipeline_stop.set()
        raise
    finally:
        await performance_logger.log({"event": "session_end"})
        performance_logger.emit_event({"type": "status", "status": "Idle"})

    print("所有任务正常退出")
