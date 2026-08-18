import asyncio
import re
import time
from collections import deque
from datetime import datetime
from functools import lru_cache
from statistics import median

import numpy as np

from config import (
    CHUNK_DURATION_SECONDS,
    ENDPOINT_ADAPTIVE_ENABLED,
    ENDPOINT_BASE_SECONDS,
    ENDPOINT_MAX_SECONDS,
    ENDPOINT_MIN_INTRA_PAUSE_SECONDS,
    ENDPOINT_MIN_SPEECH_SECONDS,
    ENDPOINT_MIN_SECONDS,
    ENDPOINT_SAFETY_MARGIN_SECONDS,
    ENDPOINT_SMOOTHING_ALPHA,
    ENDPOINT_SHORT_UTTERANCE_EXTRA_WAIT_SECONDS,
    SENSEVOICE_MODEL_NAME,
    VAD_RMS_THRESHOLD,
)


@lru_cache(maxsize=1)
def load_sensevoice_model():
    """延迟导入 FunASR，并在 CPU 上加载 SenseVoiceSmall。"""
    # Lazy Import（延迟导入）确保 Normal Mode 不加载 FunASR/PyTorch。
    from funasr import AutoModel

    return AutoModel(
        model=SENSEVOICE_MODEL_NAME,
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )


def recognize_sensevoice_sync(model, utterance_audio: np.ndarray) -> str:
    """对一整句 float32 音频执行 SenseVoiceSmall 离线推理。"""
    results = model.generate(
        input=utterance_audio,
        language="zh",
        use_itn=True,
        batch_size_s=60,
    )
    if not results:
        return ""

    raw_text = results[0].get("text", "")
    # SenseVoice 原始结果含 <|zh|>、<|Speech|> 等元数据标签。
    # Benchmark 只比较 ASR 文本，所以在 ASR Provider 层移除这些标签。
    return re.sub(r"<\|[^>]+\|>", "", raw_text).strip()


def calculate_adaptive_endpoint_threshold(
    current_threshold: float,
    recent_pause_durations: deque,
) -> tuple[float, float, float]:
    """用近期句中停顿中位数、Safety Margin 和 EMA 更新阈值。"""
    pause_statistic = median(recent_pause_durations)

    if len(recent_pause_durations) < 3:
        target_threshold = ENDPOINT_BASE_SECONDS
        new_threshold = current_threshold
    else:
        target_threshold = min(
            ENDPOINT_MAX_SECONDS,
            max(
                ENDPOINT_MIN_SECONDS,
                pause_statistic + ENDPOINT_SAFETY_MARGIN_SECONDS,
            ),
        )
        new_threshold = (
            current_threshold * (1 - ENDPOINT_SMOOTHING_ALPHA)
            + target_threshold * ENDPOINT_SMOOTHING_ALPHA
        )
        new_threshold = min(
            ENDPOINT_MAX_SECONDS,
            max(ENDPOINT_MIN_SECONDS, new_threshold),
        )

    return pause_statistic, target_threshold, new_threshold


def should_record_intra_sentence_pause(
    pause_duration: float,
    current_threshold: float,
) -> bool:
    """过滤VAD抖动、句尾边缘值和异常长停顿，只保留明确句中停顿。"""
    return (
        pause_duration >= ENDPOINT_MIN_INTRA_PAUSE_SECONDS
        and pause_duration < current_threshold
        and pause_duration < current_threshold * 0.9
    )


def calculate_effective_endpoint_threshold(
    current_threshold: float,
    voiced_duration_seconds: float,
    resumed_after_recent_endpoint: bool,
    adaptive_endpoint_enabled: bool,
) -> tuple[float, bool]:
    """只对刚发生过Endpoint后的短碎片增加一次保守等待。"""
    short_utterance_guard = (
        adaptive_endpoint_enabled
        and resumed_after_recent_endpoint
        and voiced_duration_seconds < ENDPOINT_MIN_SPEECH_SECONDS
    )
    if not short_utterance_guard:
        return current_threshold, False
    return (
        min(
            ENDPOINT_MAX_SECONDS,
            current_threshold
            + ENDPOINT_SHORT_UTTERANCE_EXTRA_WAIT_SECONDS,
        ),
        True,
    )


async def sensevoice_asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    asr_ready: asyncio.Event,
    benchmark_mode: bool = False,
    trace_session_id: str | None = None,
    event_callback=None,
) -> None:
    """用现有 RMS VAD 收集整句音频，再交给 SenseVoiceSmall 推理。"""
    print("ASR 正在加载 SenseVoiceSmall...")
    try:
        model = await asyncio.to_thread(load_sensevoice_model)
    except Exception as error:
        print(
            "[SenseVoice Error] 模型加载失败："
            f"{type(error).__name__}: {error}"
        )
        if event_callback is not None:
            try:
                event_callback(
                    {
                        "type": "error",
                        "message": (
                            "SenseVoice model failed to load: "
                            f"{type(error).__name__}"
                        ),
                    }
                )
            except Exception as callback_error:
                print(
                    "[Event Callback Warning] "
                    f"{type(callback_error).__name__}"
                )
        asr_ready.set()
        await text_queue.put(None)

        # 继续消费到 Sentinel，避免 audio_queue 满后阻塞停止流程。
        while await audio_queue.get() is not None:
            pass
        print("SenseVoice ASR Worker 已结束")
        return

    print("ASR 已就绪：SenseVoiceSmall 整句离线识别（CPU）")
    asr_ready.set()
    if event_callback is not None:
        try:
            event_callback({"type": "status", "status": "Listening"})
        except Exception as callback_error:
            print(
                "[Event Callback Warning] "
                f"{type(callback_error).__name__}"
            )

    is_speaking = False
    last_voice_time = None
    utterance_chunks = []
    silence_started_at = None
    recent_pause_durations = deque(maxlen=10)
    current_endpoint_threshold = ENDPOINT_BASE_SECONDS
    voiced_duration_seconds = 0.0
    voiced_chunk_count = 0
    last_endpoint_time = None
    resumed_after_recent_endpoint = False
    time_since_last_endpoint = None
    endpoint_evaluation_logged = False
    speech_started_at = None
    normal_sentence_id = 0
    adaptive_endpoint_enabled = (
        ENDPOINT_ADAPTIVE_ENABLED and not benchmark_mode
    )

    if not benchmark_mode:
        print(
            "Adaptive Endpointing: "
            f"{'ON' if adaptive_endpoint_enabled else 'OFF'}"
        )
        print(f"Base Threshold: {ENDPOINT_BASE_SECONDS * 1000:.0f} ms")
        print(f"Min Threshold: {ENDPOINT_MIN_SECONDS * 1000:.0f} ms")
        print(f"Max Threshold: {ENDPOINT_MAX_SECONDS * 1000:.0f} ms")
        print(
            "Current Endpoint Threshold: "
            f"{current_endpoint_threshold * 1000:.0f} ms"
        )
        print(
            "Short Utterance Guard: speech < "
            f"{ENDPOINT_MIN_SPEECH_SECONDS * 1000:.0f} ms, "
            "extra wait "
            f"{ENDPOINT_SHORT_UTTERANCE_EXTRA_WAIT_SECONDS * 1000:.0f} ms"
        )

    async def recognize_and_publish(
        last_voice_at: float,
        endpoint_triggered_at: float,
        current_speech_started_at: float | None,
        endpoint_latency_ms: float | None,
        speech_duration_seconds: float,
        speech_voice_chunk_count: int,
    ) -> None:
        nonlocal normal_sentence_id
        utterance_audio = np.concatenate(utterance_chunks).astype(
            np.float32,
            copy=False,
        )

        inference_started_at = time.perf_counter()
        if event_callback is not None:
            try:
                event_callback(
                    {"type": "status", "status": "Recognizing"}
                )
            except Exception as callback_error:
                print(
                    "[Event Callback Warning] "
                    f"{type(callback_error).__name__}"
                )
        try:
            recognized_text = await asyncio.to_thread(
                recognize_sensevoice_sync,
                model,
                utterance_audio,
            )
        except Exception as error:
            print(
                "[SenseVoice Error] 推理失败："
                f"{type(error).__name__}: {error}"
            )
            if event_callback is not None:
                try:
                    event_callback(
                        {
                            "type": "error",
                            "message": (
                                "SenseVoice recognition failed: "
                                f"{type(error).__name__}"
                            ),
                        }
                    )
                except Exception as callback_error:
                    print(
                        "[Event Callback Warning] "
                        f"{type(callback_error).__name__}"
                    )
            recognized_text = ""
        result_received_at = time.perf_counter()

        asr_inference_latency_ms = (
            result_received_at - inference_started_at
        ) * 1000
        speech_end_to_result_latency_ms = (
            result_received_at - last_voice_at
        ) * 1000
        endpoint_to_result_latency_ms = (
            result_received_at - endpoint_triggered_at
        ) * 1000
        result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(
            f"[ASR SenseVoice Final {result_time}] {recognized_text} "
            f"| SenseVoice Inference Latency: "
            f"{asr_inference_latency_ms:.0f} ms "
            f"| Speech End To Result Latency: "
            f"{speech_end_to_result_latency_ms:.0f} ms "
            f"| Endpoint Trigger To Result: "
            f"{endpoint_to_result_latency_ms:.0f} ms"
        )
        if benchmark_mode:
            await text_queue.put(
                (
                    recognized_text,
                    endpoint_latency_ms,
                    asr_inference_latency_ms,
                    speech_end_to_result_latency_ms,
                )
            )
        else:
            normal_sentence_id += 1
            trace_id = (
                f"{trace_session_id}-{normal_sentence_id}"
                if trace_session_id
                else f"normal-{normal_sentence_id}"
            )
            # Normal Mode 从 Endpoint 开始携带轻量 Trace Metadata；
            # Benchmark 继续保持上面的历史四元组结构。
            await text_queue.put(
                {
                    "trace_id": trace_id,
                    "sentence_id": normal_sentence_id,
                    "raw_text": recognized_text,
                    "text": recognized_text,
                    "speech_started_at": current_speech_started_at,
                    "last_voice_at": last_voice_at,
                    "speech_duration_ms": speech_duration_seconds * 1000,
                    "voice_chunk_count": speech_voice_chunk_count,
                    "endpoint_triggered_at": endpoint_triggered_at,
                    "speech_end_detected_at": endpoint_triggered_at,
                    "endpoint_wait_ms": (
                        endpoint_triggered_at - last_voice_at
                    ) * 1000,
                    "asr_started_at": inference_started_at,
                    "asr_result_at": result_received_at,
                    "hotword_done_at": None,
                }
            )
            print(
                f"[Trace] {trace_id} | Sentence ID: "
                f"{normal_sentence_id}"
            )

    while True:
        queue_item = await audio_queue.get()

        if queue_item is None:
            if is_speaking and utterance_chunks:
                last_voice_at = last_voice_time or time.perf_counter()
                await recognize_and_publish(
                    last_voice_at,
                    time.perf_counter(),
                    speech_started_at,
                    None,
                    voiced_duration_seconds,
                    voiced_chunk_count,
                )

            await text_queue.put(None)
            print("SenseVoice ASR Worker 已结束")
            break

        audio_chunk, chunk_created_at, chunk_clock_time = queue_item
        rms = float(np.sqrt(np.mean(audio_chunk**2))) * 1000
        print(
            f"[Audio {chunk_clock_time}] Chunk: {audio_chunk.size} samples "
            f"| RMS: {rms:.1f}"
        )

        # 使用 Audio Worker 记录的采集时间，避免 ASR 推理期间 Queue 积压后，
        # 用处理时间误算 Silence（静音）和句间恢复间隔。
        vad_now = chunk_created_at
        if rms > VAD_RMS_THRESHOLD:
            if not is_speaking:
                utterance_chunks = []
                silence_started_at = None
                speech_started_at = vad_now
                voiced_duration_seconds = 0.0
                voiced_chunk_count = 0
                endpoint_evaluation_logged = False
                time_since_last_endpoint = (
                    vad_now - last_endpoint_time
                    if last_endpoint_time is not None
                    else None
                )
                # 刚刚发生过 Endpoint 又快速恢复语音，下一段短音频更像误切碎片。
                resumed_after_recent_endpoint = (
                    adaptive_endpoint_enabled
                    and time_since_last_endpoint is not None
                    and time_since_last_endpoint <= ENDPOINT_MAX_SECONDS
                )
            elif silence_started_at is not None:
                pause_duration = vad_now - silence_started_at

                # 只有 voice → silence → voice 且尚未达到句尾阈值，
                # 且不属于极短VAD抖动或接近句尾的边缘值，才记录为句中停顿。
                if (
                    adaptive_endpoint_enabled
                    and should_record_intra_sentence_pause(
                        pause_duration,
                        current_endpoint_threshold,
                    )
                ):
                    recent_pause_durations.append(pause_duration)
                    (
                        pause_statistic,
                        target_threshold,
                        current_endpoint_threshold,
                    ) = calculate_adaptive_endpoint_threshold(
                        current_endpoint_threshold,
                        recent_pause_durations,
                    )
                    print("[Adaptive Endpoint]")
                    print(
                        "Intra-sentence Pause: "
                        f"{pause_duration * 1000:.0f} ms"
                    )
                    print(
                        "Recent Median Pause: "
                        f"{pause_statistic * 1000:.0f} ms"
                    )
                    print(
                        "Target Threshold: "
                        f"{target_threshold * 1000:.0f} ms"
                    )
                    print(
                        "Current Threshold: "
                        f"{current_endpoint_threshold * 1000:.0f} ms"
                    )

                silence_started_at = None
                endpoint_evaluation_logged = False
            is_speaking = True
            last_voice_time = vad_now
            voiced_duration_seconds += CHUNK_DURATION_SECONDS
            voiced_chunk_count += 1
            utterance_chunks.append(audio_chunk)
        elif is_speaking and last_voice_time is not None:
            utterance_chunks.append(audio_chunk)
            if silence_started_at is None:
                # 最后一次检测到语音的时间近似表示静音开始时间。
                silence_started_at = last_voice_time
            silence_seconds = vad_now - last_voice_time
            (
                effective_endpoint_threshold,
                short_utterance_guard,
            ) = calculate_effective_endpoint_threshold(
                current_endpoint_threshold,
                voiced_duration_seconds,
                resumed_after_recent_endpoint,
                adaptive_endpoint_enabled,
            )

            if (
                not benchmark_mode
                and silence_seconds >= current_endpoint_threshold
                and not endpoint_evaluation_logged
            ):
                pause_statistic = (
                    median(recent_pause_durations)
                    if recent_pause_durations
                    else None
                )
                print("[Adaptive Endpoint]")
                print(
                    "Speech Duration: "
                    f"{voiced_duration_seconds * 1000:.0f} ms "
                    f"({voiced_chunk_count} voiced chunks)"
                )
                print(f"Silence: {silence_seconds * 1000:.0f} ms")
                print(
                    "Base Threshold: "
                    f"{ENDPOINT_BASE_SECONDS * 1000:.0f} ms"
                )
                print(
                    "Adaptive Threshold: "
                    f"{current_endpoint_threshold * 1000:.0f} ms"
                )
                print(
                    "Effective Threshold: "
                    f"{effective_endpoint_threshold * 1000:.0f} ms"
                )
                print(
                    "Short Utterance Guard: "
                    f"{'ON' if short_utterance_guard else 'OFF'}"
                )
                print(f"Pause Sample Count: {len(recent_pause_durations)}")
                print(
                    "Pause Statistic (median): "
                    + (
                        f"{pause_statistic * 1000:.0f} ms"
                        if pause_statistic is not None
                        else "N/A"
                    )
                )
                print(
                    "Time Since Last Endpoint: "
                    + (
                        f"{time_since_last_endpoint * 1000:.0f} ms"
                        if time_since_last_endpoint is not None
                        else "N/A"
                    )
                )
                endpoint_evaluation_logged = True

            if silence_seconds >= effective_endpoint_threshold:
                last_voice_at = last_voice_time
                endpoint_latency_ms = (
                    vad_now - last_voice_at
                ) * 1000
                endpoint_reason = (
                    "short_utterance_confirmed"
                    if short_utterance_guard
                    else "normal_endpoint"
                )
                print("[Speech End]")
                print(
                    "speech_duration_ms = "
                    f"{voiced_duration_seconds * 1000:.0f}"
                )
                print(f"silence_ms = {endpoint_latency_ms:.0f}")
                print(
                    "effective_threshold_ms = "
                    f"{effective_endpoint_threshold * 1000:.0f}"
                )
                print(f"reason = {endpoint_reason}")

                last_endpoint_time = vad_now

                await recognize_and_publish(
                    last_voice_at,
                    vad_now,
                    speech_started_at,
                    endpoint_latency_ms,
                    voiced_duration_seconds,
                    voiced_chunk_count,
                )

                # 一句提交后清空缓存，等待下一句话重新进入 speaking。
                is_speaking = False
                last_voice_time = None
                utterance_chunks = []
                silence_started_at = None
                voiced_duration_seconds = 0.0
                voiced_chunk_count = 0
                resumed_after_recent_endpoint = False
                time_since_last_endpoint = None
                endpoint_evaluation_logged = False
                speech_started_at = None
