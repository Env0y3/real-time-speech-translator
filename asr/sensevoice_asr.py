import asyncio
import re
import time
from collections import deque
from datetime import datetime

import numpy as np

from config import (
    ENDPOINT_ADAPTIVE_ENABLED,
    ENDPOINT_BASE_SECONDS,
    ENDPOINT_MAX_SECONDS,
    ENDPOINT_MIN_SECONDS,
    ENDPOINT_SAFETY_MARGIN_SECONDS,
    ENDPOINT_SMOOTHING_ALPHA,
    SENSEVOICE_MODEL_NAME,
    VAD_RMS_THRESHOLD,
)


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
    """根据近期句中停顿计算平均值、目标阈值和EMA平滑后的阈值。"""
    average_pause = sum(recent_pause_durations) / len(
        recent_pause_durations
    )

    if len(recent_pause_durations) < 3:
        target_threshold = ENDPOINT_BASE_SECONDS
        new_threshold = current_threshold
    else:
        target_threshold = min(
            ENDPOINT_MAX_SECONDS,
            max(
                ENDPOINT_MIN_SECONDS,
                average_pause + ENDPOINT_SAFETY_MARGIN_SECONDS,
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

    return average_pause, target_threshold, new_threshold


async def sensevoice_asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    asr_ready: asyncio.Event,
    benchmark_mode: bool = False,
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
        asr_ready.set()
        await text_queue.put(None)

        # 继续消费到 Sentinel，避免 audio_queue 满后阻塞停止流程。
        while await audio_queue.get() is not None:
            pass
        print("SenseVoice ASR Worker 已结束")
        return

    print("ASR 已就绪：SenseVoiceSmall 整句离线识别（CPU）")
    asr_ready.set()

    is_speaking = False
    last_voice_time = None
    utterance_chunks = []
    silence_started_at = None
    recent_pause_durations = deque(maxlen=10)
    current_endpoint_threshold = ENDPOINT_BASE_SECONDS
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

    async def recognize_and_publish(
        speech_end_time: float,
        endpoint_latency_ms: float | None,
    ) -> None:
        utterance_audio = np.concatenate(utterance_chunks).astype(
            np.float32,
            copy=False,
        )

        inference_started_at = time.perf_counter()
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
            recognized_text = ""
        result_received_at = time.perf_counter()

        asr_inference_latency_ms = (
            result_received_at - inference_started_at
        ) * 1000
        speech_end_to_result_latency_ms = (
            result_received_at - speech_end_time
        ) * 1000
        result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        print(
            f"[ASR SenseVoice Final {result_time}] {recognized_text} "
            f"| SenseVoice Inference Latency: "
            f"{asr_inference_latency_ms:.0f} ms "
            f"| Speech End To Result Latency: "
            f"{speech_end_to_result_latency_ms:.0f} ms"
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
            # Normal Mode 下游 Translation Worker 只接收字符串。
            await text_queue.put(recognized_text)

    while True:
        queue_item = await audio_queue.get()

        if queue_item is None:
            if is_speaking and utterance_chunks:
                speech_end_time = last_voice_time or time.perf_counter()
                await recognize_and_publish(speech_end_time, None)

            await text_queue.put(None)
            print("SenseVoice ASR Worker 已结束")
            break

        audio_chunk, _chunk_created_at, chunk_clock_time = queue_item
        rms = float(np.sqrt(np.mean(audio_chunk**2))) * 1000
        print(
            f"[Audio {chunk_clock_time}] Chunk: {audio_chunk.size} samples "
            f"| RMS: {rms:.1f}"
        )

        vad_now = time.perf_counter()
        if rms > VAD_RMS_THRESHOLD:
            if not is_speaking:
                utterance_chunks = []
                silence_started_at = None
            elif silence_started_at is not None:
                pause_duration = vad_now - silence_started_at

                # 只有 voice → silence → voice 且尚未达到句尾阈值，
                # 才属于 Intra-sentence Pause（句中停顿）。
                if (
                    adaptive_endpoint_enabled
                    and pause_duration < current_endpoint_threshold
                ):
                    recent_pause_durations.append(pause_duration)
                    (
                        average_pause,
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
                        "Recent Average Pause: "
                        f"{average_pause * 1000:.0f} ms"
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
            is_speaking = True
            last_voice_time = vad_now
            utterance_chunks.append(audio_chunk)
        elif is_speaking and last_voice_time is not None:
            utterance_chunks.append(audio_chunk)
            if silence_started_at is None:
                # 最后一次检测到语音的时间近似表示静音开始时间。
                silence_started_at = last_voice_time
            silence_seconds = vad_now - last_voice_time

            if silence_seconds >= current_endpoint_threshold:
                speech_end_time = last_voice_time
                endpoint_latency_ms = (
                    vad_now - speech_end_time
                ) * 1000
                print(
                    "[VAD] Speech End detected "
                    f"| silence: {endpoint_latency_ms:.0f} ms "
                    f"| threshold: "
                    f"{current_endpoint_threshold * 1000:.0f} ms"
                )

                await recognize_and_publish(
                    speech_end_time,
                    endpoint_latency_ms,
                )

                # 一句提交后清空缓存，等待下一句话重新进入 speaking。
                is_speaking = False
                last_voice_time = None
                utterance_chunks = []
                silence_started_at = None
