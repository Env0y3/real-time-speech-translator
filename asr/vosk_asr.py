import asyncio
import json
import time
from datetime import datetime

import numpy as np
from vosk import KaldiRecognizer, Model

from config import (
    ENDPOINT_SILENCE_SECONDS,
    MODEL_PATH,
    PREFERRED_SAMPLE_RATE,
    VAD_RMS_THRESHOLD,
)


async def vosk_asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    benchmark_mode: bool = False,
    asr_ready: asyncio.Event | None = None,
) -> None:
    """持续消费真实音频，输出中文流式识别结果。"""
    # ASR（自动语音识别）模型在本机运行，不需要 API Key。
    # Streaming ASR（流式语音识别）会持续复用同一个识别器。
    print("ASR 正在加载中文模型...")
    model = await asyncio.to_thread(Model, str(MODEL_PATH))
    recognizer = KaldiRecognizer(model, PREFERRED_SAMPLE_RATE)
    print("ASR 已就绪：Vosk 中文流式识别")
    if asr_ready is not None:
        asr_ready.set()

    async def publish_stable_text(
        stable_text: str,
        endpoint_latency_ms: float | None,
    ) -> None:
        """正常模式输出文本；Benchmark 模式同时输出 Endpoint 延迟。"""
        if benchmark_mode:
            await text_queue.put((stable_text, endpoint_latency_ms))
        else:
            await text_queue.put(stable_text)

    last_partial = ""
    is_speaking = False
    last_voice_time = None
    speech_end_time = None
    last_committed_text = None

    while True:
        queue_item = await audio_queue.get()

        if queue_item is None:
            # FinalResult（最终稳定结果）会刷新停止前尚未结束的语音。
            remaining_result = json.loads(recognizer.FinalResult())
            stable_text = remaining_result.get("text", "").strip()

            if stable_text and stable_text != last_committed_text:
                committed_at = time.perf_counter()
                endpoint_latency_ms = None
                result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                if speech_end_time is not None:
                    asr_final_wait_ms = (
                        committed_at - speech_end_time
                    ) * 1000
                    endpoint_latency_ms = asr_final_wait_ms
                    print(
                        f"[ASR Final {result_time}] {stable_text} "
                        f"| ASR Final Wait: {asr_final_wait_ms:.0f} ms"
                    )
                else:
                    print(f"[ASR Final {result_time}] {stable_text}")
                await publish_stable_text(
                    stable_text,
                    endpoint_latency_ms,
                )
                last_committed_text = stable_text

            speech_end_time = None

            # Sentinel（哨兵值）继续向文本下游传递。
            await text_queue.put(None)
            print("ASR Worker 已结束")
            break

        audio_chunk, chunk_created_at, chunk_clock_time = queue_item
        rms = float(np.sqrt(np.mean(audio_chunk**2))) * 1000
        print(
            f"[Audio {chunk_clock_time}] Chunk: {audio_chunk.size} samples "
            f"| RMS: {rms:.1f}"
        )

        # VAD（语音活动检测）：用 RMS 判断语音与静音，并用连续静音检测句尾。
        vad_now = time.perf_counter()
        if rms > VAD_RMS_THRESHOLD:
            if not is_speaking:
                # 新一句开始时清空上一段的防重复状态。
                speech_end_time = None
                last_committed_text = None
            is_speaking = True
            last_voice_time = vad_now
        elif is_speaking and last_voice_time is not None:
            silence_seconds = vad_now - last_voice_time
            if silence_seconds >= ENDPOINT_SILENCE_SECONDS:
                # 真实 Speech End 使用最后一次检测到语音的时间，而非检测时刻。
                speech_end_time = last_voice_time
                print(
                    "[VAD] Speech End detected "
                    f"| silence: {silence_seconds * 1000:.0f} ms"
                )

                # Partial 为空时不刷新识别器，继续让 Vosk 自动 Final 兜底。
                endpoint_partial = json.loads(recognizer.PartialResult())
                endpoint_candidate = endpoint_partial.get(
                    "partial", ""
                ).strip()

                if endpoint_candidate:
                    endpoint_result = json.loads(
                        await asyncio.to_thread(recognizer.FinalResult)
                    )
                    endpoint_text = endpoint_result.get("text", "").strip()

                    # FinalResult 会刷新当前段；Reset 后同一 recognizer
                    # 可以从头继续接收下一句话。
                    await asyncio.to_thread(recognizer.Reset)
                    last_partial = ""

                    if (
                        endpoint_text
                        and endpoint_text != last_committed_text
                    ):
                        last_committed_text = endpoint_text
                        endpoint_commit_latency_ms = (
                            time.perf_counter() - speech_end_time
                        ) * 1000
                        await publish_stable_text(
                            endpoint_text,
                            endpoint_commit_latency_ms,
                        )
                        result_time = datetime.now().strftime(
                            "%H:%M:%S.%f"
                        )[:-3]
                        print(
                            f"[ASR Endpoint Final {result_time}] "
                            f"{endpoint_text} | Endpoint Commit Latency: "
                            f"{endpoint_commit_latency_ms:.0f} ms"
                        )

                    speech_end_time = None

                # 重置后，后续静音 Chunk 不会重复触发 Speech End。
                is_speaking = False
                last_voice_time = None

        # float32 [-1, 1] 转为 Vosk 需要的 PCM16（16 位 PCM）bytes。
        pcm16_audio = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(
            np.int16
        )
        pcm16_bytes = pcm16_audio.tobytes()

        # 本地识别放入线程，避免阻塞麦克风所在的事件循环。
        is_final = await asyncio.to_thread(
            recognizer.AcceptWaveform,
            pcm16_bytes,
        )

        latency_ms = (time.perf_counter() - chunk_created_at) * 1000
        result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if is_final:
            final_result = json.loads(recognizer.Result())
            stable_text = final_result.get("text", "").strip()
            last_partial = ""

            if stable_text and stable_text != last_committed_text:
                committed_at = time.perf_counter()
                endpoint_latency_ms = None
                if speech_end_time is not None:
                    asr_final_wait_ms = (
                        committed_at - speech_end_time
                    ) * 1000
                    endpoint_latency_ms = asr_final_wait_ms
                    print(
                        f"[ASR Final {result_time}] {stable_text} "
                        f"| 近似延迟: {latency_ms:.0f} ms "
                        f"| ASR Final Wait: {asr_final_wait_ms:.0f} ms"
                    )
                else:
                    print(
                        f"[ASR Final {result_time}] {stable_text} "
                        f"| 近似延迟: {latency_ms:.0f} ms"
                    )
                    if last_voice_time is not None:
                        endpoint_latency_ms = (
                            committed_at - last_voice_time
                        ) * 1000
                await publish_stable_text(
                    stable_text,
                    endpoint_latency_ms,
                )
                last_committed_text = stable_text

            # 当前 Vosk Final 已处理，避免旧的句尾时间被下一句话复用。
            speech_end_time = None
            is_speaking = False
            last_voice_time = None
        else:
            # Partial Result（部分识别结果）只打印，不进入翻译队列。
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get("partial", "").strip()

            if partial_text and partial_text != last_partial:
                print(
                    f"[ASR Partial {result_time}] {partial_text} "
                    f"| 近似延迟: {latency_ms:.0f} ms"
                )
                last_partial = partial_text

