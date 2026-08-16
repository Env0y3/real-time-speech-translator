import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyttsx3
import sounddevice as sd
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from vosk import KaldiRecognizer, Model, SetLogLevel


# V2 的 Microphone（麦克风）基础配置。
PREFERRED_SAMPLE_RATE = 16_000  # Sample Rate（采样率）：每秒 16000 个样本
CHANNELS = 1  # Channel（声道）：1 表示单声道
CHUNK_DURATION_SECONDS = 0.2  # Chunk（音频块）：每块大约 200 ms
VAD_RMS_THRESHOLD = 10.0  # RMS 高于此值时认为当前有人说话
ENDPOINT_SILENCE_SECONDS = 0.6  # 连续静音达到此时长时认为一句话结束
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "vosk-model-small-cn-0.22"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# System Prompt（系统提示词）只定义单一的中译英职责。

TRANSLATION_SYSTEM_PROMPT = (
    "你是一个实时中英翻译器。"
    "请把用户提供的中文准确、自然地翻译成英文。"
    "只输出英文翻译，不要解释，不要添加额外内容。"
)

async def audio_worker(
    audio_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    microphone_ready: asyncio.Event,
) -> None:
    """持续读取真实麦克风，并把音频块放入异步队列。"""
    device_info = sd.query_devices(kind="input")
    sample_rate = PREFERRED_SAMPLE_RATE

    # 优先使用 16000 Hz；设备不支持时，兼容到设备的默认采样率。
    try:
        sd.check_input_settings(
            channels=CHANNELS,
            dtype="float32",
            samplerate=sample_rate,
        )
    except sd.PortAudioError:
        sample_rate = int(device_info["default_samplerate"])
        print(f"[Audio] 设备不支持 16000 Hz，改用 {sample_rate} Hz")

    chunk_samples = int(sample_rate * CHUNK_DURATION_SECONDS)

    try:
        # InputStream 表示 Audio Stream（音频流），不会先录制完整 wav 文件。
        with sd.InputStream(
            samplerate=sample_rate,
            channels=CHANNELS,
            dtype="float32",
        ) as stream:
            print(f"麦克风已启动：{device_info['name']}")
            print(
                f"[Audio] {sample_rate} Hz | 单声道 | "
                f"每个 Chunk {chunk_samples} samples"
            )
            microphone_ready.set()

            while not stop_event.is_set():
                # V2 暂不使用 Callback（回调函数）模式：Callback 运行在音频线程，
                # 不能 await，也不能直接操作非 Thread-safe（线程安全）的 asyncio.Queue。
                # read() 是 Blocking（阻塞）读取，所以用 to_thread() 放到工作线程。
                # 这样不会卡住 asyncio 的 Event Loop（事件循环）。
                audio_chunk, overflowed = await asyncio.to_thread(
                    stream.read,
                    chunk_samples,
                )

                if overflowed:
                    print("[Audio] 警告：输入发生溢出，部分音频可能丢失")

                # 回到事件循环后再操作 Queue（队列），不需要跨线程直接改队列。
                # await put() 会在队列满时等待，体现 Backpressure（背压）。
                chunk_created_at = time.perf_counter()
                chunk_clock_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                await audio_queue.put(
                    (
                        audio_chunk.copy().reshape(-1),
                        chunk_created_at,
                        chunk_clock_time,
                    )
                )

    except sd.PortAudioError as error:
        print(f"[Audio] 无法打开或读取麦克风：{error}")
        stop_event.set()
    finally:
        microphone_ready.set()

        # None 仍是 Sentinel（哨兵值），通知调试 Worker 安全退出。
        await audio_queue.put(None)
        print("麦克风已停止")
        print("Audio Worker 已结束")


async def wait_for_stop(
    stop_event: asyncio.Event,
    microphone_ready: asyncio.Event,
) -> None:
    """等待用户按 Enter，通知音频任务停止。"""
    await microphone_ready.wait()

    if stop_event.is_set():
        return

    # input() 也是阻塞操作，放到线程中，避免阻塞事件循环。
    await asyncio.to_thread(input, "按 Enter 停止\n")
    stop_event.set()


async def asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
) -> None:
    """持续消费真实音频，输出中文流式识别结果。"""
    # ASR（自动语音识别）模型在本机运行，不需要 API Key。
    # Streaming ASR（流式语音识别）会持续复用同一个识别器。
    print("ASR 正在加载中文模型...")
    model = await asyncio.to_thread(Model, str(MODEL_PATH))
    recognizer = KaldiRecognizer(model, PREFERRED_SAMPLE_RATE)
    print("ASR 已就绪：Vosk 中文流式识别")

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
                result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                if speech_end_time is not None:
                    asr_final_wait_ms = (
                        time.perf_counter() - speech_end_time
                    ) * 1000
                    print(
                        f"[ASR Final {result_time}] {stable_text} "
                        f"| ASR Final Wait: {asr_final_wait_ms:.0f} ms"
                    )
                else:
                    print(f"[ASR Final {result_time}] {stable_text}")
                await text_queue.put(stable_text)
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
                        await text_queue.put(endpoint_text)
                        endpoint_commit_latency_ms = (
                            time.perf_counter() - speech_end_time
                        ) * 1000
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
                if speech_end_time is not None:
                    asr_final_wait_ms = (
                        time.perf_counter() - speech_end_time
                    ) * 1000
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
                await text_queue.put(stable_text)
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


async def translation_worker(
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
) -> None:
    """消费稳定中文文本，通过 DeepSeek 翻译成英文。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY")
        await translated_queue.put(None)
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=15.0,
        max_retries=0,
    )
    print(f"Translation 已就绪：DeepSeek {DEEPSEEK_MODEL}")

    try:
        while True:
            chinese_text = await text_queue.get()

            if chinese_text is None:
                await translated_queue.put(None)
                print("Translation Worker 已结束")
                break

            request_started_at = time.perf_counter()

            try:
                response = await client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": TRANSLATION_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": chinese_text},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    extra_body={"thinking": {"type": "disabled"}},
                )

                english_text = (
                    response.choices[0].message.content or ""
                ).strip()
                translation_finished_at = time.perf_counter()
                latency_ms = (
                    translation_finished_at - request_started_at
                ) * 1000

                if not english_text:
                    print("[Translation Error] 模型返回了空文本")
                    continue

                print("\n[Translation]")
                print(f"中文：{chinese_text}")
                print(f"英文：{english_text}")
                print(f"翻译耗时：{latency_ms:.0f} ms")
                await translated_queue.put(
                    (english_text, translation_finished_at)
                )

            except APITimeoutError:
                print("[Translation Error] 请求超时，请继续说下一句")
            except RateLimitError:
                print("[Translation Error] API 触发速率限制")
            except APIConnectionError:
                print("[Translation Error] 无法连接 DeepSeek API")
            except APIStatusError as error:
                print(
                    "[Translation Error] API 请求失败，"
                    f"状态码：{error.status_code}"
                )
            except Exception as error:
                print(
                    "[Translation Error] 翻译失败："
                    f"{type(error).__name__}"
                )
    finally:
        await client.close()


def speak_sync(english_text: str) -> float | None:
    """同步播放一句英文，并返回 TTS 首音事件的近似时间。"""
    # Windows 环境继续为每句创建新 Engine，保证连续多句可靠播放。
    tts_engine = pyttsx3.init()
    first_audio_started_at = None

    # started-word（开始朗读词语事件）是 pyttsx3 官方事件。
    # 当前 Windows SAPI5 驱动会在语音流启动时触发第一次回调；
    # 它不是扬声器硬件真正输出首帧的时间，只能作为 TTS 首音近似值。
    def on_started_word(name: str, location: int, length: int) -> None:
        nonlocal first_audio_started_at
        if first_audio_started_at is None:
            first_audio_started_at = time.perf_counter()

    callback_token = tts_engine.connect("started-word", on_started_word)

    voices = tts_engine.getProperty("voices")
    english_voice = next(
        (voice for voice in voices if "english" in voice.name.lower()),
        None,
    )
    if english_voice is not None:
        tts_engine.setProperty("voice", english_voice.id)
        

    tts_engine.say(english_text)
    tts_engine.runAndWait()
    tts_engine.disconnect(callback_token)
    tts_engine.stop()
    del tts_engine
    return first_audio_started_at


async def tts_worker(translated_queue: asyncio.Queue) -> None:
    """消费英文翻译，并通过扬声器逐句播放。"""
    print("TTS 已就绪：pyttsx3 英文语音")

    while True:
        queue_item = await translated_queue.get()

        if queue_item is None:
            print("[TTS] 播放任务结束")
            break

        english_text, translation_finished_at = queue_item

        print("\n[TTS]")
        print(f"准备播放：{english_text}", flush=True)
        playback_started_at = time.perf_counter()

        try:
            first_audio_started_at = await asyncio.to_thread(
                speak_sync,
                english_text,
            )
        except Exception as error:
            print(f"[TTS Error] 播放失败：{type(error).__name__}")
            continue

        total_playback_ms = (
            time.perf_counter() - playback_started_at
        ) * 1000
        print("[TTS] 播放完成", flush=True)
        if first_audio_started_at is not None:
            ttfa_ms = (
                first_audio_started_at - translation_finished_at
            ) * 1000
            print(f"TTFA: {ttfa_ms:.0f} ms（TTS 首音近似值）")
        else:
            print("TTFA: 无法获取 started-word 事件")
        print(f"TTS Total Playback: {total_playback_ms:.0f} ms")


async def main() -> None:
    if not MODEL_PATH.exists():
        print(f"找不到 Vosk 中文模型：{MODEL_PATH}")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY")
        return

    SetLogLevel(-1)

    audio_queue = asyncio.Queue(maxsize=5)
    text_queue = asyncio.Queue(maxsize=5)
    translated_queue = asyncio.Queue(maxsize=5)
    stop_event = asyncio.Event()
    microphone_ready = asyncio.Event()

    # 完整端到端数据流；translated_queue 只由 TTS 消费。
    await asyncio.gather(
        audio_worker(audio_queue, stop_event, microphone_ready),
        asr_worker(audio_queue, text_queue),
        translation_worker(text_queue, translated_queue),
        tts_worker(translated_queue),
        wait_for_stop(stop_event, microphone_ready),
    )

    print("所有任务正常退出")


if __name__ == "__main__":
    asyncio.run(main())
