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

# 以下假数据保留给 V0/V1；V2 的 main() 暂时不会启动假 ASR、翻译和 TTS。
AUDIO_CHUNKS = [
    "audio_chunk_1",
    "audio_chunk_2",
    "audio_chunk_3",
]

# V0 使用固定字典，模拟 ASR（自动语音识别）和翻译的处理结果。
FAKE_ASR_RESULTS = {
    "audio_chunk_1": "你好",
    "audio_chunk_2": "我今天",
    "audio_chunk_3": "想去深圳",
}

FAKE_TRANSLATION_RESULTS = {
    "你好": "Hello",
    "我今天": "Today I",
    "想去深圳": "want to go to Shenzhen",
}


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


async def audio_debug_worker(audio_queue: asyncio.Queue) -> None:
    """消费真实音频块，打印大小和基础音量。"""
    while True:
        queue_item = await audio_queue.get()

        if queue_item is None:
            print("Audio Debug Worker 已结束")
            break

        audio_chunk, _, chunk_clock_time = queue_item

        # RMS（均方根音量）：先平方、求平均，再开平方。
        # float32 音频通常在 -1 到 1，乘 1000 只是让显示数值更直观。
        rms = float(np.sqrt(np.mean(audio_chunk**2))) * 1000
        print(
            f"[Audio {chunk_clock_time}] Chunk: {audio_chunk.size} samples "
            f"| RMS: {rms:.1f}"
        )


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

    while True:
        queue_item = await audio_queue.get()

        if queue_item is None:
            # FinalResult（最终稳定结果）会刷新停止前尚未结束的语音。
            remaining_result = json.loads(recognizer.FinalResult())
            stable_text = remaining_result.get("text", "").strip()

            if stable_text:
                result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[ASR Final {result_time}] {stable_text}")

                # 只有 Stable Text（稳定文本）才能进入 text_queue，
                # 避免下游以后重复翻译不断变化的部分结果。
                await text_queue.put(stable_text)

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

        # sounddevice 给出 float32 [-1, 1]；Vosk 需要原始 PCM16（16 位 PCM）bytes。
        # PCM（脉冲编码调制）转换：限制范围、乘 32767、转 int16，再转 bytes。
        pcm16_audio = (np.clip(audio_chunk, -1.0, 1.0) * 32767).astype(
            np.int16
        )
        pcm16_bytes = pcm16_audio.tobytes()

        # 本地识别会使用 CPU，放进线程，避免阻塞麦克风所在的事件循环。
        # Vosk 用基础 Endpointing（语句结束检测）判断一句话是否已经稳定。
        is_final = await asyncio.to_thread(
            recognizer.AcceptWaveform,
            pcm16_bytes,
        )

        # Latency（延迟）是从当前 Chunk 入队到结果读取完成的近似值。
        latency_ms = (time.perf_counter() - chunk_created_at) * 1000
        result_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if is_final:
            final_result = json.loads(recognizer.Result())
            stable_text = final_result.get("text", "").strip()
            last_partial = ""

            if stable_text:
                print(
                    f"[ASR Final {result_time}] {stable_text} "
                    f"| 近似延迟: {latency_ms:.0f} ms"
                )
                await text_queue.put(stable_text)
        else:
            # Partial Result（部分识别结果）只用于观察实时变化，不进入文本队列。
            partial_result = json.loads(recognizer.PartialResult())
            partial_text = partial_result.get("partial", "").strip()

            if partial_text and partial_text != last_partial:
                print(
                    f"[ASR Partial {result_time}] {partial_text} "
                    f"| 近似延迟: {latency_ms:.0f} ms"
                )
                last_partial = partial_text


async def text_debug_worker(text_queue: asyncio.Queue) -> None:
    """验证只有稳定识别文本会进入 text_queue。"""
    while True:
        stable_text = await text_queue.get()

        if stable_text is None:
            print("Text Debug Worker 已结束")
            break

        print(f"[Text Queue] 收到稳定文本：{stable_text}")


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

    # Async Client（异步客户端）等待网络时不会阻塞 Event Loop（事件循环）。
    # Timeout（超时）设为 15 秒；关闭 SDK 自动重试，避免延迟无限增加。
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
                # Sentinel（哨兵值）继续传给翻译结果下游。
                await translated_queue.put(None)
                print("Translation Worker 已结束")
                break

            request_started_at = time.perf_counter()

            try:
                # Prompt（提示词）只包含当前稳定中文，不发送 ASR Partial。
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
                latency_ms = (
                    time.perf_counter() - request_started_at
                ) * 1000

                if not english_text:
                    print("[Translation Error] 模型返回了空文本")
                    continue

                print("\n[Translation]")
                print(f"中文：{chinese_text}")
                print(f"英文：{english_text}")
                print(f"翻译耗时：{latency_ms:.0f} ms")

                # translated_queue 中只放最终英文翻译。
                # 有界 Queue 通过 Backpressure（背压）限制无限积压。
                await translated_queue.put(english_text)

            except APITimeoutError:
                print("[Translation Error] 请求超时，请继续说下一句")
            except RateLimitError:
                # Rate Limit（速率限制）失败只影响当前句，不中断流水线。
                print("[Translation Error] API 触发速率限制")
            except APIConnectionError:
                print("[Translation Error] 无法连接 DeepSeek API")
            except APIStatusError as error:
                print(
                    "[Translation Error] API 请求失败，"
                    f"状态码：{error.status_code}"
                )
            except Exception as error:
                # 不打印请求头或 Key，只显示异常类型。
                print(
                    "[Translation Error] 翻译失败："
                    f"{type(error).__name__}"
                )
    finally:
        await client.close()


async def translation_debug_worker(
    translated_queue: asyncio.Queue,
) -> None:
    """验证最终英文已经进入 translated_queue。"""
    while True:
        english_text = await translated_queue.get()

        if english_text is None:
            print("Translation Debug Worker 已结束")
            break

        print(f"[Translated Queue] 收到英文：{english_text}")


def speak_sync(english_text: str) -> None:
    """同步创建语音引擎并完整播放一句英文。"""
    # 当前 Windows 环境复用同一个 Engine（语音引擎）时只播放第一句，
    # 因此继续为每句英文创建全新的 Engine，优先保证可靠播放。
    tts_engine = pyttsx3.init()

    # 优先选择英文 Voice（音色），避免使用系统默认的中文音色。
    voices = tts_engine.getProperty("voices")
    english_voice = next(
        (voice for voice in voices if "english" in voice.name.lower()),
        None,
    )
    if english_voice is not None:
        tts_engine.setProperty("voice", english_voice.id)

    # TTS（文字转语音）：先加入待朗读文本，再执行同步播放。
    tts_engine.say(english_text)

    # runAndWait() 是 Blocking（阻塞）操作，会等整句播放完成。
    tts_engine.runAndWait()
    tts_engine.stop()

    # 删除引用，让本句 Engine 释放；下一句会创建全新的 Engine。
    del tts_engine


async def tts_worker(translated_queue: asyncio.Queue) -> None:
    """消费英文翻译，并通过扬声器逐句播放。"""
    print("TTS 已就绪：pyttsx3 英文语音")

    while True:
        english_text = await translated_queue.get()

        if english_text is None:
            print("[TTS] 播放任务结束")
            break

        print("\n[TTS]")
        print(f"准备播放：{english_text}", flush=True)
        playback_started_at = time.perf_counter()

        try:
            # asyncio.to_thread（把同步阻塞操作放到工作线程）使播放不会
            # 卡住 Event Loop。这里仍等待本句播完，不并行播放多句。
            await asyncio.to_thread(speak_sync, english_text)
        except Exception as error:
            print(f"[TTS Error] 播放失败：{type(error).__name__}")
            continue

        playback_latency_ms = (
            time.perf_counter() - playback_started_at
        ) * 1000
        print("[TTS] 播放完成", flush=True)
        print(f"TTS 播放耗时：{playback_latency_ms:.0f} ms")

        # V5 暂未实现 Echo Cancellation（回声消除）；建议戴耳机测试，
        # 避免扬声器英文被持续工作的麦克风重新采集。


async def main() -> None:
    if not MODEL_PATH.exists():
        print(f"找不到 Vosk 中文模型：{MODEL_PATH}")
        return

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY")
        return

    # 关闭 Vosk 底层的详细日志，让终端专注显示识别结果。
    SetLogLevel(-1)

    # 有界 Queue 限制最多积压 5 个音频块，防止内存无限增长。
    audio_queue = asyncio.Queue(maxsize=5)
    text_queue = asyncio.Queue(maxsize=5)
    translated_queue = asyncio.Queue(maxsize=5)
    stop_event = asyncio.Event()
    microphone_ready = asyncio.Event()

    # V5 End-to-End（端到端）：麦克风 -> ASR -> 翻译 -> TTS。
    # translated_queue 只能有一个 Consumer（消费者），否则多个 Worker
    # 会分走不同句子；因此不再同时启动 translation_debug_worker。
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
