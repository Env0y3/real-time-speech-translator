import asyncio
import json
import os
import re
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
ASR_PROVIDER = "sensevoice"  # Benchmark 可选："vosk" 或 "sensevoice"
VOSK_MODEL_NAME = "vosk-model-small-cn-0.22"
SENSEVOICE_MODEL_NAME = "iic/SenseVoiceSmall"
BENCHMARK_REPEATS = 3
TEST_SENTENCES = [
    "你好",
    "今天天气怎么样",
    "你叫什么名字",
    "我今天想去深圳",
    "我今天下午准备去深圳找朋友吃饭",
    "明天早上八点提醒我去上课",
    "广州",
    "深圳",
    "东莞",
    "贵州",
    "杭州",
    "我最近正在学习 LangGraph",
    "我用 DeepSeek 做翻译",
    "今天想休息",
    "今天想去西安",
]
HOTWORD_TEST_SENTENCES = [
    "我用 DeepSeek 做翻译",
    "我最近正在学习 LangGraph",
    "我最近在学 Python",
    "我用 Redis 做缓存",
    "这个项目使用 WebSocket",
    "我准备使用 Docker 部署项目",
    "我使用 FastAPI 搭建后端",
    "我通过 GitHub 管理代码",
    "我正在使用 OpenAI 的模型",
    "我正在使用 ChatGPT",
]
# 当前 SenseVoiceSmall 推理代码不读取 FunASR 的通用 hotword 参数。
# 因此 ON 模式只加载并记录词表，不伪造模型已经应用 Hotword Biasing（热词偏置）。
SENSEVOICE_HOTWORD_BIASING_SUPPORTED = False
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / VOSK_MODEL_NAME
)
BENCHMARK_RESULTS_PATH = (
    Path(__file__).resolve().parent / "asr_benchmark_results.jsonl"
)
HOTWORDS_PATH = Path(__file__).resolve().parent / "hotwords.json"
NORMALIZATION_PUNCTUATION = set(
    "，。！？；：、“”‘’,.!?;:\"'"
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


async def sensevoice_asr_worker(
    audio_queue: asyncio.Queue,
    text_queue: asyncio.Queue,
    asr_ready: asyncio.Event,
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
        await text_queue.put(
            (
                recognized_text,
                endpoint_latency_ms,
                asr_inference_latency_ms,
                speech_end_to_result_latency_ms,
            )
        )

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
            is_speaking = True
            last_voice_time = vad_now
            utterance_chunks.append(audio_chunk)
        elif is_speaking and last_voice_time is not None:
            utterance_chunks.append(audio_chunk)
            silence_seconds = vad_now - last_voice_time

            if silence_seconds >= ENDPOINT_SILENCE_SECONDS:
                speech_end_time = last_voice_time
                endpoint_latency_ms = (
                    vad_now - speech_end_time
                ) * 1000
                print(
                    "[VAD] Speech End detected "
                    f"| silence: {endpoint_latency_ms:.0f} ms"
                )

                await recognize_and_publish(
                    speech_end_time,
                    endpoint_latency_ms,
                )

                # 一句提交后清空缓存，等待下一句话重新进入 speaking。
                is_speaking = False
                last_voice_time = None
                utterance_chunks = []


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


def save_benchmark_result(result: dict) -> None:
    """把一条 Benchmark 结果追加为 JSON Lines（每行一个 JSON 对象）。"""
    with BENCHMARK_RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_hotwords(path: Path) -> list[str]:
    """读取并合并 hotwords.json 中各类别的 Hotword（热词）。"""
    try:
        with path.open("r", encoding="utf-8") as file:
            categories = json.load(file)
    except FileNotFoundError:
        print(f"[Hotword] {path.name} not found")
        return []
    except json.JSONDecodeError as error:
        print(
            f"[Hotword] {path.name} JSON 格式错误："
            f"第 {error.lineno} 行，第 {error.colno} 列"
        )
        return []
    except OSError as error:
        print(f"[Hotword] 无法读取 {path.name}：{error}")
        return []

    if not isinstance(categories, dict):
        print(f"[Hotword] {path.name} 顶层必须是 JSON 对象")
        return []

    hotwords = []
    seen_hotwords = set()
    for category_name, category_words in categories.items():
        if not isinstance(category_words, list):
            print(f"[Hotword] 跳过非列表类别：{category_name}")
            continue

        for word in category_words:
            if not isinstance(word, str) or not word.strip():
                continue
            clean_word = word.strip()
            if clean_word not in seen_hotwords:
                hotwords.append(clean_word)
                seen_hotwords.add(clean_word)

    return hotwords


def normalize_text(text: str) -> str:
    """移除空白和常见标点，并把英文字母统一为小写。"""
    return "".join(
        character.lower()
        for character in text
        if not character.isspace()
        and character not in NORMALIZATION_PUNCTUATION
    )


def find_target_hotword(target_text: str, hotwords: list[str]) -> str | None:
    """从外部词表中找到目标句包含的热词，不使用品牌专属修正规则。"""
    normalized_target = normalize_text(target_text)
    for hotword in hotwords:
        if normalize_text(hotword) in normalized_target:
            return hotword
    return None


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """按 Unicode 字符计算两个字符串之间的 Levenshtein 编辑距离。"""
    previous_row = list(range(len(hypothesis) + 1))

    for reference_index, reference_character in enumerate(reference, start=1):
        current_row = [reference_index]

        for hypothesis_index, hypothesis_character in enumerate(
            hypothesis,
            start=1,
        ):
            insertion_cost = current_row[hypothesis_index - 1] + 1
            deletion_cost = previous_row[hypothesis_index] + 1
            substitution_cost = (
                previous_row[hypothesis_index - 1]
                + (reference_character != hypothesis_character)
            )
            current_row.append(
                min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]


async def benchmark_worker(
    text_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    microphone_ready: asyncio.Event,
    asr_ready: asyncio.Event,
    asr_provider: str,
    asr_model: str,
    test_sentences: list[str],
    benchmark_type: str,
    hotword_enabled: bool,
    hotwords: list[str],
) -> None:
    """逐句提示用户朗读，记录所选 ASR Provider 的最终结果。"""
    await microphone_ready.wait()
    await asr_ready.wait()
    provider_display_name = (
        "Vosk" if asr_provider == "vosk" else "SenseVoiceSmall"
    )

    total_planned = len(test_sentences) * BENCHMARK_REPEATS
    total_samples = 0
    exact_match_count = 0
    normalized_exact_match_count = 0
    total_cer = 0.0
    endpoint_latency_total = 0.0
    endpoint_latency_count = 0
    inference_latency_total = 0.0
    inference_latency_count = 0
    speech_end_to_result_total = 0.0
    speech_end_to_result_count = 0
    hotword_hit_count = 0
    hotword_sample_count = 0
    stopped_early = False

    if benchmark_type == "english_hotword":
        print("\n========== English Hotword Benchmark ==========")
        print(f"Hotword Mode: {'ON' if hotword_enabled else 'OFF'}")
        print(f"Hotword Count: {len(hotwords)}")
    else:
        print("\n========== ASR Benchmark Mode ==========")
    print(f"ASR Provider: {provider_display_name}")
    print(f"ASR Model: {asr_model}")
    print(f"测试句数：{len(test_sentences)}")
    print(f"每句重复：{BENCHMARK_REPEATS} 次")
    print(f"计划样本：{total_planned}")

    for sentence_index, target_text in enumerate(test_sentences, start=1):
        for repeat_index in range(1, BENCHMARK_REPEATS + 1):
            if stop_event.is_set():
                stopped_early = True
                break

            sample_number = total_samples + 1
            print("\n----------------------------------------")
            print(
                f"[句子 {sentence_index}/{len(test_sentences)} | "
                f"第 {repeat_index}/{BENCHMARK_REPEATS} 次 | "
                f"样本 {sample_number}/{total_planned}]"
            )
            print("请朗读：")
            print(f"“{target_text}”")

            queue_item = await text_queue.get()
            if queue_item is None:
                stopped_early = True
                break

            recognized_text = queue_item[0]
            endpoint_latency_ms = queue_item[1]
            asr_inference_latency_ms = (
                queue_item[2] if len(queue_item) > 2 else None
            )
            speech_end_to_result_latency_ms = (
                queue_item[3]
                if len(queue_item) > 3
                else endpoint_latency_ms
            )
            exact_match = recognized_text == target_text
            normalized_target = normalize_text(target_text)
            normalized_recognized = normalize_text(recognized_text)
            normalized_exact_match = (
                normalized_target == normalized_recognized
            )
            target_hotword = (
                find_target_hotword(target_text, hotwords)
                if benchmark_type == "english_hotword"
                else None
            )
            hotword_hit = (
                normalize_text(target_hotword) in normalized_recognized
                if target_hotword is not None
                else None
            )
            edit_distance = levenshtein_distance(
                normalized_target,
                normalized_recognized,
            )

            # CER（字符错误率）通常用目标字符数作为分母。
            # 空目标且识别也为空时记为 0；只有识别文本时安全记为 1。
            if normalized_target:
                cer = edit_distance / len(normalized_target)
            else:
                cer = 0.0 if not normalized_recognized else 1.0

            total_samples += 1
            if exact_match:
                exact_match_count += 1
            if normalized_exact_match:
                normalized_exact_match_count += 1
            if hotword_hit is not None:
                hotword_sample_count += 1
                if hotword_hit:
                    hotword_hit_count += 1
            total_cer += cer
            if endpoint_latency_ms is not None:
                endpoint_latency_total += endpoint_latency_ms
                endpoint_latency_count += 1
            if asr_inference_latency_ms is not None:
                inference_latency_total += asr_inference_latency_ms
                inference_latency_count += 1
            if speech_end_to_result_latency_ms is not None:
                speech_end_to_result_total += (
                    speech_end_to_result_latency_ms
                )
                speech_end_to_result_count += 1

            result = {
                "benchmark_type": benchmark_type,
                "hotword_enabled": hotword_enabled,
                "hotword_count": len(hotwords),
                "asr_provider": asr_provider,
                "asr_model": asr_model,
                "target": target_text,
                "recognized": recognized_text,
                "exact_match": exact_match,
                "normalized_target": normalized_target,
                "normalized_recognized": normalized_recognized,
                "normalized_exact_match": normalized_exact_match,
                "target_hotword": target_hotword,
                "hotword_hit": hotword_hit,
                "edit_distance": edit_distance,
                "cer": cer,
                "endpoint_latency_ms": (
                    round(endpoint_latency_ms, 1)
                    if endpoint_latency_ms is not None
                    else None
                ),
                "asr_inference_latency_ms": (
                    round(asr_inference_latency_ms, 1)
                    if asr_inference_latency_ms is not None
                    else None
                ),
                "speech_end_to_result_latency_ms": (
                    round(speech_end_to_result_latency_ms, 1)
                    if speech_end_to_result_latency_ms is not None
                    else None
                ),
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
            }
            await asyncio.to_thread(save_benchmark_result, result)

            print("目标：")
            print(target_text)
            print("识别：")
            print(recognized_text)
            print("标准化目标：")
            print(normalized_target)
            print("标准化识别：")
            print(normalized_recognized)
            print("Raw Exact Match:")
            print("PASS" if exact_match else "FAIL")
            print("Normalized Exact Match:")
            print("PASS" if normalized_exact_match else "FAIL")
            print("CER:")
            print(f"{cer * 100:.2f}%")
            if target_hotword is not None:
                print(f"Target Hotword: {target_hotword}")
                print(f"Hotword Hit: {'PASS' if hotword_hit else 'FAIL'}")
            if endpoint_latency_ms is not None:
                print(f"Endpoint Latency: {endpoint_latency_ms:.0f} ms")
            else:
                print("Endpoint Latency: N/A")
            if asr_inference_latency_ms is not None:
                print(
                    "ASR Inference Latency: "
                    f"{asr_inference_latency_ms:.0f} ms"
                )
            if speech_end_to_result_latency_ms is not None:
                print(
                    "Speech End To Result Latency: "
                    f"{speech_end_to_result_latency_ms:.0f} ms"
                )

        if stopped_early:
            break

    completed_all = total_samples == total_planned
    if completed_all:
        # 完成固定测试集后停止麦克风，并继续消费到 Sentinel，
        # 确保 ASR Worker 不会因 Queue（队列）背压而无法退出。
        stop_event.set()
        while await text_queue.get() is not None:
            pass

    exact_match_rate = (
        exact_match_count / total_samples * 100
        if total_samples
        else 0.0
    )
    normalized_exact_match_rate = (
        normalized_exact_match_count / total_samples * 100
        if total_samples
        else 0.0
    )
    average_cer = total_cer / total_samples if total_samples else 0.0
    average_endpoint_latency = (
        endpoint_latency_total / endpoint_latency_count
        if endpoint_latency_count
        else None
    )
    average_inference_latency = (
        inference_latency_total / inference_latency_count
        if inference_latency_count
        else None
    )
    average_speech_end_to_result_latency = (
        speech_end_to_result_total / speech_end_to_result_count
        if speech_end_to_result_count
        else None
    )
    hotword_hit_rate = (
        hotword_hit_count / hotword_sample_count * 100
        if hotword_sample_count
        else 0.0
    )

    if benchmark_type == "english_hotword":
        print("\n========== English Hotword Benchmark ==========")
    else:
        print("\n========== Benchmark Summary ==========")
    print(f"ASR Provider: {provider_display_name}")
    print(f"ASR Model: {asr_model}")
    if benchmark_type == "english_hotword":
        print(f"Hotword: {'ON' if hotword_enabled else 'OFF'}")
    print(f"Total Samples: {total_samples}")
    print("\nRaw Exact Match:")
    print(f"{exact_match_count} / {total_samples}")
    print(f"{exact_match_rate:.1f}%")
    print("\nNormalized Exact Match:")
    print(f"{normalized_exact_match_count} / {total_samples}")
    print(f"{normalized_exact_match_rate:.1f}%")
    print("\nAverage CER:")
    print(f"{average_cer * 100:.2f}%")
    if benchmark_type == "english_hotword":
        print("\nHotword Hit:")
        print(f"{hotword_hit_count} / {hotword_sample_count}")
        print("\nHotword Hit Rate:")
        print(f"{hotword_hit_rate:.1f}%")
    print("\nAverage Endpoint Latency:")
    if average_endpoint_latency is not None:
        print(f"{average_endpoint_latency:.0f} ms")
    else:
        print("N/A")
    print("\nAverage ASR Inference Latency:")
    if average_inference_latency is not None:
        print(f"{average_inference_latency:.0f} ms")
    else:
        print("N/A")
    print("\nAverage Speech End To Result Latency:")
    if average_speech_end_to_result_latency is not None:
        print(f"{average_speech_end_to_result_latency:.0f} ms")
    else:
        print("N/A")
    if total_samples:
        print("\nResults saved to:")
        print(BENCHMARK_RESULTS_PATH.name)
    else:
        print("尚未产生可保存的测试结果")
    if stopped_early:
        print("Benchmark 提前停止")
    print("=======================================")

    if completed_all:
        print("Benchmark 已完成，请按 Enter 退出")


def choose_run_mode() -> str | None:
    """让用户选择正常翻译或独立 ASR Benchmark 模式。"""
    print("请选择运行模式：")
    print("1. Normal Mode（正常实时翻译模式）")
    print("2. ASR Benchmark Mode（ASR 测试模式）")

    while True:
        try:
            run_mode = input("请输入 1 或 2：").strip()
        except EOFError:
            print("没有收到模式选择，程序结束")
            return None

        if run_mode in {"1", "2"}:
            return run_mode
        print("输入无效，请输入 1 或 2")


def choose_benchmark_type() -> str | None:
    """选择普通测试集或 English Hotword（英文热词）专项测试集。"""
    print("\n请选择 Benchmark 类型：")
    print("1. General Benchmark")
    print("2. English Hotword Benchmark")

    while True:
        try:
            choice = input("请输入 1 或 2：").strip()
        except EOFError:
            print("没有收到 Benchmark 类型，程序结束")
            return None

        if choice == "1":
            return "general"
        if choice == "2":
            return "english_hotword"
        print("输入无效，请输入 1 或 2")


def choose_hotword_mode() -> bool | None:
    """选择 Hotword OFF（关闭）或 ON（开启）测试标签。"""
    print("\n请选择 Hotword 模式：")
    print("1. Hotword OFF")
    print("2. Hotword ON")

    while True:
        try:
            choice = input("请输入 1 或 2：").strip()
        except EOFError:
            print("没有收到 Hotword 模式，程序结束")
            return None

        if choice == "1":
            return False
        if choice == "2":
            return True
        print("输入无效，请输入 1 或 2")


async def main() -> None:
    run_mode = choose_run_mode()
    if run_mode is None:
        return

    benchmark_type = "general"
    hotword_enabled = False
    hotwords = []
    test_sentences = TEST_SENTENCES

    if run_mode == "2":
        selected_benchmark_type = choose_benchmark_type()
        if selected_benchmark_type is None:
            return
        benchmark_type = selected_benchmark_type

        if benchmark_type == "english_hotword":
            selected_hotword_mode = choose_hotword_mode()
            if selected_hotword_mode is None:
                return
            hotword_enabled = selected_hotword_mode
            hotwords = load_hotwords(HOTWORDS_PATH)
            test_sentences = HOTWORD_TEST_SENTENCES
            print(f"Hotword Mode: {'ON' if hotword_enabled else 'OFF'}")

            if hotword_enabled:
                print(f"[Hotword] 已加载 {len(hotwords)} 个热词")
                if (
                    ASR_PROVIDER == "sensevoice"
                    and not SENSEVOICE_HOTWORD_BIASING_SUPPORTED
                ):
                    print(
                        "[Hotword] 当前 SenseVoiceSmall 路径无法直接做真正 "
                        "Hotword Biasing；不会向模型传入无效参数。"
                    )
                elif ASR_PROVIDER != "sensevoice":
                    print(
                        f"[Hotword] 当前 {ASR_PROVIDER} Benchmark 未接入"
                        "真正 Hotword Biasing。"
                    )

    if (
        run_mode == "2"
        and ASR_PROVIDER not in {"vosk", "sensevoice"}
    ):
        print('ASR_PROVIDER 只支持 "vosk" 或 "sensevoice"')
        return

    uses_vosk = run_mode == "1" or ASR_PROVIDER == "vosk"
    if uses_vosk and not MODEL_PATH.exists():
        print(f"找不到 Vosk 中文模型：{MODEL_PATH}")
        return

    if run_mode == "1" and not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY")
        return

    SetLogLevel(-1)

    audio_queue = asyncio.Queue(maxsize=5)
    text_queue = asyncio.Queue(maxsize=5)
    stop_event = asyncio.Event()
    microphone_ready = asyncio.Event()

    if run_mode == "2":
        asr_ready = asyncio.Event()

        if ASR_PROVIDER == "vosk":
            benchmark_asr_worker = asr_worker(
                audio_queue,
                text_queue,
                benchmark_mode=True,
                asr_ready=asr_ready,
            )
            benchmark_model_name = VOSK_MODEL_NAME
        else:
            benchmark_asr_worker = sensevoice_asr_worker(
                audio_queue,
                text_queue,
                asr_ready,
            )
            benchmark_model_name = SENSEVOICE_MODEL_NAME

        await asyncio.gather(
            audio_worker(audio_queue, stop_event, microphone_ready),
            benchmark_asr_worker,
            benchmark_worker(
                text_queue,
                stop_event,
                microphone_ready,
                asr_ready,
                ASR_PROVIDER,
                benchmark_model_name,
                test_sentences,
                benchmark_type,
                hotword_enabled,
                hotwords,
            ),
            wait_for_stop(stop_event, microphone_ready),
        )
    else:
        translated_queue = asyncio.Queue(maxsize=5)

        # Normal Mode 保留原来的完整端到端数据流。
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
