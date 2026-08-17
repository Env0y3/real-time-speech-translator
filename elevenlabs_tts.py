import asyncio
import base64
import binascii
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed

from config import (
    ELEVENLABS_CHANNELS,
    ELEVENLABS_DTYPE,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SAMPLE_RATE,
    ELEVENLABS_VOICE_ID,
)
from performance_logger import PerformanceLogger
from tts import speak_sync


AUDIO_QUEUE_MAXSIZE = 20
TTS_SESSION_TIMEOUT_SECONDS = 60


class ElevenLabsTTSError(Exception):
    """一次 ElevenLabs 句级 TTS Session（会话）无法继续。"""


@dataclass
class SentenceMetrics:
    """保存一句英文在 ElevenLabs 中的时间点与音频计数。"""

    first_segment_ready_at: float
    first_text_sent_at: float | None = None
    first_audio_chunk_at: float | None = None
    first_playback_at: float | None = None
    session_finished_at: float | None = None
    audio_chunk_count: int = 0
    total_audio_bytes: int = 0


@dataclass
class SentenceState:
    """保存一个中文句子对应的所有 Translation Segment（翻译语块）。"""

    sentence_id: int
    metrics: SentenceMetrics
    segments: dict[int, str] = field(default_factory=dict)
    sentence_end_received: bool = False
    pipeline_finished: bool = False

    @property
    def full_text(self) -> str:
        return " ".join(
            self.segments[index].strip()
            for index in sorted(self.segments)
            if self.segments[index].strip()
        )


def _websocket_url() -> str:
    return (
        "wss://api.elevenlabs.io/v1/text-to-speech/"
        f"{ELEVENLABS_VOICE_ID}/stream-input"
        f"?model_id={ELEVENLABS_MODEL_ID}"
        f"&output_format={ELEVENLABS_OUTPUT_FORMAT}"
    )


def _safe_error_message(error: Exception, api_key: str) -> str:
    """隐藏 API Key，只输出足够定位问题的异常类型和消息。"""
    message = str(error).replace(api_key, "[REDACTED]").strip()
    return f"{type(error).__name__}: {message or 'No additional message'}"


def _decode_audio(audio_base64: str) -> bytes:
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ElevenLabsTTSError(
            f"Base64 audio decode failed: {error}"
        ) from error


def _open_output_stream() -> sd.OutputStream:
    """每个句级 Session 只创建一个 OutputStream（音频输出流）。"""
    try:
        output_stream = sd.OutputStream(
            samplerate=ELEVENLABS_SAMPLE_RATE,
            channels=ELEVENLABS_CHANNELS,
            dtype=ELEVENLABS_DTYPE,
        )
        output_stream.start()
        return output_stream
    except Exception as error:
        raise ElevenLabsTTSError(
            "OutputStream initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error


async def _close_output_stream(
    output_stream: sd.OutputStream | None,
) -> None:
    if output_stream is None:
        return
    try:
        await asyncio.to_thread(output_stream.stop)
    finally:
        await asyncio.to_thread(output_stream.close)


def _write_audio_chunk(
    output_stream: sd.OutputStream,
    audio_samples: np.ndarray,
) -> float:
    write_started_at = time.perf_counter()
    output_stream.write(audio_samples)
    return write_started_at


def _record_segment(state: SentenceState, item: dict) -> None:
    state.segments.setdefault(item["segment_index"], item["text"])


async def _initialize_connection(websocket, api_key: str) -> None:
    """使用已验证的 ElevenLabs WebSocket 初始化协议。"""
    await websocket.send(
        json.dumps(
            {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "speed": 1.0,
                },
                "xi_api_key": api_key,
            }
        )
    )


async def _send_segment(
    websocket,
    item: dict,
    state: SentenceState,
) -> None:
    _record_segment(state, item)
    text = item["text"].strip()
    if not text:
        return

    sent_at = time.perf_counter()
    if state.metrics.first_text_sent_at is None:
        state.metrics.first_text_sent_at = sent_at

    print(f"\n[Text Segment {item['segment_index']}]")
    print(text)
    # 统一保留一个尾随空格，避免相邻 Segment 拼成 Shenzhenthis。
    # flush 只提交当前已发送文字，不关闭 WebSocket；让首个稳定语块就能
    # 开始生成音频，而不必等待 DeepSeek 完整翻译或整句 EOS。
    await websocket.send(
        json.dumps({"text": f"{text} ", "flush": True})
    )


async def _send_sentence_segments(
    websocket,
    translated_queue: asyncio.Queue,
    first_item: dict,
    state: SentenceState,
) -> None:
    """把同一句的语块立即送入同一个 WebSocket，结束后再发送 EOS。"""
    current_item = first_item
    while True:
        if current_item.get("event") == "segment":
            if current_item["sentence_id"] != state.sentence_id:
                raise ElevenLabsTTSError("Translation sentence_id 顺序异常")
            await _send_segment(websocket, current_item, state)
            if current_item.get("is_final_segment"):
                state.sentence_end_received = True
                await websocket.send(json.dumps({"text": ""}))
                return
        elif current_item.get("event") == "sentence_end":
            if current_item["sentence_id"] != state.sentence_id:
                raise ElevenLabsTTSError("sentence_end 的 sentence_id 不匹配")
            # EOS（流结束信号）发出后仍继续接收，直到服务端 is_final。
            state.sentence_end_received = True
            await websocket.send(json.dumps({"text": ""}))
            return

        next_item = await translated_queue.get()
        if next_item is None:
            state.pipeline_finished = True
            await websocket.send(json.dumps({"text": ""}))
            return
        if not isinstance(next_item, dict):
            raise ElevenLabsTTSError("ElevenLabs TTS 收到旧版 Queue 数据")
        current_item = next_item


async def _receive_audio_chunks(
    websocket,
    audio_queue: asyncio.Queue,
    metrics: SentenceMetrics,
) -> None:
    """网络 Producer（生产者）：持续接收 PCM 并放入有界音频队列。"""
    received_final = False
    try:
        async for raw_message in websocket:
            try:
                response = json.loads(raw_message)
            except (json.JSONDecodeError, TypeError) as error:
                raise ElevenLabsTTSError(
                    f"Invalid JSON response: {error}"
                ) from error

            error_detail = response.get("error")
            if error_detail:
                raise ElevenLabsTTSError(f"API error: {error_detail}")

            audio_base64 = response.get("audio")
            if isinstance(audio_base64, str) and audio_base64:
                audio_chunk = _decode_audio(audio_base64)
                if audio_chunk:
                    received_at = time.perf_counter()
                    if metrics.first_audio_chunk_at is None:
                        metrics.first_audio_chunk_at = received_at
                        first_audio_ms = (
                            received_at - metrics.first_segment_ready_at
                        ) * 1000
                        print(f"First Audio Chunk: {first_audio_ms:.0f} ms")
                    metrics.audio_chunk_count += 1
                    metrics.total_audio_bytes += len(audio_chunk)
                    # Queue 满时 await 自然形成 Backpressure（背压），不丢音频。
                    await audio_queue.put(audio_chunk)

            if response.get("is_final") or response.get("isFinal"):
                metrics.session_finished_at = time.perf_counter()
                received_final = True
                break
    except ConnectionClosed as error:
        if not received_final:
            raise ElevenLabsTTSError(
                "WebSocket closed before is_final "
                f"(code={error.code}, reason={error.reason})"
            ) from error
    finally:
        current_task = asyncio.current_task()
        if current_task is None or not current_task.cancelling():
            await audio_queue.put(None)

    if metrics.audio_chunk_count == 0:
        raise ElevenLabsTTSError("ElevenLabs returned empty audio")
    if not received_final:
        raise ElevenLabsTTSError("Session ended without is_final")


async def _playback_worker(
    audio_queue: asyncio.Queue,
    output_stream: sd.OutputStream,
    metrics: SentenceMetrics,
) -> None:
    """单一 Consumer（消费者）顺序写同一个 OutputStream。"""
    while True:
        audio_chunk = await audio_queue.get()
        if audio_chunk is None:
            return
        if len(audio_chunk) % np.dtype(np.int16).itemsize != 0:
            raise ElevenLabsTTSError("PCM Chunk 不是完整的 int16 数据")

        audio_samples = np.frombuffer(
            audio_chunk,
            dtype="<i2",
        ).reshape(-1, ELEVENLABS_CHANNELS)
        try:
            write_started_at = await asyncio.to_thread(
                _write_audio_chunk,
                output_stream,
                audio_samples,
            )
        except Exception as error:
            raise ElevenLabsTTSError(
                "OutputStream.write failed: "
                f"{type(error).__name__}: {error}"
            ) from error

        if metrics.first_playback_at is None:
            # 这是首次调用 write() 的近似值，不冒充扬声器硬件首帧时间。
            metrics.first_playback_at = write_started_at
            first_playback_ms = (
                write_started_at - metrics.first_segment_ready_at
            ) * 1000
            print(f"First Playback: {first_playback_ms:.0f} ms")


async def _run_sentence_session(
    translated_queue: asyncio.Queue,
    first_item: dict,
    state: SentenceState,
    api_key: str,
) -> None:
    """为一句中文创建一个 WebSocket，并并发发送文字、接收和播放音频。"""
    output_stream = None
    tasks = []
    try:
        output_stream = _open_output_stream()
        audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)

        async with websockets.connect(
            _websocket_url(),
            open_timeout=10,
            close_timeout=10,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            await _initialize_connection(websocket, api_key)
            tasks = [
                asyncio.create_task(
                    _send_sentence_segments(
                        websocket,
                        translated_queue,
                        first_item,
                        state,
                    )
                ),
                asyncio.create_task(
                    _receive_audio_chunks(
                        websocket,
                        audio_queue,
                        state.metrics,
                    )
                ),
                asyncio.create_task(
                    _playback_worker(
                        audio_queue,
                        output_stream,
                        state.metrics,
                    )
                ),
            ]
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await _close_output_stream(output_stream)


async def _drain_failed_sentence(
    translated_queue: asyncio.Queue,
    state: SentenceState,
) -> None:
    """失败后收齐当前句文本，避免下一句被误当作当前句。"""
    if state.sentence_end_received:
        return
    while not state.pipeline_finished:
        item = await translated_queue.get()
        if item is None:
            state.pipeline_finished = True
            return
        if not isinstance(item, dict):
            continue
        if item.get("sentence_id") != state.sentence_id:
            raise ElevenLabsTTSError("失败清理时 sentence_id 顺序异常")
        if item.get("event") == "segment":
            _record_segment(state, item)
        elif item.get("event") == "sentence_end":
            state.sentence_end_received = True
            return


async def _log_session(
    performance_logger: PerformanceLogger | None,
    state: SentenceState,
    partial_failure: bool,
    fallback_provider: str | None = None,
) -> None:
    if performance_logger is None:
        return
    metrics = state.metrics
    first_audio_chunk_ms = None
    first_playback_ms = None
    tts_session_total_ms = None
    if metrics.first_audio_chunk_at is not None:
        first_audio_chunk_ms = (
            metrics.first_audio_chunk_at - metrics.first_segment_ready_at
        ) * 1000
    if metrics.first_playback_at is not None:
        first_playback_ms = (
            metrics.first_playback_at - metrics.first_segment_ready_at
        ) * 1000
    if (
        metrics.first_text_sent_at is not None
        and metrics.session_finished_at is not None
    ):
        tts_session_total_ms = (
            metrics.session_finished_at - metrics.first_text_sent_at
        ) * 1000

    await performance_logger.log(
        {
            "event": "tts_session",
            "tts_provider": "elevenlabs",
            "fallback_provider": fallback_provider,
            "sentence_id": state.sentence_id,
            "segment_count": len(state.segments),
            "first_audio_chunk_ms": first_audio_chunk_ms,
            "first_playback_ms": first_playback_ms,
            "tts_session_total_ms": tts_session_total_ms,
            "audio_chunk_count": metrics.audio_chunk_count,
            "total_audio_bytes": metrics.total_audio_bytes,
            "partial_failure": partial_failure,
        }
    )


def _print_session_summary(state: SentenceState) -> None:
    metrics = state.metrics
    if (
        metrics.first_text_sent_at is None
        or metrics.session_finished_at is None
    ):
        return
    total_ms = (
        metrics.session_finished_at - metrics.first_text_sent_at
    ) * 1000
    print(f"Audio Chunk Count: {metrics.audio_chunk_count}")
    print(f"Total Audio Bytes: {metrics.total_audio_bytes}")
    print(f"TTS Session Total: {total_ms:.0f} ms")


async def elevenlabs_tts_worker(
    translated_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """按句消费流式翻译；一句复用一个 ElevenLabs WebSocket。"""
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("缺少 ELEVENLABS_API_KEY")
        return

    print("TTS Provider: ElevenLabs")
    print(f"Voice: {ELEVENLABS_VOICE_ID}")
    print(f"Model: {ELEVENLABS_MODEL_ID}")
    print(f"Output: {ELEVENLABS_OUTPUT_FORMAT}")

    while True:
        first_item = await translated_queue.get()
        if first_item is None:
            print("[ElevenLabs TTS] 播放任务结束")
            return
        if not isinstance(first_item, dict):
            print("[ElevenLabs TTS Error] 收到旧版 Queue 数据，跳过")
            continue
        if first_item.get("event") == "sentence_end":
            continue

        sentence_id = first_item["sentence_id"]
        metrics = SentenceMetrics(
            first_segment_ready_at=first_item["segment_ready_at"]
        )
        state = SentenceState(sentence_id=sentence_id, metrics=metrics)
        _record_segment(state, first_item)
        print("\n[ElevenLabs TTS]")
        print(f"Sentence ID: {sentence_id}")

        try:
            async with asyncio.timeout(TTS_SESSION_TIMEOUT_SECONDS):
                await _run_sentence_session(
                    translated_queue,
                    first_item,
                    state,
                    api_key,
                )
            _print_session_summary(state)
            await _log_session(performance_logger, state, False)
        except Exception as error:
            print(
                "[ElevenLabs TTS Error] "
                f"{_safe_error_message(error, api_key)}"
            )
            try:
                await _drain_failed_sentence(translated_queue, state)
            except Exception as drain_error:
                print(
                    "[ElevenLabs TTS Error] Queue 清理失败："
                    f"{type(drain_error).__name__}"
                )

            if state.metrics.first_playback_at is None and state.full_text:
                print("[ElevenLabs TTS] 当前句回退到 pyttsx3")
                try:
                    await asyncio.to_thread(speak_sync, state.full_text)
                    await _log_session(
                        performance_logger,
                        state,
                        False,
                        fallback_provider="pyttsx3",
                    )
                except Exception as fallback_error:
                    print(
                        "[TTS Fallback Error] "
                        f"{type(fallback_error).__name__}"
                    )
                    await _log_session(
                        performance_logger,
                        state,
                        False,
                        fallback_provider="pyttsx3_failed",
                    )
            else:
                print(
                    "[ElevenLabs TTS] 已播放部分音频，"
                    "为避免重复不再整句回退"
                )
                await _log_session(performance_logger, state, True)

        if state.pipeline_finished:
            print("[ElevenLabs TTS] 播放任务结束")
            return
