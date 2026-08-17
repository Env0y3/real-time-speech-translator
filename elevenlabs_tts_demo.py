import asyncio
import base64
import binascii
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus


VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_flash_v2_5"
OUTPUT_FORMAT = "pcm_24000"
SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "int16"
AUDIO_QUEUE_MAXSIZE = 20
TEXT_CHUNK_DELAY_SECONDS = 0.3
SESSION_TIMEOUT_SECONDS = 60

WEBSOCKET_URL = (
    "wss://api.elevenlabs.io/v1/text-to-speech/"
    f"{VOICE_ID}/stream-input"
    f"?model_id={MODEL_ID}&output_format={OUTPUT_FORMAT}"
)

TEXT_CHUNKS = [
    "Hello, this is a real-time streaming speech test. ",
    (
        "I am building a Chinese to English speech translation "
        "system. "
    ),
    (
        "The goal is to start speaking before the complete "
        "translation has finished."
    ),
]


class ElevenLabsProtocolError(Exception):
    """服务端返回无法继续处理的控制消息或 JSON 数据。"""


class EmptyAudioResponseError(Exception):
    """WebSocket 正常结束，但没有收到有效 Audio Chunk（音频块）。"""


class AudioPlaybackError(Exception):
    """本地扬声器初始化或连续播放失败。"""


@dataclass
class DemoMetrics:
    """保存一次 ElevenLabs Streaming TTS 会话的时间点和计数。"""

    session_started_at: float
    first_audio_chunk_at: float | None = None
    first_playback_at: float | None = None
    session_finished_at: float | None = None
    audio_chunk_count: int = 0
    total_audio_bytes: int = 0


def safe_error_message(error: Exception, api_key: str) -> str:
    """输出异常类型和消息，同时确保 API Key 不会出现在终端。"""
    message = str(error).replace(api_key, "[REDACTED]").strip()
    if not message:
        message = "No additional message"
    return f"{type(error).__name__}: {message}"


async def initialize_connection(websocket, api_key: str) -> None:
    """发送官方要求的连接初始化消息，不混入第一段真实文本。"""
    initialization_message = {
        "text": " ",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "speed": 1.0,
        },
        "xi_api_key": api_key,
    }
    await websocket.send(json.dumps(initialization_message))


async def send_text_chunks(websocket) -> None:
    """在同一个 WebSocket Session 中模拟 DeepSeek 依次产生三段文本。"""
    for chunk_index, text_chunk in enumerate(TEXT_CHUNKS, start=1):
        print(f"\n[Text Chunk {chunk_index}]")
        print(text_chunk.strip())

        text_message = {"text": text_chunk}
        if chunk_index == 2:
            # 前两段已经形成完整句子，按官方建议 Flush 以尽快触发生成。
            text_message["flush"] = True
        await websocket.send(json.dumps(text_message))

        if chunk_index < len(TEXT_CHUNKS):
            await asyncio.sleep(TEXT_CHUNK_DELAY_SECONDS)

    # 官方 Close Connection 信号：让服务端合成剩余文本并返回最终音频。
    await websocket.send(json.dumps({"text": ""}))


def decode_audio(audio_base64: str) -> bytes:
    """严格解码 Base64 Audio；损坏数据会产生清晰异常。"""
    try:
        return base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ElevenLabsProtocolError(
            f"Base64 decode error: {error}"
        ) from error


async def receive_audio_chunks(
    websocket,
    audio_queue: asyncio.Queue,
    metrics: DemoMetrics,
) -> None:
    """WebSocket Producer（生产者）：解析 JSON 并把 PCM 放入 Queue。"""
    received_final = False
    try:
        async for raw_message in websocket:
            try:
                response = json.loads(raw_message)
            except (json.JSONDecodeError, TypeError) as error:
                raise ElevenLabsProtocolError(
                    f"Invalid JSON response: {error}"
                ) from error

            audio_base64 = response.get("audio")
            if isinstance(audio_base64, str) and audio_base64:
                audio_chunk = decode_audio(audio_base64)
                if audio_chunk:
                    received_at = time.perf_counter()
                    if metrics.first_audio_chunk_at is None:
                        metrics.first_audio_chunk_at = received_at
                        first_audio_ms = (
                            received_at - metrics.session_started_at
                        ) * 1000
                        print("\n[Audio]")
                        print("First Audio Chunk received")
                        print(
                            "First Audio Chunk Latency: "
                            f"{first_audio_ms:.0f} ms"
                        )

                    metrics.audio_chunk_count += 1
                    metrics.total_audio_bytes += len(audio_chunk)
                    # 有界Queue满时自然产生Backpressure（背压），不丢音频。
                    await audio_queue.put(audio_chunk)

            # 官方示例曾使用isFinal，当前API参考使用is_final；两者都兼容。
            if response.get("is_final") or response.get("isFinal"):
                metrics.session_finished_at = time.perf_counter()
                received_final = True
                break

            error_detail = response.get("error")
            if error_detail:
                raise ElevenLabsProtocolError(
                    f"API Error: {error_detail}"
                )
    except ConnectionClosed as error:
        if not received_final:
            raise ElevenLabsProtocolError(
                "Connection closed before is_final "
                f"(code={error.code}, reason={error.reason})"
            ) from error
    finally:
        # Playback失败时Receiver会被取消；此时不要阻塞在已满Queue中放Sentinel。
        current_task = asyncio.current_task()
        if current_task is None or not current_task.cancelling():
            await audio_queue.put(None)

    if metrics.audio_chunk_count == 0:
        raise EmptyAudioResponseError
    if not received_final:
        raise ElevenLabsProtocolError("Session ended without is_final")


async def playback_worker(
    audio_queue: asyncio.Queue,
    output_stream: sd.OutputStream,
    metrics: DemoMetrics,
) -> None:
    """Audio Consumer（消费者）：在同一个 OutputStream 中顺序播放。"""
    while True:
        audio_chunk = await audio_queue.get()
        if audio_chunk is None:
            break

        if len(audio_chunk) % np.dtype(np.int16).itemsize != 0:
            raise AudioPlaybackError(
                "PCM Chunk 字节数不是 int16 的整数倍"
            )

        # pcm_24000 为 signed 16-bit little-endian mono PCM。
        audio_samples = np.frombuffer(
            audio_chunk,
            dtype="<i2",
        ).reshape(-1, CHANNELS)

        try:
            write_started_at = await asyncio.to_thread(
                write_audio_chunk,
                output_stream,
                audio_samples,
            )
        except Exception as error:
            raise AudioPlaybackError(
                f"OutputStream.write failed: {type(error).__name__}: {error}"
            ) from error

        if metrics.first_playback_at is None:
            # 记录首次进入sounddevice.write的时间，不冒充硬件首帧时间。
            metrics.first_playback_at = write_started_at
            first_playback_ms = (
                metrics.first_playback_at - metrics.session_started_at
            ) * 1000
            print(
                "First Playback Latency: "
                f"{first_playback_ms:.0f} ms"
            )


def write_audio_chunk(
    output_stream: sd.OutputStream,
    audio_samples: np.ndarray,
) -> float:
    """在线程中记录实际调用 write() 的时间，并按顺序写入 PCM。"""
    write_started_at = time.perf_counter()
    output_stream.write(audio_samples)
    return write_started_at


def open_output_stream() -> sd.OutputStream:
    """只创建一次持久 OutputStream，整个 Session 的 Chunk 共用。"""
    try:
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
        )
        output_stream.start()
        return output_stream
    except Exception as error:
        raise AudioPlaybackError(
            f"OutputStream initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error


async def close_output_stream(
    output_stream: sd.OutputStream | None,
) -> None:
    """停止并关闭本次 Demo 唯一的扬声器输出流。"""
    if output_stream is None:
        return
    try:
        await asyncio.to_thread(output_stream.stop)
    finally:
        await asyncio.to_thread(output_stream.close)


def print_summary(metrics: DemoMetrics) -> bool:
    """打印延迟、Audio Chunk 统计和真正 Streaming 判断。"""
    if (
        metrics.first_audio_chunk_at is None
        or metrics.first_playback_at is None
        or metrics.session_finished_at is None
    ):
        return False

    first_audio_ms = (
        metrics.first_audio_chunk_at - metrics.session_started_at
    ) * 1000
    first_playback_ms = (
        metrics.first_playback_at - metrics.session_started_at
    ) * 1000
    total_session_ms = (
        metrics.session_finished_at - metrics.session_started_at
    ) * 1000
    streamed_before_finish = (
        metrics.first_playback_at < metrics.session_finished_at
    )

    print("\n========== ElevenLabs Demo Summary ==========")
    print(f"First Audio Chunk Latency: {first_audio_ms:.0f} ms")
    print(f"First Playback Latency: {first_playback_ms:.0f} ms")
    print(f"Audio Chunk Count: {metrics.audio_chunk_count}")
    print(f"Total Audio Bytes: {metrics.total_audio_bytes}")
    print(f"TTS Session Total: {total_session_ms:.0f} ms")
    print(
        "Playback started before session finished: "
        f"{'YES' if streamed_before_finish else 'NO'}"
    )
    print("=============================================")
    return streamed_before_finish


async def run_demo(api_key: str) -> bool:
    """连接ElevenLabs，并发执行文本发送、网络接收和连续播放。"""
    output_stream = None
    sender_task = None
    receiver_task = None
    playback_task = None

    try:
        # 先验证扬声器，避免WebSocket已产生计费字符后才发现无法播放。
        output_stream = open_output_stream()
        audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        metrics = DemoMetrics(session_started_at=time.perf_counter())
        print("\nTTS Session Started")

        async with websockets.connect(
            WEBSOCKET_URL,
            open_timeout=10,
            close_timeout=10,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            print("WebSocket connected")
            await initialize_connection(websocket, api_key)

            sender_task = asyncio.create_task(
                send_text_chunks(websocket)
            )
            receiver_task = asyncio.create_task(
                receive_audio_chunks(websocket, audio_queue, metrics)
            )
            playback_task = asyncio.create_task(
                playback_worker(audio_queue, output_stream, metrics)
            )
            await asyncio.gather(
                sender_task,
                receiver_task,
                playback_task,
            )

        if not print_summary(metrics):
            print(
                "[ElevenLabs TTS Error] First Playback 没有早于 "
                "Session 完成"
            )
            return False

        print("\nDemo finished successfully")
        return True
    finally:
        for task in (sender_task, receiver_task, playback_task):
            if task is not None and not task.done():
                task.cancel()
        pending_tasks = [
            task
            for task in (sender_task, receiver_task, playback_task)
            if task is not None
        ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        await close_output_stream(output_stream)


async def main() -> int:
    print("========== ElevenLabs Streaming TTS Demo ==========")
    print(f"Voice ID: {VOICE_ID}")
    print(f"Model: {MODEL_ID}")
    print("Streaming: ON")
    print(
        f"Audio: {OUTPUT_FORMAT} | {SAMPLE_RATE} Hz | "
        f"{CHANNELS} channel | {DTYPE}"
    )

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY not found")
        return 1

    try:
        async with asyncio.timeout(SESSION_TIMEOUT_SECONDS):
            return 0 if await run_demo(api_key) else 1
    except InvalidStatus as error:
        print(
            "[ElevenLabs TTS Error] "
            f"{safe_error_message(error, api_key)}"
        )
    except TimeoutError as error:
        print(
            "[ElevenLabs TTS Error] "
            f"Timeout: {safe_error_message(error, api_key)}"
        )
    except (
        ElevenLabsProtocolError,
        EmptyAudioResponseError,
        AudioPlaybackError,
        ConnectionError,
    ) as error:
        print(
            "[ElevenLabs TTS Error] "
            f"{safe_error_message(error, api_key)}"
        )
    except Exception as error:
        print(
            "[ElevenLabs TTS Error] "
            f"{safe_error_message(error, api_key)}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
