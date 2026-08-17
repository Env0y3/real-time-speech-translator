import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from fishaudio import AsyncFishAudio, TTSConfig
from fishaudio.exceptions import (
    AuthenticationError,
    FishAudioError,
    WebSocketError,
)
from fishaudio.types import FlushEvent


MODEL = "s2.1-pro-free"
AUDIO_FORMAT = "pcm"
SAMPLE_RATE = 44_100
CHANNELS = 1
DTYPE = "int16"
AUDIO_QUEUE_MAXSIZE = 20
TEXT_CHUNK_DELAY_SECONDS = 0.35

TEXT_CHUNKS = [
    "Hello, this is a streaming text to speech test. ",
    (
        "I am building a real-time Chinese to English speech "
        "translation system. "
    ),
    (
        "The goal is to start speaking before the complete "
        "translation has finished."
    ),
]


class EmptyAudioResponseError(Exception):
    """Fish Audio Session 正常结束，但没有返回任何有效音频。"""


class AudioPlaybackError(Exception):
    """扬声器初始化或连续播放失败。"""


@dataclass
class DemoMetrics:
    """保存一次 Streaming TTS（流式文字转语音）会话的时间点。"""

    session_started_at: float
    first_audio_chunk_at: float | None = None
    first_playback_at: float | None = None
    session_finished_at: float | None = None
    audio_chunk_count: int = 0
    total_audio_bytes: int = 0


async def text_chunk_stream() -> AsyncIterator[str | FlushEvent]:
    """在同一个 TTS Session（会话）中依次发送三段英文文本。"""
    for chunk_index, text_chunk in enumerate(TEXT_CHUNKS, start=1):
        print(f"\n[Text Chunk {chunk_index}]")
        print(text_chunk.strip())
        yield text_chunk

        if chunk_index == 2:
            # Flush（刷新）强制服务端尽快合成当前已经缓存的文本。
            print("\n[Flush] 提交前两个 Text Chunk")
            yield FlushEvent()

        if chunk_index < len(TEXT_CHUNKS):
            # 模拟未来 DeepSeek Streaming Translation 逐段产生文本。
            await asyncio.sleep(TEXT_CHUNK_DELAY_SECONDS)


async def receive_audio_chunks(
    client: AsyncFishAudio,
    audio_queue: asyncio.Queue,
    metrics: DemoMetrics,
) -> None:
    """Network Producer（网络生产者）：接收音频并放入有界队列。"""
    config = TTSConfig(
        format=AUDIO_FORMAT,
        sample_rate=SAMPLE_RATE,
        chunk_length=200,
        latency="balanced",
    )

    try:
        audio_stream = client.tts.stream_websocket(
    text_chunk_stream(),
    format="pcm",
    latency="balanced",
    model="s2-pro",
)
        async for audio_chunk in audio_stream:
            if not audio_chunk:
                continue

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

            # Queue 满时 await 会自然形成 Backpressure（背压），不丢音频。
            await audio_queue.put(audio_chunk)
    finally:
        metrics.session_finished_at = time.perf_counter()
        await audio_queue.put(None)

    if metrics.audio_chunk_count == 0:
        raise EmptyAudioResponseError


async def playback_worker(
    audio_queue: asyncio.Queue,
    output_stream: sd.OutputStream,
    metrics: DemoMetrics,
) -> None:
    """Audio Consumer（音频消费者）：用同一个输出流连续播放 PCM。"""
    while True:
        audio_chunk = await audio_queue.get()
        if audio_chunk is None:
            break

        if len(audio_chunk) % np.dtype(np.int16).itemsize != 0:
            raise AudioPlaybackError("PCM Chunk 字节数不是 int16 的整数倍")

        # Fish Audio PCM 是 16-bit mono；转换为 sounddevice 需要的帧数组。
        audio_samples = np.frombuffer(
            audio_chunk,
            dtype=np.int16,
        ).reshape(-1, CHANNELS)

        # OutputStream.write() 会等待播放缓冲区；放到线程中避免阻塞网络接收。
        try:
            write_started_at = await asyncio.to_thread(
                write_audio_chunk,
                output_stream,
                audio_samples,
            )
        except Exception as error:
            raise AudioPlaybackError(type(error).__name__) from error

        if metrics.first_playback_at is None:
            # 这是首次进入sounddevice.write的时间，不是硬件真正输出首帧。
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
    """在线程中记录实际调用 sounddevice.write 的时间并写入 PCM。"""
    write_started_at = time.perf_counter()
    output_stream.write(audio_samples)
    return write_started_at


def open_output_stream() -> sd.OutputStream:
    """只初始化一次 OutputStream（输出音频流），供所有 Chunk 共用。"""
    try:
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
        )
        output_stream.start()
        return output_stream
    except Exception as error:
        raise AudioPlaybackError(type(error).__name__) from error


async def close_output_stream(
    output_stream: sd.OutputStream | None,
) -> None:
    """正常停止并关闭扬声器输出流。"""
    if output_stream is None:
        return
    try:
        await asyncio.to_thread(output_stream.stop)
    finally:
        await asyncio.to_thread(output_stream.close)


def print_summary(metrics: DemoMetrics) -> bool:
    """打印本次真实 Streaming Session 的延迟与音频统计。"""
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

    print("\n========== Fish Audio Demo Summary ==========")
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


async def run_demo() -> bool:
    """建立一个 Fish Audio 会话，并发执行网络接收和连续播放。"""
    output_stream = None
    client = None
    receiver_task = None
    playback_task = None

    try:
        # 先确认扬声器可用，避免音频已计费后才发现无法播放。
        output_stream = open_output_stream()
        client = AsyncFishAudio()
        audio_queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)
        metrics = DemoMetrics(session_started_at=time.perf_counter())

        print("\nTTS Session Started")
        receiver_task = asyncio.create_task(
            receive_audio_chunks(client, audio_queue, metrics)
        )
        playback_task = asyncio.create_task(
            playback_worker(audio_queue, output_stream, metrics)
        )
        await asyncio.gather(receiver_task, playback_task)

        if not print_summary(metrics):
            print(
                "[Fish TTS Error] First Playback 没有早于 Session 完成"
            )
            return False

        print("\nDemo finished successfully")
        return True
    finally:
        for task in (receiver_task, playback_task):
            if task is not None and not task.done():
                task.cancel()
        pending_tasks = [
            task
            for task in (receiver_task, playback_task)
            if task is not None
        ]
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if client is not None:
            await client.close()
        await close_output_stream(output_stream)


async def main() -> int:
    print("========== Fish Audio Streaming TTS Demo ==========")
    print(f"Model: {MODEL}")
    print("Streaming: ON")
    print(
        f"Audio: PCM {SAMPLE_RATE} Hz | {CHANNELS} channel | {DTYPE}"
    )

    if not os.environ.get("FISH_API_KEY"):
        print("FISH_API_KEY not found")
        return 1

    try:
        return 0 if await run_demo() else 1
    except AuthenticationError:
        print("[Fish TTS Error] Authentication Error")
    except WebSocketError as error:
        print(f"[Fish TTS Error] WebSocket disconnected: {error}")
    except EmptyAudioResponseError:
        print("[Fish TTS Error] Empty audio response")
    except AudioPlaybackError as error:
        print(f"[Fish TTS Error] Audio playback error: {error}")
    except FishAudioError as error:
        print(f"[Fish TTS Error] API Error: {type(error).__name__}")
    except (ConnectionError, TimeoutError) as error:
        print(f"[Fish TTS Error] Connection Error: {type(error).__name__}")
    except Exception as error:
        print(
            f"[Fish TTS Error] Unexpected error: "
            f"{type(error).__name__}: {error}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
