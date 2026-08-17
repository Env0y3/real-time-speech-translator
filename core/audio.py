import asyncio
import time
from datetime import datetime

import sounddevice as sd

from config import (
    CHANNELS,
    CHUNK_DURATION_SECONDS,
    PREFERRED_SAMPLE_RATE,
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

