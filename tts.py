import asyncio
import time

import pyttsx3

from performance_logger import PerformanceLogger


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


async def tts_worker(
    translated_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """消费英文翻译，并通过扬声器逐句播放。"""
    print("TTS 已就绪：pyttsx3 英文语音")
    previous_sentence_id = None
    previous_playback_finished_at = None

    while True:
        queue_item = await translated_queue.get()

        if queue_item is None:
            print("[TTS] 播放任务结束")
            break

        (
            english_text,
            segment_ready_at,
            sentence_id,
            segment_index,
        ) = queue_item
        processing_started_at = time.perf_counter()
        queue_wait_ms = (
            processing_started_at - segment_ready_at
        ) * 1000

        print("\n[TTS]")
        print(f"准备播放：{english_text}", flush=True)

        try:
            first_audio_started_at = await asyncio.to_thread(
                speak_sync,
                english_text,
            )
        except Exception as error:
            print(f"[TTS Error] 播放失败：{type(error).__name__}")
            continue

        playback_finished_at = time.perf_counter()
        total_playback_ms = (
            playback_finished_at - processing_started_at
        ) * 1000
        print("[TTS] 播放完成", flush=True)

        tts_startup_ms = None
        segment_ready_to_audio_ms = None
        segment_gap_ms = None
        if first_audio_started_at is not None:
            # TTS Startup：Worker开始处理到started-word首次触发。
            tts_startup_ms = (
                first_audio_started_at - processing_started_at
            ) * 1000
            # Segment Ready → First Audio，沿用终端里的TTFA近似指标。
            segment_ready_to_audio_ms = (
                first_audio_started_at - segment_ready_at
            ) * 1000
            if (
                segment_index > 1
                and previous_sentence_id == sentence_id
                and previous_playback_finished_at is not None
            ):
                segment_gap_ms = (
                    first_audio_started_at
                    - previous_playback_finished_at
                ) * 1000
            print(
                "TTFA: "
                f"{segment_ready_to_audio_ms:.0f} ms"
                "（TTS 首音近似值）"
            )
        else:
            print("TTFA: 无法获取 started-word 事件")
        print(f"TTS Total Playback: {total_playback_ms:.0f} ms")

        if performance_logger is not None:
            await performance_logger.log(
                {
                    "event": "tts_segment",
                    "sentence_id": sentence_id,
                    "segment_index": segment_index,
                    "text": english_text,
                    "queue_wait_ms": queue_wait_ms,
                    "tts_startup_ms": tts_startup_ms,
                    "segment_ready_to_audio_ms": (
                        segment_ready_to_audio_ms
                    ),
                    "playback_total_ms": total_playback_ms,
                    "segment_gap_ms": segment_gap_ms,
                }
            )

        previous_sentence_id = sentence_id
        previous_playback_finished_at = playback_finished_at
