import asyncio
import os

from vosk import SetLogLevel

from asr.sensevoice_asr import sensevoice_asr_worker
from asr.vosk_asr import vosk_asr_worker
from audio import audio_worker, wait_for_stop
from benchmark import benchmark_worker
from config import (
    ASR_PROVIDER,
    ELEVENLABS_AUDIO_QUEUE_MAXSIZE,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_VOICE_ID,
    HOTWORDS_PATH,
    HOTWORD_TEST_SENTENCES,
    MODEL_PATH,
    NORMAL_ASR_PROVIDER,
    NORMAL_HOTWORD_CORRECTION_ENABLED,
    RUNTIME_LATENCY_LOG_PATH,
    SENSEVOICE_MODEL_NAME,
    STREAMING_TRANSLATION_ENABLED,
    STREAMING_TRANSLATION_MAX_CHARS,
    STREAMING_TRANSLATION_MIN_CHARS,
    STREAMING_TRANSLATION_TARGET_CHARS,
    TEST_SENTENCES,
    TTS_PROVIDER,
    VOSK_MODEL_NAME,
)
from elevenlabs_tts import elevenlabs_tts_worker
from hotwords import hotword_correction_worker, load_hotwords
from performance_logger import PerformanceLogger, create_session_id
from translation import streaming_translation_worker, translation_worker
from tts import tts_worker


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


def choose_correction_mode() -> bool | None:
    """选择 Hotword Post Correction（热词后处理纠错）开关。"""
    print("\n请选择 Hotword Post Correction 模式：")
    print("1. Correction OFF")
    print("2. Correction ON")

    while True:
        try:
            choice = input("请输入 1 或 2：").strip()
        except EOFError:
            print("没有收到 Correction 模式，程序结束")
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
    correction_enabled = False
    hotwords = []
    test_sentences = TEST_SENTENCES

    if run_mode == "2":
        selected_benchmark_type = choose_benchmark_type()
        if selected_benchmark_type is None:
            return
        benchmark_type = selected_benchmark_type

        if benchmark_type == "english_hotword":
            selected_correction_mode = choose_correction_mode()
            if selected_correction_mode is None:
                return
            correction_enabled = selected_correction_mode
            hotwords = load_hotwords(HOTWORDS_PATH)
            test_sentences = HOTWORD_TEST_SENTENCES
            print(
                "Hotword Post Correction: "
                f"{'ON' if correction_enabled else 'OFF'}"
            )

            if correction_enabled:
                print(f"[Hotword] 已加载 {len(hotwords)} 个热词")
                print("[Hotword] Post Correction enabled")
                print(
                    "[Hotword] 这是识别后纠错，不是模型级 Hotword Biasing。"
                )

    if (
        run_mode == "2"
        and ASR_PROVIDER not in {"vosk", "sensevoice"}
    ):
        print('ASR_PROVIDER 只支持 "vosk" 或 "sensevoice"')
        return

    if (
        run_mode == "1"
        and NORMAL_ASR_PROVIDER not in {"vosk", "sensevoice"}
    ):
        print('NORMAL_ASR_PROVIDER 只支持 "vosk" 或 "sensevoice"')
        return

    uses_vosk = (
        run_mode == "1" and NORMAL_ASR_PROVIDER == "vosk"
    ) or (
        run_mode == "2" and ASR_PROVIDER == "vosk"
    )
    if uses_vosk and not MODEL_PATH.exists():
        print(f"找不到 Vosk 中文模型：{MODEL_PATH}")
        return

    if run_mode == "1" and not os.environ.get("DEEPSEEK_API_KEY"):
        print("缺少 DEEPSEEK_API_KEY")
        return

    if run_mode == "1" and TTS_PROVIDER not in {"pyttsx3", "elevenlabs"}:
        print('TTS_PROVIDER 只支持 "pyttsx3" 或 "elevenlabs"')
        return

    if (
        run_mode == "1"
        and TTS_PROVIDER == "elevenlabs"
        and not os.environ.get("ELEVENLABS_API_KEY")
    ):
        print("缺少 ELEVENLABS_API_KEY")
        return

    SetLogLevel(-1)

    audio_queue = asyncio.Queue(maxsize=5)
    text_queue = asyncio.Queue(maxsize=5)
    stop_event = asyncio.Event()
    microphone_ready = asyncio.Event()

    if run_mode == "2":
        asr_ready = asyncio.Event()

        if ASR_PROVIDER == "vosk":
            benchmark_asr_worker = vosk_asr_worker(
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
                benchmark_mode=True,
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
                correction_enabled,
                hotwords,
            ),
            wait_for_stop(stop_event, microphone_ready),
        )
    else:
        session_id = create_session_id()
        performance_logger = PerformanceLogger(
            RUNTIME_LATENCY_LOG_PATH,
            session_id,
        )
        await performance_logger.log(
            {
                "event": "session_start",
                "streaming_translation_enabled": (
                    STREAMING_TRANSLATION_ENABLED
                ),
                "tts_provider": TTS_PROVIDER,
                "audio_playback_queue_maxsize": (
                    ELEVENLABS_AUDIO_QUEUE_MAXSIZE
                    if TTS_PROVIDER == "elevenlabs"
                    else None
                ),
                "translation_min_chars": (
                    STREAMING_TRANSLATION_MIN_CHARS
                ),
                "translation_target_chars": (
                    STREAMING_TRANSLATION_TARGET_CHARS
                ),
                "translation_max_chars": (
                    STREAMING_TRANSLATION_MAX_CHARS
                ),
            }
        )
        print(f"Latency Session ID: {session_id}")

        # 流式模式允许短时间缓存多个英文语块，避免TTS播放期间过早触发
        # Backpressure（背压）并暂停DeepSeek流读取；旧模式仍保持原容量。
        translated_queue = asyncio.Queue(
            maxsize=20 if STREAMING_TRANSLATION_ENABLED else 5
        )
        selected_translation_worker = (
            streaming_translation_worker
            if STREAMING_TRANSLATION_ENABLED
            else translation_worker
        )
        selected_tts_worker = (
            elevenlabs_tts_worker
            if TTS_PROVIDER == "elevenlabs"
            else tts_worker
        )
        print(
            "Streaming Translation: "
            f"{'ON' if STREAMING_TRANSLATION_ENABLED else 'OFF'}"
        )
        print(f"TTS Provider: {TTS_PROVIDER}")
        if TTS_PROVIDER == "elevenlabs":
            print(f"Voice: {ELEVENLABS_VOICE_ID}")
            print(f"Model: {ELEVENLABS_MODEL_ID}")
            print(f"Output: {ELEVENLABS_OUTPUT_FORMAT}")

        if NORMAL_ASR_PROVIDER == "vosk":
            print("Normal ASR: Vosk")

            # Vosk Normal Mode 保留原来的完整端到端数据流。
            await asyncio.gather(
                audio_worker(audio_queue, stop_event, microphone_ready),
                vosk_asr_worker(audio_queue, text_queue),
                selected_translation_worker(
                    text_queue,
                    translated_queue,
                    performance_logger,
                ),
                selected_tts_worker(translated_queue, performance_logger),
                wait_for_stop(stop_event, microphone_ready),
            )
        else:
            print("Normal ASR: SenseVoiceSmall")
            print(
                "Hotword Post Correction: "
                f"{'ON' if NORMAL_HOTWORD_CORRECTION_ENABLED else 'OFF'}"
            )
            normal_asr_ready = asyncio.Event()
            raw_text_queue = text_queue

            if NORMAL_HOTWORD_CORRECTION_ENABLED:
                corrected_text_queue = asyncio.Queue(maxsize=5)
                normal_hotwords = load_hotwords(HOTWORDS_PATH)

                await asyncio.gather(
                    audio_worker(audio_queue, stop_event, microphone_ready),
                    sensevoice_asr_worker(
                        audio_queue,
                        raw_text_queue,
                        normal_asr_ready,
                        benchmark_mode=False,
                    ),
                    hotword_correction_worker(
                        raw_text_queue,
                        corrected_text_queue,
                        normal_hotwords,
                    ),
                    selected_translation_worker(
                        corrected_text_queue,
                        translated_queue,
                        performance_logger,
                    ),
                    selected_tts_worker(
                        translated_queue,
                        performance_logger,
                    ),
                    wait_for_stop(stop_event, microphone_ready),
                )
            else:
                await asyncio.gather(
                    audio_worker(audio_queue, stop_event, microphone_ready),
                    sensevoice_asr_worker(
                        audio_queue,
                        raw_text_queue,
                        normal_asr_ready,
                        benchmark_mode=False,
                    ),
                    selected_translation_worker(
                        raw_text_queue,
                        translated_queue,
                        performance_logger,
                    ),
                    selected_tts_worker(
                        translated_queue,
                        performance_logger,
                    ),
                    wait_for_stop(stop_event, microphone_ready),
                )

        await performance_logger.log({"event": "session_end"})

    print("所有任务正常退出")


if __name__ == "__main__":
    asyncio.run(main())
