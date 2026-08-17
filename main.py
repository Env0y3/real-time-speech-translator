import asyncio

from vosk import SetLogLevel

from asr.sensevoice_asr import sensevoice_asr_worker
from asr.vosk_asr import vosk_asr_worker
from config import (
    ASR_PROVIDER,
    HOTWORDS_PATH,
    HOTWORD_TEST_SENTENCES,
    MODEL_PATH,
    SENSEVOICE_MODEL_NAME,
    TEST_SENTENCES,
    VOSK_MODEL_NAME,
)
from core.audio import audio_worker, wait_for_stop
from core.audio_devices import AudioRoutingError
from core.hotwords import load_hotwords
from core.pipeline import run_normal_pipeline
from scripts.benchmark import benchmark_worker


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


async def run_benchmark_mode() -> None:
    benchmark_type = choose_benchmark_type()
    if benchmark_type is None:
        return

    correction_enabled = False
    hotwords = []
    test_sentences = TEST_SENTENCES
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
            print("[Hotword] 这是识别后纠错，不是模型级 Hotword Biasing。")

    if ASR_PROVIDER not in {"vosk", "sensevoice"}:
        print('ASR_PROVIDER 只支持 "vosk" 或 "sensevoice"')
        return
    if ASR_PROVIDER == "vosk" and not MODEL_PATH.exists():
        print(f"找不到 Vosk 中文模型：{MODEL_PATH}")
        return

    SetLogLevel(-1)
    audio_queue = asyncio.Queue(maxsize=5)
    text_queue = asyncio.Queue(maxsize=5)
    stop_event = asyncio.Event()
    microphone_ready = asyncio.Event()
    asr_ready = asyncio.Event()

    if ASR_PROVIDER == "vosk":
        asr_worker = vosk_asr_worker(
            audio_queue,
            text_queue,
            benchmark_mode=True,
            asr_ready=asr_ready,
        )
        model_name = VOSK_MODEL_NAME
    else:
        asr_worker = sensevoice_asr_worker(
            audio_queue,
            text_queue,
            asr_ready,
            benchmark_mode=True,
        )
        model_name = SENSEVOICE_MODEL_NAME

    await asyncio.gather(
        audio_worker(audio_queue, stop_event, microphone_ready),
        asr_worker,
        benchmark_worker(
            text_queue,
            stop_event,
            microphone_ready,
            asr_ready,
            ASR_PROVIDER,
            model_name,
            test_sentences,
            benchmark_type,
            correction_enabled,
            hotwords,
        ),
        wait_for_stop(stop_event, microphone_ready),
    )
    print("所有任务正常退出")


async def main() -> None:
    run_mode = choose_run_mode()
    if run_mode is None:
        return
    if run_mode == "1":
        await run_normal_pipeline(interactive_stop=True)
    else:
        await run_benchmark_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (AudioRoutingError, ValueError) as error:
        print("[Startup Error]")
        print(error)
