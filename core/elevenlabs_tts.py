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
    ELEVENLABS_AUDIO_QUEUE_MAXSIZE,
    ELEVENLABS_CHANNELS,
    ELEVENLABS_DTYPE,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_OUTPUT_FORMAT,
    ELEVENLABS_SAMPLE_RATE,
    ELEVENLABS_VOICE_ID,
    LOCAL_MONITOR_DEVICE,
    LOCAL_MONITOR_ENABLED,
    TRANSLATION_OUTPUT_DEVICE,
)
from core.audio_devices import (
    AudioRoutingError,
    AudioRoutingPlan,
    build_audio_routing_plan,
    format_device,
)
from core.performance_logger import PerformanceLogger
from core.tts import speak_sync, tts_worker


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
    playback_finished_at: float | None = None
    last_audio_write_finished_at: float | None = None
    tts_queue_wait_ms: float | None = None
    cross_sentence_gap_ms: float | None = None
    audio_chunk_count: int = 0
    total_audio_bytes: int = 0
    playback_error: bool = False


@dataclass
class SentenceState:
    """保存一个中文句子对应的所有 Translation Segment（翻译语块）。"""

    sentence_id: int
    metrics: SentenceMetrics
    segments: dict[int, str] = field(default_factory=dict)
    sentence_end_received: bool = False
    pipeline_finished: bool = False
    partial_failure: bool = False
    fallback_provider: str | None = None
    trace: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return " ".join(
            self.segments[index].strip()
            for index in sorted(self.segments)
            if self.segments[index].strip()
        )


@dataclass
class AudioOutputStreams:
    """主输出始终存在；Monitor 可在启动或运行失败后独立关闭。"""

    translation: sd.OutputStream
    monitor: sd.OutputStream | None = None
    monitor_label: str | None = None


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


def _open_output_stream(
    device: int,
    role: str,
) -> sd.OutputStream:
    """按已验证的设备索引创建 Continuous OutputStream。"""
    output_stream = None
    try:
        output_stream = sd.OutputStream(
            device=device,
            samplerate=ELEVENLABS_SAMPLE_RATE,
            channels=ELEVENLABS_CHANNELS,
            dtype=ELEVENLABS_DTYPE,
        )
        output_stream.start()
        return output_stream
    except Exception as error:
        if output_stream is not None:
            try:
                output_stream.close()
            except Exception:
                pass
        raise ElevenLabsTTSError(
            f"{role} OutputStream initialization failed: "
            f"{type(error).__name__}: {error}"
        ) from error


def _open_audio_output_streams(
    routing_plan: AudioRoutingPlan,
) -> AudioOutputStreams:
    """主输出打开失败向上抛出；Monitor 失败只产生 Warning。"""
    translation_stream = _open_output_stream(
        routing_plan.translation_device.index,
        "Translation output",
    )
    monitor_stream = None
    monitor_label = None
    if routing_plan.monitor_enabled and routing_plan.monitor_device:
        monitor_label = format_device(routing_plan.monitor_device)
        try:
            monitor_stream = _open_output_stream(
                routing_plan.monitor_device.index,
                "Local monitor",
            )
        except Exception as error:
            print(
                "[Audio Routing Warning] Local monitor could not be "
                f"opened and was disabled: {type(error).__name__}: {error}"
            )
    return AudioOutputStreams(
        translation=translation_stream,
        monitor=monitor_stream,
        monitor_label=monitor_label,
    )


async def _close_output_stream(
    output_stream: sd.OutputStream | None,
) -> None:
    if output_stream is None:
        return
    try:
        await asyncio.to_thread(output_stream.stop)
    finally:
        await asyncio.to_thread(output_stream.close)


async def _close_audio_output_streams(
    output_streams: AudioOutputStreams,
) -> None:
    if output_streams.monitor is not None:
        try:
            await _close_output_stream(output_streams.monitor)
        except Exception as error:
            print(
                "[Audio Routing Warning] Local monitor close failed: "
                f"{type(error).__name__}"
            )
        output_streams.monitor = None
    await _close_output_stream(output_streams.translation)


def _write_audio_chunk(
    output_stream: sd.OutputStream,
    audio_samples: np.ndarray,
) -> tuple[float, float]:
    write_started_at = time.perf_counter()
    output_stream.write(audio_samples)
    return write_started_at, time.perf_counter()


async def _write_routed_audio_chunk(
    output_streams: AudioOutputStreams,
    audio_samples: np.ndarray,
) -> tuple[float, float]:
    """同一 PCM Chunk 并行写入两个不同流，主输出结果作为 Trace 基准。"""
    translation_write = asyncio.to_thread(
        _write_audio_chunk,
        output_streams.translation,
        audio_samples,
    )
    if output_streams.monitor is None:
        return await translation_write

    monitor_stream = output_streams.monitor
    translation_result, monitor_result = await asyncio.gather(
        translation_write,
        asyncio.to_thread(
            _write_audio_chunk,
            monitor_stream,
            audio_samples,
        ),
        return_exceptions=True,
    )
    if isinstance(monitor_result, Exception):
        print(
            "[Audio Routing Warning] Local monitor write failed and was "
            f"disabled ({output_streams.monitor_label}): "
            f"{type(monitor_result).__name__}: {monitor_result}"
        )
        try:
            await _close_output_stream(monitor_stream)
        except Exception:
            pass
        output_streams.monitor = None
    if isinstance(translation_result, Exception):
        raise translation_result
    return translation_result


def _record_segment(state: SentenceState, item: dict) -> None:
    state.segments.setdefault(item["segment_index"], item["text"])
    state.trace.update(item.get("trace", {}))


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
        state.trace["tts_first_text_sent_at"] = sent_at

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
            state.trace.update(current_item.get("trace", {}))
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
    audio_playback_queue: asyncio.Queue,
    state: SentenceState,
    runtime_session_id: str | None,
) -> None:
    """网络 Producer（生产者）：接收 PCM 后立即交给全局播放队列。"""
    metrics = state.metrics
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
                    is_first_chunk = metrics.first_audio_chunk_at is None
                    if is_first_chunk:
                        metrics.first_audio_chunk_at = received_at
                        state.trace["tts_first_audio_received_at"] = (
                            received_at
                        )
                        network_first_audio_ms = (
                            received_at - metrics.first_segment_ready_at
                        ) * 1000
                        print(
                            "Network First Audio: "
                            f"{network_first_audio_ms:.0f} ms"
                        )
                    metrics.audio_chunk_count += 1
                    metrics.total_audio_bytes += len(audio_chunk)
                    # Queue 满时自然产生 Backpressure（背压），不丢 Chunk。
                    await audio_playback_queue.put(
                        {
                            "type": "audio",
                            "session_id": runtime_session_id,
                            "trace_id": state.trace.get("trace_id"),
                            "sentence_id": state.sentence_id,
                            "audio_chunk": audio_chunk,
                            "audio_received_at": received_at,
                            "is_first_chunk": is_first_chunk,
                        }
                    )

            if response.get("is_final") or response.get("isFinal"):
                metrics.session_finished_at = time.perf_counter()
                state.trace["tts_session_finished_at"] = (
                    metrics.session_finished_at
                )
                received_final = True
                break
    except ConnectionClosed as error:
        if not received_final:
            raise ElevenLabsTTSError(
                "WebSocket closed before is_final "
                f"(code={error.code}, reason={error.reason})"
            ) from error
    if metrics.audio_chunk_count == 0:
        raise ElevenLabsTTSError("ElevenLabs returned empty audio")
    if not received_final:
        raise ElevenLabsTTSError("Session ended without is_final")

async def _global_playback_worker(
    audio_playback_queue: asyncio.Queue,
    output_streams: AudioOutputStreams,
    sentence_states: dict[int, SentenceState],
    performance_logger: PerformanceLogger | None,
) -> None:
    """唯一播放 Consumer；每个 OutputStream 同一时间只有一次 write。"""
    previous_sentence_finished_at = None
    session_log_tasks = []

    while True:
        message = await audio_playback_queue.get()
        if message is None:
            if session_log_tasks:
                log_results = await asyncio.gather(
                    *session_log_tasks,
                    return_exceptions=True,
                )
                for log_result in log_results:
                    if isinstance(log_result, Exception):
                        print(
                            "[Trace Log Warning] "
                            f"{type(log_result).__name__}"
                        )
            return
        sentence_id = message["sentence_id"]
        state = sentence_states.get(sentence_id)
        if state is None:
            print(
                "[ElevenLabs Playback Warning] "
                f"找不到 Sentence ID {sentence_id}"
            )
            continue
        metrics = state.metrics
        message_trace_id = message.get("trace_id")
        state_trace_id = state.trace.get("trace_id")
        if (
            message_trace_id is not None
            and state_trace_id is not None
            and message_trace_id != state_trace_id
        ):
            print(
                "[Trace Warning] Playback trace_id 不匹配："
                f"{message_trace_id} != {state_trace_id}"
            )

        if message["type"] == "audio":
            if metrics.playback_error:
                continue
            audio_chunk = message["audio_chunk"]
            if len(audio_chunk) % np.dtype(np.int16).itemsize != 0:
                metrics.playback_error = True
                print("[ElevenLabs Playback Error] PCM Chunk 不完整")
                continue

            audio_samples = np.frombuffer(
                audio_chunk,
                dtype="<i2",
            ).reshape(-1, ELEVENLABS_CHANNELS)
            try:
                write_started_at, write_finished_at = (
                    await _write_routed_audio_chunk(
                        output_streams,
                        audio_samples,
                    )
                )
            except Exception as error:
                metrics.playback_error = True
                print(
                    "[ElevenLabs Playback Error] "
                    f"{type(error).__name__}: {error}"
                )
                continue

            metrics.last_audio_write_finished_at = write_finished_at
            if metrics.first_playback_at is None:
                # write() 开始时间是软件层近似值，不冒充硬件首帧时间。
                metrics.first_playback_at = write_started_at
                state.trace["first_playback_at"] = write_started_at
                metrics.tts_queue_wait_ms = (
                    write_started_at - message["audio_received_at"]
                ) * 1000
                if previous_sentence_finished_at is not None:
                    metrics.cross_sentence_gap_ms = (
                        write_started_at - previous_sentence_finished_at
                    ) * 1000
                first_playback_ms = (
                    write_started_at - metrics.first_segment_ready_at
                ) * 1000
                print(f"First Playback: {first_playback_ms:.0f} ms")
                print(
                    "TTS Queue Wait: "
                    f"{metrics.tts_queue_wait_ms:.0f} ms"
                )
                if performance_logger is not None:
                    performance_logger.emit_event(
                        {
                            "type": "tts_started",
                            "status": "Speaking",
                            "sentence_id": sentence_id,
                            "trace_id": state.trace.get("trace_id"),
                        }
                    )

        elif message["type"] == "pyttsx3_fallback":
            state.fallback_provider = "pyttsx3"
            fallback_started_at = time.perf_counter()
            if previous_sentence_finished_at is not None:
                metrics.cross_sentence_gap_ms = (
                    fallback_started_at - previous_sentence_finished_at
                ) * 1000
            try:
                first_audio_started_at = await asyncio.to_thread(
                    speak_sync,
                    message["text"],
                )
                metrics.first_playback_at = (
                    first_audio_started_at or fallback_started_at
                )
                state.trace["first_playback_at"] = (
                    metrics.first_playback_at
                )
            except Exception as error:
                state.fallback_provider = "pyttsx3_failed"
                metrics.playback_error = True
                state.partial_failure = True
                print(
                    "[TTS Fallback Error] "
                    f"{type(error).__name__}"
                )
            metrics.playback_finished_at = time.perf_counter()
            state.trace["playback_finished_at"] = (
                metrics.playback_finished_at
            )
            previous_sentence_finished_at = metrics.playback_finished_at
            # 日志放到后台，不能阻塞下一句从全局播放队列开始写入。
            session_log_tasks.append(
                asyncio.create_task(
                    _print_and_log_session(performance_logger, state)
                )
            )
            sentence_states.pop(sentence_id, None)

        elif message["type"] == "sentence_end":
            metrics.playback_finished_at = (
                metrics.last_audio_write_finished_at or time.perf_counter()
            )
            state.trace["playback_finished_at"] = (
                metrics.playback_finished_at
            )
            if metrics.first_playback_at is not None:
                previous_sentence_finished_at = metrics.playback_finished_at
            if metrics.playback_error:
                state.partial_failure = True
            session_log_tasks.append(
                asyncio.create_task(
                    _print_and_log_session(performance_logger, state)
                )
            )
            sentence_states.pop(sentence_id, None)


async def _run_sentence_session(
    translated_queue: asyncio.Queue,
    audio_playback_queue: asyncio.Queue,
    first_item: dict,
    state: SentenceState,
    api_key: str,
    runtime_session_id: str | None,
) -> None:
    """一句使用一个 WebSocket；这里只生成音频，不等待扬声器播放。"""
    tasks = []
    try:
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
                        audio_playback_queue,
                        state,
                        runtime_session_id,
                    )
                ),
            ]
            await asyncio.gather(*tasks)
            # 两个任务都成功后再放 Sentence End Marker（句尾标记）。
            # None 仍只用于整个 Playback Worker 的 shutdown。
            await audio_playback_queue.put(
                {
                    "type": "sentence_end",
                    "session_id": runtime_session_id,
                    "trace_id": state.trace.get("trace_id"),
                    "sentence_id": state.sentence_id,
                }
            )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


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
            state.trace.update(item.get("trace", {}))
            state.sentence_end_received = True
            return


def _elapsed_ms(start_at, end_at) -> float | None:
    """只用 perf_counter 同源时间点计算延迟；缺失值返回 None。"""
    if not isinstance(start_at, (int, float)) or not isinstance(
        end_at,
        (int, float),
    ):
        return None
    return (end_at - start_at) * 1000


def _format_metric(value: float | None) -> str:
    return f"{value:.0f} ms" if value is not None else "N/A"


async def _print_and_log_session(
    performance_logger: PerformanceLogger | None,
    state: SentenceState,
) -> None:
    """在本地播放结束后统一打印并记录网络与播放两组指标。"""
    metrics = state.metrics
    network_first_audio_ms = None
    first_playback_ms = None
    tts_session_total_ms = None
    sentence_playback_ms = None
    if metrics.first_audio_chunk_at is not None:
        network_first_audio_ms = (
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
    if (
        metrics.first_playback_at is not None
        and metrics.playback_finished_at is not None
    ):
        sentence_playback_ms = (
            metrics.playback_finished_at - metrics.first_playback_at
        ) * 1000

    trace = state.trace
    endpoint_triggered_at = trace.get("endpoint_triggered_at")
    last_voice_at = trace.get("last_voice_at")
    asr_result_at = trace.get("asr_result_at")
    translation_first_token_at = trace.get("translation_first_token_at")
    translation_first_segment_at = trace.get(
        "translation_first_segment_at"
    )
    first_audio_received_at = trace.get("tts_first_audio_received_at")
    first_playback_at = trace.get("first_playback_at")

    speech_end_to_asr_result_ms = _elapsed_ms(
        endpoint_triggered_at,
        asr_result_at,
    )
    speech_end_to_translation_first_token_ms = _elapsed_ms(
        endpoint_triggered_at,
        translation_first_token_at,
    )
    speech_end_to_translation_first_segment_ms = _elapsed_ms(
        endpoint_triggered_at,
        translation_first_segment_at,
    )
    speech_end_to_tts_first_audio_ms = _elapsed_ms(
        endpoint_triggered_at,
        first_audio_received_at,
    )
    speech_end_to_first_playback_ms = _elapsed_ms(
        endpoint_triggered_at,
        first_playback_at,
    )
    last_voice_to_first_playback_ms = _elapsed_ms(
        last_voice_at,
        first_playback_at,
    )
    endpoint_wait_ms = _elapsed_ms(
        last_voice_at,
        endpoint_triggered_at,
    )
    translation_after_asr_ms = _elapsed_ms(
        asr_result_at,
        translation_first_segment_at,
    )
    tts_after_first_segment_ms = _elapsed_ms(
        translation_first_segment_at,
        first_audio_received_at,
    )

    print(f"Audio Chunk Count: {metrics.audio_chunk_count}")
    print(f"Total Audio Bytes: {metrics.total_audio_bytes}")
    print(
        "Network First Audio: "
        f"{network_first_audio_ms:.0f} ms"
        if network_first_audio_ms is not None
        else "Network First Audio: N/A"
    )
    print(
        f"TTS Queue Wait: {metrics.tts_queue_wait_ms:.0f} ms"
        if metrics.tts_queue_wait_ms is not None
        else "TTS Queue Wait: N/A"
    )
    print(
        f"Sentence Playback: {sentence_playback_ms:.0f} ms"
        if sentence_playback_ms is not None
        else "Sentence Playback: N/A"
    )
    print(
        f"Cross-Sentence Gap: {metrics.cross_sentence_gap_ms:.0f} ms"
        if metrics.cross_sentence_gap_ms is not None
        else "Cross-Sentence Gap: N/A"
    )

    print("\n========== Trace Summary ==========")
    print(f"Trace ID: {trace.get('trace_id', 'N/A')}")
    print(f"Sentence ID: {state.sentence_id}")
    print(
        "Speech End → ASR Result: "
        f"{_format_metric(speech_end_to_asr_result_ms)}"
    )
    print(
        "Speech End → First Translation Segment: "
        f"{_format_metric(speech_end_to_translation_first_segment_ms)}"
    )
    print(
        "Speech End → First TTS Audio: "
        f"{_format_metric(speech_end_to_tts_first_audio_ms)}"
    )
    print(
        "Speech End → First English Playback: "
        f"{_format_metric(speech_end_to_first_playback_ms)}"
    )
    print(
        "Last Voice → First English Playback: "
        f"{_format_metric(last_voice_to_first_playback_ms)}"
    )
    print("Breakdown:")
    print(f"Endpoint Wait: {_format_metric(endpoint_wait_ms)}")
    print(f"ASR: {_format_metric(speech_end_to_asr_result_ms)}")
    print(
        "Translation after ASR: "
        f"{_format_metric(translation_after_asr_ms)}"
    )
    print(
        "ElevenLabs after First Segment: "
        f"{_format_metric(tts_after_first_segment_ms)}"
    )
    print(
        "Playback Queue: "
        f"{_format_metric(metrics.tts_queue_wait_ms)}"
    )
    print("===================================")

    if performance_logger is None:
        return

    await performance_logger.log(
        {
            "event": "tts_session",
            "tts_provider": "elevenlabs",
            "trace_id": trace.get("trace_id"),
            "fallback_provider": state.fallback_provider,
            "sentence_id": state.sentence_id,
            "segment_count": len(state.segments),
            # 保留 V11.1 字段，并新增语义更清晰的同值字段。
            "first_audio_chunk_ms": network_first_audio_ms,
            "network_first_audio_ms": network_first_audio_ms,
            "tts_queue_wait_ms": metrics.tts_queue_wait_ms,
            "first_playback_ms": first_playback_ms,
            "tts_session_total_ms": tts_session_total_ms,
            "sentence_playback_ms": sentence_playback_ms,
            "cross_sentence_gap_ms": metrics.cross_sentence_gap_ms,
            "audio_chunk_count": metrics.audio_chunk_count,
            "total_audio_bytes": metrics.total_audio_bytes,
            "partial_failure": state.partial_failure,
        }
    )
    await performance_logger.log(
        {
            "event": "trace_summary",
            "trace_id": trace.get("trace_id"),
            "sentence_id": state.sentence_id,
            "raw_text": trace.get("raw_text"),
            "source_text": trace.get("source_text"),
            "translated_text": trace.get("translated_text"),
            "speech_started_at": trace.get("speech_started_at"),
            "last_voice_at": last_voice_at,
            "endpoint_triggered_at": endpoint_triggered_at,
            "asr_started_at": trace.get("asr_started_at"),
            "asr_result_at": asr_result_at,
            "hotword_done_at": trace.get("hotword_done_at"),
            "translation_request_started_at": trace.get(
                "translation_request_started_at"
            ),
            "translation_first_token_at": translation_first_token_at,
            "translation_first_segment_at": (
                translation_first_segment_at
            ),
            "translation_finished_at": trace.get(
                "translation_finished_at"
            ),
            "tts_first_text_sent_at": trace.get(
                "tts_first_text_sent_at"
            ),
            "tts_first_audio_received_at": first_audio_received_at,
            "first_playback_at": first_playback_at,
            "tts_session_finished_at": trace.get(
                "tts_session_finished_at"
            ),
            "playback_finished_at": trace.get("playback_finished_at"),
            "endpoint_wait_ms": endpoint_wait_ms,
            "speech_end_to_asr_result_ms": (
                speech_end_to_asr_result_ms
            ),
            "speech_end_to_translation_first_token_ms": (
                speech_end_to_translation_first_token_ms
            ),
            "speech_end_to_translation_first_segment_ms": (
                speech_end_to_translation_first_segment_ms
            ),
            "speech_end_to_tts_first_audio_ms": (
                speech_end_to_tts_first_audio_ms
            ),
            "speech_end_to_first_playback_ms": (
                speech_end_to_first_playback_ms
            ),
            "endpoint_trigger_to_first_playback_ms": (
                speech_end_to_first_playback_ms
            ),
            "last_voice_to_first_playback_ms": (
                last_voice_to_first_playback_ms
            ),
            "asr_stage_ms": speech_end_to_asr_result_ms,
            "translation_after_asr_ms": translation_after_asr_ms,
            "tts_after_first_segment_ms": tts_after_first_segment_ms,
            "playback_queue_ms": metrics.tts_queue_wait_ms,
            "translation_ttft_ms": _elapsed_ms(
                trace.get("translation_request_started_at"),
                translation_first_token_at,
            ),
            "translation_ttfs_ms": _elapsed_ms(
                trace.get("translation_request_started_at"),
                translation_first_segment_at,
            ),
            "network_first_audio_ms": network_first_audio_ms,
            "tts_queue_wait_ms": metrics.tts_queue_wait_ms,
        }
    )
    performance_logger.emit_event(
        {
            "type": "status",
            "status": "Listening",
            "sentence_id": state.sentence_id,
        }
    )


def _print_network_session_summary(state: SentenceState) -> None:
    """WebSocket 完成时只打印服务器会话指标，不冒充播放完成。"""
    metrics = state.metrics
    if (
        metrics.first_text_sent_at is None
        or metrics.session_finished_at is None
    ):
        return
    total_ms = (
        metrics.session_finished_at - metrics.first_text_sent_at
    ) * 1000
    print(f"TTS Session Total: {total_ms:.0f} ms")


def _should_use_local_tts_fallback(
    state: SentenceState,
    audio_routing: AudioRoutingPlan,
) -> bool:
    """显式设备路由不能被默认扬声器 fallback 绕过。"""
    return (
        state.metrics.audio_chunk_count == 0
        and bool(state.full_text)
        and audio_routing.translation_uses_default
    )


async def elevenlabs_tts_worker(
    translated_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
    audio_routing: AudioRoutingPlan | None = None,
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
    print(
        "Global Audio Queue: "
        f"maxsize={ELEVENLABS_AUDIO_QUEUE_MAXSIZE}"
    )

    if audio_routing is None:
        audio_routing = build_audio_routing_plan(
            TRANSLATION_OUTPUT_DEVICE,
            LOCAL_MONITOR_ENABLED,
            LOCAL_MONITOR_DEVICE,
            ELEVENLABS_SAMPLE_RATE,
            ELEVENLABS_CHANNELS,
            ELEVENLABS_DTYPE,
        )

    try:
        output_streams = _open_audio_output_streams(audio_routing)
    except Exception as error:
        if not audio_routing.translation_uses_default:
            raise AudioRoutingError(
                "Translation output device "
                f"{format_device(audio_routing.translation_device)} "
                f"cannot be opened: {type(error).__name__}: {error}"
            ) from error
        print(
            "[ElevenLabs Playback Error] 无法初始化全局 OutputStream："
            f"{type(error).__name__}，改用 pyttsx3"
        )
        await tts_worker(translated_queue, performance_logger)
        return

    runtime_session_id = (
        performance_logger.session_id
        if performance_logger is not None
        else None
    )
    audio_playback_queue = asyncio.Queue(
        maxsize=ELEVENLABS_AUDIO_QUEUE_MAXSIZE
    )
    sentence_states: dict[int, SentenceState] = {}
    playback_task = asyncio.create_task(
        _global_playback_worker(
            audio_playback_queue,
            output_streams,
            sentence_states,
            performance_logger,
        )
    )

    try:
        while True:
            first_item = await translated_queue.get()
            if first_item is None:
                break
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
            sentence_states[sentence_id] = state
            _record_segment(state, first_item)
            print("\n[ElevenLabs TTS]")
            print(f"Sentence ID: {sentence_id}")
            print(f"Trace ID: {state.trace.get('trace_id', 'N/A')}")

            try:
                async with asyncio.timeout(TTS_SESSION_TIMEOUT_SECONDS):
                    await _run_sentence_session(
                        translated_queue,
                        audio_playback_queue,
                        first_item,
                        state,
                        api_key,
                        runtime_session_id,
                    )
                _print_network_session_summary(state)
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

                if _should_use_local_tts_fallback(state, audio_routing):
                    print("[ElevenLabs TTS] 当前句排队回退到 pyttsx3")
                    await audio_playback_queue.put(
                        {
                            "type": "pyttsx3_fallback",
                            "session_id": runtime_session_id,
                            "trace_id": state.trace.get("trace_id"),
                            "sentence_id": sentence_id,
                            "text": state.full_text,
                        }
                    )
                else:
                    state.partial_failure = True
                    if (
                        state.metrics.audio_chunk_count == 0
                        and not audio_routing.translation_uses_default
                    ):
                        print(
                            "[ElevenLabs TTS] 显式输出设备模式禁止回退到"
                            "默认扬声器"
                        )
                    elif state.metrics.audio_chunk_count == 0:
                        print("[ElevenLabs TTS] 没有可用于本地回退的文本")
                    else:
                        print(
                            "[ElevenLabs TTS] 已收到部分音频，"
                            "为避免重复不再整句回退"
                        )
                    await audio_playback_queue.put(
                        {
                            "type": "sentence_end",
                            "session_id": runtime_session_id,
                            "trace_id": state.trace.get("trace_id"),
                            "sentence_id": sentence_id,
                        }
                    )

            if state.pipeline_finished:
                break
    finally:
        # None 只在所有句级网络 Session 都结束后关闭全局 Playback Worker。
        await audio_playback_queue.put(None)
        await playback_task
        await _close_audio_output_streams(output_streams)

    print("[ElevenLabs TTS] 播放任务结束")
