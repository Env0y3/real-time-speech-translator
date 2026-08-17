import asyncio
import inspect
import os
import re
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    STREAMING_TRANSLATION_MAX_CHARS,
    STREAMING_TRANSLATION_MIN_CHARS,
    STREAMING_TRANSLATION_TARGET_CHARS,
    TRANSLATION_SYSTEM_PROMPT,
)
from core.performance_logger import PerformanceLogger


TRANSLATION_BOUNDARY_PUNCTUATION = ",.!?;:"


def _emit_pipeline_event(
    performance_logger: PerformanceLogger | None,
    event: dict,
) -> None:
    if performance_logger is not None:
        performance_logger.emit_event(event)


def has_translation_output(text: str) -> bool:
    """空字符串或纯空白翻译不能产生任何 TTS Queue 消息。"""
    return bool(text.strip())


def _translation_segment_message(
    text: str,
    segment_ready_at: float,
    sentence_id: int,
    segment_index: int,
    trace_metadata: dict,
) -> dict:
    """构造 Translation → TTS 的语块消息。"""
    return {
        "event": "segment",
        "sentence_id": sentence_id,
        "segment_index": segment_index,
        "text": text,
        "segment_ready_at": segment_ready_at,
        "is_final_segment": False,
        "trace": dict(trace_metadata),
    }


async def _publish_sentence_end(
    translated_queue: asyncio.Queue,
    sentence_id: int,
    full_translation: str,
    segment_count: int,
    trace_metadata: dict,
) -> None:
    """通知 TTS：当前中文句子的所有英文语块已经产生完毕。"""
    await translated_queue.put(
        {
            "event": "sentence_end",
            "sentence_id": sentence_id,
            "full_translation": full_translation,
            "segment_count": segment_count,
            "trace": dict(trace_metadata),
        }
    )


def _prepare_translation_input(
    queue_item,
    sentence_counter: int,
    performance_logger: PerformanceLogger | None,
) -> tuple[str, int, int, dict]:
    """兼容旧字符串输入，并让 ASR 生成的 sentence_id 成为权威编号。"""
    if isinstance(queue_item, dict):
        trace_metadata = dict(queue_item)
        chinese_text = str(trace_metadata.get("text", ""))
        sentence_id = int(
            trace_metadata.get("sentence_id", sentence_counter + 1)
        )
    else:
        chinese_text = str(queue_item)
        sentence_id = sentence_counter + 1
        trace_metadata = {
            "sentence_id": sentence_id,
            "raw_text": chinese_text,
            "text": chinese_text,
        }

    session_id = (
        performance_logger.session_id
        if performance_logger is not None
        else "translation"
    )
    trace_metadata.setdefault(
        "trace_id",
        f"{session_id}-{sentence_id}",
    )
    trace_metadata["sentence_id"] = sentence_id
    trace_metadata["source_text"] = chinese_text
    return (
        chinese_text,
        sentence_id,
        max(sentence_counter, sentence_id),
        trace_metadata,
    )


async def request_full_translation(
    client: AsyncOpenAI,
    chinese_text: str,
) -> str:
    """使用原有非流式参数请求完整英文翻译。"""
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
    return (response.choices[0].message.content or "").strip()


def _word_boundary_at_or_before(
    text: str,
    limit: int,
) -> int | None:
    """寻找不超过指定长度的最后一个空白单词边界。"""
    boundaries = list(re.finditer(r"\s+", text[:limit + 1]))
    for boundary in reversed(boundaries):
        if len(text[:boundary.start()].strip()) >= (
            STREAMING_TRANSLATION_MIN_CHARS
        ):
            return boundary.start()
    return None


def extract_translation_segments(
    buffer: str,
    stream_finished: bool = False,
) -> tuple[list[str], str]:
    """从流式翻译Buffer（缓冲区）中提取可稳定发送的英文语块。"""
    segments = []
    remaining_buffer = buffer.lstrip()

    while remaining_buffer:
        if stream_finished:
            final_segment = remaining_buffer.strip()
            if final_segment:
                segments.append(final_segment)
            remaining_buffer = ""
            break

        boundary_index = None

        # 标点优先，但仍要求达到最小长度，避免把很短的词组单独送给TTS。
        for punctuation_match in re.finditer(
            f"[{re.escape(TRANSLATION_BOUNDARY_PUNCTUATION)}]",
            remaining_buffer,
        ):
            candidate_end = punctuation_match.end()
            if len(remaining_buffer[:candidate_end].strip()) >= (
                STREAMING_TRANSLATION_MIN_CHARS
            ):
                boundary_index = candidate_end
                break

        if boundary_index is None and len(remaining_buffer) >= (
            STREAMING_TRANSLATION_TARGET_CHARS
        ):
            boundary_index = _word_boundary_at_or_before(
                remaining_buffer,
                STREAMING_TRANSLATION_TARGET_CHARS,
            )

        if boundary_index is None and len(remaining_buffer) >= (
            STREAMING_TRANSLATION_MAX_CHARS
        ):
            boundary_index = _word_boundary_at_or_before(
                remaining_buffer,
                STREAMING_TRANSLATION_MAX_CHARS,
            )
            if boundary_index is None:
                # 极长单词不从中间切开；等待其后的第一个空白边界。
                later_boundary = re.search(
                    r"\s+",
                    remaining_buffer[STREAMING_TRANSLATION_MAX_CHARS:],
                )
                if later_boundary is not None:
                    boundary_index = (
                        STREAMING_TRANSLATION_MAX_CHARS
                        + later_boundary.start()
                    )

        if boundary_index is None:
            break

        segment = remaining_buffer[:boundary_index].strip()
        if not segment:
            break
        segments.append(segment)
        remaining_buffer = remaining_buffer[boundary_index:].lstrip()

    return segments, remaining_buffer


async def translation_worker(
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """消费稳定中文文本，通过 DeepSeek 翻译成英文。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY")
        _emit_pipeline_event(
            performance_logger,
            {"type": "error", "message": "Missing DEEPSEEK_API_KEY"},
        )
        await translated_queue.put(None)
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=15.0,
        max_retries=0,
    )
    print(f"Translation 已就绪：DeepSeek {DEEPSEEK_MODEL}")
    sentence_counter = 0

    try:
        while True:
            queue_item = await text_queue.get()

            if queue_item is None:
                await translated_queue.put(None)
                print("Translation Worker 已结束")
                break

            (
                chinese_text,
                sentence_id,
                sentence_counter,
                trace_metadata,
            ) = _prepare_translation_input(
                queue_item,
                sentence_counter,
                performance_logger,
            )
            _emit_pipeline_event(
                performance_logger,
                {
                    "type": "asr_result",
                    "sentence_id": sentence_id,
                    "text": chinese_text,
                    "trace_id": trace_metadata.get("trace_id"),
                },
            )
            _emit_pipeline_event(
                performance_logger,
                {"type": "status", "status": "Translating"},
            )
            request_started_at = time.perf_counter()
            trace_metadata["translation_request_started_at"] = (
                request_started_at
            )

            try:
                english_text = await request_full_translation(
                    client,
                    chinese_text,
                )
                translation_finished_at = time.perf_counter()
                latency_ms = (
                    translation_finished_at - request_started_at
                ) * 1000

                if not has_translation_output(english_text):
                    print("[Translation Error] 模型返回了空文本")
                    continue

                trace_metadata.update(
                    {
                        "translation_first_segment_at": (
                            translation_finished_at
                        ),
                        "translation_finished_at": translation_finished_at,
                        "translated_text": english_text,
                    }
                )

                print("\n[Translation]")
                print(f"中文：{chinese_text}")
                print(f"英文：{english_text}")
                print(f"翻译耗时：{latency_ms:.0f} ms")
                await translated_queue.put(
                    _translation_segment_message(
                        english_text,
                        translation_finished_at,
                        sentence_id,
                        1,
                        trace_metadata,
                    )
                )
                _emit_pipeline_event(
                    performance_logger,
                    {
                        "type": "translation_segment",
                        "sentence_id": sentence_id,
                        "segment_index": 1,
                        "text": english_text,
                    },
                )
                await _publish_sentence_end(
                    translated_queue,
                    sentence_id,
                    english_text,
                    1,
                    trace_metadata,
                )
                await _log_translation_result(
                    performance_logger,
                    sentence_id,
                    chinese_text,
                    english_text,
                    None,
                    latency_ms,
                    latency_ms,
                    1,
                    False,
                    trace_metadata,
                )

            except APITimeoutError:
                print("[Translation Error] 请求超时，请继续说下一句")
                _emit_pipeline_event(
                    performance_logger,
                    {"type": "error", "message": "DeepSeek request timed out"},
                )
            except RateLimitError:
                print("[Translation Error] API 触发速率限制")
                _emit_pipeline_event(
                    performance_logger,
                    {"type": "error", "message": "DeepSeek rate limit reached"},
                )
            except APIConnectionError:
                print("[Translation Error] 无法连接 DeepSeek API")
                _emit_pipeline_event(
                    performance_logger,
                    {"type": "error", "message": "DeepSeek connection error"},
                )
            except APIStatusError as error:
                print(
                    "[Translation Error] API 请求失败，"
                    f"状态码：{error.status_code}"
                )
                _emit_pipeline_event(
                    performance_logger,
                    {
                        "type": "error",
                        "message": f"DeepSeek API error ({error.status_code})",
                    },
                )
            except Exception as error:
                print(
                    "[Translation Error] 翻译失败："
                    f"{type(error).__name__}"
                )
                _emit_pipeline_event(
                    performance_logger,
                    {
                        "type": "error",
                        "message": f"Translation failed: {type(error).__name__}",
                    },
                )
    finally:
        await client.close()


def _describe_translation_error(error: Exception) -> str:
    """把流式或fallback异常转换成简短、不会泄露敏感信息的日志。"""
    if isinstance(error, APITimeoutError):
        return "请求超时"
    if isinstance(error, RateLimitError):
        return "API 触发速率限制"
    if isinstance(error, APIConnectionError):
        return "无法连接 DeepSeek API"
    if isinstance(error, APIStatusError):
        return f"API 请求失败，状态码：{error.status_code}"
    return f"{type(error).__name__}"


def _print_streaming_metrics(
    request_started_at: float,
    first_token_at: float | None,
    first_segment_at: float | None,
    translation_finished_at: float,
    segment_count: int,
) -> tuple[float | None, float | None, float]:
    """打印TTFT、TTFS和完整流式翻译耗时。"""
    print("\n[Streaming Translation]")
    ttft_ms = None
    ttfs_ms = None
    if first_token_at is not None:
        # TTFT（Time To First Token，首Token延迟）。
        ttft_ms = (first_token_at - request_started_at) * 1000
        print(f"First Token Latency: {ttft_ms:.0f} ms")
    else:
        print("First Token Latency: N/A")

    if first_segment_at is not None:
        # TTFS（Time To First Segment，首语块延迟）直接决定TTS何时可开始。
        ttfs_ms = (first_segment_at - request_started_at) * 1000
        print(f"First Segment Latency: {ttfs_ms:.0f} ms")
    else:
        print("First Segment Latency: N/A")

    total_latency_ms = (
        translation_finished_at - request_started_at
    ) * 1000
    print(f"Translation Total Latency: {total_latency_ms:.0f} ms")
    print(f"Segment Count: {segment_count}")
    return ttft_ms, ttfs_ms, total_latency_ms


async def _log_translation_result(
    performance_logger: PerformanceLogger | None,
    sentence_id: int,
    source_text: str,
    full_translation: str,
    ttft_ms: float | None,
    ttfs_ms: float | None,
    translation_total_ms: float,
    segment_count: int,
    streaming_enabled: bool,
    trace_metadata: dict,
) -> None:
    """保存一句翻译的结构化延迟数据；缺失指标使用 JSON null。"""
    if performance_logger is None:
        return
    await performance_logger.log(
        {
            "event": "translation",
            "trace_id": trace_metadata.get("trace_id"),
            "sentence_id": sentence_id,
            "source_text": source_text,
            "full_translation": full_translation,
            "ttft_ms": ttft_ms,
            "ttfs_ms": ttfs_ms,
            "translation_total_ms": translation_total_ms,
            "segment_count": segment_count,
            "streaming_enabled": streaming_enabled,
        }
    )


async def _fallback_to_full_translation(
    client: AsyncOpenAI,
    chinese_text: str,
    translated_queue: asyncio.Queue,
    request_started_at: float,
    first_token_at: float | None,
    sentence_id: int,
    performance_logger: PerformanceLogger | None,
    trace_metadata: dict,
) -> None:
    """流式请求在首段输出前失败时，回退到原来的完整翻译。"""
    print("[Streaming Translation] 回退到完整翻译")
    try:
        english_text = await request_full_translation(client, chinese_text)
    except Exception as error:
        print(
            "[Translation Fallback Error] "
            f"{_describe_translation_error(error)}"
        )
        _emit_pipeline_event(
            performance_logger,
            {
                "type": "error",
                "message": f"DeepSeek fallback failed: {_describe_translation_error(error)}",
            },
        )
        return

    if not has_translation_output(english_text):
        print("[Translation Fallback Error] 模型返回了空文本")
        return

    segment_ready_at = time.perf_counter()
    trace_metadata.update(
        {
            "translation_first_segment_at": segment_ready_at,
            "translation_finished_at": segment_ready_at,
            "translated_text": english_text,
        }
    )
    print("\n[Translation Segment 1]")
    print(english_text)
    print("\n[Full Streaming Translation]")
    print(english_text)
    await translated_queue.put(
        _translation_segment_message(
            english_text,
            segment_ready_at,
            sentence_id,
            1,
            trace_metadata,
        )
    )
    _emit_pipeline_event(
        performance_logger,
        {
            "type": "translation_segment",
            "sentence_id": sentence_id,
            "segment_index": 1,
            "text": english_text,
        },
    )
    await _publish_sentence_end(
        translated_queue,
        sentence_id,
        english_text,
        1,
        trace_metadata,
    )
    ttft_ms, ttfs_ms, total_latency_ms = _print_streaming_metrics(
        request_started_at,
        first_token_at,
        segment_ready_at,
        segment_ready_at,
        1,
    )
    await _log_translation_result(
        performance_logger,
        sentence_id,
        chinese_text,
        english_text,
        ttft_ms,
        ttfs_ms,
        total_latency_ms,
        1,
        True,
        trace_metadata,
    )


async def streaming_translation_worker(
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
    performance_logger: PerformanceLogger | None = None,
) -> None:
    """消费中文文本，持续接收DeepSeek增量内容并按稳定语块输出。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("缺少 DEEPSEEK_API_KEY")
        _emit_pipeline_event(
            performance_logger,
            {"type": "error", "message": "Missing DEEPSEEK_API_KEY"},
        )
        await translated_queue.put(None)
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=15.0,
        max_retries=0,
    )
    print(f"Streaming Translation 已就绪：DeepSeek {DEEPSEEK_MODEL}")
    sentence_counter = 0

    try:
        while True:
            queue_item = await text_queue.get()

            if queue_item is None:
                await translated_queue.put(None)
                print("Streaming Translation Worker 已结束")
                break

            (
                chinese_text,
                sentence_id,
                sentence_counter,
                trace_metadata,
            ) = _prepare_translation_input(
                queue_item,
                sentence_counter,
                performance_logger,
            )
            _emit_pipeline_event(
                performance_logger,
                {
                    "type": "asr_result",
                    "sentence_id": sentence_id,
                    "text": chinese_text,
                    "trace_id": trace_metadata.get("trace_id"),
                },
            )
            _emit_pipeline_event(
                performance_logger,
                {"type": "status", "status": "Translating"},
            )
            request_started_at = time.perf_counter()
            trace_metadata["translation_request_started_at"] = (
                request_started_at
            )
            print("\n[Streaming Translation Request]")
            print(f"中文：{chinese_text}")
            print(f"Translation Request Start: {time.strftime('%H:%M:%S')}")

            first_token_at = None
            first_segment_at = None
            segment_count = 0
            buffer = ""
            full_translation_parts = []
            stream_error = None

            async def publish_segment(segment: str) -> None:
                nonlocal first_segment_at, segment_count
                segment_ready_at = time.perf_counter()
                if first_segment_at is None:
                    first_segment_at = segment_ready_at
                    trace_metadata["translation_first_segment_at"] = (
                        segment_ready_at
                    )
                segment_count += 1
                print(f"\n[Translation Segment {segment_count}]")
                print(segment)

                # 同一Queue继续传递文本、准备时间、句子编号和语块编号。
                await translated_queue.put(
                    _translation_segment_message(
                        segment,
                        segment_ready_at,
                        sentence_id,
                        segment_count,
                        trace_metadata,
                    )
                )
                _emit_pipeline_event(
                    performance_logger,
                    {
                        "type": "translation_segment",
                        "sentence_id": sentence_id,
                        "segment_index": segment_count,
                        "text": segment,
                    },
                )

            stream = None
            try:
                stream = await client.chat.completions.create(
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
                    stream=True,
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta_content = chunk.choices[0].delta.content
                    if not isinstance(delta_content, str) or not delta_content:
                        continue

                    if first_token_at is None and delta_content.strip():
                        first_token_at = time.perf_counter()
                        trace_metadata["translation_first_token_at"] = (
                            first_token_at
                        )

                    # OpenAI Compatible流返回的是delta.content增量，只追加一次。
                    full_translation_parts.append(delta_content)
                    buffer += delta_content
                    ready_segments, buffer = extract_translation_segments(
                        buffer
                    )
                    for ready_segment in ready_segments:
                        await publish_segment(ready_segment)

            except Exception as error:
                stream_error = error
            finally:
                # 显式关闭SDK的异步流，及时释放响应连接。部分Python 3.14
                # 环境的底层httpcore2仍可能额外输出生成器清理警告。
                close_stream = getattr(stream, "close", None)
                if close_stream is not None:
                    try:
                        close_result = close_stream()
                        if inspect.isawaitable(close_result):
                            await close_result
                    except Exception as close_error:
                        # 翻译内容已经接收完成时，连接清理异常不应触发整句重译。
                        print(
                            "[Streaming Translation Warning] "
                            f"流关闭失败：{close_error}"
                        )

            full_translation = "".join(full_translation_parts).strip()

            if stream_error is not None:
                print(
                    "[Streaming Translation Error] "
                    f"{_describe_translation_error(stream_error)}"
                )
                _emit_pipeline_event(
                    performance_logger,
                    {
                        "type": "error",
                        "message": (
                            "DeepSeek stream error: "
                            f"{_describe_translation_error(stream_error)}"
                        ),
                    },
                )
                if segment_count == 0:
                    await _fallback_to_full_translation(
                        client,
                        chinese_text,
                        translated_queue,
                        request_started_at,
                        first_token_at,
                        sentence_id,
                        performance_logger,
                        trace_metadata,
                    )
                    continue

                # 已经播放过语块时不能再回退整句，否则会重复TTS。
                remaining_segments, buffer = extract_translation_segments(
                    buffer,
                    stream_finished=True,
                )
                for remaining_segment in remaining_segments:
                    await publish_segment(remaining_segment)
                print("\n[Partial Streaming Translation]")
                print(full_translation)
                translation_finished_at = time.perf_counter()
                trace_metadata.update(
                    {
                        "translation_finished_at": translation_finished_at,
                        "translated_text": full_translation,
                    }
                )
                ttft_ms, ttfs_ms, total_latency_ms = (
                    _print_streaming_metrics(
                        request_started_at,
                        first_token_at,
                        first_segment_at,
                        translation_finished_at,
                        segment_count,
                    )
                )
                await _log_translation_result(
                    performance_logger,
                    sentence_id,
                    chinese_text,
                    full_translation,
                    ttft_ms,
                    ttfs_ms,
                    total_latency_ms,
                    segment_count,
                    True,
                    trace_metadata,
                )
                await _publish_sentence_end(
                    translated_queue,
                    sentence_id,
                    full_translation,
                    segment_count,
                    trace_metadata,
                )
                continue

            if not has_translation_output(full_translation):
                print("[Streaming Translation Error] 模型返回了空流")
                # 成功结束的空流代表模型按约定返回空内容；不要再次请求，
                # 也不要向 TTS 发布 segment 或 sentence_end。
                continue

            remaining_segments, buffer = extract_translation_segments(
                buffer,
                stream_finished=True,
            )
            for remaining_segment in remaining_segments:
                await publish_segment(remaining_segment)

            print("\n[Full Streaming Translation]")
            print(full_translation)
            translation_finished_at = time.perf_counter()
            trace_metadata.update(
                {
                    "translation_finished_at": translation_finished_at,
                    "translated_text": full_translation,
                }
            )
            ttft_ms, ttfs_ms, total_latency_ms = _print_streaming_metrics(
                request_started_at,
                first_token_at,
                first_segment_at,
                translation_finished_at,
                segment_count,
            )
            await _log_translation_result(
                performance_logger,
                sentence_id,
                chinese_text,
                full_translation,
                ttft_ms,
                ttfs_ms,
                total_latency_ms,
                segment_count,
                True,
                trace_metadata,
            )
            await _publish_sentence_end(
                translated_queue,
                sentence_id,
                full_translation,
                segment_count,
                trace_metadata,
            )
    finally:
        await client.close()
