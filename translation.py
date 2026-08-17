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


TRANSLATION_BOUNDARY_PUNCTUATION = ",.!?;:"


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
                english_text = await request_full_translation(
                    client,
                    chinese_text,
                )
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
) -> None:
    """打印TTFT、TTFS和完整流式翻译耗时。"""
    print("\n[Streaming Translation]")
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


async def _fallback_to_full_translation(
    client: AsyncOpenAI,
    chinese_text: str,
    translated_queue: asyncio.Queue,
    request_started_at: float,
    first_token_at: float | None,
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
        return

    if not english_text:
        print("[Translation Fallback Error] 模型返回了空文本")
        return

    segment_ready_at = time.perf_counter()
    print("\n[Translation Segment 1]")
    print(english_text)
    print("\n[Full Streaming Translation]")
    print(english_text)
    await translated_queue.put((english_text, segment_ready_at))
    _print_streaming_metrics(
        request_started_at,
        first_token_at,
        segment_ready_at,
        segment_ready_at,
        1,
    )


async def streaming_translation_worker(
    text_queue: asyncio.Queue,
    translated_queue: asyncio.Queue,
) -> None:
    """消费中文文本，持续接收DeepSeek增量内容并按稳定语块输出。"""
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
    print(f"Streaming Translation 已就绪：DeepSeek {DEEPSEEK_MODEL}")

    try:
        while True:
            chinese_text = await text_queue.get()

            if chinese_text is None:
                await translated_queue.put(None)
                print("Streaming Translation Worker 已结束")
                break

            request_started_at = time.perf_counter()
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
                segment_count += 1
                print(f"\n[Translation Segment {segment_count}]")
                print(segment)

                # TTS仍消费原有二元组；此时间表示该segment准备完成的时刻。
                await translated_queue.put((segment, segment_ready_at))

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
                if segment_count == 0:
                    await _fallback_to_full_translation(
                        client,
                        chinese_text,
                        translated_queue,
                        request_started_at,
                        first_token_at,
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
                _print_streaming_metrics(
                    request_started_at,
                    first_token_at,
                    first_segment_at,
                    translation_finished_at,
                    segment_count,
                )
                continue

            if not full_translation:
                print("[Streaming Translation Error] 模型返回了空流")
                await _fallback_to_full_translation(
                    client,
                    chinese_text,
                    translated_queue,
                    request_started_at,
                    first_token_at,
                )
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
            _print_streaming_metrics(
                request_started_at,
                first_token_at,
                first_segment_at,
                translation_finished_at,
                segment_count,
            )
    finally:
        await client.close()
