import asyncio
import os
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
    TRANSLATION_SYSTEM_PROMPT,
)


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

                english_text = (
                    response.choices[0].message.content or ""
                ).strip()
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

