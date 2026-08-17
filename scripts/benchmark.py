import asyncio
import json
from datetime import datetime

from config import BENCHMARK_REPEATS, BENCHMARK_RESULTS_PATH
from core.hotwords import correct_hotwords, find_target_hotword
from core.text_utils import levenshtein_distance, normalize_text


def save_benchmark_result(result: dict) -> None:
    """把一条 Benchmark 结果追加为 JSON Lines（每行一个 JSON 对象）。"""
    with BENCHMARK_RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


async def benchmark_worker(
    text_queue: asyncio.Queue,
    stop_event: asyncio.Event,
    microphone_ready: asyncio.Event,
    asr_ready: asyncio.Event,
    asr_provider: str,
    asr_model: str,
    test_sentences: list[str],
    benchmark_type: str,
    correction_enabled: bool,
    hotwords: list[str],
) -> None:
    """逐句提示用户朗读，记录所选 ASR Provider 的最终结果。"""
    await microphone_ready.wait()
    await asr_ready.wait()
    provider_display_name = (
        "Vosk" if asr_provider == "vosk" else "SenseVoiceSmall"
    )

    total_planned = len(test_sentences) * BENCHMARK_REPEATS
    total_samples = 0
    exact_match_count = 0
    normalized_exact_match_count = 0
    total_cer = 0.0
    endpoint_latency_total = 0.0
    endpoint_latency_count = 0
    inference_latency_total = 0.0
    inference_latency_count = 0
    speech_end_to_result_total = 0.0
    speech_end_to_result_count = 0
    raw_hotword_hit_count = 0
    corrected_hotword_hit_count = 0
    hotword_sample_count = 0
    corrected_total_cer = 0.0
    correction_count = 0
    false_correction_count = 0
    correction_improved_count = 0
    correction_worsened_count = 0
    correction_unchanged_count = 0
    stopped_early = False

    if benchmark_type == "english_hotword":
        print("\n========== Hotword Correction Benchmark ==========")
        print(
            "Post Correction: "
            f"{'ON' if correction_enabled else 'OFF'}"
        )
        print(f"Hotword Count: {len(hotwords)}")
    else:
        print("\n========== ASR Benchmark Mode ==========")
    print(f"ASR Provider: {provider_display_name}")
    print(f"ASR Model: {asr_model}")
    print(f"测试句数：{len(test_sentences)}")
    print(f"每句重复：{BENCHMARK_REPEATS} 次")
    print(f"计划样本：{total_planned}")

    for sentence_index, target_text in enumerate(test_sentences, start=1):
        for repeat_index in range(1, BENCHMARK_REPEATS + 1):
            if stop_event.is_set():
                stopped_early = True
                break

            sample_number = total_samples + 1
            print("\n----------------------------------------")
            print(
                f"[句子 {sentence_index}/{len(test_sentences)} | "
                f"第 {repeat_index}/{BENCHMARK_REPEATS} 次 | "
                f"样本 {sample_number}/{total_planned}]"
            )
            print("请朗读：")
            print(f"“{target_text}”")

            queue_item = await text_queue.get()
            if queue_item is None:
                stopped_early = True
                break

            recognized_text = queue_item[0]
            endpoint_latency_ms = queue_item[1]
            asr_inference_latency_ms = (
                queue_item[2] if len(queue_item) > 2 else None
            )
            speech_end_to_result_latency_ms = (
                queue_item[3]
                if len(queue_item) > 3
                else endpoint_latency_ms
            )
            raw_recognized_text = recognized_text
            corrected_recognized_text = raw_recognized_text
            corrections = []
            if (
                benchmark_type == "english_hotword"
                and correction_enabled
            ):
                corrected_recognized_text, corrections = correct_hotwords(
                    raw_recognized_text,
                    hotwords,
                )

            exact_match = raw_recognized_text == target_text
            normalized_target = normalize_text(target_text)
            normalized_recognized = normalize_text(raw_recognized_text)
            corrected_normalized = normalize_text(corrected_recognized_text)
            normalized_exact_match = (
                normalized_target == normalized_recognized
            )
            target_hotword = (
                find_target_hotword(target_text, hotwords)
                if benchmark_type == "english_hotword"
                else None
            )
            raw_hotword_hit = (
                normalize_text(target_hotword) in normalized_recognized
                if target_hotword is not None
                else None
            )
            corrected_hotword_hit = (
                normalize_text(target_hotword) in corrected_normalized
                if target_hotword is not None
                else None
            )
            edit_distance = levenshtein_distance(
                normalized_target,
                normalized_recognized,
            )
            corrected_edit_distance = levenshtein_distance(
                normalized_target,
                corrected_normalized,
            )

            # CER（字符错误率）通常用目标字符数作为分母。
            # 空目标且识别也为空时记为 0；只有识别文本时安全记为 1。
            if normalized_target:
                cer = edit_distance / len(normalized_target)
                corrected_cer = (
                    corrected_edit_distance / len(normalized_target)
                )
            else:
                cer = 0.0 if not normalized_recognized else 1.0
                corrected_cer = (
                    0.0 if not corrected_normalized else 1.0
                )

            correction_improved = (
                corrected_cer < cer
                or (
                    raw_hotword_hit is False
                    and corrected_hotword_hit is True
                    and corrected_cer <= cer
                )
            )
            correction_worsened = (
                not correction_improved
                and (
                    corrected_cer > cer
                    or (
                        raw_hotword_hit is True
                        and corrected_hotword_hit is False
                    )
                )
            )
            false_correction = bool(corrections) and correction_worsened

            total_samples += 1
            if exact_match:
                exact_match_count += 1
            if normalized_exact_match:
                normalized_exact_match_count += 1
            if raw_hotword_hit is not None:
                hotword_sample_count += 1
                if raw_hotword_hit:
                    raw_hotword_hit_count += 1
                if corrected_hotword_hit:
                    corrected_hotword_hit_count += 1
            total_cer += cer
            corrected_total_cer += corrected_cer
            correction_count += len(corrections)
            if false_correction:
                false_correction_count += 1
            if correction_improved:
                correction_improved_count += 1
            elif correction_worsened:
                correction_worsened_count += 1
            else:
                correction_unchanged_count += 1
            if endpoint_latency_ms is not None:
                endpoint_latency_total += endpoint_latency_ms
                endpoint_latency_count += 1
            if asr_inference_latency_ms is not None:
                inference_latency_total += asr_inference_latency_ms
                inference_latency_count += 1
            if speech_end_to_result_latency_ms is not None:
                speech_end_to_result_total += (
                    speech_end_to_result_latency_ms
                )
                speech_end_to_result_count += 1

            result = {
                "benchmark_type": benchmark_type,
                # 保留 hotword_enabled 兼容 V7.4 历史记录；
                # V7.5 中它与 correction_enabled 含义相同，均表示后处理开关。
                "hotword_enabled": correction_enabled,
                "correction_enabled": correction_enabled,
                "hotword_count": len(hotwords),
                "asr_provider": asr_provider,
                "asr_model": asr_model,
                "target": target_text,
                "recognized": raw_recognized_text,
                "raw_recognized": raw_recognized_text,
                "corrected_recognized": corrected_recognized_text,
                "exact_match": exact_match,
                "normalized_target": normalized_target,
                "normalized_recognized": normalized_recognized,
                "raw_normalized": normalized_recognized,
                "corrected_normalized": corrected_normalized,
                "normalized_exact_match": normalized_exact_match,
                "target_hotword": target_hotword,
                "hotword_hit": raw_hotword_hit,
                "raw_hotword_hit": raw_hotword_hit,
                "corrected_hotword_hit": corrected_hotword_hit,
                "edit_distance": edit_distance,
                "cer": cer,
                "raw_cer": cer,
                "corrected_edit_distance": corrected_edit_distance,
                "corrected_cer": corrected_cer,
                "corrections": corrections,
                "correction_improved": correction_improved,
                "correction_worsened": correction_worsened,
                "correction_unchanged": (
                    not correction_improved and not correction_worsened
                ),
                "false_correction": false_correction,
                "endpoint_latency_ms": (
                    round(endpoint_latency_ms, 1)
                    if endpoint_latency_ms is not None
                    else None
                ),
                "asr_inference_latency_ms": (
                    round(asr_inference_latency_ms, 1)
                    if asr_inference_latency_ms is not None
                    else None
                ),
                "speech_end_to_result_latency_ms": (
                    round(speech_end_to_result_latency_ms, 1)
                    if speech_end_to_result_latency_ms is not None
                    else None
                ),
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
            }
            await asyncio.to_thread(save_benchmark_result, result)

            print("目标：")
            print(target_text)
            print("Raw ASR（原始识别）：")
            print(raw_recognized_text)
            if benchmark_type == "english_hotword":
                print("Corrected（纠错后）：")
                print(corrected_recognized_text)
            print("标准化目标：")
            print(normalized_target)
            print("Raw 标准化识别：")
            print(normalized_recognized)
            print("Raw Exact Match:")
            print("PASS" if exact_match else "FAIL")
            print("Normalized Exact Match:")
            print("PASS" if normalized_exact_match else "FAIL")
            print("Raw CER:")
            print(f"{cer * 100:.2f}%")
            if benchmark_type == "english_hotword":
                print("Corrected CER:")
                print(f"{corrected_cer * 100:.2f}%")
            if target_hotword is not None:
                print(f"Target Hotword: {target_hotword}")
                print(
                    "Raw Hotword Hit: "
                    f"{'PASS' if raw_hotword_hit else 'FAIL'}"
                )
                print(
                    "Corrected Hotword Hit: "
                    f"{'PASS' if corrected_hotword_hit else 'FAIL'}"
                )
                print(
                    "Corrections: "
                    + (
                        json.dumps(corrections, ensure_ascii=False)
                        if corrections
                        else "[]"
                    )
                )
            if endpoint_latency_ms is not None:
                print(f"Endpoint Latency: {endpoint_latency_ms:.0f} ms")
            else:
                print("Endpoint Latency: N/A")
            if asr_inference_latency_ms is not None:
                print(
                    "ASR Inference Latency: "
                    f"{asr_inference_latency_ms:.0f} ms"
                )
            if speech_end_to_result_latency_ms is not None:
                print(
                    "Speech End To Result Latency: "
                    f"{speech_end_to_result_latency_ms:.0f} ms"
                )

        if stopped_early:
            break

    completed_all = total_samples == total_planned
    if completed_all:
        # 完成固定测试集后停止麦克风，并继续消费到 Sentinel，
        # 确保 ASR Worker 不会因 Queue（队列）背压而无法退出。
        stop_event.set()
        while await text_queue.get() is not None:
            pass

    exact_match_rate = (
        exact_match_count / total_samples * 100
        if total_samples
        else 0.0
    )
    normalized_exact_match_rate = (
        normalized_exact_match_count / total_samples * 100
        if total_samples
        else 0.0
    )
    average_cer = total_cer / total_samples if total_samples else 0.0
    average_corrected_cer = (
        corrected_total_cer / total_samples if total_samples else 0.0
    )
    average_endpoint_latency = (
        endpoint_latency_total / endpoint_latency_count
        if endpoint_latency_count
        else None
    )
    average_inference_latency = (
        inference_latency_total / inference_latency_count
        if inference_latency_count
        else None
    )
    average_speech_end_to_result_latency = (
        speech_end_to_result_total / speech_end_to_result_count
        if speech_end_to_result_count
        else None
    )
    raw_hotword_hit_rate = (
        raw_hotword_hit_count / hotword_sample_count * 100
        if hotword_sample_count
        else 0.0
    )
    corrected_hotword_hit_rate = (
        corrected_hotword_hit_count / hotword_sample_count * 100
        if hotword_sample_count
        else 0.0
    )

    if benchmark_type == "english_hotword":
        print("\n========== Hotword Correction Benchmark ==========")
    else:
        print("\n========== Benchmark Summary ==========")
    print(f"ASR Provider: {provider_display_name}")
    print(f"ASR Model: {asr_model}")
    if benchmark_type == "english_hotword":
        print(
            "Post Correction: "
            f"{'ON' if correction_enabled else 'OFF'}"
        )
    print(f"Total Samples: {total_samples}")
    print("\nRaw Exact Match:")
    print(f"{exact_match_count} / {total_samples}")
    print(f"{exact_match_rate:.1f}%")
    print("\nNormalized Exact Match:")
    print(f"{normalized_exact_match_count} / {total_samples}")
    print(f"{normalized_exact_match_rate:.1f}%")
    if benchmark_type == "english_hotword":
        print("\nRaw Average CER:")
        print(f"{average_cer * 100:.2f}%")
        print("\nCorrected Average CER:")
        print(f"{average_corrected_cer * 100:.2f}%")
        print("\nRaw Hotword Hit:")
        print(f"{raw_hotword_hit_count} / {hotword_sample_count}")
        print(f"{raw_hotword_hit_rate:.1f}%")
        print("\nCorrected Hotword Hit:")
        print(f"{corrected_hotword_hit_count} / {hotword_sample_count}")
        print(f"{corrected_hotword_hit_rate:.1f}%")
        print(f"\nCorrection Count: {correction_count}")
        print(f"False Correction Count: {false_correction_count}")
        print(f"Correction Improved: {correction_improved_count}")
        print(f"Correction Worsened: {correction_worsened_count}")
        print(f"Unchanged: {correction_unchanged_count}")
    else:
        print("\nAverage CER:")
        print(f"{average_cer * 100:.2f}%")
    print("\nAverage Endpoint Latency:")
    if average_endpoint_latency is not None:
        print(f"{average_endpoint_latency:.0f} ms")
    else:
        print("N/A")
    print("\nAverage ASR Inference Latency:")
    if average_inference_latency is not None:
        print(f"{average_inference_latency:.0f} ms")
    else:
        print("N/A")
    print("\nAverage Speech End To Result Latency:")
    if average_speech_end_to_result_latency is not None:
        print(f"{average_speech_end_to_result_latency:.0f} ms")
    else:
        print("N/A")
    if total_samples:
        print("\nResults saved to:")
        print(BENCHMARK_RESULTS_PATH.name)
    else:
        print("尚未产生可保存的测试结果")
    if stopped_early:
        print("Benchmark 提前停止")
    print("=======================================")

    if completed_all:
        print("Benchmark 已完成，请按 Enter 退出")
