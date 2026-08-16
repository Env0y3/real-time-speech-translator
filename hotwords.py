import json
import re
from pathlib import Path

from config import (
    HOTWORD_CORRECTION_THRESHOLD,
    HOTWORD_MARGIN_THRESHOLD,
    HOTWORD_MAX_PHRASE_TOKENS,
)
from text_utils import levenshtein_distance, normalize_text


def load_hotwords(path: Path) -> list[str]:
    """读取并合并 hotwords.json 中各类别的 Hotword（热词）。"""
    try:
        with path.open("r", encoding="utf-8") as file:
            categories = json.load(file)
    except FileNotFoundError:
        print(f"[Hotword] {path.name} not found")
        return []
    except json.JSONDecodeError as error:
        print(
            f"[Hotword] {path.name} JSON 格式错误："
            f"第 {error.lineno} 行，第 {error.colno} 列"
        )
        return []
    except OSError as error:
        print(f"[Hotword] 无法读取 {path.name}：{error}")
        return []

    if not isinstance(categories, dict):
        print(f"[Hotword] {path.name} 顶层必须是 JSON 对象")
        return []

    hotwords = []
    seen_hotwords = set()
    for category_name, category_words in categories.items():
        if not isinstance(category_words, list):
            print(f"[Hotword] 跳过非列表类别：{category_name}")
            continue

        for word in category_words:
            if not isinstance(word, str) or not word.strip():
                continue
            clean_word = word.strip()
            if clean_word not in seen_hotwords:
                hotwords.append(clean_word)
                seen_hotwords.add(clean_word)

    return hotwords


def find_target_hotword(target_text: str, hotwords: list[str]) -> str | None:
    """从外部词表中找到目标句包含的热词，不使用品牌专属修正规则。"""
    normalized_target = normalize_text(target_text)
    for hotword in hotwords:
        if normalize_text(hotword) in normalized_target:
            return hotword
    return None


def extract_english_candidates(text: str) -> list[dict]:
    """提取 ASCII 英文 token，并组合相邻的短英文 phrase（短语）。"""
    token_matches = list(re.finditer(r"[A-Za-z]+", text))
    token_groups = []
    current_group = []

    for token_match in token_matches:
        if not current_group:
            current_group = [token_match]
            continue

        previous_match = current_group[-1]
        separator = text[previous_match.end():token_match.start()]
        if separator and separator.isspace():
            current_group.append(token_match)
        else:
            token_groups.append(current_group)
            current_group = [token_match]

    if current_group:
        token_groups.append(current_group)

    candidates = []
    for token_group in token_groups:
        max_tokens = min(HOTWORD_MAX_PHRASE_TOKENS, len(token_group))
        for phrase_size in range(1, max_tokens + 1):
            for start_index in range(len(token_group) - phrase_size + 1):
                end_index = start_index + phrase_size - 1
                start = token_group[start_index].start()
                end = token_group[end_index].end()
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                    }
                )

    return candidates


def normalized_edit_similarity(first_text: str, second_text: str) -> float:
    """计算忽略空白、标点和英文大小写后的编辑距离相似度。"""
    normalized_first = normalize_text(first_text)
    normalized_second = normalize_text(second_text)
    longest_length = max(len(normalized_first), len(normalized_second))
    if longest_length == 0:
        return 1.0

    distance = levenshtein_distance(normalized_first, normalized_second)
    return 1.0 - distance / longest_length


def correct_hotwords(
    text: str,
    hotwords: list[str],
) -> tuple[str, list[dict]]:
    """仅用外部热词表保守修正英文片段，并保留每次修改的信息。"""
    usable_hotwords = [word for word in hotwords if normalize_text(word)]
    if not text or not usable_hotwords:
        return text, []

    correction_candidates = []
    for english_candidate in extract_english_candidates(text):
        similarity_scores = sorted(
            (
                (
                    normalized_edit_similarity(
                        english_candidate["text"],
                        hotword,
                    ),
                    hotword,
                )
                for hotword in usable_hotwords
            ),
            reverse=True,
        )
        best_similarity, best_hotword = similarity_scores[0]
        second_similarity = (
            similarity_scores[1][0]
            if len(similarity_scores) > 1
            else 0.0
        )
        similarity_margin = best_similarity - second_similarity

        # Threshold（阈值）和 Margin（领先幅度）都满足才修改。
        # 如果不确定则保留 Raw ASR（原始识别结果），优先保证纠错精确率。
        if (
            best_similarity >= HOTWORD_CORRECTION_THRESHOLD
            and similarity_margin >= HOTWORD_MARGIN_THRESHOLD
            and english_candidate["text"] != best_hotword
        ):
            correction_candidates.append(
                {
                    **english_candidate,
                    "replacement": best_hotword,
                    "similarity": best_similarity,
                }
            )

    # 优先选择相似度更高、覆盖范围更长的候选，避免重叠片段被重复修改。
    correction_candidates.sort(
        key=lambda item: (
            -item["similarity"],
            -(item["end"] - item["start"]),
            item["start"],
        )
    )
    selected_corrections = []
    for correction in correction_candidates:
        overlaps = any(
            correction["start"] < selected["end"]
            and correction["end"] > selected["start"]
            for selected in selected_corrections
        )
        if not overlaps:
            selected_corrections.append(correction)

    corrected_text = text
    for correction in sorted(
        selected_corrections,
        key=lambda item: item["start"],
        reverse=True,
    ):
        corrected_text = (
            corrected_text[:correction["start"]]
            + correction["replacement"]
            + corrected_text[correction["end"]:]
        )

    corrections = [
        {
            "original": correction["text"],
            "replacement": correction["replacement"],
            "similarity": round(correction["similarity"], 4),
        }
        for correction in sorted(
            selected_corrections,
            key=lambda item: item["start"],
        )
    ]
    return corrected_text, corrections

