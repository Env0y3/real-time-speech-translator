from config import NORMALIZATION_PUNCTUATION


def normalize_text(text: str) -> str:
    """移除空白和常见标点，并把英文字母统一为小写。"""
    return "".join(
        character.lower()
        for character in text
        if not character.isspace()
        and character not in NORMALIZATION_PUNCTUATION
    )


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """按 Unicode 字符计算两个字符串之间的 Levenshtein 编辑距离。"""
    previous_row = list(range(len(hypothesis) + 1))

    for reference_index, reference_character in enumerate(reference, start=1):
        current_row = [reference_index]

        for hypothesis_index, hypothesis_character in enumerate(
            hypothesis,
            start=1,
        ):
            insertion_cost = current_row[hypothesis_index - 1] + 1
            deletion_cost = previous_row[hypothesis_index] + 1
            substitution_cost = (
                previous_row[hypothesis_index - 1]
                + (reference_character != hypothesis_character)
            )
            current_row.append(
                min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )
            )

        previous_row = current_row

    return previous_row[-1]

