import json
from pathlib import Path
from typing import Any

from core.paths import get_user_data_path
from gui.i18n import normalize_language


def default_settings_path() -> Path:
    return get_user_data_path("user_settings.json")


def load_user_settings(path: Path | None = None) -> dict[str, Any]:
    settings_path = path or default_settings_path()
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"ui_language": "zh"}
    if not isinstance(data, dict):
        return {"ui_language": "zh"}
    return {**data, "ui_language": normalize_language(data.get("ui_language"))}


def save_user_settings(
    settings: dict[str, Any],
    path: Path | None = None,
) -> None:
    settings_path = path or default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = settings_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(settings_path)
