import os
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "RealTimeSpeechTranslator"


def get_resource_root() -> Path:
    """返回源码项目根目录或 PyInstaller 解包资源目录。"""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root).resolve()
    return Path(__file__).resolve().parents[1]


def get_resource_path(*parts: str) -> Path:
    return get_resource_root().joinpath(*parts)


def get_user_data_root() -> Path:
    """返回可写用户目录；环境变量 override 仅用于测试和便携部署。"""
    override = os.environ.get("REALTIME_TRANSLATOR_USER_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIRECTORY_NAME


def get_user_data_path(*parts: str) -> Path:
    return get_user_data_root().joinpath(*parts)
