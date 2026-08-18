import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.paths import get_resource_path, get_user_data_path


class PathTests(unittest.TestCase):
    def test_user_data_path_supports_deployment_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {"REALTIME_TRANSLATOR_USER_DATA_DIR": temp_dir},
            ):
                result = get_user_data_path("logs", "runtime.jsonl")
        self.assertEqual(
            result,
            Path(temp_dir).resolve() / "logs" / "runtime.jsonl",
        )

    def test_resource_path_uses_pyinstaller_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(sys, "_MEIPASS", temp_dir, create=True):
                result = get_resource_path("data", "hotwords.json")
        self.assertEqual(
            result,
            Path(temp_dir).resolve() / "data" / "hotwords.json",
        )


if __name__ == "__main__":
    unittest.main()
