import sys
import types
import unittest
from unittest.mock import Mock, patch

from asr.sensevoice_asr import load_sensevoice_model


class SenseVoicePreloadTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_sensevoice_model.cache_clear()

    def test_preloaded_model_is_reused_by_pipeline_worker(self) -> None:
        model = object()
        auto_model = Mock(return_value=model)
        fake_funasr = types.SimpleNamespace(AutoModel=auto_model)
        load_sensevoice_model.cache_clear()

        with patch.dict(sys.modules, {"funasr": fake_funasr}):
            first = load_sensevoice_model()
            second = load_sensevoice_model()

        self.assertIs(first, model)
        self.assertIs(second, model)
        auto_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
