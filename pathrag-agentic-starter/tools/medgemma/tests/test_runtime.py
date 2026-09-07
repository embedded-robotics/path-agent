import importlib
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.runtime import MedGemmaConfigurationError, pipeline_kwargs, runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_default_nonquantized_settings_do_not_import_bitsandbytes(self):
        settings = runtime_settings({"models": {"medgemma_repo": "google/medgemma-4b-it"}, "generation": {}, "runtime": {}})
        with patch.object(importlib, "import_module", side_effect=AssertionError("unexpected import")):
            kwargs = pipeline_kwargs(settings)
        self.assertNotIn("quantization_config", kwargs)

    def test_4bit_requires_bitsandbytes_before_model_loading(self):
        settings = runtime_settings({"models": {"medgemma_repo": "model"}, "runtime": {"quantization": "4bit"}})
        with patch.object(importlib, "import_module", side_effect=ImportError):
            with self.assertRaisesRegex(MedGemmaConfigurationError, "bitsandbytes"):
                pipeline_kwargs(settings)

    def test_invalid_quantization_is_rejected(self):
        with patch.dict("os.environ", {"MEDGEMMA_QUANTIZATION": "bad"}):
            with self.assertRaisesRegex(MedGemmaConfigurationError, "QUANTIZATION"):
                runtime_settings({"models": {"medgemma_repo": "model"}})

    def test_only_sequential_microbatch_one_is_accepted(self):
        settings = runtime_settings({"models": {"medgemma_repo": "model"}, "runtime": {"microbatch_size": 1}})
        self.assertEqual(settings.microbatch_size, 1)
        for value in (0, 2, -1):
            with patch.dict("os.environ", {"MEDGEMMA_MICROBATCH_SIZE": str(value)}):
                with self.assertRaisesRegex(MedGemmaConfigurationError, "sequential single-item"):
                    runtime_settings({"models": {"medgemma_repo": "model"}})


if __name__ == "__main__":
    unittest.main()
