import importlib
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.graph.runtime as runtime
from src.graph.runtime import MedGemmaConfigurationError, load_yaml, pipeline_kwargs, runtime_settings


class RuntimeSettingsTests(unittest.TestCase):
    def test_default_nonquantized_settings_do_not_import_bitsandbytes(self):
        settings = runtime_settings({"models": {"medgemma_repo": "model"}, "generation": {}, "runtime": {"quantization": "none"}})
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

    def test_pipeline_dtype_preserves_selected_precision(self):
        settings = runtime_settings({"models": {"medgemma_repo": "model"}, "runtime": {"precision": "fp16"}})
        def modern_pipeline(*_args, dtype=None, **_kwargs):
            return None
        with patch.object(runtime, "pipeline", modern_pipeline):
            kwargs = pipeline_kwargs(settings)
        self.assertEqual(kwargs["dtype"], runtime.torch.float16)
        self.assertNotIn("torch_dtype", kwargs)

    def test_4bit_configuration_is_unchanged(self):
        settings = runtime_settings({"models": {"medgemma_repo": "model"}, "runtime": {"quantization": "4bit", "precision": "bf16"}})
        kwargs = pipeline_kwargs(settings)
        self.assertTrue(kwargs["quantization_config"].load_in_4bit)
        self.assertEqual(kwargs["quantization_config"].bnb_4bit_quant_type, "nf4")
        self.assertTrue(kwargs["quantization_config"].bnb_4bit_use_double_quant)

    def test_internal_defaults_match_validated_configuration(self):
        config = load_yaml(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
        with patch.dict(os.environ, {}, clear=True):
            settings = runtime_settings(config)
        self.assertEqual(settings.model_id, "google/medgemma-1.5-4b-it")
        self.assertEqual(settings.precision, "bf16")
        self.assertEqual(settings.quantization, "4bit")
        self.assertEqual(settings.max_new_tokens, 96)
        self.assertEqual(settings.microbatch_size, 1)

    def test_environment_overrides_yaml_defaults(self):
        config = {"models": {"medgemma_repo": "yaml-model"}, "generation": {"max_new_tokens": 96}, "runtime": {"precision": "bf16", "quantization": "4bit", "microbatch_size": 1}}
        env = {"MEDGEMMA_MODEL": "env-model", "MEDGEMMA_PRECISION": "fp16", "MEDGEMMA_QUANTIZATION": "8bit", "MEDGEMMA_MAX_NEW_TOKENS": "123"}
        with patch.dict(os.environ, env, clear=False):
            settings = runtime_settings(config)
        self.assertEqual((settings.model_id, settings.precision, settings.quantization, settings.max_new_tokens), ("env-model", "fp16", "8bit", 123))

    def test_cli_model_override_precedes_environment(self):
        with patch.dict(os.environ, {"MEDGEMMA_MODEL": "env-model"}, clear=False):
            settings = runtime_settings({}, model_override="cli-model")
        self.assertEqual(settings.model_id, "cli-model")

    def test_runtime_does_not_force_hugging_face_offline_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            pipeline_kwargs(runtime_settings({"runtime": {"quantization": "none"}}))
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)
            self.assertNotIn("TRANSFORMERS_OFFLINE", os.environ)


if __name__ == "__main__":
    unittest.main()
