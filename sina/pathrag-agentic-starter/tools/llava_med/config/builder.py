from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from llava.model import LlavaMistralForCausalLM
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit: bool = False,
    load_4bit: bool = False,
    device_map: str | dict = "auto",
    device: str = "cuda",
):
    """
    Patched version for Colab / CUDA 12.x:

    - Completely ignores `load_8bit` / `load_4bit` and DOES NOT use BitsAndBytes / quantization.
    - Always loads models in float16 (fp16) and relies on standard PyTorch GPU execution.
    - Avoids importing or configuring BitsAndBytesConfig so that `bitsandbytes` is not required.

    This is intended to work around bitsandbytes/CUDA incompatibilities while keeping
    LLaVA-Med usable for inference.
    """

    kwargs: dict = {}

    # Device handling
    if device != "cuda":
        # Explicit single-device mapping (e.g. "cpu")
        kwargs["device_map"] = {"": device}
    else:
        # For GPU, allow "auto" or a passed-in device_map.
        kwargs["device_map"] = device_map

    # Always use float16; no quantization flags
    kwargs["torch_dtype"] = torch.float16

    # ------------------------------------------------------------------
    # LLaVA models (vision + language)
    # ------------------------------------------------------------------
    if "llava" in model_name.lower():
        if "mistral" in model_name.lower():
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # LLaVA-Mistral model (no 4/8-bit, no bitsandbytes)
            model = LlavaMistralForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                use_flash_attention_2=False,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported LLaVA variant in model_name={model_name!r}; "
                "this patched builder currently expects a mistral-based LLaVA."
            )

    # ------------------------------------------------------------------
    # Plain language models (no vision tower)
    # ------------------------------------------------------------------
    else:
        # PEFT (LoRA) model
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                **kwargs,
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)

        # Base Causal LM
        else:
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    **kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    use_fast=use_fast,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )

    # ------------------------------------------------------------------
    # Vision tower & image processor for LLaVA
    # ------------------------------------------------------------------
    image_processor = None

    if "llava" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True,
            )

        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        # Move vision tower + projector to device in fp16
        vision_tower.to(device=device, dtype=torch.float16)
        model.model.mm_projector.to(device=device, dtype=torch.float16)
        model.to(device=device, dtype=torch.float16)

        image_processor = vision_tower.image_processor

    # ------------------------------------------------------------------
    # Context length
    # ------------------------------------------------------------------
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

