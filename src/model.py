"""
Model and tokenizer loading utilities for LoRA/QLoRA fine-tuning.
"""
from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str | None) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer.chat_template:
        raise ValueError(
            f"Model tokenizer '{model_id}' has no chat_template; provide a model-specific template before SFT."
        )
    return tokenizer


def load_model(
    model_id: str,
    quantization_config: dict[str, Any] | None = None,
    use_gradient_checkpointing: bool = True,
    device_map: str | dict[str, Any] | None = None,
):
    logger.info("Loading model: %s", model_id)
    quantization_config = quantization_config or {}
    bnb_config = None
    if quantization_config.get("load_in_4bit", False):
        compute_dtype = _resolve_dtype(quantization_config.get("bnb_4bit_compute_dtype"))
        if compute_dtype == torch.float32:
            raise ValueError("4-bit QLoRA requires CUDA; disable quantization for CPU environments")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(
                quantization_config.get("bnb_4bit_use_double_quant", True)
            ),
        )

    dtype = _resolve_dtype(
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "trust_remote_code": False,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if dtype != torch.float32:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model


def prepare_model_for_lora(model, lora_config_dict: dict[str, Any]):
    logger.info("Preparing model for LoRA adaptation")
    if getattr(model, "is_quantized", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(lora_config_dict.get("r", 16)),
        lora_alpha=int(lora_config_dict.get("lora_alpha", 32)),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=float(lora_config_dict.get("lora_dropout", 0.05)),
        bias=lora_config_dict.get("bias", "none"),
        task_type="CAUSAL_LM",
        modules_to_save=lora_config_dict.get("modules_to_save"),
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA parameters: trainable=%d total=%d ratio=%.4f%%",
        trainable,
        total,
        100.0 * trainable / total if total else 0.0,
    )
    return model
