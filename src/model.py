"""
Model loader module for the fine-tuning pipeline.
Supports bitsandbytes quantization (QLoRA) and PEFT adapter wrapping.
"""

import logging
import torch
from typing import Tuple, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

def load_tokenizer(model_id: str) -> PreTrainedTokenizer:
    """
    Load tokenizer from Hugging Face Hub, ensuring correct padding token and chat template.
    """
    logger.info("Loading tokenizer for model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Ensure pad token is set (important for batching in SFT)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            
    tokenizer.padding_side = "right"  # Standard for multi-turn training (avoid issues with attention mask)
    return tokenizer

def load_model(
    model_id: str,
    quantization_config: Dict[str, Any] = None,
    use_gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """
    Load a pretrained causal language model with dynamic device placement.
    """
    logger.info("Loading model: %s", model_id)
    
    # Configure quantization if specified
    bnb_config = None
    if quantization_config and quantization_config.get("load_in_4bit", False):
        logger.info("Configuring 4-bit bitsandbytes quantization (QLoRA)...")
        compute_dtype = torch.bfloat16 if quantization_config.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", True)
        )
    
    # Load raw model with configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically partition weights across available GPUs
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    # Configure gradient checkpointing for memory conservation
    if use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing.")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # Must be False when using gradient checkpointing
        
    return model

def prepare_model_for_lora(
    model: PreTrainedModel,
    lora_config_dict: Dict[str, Any],
) -> PreTrainedModel:
    """
    Prepares and wraps a base model with LoRA parameters.
    """
    logger.info("Preparing model for LoRA adaptation...")
    
    # Check if the model is quantized
    is_quantized = getattr(model, "is_quantized", False) or hasattr(model, "quantization_method")
    if is_quantized:
        logger.info("Preparing quantized model for k-bit training.")
        model = prepare_model_for_kbit_training(model)

    # Initialize LoRA Config
    lora_config = LoraConfig(
        r=lora_config_dict.get("r", 16),
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
        bias=lora_config_dict.get("bias", "none"),
        task_type="CAUSAL_LM",
        modules_to_save=lora_config_dict.get("modules_to_save", None)
    )
    
    # Wrap model with PEFT wrapper
    model = get_peft_model(model, lora_config)
    
    # Log information about trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(
        "Trainable parameters: %d | All parameters: %d | Proportion Trainable: %.4f%%",
        trainable_params, all_param, 100 * trainable_params / all_param
    )
    
    return model
