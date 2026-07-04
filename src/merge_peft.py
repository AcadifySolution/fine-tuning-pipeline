"""
Script to merge trained LoRA adapter weights back into the base model.
Enables high-performance serving by compiling adapters and base model into a single weight check.
"""

import os
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model.")
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Path or Hugging Face model identifier of the original base model."
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the directory containing the trained PEFT/LoRA adapter weights."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the full merged model will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on ('cpu' or 'cuda'). CPU is recommended to avoid GPU VRAM limits."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, pushes the merged model to the Hugging Face Hub."
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Hugging Face repo identifier if pushing to Hub (e.g. 'acadify-solution/llama-3-sft')."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Loading base model: %s on device: %s", args.base_model_name, args.device)
    
    # We load base model in FP16 or BF16 (matching the target precision)
    # Quantized models CANNOT be merged directly; hence we load the unquantized base model.
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map=args.device,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    logger.info("Loading tokenizer matching base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    
    logger.info("Loading PEFT model with adapter from: %s", args.adapter_dir)
    # Wrap base model with adapter layers
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        device_map=args.device,
        torch_dtype=torch_dtype
    )
    
    logger.info("Merging LoRA adapters into the base model weights...")
    # This modifies the base model in-place and removes adapter layers
    merged_model = model.merge_and_unload()
    
    logger.info("Saving merged model and tokenizer to: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Model merge successfully completed!")

    if args.push_to_hub:
        if not args.hub_repo_id:
            raise ValueError("Must provide --hub_repo_id when --push_to_hub is active.")
        logger.info("Pushing merged model to Hugging Face Hub repo: %s", args.hub_repo_id)
        merged_model.push_to_hub(args.hub_repo_id)
        tokenizer.push_to_hub(args.hub_repo_id)
        logger.info("Successfully uploaded merged model to Hugging Face Hub.")

if __name__ == "__main__":
    main()
