"""
Merge a LoRA adapter into an unquantized base model.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model.")
    parser.add_argument("--base_model_name", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.push_to_hub and not args.hub_repo_id:
        raise ValueError("--hub_repo_id is required with --push_to_hub")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    dtype = torch.float32 if args.device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    logger.info("Loading base model %s on %s", args.base_model_name, args.device)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=False)
    adapter_path = Path(args.adapter_dir)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    merged = model.merge_and_unload()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, max_shard_size="5GB", safe_serialization=True)
    tokenizer.save_pretrained(output)

    if args.push_to_hub:
        logger.info("Pushing merged model to %s", args.hub_repo_id)
        merged.push_to_hub(args.hub_repo_id, safe_serialization=True)
        tokenizer.push_to_hub(args.hub_repo_id)

    logger.info("Merged model written to %s", output)


if __name__ == "__main__":
    main()
