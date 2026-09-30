"""
Production-oriented SFT training entry point.

Supports Hugging Face Trainer (default) and an Accelerate custom loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, get_scheduler

from data.dataset import SFTDataset
from src.model import load_model, load_tokenizer, prepare_model_for_lora
from src.utils import count_trainable_parameters, log_gpu_memory, set_seed, write_run_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal language model with LoRA/QLoRA.")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--lora_config", default="configs/lora_config.yaml")
    parser.add_argument("--deepspeed_config")
    parser.add_argument("--output_dir", default="checkpoints/sft_model")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--run_custom_loop", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_only_model", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = [
        ("num_epochs", args.num_epochs),
        ("per_device_train_batch_size", args.per_device_train_batch_size),
        ("per_device_eval_batch_size", args.per_device_eval_batch_size),
        ("gradient_accumulation_steps", args.gradient_accumulation_steps),
        ("max_seq_length", args.max_seq_length),
    ]
    for name, value in positive:
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def _precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = torch.cuda.is_bf16_supported()
    return bf16, not bf16


def _build_collator(tokenizer, model, max_seq_length: int):
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=max_seq_length,
        label_pad_token_id=-100,
    )


def run_hf_trainer(args, model, tokenizer, train_dataset, val_dataset):
    bf16, fp16 = _precision_flags()
    training_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=bf16,
        fp16=fp16,
        deepspeed=args.deepspeed_config,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_safetensors=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=_build_collator(tokenizer, model, args.max_seq_length),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    return trainer


def run_custom_accelerator_loop(args, model, tokenizer, train_dataset, val_dataset):
    accelerator = Accelerator(
        mixed_precision="bf16" if _precision_flags()[0] else ("fp16" if torch.cuda.is_available() else "no"),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    collator = _build_collator(tokenizer, model, args.max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=max(1, int(0.03 * total_steps)),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        update_count = 0
        progress = tqdm(total=steps_per_epoch, disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                running_loss += float(loss.detach().item())
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                update_count += 1
                progress.update(1)
                if global_step % 10 == 0 and accelerator.is_local_main_process:
                    logger.info(
                        "epoch=%d step=%d train_loss=%.4f lr=%.6e",
                        epoch + 1,
                        global_step,
                        running_loss / max(1, step + 1),
                        scheduler.get_last_lr()[0],
                    )
                    log_gpu_memory()

        progress.close()

        model.eval()
        val_total = torch.tensor(0.0, device=accelerator.device)
        val_batches = torch.tensor(0.0, device=accelerator.device)
        for batch in val_loader:
            with torch.no_grad():
                val_total += model(**batch).loss.detach()
                val_batches += 1

        gathered = accelerator.gather_for_metrics(torch.stack([val_total, val_batches]))
        val_loss = gathered[0].sum().item() / max(1.0, gathered[1].sum().item())

        if accelerator.is_local_main_process:
            logger.info("epoch=%d val_loss=%.4f", epoch + 1, val_loss)
            epoch_dir = Path(args.output_dir) / f"epoch_{epoch + 1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                epoch_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(epoch_dir)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(
            final_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(final_dir)


def main():
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed, deterministic=args.deterministic)

    with open(args.lora_config, "r", encoding="utf-8") as handle:
        lora_config = yaml.safe_load(handle) or {}

    tokenizer = load_tokenizer(args.model_id)
    model = load_model(
        args.model_id,
        quantization_config=lora_config.get("quantization"),
        use_gradient_checkpointing=True,
    )
    model = prepare_model_for_lora(model, lora_config)
    count_trainable_parameters(model)

    train_dataset = SFTDataset(args.train_path, tokenizer, args.max_seq_length)
    val_dataset = SFTDataset(args.val_path, tokenizer, args.max_seq_length)

    manifest = {
        "model_id": args.model_id,
        "train_path": os.path.abspath(args.train_path),
        "val_path": os.path.abspath(args.val_path),
        "lora_config": os.path.abspath(args.lora_config),
        "output_dir": os.path.abspath(args.output_dir),
        "seed": args.seed,
        "deterministic": args.deterministic,
        "max_seq_length": args.max_seq_length,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
    }
    write_run_manifest(args.output_dir, manifest)

    if args.run_custom_loop:
        run_custom_accelerator_loop(args, model, tokenizer, train_dataset, val_dataset)
    else:
        run_hf_trainer(args, model, tokenizer, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
