"""
Fine-tuning execution script.
Supports distributed training via Accelerate/DeepSpeed and custom PyTorch / HF Trainer pipelines.
"""

import os
import sys
import argparse
import yaml
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler
)
from accelerate import Accelerator

# Add repository root to path for absolute imports in SageMaker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SFTDataset
from src.model import load_tokenizer, load_model, prepare_model_for_lora
from src.utils import set_seed, log_gpu_memory, count_trainable_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune open-source LLMs.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Hugging Face model identifier.")
    parser.add_argument("--train_path", type=str, required=True, help="Path to SFT train JSONL dataset.")
    parser.add_argument("--val_path", type=str, required=True, help="Path to SFT validation JSONL dataset.")
    parser.add_argument("--lora_config", type=str, default="configs/lora_config.yaml", help="Path to LoRA YAML config file.")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed JSON configuration file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_model", help="Directory to save checkpoint models.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate during fine-tuning.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Micro-batch size per device for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Micro-batch size per device for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Steps before applying backward pass and optimizer step.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length configuration for tokenizer.")
    parser.add_argument("--run_custom_loop", action="store_true", help="If True, runs custom PyTorch loop with Accelerate. If False, runs Hugging Face Trainer.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic training seed.")
    
    return parser.parse_args()

def run_hf_trainer(args, model, tokenizer, train_dataset, val_dataset):
    """
    Orchestrates training using Hugging Face Trainer API.
    """
    logger.info("Starting SFT training using Hugging Face Trainer...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        deepspeed=args.deepspeed_config,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        logging_dir=os.path.join(args.output_dir, "logs"),
        disable_tqdm=False
    )

    # Use data collator designed for sequence-to-sequence generation (which also handles label padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=args.max_seq_length,
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    
    # Save the final adapter model weights
    logger.info("Training complete. Saving final LoRA adapter...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def run_custom_accelerator_loop(args, model, tokenizer, train_dataset, val_dataset):
    """
    Orchestrates training using a custom PyTorch loop wrapped in Accelerate for full control.
    """
    logger.info("Starting training using Custom PyTorch + Accelerate loop...")
    
    # Initialize the Accelerator
    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Create DataLoaders
    # Custom SFTDataset handles formatting, so we can use standard collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=args.max_seq_length,
        label_pad_token_id=-100
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    total_training_steps = args.num_epochs * num_update_steps_per_epoch

    # Setup Scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.03 * total_training_steps),
        num_training_steps=total_training_steps
    )

    # Prepare for distributed execution
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    logger.info("***** Running training loop *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Total optimization steps = {total_training_steps}")
    
    # Track metrics
    progress_bar = tqdm(range(total_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Check if gradient updates happened
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
                if completed_steps % 10 == 0 and accelerator.is_local_main_process:
                    current_loss = total_loss.item() / (step + 1)
                    logger.info(f"Epoch {epoch} | Step {completed_steps} | Loss: {current_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6e}")
                    log_gpu_memory()

        # Run validation per epoch
        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.detach().float()
                
        # Gather losses across all GPUs in multi-GPU distributed settings
        val_loss = accelerator.gather(val_loss).mean().item() / len(val_dataloader)
        if accelerator.is_local_main_process:
            logger.info(f"--- Epoch {epoch} Validation Loss: {val_loss:.4f} ---")
            
            # Save Checkpoint
            epoch_output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_output_dir, exist_ok=True)
            
            # Unwrap model for standard state_dict saving
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                epoch_output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            tokenizer.save_pretrained(epoch_output_dir)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load LoRA configuration
    with open(args.lora_config, "r") as f:
        lora_config_dict = yaml.safe_load(f)

    # 1. Load Tokenizer & Model
    tokenizer = load_tokenizer(args.model_id)
    model = load_model(
        model_id=args.model_id,
        quantization_config=lora_config_dict.get("quantization"),
        use_gradient_checkpointing=True
    )
    
    # 2. Prepare Model for LoRA (Quantization and Adapter wrapper)
    model = prepare_model_for_lora(model, lora_config_dict)
    
    # Print architecture metrics
    count_trainable_parameters(model)

    # 3. Load Datasets
    logger.info("Initializing SFT Train Dataset...")
    train_dataset = SFTDataset(args.train_path, tokenizer, args.max_seq_length)
    logger.info("Initializing SFT Validation Dataset...")
    val_dataset = SFTDataset(args.val_path, tokenizer, args.max_seq_length)

    # 4. Trigger Training Loop
    if args.run_custom_loop:
        run_custom_accelerator_loop(args, model, tokenizer, train_dataset, val_dataset)
    else:
        run_hf_trainer(args, model, tokenizer, train_dataset, val_dataset)

if __name__ == "__main__":
    main()
