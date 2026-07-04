"""
Custom PyTorch Dataset class for Supervised Fine-Tuning (SFT) of LLMs.
Implements tokenization, formatting with Chat Templates, and attention/label masking.
"""

import json
from typing import Dict, List, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SFTDataset(Dataset):
    """
    Dataset class that prepares conversational data for supervised fine-tuning.
    Formats data using the tokenizer's chat template and mask inputs to ensure
    loss is only calculated on the assistant's responses (not the user prompts).
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
    ):
        """
        Args:
            jsonl_path (str): Path to the JSON Lines data file.
            tokenizer (PreTrainedTokenizer): Tokenizer matching the target model.
            max_seq_length (int): Maximum sequence length for padding/truncation.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples = []
        
        # Load samples from JSONL
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        # Check if tokenizer has padding token defined, else configure it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Processes a single conversation example, tokenizes it, and returns inputs/labels.
        Supports multi-turn messages structure:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        example = self.examples[idx]
        messages = example.get("messages", [])

        # Apply Chat Template (e.g., Llama 3 jinja template)
        # We don't tokenize yet because we want to calculate label masks for loss
        formatted_chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the complete conversation
        tokenized = self.tokenizer(
            formatted_chat,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Clone labels from input_ids. We will mask out prompt text with -100
        labels = input_ids.clone()

        # Label Masking: Only calculate loss on assistant turns.
        # We find assistant sections by tokenizing segments.
        self._mask_user_prompts(messages, labels)

        # Set pad token labels to -100 as well
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _mask_user_prompts(self, messages: List[Dict[str, str]], labels: torch.Tensor):
        """
        Calculates offsets and masks user roles, system rules, and prompts, setting their label indexes to -100.
        """
        # Start matching tokens
        curr_idx = 0
        
        # We loop through messages and calculate token indices to construct the label mask
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Tokenize individual message formatted as its own turn to see its length
            # Note: Depending on tokenizer jinja settings, we construct exact tokens.
            # For robustness, we reconstruct up to the target message segment.
            segment_chat = self.tokenizer.apply_chat_template(
                messages[:i+1],
                tokenize=False,
                add_generation_prompt=False
            )
            tokenized_segment = self.tokenizer(
                segment_chat,
                add_special_tokens=False,
                return_tensors="pt"
            )
            segment_len = tokenized_segment["input_ids"].squeeze(0).size(0)

            # If the current message role is user or system, mask out these tokens
            if role in ["user", "system"]:
                end_idx = min(segment_len, self.max_seq_length)
                if curr_idx < end_idx:
                    labels[curr_idx:end_idx] = -100
            
            curr_idx = min(segment_len, self.max_seq_length)
            if curr_idx >= self.max_seq_length:
                break
