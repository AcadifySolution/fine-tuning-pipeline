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
        
        # Mask out all tokens initially
        labels.fill_(-100)

        # Label Masking: Only calculate loss on assistant turns.
        self._mask_user_prompts(input_ids, labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _mask_user_prompts(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Locates assistant response segments in input_ids and unmasks only those tokens (plus EOT/EOS) in labels.
        Uses a robust token-matching search strategy to avoid BPE offset discrepancies.
        """
        # Determine the response template and the end of turn token based on tokenizer
        tokenizer_name = self.tokenizer.name_or_path.lower() if self.tokenizer.name_or_path else ""
        
        if "llama-3" in tokenizer_name or "llama3" in tokenizer_name:
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            eot_token = "<|eot_id|>"
        elif "llama" in tokenizer_name:
            response_template = "[/INST]"
            eot_token = "</s>"
        elif "qwen" in tokenizer_name or "chatml" in tokenizer_name:
            response_template = "<|im_start|>assistant\n"
            eot_token = "<|im_end|>"
        else:
            # Fallback default
            response_template = "### Assistant:"
            eot_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"

        # Encode the templates to token IDs
        # We use add_special_tokens=False to prevent prepending BOS tokens inside the sequence
        response_tokens = self.tokenizer.encode(response_template, add_special_tokens=False)
        
        if isinstance(eot_token, int):
            eot_token_id = eot_token
        else:
            eot_ids = self.tokenizer.encode(eot_token, add_special_tokens=False)
            eot_token_id = eot_ids[-1] if eot_ids else self.tokenizer.eos_token_id

        input_ids_list = input_ids.tolist()
        n = len(input_ids_list)
        m = len(response_tokens)
        
        if m == 0:
            # If template cannot be encoded, fallback to full sequence training (standard SFT)
            labels.copy_(input_ids)
            # Still mask out pad tokens
            labels[input_ids == self.tokenizer.pad_token_id] = -100
            return

        # Find all occurrences of response_tokens in input_ids
        match_indices = []
        for i in range(n - m + 1):
            if input_ids_list[i : i + m] == response_tokens:
                match_indices.append(i)

        # For each matched assistant response, unmask the assistant's generation tokens
        # from the end of response_tokens up to the corresponding EOT/EOS token.
        for start_idx in match_indices:
            response_start = start_idx + m
            
            # Find the next EOT or EOS token
            response_end = n
            for j in range(response_start, n):
                if input_ids_list[j] == eot_token_id or input_ids_list[j] == self.tokenizer.eos_token_id:
                    response_end = j + 1  # Include the EOT/EOS token in loss calculation
                    break
            
            # Unmask this range in labels
            if response_start < response_end:
                labels[response_start:response_end] = input_ids[response_start:response_end]

