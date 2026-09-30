"""
Validated SFT dataset with chat-template formatting and assistant-only loss masking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class SFTDataset(Dataset):
    """Load JSONL conversations and tokenize them lazily."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2048,
    ) -> None:
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be greater than zero")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict[str, Any]] = []

        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
                messages = item.get("messages") if isinstance(item, dict) else None
                if not isinstance(messages, list):
                    continue
                if not any(
                    isinstance(m, dict)
                    and m.get("role") == "assistant"
                    and isinstance(m.get("content"), str)
                    and m.get("content", "").strip()
                    for m in messages
                ):
                    continue
                self.examples.append({"messages": messages})

        if not self.examples:
            raise ValueError(f"No valid assistant-training examples found in {path}")

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must provide either pad_token or eos_token")
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.examples[idx]["messages"]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            raise ValueError(f"Tokenizer chat template failed for sample {idx}") from exc

        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        labels = torch.full_like(input_ids, -100)

        self._mask_assistant_spans(messages, input_ids, labels)
        labels[attention_mask == 0] = -100

        if not torch.any(labels != -100):
            raise ValueError(f"Sample {idx} contains no assistant tokens within max_seq_length")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_assistant_spans(
        self,
        messages: list[dict[str, Any]],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Mask using re-rendered assistant prefixes, avoiding model-name heuristics."""
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Tokenizer must provide a chat_template for assistant-only masking")

        ids = input_ids.tolist()
        cursor = 0

        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            prefix_messages = messages[:index] + [
                {"role": "assistant", "content": ""}
            ]
            try:
                prefix_text = self.tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                full_text = self.tokenizer.apply_chat_template(
                    messages[: index + 1],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                raise ValueError(f"Unable to render chat template for assistant turn {index}") from exc

            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
            if not prefix_ids or len(full_ids) <= len(prefix_ids):
                continue

            content_ids = full_ids[len(prefix_ids):]
            match_start = _find_subsequence(ids, content_ids, start=cursor)
            if match_start is None:
                continue

            end = min(match_start + len(content_ids), len(ids))
            labels[match_start:end] = input_ids[match_start:end]
            cursor = end


def _find_subsequence(sequence: list[int], needle: list[int], start: int = 0) -> int | None:
    if not needle:
        return None
    limit = len(sequence) - len(needle) + 1
    for index in range(max(0, start), max(0, limit)):
        if sequence[index:index + len(needle)] == needle:
            return index
    return None
