"""
Dataset preparation for supervised fine-tuning.

The script normalizes common instruction/chat dataset schemas into:
{"messages": [{"role": "...", "content": "..."}]}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"system", "user", "assistant"}
MAX_MESSAGE_CHARS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and split an SFT dataset.")
    parser.add_argument("--dataset_name", required=True, help="HF dataset id or local JSON/JSONL path.")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--test_split_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _clean_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []

    cleaned: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content", "")
        if role not in ALLOWED_ROLES or not isinstance(content, str):
            continue
        content = content.strip()
        if not content or len(content) > MAX_MESSAGE_CHARS:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def convert_to_messages_format(example: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Convert common dataset formats into a validated messages list."""
    messages = _clean_messages(example.get("messages"))
    if messages:
        return {"messages": messages}

    text = example.get("text")
    if isinstance(text, str):
        parsed: list[dict[str, str]] = []
        for part in text.split("###"):
            item = part.strip()
            if item.startswith("Human:"):
                parsed.append({"role": "user", "content": item.removeprefix("Human:").strip()})
            elif item.startswith("Assistant:"):
                parsed.append(
                    {"role": "assistant", "content": item.removeprefix("Assistant:").strip()}
                )
        messages = _clean_messages(parsed)
        if messages:
            return {"messages": messages}

    prompt = example.get("instruction", example.get("prompt", example.get("question", "")))
    response = example.get("output", example.get("response", example.get("answer", "")))
    if isinstance(prompt, str) and isinstance(response, str):
        messages = _clean_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )
    return {"messages": messages}


def _load_examples(source: str) -> list[dict[str, Any]]:
    path = Path(source)
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else payload.get("data", [])
        raise ValueError("Local dataset must be .json or .jsonl")

    dataset = load_dataset(source)
    split = "train" if "train" in dataset else next(iter(dataset))
    return list(dataset[split])


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_split_size < 1.0:
        raise ValueError("--test_split_size must be between 0 and 1")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset: %s", args.dataset_name)
    raw_examples = _load_examples(args.dataset_name)
    processed = [
        formatted
        for item in raw_examples
        if (formatted := convert_to_messages_format(item))["messages"]
        and any(m["role"] == "assistant" for m in formatted["messages"])
    ]

    if len(processed) < 2:
        raise ValueError("At least two valid examples containing an assistant message are required")

    train_data, val_data = train_test_split(
        processed,
        test_size=args.test_split_size,
        random_state=args.seed,
        shuffle=True,
    )

    for name, rows in (("train", train_data), ("val", val_data)):
        target = output_dir / f"{name}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Wrote %d examples to %s", len(rows), target)


if __name__ == "__main__":
    main()
