"""
Dataset Preprocessing and Preparation Script.
Loads raw text/instruct datasets, formats them to the required multi-turn schema,
and partitions them into JSONL splits for SFT training.
"""

import os
import argparse
import logging
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and partition dataset for SFT pipeline.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="timdettmers/openassistant-guanaco",
        help="Hugging Face dataset name or path to local CSV/JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed train/validation JSONL splits."
    )
    parser.add_argument(
        "--test_split_size",
        type=float,
        default=0.1,
        help="Fraction of data to allocate for validation/testing."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val partition."
    )
    return parser.parse_args()

def convert_to_messages_format(example: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """
    Standardizes typical QA dataset items into a conversational messages schema.
    Customize this function based on the target raw dataset schema.
    """
    # Example conversion logic for guanaco-like single-prompt-response formats:
    # "### Human: ... ### Assistant: ..."
    text = example.get("text", "")
    messages = []
    
    parts = text.split("###")
    for part in parts:
        part = part.strip()
        if part.startswith("Human:"):
            content = part.replace("Human:", "").strip()
            messages.append({"role": "user", "content": content})
        elif part.startswith("Assistant:"):
            content = part.replace("Assistant:", "").strip()
            messages.append({"role": "assistant", "content": content})
            
    # Fallback to standard QA fields if custom format is missing
    if not messages:
        prompt = example.get("instruction", example.get("prompt", example.get("question", "")))
        response = example.get("output", example.get("response", example.get("answer", "")))
        if prompt and response:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
    return {"messages": messages}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Loading raw dataset: %s", args.dataset_name)
    try:
        # Load dataset from Hugging Face Datasets Hub
        raw_dataset = load_dataset(args.dataset_name)
        # Select the 'train' split if present, or work with the main list
        split_name = "train" if "train" in raw_dataset else list(raw_dataset.keys())[0]
        data_list = list(raw_dataset[split_name])
    except Exception as e:
        logger.error("Failed to load Hugging Face dataset. Attempting local file read... %s", e)
        if os.path.exists(args.dataset_name):
            if args.dataset_name.endswith(".json"):
                import json
                with open(args.dataset_name, "r") as f:
                    data_list = json.load(f)
            else:
                raise ValueError("Unsupported local file format. Please provide a JSON dataset.")
        else:
            raise FileNotFoundError(f"Could not load local or remote dataset: {args.dataset_name}")

    logger.info("Loaded %d raw examples.", len(data_list))
    
    # Process examples
    processed_examples = []
    for item in data_list:
        formatted = convert_to_messages_format(item)
        if formatted["messages"]:  # Only keep non-empty conversations
            processed_examples.append(formatted)
            
    logger.info("Formatted %d conversations successfully.", len(processed_examples))

    # Split into train and validation sets
    train_data, val_data = train_test_split(
        processed_examples,
        test_size=args.test_split_size,
        random_state=args.seed
    )
    
    logger.info("Splitting dataset into %d training and %d validation items.", len(train_data), len(val_data))

    # Save to JSONL splits
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
            
    logger.info("Dataset preprocessing complete. Files written to: \n  - Train: %s\n  - Validation: %s", train_path, val_path)

if __name__ == "__main__":
    main()
