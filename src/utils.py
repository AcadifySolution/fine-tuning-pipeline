"""
Utilities for reproducible and observable LLM fine-tuning runs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch and CUDA.

    Deterministic kernels can materially reduce throughput, so they are opt-in.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    logger.info("Setting global training seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_gpu_memory() -> None:
    """Log per-GPU allocated, reserved and peak memory."""
    if not torch.cuda.is_available():
        logger.info("CUDA is not available; skipping GPU memory metrics.")
        return

    for device_idx in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / 1024**3
        reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
        logger.info(
            "GPU[%d] allocated=%.2fGiB peak=%.2fGiB reserved=%.2fGiB",
            device_idx,
            allocated,
            max_allocated,
            reserved,
        )


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters and log the ratio."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / total if total else 0.0
    logger.info(
        "Model parameters: trainable=%d total=%d ratio=%.4f%%",
        trainable,
        total,
        ratio,
    )
    return trainable


def write_run_manifest(output_dir: str, manifest: dict[str, Any]) -> Path:
    """Persist a JSON manifest describing the training run."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest_path


def validate_positive_int(name: str, value: int) -> int:
    """Validate CLI integer settings that must be strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
