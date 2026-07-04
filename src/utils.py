"""
Utility functions for the fine-tuning pipeline.
Includes logging, seeding, memory profiling, and metrics calculations.
"""

import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Sets the random seed across Python, NumPy, PyTorch, and CUDA to ensure training runs
    are fully reproducible and deterministic.
    """
    logger.info("Setting global training seed to: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic algorithms if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_gpu_memory():
    """
    Profiles current GPU memory usage and logs max allocated and reserved VRAM.
    Helps diagnose OOM (Out Of Memory) issues in distributed architectures.
    """
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            # Get statistics
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
            
            logger.info(
                "GPU [%d] Memory Status: Allocated: %.2f GB | Peak Allocated: %.2f GB | Reserved: %.2f GB",
                device_idx, allocated, max_allocated, reserved
            )
    else:
        logger.info("CUDA is not available. GPU Memory status log skipped.")

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Counts and prints statistics of trainable weights vs frozen base model parameters.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    logger.info(
        "Model Parameter Summary: Trainable Params: %d | Total Params: %d | Ratio: %.4f%%",
        trainable_params, all_params, (100.0 * trainable_params / all_params) if all_params > 0 else 0.0
    )
    return trainable_params
