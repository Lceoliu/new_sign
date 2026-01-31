"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(rank: int = 0, world_size: int = 1, backend: str = "nccl") -> None:
    """Setup distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend to use (nccl, gloo, mpi)
    """
    if world_size > 1:
        # Initialize process group
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )

        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_tensor(tensor: torch.Tensor, world_size: Optional[int] = None) -> torch.Tensor:
    """Reduce tensor across all processes.

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes (auto-detected if None)

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    if world_size is None:
        world_size = get_world_size()

    # Clone tensor to avoid modifying original
    reduced_tensor = tensor.clone()

    # All-reduce operation
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

    # Average across processes
    reduced_tensor /= world_size

    return reduced_tensor