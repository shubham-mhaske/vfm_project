import torch
import platform
import os

IS_MAC = platform.system() == "Darwin"

def get_device(force_cpu=False):
    """
    Selects the appropriate device (MPS, CUDA, or CPU) based on availability.
    
    Args:
        force_cpu: If True, always return CPU device regardless of GPU availability
    """
    if force_cpu:
        print("Forcing CPU usage as requested.")
        return torch.device("cpu")
    
    if IS_MAC:
        if torch.backends.mps.is_available():
            print("Using Apple MPS backend.")
            return torch.device("mps")
        else:
            print("No MPS found. Using CPU.")
            return torch.device("cpu")
    elif torch.cuda.is_available():
        print("Using Nvidia CUDA backend.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")

def get_num_workers():
    """
    Determines the optimal number of workers for DataLoader.
    """
    if IS_MAC:
        return 0  # or 1, for PyTorch dataloader
    else:
        # Get the number of CPUs allocated by SLURM, default to 4 if not set
        return int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

device = get_device()
