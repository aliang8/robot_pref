import torch
import numpy as np
import random
import os


def set_seed(seed=42, deterministic_cudnn=True):
    """Set random seed for reproducibility across multiple libraries.

    Args:
        seed: Integer seed for random number generators
        deterministic_cudnn: If True, set CuDNN to use deterministic algorithms
            (may impact performance but ensures reproducibility)

    Returns:
        The seed used (useful when seed=None to know what seed was randomly chosen)
    """
    # If no seed is provided, generate one
    if seed is None:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        print(f"No seed provided, using randomly generated seed: {seed}")

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)

    # Set CuDNN to deterministic mode if requested and CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("CUDA is available: Set CuDNN to deterministic mode")
        else:
            torch.backends.cudnn.benchmark = True
            print("CUDA is available: Using CuDNN benchmark for better performance")

    # Set environment variable for potential subprocesses
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Random seed set to {seed}")
    return seed
