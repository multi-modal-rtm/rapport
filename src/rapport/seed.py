import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Required by torch's deterministic matmul/cuBLAS kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
