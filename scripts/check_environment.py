
import sys
import platform

import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
import requests

def main():
    print("=" * 55)
    print("FINANCIAL COMPLAINT INTELLIGENCE")
    print("ENVIRONMENT VERIFICATION")
    print("=" * 55)

    print("\n[1] Python")
    print("Version:", sys.version.split()[0])
    print("Platform:", platform.platform())

    print("\n[2] Installed Libraries")
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("Requests:", requests.__version__)

    print("\n[3] Compute Device")

    if torch.cuda.is_available():
        device = torch.device("cuda")

        print("CUDA available: Yes")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

    else:
        device = torch.device("cpu")

        print("CUDA available: No")
        print("Using CPU.")

    print("Selected device:", device)

    print("\n[4] PyTorch Tensor Test")

    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = tensor * 2

    print("Input tensor:", tensor)
    print("Output tensor:", result)

    assert torch.equal(
        result,
        torch.tensor([2.0, 4.0, 6.0])
    )

    print("\n[5] Dependency Import Test")

    print("All required libraries imported successfully.")

    print("\n" + "=" * 55)
    print("ENVIRONMENT VERIFICATION SUCCESSFUL")
    print("=" * 55)


if __name__ == "__main__":
    main()
