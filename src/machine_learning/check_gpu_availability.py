import subprocess
import shutil


def is_cuda_available():
    # Default to using CPU models and set gpu_available to False
    gpu_available = False

    # Check if nvidia-smi is available on the system
    if shutil.which('nvidia-smi'):
        try:
            # Run the nvidia-smi command to check for an NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            gpu_available = True
        except subprocess.CalledProcessError as e:
            # Print the error message and continue execution
            print(f"GPU acceleration is unavailable: {e}. Defaulting to CPU models.")
    else:
        print("nvidia-smi command not found. Assuming no NVIDIA GPU.")

    # If GPU is available, import the GPU-specific models
    if gpu_available:
        try:
            from cuml.cluster import KMeans
            from cuml.linear_model import LogisticRegression
            from src.machine_learning.gpu.ml import KMeansGPU, LogisticRegressionGPU
        except ImportError as e:
            print(f"Error importing GPU models: {e}. Defaulting to CPU models.")
            gpu_available = False  # If import fails, fall back to CPU

    return gpu_available