import logging
import shutil
import subprocess

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@staticmethod
def is_cuda_available():
    """
    Checks if CUDA (GPU support) is available.
    """
        
    gpu_available = False
    if shutil.which('nvidia-smi'):
        try:
            subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            gpu_available = True
        except subprocess.CalledProcessError as e:
            LOGGER.debug("GPU acceleration is unavailable. Defaulting to CPU models.")
    else:
        LOGGER.debug("nvidia-smi command not found. Assuming no NVIDIA GPU.")
    return gpu_available