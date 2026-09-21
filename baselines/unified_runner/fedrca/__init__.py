"""Unified MedMNIST federated baselines and the FedRCA research method."""
import os
import warnings


# Set these before scikit-learn is imported.  This avoids the documented
# Windows MKL KMeans leak path and keeps both the launcher and direct CLI quiet.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

warnings.filterwarnings(
    "ignore",
    message=r"adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"KMeans is known to have a memory leak on Windows with MKL.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Plan failed with a cudnnException.*",
    category=UserWarning,
)

__version__ = "0.2.0"
