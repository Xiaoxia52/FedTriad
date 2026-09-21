"""FedRCA command-line entry point."""
import os


# Configure numerical libraries before importing the training package.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fedrca.cli import main


if __name__ == "__main__":
    main()
