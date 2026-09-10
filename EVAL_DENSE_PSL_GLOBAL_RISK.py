import argparse
import os
from pathlib import Path
import warnings

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
warnings.filterwarnings("ignore", message=r"Plan failed with a cudnnException.*")
warnings.filterwarnings(
    "ignore", message=r"Please use the new API settings to control TF32 behavior.*"
)

from fedtriad.dense_psl_risk import evaluate_root


def main():
    parser = argparse.ArgumentParser(
        description="Offline final risk ablations for completed dense PSL checkpoints"
    )
    parser.add_argument(
        "--root", default="runs/fedtriad_3datasets_300r_ablations"
    )
    parser.add_argument(
        "--output", default="output/final-global-class-risk"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["bloodmnist", "organamnist", "pathmnist"]
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = evaluate_root(
        Path(args.root), Path(args.output), tuple(args.datasets), args.device, args.force
    )
    print("FedTriad final risk evaluation: " + str(manifest), flush=True)


if __name__ == "__main__":
    main()
