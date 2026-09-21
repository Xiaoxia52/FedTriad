"""Run the unified three-dataset comparison suite."""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import warnings


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

from fedrca.config import read_config
from fedrca.suite import execute_suite, expand_suite

SAFE_MAX_GPU_JOBS = 8
PROJECT_ROOT = Path(__file__).resolve().parent


def _process_is_running(process_id):
    """Return whether a PID is alive without using broken os.kill(pid, 0) on Windows."""
    if process_id is None or process_id <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    synchronize = 0x00100000
    wait_timeout = 0x00000102
    handle = kernel32.OpenProcess(synchronize, False, process_id)
    if not handle:
        # Access denied still proves that the process exists.
        return ctypes.get_last_error() == 5
    try:
        return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
    finally:
        kernel32.CloseHandle(handle)


def _prepare_resume_outputs(configs):
    """Clean only verified transient crash artifacts before suite resume.

    This lives in the launcher rather than the scientific package so repairing
    orchestration does not change the implementation hash stored in checkpoints.
    """
    stale_locks = 0
    transient_shells = 0
    resumable = 0
    active = []
    for config in configs:
        output = Path(config.output_dir)
        if not output.exists():
            continue

        result_path = output / "results.json"
        if result_path.is_file():
            try:
                import json
                if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                    continue
            except (OSError, ValueError):
                pass

        lock = output / "run.lock"
        if lock.exists():
            try:
                process_id = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                process_id = None
            if _process_is_running(process_id):
                active.append((str(output), process_id))
                continue
            lock.unlink(missing_ok=True)
            stale_locks += 1

        checkpoint = output / "last.pt"
        if checkpoint.is_file():
            # A previous failure report is stale once this checkpoint is resumed.
            (output / "failure.log").unlink(missing_ok=True)
            resumable += 1
            continue

        entries = {path.name for path in output.iterdir()}
        if entries and entries <= {"failure.log"}:
            shutil.rmtree(output)
            transient_shells += 1

    if active:
        details = ", ".join("%s (pid=%s)" % item for item in active)
        raise RuntimeError("Refusing duplicate resume; jobs are still active: " + details)
    print(
        "Resume preflight: resumable=%s; stale_locks_removed=%s; transient_shells_removed=%s"
        % (resumable, stale_locks, transient_shells),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified 28x28 medical FL comparison (3 datasets, 11 methods, 2 alphas)."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/main_3datasets_11methods.json"))
    parser.add_argument("--data-dir", type=Path,
                        help="Directory containing the three official MedMNIST NPZ files")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--job", "--jobs", type=int, default=1,
                        help="Concurrent GPU experiments (allowed range: 1-8).")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # Resolve user paths before anchoring config-relative outputs and caches.
    args.config = str(Path(args.config).expanduser().resolve())
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
    os.chdir(PROJECT_ROOT)
    if not 1 <= args.job <= SAFE_MAX_GPU_JOBS:
        parser.error("--job must be between 1 and 8")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    spec = read_config(args.config)
    spec["seeds"] = sorted(set(args.seeds))
    if args.data_dir is not None:
        data_root = str(args.data_dir.resolve())
        for dataset in spec["datasets"]:
            spec["datasets"][dataset] = {"data_file": "", "data_root": data_root}
    configs = expand_suite(spec)
    count = len(configs)
    print("Unified suite: %s jobs; seeds=%s; parallel=%s" % (
        count, spec["seeds"], args.job
    ), flush=True)
    if args.preview:
        print("datasets=%s" % ", ".join(spec["datasets"]), flush=True)
        print("methods=%s" % ", ".join(spec["algorithms"]), flush=True)
        print("alphas=%s" % spec["alphas"], flush=True)
        return
    if args.resume:
        _prepare_resume_outputs(configs)
    execute_suite(spec, execute=not args.preview, resume=args.resume, jobs=args.job)


if __name__ == "__main__":
    main()
