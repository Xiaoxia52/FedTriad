"""Run the protocol-matched CWT and FedSeq supplementary baselines."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import warnings


for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent
CALLER_DIRECTORY = Path.cwd()
os.chdir(PROJECT_ROOT)
warnings.filterwarnings("ignore", message=r"Plan failed with a cudnnException.*")
warnings.filterwarnings(
    "ignore", message=r"Please use the new API settings to control TF32 behavior.*"
)

from fedrca.config import read_config
from fedrca.suite import execute_suite, expand_suite, summarize


SAFE_MAX_GPU_JOBS = 8


def _process_is_running(process_id):
    if process_id is None or process_id <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(process_id, 0)
        except (ProcessLookupError, OSError):
            return False
        except PermissionError:
            return True
        return True
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, process_id)
    if not handle:
        return ctypes.get_last_error() == 5
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
    finally:
        kernel32.CloseHandle(handle)


def _prepare_resume_outputs(configs):
    stale_locks = transient_shells = resumable = complete = 0
    active = []
    for config in configs:
        output = Path(config.output_dir)
        if not output.exists():
            continue
        result_path = output / "results.json"
        if result_path.is_file():
            try:
                if json.loads(result_path.read_text(encoding="utf-8")).get("status") == "complete":
                    complete += 1
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
        if (output / "last.pt").is_file():
            (output / "failure.log").unlink(missing_ok=True)
            resumable += 1
            continue
        entries = {path.name for path in output.iterdir()}
        if entries and entries <= {"failure.log"}:
            shutil.rmtree(output)
            transient_shells += 1
    if active:
        details = ", ".join("%s (pid=%s)" % item for item in active)
        raise RuntimeError("Refusing duplicate resume; jobs are active: " + details)
    print(
        "Resume preflight: complete=%s; resumable=%s; stale_locks_removed=%s; "
        "transient_shells_removed=%s" %
        (complete, resumable, stale_locks, transient_shells), flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CWT and FedSeq under the frozen three-dataset 300-round protocol."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/serial_baselines_3datasets_300r.json"))
    parser.add_argument("--data-dir", type=Path,
                        help="Directory containing the three official MedMNIST NPZ files")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--job", "--jobs", type=int, default=8)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser()
    args.config = str((CALLER_DIRECTORY / config_path).resolve())
    if args.data_dir is not None:
        args.data_dir = (CALLER_DIRECTORY / args.data_dir.expanduser()).resolve()
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
    print(
        "Serial baseline suite: %s jobs; methods=%s; seeds=%s; parallel=%s" %
        (len(configs), spec["algorithms"], spec["seeds"], args.job), flush=True,
    )
    if args.preview:
        for config in configs:
            result = Path(config.output_dir) / "results.json"
            complete = False
            if result.is_file():
                try:
                    complete = json.loads(result.read_text(encoding="utf-8")).get("status") == "complete"
                except (OSError, ValueError):
                    pass
            print(("SKIP " if complete else "RUN  ") + config.output_dir)
        return

    if args.resume:
        _prepare_resume_outputs(configs)
    execute_suite(
        spec, execute=True, resume=args.resume, jobs=args.job, skip_complete=True
    )
    summarize(
        Path(spec["output_root"]), Path(spec["output_root"]) / "summary.csv"
    )
    print("Serial baseline pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
