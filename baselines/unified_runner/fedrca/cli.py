import argparse
from dataclasses import fields
import json
import os

from .config import Config, read_config


def _overrides(args, base):
    values = dict(base)
    for field in fields(Config):
        value = getattr(args, field.name, None)
        if value is not None:
            values[field.name] = value
    for item in getattr(args, "set_values", []) or []:
        if "=" not in item:
            raise ValueError("--set requires key=value")
        key, raw = item.split("=", 1)
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    return Config.from_dict(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description="FedRCA medical federated learning (CPU by default)")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run", help="Train/evaluate one method; no test-set model selection")
    train.add_argument("--config")
    train.add_argument("--resume", action="store_true", help="Load only your own trusted local checkpoint")
    train.add_argument("--set", dest="set_values", action="append", default=[])
    defaults = Config()
    for field in fields(Config):
        value = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            train.add_argument(flag, action="store_true", default=None)
        else:
            train.add_argument(flag, type=type(value), default=None)
    inspect = sub.add_parser("inspect", help="Read array headers only, without loading image data")
    inspect.add_argument("path")
    suite = sub.add_parser("suite", help="Preview an experiment matrix; --execute starts it")
    suite.add_argument("--config", required=True)
    suite.add_argument("--execute", action="store_true")
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--job", "--jobs", type=int, default=1,
                       help="Number of independent jobs to run concurrently")
    summary = sub.add_parser("summarize", help="Aggregate compatible completed seed runs")
    summary.add_argument("root")
    summary.add_argument("--output", default="summary.csv")
    summary.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        config = _overrides(args, read_config(args.config) if args.config else {})
        if config.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from .runner import run
        result = run(config, args.resume)
        print(json.dumps({"output_dir": config.output_dir, "smoke_only": result["smoke_only"],
                          "selected_round": result["selected_round"],
                          "local_test": result["local_test"]["aggregate"]}, ensure_ascii=False, indent=2))
    elif args.command == "inspect":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        from .data import inspect_npz
        print(json.dumps(inspect_npz(args.path), ensure_ascii=False, indent=2))
    elif args.command == "suite":
        from .suite import execute_suite
        execute_suite(read_config(args.config), args.execute, args.resume, args.job)
    else:
        from .suite import summarize
        summarize(args.root, args.output, args.include_smoke)


if __name__ == "__main__":
    main()
