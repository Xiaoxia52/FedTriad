"""Summarize completed non-smoke FedTriad runs without changing them."""
import argparse
from pathlib import Path

from fedtriad.suite import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/fedtriad_3datasets_300r_ablations")
    parser.add_argument("--output", default="runs/fedtriad_summary.csv")
    args = parser.parse_args()
    records = summarize(args.root, args.output)
    print("Wrote %s groups to %s" % (len(records), Path(args.output)), flush=True)


if __name__ == "__main__":
    main()
