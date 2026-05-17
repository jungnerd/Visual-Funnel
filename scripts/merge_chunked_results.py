#!/usr/bin/env python3
"""Merge chunked run.py result JSON files into standard full-result files."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_root", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model", default="qwen2_5")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def result_filename(model, task, method):
    return f"{model}-{task}-{method}.json"


def main():
    args = parse_args()
    chunk_root = Path(args.chunk_root)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        for method in args.methods:
            merged = []
            for chunk_id in range(args.total_chunks):
                chunk_dir = chunk_root / f"{task}_chunk{chunk_id}_of_{args.total_chunks}"
                chunk_file = chunk_dir / result_filename(args.model, task, method)
                if not chunk_file.exists():
                    raise FileNotFoundError(f"Missing chunk result: {chunk_file}")
                with chunk_file.open() as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"Expected list in {chunk_file}, got {type(data)}")
                merged.extend(data)

            out_file = save_path / result_filename(args.model, task, method)
            if out_file.exists() and not args.overwrite:
                raise FileExistsError(f"{out_file} exists; pass --overwrite to replace it")
            with out_file.open("w") as f:
                json.dump(merged, f, indent=4)
            print(f"{out_file}: {len(merged)} rows")


if __name__ == "__main__":
    main()
