import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from info import model_to_fullname
from run import load_model_and_processor, run_visual_funnel


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description="Run a one-image Visual Funnel smoke test.")
    parser.add_argument("--model", choices=model_to_fullname.keys(), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, processor = load_model_and_processor(
        args.model,
        model_to_fullname[args.model],
        args.device,
    )
    result = run_visual_funnel(
        args.model,
        args.image,
        args.question,
        model,
        processor,
    )
    result.update(
        {
            "model": args.model,
            "image": args.image,
            "question": args.question,
        }
    )
    print(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
