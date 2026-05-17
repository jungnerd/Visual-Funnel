import argparse
import json
import random
from pathlib import Path


def convert_textvqa_val(input_path, output_path, limit=None, seed=0, shuffle=False):
    with open(input_path, "r") as f:
        raw = json.load(f)

    rows = []
    for data_id, item in enumerate(raw["data"]):
        rows.append({
            "id": str(data_id).zfill(10),
            "question": item["question"],
            "labels": item["answers"],
            "image_path": f"{item['image_id']}.jpg",
        })

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if limit is not None:
        rows = rows[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=4)

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert TextVQA val JSON to this repo's benchmark format.")
    parser.add_argument("--input", default="data/textvqa/TextVQA_0.5.1_val.json")
    parser.add_argument("--output", default="data/textvqa/data.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    count = convert_textvqa_val(
        input_path=args.input,
        output_path=args.output,
        limit=limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    print(f"Wrote {count} examples to {args.output}")


if __name__ == "__main__":
    main()
