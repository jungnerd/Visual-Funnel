import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def prepare_gqa_subset(dataset_name, split, output_dir, limit, seed, shuffle, streaming, qa_per_image):
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    if shuffle and streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=1000)
    elif shuffle:
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        dataset = dataset.select(indices)

    rows = []
    for image_idx, item in enumerate(dataset):
        image = item["image"].convert("RGB")

        if "qa" in item:
            qa_items = item["qa"][:qa_per_image]
        else:
            qa_items = [{"question": item["question"], "answer": item["answer"]}]

        for qa_idx, qa in enumerate(qa_items):
            question_id = str(item.get("question_id", item.get("id", f"gqa_{image_idx:06d}_{qa_idx:02d}")))
            if len(qa_items) > 1:
                question_id = f"{question_id}_{qa_idx:02d}"
            image_path = f"{question_id}.jpg"

            image.save(image_dir / image_path)

            rows.append({
                "id": question_id,
                "question": qa["question"],
                "labels": [qa["answer"]],
                "image_path": image_path,
            })

            if limit is not None and len(rows) >= limit:
                break

        if limit is not None and len(rows) >= limit:
            break

    data_path = output_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(rows, f, indent=4)

    return data_path, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare a small GQA subset in this repo's benchmark format.")
    parser.add_argument("--dataset-name", default="vikhyatk/gqa-val")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="data/gqa")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qa-per-image", type=int, default=1)
    args = parser.parse_args()

    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    data_path, count = prepare_gqa_subset(
        dataset_name=args.dataset_name,
        split=args.split,
        output_dir=args.output_dir,
        limit=limit,
        seed=args.seed,
        shuffle=args.shuffle,
        streaming=args.streaming,
        qa_per_image=args.qa_per_image,
    )
    print(f"Wrote {count} examples to {data_path}")


if __name__ == "__main__":
    main()
