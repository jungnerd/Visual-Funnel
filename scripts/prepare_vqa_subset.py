import argparse
import ast
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


TASK_SPECS = {
    "docvqa": {
        "dataset_name": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "split": "validation",
        "id_field": "questionId",
        "question_field": "question",
        "answers_field": "answers",
    },
    "infovqa": {
        "dataset_name": "weijiezz/infovqa_val_split_test",
        "config": "InfographicVQA",
        "split": "validation",
        "id_field": "questionId",
        "question_field": "question",
        "answers_field": "answers",
    },
    "pope": {
        "dataset_name": "lmms-lab/POPE",
        "config": "Full",
        "split": "adversarial",
        "id_field": "question_id",
        "question_field": "question",
        "answers_field": "answer",
    },
    "aokvqa": {
        "dataset_name": "HuggingFaceM4/A-OKVQA",
        "config": "default",
        "split": "validation",
        "id_field": "question_id",
        "question_field": "question",
        "answers_field": "direct_answers",
    },
    "vqav2": {
        "dataset_name": "HuggingFaceM4/VQAv2",
        "config": "default",
        "split": "validation",
        "id_field": "question_id",
        "question_field": "question",
        "answers_field": "answers",
    },
}


def safe_id(value, fallback):
    text = str(value) if value is not None else fallback
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text or fallback


def normalize_answers(value):
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                return normalize_answers(parsed)
            except (SyntaxError, ValueError):
                pass
        return [stripped]
    if isinstance(value, dict):
        if "answer" in value:
            return [str(value["answer"])]
        return [str(value)]
    if isinstance(value, list):
        answers = []
        for item in value:
            answers.extend(normalize_answers(item))
        return answers
    return [str(value)]


def prepare_subset(task, output_dir, limit, seed, shuffle, streaming):
    spec = TASK_SPECS[task]
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        spec["dataset_name"],
        spec["config"],
        split=spec["split"],
        streaming=streaming,
        trust_remote_code=True,
    )
    if shuffle:
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=1000)
        else:
            indices = list(range(len(dataset)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            dataset = dataset.select(indices)

    rows = []
    for row_idx, item in enumerate(dataset):
        answers = normalize_answers(item.get(spec["answers_field"]))
        if not answers:
            continue

        row_id = safe_id(item.get(spec["id_field"]), f"{task}_{row_idx:06d}")
        image_path = f"{row_id}.jpg"
        image = item["image"].convert("RGB")
        image.save(image_dir / image_path)

        rows.append({
            "id": row_id,
            "question": item[spec["question_field"]],
            "labels": answers,
            "image_path": image_path,
        })

        if limit is not None and len(rows) >= limit:
            break

    data_path = output_dir / "data.json"
    with open(data_path, "w") as f:
        json.dump(rows, f, indent=4)

    return data_path, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare small VQA benchmark subsets for Visual Funnel pilots.")
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = args.output_dir or f"data/{args.task}"
    limit = None if args.limit is not None and args.limit <= 0 else args.limit
    data_path, count = prepare_subset(
        task=args.task,
        output_dir=output_dir,
        limit=limit,
        seed=args.seed,
        shuffle=args.shuffle,
        streaming=args.streaming,
    )
    print(f"Wrote {count} examples to {data_path}")


if __name__ == "__main__":
    main()
