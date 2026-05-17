import argparse
import difflib
import json
import os
import re

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "howd": "how'd",
    "hows": "how's",
    "im": "i'm",
    "ive": "i've",
    "isnt": "isn't",
    "itd": "it'd",
    "itll": "it'll",
    "mightve": "might've",
    "mustve": "must've",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyd": "they'd",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "whos": "who's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [';', "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def process_punctuation(text):
    out = text
    for punct in PUNCT:
        if punct + " " in text or " " + punct in text or re.search(COMMA_STRIP, text):
            out = out.replace(punct, "")
        else:
            out = out.replace(punct, " ")
    return PERIOD_STRIP.sub("", out, re.UNICODE)


def normalize_answer(text):
    words = []
    for word in process_punctuation(str(text)).lower().split():
        word = MANUAL_MAP.get(word, word)
        if word in ARTICLES:
            continue
        words.append(CONTRACTIONS.get(word, word))
    return " ".join(words)


def vqa_score(prediction, labels):
    prediction = normalize_answer(prediction)
    labels = [normalize_answer(label) for label in labels]
    matches = sum(1 for label in labels if prediction == label)
    return 100 * min(0.3 * matches, 1)


def gqa_score(prediction, labels):
    prediction = normalize_answer(prediction)
    labels = [normalize_answer(label) for label in labels]
    return 100 * sum(1 for label in labels if label in prediction)


def doc_score(prediction, labels):
    prediction = normalize_answer(prediction)
    labels = [normalize_answer(label) for label in labels]
    return 100 if any(prediction in label or label in prediction for label in labels) else 0


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def match_choice(answer, candidates):
    scores = [similarity(answer, candidate) for candidate in candidates]
    return candidates[scores.index(max(scores))]


def score_row(task, row):
    labels = row["labels"]
    if isinstance(labels, str):
        labels = [labels]

    original = row["original_answer"]
    visual_funnel = row["visual_funnel_answer"]

    if task in {"textvqa", "aokvqa", "vqav2"}:
        return vqa_score(original, labels), vqa_score(visual_funnel, labels)
    if task == "gqa":
        return gqa_score(original, labels), gqa_score(visual_funnel, labels)
    if task in {"docvqa", "infovqa"}:
        return doc_score(original, labels), doc_score(visual_funnel, labels)
    if task == "pope":
        label = labels[0]
        return (
            100 if match_choice(original, ["yes", "no"]) == label else 0,
            100 if match_choice(visual_funnel, ["yes", "no"]) == label else 0,
        )
    if task == "vstar":
        candidates = row["question"].split("\n")[1:-1]
        label = labels[0]
        original_choice = "ABCD"[candidates.index(match_choice(original, candidates))]
        vf_choice = "ABCD"[candidates.index(match_choice(visual_funnel, candidates))]
        return vqa_score(original_choice, [label]), vqa_score(vf_choice, [label])
    raise ValueError(f"Unsupported task: {task}")


def evaluate_file(path, task):
    with open(path, "r") as f:
        rows = json.load(f)

    original_scores = []
    vf_scores = []
    for row in tqdm(rows, desc=os.path.basename(path), ncols=100):
        original_score, vf_score = score_row(task, row)
        original_scores.append(original_score)
        vf_scores.append(vf_score)

    return sum(original_scores) / len(original_scores), sum(vf_scores) / len(vf_scores)


def main(args):
    import pandas as pd

    os.makedirs(args.save_path, exist_ok=True)
    results = []
    json_files = sorted(file for file in os.listdir(args.data_dir) if file.endswith(".json"))

    for json_file in json_files:
        parts = json_file[:-5].split("-")
        if len(parts) < 3:
            continue
        model_name, task = parts[:2]
        method = "-".join(parts[2:])
        if args.tasks and task not in args.tasks:
            continue
        if not method.startswith("visual_funnel"):
            continue

        original_acc, visual_funnel_acc = evaluate_file(
            os.path.join(args.data_dir, json_file),
            task,
        )
        results.append(
            {
                "model_name": model_name,
                "task": task,
                "method": method,
                "original_acc": original_acc,
                "visual_funnel_acc": visual_funnel_acc,
            }
        )

    report_path = os.path.join(args.save_path, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(os.path.join(args.save_path, "evaluation_report.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Visual Funnel result JSON files.")
    parser.add_argument("--data_dir", type=str, default="./data/results")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["textvqa", "vstar", "gqa", "pope", "aokvqa", "docvqa", "infovqa", "vqav2"],
    )
    main(parser.parse_args())
