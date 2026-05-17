import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from info import model_to_fullname, task_to_image_path, task_to_question_path
from visual_funnel_attention import (
    encode_image_base64,
    extract_contextual_attention,
    prepare_qwen2_5_input,
)
from visual_funnel_utils import build_visual_funnel_portfolio, visual_funnel_images


MODEL_INPUT_SIZE = {
    "qwen2_5": 224,
}


def write_result_json(output_path, rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(rows, f, indent=4)
    os.replace(tmp_path, output_path)


def load_model_and_processor(model_name, model_id, device):
    if model_name == "qwen2_5":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        max_pixels = 256 * 28 * 28
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        processor.image_processor.size["longest_edge"] = max_pixels
        return model, processor

    raise ValueError(f"Unsupported model: {model_name}")


def generate_answer(model_name, images, question, model, processor):
    prompt = f"{question} Answer the question using a single word or phrase."

    if model_name == "qwen2_5":
        content = [
            {"type": "image", "image": f"data:image;base64,{encode_image_base64(image)}"}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)
        generate_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generate_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generate_ids)
        ]
        text = processor.batch_decode(
            generate_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        num_img_tokens = int((inputs.input_ids[0] == image_token_id).sum().item())
        return text.strip(), num_img_tokens

    raise ValueError(f"Unsupported model: {model_name}")


def run_visual_funnel(
    model_name,
    image_path,
    question,
    model,
    processor,
    vf_config=None,
):
    vf_config = vf_config or {}
    image = Image.open(image_path).convert("RGB")
    model.eval()

    original_answer, _ = generate_answer(model_name, [image], question, model, processor)

    att_map = extract_contextual_attention(model_name, image, question, model, processor)
    portfolio = build_visual_funnel_portfolio(
        att_map,
        image.size,
        MODEL_INPUT_SIZE[model_name],
        base_scale=vf_config.get("base_scale", 1.0),
        alpha1_base=vf_config.get("alpha1_base", 1.2),
        alpha1_entropy=vf_config.get("alpha1_entropy", 0.6),
        alpha2_base=vf_config.get("alpha2_base", 1.6),
        alpha2_entropy=vf_config.get("alpha2_entropy", 1.2),
    )
    portfolio_images, selected_views = visual_funnel_images(
        image,
        portfolio["bboxes"],
        vf_config.get("views", "original,focal,alpha1,alpha2"),
    )
    portfolio["views"] = selected_views

    visual_funnel_answer, num_img_tokens = generate_answer(
        model_name,
        portfolio_images,
        question,
        model,
        processor,
    )

    result = {
        "original_answer": original_answer,
        "visual_funnel_answer": visual_funnel_answer,
        "focal_bbox": portfolio["bboxes"]["focal"],
        "visual_funnel": portfolio,
    }
    if num_img_tokens is not None:
        result["num_img_tokens"] = int(num_img_tokens)

    return result


def load_rows(question_path):
    if os.path.exists(question_path):
        with open(question_path, "r") as f:
            return json.load(f)
    return list(load_dataset(question_path)["test"])


def attach_image_paths(rows, image_root):
    for row in rows:
        if "image_path" in row:
            row["image_path"] = os.path.join(image_root, row["image_path"])
        else:
            row["image_path"] = os.path.join(image_root, f"{row['image_id']}.jpg")
    return rows


def main(args):
    model, processor = load_model_and_processor(args.model, args.model_id, args.device)
    rows = attach_image_paths(load_rows(args.question_path), args.image_path)
    chunked_rows = np.array_split(rows, args.total_chunks)
    current_rows = list(chunked_rows[args.chunk_id])
    if args.max_samples is not None:
        current_rows = current_rows[: args.max_samples]

    old_rows = []
    if os.path.exists(args.output_path) and not args.overwrite:
        with open(args.output_path, "r") as f:
            old_rows = json.load(f)

    new_rows = []
    vf_config = {
        "base_scale": args.vf_base_scale,
        "alpha1_base": args.vf_alpha1_base,
        "alpha1_entropy": args.vf_alpha1_entropy,
        "alpha2_base": args.vf_alpha2_base,
        "alpha2_entropy": args.vf_alpha2_entropy,
        "views": args.vf_views,
    }

    for idx, row in enumerate(tqdm(current_rows, desc="Processing", ncols=100), start=1):
        result = run_visual_funnel(
            args.model,
            row["image_path"],
            row["question"],
            model,
            processor,
            vf_config=vf_config,
        )
        row.update(result)
        new_rows.append(row)

        if args.save_every > 0 and idx % args.save_every == 0:
            write_result_json(args.output_path, old_rows + new_rows)

    write_result_json(args.output_path, old_rows + new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Visual Funnel inference.")
    parser.add_argument("--model", type=str, default="qwen2_5", choices=model_to_fullname.keys())
    parser.add_argument("--task", type=str, default="textvqa", choices=task_to_question_path.keys())
    parser.add_argument("--save_path", type=str, default="./playground/data/results")
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--question_path_override", type=str, default=None)
    parser.add_argument("--image_path_override", type=str, default=None)
    parser.add_argument("--output_tag", type=str, default=None)
    parser.add_argument("--vf_base_scale", type=float, default=1.0)
    parser.add_argument("--vf_alpha1_base", type=float, default=1.2)
    parser.add_argument("--vf_alpha1_entropy", type=float, default=0.6)
    parser.add_argument("--vf_alpha2_base", type=float, default=1.6)
    parser.add_argument("--vf_alpha2_entropy", type=float, default=1.2)
    parser.add_argument("--vf_views", type=str, default="original,focal,alpha1,alpha2")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_id = model_to_fullname[args.model]
    args.image_path = args.image_path_override or task_to_image_path[args.task]
    args.question_path = args.question_path_override or task_to_question_path[args.task]

    output_method = "visual_funnel"
    if args.output_tag:
        safe_tag = args.output_tag.replace("/", "_").replace("\\", "_")
        output_method = f"{output_method}-{safe_tag}"
    args.output_path = os.path.join(args.save_path, f"{args.model}-{args.task}-{output_method}.json")

    main(args)
