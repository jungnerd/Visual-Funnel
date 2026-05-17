import base64
from io import BytesIO

import numpy as np
import torch
from qwen_vl_utils import process_vision_info


QWEN_ATT_LAYER = 22


def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def prepare_qwen2_5_input(messages, processor):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def localization_prompt(question):
    return f"To answer '{question}', where in the image should I look?"


def contextual_attention_qwen2_5(image, question, model, processor):
    image_str = encode_image_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": localization_prompt(question)},
            ],
        }
    ]
    inputs = prepare_qwen2_5_input(messages, processor).to(model.device, torch.bfloat16)

    att_shape = (inputs["image_grid_thw"][0, 1:] / 2).cpu().numpy().astype(int).tolist()
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    input_ids = inputs["input_ids"].tolist()[0]
    pos = input_ids.index(vision_start_token_id) + 1
    pos_end = input_ids.index(vision_end_token_id)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    att = outputs["attentions"][QWEN_ATT_LAYER][0, :, -1, pos:pos_end].mean(dim=0)
    return att.to(torch.float32).detach().cpu().numpy().reshape(att_shape)


def extract_contextual_attention(model_name, image, question, model, processor):
    if model_name == "qwen2_5":
        return contextual_attention_qwen2_5(image, question, model, processor)
    raise ValueError(f"Unsupported model: {model_name}")
