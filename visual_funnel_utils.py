import math

import numpy as np


def normalize_attention_map(att_map, eps=1e-12):
    """Return a non-negative probability map with sum 1."""
    att = np.asarray(att_map, dtype=np.float64)
    if att.ndim != 2:
        raise ValueError(f"att_map must be 2D, got shape {att.shape}")

    att = np.nan_to_num(att, nan=0.0, posinf=0.0, neginf=0.0)
    att = np.maximum(att, 0.0)

    total = float(att.sum())
    if total <= eps:
        return np.full(att.shape, 1.0 / att.size, dtype=np.float64)

    return att / total


def normalized_attention_entropy(att_norm, eps=1e-12):
    """Return normalized Shannon entropy in [0, 1]."""
    att = normalize_attention_map(att_norm, eps=eps)
    if att.size <= 1:
        return 0.0

    entropy = -float(np.sum(att * np.log(att + eps)))
    norm = entropy / math.log(att.size)
    return float(np.clip(norm, 0.0, 1.0))


def visual_funnel_scales(
    entropy,
    alpha1_base=1.2,
    alpha1_entropy=0.6,
    alpha2_base=1.6,
    alpha2_entropy=1.2,
):
    """Return alpha1 and alpha2 from Visual Funnel."""
    entropy = float(np.clip(entropy, 0.0, 1.0))
    return (
        float(alpha1_base + alpha1_entropy * entropy),
        float(alpha2_base + alpha2_entropy * entropy),
    )


def _cell_centers(att_shape, image_size):
    height_blocks, width_blocks = att_shape
    image_width, image_height = image_size

    x_step = image_width / width_blocks
    y_step = image_height / height_blocks

    xs = (np.arange(width_blocks, dtype=np.float64) + 0.5) * x_step
    ys = (np.arange(height_blocks, dtype=np.float64) + 0.5) * y_step
    return np.meshgrid(xs, ys)


def attention_weighted_center(att_norm, image_size, region=None, eps=1e-12):
    """Return weighted center ``(x, y)`` in original image coordinates."""
    att = normalize_attention_map(att_norm, eps=eps)
    xs, ys = _cell_centers(att.shape, image_size)

    if region is not None:
        x1, y1, x2, y2 = region
        mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        region_att = np.where(mask, att, 0.0)
        total = float(region_att.sum())
        if total <= eps:
            return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)
        att = region_att / total

    center_x = float(np.sum(xs * att))
    center_y = float(np.sum(ys * att))
    return center_x, center_y


def square_bbox_from_center(center, side_length, image_size):
    """Return clipped square bbox ``(x1, y1, x2, y2)`` around ``center``."""
    image_width, image_height = image_size
    side = float(side_length)
    if side <= 0:
        raise ValueError(f"side_length must be positive, got {side_length}")

    width = min(side, float(image_width))
    height = min(side, float(image_height))
    center_x, center_y = map(float, center)

    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = x1 + width
    y2 = y1 + height

    if x1 < 0:
        x2 -= x1
        x1 = 0.0
    if y1 < 0:
        y2 -= y1
        y1 = 0.0
    if x2 > image_width:
        x1 -= x2 - image_width
        x2 = float(image_width)
    if y2 > image_height:
        y1 -= y2 - image_height
        y2 = float(image_height)

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_width), x2)
    y2 = min(float(image_height), y2)

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def build_visual_funnel_portfolio(
    att_map,
    image_size,
    base_size,
    base_scale=1.0,
    alpha1_base=1.2,
    alpha1_entropy=0.6,
    alpha2_base=1.6,
    alpha2_entropy=1.2,
    focal_bbox=None,
):
    """Return Visual Funnel metadata for the three hierarchical crops."""
    att_norm = normalize_attention_map(att_map)
    entropy = normalized_attention_entropy(att_norm)
    scaled_base_size = float(base_size) * float(base_scale)
    alpha1, alpha2 = visual_funnel_scales(
        entropy,
        alpha1_base=alpha1_base,
        alpha1_entropy=alpha1_entropy,
        alpha2_base=alpha2_base,
        alpha2_entropy=alpha2_entropy,
    )

    if focal_bbox is None:
        center_focal = attention_weighted_center(att_norm, image_size)
        bbox_focal = square_bbox_from_center(center_focal, scaled_base_size, image_size)
        focal_side = scaled_base_size
    else:
        bbox_focal = tuple(int(round(v)) for v in focal_bbox)
        x1, y1, x2, y2 = bbox_focal
        center_focal = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        focal_side = float(max(x2 - x1, y2 - y1))

    center_alpha1 = attention_weighted_center(att_norm, image_size, bbox_focal)
    bbox_alpha1 = square_bbox_from_center(center_alpha1, alpha1 * focal_side, image_size)

    center_alpha2 = attention_weighted_center(att_norm, image_size, bbox_alpha1)
    bbox_alpha2 = square_bbox_from_center(center_alpha2, alpha2 * focal_side, image_size)

    return {
        "attention_shape": [int(att_norm.shape[0]), int(att_norm.shape[1])],
        "entropy": float(entropy),
        "base_size": float(base_size),
        "base_scale": float(base_scale),
        "scaled_base_size": float(scaled_base_size),
        "focal_side": float(focal_side),
        "alpha1": float(alpha1),
        "alpha2": float(alpha2),
        "scale_params": {
            "alpha1_base": float(alpha1_base),
            "alpha1_entropy": float(alpha1_entropy),
            "alpha2_base": float(alpha2_base),
            "alpha2_entropy": float(alpha2_entropy),
        },
        "centers": {
            "focal": [float(center_focal[0]), float(center_focal[1])],
            "alpha1": [float(center_alpha1[0]), float(center_alpha1[1])],
            "alpha2": [float(center_alpha2[0]), float(center_alpha2[1])],
        },
        "bboxes": {
            "focal": list(bbox_focal),
            "alpha1": list(bbox_alpha1),
            "alpha2": list(bbox_alpha2),
        },
    }


def crop_portfolio_images(image, bboxes):
    """Return ``[crop_focal, crop_alpha1, crop_alpha2]`` from a PIL image."""
    return [
        image.crop(tuple(bboxes["focal"])).convert("RGB"),
        image.crop(tuple(bboxes["alpha1"])).convert("RGB"),
        image.crop(tuple(bboxes["alpha2"])).convert("RGB"),
    ]


def parse_visual_funnel_views(view_spec):
    """Parse a comma-separated Visual Funnel view list."""
    valid = {"original", "focal", "alpha1", "alpha2"}
    views = [view.strip() for view in view_spec.split(",") if view.strip()]
    if not views:
        raise ValueError("Visual Funnel view list must not be empty")

    unknown = [view for view in views if view not in valid]
    if unknown:
        raise ValueError(
            f"Unknown Visual Funnel view(s): {unknown}. "
            f"Valid views are: {sorted(valid)}"
        )

    return views


def visual_funnel_images(image, bboxes, view_spec):
    """Return images selected by a Visual Funnel view spec."""
    views = parse_visual_funnel_views(view_spec)
    image_rgb = image.convert("RGB")
    view_to_image = {
        "original": image_rgb,
        "focal": image.crop(tuple(bboxes["focal"])).convert("RGB"),
        "alpha1": image.crop(tuple(bboxes["alpha1"])).convert("RGB"),
        "alpha2": image.crop(tuple(bboxes["alpha2"])).convert("RGB"),
    }
    return [view_to_image[view] for view in views], views
