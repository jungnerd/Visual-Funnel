# Visual Funnel Implementation Spec

This branch is a standalone Visual Funnel codebase. It keeps only the pieces
needed to run the Visual Funnel inference pipeline.

## Target Pipeline

1. Load an image-question pair.
2. Generate a contextual attention map using:

   ```text
   To answer '{question}', where in the image should I look?
   ```

3. Normalize the attention map into a probability distribution.
4. Compute normalized Shannon entropy.
5. Build a hierarchical portfolio:
   - `focal`: base model image size centered at the global attention center
   - `alpha1`: immediate context centered using attention inside `focal`
   - `alpha2`: broader context centered using attention inside `alpha1`
6. Answer with `original,focal,alpha1,alpha2`.

## Default Formula

```text
alpha1 = 1.2 + 0.6 * H_norm
alpha2 = 1.6 + 1.2 * H_norm
```

## Main Files

- `run.py`: loads models/datasets and writes Visual Funnel result JSON.
- `visual_funnel_attention.py`: extracts contextual attention maps for Qwen2.5-VL.
- `visual_funnel_utils.py`: normalizes attention, computes entropy, builds crop
  boxes, and materializes portfolio images.
- `get_score.py`: scores `visual_funnel_answer` against dataset labels.
- `info.py`: stores model IDs and dataset paths.

## Output Schema

Each result row keeps the original dataset fields and adds:

```text
original_answer
visual_funnel_answer
focal_bbox
visual_funnel
num_img_tokens        # Qwen2.5 only
```

`visual_funnel` contains entropy, scale parameters, centers, crop boxes, and the
view order used for inference.

## Non-Goals

- No training or fine-tuning.
- No external detector, OCR system, segmenter, or image search module.
- No alternative cropping baselines in this standalone branch.
