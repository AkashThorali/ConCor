# coding=utf-8
"""Loading and running the released ConCor-1 checkpoint.

ConCor-1 is not autoregressive: one forward pass over
`[image tokens, text tokens, 385 bridge tokens]` yields, for every bridge token, a
presence score, a mask over the text tokens and a mask over the image. The
processor's `post_process_correspondences` turns those logits into a list of
correspondences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "UWGZQ/ConCor-1"

# The paper's inference thresholds; `post_process_correspondences` uses these when a
# threshold is left as None, so they are repeated here only for documentation.
PAPER_THRESHOLDS = {
    "presence_threshold": 0.10,
    "text_threshold": 0.45,
    "image_threshold": 0.45,
    "nms_iou_threshold": 0.50,
}


def load_concor1(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load ConCor-1 and its processor.

    Args:
        model_id: Hub repo id or a local directory holding the released files.
        device: `"cuda"`, `"cpu"`, ...; defaults to CUDA when available.
        dtype: backbone dtype. The three prediction heads always stay in fp32,
            exactly as the training checkpoint holds them.
        attn_implementation: `"flash_attention_2"` reproduces the paper's logits
            bit-for-bit; `"sdpa"` needs no extra dependency and stays within bf16
            kernel noise.

    Returns:
        `(model, processor)`, with the model on `device` and in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return model.to(device).eval(), processor


@torch.inference_mode()
def predict_batch(
    model,
    processor,
    images: Sequence[Optional[Image.Image]],
    texts: Sequence[str],
    presence_threshold: Optional[float] = None,
    text_threshold: Optional[float] = None,
    image_threshold: Optional[float] = None,
    nms_iou_threshold: Optional[float] = None,
) -> List[List[Dict]]:
    """Ground a batch of image-text pairs in a single forward pass.

    `images` may hold `None` only if *every* entry is `None` (text-only mode), since
    a batch is either multimodal or text-only. Sequences are right-padded internally,
    so images of different sizes batch fine.

    Returns:
        One list of correspondences per pair, each sorted by decreasing presence
        score. A correspondence is a dict with `bridge_index`, `presence_score`,
        `text_spans`, `text_phrases` and — with an image — `mask`, a bool array at
        the original image resolution.
    """
    if len(images) != len(texts):
        raise ValueError(f"Got {len(images)} image(s) for {len(texts)} text(s).")

    images = [None if image is None else image.convert("RGB") for image in images]
    has_images = [image is not None for image in images]
    if any(has_images) and not all(has_images):
        raise ValueError(
            "A batch is either all image-text pairs or all text-only; mixing the two "
            "would need different sequence layouts. Split the batch instead."
        )

    device = next(model.parameters()).device
    inputs = processor(
        images=list(images) if all(has_images) else None,
        text=list(texts),
        return_tensors="pt",
    ).to(device)

    # The backbone is bf16 while the heads are fp32, so run the forward pass under
    # autocast — that is the configuration the released logits were verified against.
    autocast_dtype = next(model.backbone.parameters()).dtype
    autocast = autocast_dtype in (torch.bfloat16, torch.float16)
    with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast):
        outputs = model(**inputs)

    target_sizes = (
        [(image.height, image.width) for image in images] if all(has_images) else None
    )
    return processor.post_process_correspondences(
        outputs,
        text=list(texts),
        target_sizes=target_sizes,
        presence_threshold=presence_threshold,
        text_threshold=text_threshold,
        image_threshold=image_threshold,
        nms_iou_threshold=nms_iou_threshold,
        return_masks=target_sizes is not None,
    )


def predict(
    model,
    processor,
    image: Optional[Image.Image],
    text: str,
    **thresholds: Union[float, None],
) -> List[Dict]:
    """Ground one image-text pair. See [`predict_batch`] for the returned fields."""
    return predict_batch(model, processor, [image], [text], **thresholds)[0]


def format_correspondences(correspondences: Sequence[Dict]) -> str:
    """One human-readable line per correspondence."""
    lines = []
    for index, correspondence in enumerate(correspondences):
        mask = correspondence.get("mask")
        area = f"{100.0 * mask.mean():5.2f}%" if mask is not None else "    --"
        phrases = " / ".join(correspondence["text_phrases"]) or "(no text span)"
        lines.append(
            f"{index + 1:3d}. presence={correspondence['presence_score']:.3f}  "
            f"bridge={correspondence['bridge_index']:3d}  mask={area}  "
            f"spans={correspondence['text_spans']}  |  {phrases}"
        )
    return "\n".join(lines)
