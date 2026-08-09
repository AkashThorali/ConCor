# coding=utf-8
"""ConCor-1 — vision-language grounding as bidirectional concept correspondence.

Thin inference-side utilities around the released checkpoint
[`UWGZQ/ConCor-1`](https://huggingface.co/UWGZQ/ConCor-1). The model itself lives in
the Hub repository and is loaded with `trust_remote_code=True`; this package only
adds loading defaults, batching, rendering and a reader for the released data.

    from concor1 import load_concor1, predict, render_correspondences

    model, processor = load_concor1()
    correspondences = predict(model, processor, image, text)
    render_correspondences(image, text, correspondences).save("out.png")
"""

from concor1.inference import DEFAULT_MODEL_ID, load_concor1, predict, predict_batch
from concor1.visualize import (
    PALETTE,
    overlay_masks,
    render_correspondences,
    render_text_html,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "PALETTE",
    "load_concor1",
    "overlay_masks",
    "predict",
    "predict_batch",
    "render_correspondences",
    "render_text_html",
]

__version__ = "1.0.0"
