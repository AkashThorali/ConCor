#!/usr/bin/env python3
# coding=utf-8
"""Local Gradio demo for ConCor-1.

    pip install -r demo/requirements.txt
    python demo/app.py                                   # loads UWGZQ/ConCor-1 from the Hub
    python demo/app.py --model /path/to/local/checkpoint --port 7861 --share

Give it an image and its paired text; it returns every correspondence between a
visually referential text span and an instance-level image mask, in matching colours.

The hosted version of this demo runs at
https://huggingface.co/spaces/UWGZQ/ConCor-1-demo — that Space adds ZeroGPU
scheduling on top; this file is the same app without it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

# fla compiles Triton kernels at runtime; keep its cache somewhere writable.
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concor1 import DEFAULT_MODEL_ID, load_concor1  # noqa: E402
from concor1.visualize import PALETTE, label_for, render_text_html  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HEX_PALETTE = ["#%02x%02x%02x" % color for color in PALETTE]

DESCRIPTION = """
# ConCor-1: Vision-Language Grounding as Bidirectional Concept Correspondence

**Give an image and its text.** ConCor-1 recovers *all* correspondences between visually
referential text spans and instance-level image segments. The text spans are **not**
given as queries — the model decides which parts of the text refer to something visible,
groups co-referring mentions (*"a skier"* … *"a person"* become one text mask), segments
the matching object, and decides how many correspondences exist.

A caption, a referring expression and a category list are all just "the text".
"""

USAGE_NOTES = """
### How to read the output

* **Masks** — one image mask per correspondence, in the same colour as its text mask.
* **Grounded text** — the predicted text mask of each correspondence. A text mask can be
  several disjoint spans, which is how repeated and co-referring mentions are grouped.
  One span can also carry **several underlines**: with `baseball player . baseball glove`
  over an image of two players, the phrase *"baseball player"* is claimed by two
  correspondences, one per person.
* **Presence** — the model's own score for "this is a valid image–text correspondence".

### Thresholds

Defaults are the paper's: presence 0.10, image mask 0.45, text mask 0.45, NMS IoU 0.50.
The sliders re-render from cached logits, so they are instant and cost no GPU time.
Lower the presence threshold to see more (and noisier) candidates.
"""


def load_examples() -> List[list]:
    """`examples/examples.json`, with image paths resolved against the repo root."""
    manifest = REPO_ROOT / "examples" / "examples.json"
    if not manifest.exists():
        return []
    examples = []
    for entry in json.loads(manifest.read_text()):
        image = entry.get("image")
        path = None if image is None else str((manifest.parent / image).resolve())
        if path is not None and not os.path.exists(path):
            continue
        examples.append([path, entry["text"]])
    return examples


def build_demo(model, processor) -> gr.Blocks:
    device = next(model.parameters()).device

    @torch.inference_mode()
    def run_model(image: Optional[Image.Image], text: str) -> Dict:
        """One forward pass; raw logits are cached so thresholds re-render for free."""
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        autocast_dtype = next(model.backbone.parameters()).dtype
        with torch.autocast(
            device.type, dtype=autocast_dtype,
            enabled=autocast_dtype in (torch.bfloat16, torch.float16),
        ):
            outputs = model(**inputs)
        return {
            "presence_logits": outputs.presence_logits.float().cpu(),
            "text_mask_logits": outputs.text_mask_logits.float().cpu(),
            "image_mask_logits": (
                None if outputs.image_mask_logits is None else outputs.image_mask_logits.float().cpu()
            ),
            "image_mask_grid_hw": (
                None if outputs.image_mask_grid_hw is None else outputs.image_mask_grid_hw.cpu()
            ),
        }

    def correspondences_from_cache(cache, presence, image_threshold, text_threshold, nms):
        return processor.post_process_correspondences(
            SimpleNamespace(**cache["logits"]),
            text=cache["text"],
            target_sizes=None if cache["size"] is None else [cache["size"]],
            presence_threshold=presence,
            image_threshold=image_threshold,
            text_threshold=text_threshold,
            nms_iou_threshold=nms,
            return_masks=cache["size"] is not None,
        )[0]

    def details_html(correspondences: List[Dict], has_image: bool) -> str:
        if not correspondences:
            return (
                '<div style="padding:10px;">No correspondence passed the presence threshold. '
                "Try lowering it, or check that the text describes the image.</div>"
            )
        rows = []
        for index, item in enumerate(correspondences):
            color = HEX_PALETTE[index % len(HEX_PALETTE)]
            phrases = " / ".join(f"“{phrase}”" for phrase in item["text_phrases"]) or "—"
            spans = ", ".join(f"[{start}, {end})" for start, end in item["text_spans"]) or "—"
            mask = item.get("mask")
            area = f"{100.0 * mask.mean():.2f}%" if has_image and mask is not None else "—"
            rows.append(
                f'<tr><td style="padding:4px 10px;"><span style="display:inline-block;width:11px;'
                f'height:11px;border-radius:2px;background:{color};"></span></td>'
                f'<td style="padding:4px 10px;">{index + 1}</td>'
                f'<td style="padding:4px 10px;">{item["presence_score"]:.3f}</td>'
                f'<td style="padding:4px 10px;">{phrases}</td>'
                f'<td style="padding:4px 10px;font-family:monospace;font-size:12px;">{spans}</td>'
                f'<td style="padding:4px 10px;">{area}</td>'
                f'<td style="padding:4px 10px;">{item["bridge_index"]}</td></tr>'
            )
        header = (
            '<tr style="text-align:left;"><th></th><th style="padding:4px 10px;">#</th>'
            '<th style="padding:4px 10px;">presence</th><th style="padding:4px 10px;">text mask</th>'
            '<th style="padding:4px 10px;">char spans</th><th style="padding:4px 10px;">image mask</th>'
            '<th style="padding:4px 10px;">bridge</th></tr>'
        )
        return (
            '<div style="overflow-x:auto;"><table style="border-collapse:collapse;font-size:14px;">'
            + header + "".join(rows) + "</table></div>"
        )

    def render(cache, presence, image_threshold, text_threshold, nms):
        if not cache:
            return gr.update(value=None), "", ""
        correspondences = correspondences_from_cache(
            cache, presence, image_threshold, text_threshold, nms
        )
        labels = [label_for(index, item) for index, item in enumerate(correspondences)]

        annotated = gr.update(value=None)
        if cache["image"] is not None:
            annotations = [
                (item["mask"], labels[index])
                for index, item in enumerate(correspondences)
                if item.get("mask") is not None and item["mask"].any()
            ]
            # `color_map` is what keeps an image mask the same colour as its text mask.
            annotated = gr.update(
                value=(np.array(cache["image"], dtype=np.uint8), annotations),
                color_map={
                    label: HEX_PALETTE[index % len(HEX_PALETTE)]
                    for index, label in enumerate(labels)
                },
            )
        return (
            annotated,
            render_text_html(cache["text"], correspondences),
            details_html(correspondences, has_image=cache["image"] is not None),
        )

    def predict(image, text, presence, image_threshold, text_threshold, nms):
        text = (text or "").strip()
        if not text:
            raise gr.Error(
                "ConCor-1 always grounds text — enter the image's caption, a referring "
                "expression, or a category list such as `bear . grass . tree`."
            )
        if image is not None:
            image = image.convert("RGB")
        cache = {
            "logits": run_model(image, text),
            "text": text,
            "image": image,
            "size": None if image is None else (image.height, image.width),
        }
        return (cache,) + render(cache, presence, image_threshold, text_threshold, nms)

    with gr.Blocks(title="ConCor-1 — Bidirectional Concept Correspondence") as demo:
        gr.Markdown(DESCRIPTION)
        cache_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=4):
                image_input = gr.Image(label="Image (optional)", type="pil", height=340)
                text_input = gr.TextArea(
                    label="Text to ground",
                    placeholder="A caption, a referring expression, or a category list (`bear . grass . tree`).",
                    lines=4,
                )
                run_button = gr.Button("Find correspondences", variant="primary")
                with gr.Accordion("Thresholds", open=False):
                    presence_slider = gr.Slider(0.0, 1.0, value=0.10, step=0.01, label="Presence")
                    image_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="Image mask")
                    text_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="Text mask")
                    nms_slider = gr.Slider(0.0, 1.0, value=0.50, step=0.01, label="NMS IoU")

            with gr.Column(scale=6):
                annotated_output = gr.AnnotatedImage(
                    label="Image masks (one per correspondence)", height=430
                )
                text_output = gr.HTML(label="Grounded text")
                details_output = gr.HTML(label="Correspondences")

        examples = load_examples()
        if examples:
            gr.Examples(examples=examples, inputs=[image_input, text_input], cache_examples=False)
        gr.Markdown(USAGE_NOTES)

        sliders = [presence_slider, image_slider, text_slider, nms_slider]
        predict_inputs = [image_input, text_input] + sliders
        predict_outputs = [cache_state, annotated_output, text_output, details_output]
        run_button.click(fn=predict, inputs=predict_inputs, outputs=predict_outputs)
        text_input.submit(fn=predict, inputs=predict_inputs, outputs=predict_outputs)
        for slider in sliders:
            slider.release(
                fn=render,
                inputs=[cache_state] + sliders,
                outputs=[annotated_output, text_output, details_output],
            )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("CONCOR1_MODEL_ID", DEFAULT_MODEL_ID))
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn_implementation", default="sdpa",
                        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print(f"loading {args.model} ...", flush=True)
    model, processor = load_concor1(
        args.model, device=args.device, attn_implementation=args.attn_implementation
    )
    build_demo(model, processor).queue().launch(
        server_port=args.port, share=args.share, show_error=True
    )


if __name__ == "__main__":
    main()
