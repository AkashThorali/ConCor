#!/usr/bin/env python3
# coding=utf-8
"""Ground an image-text pair (or a whole file of them) with ConCor-1.

ConCor-1 predicts the full set of correspondences between visually referential text
spans and instance-level image masks. The text spans are *not* queries — the model
decides which parts of the text are grounded, groups co-referring mentions, segments
the matching object, and decides how many correspondences exist.

Single pair:

    python scripts/run_inference.py \
        --image examples/bear.jpg \
        --text "This image depicts a close-up of a brown bear in a natural outdoor setting. \
The background consists of lush green grass. In the foreground, a large brown bear is \
positioned centrally." \
        --output_dir outputs/

A category list is just another text:

    python scripts/run_inference.py --image examples/bear.jpg --text "bear . grass . tree . person"

Many pairs, batched (`examples/examples.json` is a list of `{"image", "text"}`):

    python scripts/run_inference.py --input_json examples/examples.json --output_dir outputs/ --batch_size 4

Text only — which spans are visually referential, and how do they group?

    python scripts/run_inference.py --text "a bush of plant behind middle woman"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concor1 import DEFAULT_MODEL_ID, load_concor1, render_correspondences  # noqa: E402
from concor1.inference import format_correspondences, predict_batch  # noqa: E402
from concor1.visualize import render_text_html  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_argument_group("input (either --text, or --input_json)")
    source.add_argument("--image", type=Path, default=None, help="image to ground; omit for text-only")
    source.add_argument("--text", default=None, help="caption, referring expression or category list")
    source.add_argument(
        "--input_json",
        type=Path,
        default=None,
        help='JSON list of {"image": path, "text": str}; paths resolve relative to the JSON file',
    )

    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hub id or local directory")
    parser.add_argument("--device", default=None, help="default: cuda when available")
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="flash_attention_2 reproduces the paper's logits bit-for-bit (needs flash-attn)",
    )
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--output_dir", type=Path, default=None, help="write overlays + JSON here")
    parser.add_argument("--save_html", action="store_true", help="also write the grounded text as HTML")
    parser.add_argument("--figure_width", type=int, default=1100)

    thresholds = parser.add_argument_group("thresholds (defaults are the paper's)")
    thresholds.add_argument("--presence_threshold", type=float, default=None, help="default 0.10")
    thresholds.add_argument("--text_threshold", type=float, default=None, help="default 0.45")
    thresholds.add_argument("--image_threshold", type=float, default=None, help="default 0.45")
    thresholds.add_argument("--nms_iou_threshold", type=float, default=None, help="default 0.50")

    args = parser.parse_args()
    if args.input_json is None and args.text is None:
        parser.error("give either --text (with an optional --image) or --input_json")
    if args.input_json is not None and args.text is not None:
        parser.error("--input_json and --text are mutually exclusive")
    return args


def load_pairs(args: argparse.Namespace) -> List[Dict]:
    """Normalize the CLI input into a list of `{"image": Path|None, "text": str}`."""
    if args.input_json is not None:
        entries = json.loads(args.input_json.read_text())
        if not isinstance(entries, list):
            raise SystemExit(f"{args.input_json} must hold a JSON list of objects.")
        root = args.input_json.resolve().parent
        pairs = []
        for position, entry in enumerate(entries):
            if "text" not in entry:
                raise SystemExit(f"entry {position} in {args.input_json} has no 'text' field")
            image = entry.get("image")
            pairs.append(
                {
                    "image": None if image is None else (root / image if not Path(image).is_absolute() else Path(image)),
                    "text": entry["text"],
                }
            )
        return pairs
    return [{"image": args.image, "text": args.text}]


def output_stem(pair: Dict, index: int) -> str:
    image = pair["image"]
    return f"{index:03d}_{image.stem}" if image is not None else f"{index:03d}_text_only"


def main() -> None:
    args = parse_args()
    pairs = load_pairs(args)

    # A batch is either all image-text pairs or all text-only, so split on that.
    if any(pair["image"] is not None for pair in pairs) and any(pair["image"] is None for pair in pairs):
        raise SystemExit(
            "Mixing image-text pairs and text-only entries in one run is not supported; "
            "split them into two files."
        )

    print(f"loading {args.model} ({args.attn_implementation}) ...", flush=True)
    model, processor = load_concor1(
        args.model, device=args.device, attn_implementation=args.attn_implementation
    )

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        images: List[Optional[Image.Image]] = [
            None if pair["image"] is None else Image.open(pair["image"]).convert("RGB")
            for pair in batch
        ]
        results = predict_batch(
            model,
            processor,
            images,
            [pair["text"] for pair in batch],
            presence_threshold=args.presence_threshold,
            text_threshold=args.text_threshold,
            image_threshold=args.image_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
        )

        for offset, (pair, image, correspondences) in enumerate(zip(batch, images, results)):
            index = start + offset
            print(f"\n[{index + 1}/{len(pairs)}] {pair['image'] or '(text only)'}")
            print(f"  text: {pair['text']}")
            print(f"  {len(correspondences)} correspondence(s):")
            print(format_correspondences(correspondences))

            records.append(
                {
                    "image": None if pair["image"] is None else str(pair["image"]),
                    "text": pair["text"],
                    # Masks are dropped here: JSON carries the spans, the PNG carries the pixels.
                    "correspondences": [
                        {
                            "bridge_index": item["bridge_index"],
                            "presence_score": item["presence_score"],
                            "text_spans": item["text_spans"],
                            "text_phrases": item["text_phrases"],
                            "mask_area_fraction": (
                                float(item["mask"].mean()) if item.get("mask") is not None else None
                            ),
                        }
                        for item in correspondences
                    ],
                }
            )

            if args.output_dir is None:
                continue
            stem = output_stem(pair, index)
            figure = render_correspondences(image, pair["text"], correspondences, width=args.figure_width)
            figure.save(args.output_dir / f"{stem}.png")
            if args.save_html:
                (args.output_dir / f"{stem}.html").write_text(
                    render_text_html(pair["text"], correspondences), encoding="utf-8"
                )

    if args.output_dir is not None:
        (args.output_dir / "correspondences.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {len(records)} result(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
