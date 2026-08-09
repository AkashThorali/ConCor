#!/usr/bin/env python3
# coding=utf-8
"""Inspect and visualize the released ConCor-1 concept-correspondence data.

Every released file is a parquet of annotations only, with masks stored inline;
images live in the source datasets and are joined on `image_key` (see docs/DATA.md).

Print a sample (downloads the parquet from the Hub on first use):

    python scripts/inspect_data.py --config coconut_pancap --split validation --index 0

Render the ground-truth correspondences over the image — the same colour on a text
span and on its mask — once you have pointed the script at your COCO images:

    python scripts/inspect_data.py --config coconut_pancap --split validation --index 0 \
        --image_root coco=/data/coco --output gt.png

Decode every mask in a file as an integrity check:

    python scripts/inspect_data.py --parquet /path/to/coco.parquet --check_masks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concor1.data import (  # noqa: E402
    CONFIGS,
    IMAGE_ROOTS,
    category_prompt,
    eval_units,
    hub_parquet,
    iter_samples,
    load_sample,
)
from concor1.visualize import render_correspondences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", choices=sorted(CONFIGS), default=None, help="released config name")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--parquet", type=Path, default=None, help="a local parquet, instead of --config")
    parser.add_argument("--index", type=int, default=0, help="row to inspect")
    parser.add_argument(
        "--image_root",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help=f"where a source dataset's images live, e.g. coco=/data/coco. Keys: {sorted(IMAGE_ROOTS)}",
    )
    parser.add_argument("--output", type=Path, default=None, help="render the sample to this PNG")
    parser.add_argument(
        "--check_masks", action="store_true", help="decode every mask in the file and report totals"
    )
    args = parser.parse_args()
    if (args.config is None) == (args.parquet is None):
        parser.error("give exactly one of --config or --parquet")
    return args


def main() -> None:
    args = parse_args()

    for assignment in args.image_root:
        key, _, path = assignment.partition("=")
        if key not in IMAGE_ROOTS:
            raise SystemExit(f"Unknown image-root key {key!r}. Known keys: {sorted(IMAGE_ROOTS)}")
        IMAGE_ROOTS[key] = path

    parquet = str(args.parquet) if args.parquet else hub_parquet(args.config, args.split)
    print(f"file: {parquet}")

    if args.check_masks:
        rows = masks = pixels = 0
        for sample in iter_samples(parquet):
            rows += 1
            for correspondence in sample["correspondences"]:
                mask = correspondence.get("mask")
                if mask is not None:
                    masks += 1
                    pixels += int(mask.sum())
        print(f"{rows:,} rows, {masks:,} masks decoded, {pixels:,} foreground pixels")
        return

    sample = load_sample(parquet, args.index)
    is_caption = "text" in sample

    print(f"dataset={sample['dataset']}  split={sample['split']}  size={sample['height']}x{sample['width']}")
    print(f"image_key={sample['image_key']}")
    print(f"image_path={sample['image_path'] or '(no root configured — pass --image_root)'}")

    text = sample["text"] if is_caption else category_prompt(sample)
    print(f"\ntext ({'caption' if is_caption else 'category list'}):\n  {text}")

    # For caption files the benchmark's ground-truth unit is one entry per unique
    # instance, carrying the union of every group's spans that refer to it.
    units = eval_units(sample) if is_caption else None
    entries = units if is_caption else sample["correspondences"]
    print(f"\n{len(entries)} ground-truth correspondence(s):")
    for entry in entries[:20]:
        mask = entry.get("mask")
        area = f"{100.0 * mask.mean():5.2f}% of image" if mask is not None else "no mask"
        label = entry.get("phrases") or entry.get("category_name")
        print(f"  {str(label)[:64]:66s} {area}")
    if len(entries) > 20:
        print(f"  ... and {len(entries) - 20} more")

    if args.output is None:
        return

    if sample["image_path"] is None:
        raise SystemExit(
            "Rendering needs the source image: pass --image_root <key>=<path> "
            "(see docs/DATA.md for the per-config image roots)."
        )

    from PIL import Image

    image = Image.open(sample["image_path"]).convert("RGB")
    # Reshape the ground truth into the dicts the renderer expects.
    if is_caption:
        correspondences = [
            {
                "presence_score": 1.0,
                "text_spans": entry["char_spans"],
                "text_phrases": entry["phrases"],
                "mask": entry["mask"],
            }
            for entry in units
            if entry["mask"] is not None
        ]
    else:
        correspondences = []
        for entry in sample["correspondences"]:
            name = entry.get("category_name")
            start = text.find(name) if name else -1
            correspondences.append(
                {
                    "presence_score": 1.0,
                    "text_spans": [[start, start + len(name)]] if start >= 0 else [],
                    "text_phrases": [name] if name else [],
                    "mask": entry.get("mask"),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_correspondences(image, text, correspondences).save(args.output)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
