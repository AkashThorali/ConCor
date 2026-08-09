# coding=utf-8
"""Reader for the released ConCor-1 concept-correspondence data.

The dataset ([`UWGZQ/ConCor-1-Data`](https://huggingface.co/datasets/UWGZQ/ConCor-1-Data))
ships annotations only — every file is a parquet, every mask is inline, and no row
depends on any other file. Images are *not* included: point [`IMAGE_ROOTS`] at your
local copies of the source datasets (see `docs/DATA.md`).

    from concor1.data import IMAGE_ROOTS, hub_parquet, iter_samples

    IMAGE_ROOTS["coco"] = "/data/coco"          # holds train2017/ and val2017/
    for sample in iter_samples(hub_parquet("coconut_pancap", "validation")):
        sample["text"]                           # the text being grounded
        for correspondence in sample["correspondences"]:
            correspondence["char_spans"]         # [[start, end), ...] into sample["text"]
            correspondence["phrases"]            # the spanned substrings
            correspondence["mask"]               # bool array (height, width)
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional

import numpy as np

HUB_DATASET_ID = "UWGZQ/ConCor-1-Data"

# config -> the splits it ships
CONFIGS: Dict[str, List[str]] = {
    "coco": ["train", "validation"],
    "coconut": ["train"],
    "coconut_pancap": ["train", "validation"],
    "groundedref": ["train", "validation"],
    "goldg_flickr30k": ["train"],
    "goldg_gqa": ["train"],
    "entityseg": ["train", "validation"],
    "ade20k": ["train"],
    "pixmo_points": ["train"],
    "sa1b": ["train"],
    "roboflow_vl_100": ["train"],
    "flickr30k": ["validation"],
    "lvis_minival": ["validation"],
}

# Where each source dataset's images live locally. Fill in the ones you need; a row
# whose root is unset still decodes, it just carries `image_path=None`.
#
#   coco             a directory holding train2017/ and val2017/
#   coco2014         a directory holding train2014/            (GroundedRef validation)
#   coconut          Objects365-2020 root (train/images/v2/patchNN/...)
#   ade20k           ADE20K root holding training/img/
#   entityseg        EntitySeg images_merge/
#   flickr30k        flickr30k_images/
#   gqa              GQA / Visual Genome images/
#   sa1b             SA-1B root holding sa_000000/, sa_000001/, ...
#   roboflow_vl_100  Roboflow-VL-100 root holding <sub_dataset>/<split>/
#   pixmo_points     a directory of images named sha256(image_url), or leave unset
#                    and fetch the `image_url` column yourself
IMAGE_ROOTS: Dict[str, Optional[str]] = {
    "coco": os.environ.get("CONCOR1_COCO_ROOT"),
    "coco2014": os.environ.get("CONCOR1_COCO2014_ROOT"),
    "coconut": os.environ.get("CONCOR1_OBJECTS365_ROOT"),
    "ade20k": os.environ.get("CONCOR1_ADE20K_ROOT"),
    "entityseg": os.environ.get("CONCOR1_ENTITYSEG_ROOT"),
    "flickr30k": os.environ.get("CONCOR1_FLICKR30K_ROOT"),
    "gqa": os.environ.get("CONCOR1_GQA_ROOT"),
    "sa1b": os.environ.get("CONCOR1_SA1B_ROOT"),
    "roboflow_vl_100": os.environ.get("CONCOR1_ROBOFLOW_ROOT"),
    "pixmo_points": os.environ.get("CONCOR1_PIXMO_ROOT"),
}

# dataset column -> IMAGE_ROOTS key
_ROOT_FOR_DATASET = {
    "coco": "coco",
    "lvis": "coco",
    "coconut_pancap": "coco",
    "groundedref": "coco",          # train split: COCO train2017
    "flickr30k": "flickr30k",
    "goldg_flickr30k": "flickr30k",
    "goldg_gqa": "gqa",
    "coconut": "coconut",
    "ade20k": "ade20k",
    "entityseg": "entityseg",
    "sa1b": "sa1b",
    "roboflow_vl_100": "roboflow_vl_100",
    "pixmo_points": "pixmo_points",
}


def hub_parquet(config: str, split: str = "validation", **kwargs) -> str:
    """Download one released parquet from the Hub and return its local path.

    `split` is `"train"` or `"validation"`; extra kwargs go to `hf_hub_download`
    (`revision`, `token`, `cache_dir`, ...).
    """
    from huggingface_hub import hf_hub_download

    if config not in CONFIGS:
        raise ValueError(f"Unknown config {config!r}. Available: {sorted(CONFIGS)}")
    if split not in CONFIGS[config]:
        raise ValueError(f"Config {config!r} has no {split!r} split; it ships {CONFIGS[config]}.")

    directory = "train" if split == "train" else "test"
    return hf_hub_download(
        HUB_DATASET_ID, f"{directory}/{config}.parquet", repo_type="dataset", **kwargs
    )


def resolve_image_path(dataset: str, split: str, image_key: str) -> Optional[str]:
    """Join `image_key` onto the configured root for its source dataset."""
    root_key = _ROOT_FOR_DATASET.get(dataset)
    if dataset == "groundedref" and split == "val":
        root_key = "coco2014"          # GroundedRef validation uses COCO 2014 images
    root = IMAGE_ROOTS.get(root_key) if root_key else None
    return os.path.join(root, image_key) if root else None


def decode_mask(
    mask_entry: dict, height: Optional[int] = None, width: Optional[int] = None
) -> np.ndarray:
    """Decode one inline mask — COCO RLE (`counts`) or a polygon list — to a bool array."""
    from pycocotools import mask as mask_utils

    size = mask_entry.get("size") or [height, width]
    size = [int(size[0]), int(size[1])]

    if mask_entry.get("counts") is not None:
        counts = mask_entry["counts"]
        rle = {"size": size, "counts": counts.encode("ascii") if isinstance(counts, str) else counts}
        return mask_utils.decode(rle).astype(bool)

    polygon = mask_entry.get("polygon")
    if polygon is None:
        raise ValueError(f"mask entry has neither 'counts' nor 'polygon': {list(mask_entry)}")
    if isinstance(polygon, str):
        polygon = json.loads(polygon)
    if polygon and not isinstance(polygon[0], (list, tuple)):
        polygon = [polygon]
    rles = mask_utils.frPyObjects(
        [np.asarray(part, dtype=np.float64) for part in polygon], size[0], size[1]
    )
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def _caption_sample(row: dict, decode: bool) -> dict:
    """Build a sample from a caption-grounding row (`groups_json` + `masks_json`)."""
    groups = json.loads(row["groups_json"])
    masks = json.loads(row["masks_json"])
    height, width = row["height"], row["width"]

    correspondences = []
    for group in groups:
        entry = {
            "char_spans": group["char_spans"],
            "phrases": group.get("text")
            or [row["caption"][start:end] for start, end in group["char_spans"]],
            "instance_ids": group["instance_ids"],
        }
        if decode:
            union = None
            per_instance = {}
            for instance_id in group["instance_ids"]:
                mask_entry = masks.get(str(instance_id))
                if mask_entry is None:
                    continue
                mask = decode_mask(mask_entry, height, width)
                per_instance[str(instance_id)] = mask
                union = mask if union is None else (union | mask)
            entry["mask"] = union                        # the group's image mask (union)
            entry["_mask_by_instance"] = per_instance    # used by eval_units()
        correspondences.append(entry)

    return {
        "dataset": row["dataset"],
        "split": row["split"],
        "image_key": row["image_key"],
        "image_path": resolve_image_path(row["dataset"], row["split"], row["image_key"]),
        "image_id": row["image_id"],
        "height": height,
        "width": width,
        "text": row["caption"],
        "correspondences": correspondences,
    }


def _instance_sample(row: dict, decode: bool) -> dict:
    """Build a sample from an instance/category row (`instances_json`).

    Correspondence text for these sources is the category name: ConCor-1 takes the
    category list as its text input, and each instance's text mask is the span of its
    own category name inside that list.
    """
    instances = json.loads(row["instances_json"])
    height, width = row["height"], row["width"]

    correspondences = []
    for instance in instances:
        entry = {
            "category_name": instance.get("category_name"),
            "category_id": instance.get("category_id"),
        }
        if decode:
            segmentation = instance["segmentation"]
            if isinstance(segmentation, list):        # polygon(s)
                entry["mask"] = decode_mask({"polygon": segmentation, "size": [height, width]})
            else:
                entry["mask"] = decode_mask(segmentation, height, width)
        correspondences.append(entry)

    sample = {
        "dataset": row["dataset"],
        "split": row["split"],
        "image_key": row["image_key"],
        "image_path": resolve_image_path(row["dataset"], row["split"], row["image_key"]),
        "image_id": row["image_id"],
        "height": height,
        "width": width,
        "correspondences": correspondences,
    }
    if row.get("labels_json"):
        sample["labels"] = json.loads(row["labels_json"])
    if "image_url" in row:
        sample["image_url"] = row["image_url"]
    if "source_dataset" in row:
        sample["source_dataset"] = row["source_dataset"]
    return sample


def category_prompt(sample: dict, separator: str = " . ") -> str:
    """The text input for an instance/category sample: its label list, joined.

    Uses the file's own `labels` column when present (`pixmo_points`, `sa1b`,
    `roboflow_vl_100`), else the distinct category names of the annotated instances.
    """
    labels = sample.get("labels")
    if not labels:
        labels = list(
            OrderedDict.fromkeys(
                entry["category_name"]
                for entry in sample["correspondences"]
                if entry.get("category_name")
            )
        )
    return separator.join(labels)


def eval_units(sample: dict) -> List[dict]:
    """Ground-truth units for the paper's correspondence metrics.

    The released files store the *annotation* as groups: one group is one text mask
    plus the instance ids it refers to. The benchmark's ground-truth unit is slightly
    different — one entry per **unique instance**, carrying the union of the character
    spans of every group that refers to it. That is what merges co-referring mentions
    ("a giraffe" and "the tall one" pointing at the same animal) into one
    correspondence, and it is the unit ConCor-1's numbers are computed against.

    Returns `{"instance_id", "char_spans", "phrases", "mask"}` in order of first
    appearance.
    """
    units: "OrderedDict[str, dict]" = OrderedDict()
    for group in sample["correspondences"]:
        spans = [tuple(span) for span in group["char_spans"]]
        for instance_id in group["instance_ids"]:
            key = str(instance_id)
            if key not in units:
                units[key] = {
                    "instance_id": key,
                    "char_spans": list(spans),
                    "mask": group.get("_mask_by_instance", {}).get(key),
                }
            else:
                for span in spans:
                    if span not in units[key]["char_spans"]:
                        units[key]["char_spans"].append(span)

    text = sample.get("text", "")
    return [
        {
            "instance_id": unit["instance_id"],
            "char_spans": [list(span) for span in unit["char_spans"]],
            "phrases": [text[start:end] for start, end in unit["char_spans"]],
            "mask": unit["mask"],
        }
        for unit in units.values()
    ]


def iter_samples(
    parquet_path: str, decode_masks: bool = True, batch_size: int = 256
) -> Iterator[dict]:
    """Stream decoded samples from one released parquet."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    is_caption = "groups_json" in parquet_file.schema_arrow.names
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            row = {name: values[index] for name, values in columns.items()}
            yield _caption_sample(row, decode_masks) if is_caption else _instance_sample(row, decode_masks)


def load_sample(parquet_path: str, index: int, decode_masks: bool = True) -> dict:
    """Read a single row by position."""
    for position, sample in enumerate(iter_samples(parquet_path, decode_masks)):
        if position == index:
            return sample
    raise IndexError(f"{parquet_path} has fewer than {index + 1} rows")
