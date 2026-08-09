# Inference

ConCor-1 takes an image and its paired text and returns every correspondence between a visually
referential text span and an instance-level image mask. One forward pass produces all of them —
there is no decoding loop, and `generate()` does not apply.

## Python API

```python
from PIL import Image
from concor1 import load_concor1, predict, predict_batch, render_correspondences

model, processor = load_concor1(
    "UWGZQ/ConCor-1",                 # or a local directory
    device=None,                      # default: cuda when available
    attn_implementation="sdpa",       # or "flash_attention_2"
)

image = Image.open("examples/bear.jpg").convert("RGB")
correspondences = predict(model, processor, image, "a brown bear in lush green grass")
```

Each correspondence is a dict:

| Key | Type | Meaning |
|---|---|---|
| `presence_score` | `float` | the model's own "this is a valid correspondence" score |
| `text_spans` | `list[[int, int]]` | character spans into the text you passed, `[start, end)` |
| `text_phrases` | `list[str]` | those spans, sliced out of the text |
| `mask` | `np.ndarray[bool]` | `(height, width)` at the original image size; absent in text-only mode |
| `bridge_index` | `int` | which of the 385 bridge tokens produced it |

Results come back sorted by decreasing presence score.

### Batching

```python
results = predict_batch(model, processor, [image_a, image_b], [text_a, text_b])
```

Sequences are right-padded internally, so images of different sizes and texts of different lengths
batch fine.

### Thresholds

Post-processing keeps bridge tokens above a presence threshold, thresholds the text and image mask
probabilities, then removes duplicates with NMS. A candidate is suppressed only when **both** its
image-mask IoU and its text-span IoU with a higher-scoring candidate exceed the threshold.

| Threshold | Default | Effect of lowering it |
|---|---|---|
| `presence_threshold` | 0.10 | more, noisier candidates |
| `text_threshold` | 0.45 | wider text spans |
| `image_threshold` | 0.45 | larger masks |
| `nms_iou_threshold` | 0.50 | more aggressive de-duplication (0 disables NMS) |

```python
# example
correspondences = predict(model, processor, image, text, presence_threshold=0.1, text_threshold=0.45)
```


### Raw outputs

If you want the logits rather than the decoded correspondences:

```python
inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")
outputs = model(**inputs)
```

| Field | Shape | Meaning |
|---|---|---|
| `presence_logits` | `(B, 385)` | correspondence presence |
| `text_mask_logits` | `(B, 385, N_text)` | per-bridge mask over the text tokens |
| `image_mask_logits` | `(B, 385, N_cells)` | per-bridge mask over image cells, row-major |
| `image_mask_grid_hw` | `(B, 2)` | `(height, width)` of each sample's cell grid |

So `image_mask_logits[b, q, : h * w].reshape(h, w)` is bridge `q`'s mask map, where one cell covers
4 px of the processed image. Masks are produced by bilinearly upsampling in logit space to the
target size, then applying the sigmoid and threshold.


```python
# Caching the logits and re-thresholding on CPU
from types import SimpleNamespace

cache = {k: getattr(outputs, k).float().cpu() for k in
         ("presence_logits", "text_mask_logits", "image_mask_logits", "image_mask_grid_hw")}
correspondences = processor.post_process_correspondences(
    SimpleNamespace(**cache), text=text,
    target_sizes=[(image.height, image.width)], presence_threshold=0.1,
)[0]
```

## Input modes

All three are the same call; only the text changes.

```python
# A caption.
predict(model, processor, image, "This image depicts a close-up of a brown bear in a natural outdoor setting. The background consists of lush green grass. In the foreground, a large brown bear is positioned centrally.")

# A category list
predict(model, processor, image, "bear . grass . tree . person")
#   -> 0.995 ['bear'] ; 0.980 ['grass'] ; absent categories produce no correspondence

# A referring expression
predict(model, processor, image, "a brown bear in lush green grass")
```

## Command line

`scripts/run_inference.py` wraps all of the above.

```bash
# one pair, with a rendered overlay and a JSON dump
python scripts/run_inference.py \
    --image examples/bear.jpg \
    --text "a brown bear in lush green grass" \
    --output_dir outputs/

# many pairs, batched; --input_json is a list of {"image": path, "text": str}
python scripts/run_inference.py --input_json examples/examples.json --batch_size 4 --output_dir outputs/

# a local checkpoint, exact-logit attention, a stricter presence threshold
python scripts/run_inference.py --model /path/to/ConCor-1 \
    --attn_implementation flash_attention_2 --presence_threshold 0.1 \
    --image examples/bear.jpg --text "..."
```

| Flag | Default | |
|---|---|---|
| `--model` | `UWGZQ/ConCor-1` | Hub id or local directory |
| `--device` | cuda if available | |
| `--attn_implementation` | `sdpa` | `flash_attention_2` needs `flash-attn` |
| `--batch_size` | 1 | |
| `--output_dir` | — | writes `NNN_<stem>.png` per pair plus `correspondences.json` |
| `--save_html` | off | also writes the grounded text as a standalone HTML fragment |
| `--figure_width` | 1100 | width of the rendered figure |
| `--presence_threshold` / `--text_threshold` / `--image_threshold` / `--nms_iou_threshold` | 0.1 / 0.45 / 0.45 / 0.5 | |

`correspondences.json` carries the spans, phrases, presence scores and each mask's area fraction —
the mask pixels themselves live in the PNG.

## Rendering

```python
from concor1.visualize import overlay_masks, render_correspondences, render_text_html

overlay_masks(image, correspondences)                      # image only
render_correspondences(image, text, correspondences)       # image + grounded text + legend
render_text_html(text, correspondences)                    # an HTML fragment
```

