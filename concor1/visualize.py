# coding=utf-8
"""Rendering correspondences.

A correspondence has two sides — a text mask and an image mask — so a useful
visualization has to show both in the same colour. Three renderers are provided:

* [`overlay_masks`]         image only, masks blended in and labelled.
* [`render_text_html`]      the text with one coloured underline per correspondence
                            that claims each character (spans can be claimed by
                            several correspondences at once).
* [`render_correspondences`] both, stacked into one PNG-able image.
"""

from __future__ import annotations

from html import escape
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Visually distinct colours, assigned in order of decreasing presence score.
PALETTE: List[Tuple[int, int, int]] = [
    (239, 64, 64), (64, 204, 64), (64, 115, 255), (255, 204, 0), (255, 115, 0),
    (217, 51, 217), (0, 204, 217), (153, 89, 13), (102, 255, 102), (140, 26, 255),
    (255, 153, 204), (0, 140, 0), (179, 179, 0), (0, 26, 153), (204, 140, 51),
    (128, 128, 128), (255, 0, 128), (0, 255, 140), (140, 0, 0), (0, 140, 140),
]


def color_for(index: int) -> Tuple[int, int, int]:
    return PALETTE[index % len(PALETTE)]


def _hex(color: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % color


def _font(size: int) -> ImageFont.ImageFont:
    """A truetype font when one is available, else PIL's bitmap default."""
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_for(index: int, correspondence: Dict) -> str:
    phrases = " / ".join(correspondence["text_phrases"]) or "(no text span)"
    return f"{index + 1}. {phrases} — {correspondence['presence_score']:.2f}"


# ── Image side ────────────────────────────────────────────────────────────────


def overlay_masks(
    image: Image.Image,
    correspondences: Sequence[Dict],
    alpha: float = 0.5,
    draw_labels: bool = True,
) -> Image.Image:
    """Blend every correspondence's image mask over the image in its own colour."""
    canvas = np.array(image.convert("RGB"), dtype=np.float32)
    for index, correspondence in enumerate(correspondences):
        mask = correspondence.get("mask")
        if mask is None or not mask.any():
            continue
        color = np.array(color_for(index), dtype=np.float32)
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * color

    overlaid = Image.fromarray(canvas.astype(np.uint8))
    if not draw_labels:
        return overlaid

    draw = ImageDraw.Draw(overlaid)
    font = _font(max(13, image.width // 55))
    for index, correspondence in enumerate(correspondences):
        mask = correspondence.get("mask")
        if mask is None or not mask.any():
            continue
        rows, columns = np.nonzero(mask)
        text = " / ".join(correspondence["text_phrases"]) or "(no text span)"
        text = f"{index + 1}. {text}"
        anchor = (int(columns.min()) + 2, max(int(rows.min()) - font.size - 4, 0))
        # A dark halo keeps the label readable over both light and dark regions.
        draw.text(anchor, text, font=font, fill=color_for(index),
                  stroke_width=2, stroke_fill=(20, 20, 20))
    return overlaid


# ── Text side ─────────────────────────────────────────────────────────────────


def _owners_per_character(text: str, correspondences: Sequence[Dict]) -> Dict[int, List[int]]:
    """Which correspondences claim each character position."""
    owners: Dict[int, List[int]] = {}
    for index, correspondence in enumerate(correspondences):
        for start, end in correspondence["text_spans"]:
            for position in range(max(start, 0), min(end, len(text))):
                owners.setdefault(position, []).append(index)
    return owners


def _runs(text: str, owners: Dict[int, List[int]]):
    """Split the text into maximal runs sharing the same set of owners."""
    run_start = 0
    run_owners = tuple(owners.get(0, ()))
    for position in range(1, len(text) + 1):
        current = tuple(owners.get(position, ())) if position < len(text) else ("END",)
        if current != run_owners:
            yield text[run_start:position], run_owners
            run_start, run_owners = position, current


def render_text_html(text: str, correspondences: Sequence[Dict]) -> str:
    """The grounded text as an HTML fragment, one underline per claiming correspondence.

    A span may be claimed by several correspondences — "a baseball player" when two
    players are described identically — so underlines stack instead of the
    highest-scoring one winning.
    """
    owners = _owners_per_character(text, correspondences)
    labels = [label_for(index, item) for index, item in enumerate(correspondences)]

    pieces = []
    for fragment, run_owners in _runs(text, owners):
        if not run_owners:
            pieces.append(escape(fragment))
            continue
        colors = [_hex(color_for(index)) for index in run_owners]
        underlines = ", ".join(
            f"0 {2 + 4 * depth}px 0 0 {color}" for depth, color in enumerate(colors)
        )
        tint = "55" if len(colors) == 1 else "33"
        title = escape(" · ".join(labels[index] for index in run_owners))
        pieces.append(
            f'<span style="background:{colors[0]}{tint};box-shadow:{underlines};'
            f'border-radius:3px;padding:1px 2px;" title="{title}">{escape(fragment)}</span>'
        )

    legend = "".join(
        f'<span style="display:inline-block;margin:2px 12px 2px 0;white-space:nowrap;">'
        f'<span style="display:inline-block;width:11px;height:11px;border-radius:2px;'
        f'background:{_hex(color_for(index))};margin-right:5px;"></span>{escape(label)}</span>'
        for index, label in enumerate(labels)
    )
    return (
        '<div style="font-size:17px;line-height:2.7;padding:14px;border-radius:10px;'
        'border:1px solid rgba(128,128,128,0.35);white-space:pre-wrap;'
        f'word-wrap:break-word;">{"".join(pieces)}</div>'
        + (f'<div style="font-size:13px;padding:10px 4px 0 4px;">{legend}</div>' if legend else "")
    )


def _wrap(text: str, font, max_width: int) -> List[Tuple[int, int]]:
    """Greedy word wrap; returns `(start, end)` character offsets per line."""
    lines: List[Tuple[int, int]] = []
    line_start = 0
    last_break = None
    for position in range(len(text)):
        if text[position] == "\n":
            lines.append((line_start, position + 1))
            line_start, last_break = position + 1, None
            continue
        if text[position] == " ":
            last_break = position
        width = font.getlength(text[line_start:position + 1])
        if width > max_width and position > line_start:
            cut = (last_break + 1) if last_break is not None and last_break > line_start else position
            lines.append((line_start, cut))
            line_start, last_break = cut, None
    if line_start < len(text):
        lines.append((line_start, len(text)))
    return lines


def render_text_panel(
    text: str,
    correspondences: Sequence[Dict],
    width: int,
    font_size: int = 20,
    padding: int = 24,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """The grounded text drawn as an image, each claimed run carrying stacked underlines."""
    font = _font(font_size)
    owners = _owners_per_character(text, correspondences)
    lines = _wrap(text, font, width - 2 * padding) or [(0, len(text))]

    max_stack = max((len(value) for value in owners.values()), default=0)
    line_height = int(font_size * 1.55) + 4 * max_stack
    height = 2 * padding + line_height * len(lines)

    panel = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(panel)

    for line_index, (line_start, line_end) in enumerate(lines):
        y = padding + line_index * line_height
        x = padding
        line = text[line_start:line_end]
        for fragment, run_owners in _runs(line, {
            position - line_start: value
            for position, value in owners.items()
            if line_start <= position < line_end
        }):
            fragment_width = font.getlength(fragment)
            if run_owners:
                color = color_for(run_owners[0])
                tint = tuple(int(channel + (255 - channel) * 0.82) for channel in color)
                draw.rectangle(
                    [x - 1, y - 2, x + fragment_width + 1, y + font_size + 4], fill=tint
                )
            draw.text((x, y), fragment, font=font, fill=(15, 15, 15))
            for depth, owner in enumerate(run_owners):
                underline_y = y + font_size + 6 + 4 * depth
                draw.line(
                    [x, underline_y, x + fragment_width, underline_y],
                    fill=color_for(owner), width=3,
                )
            x += fragment_width
    return panel


def render_legend(
    correspondences: Sequence[Dict],
    width: int,
    font_size: int = 18,
    padding: int = 20,
    background: Tuple[int, int, int] = (250, 250, 250),
) -> Image.Image:
    """A colour-chip legend, one row per correspondence."""
    font = _font(font_size)
    row_height = int(font_size * 1.7)
    height = 2 * padding + row_height * max(len(correspondences), 1)
    legend = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(legend)

    if not correspondences:
        draw.text((padding, padding), "no correspondence above the presence threshold",
                  font=font, fill=(90, 90, 90))
        return legend

    for index, correspondence in enumerate(correspondences):
        y = padding + index * row_height
        chip = font_size - 4
        draw.rectangle([padding, y + 3, padding + chip, y + 3 + chip],
                       fill=color_for(index), outline=(60, 60, 60))
        mask = correspondence.get("mask")
        area = f"   ·   mask {100.0 * mask.mean():.2f}% of image" if mask is not None else ""
        spans = ", ".join(f"[{start}, {end})" for start, end in correspondence["text_spans"])
        draw.text(
            (padding + chip + 10, y),
            f"{label_for(index, correspondence)}   ·   spans {spans or '--'}{area}",
            font=font, fill=(25, 25, 25),
        )
    return legend


def render_correspondences(
    image: Optional[Image.Image],
    text: str,
    correspondences: Sequence[Dict],
    width: int = 1100,
    alpha: float = 0.5,
) -> Image.Image:
    """Image masks, grounded text and legend stacked into one figure.

    Both sides of a correspondence carry the same colour, which is the whole point:
    the mask over there belongs to the underlined span over here.
    """
    panels: List[Image.Image] = []

    if image is not None:
        overlaid = overlay_masks(image, correspondences, alpha=alpha)
        height = round(overlaid.height * width / overlaid.width)
        panels.append(overlaid.resize((width, height), Image.LANCZOS))

    panels.append(render_text_panel(text, correspondences, width=width))
    panels.append(render_legend(correspondences, width=width))

    figure = Image.new("RGB", (width, sum(panel.height for panel in panels)), (255, 255, 255))
    offset = 0
    for panel in panels:
        figure.paste(panel, (0, offset))
        offset += panel.height
    return figure
