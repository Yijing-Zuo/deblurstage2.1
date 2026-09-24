"""Out-only line geometry and bounded rescue crops. No OCR or reference text."""

import math

import numpy as np


def crop_box(box, size, padding=0):
    """Clamp an exclusive-end rectangle to image bounds."""
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or padding < 0:
        raise ValueError("Invalid crop rectangle or padding")
    width, height = size
    result = [max(0, int(math.floor(values[0]) - padding)),
              max(0, int(math.floor(values[1]) - padding)),
              min(width, int(math.ceil(values[2]) + padding)),
              min(height, int(math.ceil(values[3]) + padding))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("Empty crop rectangle")
    return result


def _ink(image):
    rgb = np.asarray(image.convert("RGB"), dtype=float)
    gray = rgb.mean(axis=2)
    background = float(np.percentile(gray, 95))
    contrast = max(15, (background - float(np.percentile(gray, 10))) * .3)
    dark = background - gray > contrast
    # These documents use dark neutral print. Saturated artifacts are diagnostic,
    # not sufficient on their own to propose another text row.
    neutral = (rgb.max(axis=2) - rgb.min(axis=2) <= 80) | (rgb.max(axis=2) < 90)
    return dark & neutral, dark & ~neutral


def _bands(active):
    edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(int)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def ink_stats(crop):
    """Image diagnostics; extent_ratio is ink width / ink height, not word count."""
    ink, colored = _ink(crop)
    yy, xx = np.nonzero(ink)
    width = int(xx.max() - xx.min() + 1) if len(xx) else 0
    height = int(yy.max() - yy.min() + 1) if len(yy) else 0
    occupied = ink[:, xx.min():xx.max() + 1].any(axis=0) if len(xx) else []
    groups = len(_bands(occupied))
    density = float(ink.sum() / max(1, width * height))
    # A solid rectangle is not evidence for a missing sentence. Connected glyphs
    # may have one column group, so also accept internal vertical whitespace.
    varied = len(xx) and bool((ink.sum(axis=0)[xx.min():xx.max() + 1] < .8 * height).any())
    return {"extent_ratio": width / max(1, height), "ink_width": width,
            "ink_height": height, "ink_fraction": float(ink.mean()),
            "colored_fraction": float(colored.mean()), "column_groups": groups,
            "text_like": bool(width >= 8 and height >= 2 and width >= 1.5 * height
                              and .015 < density < .8 and (groups >= 2 or varied))}


def _overlap(a, b, axis):
    return max(0, min(a[axis + 2], b[axis + 2]) - max(a[axis], b[axis]))


def _gutters(lines):
    if len(lines) < 4:
        return []
    left, right = min(x["box"][0] for x in lines), max(x["box"][2] for x in lines)
    height = float(np.median([x["box"][3] - x["box"][1] for x in lines]))
    narrow = [x["box"] for x in lines if x["box"][2] - x["box"][0] < .72 * (right - left)]
    intervals = sorted((b[0], b[2]) for b in narrow)
    gaps, end = [], intervals[0][1] if intervals else 0
    for start, stop in intervals[1:]:
        if start - end >= max(4, .4 * height):
            cut = (end + start) / 2
            a, b = [r for r in narrow if r[2] <= cut], [r for r in narrow if r[0] >= cut]
            if len(a) >= 2 and len(b) >= 2 and min(max(r[3] for r in a), max(r[3] for r in b)) > max(min(r[1] for r in a), min(r[1] for r in b)):
                gaps.append(cut)
        end = max(end, stop)
    return gaps


def _split(line, image):
    x0, y0, x1, y1 = line["box"]
    ink, _ = _ink(image.crop(line["box"]))
    bands = [(a, b) for a, b in _bands(ink.sum(axis=1) >= max(2, .008 * (x1 - x0))) if b - a >= 2]
    if len(bands) < 2:
        return [line]
    typical = float(np.median([b - a for a, b in bands]))
    if (max(b - a for a, b in bands) <= 2 * min(b - a for a, b in bands)
            and all(b - a <= 1.8 * typical for a, b in bands)
            and all(bands[i + 1][0] - bands[i][1] >= max(2, .2 * typical) for i in range(len(bands) - 1))):
        return [{**line, "box": [x0, max(y0, y0 + a - 1), x1, min(y1, y0 + b + 1)],
                 "split_from_multiline": True} for a, b in bands]
    return [{**line, "geometry_warning": "possible_multiple_rows"}]


def tight_box(image, box, padding=1):
    """Trim vertical whitespace only; preserve original horizontal coverage."""
    if padding < 0:
        raise ValueError("Crop padding must not be negative")
    x0, y0, x1, y1 = crop_box(box, image.size)
    ink, _ = _ink(image.crop((x0, y0, x1, y1)))
    active = np.flatnonzero(ink.sum(axis=1) >= 2)
    if not len(active):
        return [x0, y0, x1, y1]
    return [x0, max(y0, y0 + int(active[0]) - padding),
            x1, min(y1, y0 + int(active[-1]) + 1 + padding)]


def _ordered(lines):
    columns = {}

    def visit(group, path=()):
        cuts = _gutters(group)
        if cuts:
            cut = cuts[0]
            left = [x for x in group if x["box"][2] <= cut]
            right = [x for x in group if x["box"][0] >= cut]
            spans = [x for x in group if x not in left and x not in right]
            if not spans:
                return visit(left, path + (0,)) + visit(right, path + (1,))
            result, remaining = [], left + right
            for span in sorted(spans, key=lambda x: x["box"][1]):
                center = (span["box"][1] + span["box"][3]) / 2
                before = [x for x in remaining if (x["box"][1] + x["box"][3]) / 2 < center]
                remaining = [x for x in remaining if x not in before]
                result.extend(visit(before, path))
                result.extend(visit([span], path))
            return result + visit(remaining, path)
        column = columns.setdefault(path, len(columns))
        return [{**line, "column": column} for line in sorted(group, key=lambda x: (x["box"][1], x["box"][0]))]

    return [{**line, "order": i} for i, line in enumerate(visit(lines))]


def build_lines(image, detections, settings):
    """Split rows, merge same-column fragments, suppress contained duplicates."""
    lines, suppressed = [], []
    for detection in detections:
        points = np.asarray(detection.get("polygon", []), dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all():
            suppressed.append({"reason": "invalid_detection"})
            continue
        try:
            box = crop_box([*points.min(axis=0), *points.max(axis=0)], image.size)
        except ValueError:
            suppressed.append({"reason": "empty_detection"})
            continue
        lines.extend(_split({"box": box, "detections": [detection]}, image))
    gutters = _gutters(lines)
    changed = True
    while changed:
        changed = False
        for i, line in enumerate(lines):
            a = line["box"]
            for j in range(i + 1, len(lines)):
                b = lines[j]["box"]
                ah, bh = a[3] - a[1], b[3] - b[1]
                same = (_overlap(a, b, 1) >= .65 * min(ah, bh)
                        and abs(sum(a[1::2]) - sum(b[1::2])) < .9 * max(ah, bh)
                        and max(ah, bh) < 1.8 * min(ah, bh))
                overlap = _overlap(a, b, 0)
                duplicate = overlap >= .9 * min(a[2] - a[0], b[2] - b[0])
                crosses = any((a[2] <= cut <= b[0]) or (b[2] <= cut <= a[0]) for cut in gutters)
                gap = max(a[0], b[0]) - min(a[2], b[2])
                fragment = not crosses and gap <= .6 * min(ah, bh)
                if same and (duplicate or fragment):
                    line["box"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    line["detections"].extend(lines[j]["detections"])
                    suppressed.append({"box": b, "reason": "contained_fragment" if duplicate else "joined_fragment"})
                    lines.pop(j)
                    changed = True
                    break
            if changed:
                break
    padding = int(settings.get("crop_padding", 1))
    for line in lines:
        line["box"] = tight_box(image, line["box"], padding)
    return _ordered(lines), suppressed


def coverage(image, lines):
    ink, colored = _ink(image)
    covered = np.zeros(ink.shape, dtype=bool)
    for line in lines:
        x0, y0, x1, y1 = crop_box(line["box"], image.size)
        covered[y0:y1, x0:x1] = True
    uncovered = ink & ~covered
    return {"uncovered_ink_fraction": float(uncovered.sum() / max(1, ink.sum())),
            "uncovered_ink_bands": [[a, b] for a, b in _bands(uncovered.sum(axis=1) >= max(2, image.width * .01)) if b - a >= 2],
            "colored_noise_fraction": float(colored.mean())}


def residual_regions(image, lines, settings):
    """Propose same-column local detector ROIs; do not manufacture new text lines."""
    maximum = int(settings.get("max_residual_regions", 8))
    if maximum < 1:
        return []
    ink, _ = _ink(image)
    uncovered = ink.copy()
    for line in lines:
        x0, y0, x1, y1 = crop_box(line["box"], image.size)
        uncovered[y0:y1, x0:x1] = False
    bounds = [0] + [int(round(x)) for x in _gutters(lines)] + [image.width]
    typical = float(np.median([x["box"][3] - x["box"][1] for x in lines])) if lines else 12
    regions = []
    for left, right in zip(bounds, bounds[1:]):
        proposed = []
        for top, bottom in _bands(uncovered[:, left:right].sum(axis=1) >= 2):
            if bottom - top < 2:
                continue
            if not ink_stats(image.crop((left, top, right, bottom)))["text_like"]:
                continue
            xs = np.flatnonzero(uncovered[top:bottom, left:right].any(axis=0))
            x0, x1 = left + int(xs[0]), left + int(xs[-1]) + 1
            y0, y1 = top, bottom
            for line in lines:
                b = line["box"]
                if b[0] >= left and b[2] <= right and _overlap(b, [left, top, right, bottom], 1) >= .4 * min(b[3] - b[1], bottom - top):
                    x0, x1, y0, y1 = min(x0, b[0]), max(x1, b[2]), min(y0, b[1]), max(y1, b[3])
            pad = max(2, int(round(typical * .5)))
            box = [max(left, x0 - pad), max(0, y0 - pad), min(right, x1 + pad), min(image.height, y1 + pad)]
            if proposed and box[1] - proposed[-1][3] <= typical:
                old = proposed[-1]
                proposed[-1] = [min(old[0], box[0]), old[1], max(old[2], box[2]), max(old[3], box[3])]
            else:
                proposed.append(box)
        regions.extend(proposed)
    regions.sort(key=lambda b: int(uncovered[b[1]:b[3], b[0]:b[2]].sum()), reverse=True)
    return sorted(regions[:maximum], key=lambda b: (b[1], b[0]))


def segment_boxes(image, box, max_segments=3, overlap=.15):
    """At most three overlapping horizontal crops; prefer whitespace near cuts."""
    if max_segments < 1 or not 0 <= overlap < .5:
        raise ValueError("Invalid segment budget or overlap")
    x0, y0, x1, y1 = crop_box(box, image.size)
    width = x1 - x0
    stats = ink_stats(image.crop((x0, y0, x1, y1)))
    height = max(1, stats["ink_height"])
    count = min(3, int(max_segments), max(1, int(math.ceil(width / max(64, 12 * height)))))
    if count == 1:
        return [[x0, y0, x1, y1]]
    ink, _ = _ink(image.crop((x0, y0, x1, y1)))
    profile = ink.sum(axis=0)
    cuts = [0]
    for i in range(1, count):
        target = width * i / count
        radius = max(2, int(width / count * .15))
        candidates = range(max(cuts[-1] + 1, int(target) - radius), min(width - 1, int(target) + radius) + 1)
        cuts.append(min(candidates, key=lambda x: (profile[x], abs(x - target))))
    cuts.append(width)
    margin = max(1, int(round(width / count * overlap / 2))) if overlap else 0
    return [[x0 + max(0, a - (margin if i else 0)), y0,
             x0 + min(width, b + (margin if i < count - 1 else 0)), y1]
            for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
